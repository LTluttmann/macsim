import abc
import torch
import torch.nn as nn

from einops import rearrange
from tensordict import TensorDict

from macsim.utils.ops import gather_by_index
from macsim.models.nn.dynamic import get_dynamic_emb
from macsim.models.nn.misc import CommunicationLayer
from macsim.models.encoder.base import MatNetEncoderOutput
from macsim.models.policy_args import PolicyParams, MacsimParams


class BaseKVL(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, params: PolicyParams):
        super().__init__()
        self.cache = None
        self.params = params
        self.embed_dim = params.embed_dim
        self.dynamic_embedding = get_dynamic_emb(params)
        self.Wkvl = nn.Linear(params.embed_dim, 3 * params.embed_dim, bias=params.bias)


    def compute_cache(self, embs: TensorDict, td: TensorDict):
        # 3 * (bs, n_j, n_o, emb_dim)
        self.cache = self.Wkvl(embs["nodes"]).chunk(3, dim=-1)

    def forward(self, embs: TensorDict, td: TensorDict, cache = None):
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(embs, td)
        if cache is not None:
            k, v, l = self.cache
        else:
            k, v, l = self.Wkvl(embs["nodes"]).chunk(3, dim=-1)

        k_dyn = k + glimpse_k_dyn
        v_dyn = v + glimpse_v_dyn
        l_dyn = l + logit_k_dyn

        bs, num_agents, _ = embs["nodes"].shape
        node_mask = torch.full((bs, num_agents), fill_value=False, device=td.device)
        return k_dyn, v_dyn, l_dyn, node_mask
    


def get_kvl_emb(params: PolicyParams) -> BaseKVL:
    kvl_layer = {
        ####### FJSP #############
        "fjsp": JobShopKVL,
        "ma_fjsp": JobShopKVL,
        ####### FFSP #############
        "ffsp": FlowShopKVL,
        "matnet_ffsp": FlowShopKVL,
        "ma_ffsp": FlowShopKVL,
        ####### HCVRP #############
        "hcvrp": HCVRP_KVL,
        "ma_hcvrp": HCVRP_KVL,
    }
    KVL_Class = kvl_layer.get(params.env.env, BaseKVL)
    return KVL_Class(params)


class JobShopKVL(BaseKVL):
    """Generates K, V and L embeddings from the next operation per job"""
    def __init__(self, params: MacsimParams):
        super().__init__(params)
        if not params.env.use_job_emb:
            self.job_emb_attn = CommunicationLayer(params)
        self.wait_emb = nn.Parameter(
            (
                torch.distributions.Uniform(low=-1/self.embed_dim**0.5, high=1/self.embed_dim**0.5)
                .sample((1, 1, self.embed_dim))
            ), 
            requires_grad=True
        )

    def compute_cache(self, embs: MatNetEncoderOutput, td: TensorDict) -> None:
        _, n_jobs, n_ops = td["op_scheduled"].shape
        # (bs, n_j, n_o, emb_dim)
        ops_emb = rearrange(embs["nodes"], "b (j o) d -> b j o d", j=n_jobs, o=n_ops)
        # 3 * (bs, n_j, n_o, emb_dim)
        self.cache = self.Wkvl(ops_emb).chunk(3, dim=-1)

    def _get_self_attn_mask(self, td):
        # True means not to attend
        # (bs * num_heads, num_jobs)
        attn_mask = td["job_done"].repeat_interleave(
            self.params.num_heads, dim=0
        )
        # shape: (bs * num_heads, 1)
        dummy_job_mask = torch.full_like(attn_mask[:, :1], fill_value=False)  # dummy job always attends
        # (bs * num_heads, num_jobs+1)
        attn_mask = torch.cat((dummy_job_mask, attn_mask), dim=1)
        # (bs * num_heads, num_jobs+1, num_jobs+1)
        attn_mask = attn_mask.unsqueeze(1).expand((-1, attn_mask.size(1), attn_mask.size(1)))
        return attn_mask

    def forward(self, emb: MatNetEncoderOutput, td: TensorDict, cache = None):
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(emb, td)
        # (bs, 1, d)
        wait_emb = self.wait_emb.expand(td.size(0), 1, -1)
        if cache is not None:
            k, v, l = tuple(map(
                lambda x: gather_by_index(x, td["next_op"], dim=2), 
                self.cache
            ))
            k, v, l = tuple(map(lambda x: torch.cat((wait_emb, x), dim=1), cache))
            
        else:
            _, n_j, n_o = td["op_scheduled"].shape
            ops_emb = rearrange(emb["nodes"], "b (j o) d -> b j o d", j=n_j, o=n_o)
            job_emb = gather_by_index(ops_emb, td["next_op"], dim=2)
            job_emb_w_wait = torch.cat((wait_emb, job_emb), dim=1)
            if hasattr(self, "job_emb_attn"):
                job_emb_w_wait = self.job_emb_attn(job_emb_w_wait, attn_mask=self._get_self_attn_mask(td))
            k, v, l = self.Wkvl(job_emb_w_wait).chunk(3, dim=-1)

        k_dyn = k + glimpse_k_dyn
        v_dyn = v + glimpse_v_dyn
        l_dyn = l + logit_k_dyn

        node_mask = td["job_done"].clone()
        dummy_mask = torch.full_like(node_mask[:, :1], fill_value=False)
        node_mask_w_dummy = torch.cat((dummy_mask, node_mask), dim=1)
        return k_dyn, v_dyn, l_dyn, node_mask_w_dummy


class FlowShopKVL(BaseKVL):
    """Generates K, V and L embeddings from the current job embeddings"""
    def __init__(self, params: MacsimParams):
        super().__init__(params)

        self.wait_emb = nn.Parameter(
            (
                torch.distributions.Uniform(low=-1/self.embed_dim**0.5, high=1/self.embed_dim**0.5)
                .sample((1, 1, self.embed_dim))
            ), 
            requires_grad=True
        )

    def compute_cache(self, embs: MatNetEncoderOutput, td: TensorDict) -> None:
        # shape: 3 * (bs, n, emb_dim)
        self.cache = self.Wkvl(embs["nodes"]).chunk(3, dim=-1)

    def forward(self, embs: MatNetEncoderOutput, td: TensorDict, cache = None):
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(embs, td)
        wait_emb = self.wait_emb.expand(td.size(0), 1, -1)  
        if cache is not None:
            k, v, l = self.cache
            k, v, l = tuple(map(lambda x: torch.cat((x, wait_emb), dim=1), cache))
            
        else:
            job_emb_w_wait = torch.cat((embs["nodes"], wait_emb), dim=1)
            k, v, l = self.Wkvl(job_emb_w_wait).chunk(3, dim=-1)

        k_dyn = k + glimpse_k_dyn
        v_dyn = v + glimpse_v_dyn
        l_dyn = l + logit_k_dyn

        node_mask = td["job_done"].clone()
        dummy_mask = torch.full_like(node_mask[:, :1], fill_value=False)
        node_mask_w_dummy = torch.cat((node_mask, dummy_mask), dim=1)
        return k_dyn, v_dyn, l_dyn, node_mask_w_dummy

    
class HCVRP_KVL(BaseKVL):
    def __init__(self, params: MacsimParams):
        super().__init__(params)

        self.wait_emb = nn.Parameter(
            (
                torch.distributions.Uniform(low=-1/self.embed_dim**0.5, high=1/self.embed_dim**0.5)
                .sample((1, 1, self.embed_dim))
            ), 
            requires_grad=True
        )
    
    def compute_cache(self, embs: TensorDict, td: TensorDict):
        self.cache = self.Wkvl(embs["nodes"]).chunk(3, dim=-1)

    def forward(self, embs: TensorDict, td: TensorDict, cache = None):
        bs = td.size(0)
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(embs, td)
        wait_emb = self.wait_emb.expand(bs, 1, -1)  
        if cache is not None:
            k, v, l = tuple(map(lambda x: torch.cat((x, wait_emb), dim=1), cache))
        else:
            node_emb_w_wait = torch.cat((embs["nodes"], wait_emb), dim=1)
            k, v, l = self.Wkvl(node_emb_w_wait).chunk(3, dim=-1)

        k_dyn = k + glimpse_k_dyn
        v_dyn = v + glimpse_v_dyn
        l_dyn = l + logit_k_dyn

        node_mask = td["visited"].clone()
        node_mask[:, 0] = False
        dummy_mask = torch.full_like(node_mask[:, :1], fill_value=False)
        node_mask_w_dummy = torch.cat((node_mask, dummy_mask), dim=1)
        return k_dyn, v_dyn, l_dyn, node_mask_w_dummy
    