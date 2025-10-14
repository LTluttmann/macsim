import abc
import torch
import torch.nn as nn

from einops import rearrange
from tensordict import TensorDict

from macsim.utils.ops import gather_by_index
from macsim.models.encoder.base import MatNetEncoderOutput
from macsim.models.policy_args import PolicyParams, MacsimParams


def get_action_emb(params: PolicyParams):
    action_layer = {
        ####### FJSP #############
        "fjsp": JobShopAction,
        "ma_fjsp": JobShopAction,
        ####### FFSP #############
        "ffsp": FlowShopAction,
        "matnet_ffsp": FlowShopAction,
        "ma_ffsp": FlowShopAction,    
        ###### HCVRP ###############
        "hcvrp": HCVRP_Action,
        "ma_hcvrp": HCVRP_Action,    
    }
    action_emb = action_layer[params.env.env](params)
    return action_emb


class BaseActionEmbedding(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self):
        super().__init__()

    @staticmethod
    @abc.abstractmethod
    def get_action_mask(td):
        ...

    @abc.abstractmethod
    def forward(self, embs: TensorDict, td: TensorDict, cache = None):
        ...


class JobShopAction(BaseActionEmbedding):
    def __init__(self, params: MacsimParams):
        super().__init__()
        self.params = params
        self.embed_dim = params.embed_dim
        self.skip_token = nn.Parameter(
            (
                torch.distributions.Uniform(low=-1/self.embed_dim**0.5, high=1/self.embed_dim**0.5)
                .sample((1, 1, self.embed_dim))
            ), 
            requires_grad=True
        )

    @staticmethod
    def get_action_mask(td):
        # mask
        node_mask = td["job_done"]
        skip_token_mask = torch.full_like(node_mask[:, :1], fill_value=False)
        node_mask_w_skip_token = torch.cat((skip_token_mask, node_mask), dim=1)
        return node_mask_w_skip_token

    def forward(self, embs: MatNetEncoderOutput, td: TensorDict, return_mask: bool = False):
        skip_token = self.skip_token.expand(td.size(0), 1, -1)
        _, n_j, n_o = td["op_scheduled"].shape
        ops_emb = rearrange(embs["nodes"], "b (j o) d -> b j o d", j=n_j, o=n_o)
        job_emb = gather_by_index(ops_emb, td["next_op"], dim=2)
        job_emb_w_skip_token = torch.cat((skip_token, job_emb), dim=1)
        if return_mask:
            return job_emb_w_skip_token, self.get_action_mask(td)
        return job_emb_w_skip_token


class FlowShopAction(BaseActionEmbedding):
    def __init__(self, params: MacsimParams):
        super().__init__()
        self.embed_dim = params.embed_dim
        self.skip_token = nn.Parameter(
            (
                torch.distributions.Uniform(low=-1/self.embed_dim**0.5, high=1/self.embed_dim**0.5)
                .sample((1, 1, self.embed_dim))
            ), 
            requires_grad=True
        )

    @staticmethod
    def get_action_mask(td):
        # mask
        node_mask = td["job_done"]
        skip_tok_mask = torch.full_like(node_mask[:, :1], fill_value=False)
        node_mask_w_skip_token = torch.cat((node_mask, skip_tok_mask), dim=1)
        return node_mask_w_skip_token
    
    def forward(self, embs: MatNetEncoderOutput, td: TensorDict, return_mask: bool = False):
        skip_token = self.skip_token.expand(td.size(0), 1, -1)  
        job_emb_w_skip_token = torch.cat((embs["nodes"], skip_token), dim=1)
        if return_mask:
            return job_emb_w_skip_token, self.get_action_mask(td)
        return job_emb_w_skip_token


# class HFSP_Action(JobShopAction):
#     def __init__(self, params):
#         super().__init__(params)

#     def forward(self, embs: MatNetEncoderOutput, td: TensorDict):
#         skip_token = self.skip_token.expand(td.size(0), 1, -1)
#         _, n_job, n_stage = td["op_scheduled"].shape
#         ops_emb = rearrange(embs["nodes"], "b (j o) d -> b j o d", j=n_job, o=n_stage)
#         job_emb = gather_by_index(ops_emb, td["job_location"].clamp(max=n_stage-1), dim=2)
#         job_emb_w_skip_token = torch.cat((skip_token, job_emb), dim=1)
#         if hasattr(self, "job_emb_attn"):
#             job_emb_w_skip_token = self.job_emb_attn(job_emb_w_skip_token, attn_mask=self._get_self_attn_mask(td))
#         return job_emb_w_skip_token


class MTSP_Action(BaseActionEmbedding):
    def __init__(self, params: MacsimParams):
        super().__init__()
        self.embed_dim = params.embed_dim
        self.skip_token = nn.Parameter(
            (
                torch.distributions.Uniform(low=-1/self.embed_dim**0.5, high=1/self.embed_dim**0.5)
                .sample((1, 1, self.embed_dim))
            ), 
            requires_grad=True
        )

    @staticmethod
    def get_action_mask(td):
        # mask
        num_agents = td["current_length"].size(1)
        node_mask = ~td["available"][:, num_agents:]
        skip_token_mask = torch.full_like(node_mask[:, :1], fill_value=False)
        node_mask_w_skip_token = torch.cat((node_mask, skip_token_mask), dim=1)
        return node_mask_w_skip_token

    def forward(self, embs: MatNetEncoderOutput, td: TensorDict, return_mask: bool = False):
        skip_token = self.skip_token.expand(td.size(0), 1, -1)  
        city_embs = embs["nodes"][:, 1:]
        city_embs_w_skip_token = torch.cat((city_embs, skip_token), dim=1)
        if return_mask:
            return city_embs_w_skip_token, self.get_action_mask(td)
        return city_embs_w_skip_token


class HCVRP_Action(BaseActionEmbedding):
    def __init__(self, params: MacsimParams):
        super().__init__()
        self.embed_dim = params.embed_dim
        self.skip_token = nn.Parameter(
            (
                torch.distributions.Uniform(low=-1/self.embed_dim**0.5, high=1/self.embed_dim**0.5)
                .sample((1, 1, self.embed_dim))
            ), 
            requires_grad=True
        )

    @staticmethod
    def get_action_mask(td):
        node_mask = td["visited"].clone()
        node_mask[:, 0] = False
        skip_token_mask = torch.full_like(node_mask[:, :1], fill_value=False)
        node_mask_w_skip_token = torch.cat((node_mask, skip_token_mask), dim=1)
        return node_mask_w_skip_token

    def forward(self, embs: MatNetEncoderOutput, td: TensorDict, return_mask: bool = False):
        skip_token = self.skip_token.expand(td.size(0), 1, -1)  
        action_embs = torch.cat((embs["nodes"], skip_token), dim=1)
        if return_mask:
            return action_embs, self.get_action_mask(td)
        return action_embs
