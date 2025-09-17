import torch
import torch.nn as nn

from tensordict import TensorDict

from macsim.models.policy_args import MacsimParams
from macsim.utils.config import PolicyParams
from macsim.models.encoder.base import MatNetEncoderOutput
from macsim.models.nn.misc import PositionalEncoding, AttentionGraphPooling, CommunicationLayer


def get_context_emb(params: PolicyParams):
    
    emb_registry = {
        "ma_fjsp": JobShopContext,
        "ma_ffsp": FlowShopContext,
        "ma_hfsp": HFSP_Context if getattr(params.env, "flatten_stages", False) else FlowShopContext,
        "ma_mtsp": MTSP_Context,
        "ma_hcvrp": HCVRP_Context,
    }

    EmbCls = emb_registry.get(params.env.env, SimpleContext)
    return EmbCls(params)


class SimpleContext(nn.Module):

    def __init__(self, params: PolicyParams):
        super().__init__()

    def forward(self, embs: MatNetEncoderOutput, td: TensorDict): 
        bs, num_agents, _ = embs["agents"].shape
        agent_mask = torch.full((bs, num_agents), fill_value=False, device=td.device)
        return embs["agents"], agent_mask
    

class JobShopContext(nn.Module):

    def __init__(self, params: MacsimParams):
        super().__init__()
        self.use_context_emb = params.use_context_emb
        self.stepwise = params.stepwise_encoding

        if self.use_context_emb:
            self.op_pooling = AttentionGraphPooling(params)
            self.workload_enc = PositionalEncoding(embed_dim=params.embed_dim, dropout=params.dropout, max_len=1_000)

        if not self.stepwise:
            self.proj_ma_time = nn.Linear(1, params.embed_dim, bias=False)

        if params.use_communication:
            self.communcation_layer = CommunicationLayer(params)

    def _agent_emb(self, agent_embs, td)-> torch.Tensor:
        if not self.stepwise:
            # b m 1
            ma_time = td["t_ma_idle"].unsqueeze(-1) / td["scale_factor"]
            # b m d
            ma_time_proj = self.proj_ma_time(ma_time)
        else:
            ma_time_proj = 0

        agent_embs = agent_embs + ma_time_proj
        return agent_embs
        
    def _graph_emb(self, embs: MatNetEncoderOutput, td: TensorDict) -> torch.Tensor:
        num_agents = embs["agents"].size(1)
        # NOTE op_scheduled is also True for padded entries
        op_mask = torch.flatten(td["op_scheduled"], 1, 2)
        # (bs, 1)
        remaining_ops = torch.sum(~op_mask, dim=1, keepdim=True)
        # (bs, 1, d)
        graph_emb = self.op_pooling(embs["nodes"], op_mask)
        # (bs, 1, d)
        graph_emb = self.workload_enc(graph_emb, remaining_ops)
        return graph_emb

    def forward(self, embs: MatNetEncoderOutput, td: TensorDict):
        bs, num_agents = td["t_ma_idle"].shape
        # (bs, m, d)
        agent_embs = self._agent_emb(embs["agents"], td)

        if self.use_context_emb:
            # (bs, m, d)
            graph_emb = self._graph_emb(embs, td)
            agent_embs += graph_emb
        
        if hasattr(self, "communcation_layer"): 
            agent_embs = self.communcation_layer(agent_embs)

        agent_mask = torch.full((bs, num_agents), fill_value=False, device=td.device)
        return agent_embs, agent_mask



class FlowShopContext(nn.Module):

    def __init__(self, params: MacsimParams):
        super().__init__()
        self.stepwise = params.stepwise_encoding
        self.use_context_emb = params.use_context_emb
        
        if self.use_context_emb:
            self.pooling = AttentionGraphPooling(params)
            self.pos_enc = PositionalEncoding(embed_dim=params.embed_dim, dropout=params.dropout, max_len=10_000)

        if not self.stepwise:
            self.proj_ma_time = nn.Linear(1, params.embed_dim, bias=False)

        if params.use_communication:
            self.communcation_layer = CommunicationLayer(params)

    def _agent_updates(self, agent_embs, td):
        if not self.stepwise:
            # b m 1
            ma_time = td["t_ma_idle"].unsqueeze(-1) / td["scale_factor"]
            # b m d
            ma_time_proj = self.proj_ma_time(ma_time)
        else:
            ma_time_proj = 0

        agent_embs = agent_embs + ma_time_proj
        return agent_embs
    
    def _graph_emb(self, embs, td):
        job_mask = td["job_done"]
        graph_emb = self.pooling(embs["nodes"], job_mask)
        # bs, 1
        remaining_jobs = torch.sum(~job_mask, dim=1, keepdim=True)
        # bs, n_ma
        remaining_ops = remaining_jobs.expand(-1, embs["agents"].size(1))
        # embedding for the amount of remaining operations
        graph_emb = self.pos_enc(graph_emb, remaining_ops)
        return graph_emb

    def forward(self, embs: MatNetEncoderOutput, td: TensorDict):
        agent_embs = self._agent_updates(embs["agents"], td)
        
        if self.use_context_emb:
            graph_emb = self._graph_emb(embs, td)
            agent_embs = agent_embs + graph_emb

        if hasattr(self, "communcation_layer"): 
            agent_embs = self.communcation_layer(agent_embs)

        agent_mask = td["stage_table"] == -1
        return agent_embs, agent_mask



class HFSP_Context(FlowShopContext):

    def __init__(self, params: MacsimParams):
        super().__init__(params)
    
    def _graph_emb(self, embs, td):
        op_mask = td["op_scheduled"].flatten(1,2)
        graph_emb = self.pooling(embs["nodes"], op_mask)
        # bs, 1
        remaining_jobs = torch.sum(~op_mask, dim=1, keepdim=True)
        # bs, n_ma
        remaining_ops = remaining_jobs.expand(-1, embs["agents"].size(1))
        # embedding for the amount of remaining operations
        graph_emb = self.pos_enc(graph_emb, remaining_ops)
        return graph_emb



class MTSP_Context(nn.Module):

    def __init__(self, params: MacsimParams):
        super().__init__()
        self.stepwise = params.stepwise_encoding
        self.use_context_emb = params.use_context_emb
        if params.use_communication:
            self.communcation_layer = CommunicationLayer(params)

        if self.use_context_emb:
            self.pooling = AttentionGraphPooling(params)
            self.graph_weight = nn.Parameter(torch.full((1, 1, 1), fill_value=0.5), requires_grad=True)
            self.pooled_proj = nn.Linear(params.embed_dim, params.embed_dim, bias=params.embed_dim)

    def _graph_emb(self, embs, td):
        bs, num_agents, _ = embs["agents"].shape
        city_embs = embs["nodes"][:, 1:]
        city_mask = ~td["available"][:, num_agents:]
        graph_emb = self.pooling(city_embs, city_mask)
        graph_emb = self.pooled_proj(graph_emb)
        graph_emb = graph_emb.expand(-1, num_agents, -1).contiguous()
        # # bs, 1
        # remaining_nodes = torch.sum(~city_mask, dim=1, keepdim=True)
        # # bs, n_ma
        # remaining_nodes = remaining_nodes.expand(-1, embs["agents"].size(1))
        # # embedding for the amount of remaining operations
        # graph_emb = self.pos_enc(graph_emb, remaining_nodes)
        # graph_emb += self.graph_feat_proj(torch.log(1 + remaining_nodes).unsqueeze(-1))
        return graph_emb
    
    def forward(self, embs: MatNetEncoderOutput, td: TensorDict):
        bs, num_agents, _ = embs["agents"].shape
        agent_embs = embs["agents"]

        if self.use_context_emb:
            graph_emb = self._graph_emb(embs, td)
            agent_embs = agent_embs + self.graph_weight * graph_emb

        if hasattr(self, "communication_layer"):
            agent_embs = self.communcation_layer(agent_embs)

        agent_mask = torch.full((bs, num_agents), fill_value=False, device=td.device)
        return agent_embs, agent_mask


class HCVRP_Context(nn.Module):

    def __init__(self, params: MacsimParams):
        super().__init__()
        self.stepwise = params.stepwise_encoding
        self.use_context_emb = params.use_context_emb
        if params.use_communication:
            self.communcation_layer = CommunicationLayer(params)

        if self.use_context_emb:
            self.pooling = AttentionGraphPooling(params)
            # self.proj_context = nn.Linear(3 * params.embed_dim, params.embed_dim, bias=params.bias)
            self.graph_weight = nn.Parameter(torch.full((1, 1, 1), fill_value=0.5), requires_grad=True)            
            self.pooled_proj = nn.Linear(params.embed_dim, params.embed_dim, bias=params.embed_dim)


    def _graph_emb(self, embs, td):
        bs, num_agents, _ = embs["agents"].shape
        city_embs = embs["nodes"]#[:, 1:]
        city_mask = td["visited"].clone()#[:, 1:]
        city_mask[:, 0] = False
        graph_emb = self.pooling(city_embs, city_mask)
        graph_emb = self.pooled_proj(graph_emb)
        graph_emb = graph_emb.expand(-1, num_agents, -1).contiguous()
        return graph_emb
    
    def forward(self, embs: MatNetEncoderOutput, td: TensorDict):
        bs, num_agents, _ = embs["agents"].shape
        agent_embs = embs["agents"]

        if self.use_context_emb:
            # depot_node = embs["nodes"]
            graph_emb = self._graph_emb(embs, td)
            agent_embs = agent_embs + self.graph_weight * graph_emb
            # cur_node_embedding = gather_by_index(
            #     embs["nodes"], td["current_node"]
            # )  # [B, M, hdim]
            # agent_embs = self.proj_context(torch.cat((agent_embs, cur_node_embedding, graph_emb), dim=-1))

        if hasattr(self, "communication_layer"):
            agent_embs = self.communcation_layer(agent_embs)

        agent_mask = torch.full((bs, num_agents), fill_value=False, device=td.device)
        return agent_embs, agent_mask

