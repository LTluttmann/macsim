import torch
import torch.nn as nn

from typing import Union
from einops import rearrange
from tensordict import TensorDict
from torch.nn.modules import TransformerEncoderLayer

from macsim.models.nn.misc import MLP
from macsim.models.nn.context import get_context_emb
from macsim.models.policy_args import TransformerParams
from macsim.models.nn.action_embeddings import get_action_emb
from macsim.envs.base import Environment, MultiAgentEnvironment
from macsim.models.encoder.base import MatNetEncoderOutput



class AgentActionPointer(nn.Module):
    """Decodes a job-machine pair given job and machine embeddings"""

    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.embed_dim = params.embed_dim
        self.num_heads = params.num_heads
        self.input_dim = params.embed_dim

        # self.final_trf_block = TransformerEncoderLayer(
        #     d_model=self.input_dim,
        #     nhead=params.num_heads,
        #     dim_feedforward=params.feed_forward_hidden,
        #     dropout=params.dropout,
        #     activation=params.activation,
        #     norm_first=params.norm_first,
        #     batch_first=True,
        # )

        self.mlp = MLP(
            input_dim=self.input_dim,
            output_dim=1,
            num_neurons=[self.input_dim] * params.num_decoder_ff_layers,
            hidden_act=params.activation,
        )

        self.is_multiagent_policy = params.is_multiagent_policy
        self.agent_emb = get_context_emb(params)
        self.action_emb = get_action_emb(params)

    def forward(self, embeddings: MatNetEncoderOutput, td: TensorDict, env: Union[Environment, MultiAgentEnvironment]):

        node_emb = self.action_emb(embeddings, td)
        agent_emb, _ = self.agent_emb(embeddings, td)
        
        # (bs, n_ma, 1 + n_jobs, 2*emb)
        agent_action_embs = agent_action_emb_combine(node_emb, agent_emb)
        # self attention
        # agent_action_embs = self._sa_block(agent_action_embs, td, env)

        # (bs, ma, jobs+1)
        agent_action_logits = self.mlp(agent_action_embs).squeeze(-1)
        return agent_action_logits
    

    # def _sa_block(self, job_ma_embs: torch.Tensor, td: TensorDict, env: Union[Environment, MultiAgentEnvironment]):
    #     bs, nm, nj, emb = job_ma_embs.shape

    #     attn_mask = self._get_attn_mask(td, env)

    #     # transformer layer over final embeddings
    #     job_ma_embs = self.final_trf_block(
    #         src=rearrange(job_ma_embs, "b m j d -> b (m j) d"), 
    #         src_mask=attn_mask
    #     )
    #     job_ma_embs = job_ma_embs.view(bs, nm, nj, emb)
    #     return job_ma_embs
    

    def _get_attn_mask(self, td: torch.Tensor, env: Union[Environment, MultiAgentEnvironment]):
        mask = td["action_mask"].clone()
        # NOTE in multiagent settings, wait op is a valid action, thus should attend to other ops
        if self.is_multiagent_policy:
            mask[..., env.wait_op_idx] = True
        # !!!In TransformerEncoderLayer, True mean NOT attent!!!
        mask = ~rearrange(mask, "b m j -> b (m j)")
        # get statistics
        bs, n_actions = mask.shape
        # expand self
        attn_mask = (
            mask
            .unsqueeze(1)
            .expand(bs, n_actions, n_actions)
            .contiguous()
        )
        # make all actions attend to at least themselves
        attn_mask = attn_mask.diagonal_scatter(
            torch.full_like(mask, fill_value=False),
            dim1=-2, dim2=-1
        )
        # make head dim
        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
        return attn_mask
    


def agent_action_emb_combine(action_emb, agent_emb):
    n_ma = agent_emb.size(1)
    # (bs, n_agents, n_actions, emb)
    action_emb_expanded = action_emb.unsqueeze(1).expand(-1, n_ma, -1, -1)
    agent_emb_expanded = agent_emb.unsqueeze(2).expand_as(action_emb_expanded)
    # (bs, n_agents, n_actions, 2*emb)
    # h_actions = torch.cat((agent_emb_expanded, action_emb_expanded), dim=-1)
    h_actions = action_emb_expanded + agent_emb_expanded
    return h_actions
