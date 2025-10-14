import abc
from typing import Union

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict

from macsim.utils.config import PolicyParams
from macsim.models.nn.misc import MaskedPooling
from macsim.models.policy_args import TransformerParams
from macsim.models.encoder.base import MatNetEncoderOutput
from macsim.models.decoder.mlp import agent_action_emb_combine
from macsim.models.nn.action_embeddings import get_action_emb


def get_pooling_layer(params: PolicyParams):
    mapping = {
        "fjsp": AgentActionPoolingLayer,
        "ma_fjsp": AgentActionPoolingLayer,
        "ffsp": AgentActionPoolingLayer,
        "ma_ffsp": AgentActionPoolingLayer,
    }

    pooling_layer = mapping.get(params.env.env, default=AgentActionPoolingLayer)(params)
    return pooling_layer


class BasePoolingLayer(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, params: PolicyParams):
        super().__init__()
        self.params = params

    @abc.abstractmethod
    def forward(self, embeddings: MatNetEncoderOutput, td: TensorDict, env) -> torch.Tensor:
        ...


class ActionPoolingLayer(BasePoolingLayer):
    def __init__(self, params: PolicyParams):
        super().__init__(params)
        self.action_emb = get_action_emb(params)
        self.pooling = MaskedPooling(params)


    def forward(
        self, 
        embeddings: MatNetEncoderOutput, 
        td: TensorDict,
        env
    ):
        # (bs, n_actions, emb)
        action_emb = self.action_emb(embeddings, td)
        # True means to mask action embedding in pooling operation
        attn_mask = ~td["action_mask"]
        assert attn_mask.dim() == 2, "can't use ActionPoolingLayer when an agent dimension exist. Use AgentActionPoolingLayer instead"
        # (bs, emb)
        pooled_emb = self.pooling(action_emb, attn_mask)
        return pooled_emb
    


class AgentActionPoolingLayer(BasePoolingLayer):

    def __init__(self, params: TransformerParams) -> None:
        super().__init__(params)
        self.action_emb = get_action_emb(params)
        self.compress = nn.Linear(2 * params.embed_dim, params.embed_dim, bias=params.bias)
        self.pooling = MaskedPooling(params)

    def forward(
        self, 
        embeddings: MatNetEncoderOutput, 
        td: TensorDict, 
        env
    ):
        # (bs, n_actions, emb)
        action_emb = self.action_emb(embeddings, td)
        # (bs, n_agents, emb)
        agent_emb = embeddings["agents"]
        # (bs, n_agents, n_actions, emb)
        agent_action_emb = self.compress(agent_action_emb_combine(action_emb, agent_emb))
        agent_action_emb_flat = rearrange(agent_action_emb, "b m j d -> b (m j) d")
        # self attetion
        attn_mask = ~td["action_mask"]
        attn_mask_flat = rearrange(attn_mask, "b m j -> b (m j)")
        # (bs, n_agents * n_actions, emb) -> (bs, emb)
        pooled_emb = self.pooling(agent_action_emb_flat, attn_mask_flat)
        return pooled_emb
