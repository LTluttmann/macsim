import torch.nn as nn

from macsim.models.nn.misc import MLP
from macsim.utils.config import PolicyParams
from macsim.models.nn.pooling import get_pooling_layer


class Critic(nn.Module):
    def __init__(self, params: PolicyParams):
        super().__init__()
        self.pooling = get_pooling_layer(params)
        self.critic = MLP(params.embed_dim, 1, [params.embed_dim] * 2)

    def forward(self, embeddings, td, env):
        pooled_emb = self.pooling(embeddings, td, env)
        return self.critic(pooled_emb).squeeze(-1)


