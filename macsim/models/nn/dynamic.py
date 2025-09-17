import torch
import torch.nn as nn
from tensordict import TensorDict
from macsim.utils.config import ModelParams


def get_dynamic_emb(params: ModelParams, key: str = None) -> nn.Module:
    # if we encode stepwise, dynamic embeddings are not needed
    if params.stepwise_encoding:
        return StaticEmbedding()
    
    raise NotImplementedError

    EmbCls = emb_registry[params.policy.env.env]
    
    if key is not None:
        EmbCls = EmbCls[key]

    return EmbCls(params)


class StaticEmbedding(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td: TensorDict, emb: torch.Tensor):
        return 0, 0, 0
