from macsim.utils.config import PolicyParams

from .matnet import MatNetEncoder
from .am import AttentionModelEncoder


__all__ = [
    "get_encoder_network",
    "MatNetEncoder",
    "AttentionModelEncoder"
]


def get_encoder_network(params: PolicyParams):
    encoder_map = {
        "am": AttentionModelEncoder,
        "matnet": MatNetEncoder
    }
    net = encoder_map[params.encoder](params)
    return net