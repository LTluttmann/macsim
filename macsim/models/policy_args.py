from dataclasses import dataclass, field
from rl4co.utils import pylogger

from macsim.utils.config import PolicyParams

log = pylogger.get_pylogger(__name__)


@dataclass(kw_only=True)
class TransformerParams(PolicyParams):
    policy: str = "transformer"
    num_heads: int = 8
    feed_forward_hidden: int = None
    qkv_dim: int = field(init=False)
    input_dropout: float = 0.1 # dropout after positional encoding
    activation: str = "gelu"
    norm_first: bool = True # True
    use_block_attn: bool = False
    is_multiagent_policy: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.feed_forward_hidden = self.feed_forward_hidden or 2*self.embed_dim
        self.qkv_dim = self.embed_dim // self.num_heads
        assert self.embed_dim % self.num_heads == 0, "self.kdim must be divisible by num_heads"


@dataclass(kw_only=True)
class MatNetParams(TransformerParams):
    policy: str = "matnet"
    ms_hidden_dim: int = None
    mask_no_edge: bool = True
    param_sharing: bool = True
    cost_mat_dims: int = 1
    chunk_ms_scores_batch: int = 0
    ms_scores_softmax_temp: float = 1.0
    ms_scores_tanh_clip: float = 0.0
    def __post_init__(self):
        super().__post_init__()
        self.ms_hidden_dim = self.ms_hidden_dim or self.qkv_dim


@dataclass(kw_only=True)
class SingleAgentMlpPolicyParams(MatNetParams):
    policy: str = "sa_mlp"
    encoder: str = "matnet"
    decoder: str = "mlp"
    is_multiagent_policy: bool = False
    num_decoder_ff_layers: int = 1
    use_context_emb: bool = False # no effect but put it here to avoid init issues


@dataclass(kw_only=True)
class SingleAgentAttnPolicyParams(MatNetParams):
    policy: str = "sa_attn"
    encoder: str = "matnet"
    decoder: str = "attn"
    is_multiagent_policy: bool = False
    use_decoder_attn_mask: bool = True
    use_communication: bool = False
    use_rezero: bool = False
    use_context_emb: bool = False


@dataclass(kw_only=True)
class MacsimMlpParams(MatNetParams):
    policy: str = "macsim_mlp"
    encoder: str = "matnet"
    decoder: str = "mlp"
    is_multiagent_policy: bool = True
    use_communication: bool = False
    num_decoder_ff_layers: int = 1
    ranking_strategy: str = "learned"
    use_context_emb: bool = False # no effect but put it here to avoid init issues
    

@dataclass(kw_only=True)
class MacsimParams(MacsimMlpParams):
    policy: str = "macsim"
    encoder: str = "matnet"
    decoder: str = "attn"
    use_decoder_attn_mask: bool = True
    use_rezero: bool = False
    
