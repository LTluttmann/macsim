import abc
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import TypedDict, Tuple
from torchrl.data import LazyTensorStorage
from torch.distributions import Categorical

from macsim.envs.base import Environment
from macsim.utils.ops import gather_by_index
from macsim.utils.config import DecodingConfig, PolicyParams
from macsim.decoding.strategies import DecodingStrategy, get_decode_type


from .attn import MultiAgentAttentionPointer, AttentionPointer
from .mlp import AgentActionPointer



class AgentAction(TypedDict):
    idx: torch.Tensor
    node: torch.Tensor
    agent: torch.Tensor


def get_pointer_network(params: PolicyParams):
    single_pointer_map = {
        "attn": AttentionPointer,
        "mlp": AgentActionPointer,
    }
    multi_pointer_map = {
        "attn": MultiAgentAttentionPointer,
        "mlp": AgentActionPointer,
    }
    pointer_map = multi_pointer_map if params.is_multiagent_policy else single_pointer_map
    net = pointer_map[params.decoder](params)
    return net


class BaseDecoder(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, params: PolicyParams, pointer: nn.Module = None) -> None:
        super().__init__() 
        self.pointer = pointer or get_pointer_network(params)
        self.dec_strategy: DecodingStrategy = None
        self.stepwise_encoding = params.stepwise_encoding

    def maybe_cache(self, embeddings: TensorDict, td: TensorDict):
        if not self.stepwise_encoding and hasattr(self.pointer, "compute_cache"):
            self.pointer.compute_cache(embeddings, td)

    def pre_rollout_hook(self, td: TensorDict, env: Environment) -> Tuple[TensorDict, Environment]:
        # logic to be applied after first encoder forward pass
        td, env = self.dec_strategy.setup(td, env)
        return td, env

    def post_rollout_hook(
        self, 
        td: TensorDict, 
        storage: LazyTensorStorage = None
    ):
        td, storage = self.dec_strategy.post_decoding_hook(td, storage)
        return td, storage

    def _set_decode_strategy(self, decoding_params: DecodingConfig):
        self.dec_strategy = get_decode_type(decoding_params)

    @abc.abstractmethod
    def forward(
        self, 
        embeddings: TensorDict, 
        td: TensorDict, 
        env: Environment, 
        return_logits: bool = False, 
        return_logp: bool = False
    ) -> TensorDict:
        pass

    @abc.abstractmethod
    def get_logp_of_action(self, embeddings: TensorDict, td: TensorDict, env):
        pass

    def get_logits_of_action(self, embeddings: TensorDict, td: TensorDict, env):
        bs = embeddings.size(0)
        logits = self.get_logits(embeddings, td, env)
        action_logits = gather_by_index(logits.view(bs, -1), td["action"]["idx"])
        dist_entropys = Categorical(logits=logits).entropy()
        return action_logits, dist_entropys, None 

    def get_logits(self, embeddings: TensorDict, td: TensorDict, env):
        logits = self.pointer(embeddings, td, env)
        return logits
