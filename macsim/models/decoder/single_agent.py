import torch
import torch.nn as nn

from tensordict import TensorDict
from typing import Tuple
from torch.distributions import Categorical

from macsim.envs.base import Environment
from macsim.utils.ops import gather_by_index
from macsim.models.policy_args import PolicyParams, TransformerParams
from macsim.models.encoder.base import MatNetEncoderOutput

from .attn import AttentionPointer
from .mlp import AgentActionPointer
from .base import BaseDecoder, AgentAction


__all__ = [
    "SingleAgentDecoder",
    "SingleAgentAttnDecoder",
    "SingleAgentMLPDecoder",
]


class SingleAgentDecoder(BaseDecoder):
    """Baseclass for decoders outputting a single agent action"""
    def __init__(self, params: PolicyParams, pointer: nn.Module = None) -> None:
        super().__init__(params, pointer=pointer)

    def forward(
        self, 
        embeddings: MatNetEncoderOutput, 
        td: TensorDict, 
        env: Environment, 
        mask: torch.Tensor = None,
        return_logits: bool = False,
        **dec_strategy_kwargs
    ):
        if mask is None:
            mask = td["action_mask"].clone()
        logits = self.pointer(embeddings, td, env)
        logp = self._logits_to_logp(logits, mask)
        action, logp = self.dec_strategy.step(logp, **dec_strategy_kwargs)
        action = self._translate_action(action, td, logp)
        if return_logits:
            td.set("logits", logits)
        # insert action td
        td.set("action", action)
        return td
    
    def get_logp_of_action(self, embeddings: TensorDict, td: TensorDict, env: Environment):
        mask = td["action_mask"].clone()
        logits = self.pointer(embeddings, td, env)
        logp = self._logits_to_logp(logits, mask)
        action_logp = gather_by_index(logp, td["action"]["idx"])
        dist_entropys = Categorical(logp.exp()).entropy()
        return action_logp, dist_entropys, None  # no mask due to padding in single agent settings
    
    def _logits_to_logp(self, logits: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = logits.size(0)
        flat_logits = logits.contiguous().view(bs, -1)
        flat_mask = mask.contiguous().view(bs, -1)
        logp = self.dec_strategy.logits_to_logp(logits=flat_logits, mask=flat_mask)
        return logp
    
    def _translate_action(self, action: torch.Tensor, td: TensorDict, logp: torch.Tensor = None) -> AgentAction:
        # translate and store action
        n_actions = td["action_mask"].size(-1)
        selected_agent = action // n_actions
        selected_job = action % n_actions
        action = TensorDict(
            {
                "idx": action, 
                "node": selected_job, 
                "agent": selected_agent,
            },
            batch_size=td.batch_size
        )
        if logp is not None:
            action.set("logp", logp)
        return action



class SingleAgentMLPDecoder(SingleAgentDecoder):
    def __init__(self, params: TransformerParams) -> None:
        pointer = AgentActionPointer(params)
        super().__init__(pointer, params)


class SingleAgentAttnDecoder(SingleAgentDecoder):
    def __init__(self, params: TransformerParams) -> None:
        pointer = AttentionPointer(params)
        super().__init__(pointer, params)

    def pre_rollout_hook(self, td, env, embeddings, store_trajectories: bool = False):
        td, env, embeddings = super().pre_rollout_hook(td, env, embeddings, store_trajectories)
        if not self.stepwise_encoding:
            self.pointer.compute_cache(embeddings, td)
        return td, env, embeddings