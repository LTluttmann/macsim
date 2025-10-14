import torch
import torch.nn as nn

from torch import Tensor
from typing import Union
from einops import rearrange
from tensordict import TensorDict
from torch.distributions import Categorical

from macsim.utils.ops import gather_by_index

from macsim.envs.base import MultiAgentEnvironment
from macsim.models.decoder.base import AgentAction
from macsim.models.encoder.base import MatNetEncoderOutput
from macsim.models.policy_args import MacsimParams, MacsimMlpParams

from .base import BaseDecoder
from .mlp import AgentActionPointer
from .attn import MultiAgentAttentionPointer


__all__ = [
    "MultiAgentDecoder",
    "MultiAgentAttnDecoder",
    "MultiAgentMLPDecoder",
]


class MultiAgentDecoder(BaseDecoder):
    def __init__(self, params: Union[MacsimParams, MacsimMlpParams], pointer: nn.Module = None) -> None:
        super().__init__(params, pointer=pointer)
        self.eval_multistep = params.eval_multistep
        self.eval_per_agent = params.eval_per_agent
        self.use_masked_softmax_for_eval = params.use_masked_softmax_for_eval
        self.pad = params.eval_multistep
        self.max_steps = params.max_steps or float("inf")
        self.ranking_strategy = params.ranking_strategy
    

    def _init_actions(self, max_steps: int, td: TensorDict) -> AgentAction:
        batch_size = td.size(0)
        actions = TensorDict(
            {
                "idx": torch.zeros((batch_size, max_steps), device=td.device, dtype=torch.long),
                "node": torch.zeros((batch_size, max_steps), device=td.device, dtype=torch.long),
                "agent": torch.zeros((batch_size, max_steps), device=td.device, dtype=torch.long),
            },
            batch_size=(batch_size, max_steps),
        )
        return actions
    

    def forward(
        self, 
        embeddings: MatNetEncoderOutput, 
        td: TensorDict, 
        env: MultiAgentEnvironment, 
        mask: torch.Tensor = None,
        return_logits: bool = False,
        **dec_strategy_kwargs
    ):
        bs, num_agents, _ = td["action_mask"].shape
        steps = min(self.max_steps, num_agents)
        # get logits and mask
        logits = self.pointer(embeddings, td, env)
        # in mask, True means feasible
        if mask is None:
            mask = td["action_mask"].clone()
        # initialize action buffer
        td["action"] = self._init_actions(steps, td)
        busy_agents = torch.full(size=(bs, num_agents), fill_value=False, device=td.device)
        # agent rollout
        for i in range(steps):
            logps = self._logits_to_logp(logits, mask)
            # decoding step and action translation
            action, logp = self.dec_strategy.step(logps, **dec_strategy_kwargs)
            action = self._translate_action(action, td, logp)
            td["action"][:, i] = action
            busy_agents.scatter_(1, action["agent"][:, None], True)
            mask = env.update_mask(mask, action, td, busy_agents)

        if return_logits:
            td["logits"] = logits
        return td

    
    def get_logp_of_action(self,  embeddings: TensorDict, td: TensorDict, env: MultiAgentEnvironment):
        if self.use_masked_softmax_for_eval:
            return self._get_masked_logp_of_actions(embeddings, td, env)
        else:
            return self._get_unmasked_logp_of_actions(embeddings, td, env)

    def _get_unmasked_logp_of_actions(self, embeddings: TensorDict, td: TensorDict, env: MultiAgentEnvironment):
        # get flat action indices
        action_indices = td["action"]["idx"]
        # get logits and mask once
        logits = self.pointer(embeddings, td, env)
        mask = td["action_mask"].clone()
        # NOTE unmask "noop" to avoid nans and infs: in principle, every agent always has the chance to wait
        mask[..., env.wait_op_idx] = True
        # (bs, num_actions)
        logp = self._logits_to_logp(logits, mask)
        #(bs, num_agents)
        selected_logp = gather_by_index(logp, action_indices, dim=1, squeeze=False)
        # get entropy
        if self.eval_multistep and selected_logp.dim() == 2:
            #(bs, num_agents)
            dist_entropys = Categorical(logits=logits).entropy()
        else:
            #(bs)
            dist_entropys = Categorical(probs=logp.exp()).entropy().unsqueeze(1)           

        if self.eval_multistep:
            # mask loss for agents which have only one available option
            # (bs, n_agents)
            loss_mask = gather_by_index(mask.sum(-1)==1, td["action"]["agent"], dim=1)
        else:
            loss_mask = None
        return selected_logp, dist_entropys, loss_mask
       
    def _get_masked_logp_of_actions(self, embeddings: TensorDict, td: TensorDict, env: MultiAgentEnvironment):
        bs, _num_agents, num_actions = td["action_mask"].shape
        num_agents = td["action"]["idx"].size(1)
        assert _num_agents == num_agents, "what happened here?"
        # get logits and mask once
        logits = self.pointer(embeddings, td, env)
        mask: Tensor = td["action_mask"].clone()

        if self.eval_per_agent:
            idle_agents = torch.full(size=(bs, num_agents), fill_value=True, device=td.device)
            for i in range(num_agents):
                curr_action = td["action"][:, i]
                agent = curr_action["agent"]
                node = curr_action["node"]
                idle_agents.scatter_(1, agent[:, None], False)
                node = node.repeat_interleave(idle_agents.sum(-1))
                b, a = torch.where(idle_agents)
                mask[b, a, node] = False
            # unmask "no op" to avoid nans and infs: in principle, every agent always has the chance to wait
            mask[..., env.wait_op_idx] = True
            loss_mask = gather_by_index(mask.sum(-1)==1, td["action"]["agent"], dim=1)
            # get flat action indices
            action_indices = td["action"]["idx"]
            # (bs, num_actions)
            logp = self._logits_to_logp(logits, mask)
            #(bs, num_agents)
            selected_logps = gather_by_index(logp, action_indices, dim=1, squeeze=False)

        else:
            # mask for the loss. True means to mask the corresponding entry in the loss function
            loss_mask = torch.full((bs, num_agents), fill_value=False, device=mask.device)
            selected_logps = []
            busy_agents = torch.full(size=(bs, num_agents), fill_value=False, device=td.device)
            for i in range(num_agents):
                # unmask "no op" to avoid nans and infs: in principle, every agent always has the chance to wait
                mask[..., env.wait_op_idx] = True
                curr_action = td["action"][:, i]
                agent = curr_action["agent"]
                idx = curr_action["idx"]
                # mask agents that have only one feasible action
                loss_mask.scatter_(1, agent[:, None], gather_by_index(mask, agent, dim=1).sum(1, keepdims=True) == 1)
                # (bs, num_actions)
                logp = self._logits_to_logp(logits, mask)
                #(bs, 1)
                selected_logp = gather_by_index(logp, idx, dim=1, squeeze=False)
                selected_logps.append(selected_logp)
                # mask all predecessor actions for coming successor (like in listmle)
                # mask = mask.scatter(1, agent[:, None, None].expand(bs, 1, num_actions), False)
                busy_agents.scatter_(1, agent[:, None], True)
                mask = env.update_mask(mask, curr_action, td, busy_agents)

            selected_logps = torch.cat(selected_logps, dim=1)

        assert selected_logps.isfinite().all()
        dist_entropys = Categorical(logits=logits).entropy()
        return selected_logps, dist_entropys, loss_mask

    def _prepare_logits_for_rollout(self, logits: torch.Tensor, mask: torch.Tensor):
        if self.ranking_strategy != "learned":
            batch_idx = torch.arange(logits.size(0), device=logits.device)
            new_mask = mask.clone()
            if self.ranking_strategy == "index":
                # follow the agent order defined by the set of agents, i.e. m=1,...,M
                next_agent = mask.any(-1).float().argmax(1)
            elif self.ranking_strategy == "random":
                # select a random action to perform an action
                valid_agent = mask.any(-1)
                next_agent = (
                    torch.rand_like(valid_agent.float())
                    .masked_fill(~valid_agent, float("-inf"))
                    .softmax(dim=1)
                    .multinomial(1)
                    .squeeze(1)
                )
            else:
                raise ValueError
            # mask all actions in the new mask
            new_mask[...] = False  
            # ...except for the selected agent
            new_mask[batch_idx, next_agent, :] = mask[batch_idx, next_agent, :]
            mask = rearrange(new_mask, "b m j -> b (m j)")
        else:
            mask = rearrange(mask, "b m j -> b (m j)")

        logits = rearrange(logits, "b m j -> b (m j)")
        return logits, mask

    def _logits_to_logp(self, logits: torch.Tensor, mask: torch.Tensor):
        if torch.is_grad_enabled() and self.eval_per_agent:
            # when training we evaluate on a per agent basis
            # perform softmax per agent
            logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)
            # flatten logp for selection
            logp = rearrange(logp, "b m j -> b (m j)")

        elif torch.is_grad_enabled():
            # when rolling out, we sample iteratively from flattened prob dist
            logp = self.dec_strategy.logits_to_logp(logits=logits.flatten(1,2), mask=mask.flatten(1,2))
        else:
            # when rolling out, we sample iteratively from flattened prob dist
            logits, mask = self._prepare_logits_for_rollout(logits, mask)
            logp = self.dec_strategy.logits_to_logp(logits=logits, mask=mask)
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


class MultiAgentAttnDecoder(MultiAgentDecoder):
    def __init__(self, params: MacsimParams) -> None:
        pointer = MultiAgentAttentionPointer(params)
        super().__init__(params, pointer=pointer)
        self.pointer: MultiAgentAttentionPointer
    

class MultiAgentMLPDecoder(MultiAgentDecoder):
    def __init__(self, params: MacsimMlpParams) -> None:
        pointer = AgentActionPointer(params)
        super().__init__(params, pointer=pointer)
