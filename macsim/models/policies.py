import warnings
import torch
import torch.nn as nn

from typing import Tuple, Union
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage
from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from macsim.models.critic import Critic
from macsim.models.decoder.base import BaseDecoder
from macsim.models.decoder import MultiAgentDecoder, SingleAgentDecoder
from macsim.utils.utils import Registry
from macsim.envs.base import Environment
from macsim.utils.config import DecodingConfig
from macsim.models.policy_args import PolicyParams
from macsim.models.encoder import get_encoder_network


from .policy_args import *


policy_registry = Registry()


class SchedulingPolicy(nn.Module):
    
    _critic_cls = Critic

    def __init__(self, params: PolicyParams, encoder: nn.Module = None, decoder: BaseDecoder = None):
        super().__init__()  
        # Encoder
        self.encoder = encoder or get_encoder_network(params)
        # Decoder
        if decoder is None:
            if params.is_multiagent_policy:
                self.ma_policy = True
                decoder = MultiAgentDecoder(params)
            else:
                self.ma_policy = False
                decoder = SingleAgentDecoder(params)

        self.decoder = decoder
        self.params = params
        self.critic = self.get_critic(params)

    def get_critic(self, params: PolicyParams):
        if not params.use_critic:
            return None
        else:
            return self._critic_cls(params) 

    @classmethod
    def initialize(cls, params: PolicyParams) -> "SchedulingPolicy":
        PolicyCls = policy_registry.get(params.policy, cls)
        return PolicyCls(params)

    def _setup_storage(self, size, device = "cpu", **kwargs) -> LazyTensorStorage:
        return LazyTensorStorage(size, device=device, **kwargs)
    
    def forward(
        self, 
        td: TensorDict, 
        env: Environment, 
        return_logp: bool = False,
        return_logits: bool = False,
        return_trajectories: bool = False,
        storage: LazyTensorStorage = None,
        **storage_kwargs
    ) -> Union[TensorDict, Tuple[TensorDict, TensorDict]]:
        """Function for a full Policy Rollout"""
        next_td = td.clone()
        next_td, env = self.decoder.pre_rollout_hook(next_td, env)
        # encoding once
        embeddings = self.encoder(next_td)
        # setup optional trajectory buffer
        if storage is None and (return_trajectories or return_logp or return_logits):
            storage = self._setup_storage(env.max_num_steps, **storage_kwargs)
        # generation loop
        step = 0
        while not next_td["done"].all():
            # autoregressive decoding
            act_td = self.decoder(embeddings, next_td, env, return_logits=return_logits, return_logp=return_logp)
            if storage is not None:
                storage.set(slice(step, step+1), act_td.unsqueeze(0).clone())
            # step the env
            next_td = env.step(act_td)
            if self.params.stepwise_encoding:
                # optionally, re-encode
                embeddings = self.encoder(next_td)
            # increment step
            step += 1

        # prepare return td
        next_td["reward"] = env.get_reward(next_td)
        # postprocessing
        done_td, storage = self.decoder.post_rollout_hook(next_td, storage)
        # return storage or not
        if storage is not None:
            return done_td, storage
        return done_td
    

    def faster_rollout(
            self, 
            td: TensorDict, 
            env: Environment, 
            return_logits: bool = False,
            storage: TensorStorage = None,
        ) -> Union[TensorDict, Tuple[TensorDict, TensorDict]]:
        """Performs the same rollout as self.forward but excludes done instances in each iteration
        making it potentially faster, especially in multi-agent settings, where the number of decoding
        steps can significantly differ between instances.
        """
        if storage is not None:
            warnings.warn("Filling a storage in faster_rollout() is experimental and might lead to unintended results.")

        td = td.clone()
        td, env = self.decoder.pre_rollout_hook(td, env, embeddings)
        # encoding
        embeddings = self.encoder(td)
        step = 0
        batch_idx = torch.arange(td.size(0), device=td.device)
        done_idx = batch_idx[td["done"]]
        while not (td["done"].all() or td.size(0) == 0):
            # autoregressive decoding
            act_td = self.decoder(embeddings, td, env, return_logits=return_logits)
            # in the first iteration, we initialize the td buffer of completed instances.
            if step == 0:
                # NOTE: must happen after decoder to include action key
                done_tds = torch.empty_like(act_td)
                done_tds[done_idx] = td[td["done"]]

            # fill the storage if it is passed to the rollout function
            if storage is not None:
                # optionally save the state and the action of the still rolled out instances in a buffer
                storage.set((slice(step, step+1), batch_idx), act_td.unsqueeze(0).clone())
                if done_idx.size(0) > 0:
                    # Here we pad the storage of finished instances with their done state. 
                    # TODO there is a tensordict.repeat method in newer versions. Replace this with .repeat()
                    # NOTE we need to do this here. Doing it after env.step, maybe new instances are done and
                    # then we overwrite their act_state in the storage with the done_state. 
                    num_done = done_idx.size(0)
                    remaining_steps = storage.max_size - step
                    done_tds_exp = done_tds[done_idx].unsqueeze(0).expand(remaining_steps, num_done).contiguous().clone()
                    storage.set((slice(step, storage.max_size), done_idx), done_tds_exp)

            # step the env
            td = env.step(act_td)
            # get batch ids of instances that are completed
            done_idx = batch_idx[td["done"]]
            # write the states of the respective instances to the buffer
            done_tds[done_idx] = td[td["done"]]
            # update the set of remaining batch ids...
            batch_idx = batch_idx[~td["done"]]
            # ...as well as the remaining states
            td = td[~td["done"]]

            if self.params.stepwise_encoding and not td.size(0) == 0:
                embeddings = self.encoder(td)

            step +=1

        done_tds["reward"] = env.get_reward(done_tds)

        return done_tds


    def act(self, td, env, return_logits: bool = False, **decoder_kwargs) -> TensorDict:
        embeddings = self.encoder(td)
        td = self.decoder(embeddings, td, env, return_logits=return_logits, **decoder_kwargs)
        return td
    
    
    def n_step_rollout(self, td: TensorDict, env: Environment, n_steps: int, record_trajectory: bool = True, **decoder_kwargs):
        assert self.params.stepwise_encoding
        next_td = td.clone()
        state_stack = LazyTensorStorage(n_steps, device="auto") if record_trajectory else None
        for i in range(n_steps):
            act_td = self.act(next_td, env, **decoder_kwargs)
            
            if record_trajectory:
                state_stack.set(slice(i, i+1), act_td.unsqueeze(0).clone())

            next_td = env.step(act_td)
        
        if record_trajectory:
            state_stack = state_stack[:n_steps].permute(1,0)
            return next_td, state_stack
        
        return next_td

    def act_and_eval(self, td: TensorDict, env: Environment, return_logits = False, **decoder_kwargs) -> Tuple[TensorDict, torch.Tensor]:
        embeddings = self.encoder(td)
        # pred value via the value head
        if self.critic is not None:
            value_pred = self.critic(embeddings, td, env)
        else:
            value_pred = None

        td = self.decoder(embeddings, td, env, return_logits=return_logits, **decoder_kwargs)
        return td, value_pred
    
    def get_logits(self, td, env) -> torch.Tensor:
        # Encoder: get encoder output and initial embeddings from initial state
        embeddings = self.encoder(td)
        logits = self.decoder.get_logits(embeddings, td, env)
        return logits
    
    def evaluate(self, td, env, normalize=True):
        # Encoder: get encoder output and initial embeddings from initial state
        embeddings = self.encoder(td)
        # pred value via the value head
        if self.critic is not None:
            value_pred = self.critic(embeddings, td, env)
        else:
            value_pred = None

        if normalize:
            action_logprobs, entropies, mask = self.decoder.get_logp_of_action(embeddings, td, env)
        else:
            action_logprobs, entropies, mask = self.decoder.get_logits_of_action(embeddings, td, env)

        return action_logprobs, value_pred, entropies, mask

    def set_decode_type(self, decoding_params: DecodingConfig) -> None:
        self.decoder._set_decode_strategy(decoding_params)


    @torch.no_grad()
    def generate(self, td, env=None, **kwargs) -> TensorDict:
        is_training = self.training
        self.train(False)
        out = super().__call__(td, env, **kwargs)
        self.train(is_training)
        return out
    