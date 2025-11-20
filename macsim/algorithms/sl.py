import copy

import torch
import torch.nn as nn
from tensordict import TensorDict
from omegaconf import DictConfig
from torchrl.data import LazyTensorStorage

from macsim.envs.base import Environment
from macsim.algorithms.model_args import SelfImprovementParameters
from macsim.algorithms.base import LearningAlgorithmWithReplayBuffer
from macsim.utils.config import TrainingParams, ValidationParams, TestParams

from macsim.utils.ops import unbatchify, gather_by_index
from macsim.algorithms.losses import loss_map, calc_adv_weights, listnet_loss
from .utils import count_conflicting_agents_per_step

_float = torch.get_default_dtype()


class SelfImprovement(LearningAlgorithmWithReplayBuffer):

    name = "sl"

    def __init__(
        self, 
        env: Environment,
        policy: nn.Module,
        model_params: SelfImprovementParameters,
        train_params: TrainingParams,
        val_params: ValidationParams,
        test_params: TestParams
    ) -> None:

        super().__init__(
            env=env,
            policy=policy,
            model_params=model_params,
            train_params=train_params,
            val_params=val_params,
            test_params=test_params
        )

        self.model_params: SelfImprovementParameters
        # for lookback / revert operations save the current state
        if self.model_params.lookback_intervals:
            self.old_tgt_policy_state = copy.deepcopy(self.policy.state_dict())

        # self.num_starts = self.setup_parameter(model_params.num_starts, dtype=int)
        self.num_starts = model_params.ref_model_decoding.num_samples
        self.entropy_coef = self.setup_parameter(model_params.entropy_coef, dtype=float)
        self.penalty_coef = self.setup_parameter(model_params.penalty_coef, dtype=float)
        self.lookback_intervals = self.setup_parameter(
            model_params.lookback_intervals,
            dtype=lambda x: int(x) or float("inf")
        )
       
        self.loss_fn = loss_map[model_params.loss]
        self.update_after_every_batch = model_params.update_after_every_batch
        self.return_logits = self.loss_fn.__name__ == listnet_loss.__name__
        if self.update_after_every_batch:
            # if we update after every batch, we always need to flush the buffer afterwards, otherwise
            # examples of earlier batches are seen more often during training then thoss of later batches
            self.always_clear_buffer = True
        else:
            self.always_clear_buffer = model_params.always_clear_buffer

    def _update_rb_sampler_probs(self):
        if len(self.rb_ensamble.storage) == 1:
            return
        rb_distribution = torch.tensor([len(x) for x in self.rb_ensamble.storage], device=self.device)
        rb_distribution = torch.where(rb_distribution == 0, -torch.inf, rb_distribution)
        rb_distribution = torch.softmax(rb_distribution / 1000, dim=0)
        self.rb_ensamble.sampler.p = rb_distribution

    def get_buffer_id(self, sub_td):
        return list(self.rbs.keys())[sub_td["index"]["buffer_ids"][0]]

    def _update(self):
        losses = []
        entropies = []
        self._update_rb_sampler_probs()
        for _ in range(self.num_training_batches):
            sub_td = self.rb_ensamble.sample().clone().to(self.device).squeeze(0)
            buffer_id = self.get_buffer_id(sub_td)
            # get logp of target policy for pseudo expert actions 
            logp, _, entropy, mask = self.policy.evaluate(sub_td, self.env)
            # (bs)
            loss = self.loss_fn(logp, entropy, sub_td, mask=mask, entropy_coef=self.entropy_coef)
            # (bs)
            adv_weights = sub_td.get("adv_weights", None)
            if adv_weights is not None:
                loss = loss * adv_weights
            # (bs)
            weight = sub_td.get("_weight", None)
            # aggregate loss: bs -> 1
            if weight is not None:
                loss = (loss * weight).sum() / weight.sum()
            else:
                loss = loss.mean()
            self.manual_opt_step(loss, buffer_id)
            losses.append(loss.detach())
            entropies.append(entropy.mean())

        loss = torch.stack(losses, dim=0).mean()
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        entropy = torch.stack(entropies, dim=0).mean()
        self.log("train/entropy", entropy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.always_clear_buffer:
            self.clear_buffer()


    @torch.no_grad
    def _collect_experience(self, next_td: TensorDict, instance_id: str):
        # init storage and perform rollout
        state_stack = LazyTensorStorage(self.env.max_num_steps, device=self.model_params.buffer_storage_device)
        done_tds, state_stack = self.ref_policy(next_td, self.env, storage=state_stack, return_logits=self.return_logits)
        
        # (bs, #samples, #steps)
        state_stack = unbatchify(state_stack, self.num_starts)
        # (bs, #samples)
        # rewards = self.env.get_reward(done_tds).to(state_stack.device)
        rewards = unbatchify(done_tds["reward"], self.num_starts).to(state_stack.device)
        steps = (~state_stack["done"]).sum(-1).to(_float)
        # (bs); add very small noise to randomize tie breaks
        best_idx = torch.argmax(rewards - self.penalty_coef * steps + torch.rand_like(rewards) * 1e-9, dim=1)
        best_reward = gather_by_index(rewards, best_idx)
        # store rollout results
        self.max_rewards.append(best_reward)
        self.train_rewards.append(rewards.mean(1))
        self.reward_std.append(rewards.std(1) / (-rewards.mean(1) + 1e-6))
        self.avg_steps.append(steps.mean())

        # (bs)
        best_states = gather_by_index(state_stack, best_idx, dim=1)
        del state_stack
        torch.cuda.empty_cache()

        if self.model_params.use_advantage_weights:
            advantage = best_reward - rewards.mean(1, keepdim=False)
            adv_weights = calc_adv_weights(advantage, temp=2.5, weight_clip=10.)
            best_states["adv_weights"] = adv_weights.unsqueeze(1).expand(*best_states.shape)

        # flatten so that every step is an experience
        best_states = best_states.reshape(-1).contiguous()
        # filter out steps where the instance is already in terminal state. There is nothing to learn from
        best_states = best_states[~best_states["done"]]
        if not self.model_params.eval_multistep and self.ref_policy.ma_policy:
            # if we have a multi-agent policy (generating M action per step) but want to train the model only
            # on a single action, we need to flatten the agent index in the batch dimension 
            best_states = self.flatten_multi_action_td(best_states)
        if hasattr(self.env, "augment_states"):
            best_states = self.env.augment_states(best_states)
        # save to buffer
        self.rbs[instance_id].extend(best_states)

    def training_step(self, batch: TensorDict, batch_idx: int):
        orig_state, instance_id = self.env.reset(batch)
        bs = orig_state.size(0)

        if isinstance(self.rollout_batch_size, (dict, DictConfig)):
            rollout_batch_size = self.rollout_batch_size.get(instance_id)
        else:
            rollout_batch_size = self.rollout_batch_size

        # data gathering loop
        for i in range(0, bs, rollout_batch_size):
            next_td = orig_state[i : i + rollout_batch_size]
            self._collect_experience(next_td, instance_id)

        if self.trainer.is_last_batch or self.update_after_every_batch:
            self._update()


    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        state, instance_id = self.env.reset(batch)
        eval_conflicts = self.val_params.eval_conflicts and self.policy.ma_policy
        out = self.policy.generate(state, self.env, return_logits=eval_conflicts)

        if eval_conflicts:
            reward = out[0]["reward"]
            logits = out[1]["logits"]
            skip_tok_idx = logits.size(-1) - 1 if self.env.wait_op_idx == -1 else 0
            actions = torch.argmax(logits, dim=-1)
            agent_conf_per_step = count_conflicting_agents_per_step(actions, skip_idx=skip_tok_idx)
            self.log(
                "val/n_agent_conflicts", 
                agent_conf_per_step.float().mean(), 
                prog_bar=False, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True,
                add_dataloader_idx=True,
            )
        else:
            reward = out["reward"]

        val_set_name = self.val_set_names[dataloader_idx]

        if val_set_name == "synthetic":
            self.validation_step_rewards[instance_id].append(reward)
            metric_name = f"val/synthetic/{instance_id}/reward"
        else:
            metric_name = f"val/files/{val_set_name}/reward"

        self.log(
            metric_name, 
            reward.mean(), 
            prog_bar=True, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True,
            add_dataloader_idx=False,
        )


    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        
        _, improved = super().on_validation_epoch_end()

        if improved:
            self.pylogger.info("Updating Reference Policy...")
            self.ref_policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))

            if not self.always_clear_buffer:
                self.clear_buffer()

        if (
            ((self.current_epoch + 1) % self.lookback_intervals == 0) and 
            (self.current_epoch + self.lookback_intervals < self.trainer.max_epochs)
        ):
            self.policy.load_state_dict(copy.deepcopy(self.old_tgt_policy_state))
            self.old_tgt_policy_state = copy.deepcopy(self.ref_policy.state_dict())

        if self.model_params.ref_policy_warmup == (self.current_epoch + 1):
            self._reset_target_policy()

   
    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        self.avg_steps = []
        self.reward_std = []
        self.train_rewards = []
        self.max_rewards = []

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        avg_steps = torch.stack(self.avg_steps, 0).mean()
        reward_std = torch.cat(self.reward_std, 0).mean()
        train_reward = torch.cat(self.train_rewards, 0).mean()
        max_rewards = torch.cat(self.max_rewards, 0).mean()
        self.log("train/avg_steps", avg_steps, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/reward_std", reward_std, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/reward_mean", train_reward, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/max_rewards", max_rewards, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def _reset_target_policy(self):
        for layer in self.policy.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def flatten_multi_action_td(self, td: TensorDict):
        """This function flattens out the 'action dimension' in multi agent settings. This means,
        every action from a multi agent policy will be stored as a seperate experience in the
        resulting tensordict."""
        if not (td["action"]["idx"].dim() == 2 and td["action"]["idx"].size(1) > 1):
            # tensordict is already flat
            return td
        assert td.dim() == 1
        flat_td = td.clone()
        bs, n_actions = td["action"]["idx"].shape
        action: TensorDict = flat_td["action"].clone()
        action.batch_size = (bs, n_actions)
        # (n_actions, bs)
        flat_td = flat_td.unsqueeze(0).expand(n_actions, bs).contiguous()
        flat_td["action"] = action.permute(1, 0).contiguous()
        flat_td = flat_td.view(bs * n_actions)
        return flat_td
    
