import torch
import logging
from typing import Tuple
from torchrl.data import LazyTensorStorage
from tensordict.tensordict import TensorDict

from macsim.envs.base import Environment
from macsim.utils.config import DecodingConfig
from macsim.utils.ops import augment_or_batchify, unbatchify

from .utils import process_logits


log = logging.getLogger(__name__)


def get_decode_type(cfg: DecodingConfig):
    strategy_registry = {
        "greedy": Greedy,
        "sampling": Sampling,
        "multi_sampling": MultiSampleWithoutReplacement,
        "sample_from_topk": SampleFromTopK,
    }

    if cfg.decode_type not in strategy_registry:
        log.warning(
            f"Unknown decode type '{cfg.decode_type}'. Available decode types: {strategy_registry.keys()}. Defaulting to Sampling."
        )

    return strategy_registry.get(cfg.decode_type, Sampling)(cfg)


class DecodingStrategy:
    name = "base"

    def __init__(self, config: DecodingConfig) -> None:
        
        self.config = config
        self._temperature = config.temperature
        self._per_instance_tempertature = None


    @property
    def temperature(self):
        if self._per_instance_tempertature is not None:
            return self._per_instance_tempertature
        else:
            return self._temperature

    def patch_temp_for_hybrid_decoding(self, td: TensorDict, num_starts: int = None, eps: float = 1e-9):
        """This function sets the temperature for one of M samples (in multistart mode) to
        a very low floating point (eps) in order to enable greedy decoding for this sample.
        All other samples are generated via standard trajectory sampling using the temperature
        stored in self._temperature. 

        Note we allow to pass a num_starts argument, since multistart mode can also be triggered outside
        of a DecodingStartegy instance.

        Args:
        - bs: integer indicating the batch size.
        - Optional, device: the device to store the temperature tensor on.
        - Optional, num_starts: the number of samples generated per instance. The first one will be used for greedy decoding
        """
        aug_bs = td.size(0)
        num_starts = num_starts or self.config.num_samples
        bs = aug_bs // num_starts
        self.temperature = torch.cat((torch.full((aug_bs-bs,1), self._temperature), torch.full((bs,1), eps)), dim=0).to(td.device)
        
        
    @temperature.setter
    def temperature(self, value):
        if isinstance(value, (float, int)):
            self._temperature = value
        elif isinstance(value, torch.Tensor):
            self._per_instance_tempertature = value
        else:
            raise ValueError(f"Unknown type for temperature: {type(value)}")

    def _step(self, logp: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Must be implemented by subclass")

    def setup(self, td: TensorDict, env: Environment) -> Tuple[TensorDict, Environment]:
        # Multi-start decoding: first action is chosen by ad-hoc node selection
        if self.config.num_samples > 1:
            # Expand td to batch_size * num_starts
            td, num_starts = augment_or_batchify(td, env, self.config)
            if self.config.hybrid_decoding and num_starts > 1:
                # setting low temperature for first of num_starts batch rollouts effectively lets us generate a greedy rollout
                self.patch_temp_for_hybrid_decoding(td, num_starts=num_starts)
        return td, env

    def post_decoding_hook(
        self, 
        td: TensorDict, 
        storage: LazyTensorStorage = None
    ) -> Tuple[TensorDict, TensorDict]:
        # logic to be applied after a solution has been constructed
        if storage is not None:
            # tensordict of size (bs, num_steps)
            storage = storage[:len(storage)].permute(1,0).contiguous()

        if self.config.num_samples > 1 and self.config.select_best:
            batch_size = td.size(0) // self.config.num_samples
            best_idx = torch.argmax(unbatchify(td["reward"], self.config.num_samples), dim=1)
            flat_idx = torch.arange(batch_size, device=td.device) + best_idx * batch_size
            td = td[flat_idx].clone()
            if storage is not None:
                storage = storage[flat_idx].clone()

        return td, storage

    def step(self, logp: torch.Tensor, return_logp: bool = False, **kwargs) -> Tuple[TensorDict, torch.Tensor]:
        # assert not logp.isinf().all(1).any()
        selected_actions = self._step(logp, **kwargs)

        if return_logp:
            logps_of_actions = logp.gather(-1, selected_actions.unsqueeze(-1)).squeeze(-1)
        else:
            logps_of_actions = None
        return selected_actions, logps_of_actions

    def logits_to_logp(self, logits, mask=None):
        return process_logits(
            logits,
            mask=mask,
            temperature=self.temperature,
            top_p=self.config.top_p, 
            tanh_clipping=self.config.tanh_clipping,
        )


class Greedy(DecodingStrategy):
    name = "greedy"

    def _step(self, logp: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # [BS], [BS]
        selected = logp.argmax(1)
        return selected


class Sampling(DecodingStrategy):
    name = "sampling"

    def _step(self, logp: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, n_ops, *ma = logp.shape
        assert not logp.isnan().any()
        selected = logp.exp().multinomial(1).view(bs, *ma)
        return selected
    

class MultiSampleWithoutReplacement(DecodingStrategy):
    name = "multi_sampling"

    def __init__(self, config: DecodingConfig, num_samples) -> None:
        super().__init__(config)
        self.num_samples = num_samples

    def _step(self, logp: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, n_act = logp.shape
        logp_expanded = logp.view(bs,1,n_act).expand(bs,self.num_samples,n_act).contiguous()
        logp_expanded[:,1:,:] = logp_expanded[:,1:,:] - 10
        logp_expanded_flattened = logp_expanded.view(bs, -1)
        selected = logp_expanded_flattened.exp().multinomial(self.num_samples, replacement=False) % n_act
        return selected
    

class SampleFromTopK(DecodingStrategy):
    name = "sample_from_topk"

    def __init__(self, config: DecodingConfig, k) -> None:
        super().__init__(config)
        self.k = k

    def _step(self, logp: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, n_act = logp.shape
        k = min(self.k, n_act)
        topk_val, topk_ind = logp.topk(k, dim=-1)
        topk_selected = topk_val.exp().multinomial(1)
        selected = topk_ind.gather(1, topk_selected).squeeze(1)
        return selected


