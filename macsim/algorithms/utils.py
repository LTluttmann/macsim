import math
import torch
import random

from dataclasses import dataclass
from tensordict import TensorDict, pad
from typing import Generator, Optional, Callable, Union, Type, List
from torch.optim.lr_scheduler import _LRScheduler
from torchrl.data import LazyTensorStorage
from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    LazyTensorStorage,
    TensorDictReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    ReplayBuffer
)

Numeric = Union[int, float]


def make_replay_buffer(
        buffer_size: int, 
        device="cpu", 
        priority_key: str = None,
        prefetch: int = None,
        priority_alpha: float = 0.2,
        priority_beta: float = 0.1,
        pin_memory: bool = True
    ) -> ReplayBuffer:
    
    if device == "cpu":
        storage = LazyMemmapStorage(buffer_size, device="cpu")
    else:
        storage = LazyTensorStorage(buffer_size, device=device)

    if priority_key is None:
        rb = TensorDictReplayBuffer(
            storage=storage,
            pin_memory=pin_memory,
            prefetch=prefetch,
        )

    else:
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=priority_alpha, # how much to use prioritization
            beta=priority_beta, # how much to correct for bias through importance sampling
            storage=storage,
            priority_key=priority_key,
            pin_memory=pin_memory,
            prefetch=prefetch,
        )

    return rb


class SampleGenerator:
    def __init__(self, rb: ReplayBuffer, num_iter: Optional[int] = None, num_samples: Optional[int] = None):
        self.rb = rb
        self.num_iter = num_iter
        self.num_samples = num_samples

    def __iter__(self) -> Generator[TensorDict, None, None]:
        # Case 1: No number of samples is given, return a dataset-like iterator over the replay buffer
        if self.num_samples is None:
            if isinstance(self.rb, TensorDictPrioritizedReplayBuffer):
                raise ValueError("Can't use priorities in dataset mode")
            num_iter = self.num_iter or 1
            for _ in range(num_iter):
                for item in self.rb:
                    yield item

        # Case 2: Number of samples is given, use the sampler to return n samples
        else:
            for _ in range(self.num_samples):
                yield self.rb.sample()



def samples(rb: ReplayBuffer, num_iter: int = None, num_samples: int = None) -> Generator[TensorDict, None, None]:
    # Case 1: No number of samples is given, return a dataset like iterator over the replay buffer
    if num_samples is None:
        assert not isinstance(rb, TensorDictPrioritizedReplayBuffer), "cant use priorities in dataset mode"
        num_iter = num_iter or 1
        for _ in range(num_iter):
            for item in rb:
                yield item
    
    # Case 2: Number of samples is given, use the sampler to return n samples
    else:
        for _ in range(num_samples):
            yield rb.sample()



def enable_dropout(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


class RewardScaler:
    """This class calculates the running mean and variance of a stepwise observed
    quantity, like the RL reward / advantage using the Welford online algorithm.
    The mean and variance are either used to standardize the input (scale='norm') or
    to scale it (scale='scale').

    Args:
        scale: None | 'scale' | 'mean': specifies how to transform the input; defaults to None
    """

    def __init__(self, scale: str = None):
        self.scale = scale
        self.count = 0
        self.mean = 0
        self.M2 = 0

    def __call__(self, scores: torch.Tensor):
        if self.scale is None:
            return scores
        elif isinstance(self.scale, int):
            return scores / self.scale
        # Score scaling
        self.update(scores)
        tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
        std = (self.M2 / (self.count - 1)).float().sqrt()
        score_scaling_factor = std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
        if self.scale == "norm":
            scores = (scores - self.mean.to(**tensor_to_kwargs)) / score_scaling_factor
        elif self.scale == "scale":
            scores /= score_scaling_factor
        else:
            raise ValueError("unknown scaling operation requested: %s" % self.scale)
        return scores

    @torch.no_grad()
    def update(self, batch: torch.Tensor):
        batch = batch.reshape(-1)
        self.count += len(batch)

        # newvalues - oldMean
        delta = batch - self.mean
        self.mean += (delta / self.count).sum()
        # newvalues - newMeant
        delta2 = batch - self.mean
        self.M2 += (delta * delta2).sum()


class WarmupScheduler(_LRScheduler):
    def __init__(
        self,
        after_scheduler: _LRScheduler,
        start_value: float,
        duration: int,
        end_value: Optional[float] = None,
        last_epoch: int = -1
    ):
        optimizer = after_scheduler.optimizer
        if end_value is None:
            # Use the initial optimizer lr as the warmup end value
            end_value = optimizer.param_groups[0]['lr']

        self.after_scheduler = after_scheduler
        self.warmup_start_value = start_value
        self.warmup_end_value = end_value
        self.warmup_duration = duration
        self.finished_warmup = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_duration:
            # Linear warmup calculation
            warmup_factor = (self.last_epoch + 1) / self.warmup_duration
            lr = self.warmup_start_value + warmup_factor * (self.warmup_end_value - self.warmup_start_value)
            return [lr for _ in self.base_lrs]
        else:
            if not self.finished_warmup:
                # Align the after_scheduler’s base_lrs to warmup_end_value
                self.after_scheduler.base_lrs = [self.warmup_end_value for _ in self.base_lrs]
                self.finished_warmup = True
            return self.after_scheduler.get_last_lr()

    def step(self, epoch: Optional[int] = None):
        if self.last_epoch < self.warmup_duration:
            super().step(epoch)
        else:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_duration)
            self.last_epoch += 1




# Function to generate bounded random chunk sizes
def bounded_random_chunk_sizes(total_size, min_chunk, max_chunk):
    sizes = []
    remaining = total_size
    while remaining > 0:
        max_allowed = min(max_chunk, remaining)
        min_allowed = min(min_chunk, max_allowed)

        if remaining <= max_chunk and remaining >= min_chunk:
            sizes.append(remaining)
            break

        size = random.randint(min_allowed, max_allowed)
        sizes.append(size)
        remaining -= size

    return sizes


def randomized_td_chunks_w_padding(td, min_chunk_size, max_chunk_size, dim=-1):
    N = td.shape[dim]
    chunk_sizes = bounded_random_chunk_sizes(N, min_chunk_size, max_chunk_size)
    max_size = max(chunk_sizes)
    
    # Split the TensorDict
    chunks = td.split(chunk_sizes, dim=1)
    
    # Pad each chunk to max_size, and mark padded positions with done=True
    padded_chunks = []
    for chunk in chunks:
        current_size = chunk.shape[1]    
        # Pad to max_size along dim=1
        padded_chunk = pad(chunk, pad_size=(0, 0, 0, max_size-current_size))
    
        # Mark padded "done" entries as True
        padded_chunk["done"][:, current_size:] = True
    
        padded_chunks.append(padded_chunk)
    
    # Now `padded_chunks` is a list of padded TensorDicts of shape (B, max_size)
    return padded_chunks, chunk_sizes



class LazyStorageManager:
    """
    A context manager for handling LazyTensorStorage during experience rollout.

    This context manager creates a LazyTensorStorage with a specified size and device.
    It yields the storage object to be used by a rollout function. Upon exiting
    the context, it extracts the collected experiences as a TensorDict up to
    the point where the rollout function has written, clears the storage,
    and returns the TensorDict.

    Args:
        size (int): The maximum capacity of the LazyTensorStorage.
        device (torch.device or str, optional): The device on which to store the data.
            Defaults to "cpu".
    """
    def __init__(self, size: int, device: str = "cpu"):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Size must be a positive integer.")
        self.size = size
        self.device = device
        self._storage: LazyTensorStorage = None

    def __enter__(self) -> LazyTensorStorage:
        """
        Initializes and returns the LazyTensorStorage.
        """
        self._storage = LazyTensorStorage(max_size=self.size, device=self.device)
        return self._storage

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Extracts data from the storage, clears it, and returns the TensorDict.

        The extraction assumes that the storage's internal cursor (`_cursor` or
        the length of one of its internal tensors if using `add`) correctly
        reflects the amount of data written.
        """
        current_length = self._storage._len

        if current_length == 0:
            raise ValueError("No data written to storage")
        else:
            # Extract the data up to the current length
            # The slicing behavior of LazyTensorStorage creates a TensorDict.
            extracted_td = self._storage[:current_length].clone() # Clone to ensure data is independent

        # Clear the storage for future use (resets internal state)
        self._storage.empty() # `empty()` is the method to clear and reset the storage


        self.extracted_tensordict = extracted_td
        return True # Suppresses the exception if it was a managed one (which is not the case here)


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm



class GradNormAccumulator:
    """
    Accumulates gradients from multiple batches, each potentially representing a
    different difficulty level in a curriculum learning setup.

    This class ensures each accumulated batch has an equal influence on the final
    gradient by normalizing the gradient of each batch to a unit norm before
    summation. The final gradient assigned to the model is the average of these
    normalized gradients.

    Args:
        model (torch.nn.Module): The model whose gradients are to be accumulated.
        min_norm (float): A minimum norm value to prevent division by zero or
                          by very small numbers. Gradients with a norm less than
                          this will be scaled by `1.0 / min_norm`. Defaults to 1.0.
    """
    def __init__(self, model: torch.nn.Module, min_norm: float = 1.0):
        self.model = model
        self.min_norm = min_norm
        # Stores the list of normalized gradient tensors for each parameter.
        # The outer list corresponds to model parameters, the inner list
        # to accumulated gradients from different instance_ids.
        self._grad_buffers: List[List[torch.Tensor]] = [
            [] for _ in self.model.parameters()
        ]
        self._num_accumulations = 0

    def _get_grad_norm(self) -> float:
        """Calculates the total L2 norm of the current gradients in the model."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def _normalize_and_store(self):
        """Normalizes current model grads and stores them in the buffer."""
        norm = self._get_grad_norm()
        # Prevent division by a very small number
        scale = 1.0 / max(norm, self.min_norm)

        for i, p in enumerate(self.model.parameters()):
            if p.grad is not None:
                # Normalize and store a detached clone
                norm_grad = p.grad.data * scale
                self._grad_buffers[i].append(norm_grad.detach().clone())

    def accumulate(self):
        """
        Normalizes the current gradients on the model, stores them, and then
        clears the model's gradients to prepare for the next backward pass.
        """
        self._normalize_and_store()
        self._num_accumulations += 1
        
        # Important: clear gradients after saving for the next batch
        self.model.zero_grad()

    def average_and_assign(self):
        """
        Averages the accumulated gradients and assigns them back to the model's
        .grad attribute. Resets the accumulator for the next cycle.
        """
        if self._num_accumulations == 0:
            return

        for i, p in enumerate(self.model.parameters()):
            # Only process parameters that have gradients
            if not self._grad_buffers[i]:
                continue
            
            # Stack and average the normalized gradients for this parameter
            avg_grad = torch.stack(self._grad_buffers[i], dim=0).mean(dim=0)
            
            if p.grad is None:
                p.grad = avg_grad
            else:
                p.grad.data.copy_(avg_grad)

        # Reset for the next accumulation cycle
        self.reset()

    def reset(self):
        """Clears all accumulated gradients and resets the counter."""
        self._grad_buffers = [[] for _ in self.model.parameters()]
        self._num_accumulations = 0
        
    def __len__(self) -> int:
        """Returns the number of gradients accumulated so far."""
        return self._num_accumulations
    

from dataclasses import dataclass, field
from typing import Callable, Optional, Union, Type, Dict
import math


# ----------------------------
# Scheduler factory functions
# ----------------------------

def exponential_scheduler(start: float, end: float, total_steps: int) -> Callable[[int], Numeric]:
    """Exponential decay from start → end over total_steps."""
    gamma = (end / start) ** (1 / max(total_steps, 1))
    return lambda step: start * (gamma ** step)


def linear_scheduler(start: float, end: float, total_steps: int) -> Callable[[int], Numeric]:
    """Linear interpolation from start → end."""
    return lambda step: start + (end - start) * min(step / total_steps, 1.0)


def cosine_scheduler(start: float, end: float, total_steps: int) -> Callable[[int], Numeric]:
    """Cosine annealing from start → end."""
    return lambda step: end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))


def constant_scheduler(start: float, end: float, total_steps: int) -> Callable[[int], Numeric]:
    """Constant value (always returns start)."""
    return lambda step: start


# Registry
SCHEDULER_REGISTRY: Dict[str, Callable[[float, float, int], Callable[[int], Numeric]]] = {
    "exponential": exponential_scheduler,
    "linear": linear_scheduler,
    "cosine": cosine_scheduler,
    "constant": constant_scheduler,
}


# ----------------------------
# NumericParameter class
# ----------------------------

@dataclass
class NumericParameter:
    start: Numeric
    end: Optional[Numeric] = None
    total_steps: Optional[int] = None
    scheduler_name: Optional[str] = "linear"
    scheduler: Optional[Callable[[int], Numeric]] = None  # internal callable
    dtype: Optional[Union[Type, Callable]] = None
    step_count: int = field(default=0, init=False)

    def __post_init__(self):
        # Build scheduler if given by name
        if self.scheduler_name is not None and self.end is not None and self.total_steps is not None:
            if self.scheduler_name not in SCHEDULER_REGISTRY:
                raise ValueError(f"Unknown scheduler '{self.scheduler_name}'. Available: {list(SCHEDULER_REGISTRY)}")
            
            self.scheduler = SCHEDULER_REGISTRY[self.scheduler_name](
                self.start,
                self.end if self.end is not None else self.start,
                self.total_steps if self.total_steps is not None else 1,
            )

        # Fallback: constant if nothing else provided
        if self.scheduler is None:
            self.scheduler = constant_scheduler(self.start, self.start, 1)

    @property
    def val(self) -> Numeric:
        val = self.scheduler(self.step_count)
        if self.dtype is not None:
            val = self.dtype(val)
        return val

    def update(self):
        """Advance one step."""
        self.step_count += 1
        return self