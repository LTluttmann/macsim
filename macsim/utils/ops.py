import logging
from typing import Union

import torch
from torch import Tensor
from einops import rearrange
import torch.distributed as dist
from tensordict import TensorDict

from macsim.utils.config import DecodingConfig


log = logging.getLogger(__name__)
_float = torch.get_default_dtype()


def maybe_convert(x: torch.Tensor):
    if not x.is_floating_point():
        x = x.to(dtype=torch.get_default_dtype())
    return x

def normalize_features(x: torch.Tensor, dims_to_reduce=(0,), eps=1e-6):
    """
    Normalize features along specified dimensions
    x: input tensor with arbitrary leading dimensions
    dims_to_reduce: dimensions to compute mean and std over
    """
    mean = x.mean(dim=dims_to_reduce, keepdim=True)
    var = x.var(dim=dims_to_reduce, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)


def min_max_scale(tensor: torch.Tensor, dim=None, eps=1e-6):
    tensor = tensor.to(torch.get_default_dtype()).clone()

    if dim is not None:
        min_vals = tensor.amin(dim=dim, keepdim=True)
        max_vals = tensor.amax(dim=dim, keepdim=True)
        scaled = (tensor - min_vals) / (max_vals - min_vals + eps)
    else:
        min_val = tensor.min()
        max_val = tensor.max()
        scaled = (tensor - min_val) / (max_val - min_val + eps)

    return scaled.clamp(0,1)

def scale_batch(x: torch.Tensor, scale: torch.Tensor, epsilon = 1e-6) -> torch.Tensor:
    """
    Scales each instance in the batch using a corresponding scaling factor.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, d1, d2, ...)
        scale (torch.Tensor): A tensor of shape (batch_size,) containing the scaling factors.

    Returns:
        torch.Tensor: The scaled tensor with the same shape as x.
    """
    if epsilon is not None:
        scale = scale + epsilon
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=x.device, dtype=_float)
    return x / scale.view(-1, *([1] * (x.dim() - 1)))

def feature_normalize(data, dim=None):
    keepdim = dim is not None
    mean_feat = torch.mean(data, dim=dim, keepdim=keepdim)
    std_feat = torch.std(data, dim=dim, keepdim=keepdim)
    return (data - mean_feat) / (std_feat + 1e-5)


def _batchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    """Same as repeat on dim=0 for Tensordicts as well"""
    s = x.shape
    return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])


def batchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    """Same as `einops.repeat(x, 'b ... -> (b r) ...', r=repeats)` but ~1.5x faster and supports TensorDicts.
    Repeats batchify operation `n` times as specified by each shape element.
    If shape is a tuple, iterates over each element and repeats that many times to match the tuple shape.

    Example:
    >>> x.shape: [a, b, c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a*b*c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x
    return x


def augment_or_batchify(td: TensorDict, env, num_samples: DecodingConfig):
    bs = td.size(0)
    num_augment = num_samples.num_augment
    if num_augment > 1:
        assert hasattr(env, "augment_states")
        td = env.augment_states(td, num_augment=num_augment)
        assert td.size(0) // bs == num_augment, f"Augmentation failed. Expected {num_augment} augmentations, got {td.size(0) // bs}"

    num_strategies = num_samples.num_strategies
    if num_strategies > 1: 
        bs = td.size(0)
        strategy_id = torch.arange(num_strategies, device=td.device).repeat_interleave(bs)
        td = td.repeat(num_strategies)
        td["strategy_id"] = strategy_id
        
    num_starts = num_samples.num_starts
    if num_starts > 1:
        # Expand td to batch_size * num_starts
        td = batchify(td, num_starts)
    return td, num_starts


def _unbatchify_single(
    x: Union[Tensor, TensorDict], repeats: int
) -> Union[Tensor, TensorDict]:
    """Undoes batchify operation for Tensordicts as well"""
    s = x.shape
    return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))


def unbatchify(
    x: Union[Tensor, TensorDict], shape: Union[tuple, int]
) -> Union[Tensor, TensorDict]:
    """Same as `einops.rearrange(x, '(r b) ... -> b r ...', r=repeats)` but ~2x faster and supports TensorDicts
    Repeats unbatchify operation `n` times as specified by each shape element
    If shape is a tuple, iterates over each element and unbatchifies that many times to match the tuple shape.

    Example:
    >>> x.shape: [a*b*c, ...]
    >>> shape: [a, b, c]
    >>> out.shape: [a, b, c, ...]
    """
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(
        shape
    ):  # we need to reverse the shape to unbatchify in the right order
        x = _unbatchify_single(x, s) if s > 0 else x
    return x

def gather_by_index(src: Union[torch.Tensor, TensorDict], idx: torch.Tensor, dim=1, squeeze=True):
    """Gather elements from src by index idx along specified dim

    Example:
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3)] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    idx = idx.to(src.device)
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)


def sample_n_random_actions(td: TensorDict, n: int):
    """Helper function to sample n random actions from available actions. If
    number of valid actions is less then n, we sample with replacement from the
    valid actions
    """
    action_mask = td["action_mask"]
    # check whether to use replacement or not
    n_valid_actions = torch.sum(action_mask[:, 1:], 1).min()
    if n_valid_actions < n:
        replace = True
    else:
        replace = False
    ps = torch.rand((action_mask.shape))
    ps[~action_mask] = -torch.inf
    ps = torch.softmax(ps, dim=1)
    selected = torch.multinomial(ps, n, replacement=replace).squeeze(1)
    selected = rearrange(selected, "b n -> (n b)")
    return selected.to(td.device)

def all_gather_numeric(num: Union[int, float], ws: int, device):
    local_num = torch.tensor(num, device=device)
    all_num = [torch.zeros_like(local_num) for _ in range(ws)]
    dist.all_gather(all_num, local_num)
    return all_num


def all_gather_w_padding(q: torch.Tensor, ws: int):
    """
    Gathers tensor arrays of different lengths across multiple gpus
    
    Parameters
    ----------
        q : tensor array
        ws : world size
        
    Returns
    -------
        all_q : list of gathered tensor arrays from all the gpus

    """
    local_size = torch.tensor(q.size(), device=q.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)

    size_diff = max_size.item() - local_size.item()
    if size_diff:
        padding = torch.zeros(size_diff, device=q.device, dtype=q.dtype)
        q = torch.cat((q, padding))

    all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
    dist.all_gather(all_qs_padded, q)
    all_qs = []
    for q, size in zip(all_qs_padded, all_sizes):
        all_qs.append(q[:size])
    return all_qs