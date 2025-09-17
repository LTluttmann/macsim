import os
import wandb
import time
import json
import torch
import logging
import functools
import numpy as np
import subprocess
import pandas as pd

from io import StringIO
from functools import lru_cache
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing import Any, Callable, Dict, Type, TypeVar, Union, List

from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig

from .config import ModelParams


# A logger for this file
log = logging.getLogger(__name__)

T = TypeVar('T')


class Registry:
    def __init__(self):
        self._registry: Dict[str, Type[Any]] = {}

    def register(self, name: str = None) -> Callable[[Type[T]], Type[T]]:
        def decorator(cls: Type[T]) -> Type[T]:
            nonlocal name
            if name is None:
                name = cls.__name__
            
            # Use setattr for all classes, including regular classes and dataclasses
            setattr(cls, 'name', name)
            
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str, default: Any = None) -> Type[Any]:
        return self._registry.get(name, default)

    def all(self) -> Dict[str, Type[Any]]:
        return self._registry


def read_json(path:str) -> dict:
    with open(path+".json","r",encoding="utf-8") as f:
        config = json.load(f)
    return config


def write_json(data:dict, path:str):
    with open(path+".json", 'w', encoding='UTF-8') as fp:
        fp.write(json.dumps(data, indent=2, ensure_ascii=False))


def get_free_gpus(mem_limit: int = 300):
    try:
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        gpu_df = pd.read_csv(
            StringIO(gpu_stats.decode("utf-8")),
            names=['memory.used', 'memory.free'],
            skiprows=1
        )
        gpu_df['memory.used'] = gpu_df['memory.used'].map(lambda x: int(x.rstrip(' [MiB]')))
        return gpu_df[gpu_df["memory.used"] < mem_limit].index.to_list()
    
    except:
        return []


def monitor_lr_changes(func: Callable) -> Callable:
    """
    Decorator that monitors learning rate changes before and after the wrapped function.
    Specifically designed for validation_end hooks in PyTorch Lightning.
    
    Args:
        func: The validation_end hook function to be wrapped
        
    Returns:
        Wrapped function that monitors learning rate changes
    """
    @functools.wraps(func)
    def wrapper(self: LightningModule, *args: Any, **kwargs: Any) -> Any:
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            # Get learning rates before hook
            pre_lrs = []
            for i, param_group in enumerate(optimizer.optimizer.param_groups):
                pre_lrs.append((i, param_group['lr']))
            
            # Execute the hook
            result = func(self, *args, **kwargs)
            
            # Get learning rates after hook
            post_lrs = []
            for i, param_group in enumerate(optimizer.optimizer.param_groups):
                post_lrs.append((i, param_group['lr']))
            
            # Check for changes and log them
            for (pre_group, pre_lr), (post_group, post_lr) in zip(pre_lrs, post_lrs):
                if abs(pre_lr - post_lr) > 1e-12:  # Use small epsilon to handle floating point comparison
                    log.info(
                            f"Learning rate changed in group {pre_group} of optimizer {optimizer.__class__.__name__}: "
                            f"{pre_lr:.2e} -> {post_lr:.2e} (Δ = {post_lr - pre_lr:.2e})"
                    )
            
        return result
    
    return wrapper


def determine_devices(devices: Union[List[int], int, str], check_env: bool = True, mem_limit: int = 300, assert_idle: bool = True) -> List[int]:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None and check_env:
        try:
            return [int(os.environ.get("CUDA_VISIBLE_DEVICES"))]
        except:
            raise ValueError(f"Expected a number in CUDA_VISIBLE_DEVICES, got {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    idle_gpus = get_free_gpus(mem_limit)

    if isinstance(devices, int):
        return idle_gpus[:devices]
    
    elif devices == "auto":
        return idle_gpus
    
    else:
        if assert_idle:
            assert all([d in idle_gpus for d in devices]), "Selected busy gpu, terminating..."
        return devices



class RunManager():
    def __init__(self, cfg: DictConfig, hc: HydraConfig):
        self.cfg = cfg
        self.hc = hc
        self.launcher = hc.launcher._target_
        self.is_multirun = hc.mode.name == "MULTIRUN"
        self.locked_device = None
        self.assert_idle = getattr(cfg, "assert_idle", True)

    @rank_zero_only
    def _enter(self):
        if torch.cuda.is_available() and hasattr(self.cfg.train, "devices"):
            if self.launcher == "hydra_plugins.hydra_joblib_launcher.joblib_launcher.JoblibLauncher":
                self.set_and_lock_gpu(devices=self.cfg.train.devices)
                device = [self.locked_device]
            else:
                device = determine_devices(self.cfg.train.devices, assert_idle=self.assert_idle)
            self.cfg.train.devices = device

    @rank_zero_only
    def _exit(self):
        if self.locked_device is not None:
            self.release_gpu_lock()
        if wandb.run is not None:
            wandb.finish(exit_code=0)


    def __enter__(self):
        """Called before a job starts."""
        log.info("Starting Job...")
        self._enter()

    def __exit__(self, exc_type, exc_value, traceback):
        """Called after a job ends."""
        log.info("Finishing Job...")
        self._exit()


    def set_and_lock_gpu(self, devices: Union[List[int], int, str]) -> int:
        """Acquire a GPU lock to ensure exclusivity."""
        while True:
            gpus = determine_devices(devices, check_env=False)
            for gpu_id in gpus:
                lock_file = f"/tmp/gpu_lock_{gpu_id}"
                if not os.path.exists(lock_file):
                    # Acquire lock
                    log.info(f"Locking GPU with ID {gpu_id}")
                    open(lock_file, 'w').close()
                    self.locked_device = gpu_id
                    return
            log.info(f"No GPUs available for Job No. {self.hc.job.num}. Waiting...")
            time.sleep(300)  # Retry if no GPU is available


    def release_gpu_lock(self):
        """Release the GPU lock."""
        lock_file = f"/tmp/gpu_lock_{self.locked_device}"
        if os.path.exists(lock_file):
            log.info(f"Releasing lock for GPU with ID {self.locked_device}")
            os.remove(lock_file)

def hydra_run_wrapper(func):
    """Decorator to wrap a Hydra main function."""
    @functools.wraps(func)
    def wrapper(cfg, *args, **kwargs):
        hc = HydraConfig.get()
        with RunManager(cfg, hc) as manager:
            return func(cfg, *args, **kwargs)

    return wrapper


def get_wandb_logger(cfg: DictConfig, model_params: ModelParams, hc: HydraConfig, training: bool = True):
    policy_params = model_params.policy
    env_params = policy_params.env
    default_tags = [
        model_params.algorithm, 
        policy_params.policy, 
        env_params.env,
        env_params.id,
        "eval_multi" if model_params.eval_multistep else "eval_single",
        "eval_per_agent" if model_params.eval_per_agent else "eval_all",
    ]
    logger_cfg = OmegaConf.to_container(cfg.get("logger"), resolve=True)
    new_tags = logger_cfg.pop("tags", [])
    all_tags = default_tags + new_tags
    train_tag = env_params.id if training else "evaluation"
    return WandbLogger(
        save_dir=hc.runtime.output_dir,
        tags=all_tags,
        name=f"{model_params.algorithm}-{policy_params.policy}-{train_tag}",
        **logger_cfg
    )


def get_lightning_logger(name=__name__, rzo: bool = True) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    if rzo:
        for level in logging_levels:
            setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger



def environment_distribution(env_sizes, current_epoch, epochs_until_uniform, temperature=1.0):
    """
    Compute a probability distribution over environments, shifting from size-based to uniform.

    Args:
        env_sizes (list or np.array): List of environment sizes.
        current_epoch (int): Current epoch number.
        max_epochs (int): Total number of training epochs.
        temperature (float): Controls initial preference for small environments.
        beta (float): Controls transition smoothness.

    Returns:
        np.array: Probability distribution over environments.
    """
    env_sizes = np.array(env_sizes)

    # Step 1: Compute initial probabilities favoring small environments
    initial_probs = np.exp(-env_sizes / temperature)
    initial_probs /= initial_probs.sum()  # Normalize to form a distribution

    # Step 2: Compute uniform distribution
    uniform_probs = np.ones_like(env_sizes) / len(env_sizes)

    # Step 3: Compute scheduling coefficient (alpha)
    if current_epoch < epochs_until_uniform:
        alpha = np.linspace(0, 1, epochs_until_uniform)[current_epoch]
    else:
        alpha = 1

    # Step 4: Interpolate between the two distributions
    final_probs = (1 - alpha) * initial_probs + alpha * uniform_probs

    # Ensure normalization (to account for numerical issues)
    final_probs /= final_probs.sum()

    return final_probs


def get_environment_distribution_for_current_epoch(env_sizes, current_epoch, epochs_until_uniform, **kwargs):
    if current_epoch < epochs_until_uniform:
        all_dists = generate_env_curriculum(env_sizes, epochs_until_uniform, **kwargs)
        return all_dists[current_epoch]
    else:
        return np.ones_like(env_sizes) / len(env_sizes)
    

lru_cache(maxsize=1)
def generate_env_curriculum(
    env_sizes: list[int], 
    steps_until_uniform: int,     
    uniformity_threshold: float = 0.3, # Wie gleichmäßig am Ende (P_best/P_worst - 1)
    dominance_ratio_threshold: float = 200.0, # Wie dominant die kleinste Größe am Start sein soll (P_min/P_second_min)
    curve_factor: float = 2.0 # Form der Temperaturkurve
):
    env_sizes = np.array(env_sizes)
    sizes_scaled = (env_sizes - env_sizes.min()) / (env_sizes.max() - env_sizes.min())

    scores = -sizes_scaled.astype("float")
    S_min_val = sizes_scaled[0]
    S_second_min_val = sizes_scaled[1]

    score_diff_for_Tmin = S_second_min_val - S_min_val

    log_dominance_factor = np.log(dominance_ratio_threshold)
    if log_dominance_factor <= 1e-9: # Sicherheit gegen Division durch Null
        log_dominance_factor = 1e-9
    T_min_auto = score_diff_for_Tmin / log_dominance_factor

    # --- Automatische Bestimmung von T_max ---
    S_max_val = sizes_scaled[-1]
    score_diff_for_Tmax = S_max_val - S_min_val

    log_uniformity_factor = np.log(1 + uniformity_threshold)
    if log_uniformity_factor <= 1e-9:
        log_uniformity_factor = 1e-9

    T_max_auto = score_diff_for_Tmax / log_uniformity_factor

    temps = T_min_auto + (T_max_auto - T_min_auto) * (np.linspace(0, 1, steps_until_uniform) ** curve_factor)

    exp_scaled_scores = np.exp(scores[None] / temps[:, None])

    probs = exp_scaled_scores / exp_scaled_scores.sum(1, keepdims=True)
    return probs
