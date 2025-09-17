import torch
import warnings
from copy import copy
from rl4co.utils import pylogger
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
from typing import Literal, Dict, Type, List, Tuple, Union, Optional, Any


MAX_BATCH_SIZE = 32 * 2048  # (https://github.com/facebookresearch/xformers/issues/845)

log = pylogger.get_pylogger(__name__)


model_config_registry: Dict[str, Type['ModelParams']] = {}
policy_config_registry: Dict[str, Type['PolicyParams']] = {}
env_config_registry: Dict[str, Type['PolicyParams']] = {}


def config_to_dict(config_struct: Union[OmegaConf, Dict]):
    try:
        config_struct = OmegaConf.to_container(config_struct, resolve=True)
    except ValueError:
        config_struct = copy(config_struct)
    return config_struct


@dataclass(frozen=True)
class DecodingConfig:
    decode_type: Literal["greedy", "sampling"]
    tanh_clipping: float
    top_p: float
    temperature: float
    num_starts: int = None
    num_augment: int = None
    num_strategies: int = None
    select_best: bool = False
    hybrid_decoding: bool = False
    num_decoding_samples: int = None

    def __post_init__(self):
        # Normalize all fields
        object.__setattr__(self, "num_starts", self._normalize(self.num_starts))
        object.__setattr__(self, "num_augment", self._normalize(self.num_augment))
        object.__setattr__(self, "num_strategies", self._normalize(self.num_strategies))

        if self.num_decoding_samples is not None:
            if self.num_starts > 1:
                assert self.num_samples == self.num_decoding_samples, "Specified both, num_starts and num_decoding_samples. Use only one of the two"
            else:
                num_starts = self.num_decoding_samples // (self.num_strategies * self.num_augment)
                assert num_starts >= 1, "Specified num_decoding_samples smaller than the product of augmentations and strategies. Check config."
                object.__setattr__(self, "num_starts", num_starts)
                if self.num_samples < self.num_decoding_samples:
                    warnings.warn(f"Decoding samples ({self.num_samples}) smaller than specified ({self.num_decoding_samples}). Check config.")
                    

    @staticmethod
    def _normalize(value) -> int:
        # If None → default to 1
        if value is None:
            return 1
        # If not int → try to cast
        if not isinstance(value, int):
            try:
                value = int(value)
            except Exception:
                return 1  # fallback if conversion fails
        # Enforce minimum of 1
        return max(1, value)

    @property
    def num_samples(self) -> int:
        return self.num_starts * self.num_augment * self.num_strategies


@dataclass(kw_only=True)
class EnvParams:
    env: str
    id: str = None
    sizes: List[Tuple[int, int]] = None
    file_path: str = None
    multiagent: bool = field(init=False)
    use_skip_token: bool = True
    num_augment: int = None

    def __init_subclass__(cls, *args, **kw):
        super().__init_subclass__(*args, **kw)
        env_config_registry[cls.env] = cls

    @classmethod
    def initialize(cls, env: str = None, **kwargs):
        try:
            env = env or cls.env
            Config = env_config_registry[env]
            return Config(**kwargs)
        except KeyError:
            raise ValueError(f"No Config found for environment {env}. Configs available for {','.join(list(env_config_registry.keys()))}")
        

@dataclass
class EnvParamList:

    envs: list[EnvParams]
    id: str = "multi_instance"
        
    def append(self, item):
        self.envs.append(item)

    def __getitem__(self, index):
        # Allows element access using indexing
        return self.envs[index]

    def __getattr__(self, attr):
        envs = object.__getattribute__(self, 'envs')
        # Check if the first env has this attribute
        if hasattr(envs[0], attr):
            return getattr(envs[0], attr)
        else:
            raise AttributeError(f"'Environment' has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        if attr == 'envs':
            # Handle setting internal attribute
            super().__setattr__(attr, value)
        else:
            for env in self.envs:
                setattr(env, attr, value)

    @classmethod
    def initialize(cls, env_params: OmegaConf):
        env_params = OmegaConf.to_container(env_params, resolve=True)
        cfg_list = env_params.pop("list")
        cfg_list = {k: {**v, **env_params} for k,v in cfg_list.items()}
        param_list = []
        for params in cfg_list.values():
            params = EnvParams.initialize(**params)
            param_list.append(params)
        return cls(param_list)
    

    def __len__(self):
        # Returns the length of the list
        return len(self.envs)


@dataclass(kw_only=True)
class PolicyParams:
    policy: str
    env: Type["EnvParams"]
    encoder: str = None
    decoder: str = None
    # hyperparams
    embed_dim: int = 256
    bias: bool = True
    num_encoder_layers: int = 4
    dropout: float = 0.0
    normalization: Literal["batch", "instance", "layer", "rms", "none"] = "layer"
    is_multiagent_policy: bool = None
    max_steps: int = None
    use_cross_mha: bool = None
    # to be specified by the learning algorithm
    eval_multistep: bool = field(init=False)
    eval_per_agent: bool = field(init=False)
    use_masked_softmax_for_eval: bool = field(init=False)
    use_critic: bool = field(init=False)
    critic_pooling: Literal["mean", "max", "attn"] = "mean"
    critic_use_sa: bool = True
    stepwise_encoding: bool = field(init=False)

    def __init_subclass__(cls, *args, **kw):
        super().__init_subclass__(*args, **kw)
        policy_config_registry[cls.policy] = cls

    @classmethod
    def initialize(cls, policy: str = None, **kwargs):
        try:
            Config = policy_config_registry[policy]
            return Config(**kwargs)
        except KeyError:
            log.info("Policy of type {policy} has no Config. Use default config instead")
            return PolicyParams(policy=policy, **kwargs)

    def __post_init__(self):
        if self.is_multiagent_policy:
            self.env.env = "ma_" + self.env.env
            self.env.multiagent = True
        else:
            self.env.multiagent = False
    
        if self.use_cross_mha is False:
            self.encoder = "am"

@dataclass(kw_only=True)
class ModelParams:
    
    policy: Type["PolicyParams"]

    # model architecture
    algorithm: str = None

    stepwise_encoding: bool = False
    warmup_params: "ModelParams" = None
    eval_multistep: bool = False
    eval_per_agent: bool = True
    use_masked_softmax_for_eval: bool = False
    use_critic: bool = False
    log_grad_norm: bool = False
    
    def __post_init__(self):
        self.policy.stepwise_encoding = self.stepwise_encoding
        self.policy.use_critic = self.use_critic
        self.eval_multistep = self.eval_multistep and self.policy.is_multiagent_policy
        self.eval_per_agent = self.eval_per_agent and self.eval_multistep
        self.policy.eval_multistep = self.eval_multistep
        self.policy.eval_per_agent = self.eval_per_agent
        self.policy.use_masked_softmax_for_eval = self.use_masked_softmax_for_eval and self.eval_multistep

    def __init_subclass__(cls, *args, **kw):
        super().__init_subclass__(*args, **kw)
        model_config_registry[cls.algorithm] = cls
    
    @classmethod
    def initialize(
        cls, 
        policy_params: PolicyParams, 
        algorithm: str = None, 
        **kwargs
    ):
        assert algorithm is not None, (f"specify the algorithm to use")
        Config = model_config_registry[algorithm]
        if kwargs.get("warmup_params", None) is not None:
            warmup_params = kwargs.pop("warmup_params")
            warmup_cfg = cls.initialize(policy_params=policy_params, **warmup_params) 
            return Config(policy=policy_params, warmup_params=warmup_cfg, **kwargs)
        else:
            return Config(policy=policy_params, **kwargs)



@dataclass(kw_only=True)
class ModelWithReplayBufferParams(ModelParams):
    inner_epochs: int = 1
    num_batches: int = None
    mini_batch_size: int = None
    rollout_batch_size: int = None
    buffer_storage_device: Literal["gpu", "cpu"] = "auto"
    buffer_size: int = 100_000
    # replay buffers allow us to gather experience in evaluation mode
    # Thus, memory leakage through growing gradient information is avoided
    stepwise_encoding: bool = True 
    buffer_kwargs: dict = field(default_factory= lambda: {
        "priority_alpha": 0.5,
        "priority_beta": 0.8
    })
    priority_key: str = None

    # ref model decoding
    ref_model_decode_type: Literal["greedy", "sampling"] = "sampling"
    ref_model_top_p: float = 1
    ref_model_tanh_clipping: float = 10.0
    ref_model_temp: float = 1
    ref_model_num_starts: int = None
    ref_model_num_augment: int = None
    ref_model_num_strategies: int = None
    ref_model_select_best: bool = True
    ref_model_hybrid_decoding: bool = False
    ref_model_num_decoding_samples: int = None

    def __post_init__(self):
        super().__post_init__()
    
    
    @property
    def ref_model_decoding(self):
        return DecodingConfig(
            decode_type=self.ref_model_decode_type,
            top_p=self.ref_model_top_p,
            tanh_clipping=self.ref_model_tanh_clipping,
            temperature=self.ref_model_temp,
            num_starts=self.ref_model_num_starts,
            num_augment=self.ref_model_num_augment,
            num_strategies=self.ref_model_num_strategies,
            select_best=self.ref_model_select_best,
            hybrid_decoding=self.ref_model_hybrid_decoding,
            num_decoding_samples=self.ref_model_num_decoding_samples
        )

    
@dataclass(kw_only=True)
class PhaseParams:
    #data
    batch_size: int
    dataset_size: int
    data_dir: str = None
    num_file_instances: int = None
    file_loader_kwargs: dict = field(default_factory=dict)

    # decoding
    decode_type: Literal["greedy", "sampling"] = "sampling"
    top_p: float = 1
    tanh_clipping: float = 10.0
    temperature: float = 1
    num_starts: int = None
    num_augment: int = None
    num_strategies: int = None
    select_best: bool = True
    hybrid_decoding: bool = False
    num_decoding_samples: int = None

    @property
    def decoding(self) -> DecodingConfig:
        return DecodingConfig(
            decode_type=self.decode_type,
            top_p=self.top_p,
            tanh_clipping=self.tanh_clipping,
            temperature=self.temperature,
            num_starts=self.num_starts,
            num_augment=self.num_augment,
            num_strategies=self.num_strategies,
            select_best=self.select_best,
            hybrid_decoding=self.hybrid_decoding,
            num_decoding_samples=self.num_decoding_samples
        )


@dataclass(kw_only=True)
class TrainingParams(PhaseParams):
    # training
    checkpoint: Optional[str] = None
    train: bool = True
    debug: bool = False
    profiler_filename: str = "profiler_log.txt"


    optimizer_kwargs: Dict[str, Any] = field(default_factory= lambda: {
        "policy_lr": 2e-4,
    })
    lr_scheduler: Optional[Union[OmegaConf, DictConfig]] = None
    lr_scheduler_interval: int = 1
    lr_scheduler_monitor: str = "val/reward/avg"
    lr_reduce_on_plateau_patience: int = 5
    lr_warmup_epochs: int = 0

    max_grad_norm: float = 1.
    epochs: int = 10
    accumulate_grad_batches: int = 1
    norm_curriculum_grad: bool = False

    precision: str = "32-true"  # 16-mixed"
    distribution_strategy: str = "auto"
    accelerator: str = "auto"
    devices: Union[str, List[int]] = "auto"
    reload_every_n: int = 1
    
    seed: int = 1234567
    data_dir: str = None
    monitor_instance: str = None  # instance used for monitoring

    def __post_init__(self):
        if not torch.cuda.is_available():
            # fallback to cpu if cuda not available (MPS not supported yet)
            self.accelerator = "cpu"
            self.devices = "auto"

@dataclass(kw_only=True)
class ValidationParams(PhaseParams):
    ...


@dataclass(kw_only=True)
class TestParams(PhaseParams):
    devices: Union[str, List[int]] = "auto"
    checkpoint: str = None
    seed: int = 1234567

    as_time_budget: int = None
    as_lr: float = 1e-9
    as_bs: int = 32
    as_inner_epochs: int = 3

    @property
    def active_search_params(self) -> "ActiveSearchParams":
        if self.as_time_budget is None:
            return None
        return ActiveSearchParams(
            self.as_time_budget,
            self.as_lr,
            self.as_bs,
            self.as_inner_epochs
        )


@dataclass
class ActiveSearchParams:
    time_budget: int
    lr: float
    bs: int
    inner_epochs: int