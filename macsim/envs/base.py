import abc
import torch
import numpy as np

from torch import Tensor
from einops import reduce
from dataclasses import dataclass
from tensordict.tensordict import TensorDict
from typing import Type, Dict, List, Optional, Union, TypedDict, Tuple

from macsim.utils.config import EnvParams

env_registry: Dict[str, Type['Environment']] = {}


class AgentAction(TypedDict):
    idx: torch.Tensor
    node: torch.Tensor
    agent: torch.Tensor


@dataclass
class ProblemSize:
    num_jobs: int
    num_ops: int
    num_mas: int


class Generator(metaclass=abc.ABCMeta):
    """Base data generator class, to be called with `env.generator(batch_size)`"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = None

    def __call__(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        return self._generate(batch_size)

    @abc.abstractmethod
    def _generate(self, batch_size, **kwargs) -> TensorDict:
        pass

    @property
    @abc.abstractmethod
    def id(self):
        pass

    @property
    @abc.abstractmethod
    def size(self):
        pass


class Environment(metaclass=abc.ABCMeta):
    name = ...

    def __init__(self, generators: Optional[List[Generator]]):
        if generators is not None:
            generators = [generators] if not isinstance(generators, list) else generators
            self.generators: List[Generator] = sorted(generators, key=lambda x: x.size)
        else:
            self.generators = None
        # Protected attribute for stepwise reward
        self._stepwise_reward = False
        # Class attribute to indicate if stepwise rewards are supported
        if not hasattr(self.__class__, '_supports_stepwise_reward'):
            self.__class__._supports_stepwise_reward = False
        # whether env should be cheked for validity or not
        self.check_mask = False

    @property
    def max_num_steps(self):
        return max([self._max_num_steps(g) for g in self.generators])

    @staticmethod
    @abc.abstractmethod
    def _max_num_steps(g: Generator):
        pass

    @abc.abstractmethod
    def get_action_mask(self, td):
        """Defines the action mask of the next step. True means feasible, False means infeasible action"""
        ...

    @abc.abstractmethod
    def _step(self, td) -> TensorDict:
        """Defines how the env transitions to the next state s_{t+1} given (s_t,a_t)"""
        ...

    def _reset(self, td: TensorDict) -> TensorDict:
        """Initializes the state s_0 given the problem instance x. In the simplest case, s_0 = x"""
        return td
    
    def generate(self, batch_size, generator_idx = None):
        assert self.generators is not None, "Need to pass a generator to Environment class to generate data"
        if generator_idx is None:
            generator = np.random.choice(self.generators)
        else:
            if isinstance(generator_idx, str):
                generator_idx = [g.id for g in self.generators].index(generator_idx)
            generator = self.generators[generator_idx]
        td = generator(batch_size=batch_size)
        return td

    def reset(self, td: Optional[TensorDict] = None, batch_size=None) -> Tuple[TensorDict, str]:
        """Reset function to call at the beginning of each episode"""
        assert not (td is None and batch_size is None), "Must pass either td or batch_size to reset"
        if batch_size is None:
            batch_size = td.batch_size
        if td is None:
            td = self.generate(batch_size)
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        _id = self._set_instance_params_and_id(td)
        reset_td = self._reset(td)
        reset_td.set("action_mask", self.get_action_mask(reset_td))

        if self.check_mask:
            assert reduce(reset_td["action_mask"], "bs ... -> bs", "any").all(), (
                "In some instances, no action can be taken in the current state."
            )

        return reset_td, _id

    def step(self, td: TensorDict) -> TensorDict:
        # cloning required to avoid inplace operation which avoids gradient backtracking
        td = td.clone()
        next_state = self._step(td)
        # td.set("next", next_state)
        next_state.set("action_mask", self.get_action_mask(next_state))

        if self.check_mask:
            assert reduce(next_state["action_mask"], "bs ... -> bs", "any").all(), (
                "In some instances, no action can be taken in the current state."
            )

        return next_state
    


    @property
    def stepwise_reward(self) -> bool:
        """Whether the environment uses stepwise rewards."""
        return self._stepwise_reward

    @stepwise_reward.setter
    def stepwise_reward(self, value: bool) -> None:
        """Set stepwise reward mode if supported by the environment.
        
        Args:
            value: Boolean indicating whether to use stepwise rewards
            
        Raises:
            ValueError: If stepwise rewards are not supported by this environment
        """
        if not self.__class__._supports_stepwise_reward:
            raise ValueError(
                f"Environment '{self.__class__.__name__}' does not support stepwise rewards"
            )
        self._stepwise_reward = value


    @abc.abstractmethod
    def _set_instance_params_and_id(self, td):
        pass

    def __init_subclass__(cls, *args, **kw):
        super().__init_subclass__(*args, **kw)
        env_registry[cls.name] = cls

    @classmethod
    def initialize(cls, params: EnvParams) -> Type["Environment"]:
        env = params.env
        return env_registry[env](params=params)

    @staticmethod
    @abc.abstractmethod
    def load_instances(path) -> Union[TensorDict, List[TensorDict]]:
        pass

    @classmethod
    def supports_stepwise_reward(cls) -> bool:
        """Check if this environment supports stepwise rewards."""
        return getattr(cls, '_supports_stepwise_reward', False)
    
    @abc.abstractmethod
    def get_reward(self, td: TensorDict, **kwargs) -> torch.Tensor:
        """Function to compute the reward. Can be called by the agent to compute the reward of the current state
        This is faster than calling step() and getting the reward from the returned TensorDict at each time for CO tasks
        """
        pass

    def _get_step_reward(self, td):
        raise NotImplementedError

    def get_step_reward(self, td):
        if self.stepwise_reward:
            return self._get_step_reward(td)
        else:
            raise ValueError("Called get_step_reward on environment which does not support stepwise reward")


class MultiAgentEnvironment(Environment, metaclass=abc.ABCMeta):

    name = ...
    wait_op_idx = ...
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def update_mask(self, mask: Tensor, action: AgentAction, td: TensorDict, busy_agents: Tensor) -> torch.Tensor:
        pass
