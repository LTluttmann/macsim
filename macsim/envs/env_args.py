from dataclasses import dataclass
from rl4co.utils import pylogger

from macsim.utils.config import EnvParams


log = pylogger.get_pylogger(__name__)



@dataclass(kw_only=True)
class FJSPParams(EnvParams):
    env: str = "fjsp"
    num_jobs: int = None
    num_machines: int = None
    min_processing_time: int = 1
    max_processing_time: int = 99
    scaling_factor: int = None
    # specity whether to use the difference in lower bounds as stepwise reward proxy
    stepwise_reward: bool = False  
    min_ops_per_job: int = None
    max_ops_per_job: int = None
    min_eligible_ma_per_op: int = 1
    max_eligible_ma_per_op: int = None
    same_mean_per_op: bool = True
    use_job_emb: bool = False
    max_machine_id: int = 15
    max_job_id: int = 40

    def __post_init__(self):
        self.scaling_factor = self.scaling_factor or self.max_processing_time
        self.max_eligible_ma_per_op = self.num_machines


@dataclass(kw_only=True)
class FFSPParams(EnvParams):
    env: str = "ffsp"
    num_jobs: int = 10
    num_stage: int = 3
    min_ma_per_stage: int = 4
    max_ma_per_stage: int = 4
    min_processing_time: int = 2
    max_processing_time: int = 10
    ma_cnt_prob: list = None
    scaling_factor: int = None
    ma_pad_value: int = -1

    def __post_init__(self):
        self.scaling_factor = self.scaling_factor or self.max_processing_time



@dataclass(kw_only=True)
class MatNetFFSPParams(FFSPParams):
    env: str = "matnet_ffsp"
    
    def __post_init__(self):
        self.scaling_factor = self.scaling_factor or self.max_processing_time



@dataclass(kw_only=True)
class HCVRPParams(EnvParams):
    env: str = "hcvrp"
    num_loc: int = 60
    num_agents: int = 3
    min_loc: float = 0.0
    max_loc: float = 1.0
    min_demand: int = 1
    max_demand: int = 9
    min_capacity: float = 20
    max_capacity: float = 40
    min_speed: float = 0.5
    max_speed: float = 1.0