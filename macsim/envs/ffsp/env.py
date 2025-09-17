import torch

from einops import repeat, reduce
from tensordict.tensordict import TensorDict

from macsim.envs.base import AgentAction
from macsim.envs.env_args import FFSPParams
from macsim.utils.config import EnvParamList
from macsim.envs.base import MultiAgentEnvironment, Environment

from .generator import FFSPGenerator, generate_padded_sequences



class FFSPEnv(Environment):
    """Flexible Flow Shop Problem (FFSP) environment.
    The goal is to schedule a set of jobs on a set of machines such that the makespan is minimized.

    Constraints:
        - each job has to be processed on each machine in a specific order
        - the machine has to be available to process the job
        - the job has to be available to be processed

    Finish Condition:
        - all jobs are scheduled

    Reward:
        - (minus) the makespan of the schedule

    Args:
        generator: FFSPGenerator instance as the data generator
        generator_params: parameters for the generator
    """
    name = "ffsp"

    def __init__(
        self,
        generators: FFSPGenerator = None,
        params: FFSPParams = None,
        **kwargs,
    ):
        if generators is None and params is not None:
            if isinstance(params, EnvParamList):
                generators = [FFSPGenerator(param) for param in params]
                self.params = params[0]
            else:
                generators = FFSPGenerator(params)
                self.params = params

        super().__init__(generators)

        self.num_jobs = None
        self.num_stage = None
        self.num_machine_total = None
    
    @staticmethod
    def _max_num_steps(g: FFSPGenerator):
        return g.num_job * g.num_stage
    
    def _set_instance_params_and_id(self, td):
        self.num_jobs = td["proc_times"].size(1)
        self.num_stage = td["machine_cnt"].size(1)
        self.num_machine_total = td["proc_times"].size(2)
        max_ma_per_stage = int(td['machine_cnt'].max())
        instance_id = f"{self.num_jobs}j_{max_ma_per_stage}m_{self.num_stage}s"
        return instance_id

    def get_action_mask(self, td: TensorDict):

        batch_size = td.batch_size

        mask = torch.full(
            size=(*batch_size, self.num_machine_total, self.num_jobs),
            fill_value=False,
            dtype=torch.bool,
            device=td.device
        )

        # shape: (batch, job)
        job_loc = td["job_location"]
        # shape: (batch, 1, job)
        job_finished = (job_loc >= self.num_stage).unsqueeze(-2).expand_as(mask)

        stage_table_expanded = td["stage_table"][:, :, None].expand_as(mask)
        job_not_in_machines_stage = job_loc[:, None] != stage_table_expanded

        mask.add_(job_finished)
        mask.add_(job_not_in_machines_stage)
        # NOTE only allow waiting as initial action in done envs, otherwise we want at least one machine to select a job
        no_op_mask = td["done"].view(*batch_size, 1, 1).expand(*batch_size, self.num_machine_total, 1).contiguous()
        # add mask for wait, which is allowed if machine cannot process any job
        mask = torch.cat((~mask, no_op_mask), dim=-1)
        # NOTE: 1 means feasible action, 0 means infeasible action
        return mask

    def _step(self, td: TensorDict) -> TensorDict:
        batch_size = td.batch_size
        batch_idx = torch.arange(*batch_size, dtype=torch.long, device=td.device)
        action: AgentAction = td["action"]

        job_idx = action["node"]
        machine_idx = action["agent"]

        skip = job_idx == self.num_jobs

        b_idx = batch_idx[~skip]

        job_idx = job_idx[~skip]
        machine_idx = machine_idx[~skip]

        t_job = td["t_job_ready"][b_idx, job_idx]
        t_ma = td["t_ma_idle"][b_idx, machine_idx]
        t = torch.maximum(t_job, t_ma)

        # shape: (batch)
        job_length = td["proc_times"][b_idx, job_idx, machine_idx]

        # shape: (batch, machine)
        td["t_ma_idle"][b_idx, machine_idx] = t + job_length
        td["t_job_ready"][b_idx, job_idx] = t + job_length
        # shape: (batch, job)
        td["job_location"][b_idx, job_idx] += 1
        td["job_done"][b_idx, job_idx] = td["job_location"][b_idx, job_idx] >= self.num_stage
        # shape: (batch)
        td["done"] = td["job_done"].all(dim=-1)
        return td
    
    def _reset(self, td: TensorDict) -> TensorDict:
        batch_size = td.batch_size
        device = td.device

        job_location = torch.zeros(
            size=(*batch_size, self.num_jobs),
            dtype=torch.long,
            device=device,
        )
        # time information
        t_job_ready = torch.zeros(
            size=(*batch_size, self.num_jobs), 
            dtype=torch.long,
            device=device
        )
        t_ma_idle = torch.zeros(
            size=(*batch_size, self.num_machine_total), 
            dtype=torch.long,
            device=device)
        
        # Finish status information
        reward = torch.full(
            size=(*batch_size,),
            dtype=torch.get_default_dtype(),
            device=device,
            fill_value=float("-inf"),
        )
        job_done = torch.full(
            size=(*batch_size,self.num_jobs),
            dtype=torch.bool,
            device=device,
            fill_value=False,
        )
        done = torch.full(
            size=(*batch_size,),
            dtype=torch.bool,
            device=device,
            fill_value=False,
        )

        scale_factor = reduce(
            td["proc_times"], 
            "b ... -> b", 
            reduction="max"
        )

        td = td.update(
            {
                # Index information
                "t_job_ready": t_job_ready,
                "t_ma_idle": t_ma_idle,
                # Scheduling status information
                "job_location": job_location,
                "scale_factor": scale_factor,
                # Finish status information
                "reward": reward,
                "job_done": job_done,
                "done": done
            }
        )
        return td
    
    def get_reward(self, td: TensorDict) -> torch.FloatTensor:
        assert td["done"].all(), "Cannot get reward before instance is finished"
        makespan = td["t_job_ready"].max(1).values.to(torch.get_default_dtype())
        reward = -makespan
        return reward

    @staticmethod
    def load_instances(path):
        """Dataset loading from file"""
        data = torch.load(path)
        # (bs, n_jobs, n_ma)
        proc_times = torch.cat(data, -1)
        batch_size = proc_times.size(0)
        # (bs, n_stages)
        ma_cnt = repeat(torch.tensor([x.size(-1) for x in data]), "s -> b s", b=batch_size)
        # (bs, n_ma)
        stage_table = generate_padded_sequences(ma_cnt, pad_value=-1)
        return TensorDict(
            {
                "proc_times": proc_times,
                "machine_cnt": ma_cnt,
                "stage_table": stage_table,
            },
            batch_size=(batch_size,),
        )
    

class MatNetFFSPEnv(FFSPEnv):
    """
    This is the environment logic in the MatNet paper, where the environment iterates over all machines in the order of 
    their indices. In each iteration, only the 'current machine' can choose a job.
    """

    name = "matnet_ffsp"

    def __init__(
        self,
        generators: FFSPGenerator = None,
        params: FFSPParams = None,
        **kwargs,
    ):
        super().__init__(generators, params)

    @staticmethod
    def _max_num_steps(g: FFSPGenerator):
        return g.num_job * g.max_machine_total


    def get_action_mask(self, td):
        bs = td.size(0)
        mask = super().get_action_mask(td)
        # update wait action mask
        # (bs, n_ma)
        job_in_previous_stage = torch.any(td["job_location"][:, None] < td["stage_table"][..., None], dim=-1)
        no_available_jobs = torch.logical_not(mask[..., :-1].any(-1))
        # (bs, n_ma)
        allow_wait = torch.logical_or(job_in_previous_stage, no_available_jobs)
        mask[..., -1] = allow_wait
        # update machine masks -> only current machine can act
        ma_idx = (
            torch.arange(self.num_machine_total)
            .to(td["current_ma"])
            .view(1, self.num_machine_total, 1)
            .expand_as(mask)
            .contiguous()
        )
        ma_mask = ma_idx != td["current_ma"].view(bs, 1, 1)
        # (bs, n_jobs)
        mask = mask.masked_fill(ma_mask, False)

        return mask

    def _reset(self, td):
        td = super()._reset(td)
        bs = td.size(0)
        # we start with the first machine in index order and increment in each step
        td["current_ma"] = torch.zeros((bs,), device=td.device, dtype=torch.long)
        return td
    
    def _step(self, td):
        td = super()._step(td)
        td["current_ma"] = torch.where(
            td["current_ma"] == self.num_machine_total - 1,
            torch.zeros_like(td["current_ma"]),
            td["current_ma"] + 1    
        )
        return td
        


class MultiAgentFFSPEnv(MultiAgentEnvironment, FFSPEnv):

    name = "ma_ffsp"
    wait_op_idx = -1

    def __init__(
        self,
        generators: FFSPGenerator = None,
        params: FFSPParams = None,
        **kwargs,
    ):
        super().__init__(generators, params)

    def _step(self, td: TensorDict) -> TensorDict:
        batch_size = td.batch_size
        batch_idx = torch.arange(*batch_size, dtype=torch.long, device=td.device)
        actions = td["action"].split(1, dim=-1)
        for action in actions:
            action: AgentAction = action.squeeze(1)

            job_idx = action["node"]
            machine_idx = action["agent"]

            skip = job_idx == self.num_jobs
            if skip.all():
                continue

            b_idx = batch_idx[~skip]

            job_idx = job_idx[~skip]
            machine_idx = machine_idx[~skip]

            t_job = td["t_job_ready"][b_idx, job_idx]
            t_ma = td["t_ma_idle"][b_idx, machine_idx]
            t = torch.maximum(t_job, t_ma)

            # shape: (batch)
            job_length = td["proc_times"][b_idx, job_idx, machine_idx]

            # shape: (batch, machine)
            td["t_ma_idle"][b_idx, machine_idx] = t + job_length
            td["t_job_ready"][b_idx, job_idx] = t + job_length
            # shape: (batch, job)
            td["job_location"][b_idx, job_idx] += 1
            td["job_done"][b_idx, job_idx] = td["job_location"][b_idx, job_idx] >= self.num_stage
            # shape: (batch)
            td["done"] = td["job_done"].all(dim=-1)

        return td

    def update_mask(self, mask, action: AgentAction, td, busy_agents):
        """Update mask after an agent acting during multi-agent rollout. True means feasible, False means infeasible"""
        bs, n_ma, n_jobs = mask.shape
        job_selected = action["node"]
        # mask job that has been selected in the current step so it cannot be selected by other agents
        mask = mask.scatter(-1, job_selected.view(bs, 1, 1).expand(bs, n_ma, 1), False)
        if self.params.use_skip_token:
            # always allow machines that are still idle to wait (for jobs to become available for example)
            mask[..., -1] = True
        else:
            # allow an idle machine to wait only if it cannot choose any job 
            mask[..., -1] = torch.logical_not(mask[..., :-1].any(-1))
        # lastly, mask all actions for the selected agent
        mask[busy_agents] = False
        return mask
        