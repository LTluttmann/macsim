import torch
from einops import rearrange, reduce
from tensordict.tensordict import TensorDict

from macsim.envs.base import AgentAction
from macsim.envs.env_args import FJSPParams
from macsim.utils.ops import gather_by_index
from macsim.utils.config import EnvParamList
from macsim.envs.base import Environment, MultiAgentEnvironment

from .generator import FJSPGenerator
from .parser import read, list_files, get_max_ops_from_files


class FJSPEnv(Environment):

    name = "fjsp"
    _supports_stepwise_reward = True

    def __init__(
        self,
        generators: FJSPGenerator = None,
        params: FJSPParams = None,
        **kwargs,
    ):
        
        if generators is None and params is not None:
            if isinstance(params, EnvParamList):
                generators = [FJSPGenerator(param) for param in params]
                self.params = params[0]
            else:
                generators = FJSPGenerator(params)
                self.params = params

        super().__init__(generators)


    @staticmethod
    def _max_num_steps(g: FJSPGenerator):
        return g.size

    def _set_instance_params_and_id(self, td):
        self.num_jobs = td["proc_times"].size(1)
        self.n_ops = td["proc_times"].size(2)
        self.num_mas = td["proc_times"].size(3)
        instance_id = f"{self.num_jobs}j_{self.num_mas}m"
        return instance_id

    def _reset(self, td: TensorDict) -> TensorDict:
        batch_size = td.batch_size

        time_dtype = td["proc_times"].dtype
        finish_times = torch.full((*batch_size, self.num_jobs, self.n_ops), -9999, dtype=time_dtype)

        ma_assignment = torch.full(
            size=(*batch_size, self.num_jobs, self.n_ops), 
            fill_value=-1, 
            dtype=torch.long,
            device=td.device
        )
        ma_assignment.masked_fill_(td["pad_mask"], 9999)
        next_op = torch.zeros((*batch_size, self.num_jobs), dtype=torch.long)
        t_ma_idle = torch.zeros((*batch_size, self.num_mas), dtype=time_dtype)
        t_job_ready = torch.zeros((*batch_size, self.num_jobs), dtype=time_dtype)

        # random machine ids for embeddings, which stay consistent during rollout
        ma_ids = torch.ones((*batch_size, self.params.max_machine_id)).multinomial(self.num_mas, replacement=False).long()
        
        # reset feature space
        td = td.update(
            {
                "finish_times": finish_times,
                "ma_assignment": ma_assignment,
                "t_ma_idle": t_ma_idle,
                "t_job_ready": t_job_ready,
                "next_op": next_op,
                "reward": torch.zeros((*batch_size,), dtype=torch.get_default_dtype()),
                "job_done": torch.full((*batch_size, self.num_jobs), False),
                "done": torch.full((*batch_size,), False),
                "op_scheduled": torch.logical_or(ma_assignment.ge(0), td["pad_mask"]),
                # "scale_factor": reduce(td["proc_times"], "b ... -> b", reduction="max"),
                # this really seems to work better
                "scale_factor": reduce(td["proc_times"], "b ... -> 1", reduction="max").expand(batch_size),
                "ma_ids": ma_ids,
            },
        )

        if self.params.use_job_emb:
            job_ids = torch.ones((*batch_size, self.params.max_job_id)).multinomial(self.num_jobs, replacement=False).long()
            td.set("job_id", job_ids)
        return td


    def get_action_mask(self, td: TensorDict):
        """Simple action mask, which basically masks jobs only when they are done.
        The no_op action is always masked as long as the instance is not scheduled 
        completely.
        
        Returns:
            mask [bs, 1+num_jobs]: The action mask, where True means feasible
        
        """
        # NOTE: Each Job can be selected as long as it is not done
        # NOTE: 1 means feasible action, 0 means infeasible action
        bs = td.size(0)
        next_op_exp = td["next_op"].view(bs, self.num_jobs, 1, 1).expand(-1, -1, 1, self.num_mas)
        next_op_proc_times = td["proc_times"].gather(
            dim=2, 
            index=next_op_exp
        )
        # (bs, jobs, ma)
        job_ma_mask = next_op_proc_times.squeeze(2).gt(0)
        # (bs, jobs)
        job_mask = ~td["job_done"].view(bs, self.num_jobs, 1)
        job_ma_mask = job_mask & job_ma_mask
        # (bs, 1)
        no_op_mask = td["done"].view(bs, 1, 1).expand(-1, 1, self.num_mas).contiguous()
        # (bs, jobs+1, ma)
        mask = torch.cat((no_op_mask, job_ma_mask), dim=1)
         # (bs, ma, jobs+1)
        mask = rearrange(mask, "b j m -> b m j").contiguous()
        return mask


    def _translate_action(self, td):
        """This function translates an action into a machine, job tuple."""
        selected_job = td["action"]["node"] - 1
        selected_op = gather_by_index(td["next_op"], selected_job)
        selected_machine = td["action"]["agent"]
        return selected_job, selected_op, selected_machine

    def _step(self, td: TensorDict):
        # (bs)
        done = td["done"]
        # specify which batch instances require which operation
        no_op = td["action"]["node"].eq(0)
        no_op = no_op & ~done

        req_op = ~no_op & ~done

        # select only instances that perform a scheduling action
        td_op = td.masked_select(req_op)

        td_op = self._make_step(td_op)
        # update the tensordict
        td[req_op] = td_op

        if self.stepwise_reward:
            td.set("step_reward", self.get_step_reward(td))

        return td


    def _make_step(self, td: TensorDict) -> TensorDict:
        """
        Environment transition function
        """
        bs = td.size(0)
        # tot_ops = self.num_jobs * self.num_mas
        batch_idx = torch.arange(bs, device=td.device)

        # 3*(#req_op)
        selected_job, selected_op, selected_machine = self._translate_action(td)

        # update machine state
        proc_time_of_action = td["proc_times"][batch_idx, selected_job, selected_op, selected_machine]

        ma_available = td["t_ma_idle"][batch_idx, selected_machine]
        op_available = td["t_job_ready"][batch_idx, selected_job]

        start_time = torch.maximum(ma_available, op_available)
        finish_time = start_time + proc_time_of_action
        # update schedule
        #td["start_times"][batch_idx, selected_job, selected_op] = start_time
        td["finish_times"][batch_idx, selected_job, selected_op] = finish_time
        # update the state of the selected machine
        td["t_ma_idle"][batch_idx, selected_machine] = finish_time
        td["t_job_ready"][batch_idx, selected_job] = finish_time

        ############ update job states ##############
        n_ops_per_job = torch.sum(~td["pad_mask"], dim=-1)
        next_op = torch.scatter_add(
            td["next_op"], 1, selected_job.unsqueeze(1), torch.ones_like(td["next_op"])
        )
        # specify on which machine the selected operation will be processed
        td["ma_assignment"][batch_idx, selected_job, selected_op] = selected_machine
        # if all operations of a job are assigned to a machine, the job is done
        td["job_done"] = td["ma_assignment"].ge(0).all(-1)
        td["done"] = td["job_done"].all(dim=1, keepdim=False)
        td["next_op"] = torch.minimum(next_op, n_ops_per_job-1)
        # update proc times (remove selected operation)
        td["proc_times"][batch_idx, selected_job, selected_op] = 0
        td["op_scheduled"] = torch.logical_or(td["ma_assignment"].ge(0), td["pad_mask"])
        return td
    
    
    @staticmethod
    def load_instances(path, max_ops=None):
        files = list_files(path)
        if max_ops is None:
            max_ops = get_max_ops_from_files(files)
        instances = [read(file, max_ops=max_ops) for file in files]
        return instances

    def get_reward(self, td: TensorDict) -> torch.FloatTensor:
        assert td["done"].all(), "Cannot get reward before instance is finished"
        makespan = td["finish_times"].flatten(1,2).max(1).values.to(torch.get_default_dtype())
        reward = -makespan
        return reward
    
    def _get_step_reward(self, td):
        lbs = self.calc_lower_bound(td)
        # default is necessary in _reset() method
        old_lbs = td.get("lbs", lbs)
        td["lbs"] = lbs
        return -(lbs.max(1).values - old_lbs.max(1).values)
    
    @staticmethod
    def calc_lower_bound(td: TensorDict):
        proc_times = td["proc_times"]
        
        ############ step 1: determine earliest start time for next op of each job and expand this over all ops############
        eligible_job_ma_comb = gather_by_index(proc_times, td["next_op"], dim=2).gt(0)
        job_ma_ready = torch.maximum(td["t_job_ready"][..., None], td["t_ma_idle"][:, None])
        job_ma_ready[~eligible_job_ma_comb] = float("inf")
        earliest_start = torch.where(
            td["job_done"],
            td["finish_times"].max(-1).values,
            job_ma_ready.min(-1).values
        )
        lb = earliest_start[..., None].expand_as(td["finish_times"]).contiguous()

        ############ step 2: set lb of scheduled ops to their finish times ############
        lb[td["op_scheduled"]] = td["finish_times"][td["op_scheduled"]]

        ############ step 3: determine the sum of proc times for not scheduled ops ############
        # mask already scheduled operations
        # (b j o)
        min_op_proc_times = proc_times.masked_fill(proc_times.eq(0), float("inf")).min(-1).values
        min_op_proc_times[td["op_scheduled"] | td["pad_mask"]] = 0
        # using cumulative sum of remaining ops 
        lb = lb + min_op_proc_times.cumsum(-1)
        lb[td["pad_mask"]] = 0.0
        return lb


class MultiAgentFJSPEnv(MultiAgentEnvironment, FJSPEnv):

    name = "ma_fjsp"
    wait_op_idx = 0
    
    def __init__(
        self,
        generators: FJSPGenerator = None,
        params: FJSPParams = None,
        **kwargs,
    ):
        super().__init__(generators=generators, params=params)

    @property
    def num_agents(self):
        return self.num_mas    

    def _step(self, td: TensorDict):
        bs = td.size(0)
        actions = td["action"].split(1, dim=1)
        step_rewards = []
        for action in actions:
            action = action.squeeze(1)
            # (bs)
            done = td["done"]
            # specify which batch instances require which operation
            no_op = action["node"].eq(0)
            no_op = no_op & ~done

            req_op = ~no_op & ~done

            if not req_op.any() and not self.stepwise_reward:
                continue

            # add job and ma to td
            td.set("action", action)

            # select only instances that perform a scheduling action
            td_op = td.masked_select(req_op)

            td_op = self._make_step(td_op)
            # update the tensordict
            td[req_op] = td_op

            if self.stepwise_reward:
                reward = self.get_step_reward(td)
                step_rewards.append(reward)

        if self.stepwise_reward:
            td.set("step_reward", torch.stack(step_rewards, dim=1))

        return td
    
    def update_mask(self, mask: torch.Tensor, action: AgentAction, td: TensorDict, busy_agents: torch.Tensor):
        """Update mask after an agent acting during multi-agent rollout. True means feasible, False means infeasible"""
        bs, num_mas, num_jobs_plus_one = mask.shape
        selected_job = action["node"]
        # mask selected job
        mask = mask.scatter(-1, selected_job.view(bs, 1, 1).expand(-1, num_mas, 1), False)
        if self.params.use_skip_token:
            # always allow machines that are still idle to wait (for jobs to become available for example)
            mask[..., 0] = True
        else:
            # allow an idle machine to wait only if it cannot choose any job 
            mask[..., 0] = torch.logical_not(mask[..., 1:].any(-1))
        # lastly, mask all actions for the selected agent
        mask[busy_agents] = False
        return mask
    