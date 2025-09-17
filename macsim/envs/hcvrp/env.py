from typing import Optional

import torch
import numpy as np

from tensordict.tensordict import TensorDict

from macsim.envs.base import AgentAction
from macsim.envs.env_args import HCVRPParams
from macsim.utils.ops import gather_by_index
from macsim.utils.config import EnvParamList
from macsim.envs.base import MultiAgentEnvironment, Environment

from .generator import HCVRPGenerator


def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems


class HCVRPEnv(Environment):
    """Heterogeneous Capacitated Vehicle Routing Problem (HCVRP) environment."""

    name = "hcvrp"

    def __init__(
        self,
        generators: HCVRPGenerator = None,
        params: HCVRPParams = None,
        **kwargs,
    ):
        if generators is None and params is not None:
            if isinstance(params, EnvParamList):
                generators = [HCVRPGenerator(param) for param in params]
                self.params = params[0]
            else:
                generators = HCVRPGenerator(params)
                self.params = params

        super().__init__(generators)

        self.num_cities = None
        self.num_agents = None


    @staticmethod
    def get_distance(x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Euclidean distance between two tensors of shape `[..., n, dim]`"""
        return (x - y).norm(p=2, dim=-1, keepdim=keepdim)

    @staticmethod
    def _max_num_steps(g: HCVRPGenerator):
        return g.num_loc * 2
    
    def _set_instance_params_and_id(self, td):
        self.num_cities = td["locs"].size(1) - 1
        self.num_agents = td["capacity"].size(1)
        instance_id = f"{self.num_cities}n_{self.num_agents}m"
        return instance_id

    def _reset(self, td: Optional[TensorDict] = None) -> TensorDict:
        batch_size = td.batch_size
        device = td.device

        # Record parameters
        # num_agents = self.generator.num_agents
        # num_loc_all = self.generator.num_loc + num_agents
        num_agents = td["speed"].size(-1)
        num_loc_all = td["locs"].size(-2)

        # Padding depot demand as 0 to the demand
        demand_depot = torch.zeros(
            (*batch_size, 1), dtype=torch.float32, device=device
        )
        demand = torch.cat((demand_depot, td["demand"]), -1)

        # Init current node
        current_node = torch.zeros((*batch_size, num_agents), dtype=torch.int64, device=device)
        curr_len = torch.zeros(
            (*batch_size, num_agents), dtype=torch.float32, device=device
        )

        # Init visited
        visited = torch.zeros((*batch_size, num_loc_all), dtype=torch.bool, device=device)
        # Create reset TensorDict
        td_reset = TensorDict(
            {
                "locs": td["locs"],
                "demand": demand,
                "current_length": curr_len,
                "current_node": current_node,
                "used_capacity": torch.zeros((*batch_size, num_agents), device=device),
                "agents_capacity": td["capacity"],
                "agents_speed": td["speed"],
                "visited": visited,
                "done": torch.zeros((*batch_size,), dtype=torch.bool, device=device),
            },
            batch_size=batch_size,
            device=device,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def _step(self, td: TensorDict) -> TensorDict:
        """
        Keys:
            - action [batch_size, num_agents]: action taken by each agent
        """
        num_locs = td["locs"].size(-2)
        bs = td.size(0)
        batch_idx = torch.arange(bs, device=td.device)

        action: AgentAction = td["action"]
        # (bs)
        agent_idx = action["agent"]
        # (bs)
        node_idx = action["node"].clone()
        prev_node = gather_by_index(td["current_node"], agent_idx)
        node_idx = torch.where(node_idx == num_locs, prev_node, node_idx)
        # if an agents selects the wait operation - set selected node to its current node
        stay_flag = prev_node == node_idx

        # get old and new locations
        current_loc = gather_by_index(td["locs"], node_idx)
        prev_loc = gather_by_index(td["locs"], prev_node)
        # update current node of agent
        td["current_node"] = torch.scatter(
            td["current_node"],
            dim=1,
            index=agent_idx.view(bs,1),
            src=node_idx.view(bs,1)
        )
        # Update the current length; (bs, num_agent)
        td["current_length"] = torch.scatter_add(
            td["current_length"],
            dim=1, 
            index=agent_idx.view(bs, 1),
            src=self.get_distance(prev_loc, current_loc, keepdim=True)
        )

        # Update the used capacity
        # Increase used capacity if not visiting the depot, otherwise set to 0
        selected_demand = gather_by_index(td["demand"], node_idx, dim=-1)
        selected_demand = selected_demand * (~stay_flag).float()
        # Update the capacity for current agent; (bs, num_agent)
        td["used_capacity"] = torch.scatter_add(
            td["used_capacity"],
            dim=1, 
            index=agent_idx.view(bs, 1),
            src=selected_demand.view(bs, 1)
        )
        td["used_capacity"] = torch.where(
            td["current_node"] == 0,
            0,
            td["used_capacity"] 
        )

        # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
        # Add one dimension since we write a single value
        td["visited"][batch_idx, node_idx] = True
        td["demand"][batch_idx, node_idx] = 0
        # update the done and reward
        td["done"] = td["visited"][..., 1:].all(-1)

        # td.set("action_mask", self.get_action_mask(td))
        return td


    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """action mask, True refers to feasible action"""
        num_agents = td["current_node"].size(-1)

        # Init action mask for each agent with all not visited nodes
        action_mask = torch.repeat_interleave(
            ~td["visited"][..., None, :], dim=-2, repeats=num_agents
        )
        # enable agents to wait at their current location
        # action_mask.scatter_(-1, td["current_node"][..., None], True)

        # Can not visit the node if the demand is more than the remaining capacity
        remain_capacity = td["agents_capacity"] - td["used_capacity"]
        within_capacity_flag = td["demand"][:, None] <= remain_capacity[..., None] 
        action_mask &= within_capacity_flag

        # The depot is not available if **all** the agents are at the depot and the task is not finished
        all_back_flag = torch.sum(td["current_node"] > 0, dim=-1) == 0
        has_finished_early = all_back_flag & ~td["done"]

        depot_mask = ~has_finished_early[..., None]  # 1 means we can visit

        # If no available nodes outside (all visited), make the depot always available
        all_visited_flag = (
            torch.sum(~td["visited"][..., 1:], dim=-1, keepdim=True) == 0
        )
        depot_mask |= all_visited_flag
        action_mask[..., 0] = depot_mask
        # avoid agent selects its current node -> for this we have the wait op
        action_mask.scatter_(-1, td["current_node"][..., None], False)
        # wait op mask
        no_op_mask = td["done"].view(-1, 1, 1).expand(-1, self.num_agents, 1).contiguous()
        action_mask = torch.cat((action_mask, no_op_mask), dim=-1)
        return action_mask

    def get_reward(self, td: TensorDict) -> TensorDict:
        """
        Min-max
        """
        current_length = td["current_length"]

        # Adding the final distance to the depot
        current_loc = gather_by_index(td["locs"], td["current_node"])
        depot_loc = td["locs"][:, 0]
        current_length = td["current_length"] + self.get_distance(depot_loc.unsqueeze(1), current_loc)

        # Calculate the time
        current_time = current_length / td["agents_speed"]
        max_time = current_time.max(dim=1)[0]
        return -max_time  # note: reward is negative of the total time (maximize)

    def load_instances(self, path, num_agents: int = None):
        data = np.load(path)
        ds_size = data["locs"].shape[0]
        td = TensorDict(
             {
                'locs': torch.from_numpy(data['locs']).float(),
                'demand': torch.from_numpy(data['demand']).float(),
                'capacity': torch.from_numpy(data['capacity']).float(),
                'speed': torch.from_numpy(data['speed']).float()
            },
            batch_size=[ds_size]
        )
        return td
    
    @staticmethod
    def augment_states(td: TensorDict, *args, **kwargs):
        aug_locs = augment_xy_data_by_8_fold(td["locs"])
        td_aug = td.repeat(8)
        td_aug["locs"] = aug_locs
        return td_aug

class MultiAgentHCVRPEnv(MultiAgentEnvironment, HCVRPEnv):

    name = "ma_hcvrp"
    wait_op_idx = -1

    def __init__(
        self,
        generators: HCVRPGenerator = None,
        params: HCVRPParams = None,
        **kwargs,
    ):
        super().__init__(generators=generators, params=params)


    def _step(self, td: TensorDict):
        actions = td["action"].split(1, dim=1)
        for action in actions:
            # add job and ma to td
            td.set("action", action.squeeze(1))
            td = HCVRPEnv()._step(td)

        return td
    
    def update_mask(self, mask: torch.Tensor, action: AgentAction, td: TensorDict, busy_agents: torch.Tensor):
        """Update mask after an agent acting during multi-agent rollout. True means feasible, False means infeasible"""
        selected_node = action["node"]
        depot_mask = mask[..., 0].clone()
        # mask selected node
        mask = mask.scatter(-1, selected_node.view(-1, 1, 1).expand(-1, self.num_agents, 1), False)
        # always allow vehicles to go back / wait at the depot if at least on is moving
        mask[..., 0] = depot_mask
        if self.params.use_skip_token:
            # always allow machines that are still idle to wait (for jobs to become available for example)
            mask[..., -1] = True
        else:
            # allow an idle machine to wait only if it cannot choose any job 
            mask[..., -1] = torch.logical_not(mask[..., :-1].any(-1))
        # lastly, mask all actions for the selected agent
        mask[busy_agents] = False
        return mask