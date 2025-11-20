import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
from rl4co.utils.ops import gather_by_index, get_distance


class State_HCVRP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor

    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    depot_num: torch.Tensor
    agent_idx: torch.Tensor  # index of current agent
    agent_per: torch.Tensor  # order of agents are currently moving
    cur_node: torch.Tensor  # Previous actions
    to_assign: torch.Tensor  # Previous actions
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor  # Keeps track of tour lengths corresponding to each agent
    assign: torch.Tensor
    cur_coord: torch.Tensor  # Keeps track of current coordinates
    i: torch.Tensor  # Keeps track of step
    agent_counter: torch.Tensor  # counts number of agents that finished their tour
    left_city: torch.Tensor  # Number of left cities
    depot_distance: torch.Tensor  # Distance from depot to all cities
    remain_max_distance: torch.Tensor  # Max distance from depot among left cities
    max_distance: torch.Tensor  # Max distance from depot among all cities
    agents_speed: torch.Tensor
    demand: torch.Tensor
    agents_capacity: torch.Tensor
    used_capacity: torch.Tensor
    scale_factor: float
    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    @staticmethod
    def initialize(td, agent_num, agent_per, depot_num=1, visited_dtype=torch.uint8):
        device=td["loc"].device

        left_city = td['loc'].size(1)
        batch_size, n_loc, _ = td['loc'].size()
        pomo_size = agent_per.size(0)
        all_loc = torch.cat((td['depot'][:, :depot_num, :], td['loc']), dim=1)[:, None, :, :].expand(-1, pomo_size, -1, -1)
        # cur_node = agent_per[None, :, 0:1].expand(batch_size, -1, -1) 
        current_node = torch.zeros((batch_size, pomo_size, 1), device=device, dtype=torch.long)

        depot_distance = torch.cdist(all_loc, all_loc, p=2)
        depot_distance = depot_distance[:, :, :depot_num, :].min(dim=-2)[0]
        max_distance = depot_distance.max(dim=-1, keepdim=True)[0]

        current_cord = gather_by_index(all_loc, current_node, dim=-2)
        agent_per = agent_per[None, :, :].expand(batch_size, -1, -1)

        scale_factor = td["capacity"].max().item()
        return State_HCVRP(
            loc=all_loc,
            depot_num=depot_num,
            to_assign=torch.ones(batch_size, pomo_size, dtype=torch.bool, device=device),
            dist=(all_loc[:, :, None, :] - all_loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=device)[:, None],  # Add steps dimension
            agent_idx=agent_per[..., :1],
            agent_per=agent_per,
            cur_node=current_node,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=torch.zeros(batch_size, pomo_size, left_city + depot_num, dtype=torch.uint8, device=device),
            assign=torch.zeros(batch_size, pomo_size, agent_num, dtype=torch.int64, device=device),
            lengths=torch.zeros(batch_size, pomo_size, agent_num, device=device),
            cur_coord=current_cord,
            agent_counter=torch.zeros(batch_size, pomo_size, 1, dtype=torch.int64, device=device),
            i=torch.zeros(1, dtype=torch.int64, device=device),  # Vector with length num_steps
            left_city=left_city * torch.ones(batch_size, pomo_size, 1, dtype=torch.long, device=device),
            remain_max_distance=max_distance,
            max_distance=max_distance,
            depot_distance=depot_distance,
            agents_speed=td["speed"].unsqueeze(1).expand(batch_size, pomo_size, -1),
            demand=td["demand"].unsqueeze(1).expand(batch_size, pomo_size, -1) / scale_factor,
            agents_capacity=td["capacity"].unsqueeze(1).expand(batch_size, pomo_size, -1) / scale_factor,
            used_capacity=torch.zeros(batch_size, pomo_size, dtype=torch.float, device=device),
            scale_factor=scale_factor
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths / self.agents_speed
    
    @property
    def at_depot(self):
        return self.cur_node < self.depot_num

    def update(self, selected):
        agent_num = self.lengths.size(2)
        # Update the state

        # City idices starts from depot_num + 1
        is_city = selected >= self.depot_num
        current_node = selected[:, :, None]  # Add dimension for step
        self.left_city[is_city] -= 1

        current_agent = self.agent_idx
        # if agent stays at depot, we switch to the next one
        is_last_agent = self.agent_counter == agent_num - 1
        next_agent = self.at_depot & (~is_city[..., None]) & (~is_last_agent)
        
        # If agent move to other city, then, the distance between visited city and depot is 0
        depot_distance = self.depot_distance.scatter(-1, current_node, 0)
        remain_max_distance = self.depot_distance.max(dim=-1, keepdim=True)[0]

        cur_coord = self.loc.gather(2, current_node[..., None].expand(-1, -1, -1, 2)).squeeze(-2)

        path_lengths = (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
        lengths = self.lengths.scatter_add(-1, current_agent, path_lengths[..., None])

        # selected_demand = gather_by_index(self.demand, selected, dim=-1)
        selected_demand = torch.where(is_city, self.demand.gather(-1, (selected[..., None]-1).clamp_min(0)).squeeze(-1), 0)
        used_capacity = ((self.used_capacity + selected_demand) * is_city).float()
        
        demand = self.demand.clone()
        demand[is_city] = self.demand[is_city].scatter(-1, current_node[is_city]-self.depot_num, 0)
        
        # Current agent comes back to depot when it selects its own index
        self.agent_counter[next_agent] += torch.ones(self.agent_counter[next_agent].shape, dtype=torch.int64, device=self.agent_counter.device)

        visited_ = self.visited_.scatter(-1, current_node, 1)

        if ((is_city == False).all() and (self.agent_counter == self.agent_per.size(1)).all()):
            return self._replace(agent_idx=current_agent, cur_node=current_node, visited_=visited_, demand=demand,
                                 lengths=lengths, cur_coord=cur_coord, i=self.i + 1, depot_distance=depot_distance, 
                                 remain_max_distance=remain_max_distance)
        
        next_agent_idx = self.agent_per.gather(-1, self.agent_counter)
        return self._replace(agent_idx=next_agent_idx, cur_node=current_node, visited_=visited_, demand=demand,
                             lengths=lengths, cur_coord=cur_coord, i=self.i + 1, depot_distance=depot_distance, 
                             remain_max_distance=remain_max_distance, used_capacity=used_capacity)

    def all_finished(self):
        all_visited = self.visited_.bool().all(-1)
        return (all_visited & self.at_depot.squeeze(-1)).all()

    def get_current_node(self):
        return self.cur_node



    def get_mask(self):
        visited_loc = self.visited_.bool()
        agent_num = self.lengths.size(2)

        current_agent_idx = self.agent_idx
        is_last_agent = self.agent_counter == agent_num - 1
        is_last_agent_at_depot = is_last_agent & self.at_depot
        # Init action mask for each agent with all not visited nodes
        action_mask =  ~visited_loc.clone()

        # Can not visit the node if the demand is more than the remaining capacity
        remain_capacity = self.agents_capacity.gather(-1, self.agent_idx) - self.used_capacity.unsqueeze(2)
        within_capacity_flag = self.demand <= remain_capacity
        action_mask[...,self.depot_num:] = action_mask[...,self.depot_num:] & within_capacity_flag

        # The depot is not available if a fresh agent is at the depot
        curr_length = self.lengths.gather(-1, current_agent_idx)
        depot_mask = ~(self.at_depot & curr_length.eq(0))   # 1 means we can visit [batch_size, num_agents]
        # last agent may not remain at the depot (which would typically induce an agent switch) unless all nodes have been visited
        depot_mask[is_last_agent_at_depot] = action_mask[is_last_agent_at_depot.squeeze(-1)][..., self.depot_num:].sum(-1) == 0
        # If no available nodes outside (all visited), make the depot always available
        all_visited_flag = (
            torch.sum(~visited_loc[..., self.depot_num:], dim=-1, keepdim=True) == 0
        )
        depot_mask |= all_visited_flag

        action_mask[..., :self.depot_num] = depot_mask

        return ~action_mask 


    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions
