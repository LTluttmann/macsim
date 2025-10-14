from typing import Callable, Union

import torch
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from macsim.envs.base import Generator
from macsim.envs.env_args import HCVRPParams


class HCVRPGenerator(Generator):
    """Data generator for the Heterogeneous Capacitated Vehicle Routing Problem (HCVRP).

    Args:
        - num_loc: Number of customers.
        - min_loc: Minimum location of the customers.
        - max_loc: Maximum location of the customers.
        - loc_distribution: Distribution of the locations of the customers.
        - depot_distribution: Distribution of the location of the depot.
        - min_demand: Minimum demand of the customers.
        - max_demand: Maximum demand of the customers.
        - demand_distribution: Distribution of the demand of the customers.
        - min_capacity: Minimum capacity of the agents.
        - max_capacity: Maximum capacity of the agents.
        - capacity_distribution: Distribution of the capacity of the agents.
        - min_speed: Minimum speed of the agents.
        - max_speed: Maximum speed of the agents.
        - speed_distribution: Distribution of the speed of the agents.
        - num_agents: Number of agents.

    Returns:
        A TensorDict containing the following keys:
            - locs [batch_size, num_loc, 2]: locations of the customers
            - depot [batch_size, 2]: location of the depot
            - demand [batch_size, num_loc]: demand of the customers
            - capacity [batch_size, num_agents]: capacity of the agents, different for each agents
            - speed [batch_size, num_agents]: speed of the agents, different for each agents

    Notes:
        - The capacity setting from 2D-Ptr paper is hardcoded to 20~41. It should change
            based on the size of the problem.
        - ? Demand and capacity are initialized as integers and then converted to floats.
            To avoid zero demands, we first sample from [min_demand - 1, max_demand - 1]
            and then add 1 to the demand.
        - ! Note that here the demand is not normalized by the capacity by default.
    """

    def __init__(self, params: HCVRPParams):
        self.num_loc = params.num_loc
        self.min_loc = params.min_loc
        self.max_loc = params.max_loc
        self.min_demand = params.min_demand
        self.max_demand = params.max_demand
        self.min_capacity = params.min_capacity
        self.max_capacity = params.max_capacity
        self.min_speed = params.min_speed
        self.max_speed = params.max_speed
        self.num_agents = params.num_agents

    @property
    def id(self):
        return f"{self.num_loc}n_{self.num_agents}m"
    
    @property
    def size(self):
        return self.num_loc


    def _generate(self, batch_size) -> TensorDict:

        # If depot_sampler is None, sample the depot from the locations
        locs = torch.FloatTensor(*batch_size, self.num_loc + 1, 2).uniform_(
            self.min_loc, self.max_loc
        )

        # Sample demands
        demand = torch.randint(self.min_demand, self.max_demand + 1, size=(*batch_size, self.num_loc)).float()

        # Sample capacities
        capacity = torch.randint(self.min_capacity, self.max_capacity + 1, size=(*batch_size, self.num_agents)).float()

        # Sample speed
        speed = torch.FloatTensor(*batch_size, self.num_agents).uniform_(
            self.min_speed, self.max_speed
        )

        return TensorDict(
            {
                "locs": locs,
                "demand": demand,
                "capacity": capacity,
                "speed": speed,
            },
            batch_size=batch_size,
        )
    


def generate_hcvrp_data(seed,dataset_size, hcvrp_size, veh_num):
    # https://github.com/farkguidao/2D-Ptr/blob/master/generate_data.py
    import numpy as np
    rnd = np.random.RandomState(seed)

    loc = rnd.uniform(0, 1, size=(dataset_size, hcvrp_size + 1, 2))
    depot = loc[:, -1:]
    cust = loc[:, :-1]
    
    
    d = rnd.randint(1, 10, [dataset_size, hcvrp_size + 1])
    d = d[:, :-1]  # the demand of depot is 0, which do not need to generate here

    # vehicle feature
    speed = rnd.uniform(0.5, 1, size=(dataset_size, veh_num))
    cap = rnd.randint(20, 41, size=(dataset_size, veh_num))

    locs = np.concatenate((depot, cust), axis=1)
    
    data = {
        'locs': locs.astype(np.float32),
        'demand': d.astype(np.float32),
        'capacity': cap.astype(np.float32),
        'speed': speed.astype(np.float32)
    }
    
    return data


if __name__ == "__main__":
    import numpy as np
    size = (60,5)
    seed = 24610
    data = generate_hcvrp_data(seed, 1280, *size)
    np.savez(f"hcvrp_{'_'.join([str(x) for x in size])}_seed{seed}.npz", **data)