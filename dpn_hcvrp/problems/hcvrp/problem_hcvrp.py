from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.hcvrp.state_hcvrp import State_HCVRP
from utils.beam_search import beam_search
import numpy as np
from tensordict.tensordict import TensorDict


class HCVRP(object):
    NAME = 'hcvrp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return HCVRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return State_HCVRP.initialize(*args, **kwargs)




class HCVRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(HCVRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            self.data = self.load_data(filename)
        else:
            gen = HCVRPGenerator(num_loc=size)
            self.data = gen.generate(num_samples)

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def load_data(fpath, batch_size=[]):
        import numpy as np
        data = np.load(fpath)
        ds_size, num_agents = data['capacity'].shape
        locs = torch.from_numpy(data['locs']).float()
        depot = locs[:, 0]
        locs = locs[:, 1:]
        demand = torch.from_numpy(data['demand']).float()
        capacity = torch.from_numpy(data['capacity']).float()
        speed = torch.from_numpy(data['speed']).float()
        td = [
            {
                'loc': locs[i],
                'depot': depot[i],
                'demand': demand[i],
                'capacity': capacity[i],
                'speed': speed[i]
            } 
            for i in range(ds_size)
        ]
        return td


class HCVRPGenerator:


    def __init__(self, num_loc= 60, num_agents= 3, min_demand=1, max_demand=9, min_capacity=20, max_capacity=40, min_speed=0.5, max_speed=1.0):
        self.num_loc = num_loc
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.num_agents = num_agents

    def generate(self, batch_size):
        # If depot_sampler is None, sample the depot from the locations
        locs = torch.FloatTensor(batch_size, self.num_loc + 1, 2).uniform_(
            0, 1
        )
        depot = locs[:, 0]
        locs = locs[:, 1:]
        # Sample demands
        demand = torch.randint(self.min_demand, self.max_demand + 1, size=(batch_size, self.num_loc)).float()

        # Sample capacities
        capacity = torch.randint(self.min_capacity, self.max_capacity + 1, size=(batch_size, self.num_agents)).float()

        # Sample speed
        speed = torch.FloatTensor(batch_size, self.num_agents).uniform_(
            self.min_speed, self.max_speed
        )

        return [
            {
                "depot": depot[i],
                "loc": locs[i],
                "demand": demand[i],
                "capacity": capacity[i],
                "speed": speed[i],
            }
            for i in range(batch_size)
        ]
