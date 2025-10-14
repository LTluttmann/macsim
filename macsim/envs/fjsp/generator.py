import torch
from tensordict import TensorDict
from macsim.envs.base import Generator
from macsim.envs.env_args import FJSPParams

_float = torch.get_default_dtype()

def randint_w_elementwise_bounds(min_tensor, max_tensor):
    """
    Generate random integers where each element is bounded by corresponding
    elements in min_tensor and max_tensor.
    
    Args:
        min_tensor (torch.Tensor): Tensor of lower bounds (inclusive)
        max_tensor (torch.Tensor): Tensor of upper bounds (inclusive)
    
    Returns:
        torch.Tensor: Random integers within the specified bounds
    """
    # Validate inputs have same shape
    assert min_tensor.shape == max_tensor.shape, "min and max tensors must have same shape"
    assert torch.all(min_tensor <= max_tensor), "min values must be <= max values"
    
    # Generate uniform random numbers between 0 and 1
    rand = torch.rand_like(min_tensor, dtype=_float)
    
    # Scale and shift the random numbers to the desired range
    # Adding 1 to include the max value in sampling
    range_size = (max_tensor - min_tensor + 1).to(_float)
    
    # Floor the scaled random numbers to get integers
    return torch.floor(rand * range_size + min_tensor).to(torch.int64)



class FJSPGenerator(Generator):
    """Data generator for the Flexible Job-Shop Scheduling Problem (FJSP).
    """

    def __init__(self, params: FJSPParams):

        self.num_jobs = params.num_jobs
        self.num_mas = params.num_machines
        self.min_ops_per_job = params.min_ops_per_job or params.num_machines
        self.max_ops_per_job = params.max_ops_per_job or self.min_ops_per_job
        self.min_processing_time = params.min_processing_time
        self.max_processing_time = params.max_processing_time
        self.min_eligible_ma_per_op = params.min_eligible_ma_per_op or 1
        self.max_eligible_ma_per_op = params.max_eligible_ma_per_op or params.num_machines
        self.n_ops_max = self.max_ops_per_job * params.num_jobs
        # determines whether to use a fixed number of total operations or let it vary between instances
        self.same_mean_per_op = params.same_mean_per_op
        self.max_machine_id: int = params.max_machine_id

    @property
    def id(self):
        return f"{self.num_jobs}j_{self.num_mas}m"

    @property
    def size(self):
        return self.n_ops_max

    def _simulate_processing_times(
        self, n_eligible_per_ops: torch.Tensor
    ) -> torch.Tensor:
        # n_eligible_per_ops: (bs, jobs, ops)
        bs, n_jobs, n_ops = n_eligible_per_ops.shape
        n_mas = self.num_mas

        # (bs, jobs, ops, machines)
        ma_seq_per_ops = torch.arange(1, n_mas + 1)[None, None].expand(
            bs, n_jobs, n_ops, n_mas
        )
        # generate a matrix of size (ops, mas) per job, each row having as many ones as the 
        # operation eligible machines. E.g. n_eligible_per_ops=[1,3,2]; num_mas=4
        # [[1,0,0,0],
        #   1,1,1,0],
        #   1,1,0,0]]
        # This will be shuffled randomly to generate a machine-operation mapping
        ma_ops_edges_unshuffled = (ma_seq_per_ops <= n_eligible_per_ops[..., None]).to(_float)
        # random shuffling
        idx = torch.rand_like(ma_ops_edges_unshuffled).argsort()
        ma_ops_edges = ma_ops_edges_unshuffled.gather(3, idx)

        # (bs, max_ops, machines)
        if self.same_mean_per_op:
            # simulation procedure used by Song et al.
            proc_times = torch.ones((bs, n_jobs, n_ops, n_mas))
            proc_time_means = torch.randint(
                self.min_processing_time, self.max_processing_time + 1, (bs, n_jobs, n_ops)
            )
            low_bounds = torch.maximum(
                torch.full_like(proc_times, self.min_processing_time),
                (proc_time_means * (1 - 0.2)).round().unsqueeze(-1),
            )
            high_bounds = (
                torch.minimum(
                    torch.full_like(proc_times, self.max_processing_time),
                    (proc_time_means * (1 + 0.2)).round().unsqueeze(-1),
                )
            )

            proc_times = randint_w_elementwise_bounds(low_bounds, high_bounds)

        else:
            proc_times = torch.randint(
                self.min_processing_time,
                self.max_processing_time + 1,
                size=(bs, n_jobs, n_ops, n_mas),
            )

        # remove proc_times for which there is no corresponding ma-ops connection
        proc_times = proc_times * ma_ops_edges
        return proc_times # .to(torch.int16)

    def _generate(self, batch_size) -> TensorDict:
        # simulate how many operations each job has
        n_ops_per_job = torch.randint(
            self.min_ops_per_job,
            self.max_ops_per_job + 1,
            size=(*batch_size, self.num_jobs),
        )

        # generate a mask, specifying which operations are padded
        pad_mask = torch.arange(self.max_ops_per_job)[None, None, :].expand(
            *batch_size, self.num_jobs, self.max_ops_per_job
        ).contiguous()
        pad_mask = pad_mask.ge(n_ops_per_job[..., None].expand_as(pad_mask))

        # here we simulate the eligible machines per operation and the processing times
        n_eligible_per_ops = torch.randint(
            self.min_eligible_ma_per_op,
            self.max_eligible_ma_per_op + 1,
            (*batch_size, self.num_jobs, self.max_ops_per_job),
        )
        n_eligible_per_ops[pad_mask] = 0

        # simulate processing times for machine-operation pairs
        # (bs, num_mas, n_ops_max)
        proc_times = self._simulate_processing_times(n_eligible_per_ops)

        td = TensorDict(
            {
                "proc_times": proc_times,
                "pad_mask": pad_mask,
            },
            batch_size=batch_size,
        )

        return td