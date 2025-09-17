import os
import torch
from tensordict import TensorDict
from macsim.envs.base import Generator
from macsim.envs.env_args import FFSPParams


def batch_constrained_partition(parts, sum_value, x, min_value, max_value):
    assert x * min_value <= sum_value <= x * max_value, "Impossible constraints"
    B = parts.size(0)
    # Start with random values within the range for a batch of B
    total = parts.sum(dim=1)

    # Adjust each batch separately
    diff = sum_value - total  # Compute the difference to fix
    while (diff != 0).any():  # While any row still needs adjustment
        idx = torch.randint(0, x, (B,))  # Select random indices per batch
        mask_increase = (diff > 0) & (parts[torch.arange(B), idx] < max_value)
        mask_decrease = (diff < 0) & (parts[torch.arange(B), idx] > min_value)

        # Apply changes where possible
        parts[torch.arange(B), idx] += mask_increase.to(torch.int) - mask_decrease.to(torch.int)
        diff = sum_value - parts.sum(dim=1)  # Recalculate differences

    return parts


def generate_padded_sequences(ma_cnts, pad_value=-1):
    B, x = ma_cnts.shape
    max_len = ma_cnts.sum(dim=1).max().item()  # Find the longest sequence
    max_cnt = ma_cnts.max()
    sequences = []
    for row in ma_cnts:
        seq = torch.cat([torch.full((ma_cnt.item(),), idx) for idx, ma_cnt in enumerate(row)])  # Expand indices
        padded_seq = torch.full((max_len,), pad_value)  # Create padding
        padded_seq[:len(seq)] = seq  # Fill with sequence data
        sequences.append(padded_seq)

    return torch.stack(sequences)


class FFSPGenerator(Generator):
    """Data generator for the Flexible Flow Shop Scheduling Problem (FFSP)."""

    def __init__(self, params: FFSPParams):

        self.num_stage = params.num_stage
        self.min_ma_per_stage = params.min_ma_per_stage
        self.max_ma_per_stage = params.max_ma_per_stage
        self.max_machine_total = params.max_ma_per_stage * params.num_stage
        self.num_job = params.num_jobs
        self.min_time = params.min_processing_time
        self.max_time = params.max_processing_time
        self.ma_cnt_prob = torch.tensor(params.ma_cnt_prob)[None] if params.ma_cnt_prob is not None else None
        self.ma_pad_value = params.ma_pad_value

    @property
    def id(self):
        return f"{self.num_job}j_{self.max_ma_per_stage}m_{self.num_stage}s"

    @property
    def size(self):
        return self.num_job * self.num_stage

    def _simulate_processing_times(self, stage_table):
        bs, max_num_ma = stage_table.shape
        # Init observation: running time of each job on each machine
        proc_times = torch.randint(
            low=self.min_time,
            high=self.max_time, # NOTE this is wrong but consistent with original paper
            size=(bs, self.num_job, max_num_ma),
        )
        return proc_times
    
    def _generate(self, batch_size) -> TensorDict:
        # (bs, num_stage)
        if self.ma_cnt_prob is not None:
            machine_cnt = self.ma_cnt_prob.expand(*batch_size, -1).multinomial(self.num_stage, replacement=True)
            machine_cnt = machine_cnt + self.min_ma_per_stage
            
        else:
            machine_cnt = torch.randint(
                low=self.min_ma_per_stage,
                high=self.max_ma_per_stage+1,
                size=(*batch_size, self.num_stage)
            )
        # (bs, max_num_ma)
        stage_table = generate_padded_sequences(machine_cnt, pad_value=self.ma_pad_value)
        # (bs, num_job, max_num_ma)
        proc_times = self._simulate_processing_times(stage_table)

        return TensorDict(
            {
                "proc_times": proc_times,
                "machine_cnt": machine_cnt,
                "stage_table": stage_table
            },
            batch_size=batch_size,
        )

    @staticmethod
    def save_instances(td, path):
        """Save TensorDict back to the format expected by `load_instances`."""
        proc_times = td["proc_times"]
        machine_cnt = td["machine_cnt"][0]  # shape: (n_stages,)

        # Recover the list of original tensors by slicing along the last dim
        split_sizes = machine_cnt.tolist()  # e.g., [2, 3, 1]
        data = torch.split(proc_times, split_sizes, dim=-1)
        
        # Save as a list of tensors
        torch.save(list(data), path)