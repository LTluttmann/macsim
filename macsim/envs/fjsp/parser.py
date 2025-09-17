import os

from functools import partial
from pathlib import Path
from typing import List, Tuple, Union

import torch

from tensordict import TensorDict

ProcessingData = List[Tuple[int, int]]


def list_files(path, sort_fn = None):
    files = [
        os.path.join(path, f) 
        for f in os.listdir(path) 
        if os.path.isfile(os.path.join(path, f))
    ]
    assert len(files) > 0, f"No instance files found in path {path}"
    if sort_fn is not None:
        files = sorted(files, key=sort_fn)
    return files


def parse_job_line(line: Tuple[int]) -> Tuple[ProcessingData]:
    """
    Parses a FJSPLIB job data line of the following form:

        <num operations> * (<num machines> * (<machine> <processing time>))

    In words, the first value is the number of operations. Then, for each
    operation, the first number represents the number of machines that can
    process the operation, followed by, the machine index and processing time
    for each eligible machine.

    Note that the machine indices start from 1, so we subtract 1 to make them
    zero-based.
    """
    num_operations = line[0]
    operations = []
    idx = 1

    for _ in range(num_operations):
        num_pairs = int(line[idx]) * 2
        machines = line[idx + 1 : idx + 1 + num_pairs : 2]
        durations = line[idx + 2 : idx + 2 + num_pairs : 2]
        operations.append([(m, d) for m, d in zip(machines, durations)])

        idx += 1 + num_pairs

    return operations


def get_max_ops_of_instance(file):
    lines = file2lines(file)
    max_ops = max([line[0] for line in lines[1:]])
    return max_ops

def get_max_ops_from_files(files):
    return max(map(get_max_ops_of_instance, files))


def read(loc: Path, max_ops=None):
    """
    Reads an FJSPLIB instance.

    Args:
        loc: location of instance file
        max_ops: optionally specify the maximum number of total operations (will be filled by padding)

    Returns:
        instance: the parsed instance
    """
    lines = file2lines(loc)

    # First line contains metadata.
    num_jobs, num_machines = lines[0][0], lines[0][1]

    # The remaining lines contain the job-operation data, where each line
    # represents a job and its operations.
    jobs = [parse_job_line(line) for line in lines[1:]]
    n_ops_per_job = torch.LongTensor([len(x) for x in jobs])
    max_ops = max_ops or int(n_ops_per_job.max().item())

    # generate a mask, specifying which operations are padded
    pad_mask = torch.arange(max_ops)[None, :].expand(num_jobs, max_ops).contiguous()
    pad_mask = pad_mask.ge(n_ops_per_job[:, None].expand_as(pad_mask))

    proc_times = torch.zeros((num_jobs, max_ops, num_machines), dtype=torch.get_default_dtype())
    for i, job in enumerate(jobs):
        for j, op in enumerate(job):
            for ma, dur in op:
                # subtract one to let indices start from zero
                proc_times[i, j, ma-1] = dur

    assert proc_times[pad_mask].eq(0).all()

    td = TensorDict(
        {
            "proc_times": proc_times,
            "pad_mask": pad_mask,
        },
        batch_size=[], # NOTE no batch_size to enable torch.stack
    )

    return td


def file2lines(loc: Union[Path, str]) -> List[List[int]]:
    with open(loc, "r") as fh:
        lines = [line for line in fh.readlines() if line.strip()]

    def parse_num(word: str):
        return int(word) if "." not in word else int(float(word))

    return [[parse_num(x) for x in line.split()] for line in lines]


def write_one(args, where=None):
    id, instance = args
    assert (
        len(instance["proc_times"].shape) == 2
    ), "no batch dimension allowed in write operation"
    lines = []

    # The flexibility is the average number of eligible machines per operation.
    num_eligible = (instance["proc_times"] > 0).sum()
    n_ops = (~instance["pad_mask"]).sum()
    num_jobs = instance["next_op"].size(0)
    num_machines = instance["proc_times"].size(0)
    flexibility = round(int(num_eligible) / int(n_ops), 5)

    metadata = f"{num_jobs}\t{num_machines}\t{flexibility}"
    lines.append(metadata)

    for i in range(num_jobs):
        ops_of_job = instance["job_ops_adj"][i].nonzero().squeeze(1)
        job = [len(ops_of_job)]  # number of operations of the job

        for op in ops_of_job:
            eligible_ma = instance["proc_times"][:, op].nonzero().squeeze(1)
            job.append(eligible_ma.size(0))  # num_eligible

            for machine in eligible_ma:
                duration = instance["proc_times"][machine, op]
                assert duration > 0, "something is wrong"
                # add one since in song instances ma indices start from one
                job.extend([int(machine.item()) + 1, int(duration.item())])

        line = " ".join(str(num) for num in job)
        lines.append(line)

    formatted = "\n".join(lines)

    file_name = f"{str(id+1).rjust(4, '0')}_{num_jobs}j_{num_machines}m.txt"
    full_path = os.path.join(where, file_name)

    with open(full_path, "w") as fh:
        fh.write(formatted)

    return formatted


def write(where: Union[Path, str], instances: TensorDict):
    if not os.path.exists(where):
        os.makedirs(where)

    return list(map(partial(write_one, where=where), enumerate(iter(instances))))