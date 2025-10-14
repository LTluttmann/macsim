import torch
from tensordict import TensorDict

#################################################
##################### MASKS #####################
#################################################


def operations_block_attn_mask(td: TensorDict):  
    """This function generates a attention mask for operation embeddings, 
    where only operations of the same job attend to each other. This 
    function generates the attention mask for block attention, where the job
    dimension is fused in the batch dimension:

    !!!True means to not attend!!!
    
    Returns: 
    - job_ops_mask (b * n_jobs * n_heads, n_ops, n_ops)
    """      
    bs, nj, no = td["finish_times"].shape
    # attend on ops belonging to same job
    op_scheduled = td["op_scheduled"]
    # initially, all ops in a job attend to each other
    job_ops_mask = torch.full(
        size=(bs, nj, no, no), 
        fill_value=False,
        dtype=torch.bool,
        device=td.device
    )
    # mask only ops that have been scheduled already in attention
    job_ops_mask[op_scheduled.unsqueeze(2).expand_as(job_ops_mask)] = True
    # hack to avoid nans
    job_ops_mask = job_ops_mask.diagonal_scatter(
        torch.full_like(op_scheduled, fill_value=False),
        dim1=2, dim2=3
    )
    # fuse job dimension into batch dimension, to perform memory efficient block attention
    job_ops_mask = job_ops_mask.view(bs * nj, no, no)
    return job_ops_mask


def operations_attn_mask(td: TensorDict, add_dummy_at: int = None) -> torch.Tensor:
    """This function generates a attention mask for operation embeddings, 
    where only operations of the same job attend to each other. Moreover,
    operations that have been scheduled dont attent to other operations.
    
    !!!True means to not attend!!!

    Returns: 
    - job_ops_mask (b * n_heads, n_jobs * n_ops, n_jobs * n_ops)
    """   
    bs, nj, no = td["op_scheduled"].shape
    n_ops_total = nj * no
    # attend on ops belonging to same job
    op_scheduled = td["op_scheduled"].view(bs, n_ops_total)
    job_of_op = (
        torch.arange(nj, device=td.device)
        .repeat_interleave(no)
        .unsqueeze(0).expand_as(op_scheduled)
    )
    same_job = job_of_op[..., None] == job_of_op[:, None]
    # initially, all ops in a job attend to each other
    job_ops_mask = torch.full(
        size=(bs, n_ops_total, n_ops_total), 
        fill_value=False,
        dtype=torch.bool,
        device=td.device
    )
    # mask only ops that have been scheduled already in attention
    job_ops_mask[op_scheduled.unsqueeze(1).expand_as(job_ops_mask)] = True
    job_ops_mask[~same_job] = True

    if add_dummy_at == 0:
        # pad mask for dummy op to the start of the sequence 
        job_ops_mask = torch.nn.functional.pad(job_ops_mask, [1,0,1,0], value=False)
        job_ops_mask[:, 0, 1:] = op_scheduled
        job_ops_mask[:, 1:, 0] = True
    elif add_dummy_at == -1:
        # pad mask for dummy op to the end of the sequence (dummy attends to everything)
        job_ops_mask = torch.nn.functional.pad(job_ops_mask, [0,1,0,1], value=False)
        job_ops_mask[:, -1, :-1] = op_scheduled
        job_ops_mask[:, :-1, -1] = True
    elif add_dummy_at is not None:
        raise ValueError(f"Got invalid value for add_dummy_idx. Expected None, 0 or -1, got {add_dummy_at}")
    
    mask_ndim = n_ops_total + 1 * int(add_dummy_at is not None)
    # hack to avoid nans since no dummy operation is added
    job_ops_mask = job_ops_mask.diagonal_scatter(
        torch.full((bs, mask_ndim), fill_value=False, device=td.device),
        dim1=1, dim2=2
    )

    return job_ops_mask
