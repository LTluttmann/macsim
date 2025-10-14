import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from tensordict import TensorDict
from einops import rearrange, reduce, repeat

from macsim.envs.fjsp.env import FJSPEnv
from macsim.models.nn.masks import operations_attn_mask
from macsim.models.nn.misc import EntityOffsetEmbedding
from macsim.models.encoder.base import MatNetEncoderInput
from macsim.utils.ops import gather_by_index, scale_batch
from macsim.models.policy_args import PolicyParams, MatNetParams
from macsim.models.nn.misc import PositionalEncodingWithOffset, PositionalEncoding


_float = torch.get_default_dtype()


def get_init_emb_layer(params: PolicyParams):
    env_name = params.env.env
    env_emb_mapping = {
        ####### FJSP #############
        "fjsp": FJSPInitEmbedding,
        "ma_fjsp": FJSPInitEmbedding,
        ####### FFSP #############
        "ffsp": FFSPInitEmbeddings,
        "matnet_ffsp": FFSPInitEmbeddings,
        "ma_ffsp": FFSPInitEmbeddings,
        ####### HCVRP ##############
        "hcvrp": SingleDepotHCVRPInitEmbedding,
        "ma_hcvrp": SingleDepotHCVRPInitEmbedding,
    }
    return env_emb_mapping.get(env_name)(params)


class FJSPInitEmbedding(nn.Module):
    def __init__(self, params: MatNetParams):
        super(FJSPInitEmbedding, self).__init__()
        self.time_scale_factor = None
        self.embed_dim = params.embed_dim
        self.use_job_emb = params.env.use_job_emb
        self.use_block_attn = params.use_block_attn
        self.mask_no_edge = params.mask_no_edge
        self.init_ops_embed = nn.Linear(7, self.embed_dim, bias=params.bias)
        self.init_ma_embed = nn.Linear(5, self.embed_dim, bias=params.bias)
        self.op_sequence_encoder = PositionalEncodingWithOffset(dropout=params.input_dropout)
        self.ma_dropout = nn.Dropout(p=params.input_dropout)
        # self.ma_rand_emb = EntityOffsetEmbedding(embed_dim=self.embed_dim, dropout=params.input_dropout, num_entities=params.env.max_machine_id)
        if self.use_job_emb:
            self.job_rand_emb = EntityOffsetEmbedding(embed_dim=self.embed_dim, num_entities=params.env.max_job_id)

    def _op_features(self, td: TensorDict, curr_time: torch.Tensor):
        # (bs, jobs)
        next_op: torch.Tensor = td["next_op"]
        proc_times: torch.Tensor = td["proc_times"]
        num_op, num_ma = proc_times.shape[-2:]
        # (bs, jobs, ops)
        num_eligible_ma = proc_times.gt(0).sum(-1)
        # when computing statistics, we only care about eligible ma-op pairs
        avg_proc_times = torch.where(num_eligible_ma==0, 0.0, proc_times.sum(-1) / num_eligible_ma)
        max_proc_times = proc_times.max(-1).values
        min_proc_times = proc_times.masked_fill(proc_times.eq(0), float("inf")).min(-1).values
        min_proc_times = torch.where(min_proc_times.isinf(), max_proc_times, min_proc_times)
        proc_times_span = max_proc_times - min_proc_times

        # (bs, jobs, ops)
        lb_finish_time = FJSPEnv.calc_lower_bound(td)
        # (bs, jobs, ops)
        lb_start_time = lb_finish_time - min_proc_times
        lb_start_time_shifted = lb_start_time - curr_time[:, None, None]
        lb_start_time_shifted[td["op_scheduled"]] = 0  # mask finished jobs
        is_active_op = F.one_hot(td["next_op"], num_classes=num_op).bool()

        # (bs, jobs, ops)
        op_ready_feat = torch.zeros_like(avg_proc_times)
        # (bs, jobs)
        t_job_ready_shifted = torch.clamp(td["t_job_ready"] - curr_time[:, None], min=0)
        t_job_ready_shifted[td["job_done"]] = 0  # mask finished jobs
        # (bs, jobs, ops)
        t_job_ready_shifted = t_job_ready_shifted.unsqueeze(2).expand_as(op_ready_feat)
        op_ready_feat.scatter_add_(2, next_op.unsqueeze(2), t_job_ready_shifted)

        rem_ops_per_job = torch.sum(~(td["op_scheduled"]), dim=-1, keepdim=True).expand_as(op_ready_feat)
        # stack all features 
        feats = [
            torch.log(1 + num_eligible_ma), # / num_ma,
            torch.log(1 + rem_ops_per_job), # / num_op,
            scale_batch(min_proc_times, scale=self.time_scale_factor),
            scale_batch(avg_proc_times, scale=self.time_scale_factor),
            scale_batch(proc_times_span, scale=self.time_scale_factor),
            # scale_batch(op_ready_feat, scale=self.time_scale_factor),
            # scale_batch(lb_start_time_shifted, scale=self.time_scale_factor),
            torch.log(scale_batch(op_ready_feat, scale=self.time_scale_factor) + 1),
            torch.log(scale_batch(lb_start_time_shifted, scale=self.time_scale_factor) + 1),
            # torch.log(op_ready_feat + 1),
            # torch.log(lb_start_time_shifted + 1),
        ]

        return torch.stack(feats, dim=-1)

    def _init_ops_embed(self, td: TensorDict, curr_time: torch.Tensor):
        bs, n_jobs, n_ops = td["proc_times"].shape[:-1]
        ops_feat = self._op_features(td, curr_time)
        ops_emb = self.init_ops_embed(ops_feat)
        ops_emb = self.op_sequence_encoder(ops_emb, offsets=td["next_op"])
        if self.use_job_emb:
            job_of_op = repeat(td["job_id"], "b j -> b j o", o=n_ops)
            ops_emb = self.job_rand_emb(ops_emb, job_of_op)
        # mask embeddings belonging to irrelevant operations (done or padded)
        ops_emb[td["op_scheduled"]] = 0.0
        ops_emb_flat = ops_emb.view(bs, n_jobs * n_ops, self.embed_dim)
        return ops_emb_flat

    def _init_ma_embed(self, td: TensorDict, curr_time: torch.Tensor):
        num_jobs = td["proc_times"].size(1)

        a_ma = td["t_ma_idle"]
        a_ma_shifted = torch.clamp(a_ma - curr_time[:, None], 0)

        proc_times_flat = rearrange(td["proc_times"], "b j o m -> b (j o) m")
        num_eligible_ops = proc_times_flat.gt(0).sum(1)
        num_remaining_ops = torch.sum(~(td["op_scheduled"]), dim=(1,2))

        next_op_proc_times = gather_by_index(td["proc_times"], td["next_op"], dim=2)
        num_eligible_jobs = next_op_proc_times.gt(0).sum(1)
    
        avg_proc_times = torch.where(num_eligible_jobs==0, 0.0, next_op_proc_times.sum(1) / num_eligible_jobs)
        min_proc_times = next_op_proc_times.masked_fill(next_op_proc_times.eq(0), float("inf")).min(1).values
        min_proc_times[num_eligible_jobs.eq(0)] = 0

        ma_feats = torch.stack([
            torch.log(num_eligible_jobs + 1), # / num_jobs,
            torch.log(scale_batch(a_ma_shifted, scale=self.time_scale_factor) + 1),
            # torch.log(a_ma_shifted + 1),
            # scale_batch(a_ma_shifted, scale=self.time_scale_factor),
            scale_batch(min_proc_times, scale=self.time_scale_factor),
            scale_batch(avg_proc_times, scale=self.time_scale_factor),
            scale_batch(num_eligible_ops, scale=num_remaining_ops),
        ], dim=-1)

        ma_emb = self.init_ma_embed(ma_feats)
        # ma_emb = self.ma_rand_emb(ma_emb, td["ma_ids"])
        ma_emb = self.ma_dropout(ma_emb)
        return ma_emb


    def _edge_emb(self, td: TensorDict, curr_time: torch.Tensor):
        edge_emb = rearrange(td["proc_times"], "b j o m -> b (j o) m")
        edge_emb = scale_batch(edge_emb, self.time_scale_factor)
        return edge_emb.unsqueeze(-1)
    

    def _edge_mask(self, td: TensorDict):
        # scheduled operations do not attend (True means to not attend)
        cross_mask = td["op_scheduled"][..., None].expand_as(td["proc_times"])
        if self.mask_no_edge:
            # optionally, mask not eligible op-ma pairs in attention
            # (bs, num_job, num_ma)
            cross_mask = torch.logical_or(cross_mask, td["proc_times"].eq(0))
        cross_mask = rearrange(cross_mask, "b j o m -> b (j o) m")
        return cross_mask

    def _ops_mask(self, td: TensorDict):
        # (bs, num_job*num_ma, emb)
        if self.use_job_emb:
            op_scheduled = rearrange(td["op_scheduled"], "b j o -> b (j o)")
            op_mask = op_scheduled[:, None].expand(-1, op_scheduled.size(1), op_scheduled.size(1))
            op_mask = op_mask.diagonal_scatter(
                torch.full_like(op_scheduled, fill_value=False),
                dim1=1, dim2=2
            )
        else:
            op_mask = operations_attn_mask(td)
        return op_mask
    

    def forward(self, td: TensorDict) -> Tuple[MatNetEncoderInput, MatNetEncoderInput]:
        self.time_scale_factor = td["scale_factor"]
        job_ma_ready = torch.maximum(td["t_job_ready"][..., None], td["t_ma_idle"][:, None])
        curr_time = reduce(job_ma_ready, "b ... -> b", reduction="min")

        td_emb = TensorDict(
            {  
                # (bs, jobs * ops, emb)
                "nodes": self._init_ops_embed(td, curr_time), 
                # (bs, ma, emb)
                "agents": self._init_ma_embed(td, curr_time),
                # (bs, jobs, ops, ma) 
                "edges": self._edge_emb(td, curr_time)
            }, 
            batch_size=td.batch_size
        )

        td_mask = TensorDict(
            {
                # (bs, jobs * ops, jobs * ops)
                "nodes": self._ops_mask(td), 
                "agents": None, 
                # (bs, jobs, ops, ma) 
                "edges": self._edge_mask(td)
            }, 
            batch_size=td.batch_size
        )

        return td_emb, td_mask


class FFSPInitEmbeddings(nn.Module):
    def __init__(self, params: MatNetParams):
        super().__init__()
        self.embed_dim = params.embed_dim
        self.time_scale_factor = params.env.max_processing_time
        self.mask_no_edge = params.mask_no_edge
        self.init_job_embed = nn.Linear(4, self.embed_dim, bias=params.bias)
        self.init_ma_embed = nn.Linear(4, self.embed_dim, bias=params.bias)
        self.stage_encoder = PositionalEncoding(self.embed_dim, dropout=params.input_dropout)


    def _init_job_embed(self, td: TensorDict, curr_time: torch.Tensor):
        n_stage = td["machine_cnt"].size(1)
        job_progress = td["job_location"] / n_stage  # progress of the job
        # (bs, job, ma)
        ma_in_stage = td["job_location"][..., None] == td["stage_table"][:, None]
        n_ma_in_stage = ma_in_stage.sum(-1)
        # (bs, job, ma)
        avail_ma_proc_times = td["proc_times"].masked_fill(~ma_in_stage, 0.0).to(_float)
        avg_proc_times = torch.where(n_ma_in_stage==0, 0.0, avail_ma_proc_times.sum(-1) / n_ma_in_stage)

        max_proc_times = avail_ma_proc_times.max(-1).values
        min_proc_times = avail_ma_proc_times.masked_fill(avail_ma_proc_times.eq(0), float("inf")).min(-1).values
        min_proc_times = torch.where(min_proc_times.isinf(), max_proc_times, min_proc_times)
        proc_times_span = max_proc_times - min_proc_times

        t_job_ready_shifted = (td["t_job_ready"] - curr_time[:, None])

        job_feats = torch.stack([
            # job_progress,
            # torch.log(1 + n_ma_in_stage),
            scale_batch(t_job_ready_shifted, scale=self.time_scale_factor),
            scale_batch(min_proc_times, scale=self.time_scale_factor),
            scale_batch(avg_proc_times, scale=self.time_scale_factor),
            scale_batch(proc_times_span, scale=self.time_scale_factor),
        ], dim=-1)

        job_emb = self.init_job_embed(job_feats)
        job_emb = self.stage_encoder(job_emb, td["job_location"])
        job_emb[td["job_done"]] = 0
        return job_emb


    def _init_ma_embed(self, td: TensorDict, curr_time: torch.Tensor):
        num_jobs = td["proc_times"].size(1)
        # (bs, ma)
        a_ma = td["t_ma_idle"]
        # (bs, ma)
        a_ma_shifted = torch.clamp(a_ma - curr_time[:, None], 0)
        # (bs, n_ma, n_jobs)
        job_in_stage = td["job_location"][:, None] == td["stage_table"][..., None]
        # (bs, n_ma)
        n_job_in_stage = job_in_stage.sum(-1)
        # (bs, n_ma, n_jobs)
        avail_job_proc_times = td["proc_times"].transpose(1,2).masked_fill(~job_in_stage, 0.0).to(_float)
        num_eligible_jobs = avail_job_proc_times.gt(0).sum(-1)
    
        avg_proc_times = avail_job_proc_times.sum(-1) / (num_eligible_jobs + 1e-6)
        min_proc_times = avail_job_proc_times.masked_fill(avail_job_proc_times.eq(0), float("inf")).min(-1).values
        min_proc_times[num_eligible_jobs.eq(0)] = 0

        ma_feats = torch.stack([
            scale_batch(a_ma_shifted, scale=self.time_scale_factor),
            scale_batch(min_proc_times, scale=self.time_scale_factor),
            scale_batch(avg_proc_times, scale=self.time_scale_factor),
            # torch.log(1 + n_job_in_stage),
            n_job_in_stage / num_jobs,
        ], dim=-1)

        ma_emb = self.init_ma_embed(ma_feats)
        ma_emb = self.stage_encoder(ma_emb, td["stage_table"])
        ma_emb[td["stage_table"] == -1] = 0.0
        return ma_emb

    def _job_mask(self, td: TensorDict) -> torch.Tensor:
        """This function generates an attention mask for job embeddings, 
        where done jobs are excluded from attention. 
        !!!True means to not attend!!!
        """   
        bs, num_jobs = td["job_done"].shape
        mask = td["job_done"].unsqueeze(1).expand(bs, num_jobs, num_jobs).contiguous()
        mask = mask.diagonal_scatter(
            torch.full_like(td["job_done"], fill_value=False),
            dim1=1, dim2=2
        )
        return mask


    def _edge_mask(self, td):
        """This function optionally generates a mask for job-machine attention, 
        where only jobs and machines the same stage attend to each other
        
        !!!True means to not attend!!!
        """   
        # optionally, mask not eligible op-ma pairs in attention
        if self.mask_no_edge:
            # (bs, num_job, num_ma)
            cross_mask = ~(td["job_location"][..., None] == td["stage_table"][:, None])
        else:
            # cross_mask = repeat(td["job_done"], "b j -> b j m", m=td["stage_table"].size(1))
            cross_mask = td["job_done"][..., None] | (td["stage_table"][:, None] == -1)
        return cross_mask

    def _agent_mask(self, td):
        bs, num_agents = td["stage_table"].shape
        # mask padded machines
        mask = (td["stage_table"] == -1).unsqueeze(1).expand(bs, num_agents, num_agents).contiguous()
        return mask
    
    def forward(self, td: TensorDict):
        # self.time_scale_factor = td["scale_factor"]
        # (bs, n_ma, n_jobs)
        job_in_stage = td["job_location"][:, None] == td["stage_table"][..., None]

        # compute the earliest possible start of a job among machines
        # (bs, n_ma, n_jobs)
        earliest_start = torch.maximum(td["t_job_ready"][:, None], td["t_ma_idle"][...,None])
        earliest_start[~job_in_stage] = 99999
        # (bs)
        curr_time = earliest_start.amin((1,2))

        td_emb = TensorDict(
            {  
                # (bs, jobs, emb)
                "nodes": self._init_job_embed(td, curr_time),
                # (bs, ma, emb)
                "agents": self._init_ma_embed(td, curr_time),
                # (bs, jobs, ma)
                "edges": scale_batch(td["proc_times"], self.time_scale_factor)
            }, 
            batch_size=td.batch_size
        )

        td_mask = TensorDict(
            {
                # (bs, jobs, jobs)
                "nodes": self._job_mask(td), 
                # (bs, jobs, ma)
                "agents": self._agent_mask(td), 
                # (bs, jobs, ma) 
                "edges": self._edge_mask(td)
            }, 
            batch_size=td.batch_size
        )

        return td_emb, td_mask


class SingleDepotHCVRPInitEmbedding(nn.Module):
    """Encoder for the initial state of the MTSP environment.
    Encode the initial state of the environment into a fixed-size vector.
    Features:
    - depot: initial position with positional encoding
    - cities: locations
    """

    def __init__(self, params: MatNetParams):
        super(SingleDepotHCVRPInitEmbedding, self).__init__()
        self.use_polar_feats = False
        self.init_embed_depot = nn.Linear(2, params.embed_dim, params.bias)
        self.init_node_embed = nn.Linear(3 + 2 * int(self.use_polar_feats), params.embed_dim, params.bias)
        self.init_agent_embed = nn.Linear(8, params.embed_dim, params.bias) 
        self.node_dropout = nn.Dropout(p=params.input_dropout)
        self.agent_dropout = nn.Dropout(p=params.input_dropout)
        #self.pos_encoder = PositionalEncoding(params.embed_dim, dropout=params.input_dropout)
        self.scale_factor = None

    def _init_node_embed(self, td: TensorDict):
        # (bs, N+1, 2)
        all_locs = td["locs"]
        # (bs, 1, 2)
        depot_locs = all_locs[:, :1]
        # (bs, 1, d)
        depot_embed = self.init_embed_depot(depot_locs)
        # (bs, N, 2)
        clients_locs = all_locs[:, 1:]
        clients_feats = torch.cat(
            [
                clients_locs,
                scale_batch(td["demand"][:, 1:], self.scale_factor).unsqueeze(-1),
            ], 
            dim=-1,
        )
        if self.use_polar_feats:
            # Convert to polar coordinates
            client_locs_centered = clients_locs - depot_locs  # centering
            dist_to_depot = torch.norm(client_locs_centered, p=2, dim=-1, keepdim=True)
            angle_to_depot = torch.atan2(
                client_locs_centered[..., 1:], client_locs_centered[..., :1]
            )
            clients_feats = torch.cat(
                [clients_feats, dist_to_depot, angle_to_depot], dim=-1
            )

        client_emb = self.init_node_embed(clients_feats)
        node_embs = torch.cat((depot_embed, client_emb), dim=1)
        return self.node_dropout(node_embs)
        

    def _init_agent_embed(self, td: TensorDict):
        num_agents = td["current_node"].size(1)
        depot_locs = td["locs"][:, 0]
        agent_locs = gather_by_index(td["locs"], td["current_node"]) 
        rem_cities = (~td["visited"][:, 1:]).sum(-1, keepdims=True) 
        rem_demand = td["demand"].sum(1, keepdims=True)
        progress = rem_cities / td["visited"][:, 1:].size(1)
        dist_from_depot = torch.linalg.norm(agent_locs - depot_locs[:, None], dim=-1, keepdim=True)
        status_feats = torch.cat(
            [
                agent_locs,
                dist_from_depot,
                td["current_length"].unsqueeze(-1),
                td["agents_speed"].unsqueeze(-1),
                scale_batch(td["agents_capacity"], self.scale_factor).unsqueeze(-1),
                scale_batch(td["agents_capacity"] - td["used_capacity"], self.scale_factor).unsqueeze(-1),
                progress.unsqueeze(-1).expand(-1, num_agents, 1),
                #torch.log(1 + scale_batch(rem_demand, self.scale_factor)).unsqueeze(-1).expand(-1, num_agents, 1),
                #torch.log(1 + rem_cities / num_agents).unsqueeze(-1).expand(-1, num_agents, 1),
            ],
            dim=-1
        )
        agent_embeddings = self.init_agent_embed(status_feats)
        # basically we need something to distinguish agents in the very first step since the first move is important
        # agent_ids = torch.argsort(
        #     (td["agents_capacity"] - td["used_capacity"] + 1e-3 * torch.randn_like(td["agents_capacity"])), 
        #     descending=True
        # )
        # agent_ids = torch.arange(num_agents, device=td.device)[None].expand(td.size(0),num_agents)
        # agent_embeddings = self.pos_encoder(agent_embeddings, agent_ids)
        agent_embeddings = self.agent_dropout(agent_embeddings)
        return agent_embeddings
    
    def _node_mask(self, td: TensorDict) -> torch.Tensor:
        """Cities that have been visited are excluded from attention; True means not attent"""
        num_cities = td["visited"].size(1)
        done_nodes = td["visited"].clone()
        # depot node is always available
        done_nodes[:, 0] = False
        # unmask current nodes
        done_nodes = done_nodes.scatter(1, td["current_node"], False)
        mask = done_nodes.unsqueeze(1).expand(-1, num_cities, num_cities).contiguous()
        mask = mask.diagonal_scatter(
            torch.full_like(td["visited"], fill_value=False),
            dim1=1, dim2=2
        )
        return mask
    

    def _edge_mask(self, td: TensorDict):
        """Finished agents and visited cities are excluded from attention; True means not attent"""   
        return ~td["action_mask"][...,:-1]
    
    def forward(self, td: TensorDict):
        self.scale_factor = td["agents_capacity"].max()
        td_emb = TensorDict(
            {  
                # (bs, nodes, emb)
                "nodes": self._init_node_embed(td),
                # (bs, agents, emb)
                "agents": self._init_agent_embed(td),
                "edges": None,
            }, 
            batch_size=td.batch_size
        )

        td_mask = TensorDict(
            {
                # (bs, nodes, nodes)
                "nodes": self._node_mask(td), 
                # (bs, agents, agents)
                "agents": None, 
                # (bs, nodes, agents) 
                "edges": self._edge_mask(td)
            }, 
            batch_size=td.batch_size
        )

        return td_emb, td_mask
