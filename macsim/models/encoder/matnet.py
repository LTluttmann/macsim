import torch
from einops import rearrange
from tensordict import TensorDict
from torch import nn
from torch.nn.modules import TransformerEncoderLayer

from macsim.models.nn.misc import Normalization
from macsim.models.policy_args import MatNetParams
from macsim.models.nn.init_embeddings import get_init_emb_layer

from .base import BaseEncoder, MatNetEncoderOutput
from .mixed_attention import EfficientMixedScoreMultiHeadAttentionLayer, MixedScoreMultiHeadAttentionLayer


class MatNetEncoderLayer(nn.Module):

    def __init__(self, params: MatNetParams) -> None:
        super().__init__()

        self.norm_first = params.norm_first
        self.use_block_attn = params.use_block_attn

        self.node_mha = TransformerEncoderLayer(
            d_model=params.embed_dim,
            nhead=params.num_heads,
            dim_feedforward=params.feed_forward_hidden,
            dropout=params.dropout,
            activation=params.activation,
            norm_first=False, # params.norm_first,
            batch_first=True,
        )

        self.agent_mha = TransformerEncoderLayer(
            d_model=params.embed_dim,
            nhead=params.num_heads,
            dim_feedforward=params.feed_forward_hidden,
            dropout=params.dropout,
            activation=params.activation,
            norm_first=False, # self.norm_first,
            batch_first=True,
        )

        if params.param_sharing:
            self.cross_attn = EfficientMixedScoreMultiHeadAttentionLayer(params)
        else:
            self.cross_attn = MixedScoreMultiHeadAttentionLayer(params)

        self.node_norm = Normalization(params.embed_dim, normalization=params.normalization)
        self.agent_norm = Normalization(params.embed_dim, normalization=params.normalization)


    def forward(
        self, 
        node_embs, 
        agent_embs, 
        cost_mat=None, 
        cross_mask=None,
        node_attn_mask=None,
        agent_attn_mask=None,
    ):

        #### CROSS ATTENTION ####
        node_embs_out, agent_embs_out = self.cross_attn(
            node_embs, 
            agent_embs, 
            attn_mask=cross_mask, 
            cost_mat=cost_mat
        )
        
        #### SKIP CONN AND NORM ####
        node_embs_out = self.node_norm(node_embs_out + node_embs)
        agent_embs_out = self.agent_norm(agent_embs_out + agent_embs)

        #### SELF ATTENTION ####
        # (bs, num_nodes, emb)
        node_embs_out = self.node_mha(node_embs_out, src_mask=node_attn_mask)
        # (bs, num_agents, emb)
        agent_embs_out = self.agent_mha(agent_embs_out, src_mask=agent_attn_mask)

        return node_embs_out, agent_embs_out


class MatNetEncoder(BaseEncoder):
    def __init__(self, params: MatNetParams) -> None:
        super().__init__()
        self.norm_first = params.norm_first
        self.embed_dim = params.embed_dim
        self.num_heads = params.num_heads
        self.init_embedding = get_init_emb_layer(params)
        self.encoder = nn.ModuleList([])
        for _ in range(params.num_encoder_layers):
            self.encoder.append(MatNetEncoderLayer(params))

        if self.norm_first:
            self.node_norm = Normalization(params.embed_dim, normalization=params.normalization)
            self.agent_norm = Normalization(params.embed_dim, normalization=params.normalization)


    def forward(self, td: TensorDict) -> MatNetEncoderOutput:
        # (bs, jobs, ops, emb); (bs, ma, emb); (bs, jobs*ops, ma)
        td_emb, td_mask = self.init_embedding(td)

        node_emb = td_emb["nodes"]
        agent_emb = td_emb["agents"]
        
        # add head dimension for attn masks
        node_attn_mask = td_mask["nodes"]
        if node_attn_mask is not None:
            node_attn_mask = node_attn_mask.repeat_interleave(
                self.num_heads, dim=0
            )

        agent_attn_mask = td_mask["agents"]
        if agent_attn_mask is not None:
            agent_attn_mask = agent_attn_mask.repeat_interleave(
                self.num_heads, dim=0
            )

        if self.norm_first:
            node_emb = self.node_norm(node_emb)
            agent_emb = self.agent_norm(agent_emb)

        # run through the layers 
        for layer in self.encoder:
            node_emb, agent_emb = layer(
                node_emb,
                agent_emb,
                cost_mat=td_emb["edges"], 
                cross_mask=td_mask["edges"],
                node_attn_mask=node_attn_mask,
                agent_attn_mask=agent_attn_mask
            )

        return TensorDict(
            {"nodes": node_emb, "agents": agent_emb}, 
            batch_size=td.batch_size
        )
