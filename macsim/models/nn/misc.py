import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from functools import partial
from einops import rearrange
from tensordict import TensorDict
from torch.nn.modules import TransformerEncoderLayer

from macsim.models.policy_args import PolicyParams, TransformerParams, MatNetParams


_float = torch.get_default_dtype()

class MHAWaitOperationEncoder(nn.Module):
    def __init__(self, input_size: int, params: TransformerParams) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.randn(1, 1, input_size), requires_grad=True)
        self.encoder = None


    def forward(self, ops_emb: torch.Tensor, td: TensorDict):
        # (bs, ops)
        dummy = self.dummy.expand(td.size(0), 1, -1)  
        if self.encoder is not None:
            attn_mask = ~rearrange(td["op_scheduled"] + td["pad_mask"], "b j o -> b 1 (j o)")
            attn_mask = torch.logical_or(td["done"][..., None], attn_mask)
            attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
            dummy, _ = self.encoder(
                query=dummy, 
                key=rearrange(ops_emb, "b j o e -> b (j o) e"), 
                value=rearrange(ops_emb, "b j o e -> b (j o) e"), 
                attn_mask=~attn_mask  # True means: not attend
            )
        # (bs, 1, emb)
        return dummy


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = embed_dim
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=_float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).to(_float) * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, hidden: torch.Tensor, seq_pos: torch.Tensor = None, mask = None) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            seq_pos: Tensor, shape ``[batch_size, seq_len]``
        """

        seq_pos_clipped = seq_pos.clip(min=0)
        pes = self.pe.expand(hidden.size(0), -1, -1).gather(
            1, seq_pos_clipped.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        if mask is None:
            mask = seq_pos.lt(0)
        else:
            mask = torch.logical_or(mask, seq_pos.lt(0))
        pes[mask] = 0 
        hidden = hidden + pes
        return self.dropout(hidden)


class PositionalEncodingWithOffset(nn.Module):

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, offsets: torch.Tensor = None, mask = None):
        """
        Positional Encoding with per-head offsets.
        :param x: sequence of embeddings (bs, num_heads, seq_len, d_model)
        :param offsets: per-head sequence offsets (bs, num_heads)
        """
        batch_size, num_heads, length, embed_dim = x.shape
        device = x.device

        if embed_dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(embed_dim)
            )

        # Create a position matrix for each batch instance and head
        position = torch.arange(0, length, device=device, dtype=_float).unsqueeze(0).unsqueeze(0)
        position = position.expand(batch_size, num_heads, length)  # Shape: (bs, num_heads, seq_len)
        
        # Apply offsets and clamp at 0, using offsets of shape (bs, num_heads)
        if offsets is not None:
            position = (position - offsets.unsqueeze(-1)).clamp(min=0)  # Shape: (bs, num_heads, seq_len)

        # Initialize the positional encoding tensor
        pe = torch.zeros(batch_size, num_heads, length, embed_dim, device=device)

        # Compute the div_term only once (shared across batch and heads)
        div_term = torch.exp(
            (
                torch.arange(0, embed_dim, 2, device=device, dtype=_float)
                * -(math.log(10000.0) / embed_dim)
            )
        )

        # Apply positional encoding to even (sin) and odd (cos) indices
        pe[:, :, :, 0::2] = torch.sin(position.unsqueeze(-1) * div_term)
        pe[:, :, :, 1::2] = torch.cos(position.unsqueeze(-1) * div_term)

        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(pe)
            pe[mask] = 0
        # Apply dropout and return the output
        return self.dropout(x + pe)
    

class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        return super().forward(x.float()).type_as(x)


class Normalization(nn.Module):
    def __init__(self, embed_dim: int, normalization: str):
        super().__init__()

        normalizer_class = {
            "batch": partial(nn.BatchNorm1d, affine=True),
            "instance": partial(nn.GroupNorm, num_channels=embed_dim, affine=True),
            "layer": FP32LayerNorm,
            "rms": RMSNorm,
            "none": nn.Identity
        }.get(normalization)

        self.normalizer = normalizer_class(embed_dim)

    def forward(self, x: torch.Tensor):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif isinstance(self.normalizer, nn.GroupNorm):
            batch_size, seq_len, feature_dim = x.shape
            x_reshaped = x.reshape(-1, feature_dim)
            # Apply normalization
            return self.normalizer(x_reshaped).reshape(batch_size, seq_len, feature_dim)
        else:
            return self.normalizer(x)



class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_neurons: List[int] = [64, 32],
        hidden_act: str = "ReLU",
        out_act: str = "Identity",
        input_norm: str = "none",
        output_norm: str = "none",
        bias: bool = True
    ):
        super(MLP, self).__init__()

        activation_mapping = {
            "relu": "ReLU",
            "gelu": "GELU",
            "identity": "Identity"
        }
        hidden_act_str = activation_mapping.get(hidden_act.lower(), "ReLU")
        out_act_str = activation_mapping.get(out_act.lower(), "Identity")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act_str)()
        self.out_act = getattr(nn, out_act_str)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.lins = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            self.lins.append(nn.Linear(in_dim, out_dim, bias=bias))

        self.input_norm = Normalization(input_dim, normalization=input_norm)
        self.output_norm = Normalization(output_dim, normalization=output_norm)

    def forward(self, xs):
        xs = self.input_norm(xs)
        for i, lin in enumerate(self.lins[:-1]):
            xs = lin(xs)
            xs = self.hidden_act(xs)
        xs = self.lins[-1](xs)
        xs = self.out_act(xs)
        xs = self.output_norm(xs)
        return xs

    def _get_act(self, is_last):
        return self.out_act if is_last else self.hidden_act


    
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.to(_float)).type_as(x)
        return output * self.weight
    

class EntityOffsetEmbedding(nn.Module):

    def __init__(self, embed_dim: int, num_entities: int = 100, dropout: float = 0.0):
        super().__init__()
        fixed_embs = torch.distributions.Uniform(low=-1/embed_dim**0.5, high=1/embed_dim**0.5).sample((num_entities, embed_dim))
        self.register_buffer("embeddings", fixed_embs)  # Stores as a buffer (not trainable)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, emb, entity_ids):
        offset_emb = self.embeddings[entity_ids]
        return self.dropout(emb + offset_emb)
    

class AttentionGraphPooling(nn.Module):
    def __init__(self, params: PolicyParams):
        super().__init__()
        self.embed_dim = params.embed_dim
        # Learnable query vector for attention
        self.query = nn.Parameter(torch.randn(1, 1, params.embed_dim))
        
    def forward(self, node_embeddings, node_mask = None):
        """
        Args:
            embeddings: Tensor of shape (B, S, D)
            mask: Tensor of shape (B, S) with !!!True indicating invalid positions!!!
        Returns:
            Pooled tensor of shape (B, 1, D)
        """
        bs = node_embeddings.size(0)
        # (b 1 N)
        # (b 1 d)
        query = self.query.expand(bs, 1, -1)
        # Calculate attention scores using dot product between query and node embeddings
        # (b 1 N)
        attn_scores = torch.bmm(query, node_embeddings.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        if node_mask is not None:
            node_mask = node_mask.unsqueeze(1)
            all_masked = node_mask.all(-1, keepdim=True)
            # Apply mask (set attention scores for invalid nodes to -inf)
            attn_scores = attn_scores.masked_fill(node_mask, float("-inf"))
            # Normalize with softmax
            attn_weights = F.softmax(attn_scores, dim=-1) 
            attn_weights = attn_weights.masked_fill(all_masked.expand_as(attn_weights), 0.0)
        else:
            attn_weights = F.softmax(attn_scores, dim=-1) 
        # Weighted sum of node embeddings: 
        graph_embedding = torch.bmm(attn_weights, node_embeddings)
        # (b 1 d)
        return graph_embedding
    

class SelfAttnBlock(nn.Module):
    def __init__(self, params: TransformerParams):
        super().__init__()
        self.num_heads = params.num_heads
        self.trf_block = TransformerEncoderLayer(
            d_model=params.embed_dim,
            nhead=params.num_heads,
            dim_feedforward=params.feed_forward_hidden,
            dropout=params.dropout,
            activation=params.activation,
            norm_first=params.norm_first,
            bias=params.bias,
            batch_first=True,
        )

    def forward(self, embeddings, mask):
        attn_mask = self._get_attn_mask(mask)
        # transformer layer over final embeddings
        embeddings = self.trf_block(
            src=embeddings, 
            src_mask=attn_mask
        )
        return embeddings
        
    def _get_attn_mask(self, mask: torch.Tensor):
        # get statistics
        bs, n_actions = mask.shape
        # expand self
        attn_mask = (
            mask
            .unsqueeze(1)
            .expand(bs, n_actions, n_actions)
            .contiguous()
        )
        # make all actions attend to at least themselves
        attn_mask = attn_mask.diagonal_scatter(
            torch.full_like(mask, fill_value=False),
            dim1=-2, dim2=-1
        )
        # make head dim
        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)
        return attn_mask
    


class MaskedPooling(nn.Module):

    def __init__(self, params: PolicyParams):
        super().__init__()

        pooling = params.critic_pooling.lower()
        if pooling not in {'mean', 'max', 'attn'}:
            raise ValueError("Pooling must be one of ['mean', 'max', 'attn']")
        self.pooling = pooling
        self.attn_pool = AttentionGraphPooling(params) if pooling == 'attn' else None
        self.sa_block = SelfAttnBlock(params) if params.critic_use_sa else None

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Tensor of shape (B, S, D)
            mask: Tensor of shape (B, S) with !!!True indicating invalid positions!!!
        Returns:
            Pooled tensor of shape (B, D)
        """
        if self.sa_block is not None:
            embeddings = self.sa_block(embeddings, mask)

        if self.pooling == 'mean':
            mask_expanded = mask.unsqueeze(-1)  # (B, S, 1)
            masked = embeddings.masked_fill(mask_expanded, 0.0)
            summed = masked.sum(dim=1)
            lengths = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
            return summed / lengths

        elif self.pooling == 'max':
            mask_expanded = mask.unsqueeze(-1)
            masked = embeddings.masked_fill(mask_expanded, float("-inf"))
            return masked.max(dim=1).values

        elif self.pooling == 'attn':
            return self.attn_pool(embeddings, node_mask=mask)
        

class CommunicationLayer(nn.Module):
    def __init__(self, params: TransformerParams) -> None:
        super().__init__()
        self.comm_layer = TransformerEncoderLayer(
            d_model=params.embed_dim,
            nhead=params.num_heads,
            dim_feedforward=params.feed_forward_hidden,
            dropout=params.dropout,
            activation=params.activation,
            norm_first=params.norm_first,
            batch_first=True,
        )

    def forward(self, x, attn_mask=None):
        h = self.comm_layer(x, src_mask=attn_mask)
        return h
    