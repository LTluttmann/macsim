import math

import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from tensordict import TensorDict

from macsim.models.nn.kvl import get_kvl_emb
from macsim.models.nn.context import get_context_emb
from macsim.models.encoder.base import MatNetEncoderOutput
from macsim.envs.base import MultiAgentEnvironment, Environment
from macsim.models.policy_args import SingleAgentAttnPolicyParams, MacsimParams


class AttentionPointerMechanism(nn.Module):
    """Calculate logits given query, key and value and logit key.

    Note:
        With Flash Attention, masking is not supported

    Performs the following:
        1. Apply cross attention to get the heads
        2. Project heads to get glimpse
        3. Compute attention score between glimpse and logit key

    Args:
        embed_dim: total dimension of the model
        num_heads: number of heads
        mask_inner: whether to mask inner attention
        linear_bias: whether to use bias in linear projection
        sdp_fn: scaled dot product attention function (SDPA)
        check_nan: whether to check for NaNs in logits
    """

    def __init__(self, params: MacsimParams, check_nan=True):
        super(AttentionPointerMechanism, self).__init__()
        self.num_heads = params.num_heads
        # Projection - query, key, value already include projections
        self.project_out = nn.Linear(params.embed_dim, params.embed_dim, bias=params.bias)
        self.use_rezero = params.use_rezero
        if self.use_rezero:
            self.resweight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.dropout = nn.Dropout(params.dropout)
        self.check_nan = check_nan

    def forward(self, query, key, value, logit_key, attn_mask=None):
        """Compute attention logits given query, key, value, logit key and attention mask.

        Args:
            query: query tensor of shape [B, ..., L, E]
            key: key tensor of shape [B, ..., S, E]
            value: value tensor of shape [B, ..., S, E]
            logit_key: logit key tensor of shape [B, ..., S, E]
            attn_mask: attention mask tensor of shape [B, ..., S]. Note that `True` means that the value _should_ take part in attention
                as described in the [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
        """
        # Compute inner multi-head attention with no projections.
        # (bs, q_dim, d_model)
        heads = self._inner_mha(query, key, value, attn_mask)
        # (bs, q_dim, d_model)
        glimpse = self.dropout(self.project_out(heads))
        if self.use_rezero:
            glimpse = query + self.resweight * glimpse
        # Batch matrix multiplication to compute logits (bs, q_dim, graph_size)
        logits = torch.bmm(glimpse, logit_key.transpose(-2, -1)) / math.sqrt(glimpse.size(-1))

        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"

        return logits

    def _inner_mha(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        attn_mask: torch.Tensor | None
    ):
        q = self._make_heads(query)
        k = self._make_heads(key)
        v = self._make_heads(value)
        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(1)
                if attn_mask.ndim == 3
                else attn_mask.unsqueeze(1).unsqueeze(2)
            )
        heads = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return rearrange(heads, "... h n g -> ... n (h g)", h=self.num_heads)

    def _make_heads(self, v):
        return rearrange(v, "... g (h s) -> ... h g s", h=self.num_heads)



class AttentionPointer(nn.Module):
    def __init__(
        self,
        params: SingleAgentAttnPolicyParams,
        check_nan: bool = True,
    ):
        super(AttentionPointer, self).__init__()
        self.kvl_emb = get_kvl_emb(params)
        self.context_embedding = get_context_emb(params)
        self.pointer = AttentionPointerMechanism(params, check_nan)
        self.use_decoder_attn_mask = params.use_decoder_attn_mask

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_cache(self, embs: MatNetEncoderOutput, td: TensorDict) -> None:
        # shape: 3 * (bs, n, emb_dim)
        self.kvl_emb.compute_cache(embs, td)

    def forward(self, embs: MatNetEncoderOutput, td: TensorDict, env: Environment):
        # (bs, a, emb) | (bs, a)
        q, q_mask = self.context_embedding(embs, td)
        # (bs, heads, nodes, key_dim) | (bs, heads, nodes, key_dim)  |  (bs, nodes, emb_dim) | (bs, nodes)
        k, v, logit_key, k_mask = self.kvl_emb(embs, td)
        
        if self.use_decoder_attn_mask:
            # # (bs, a, nodes); in F.scaled_dot_product_attention: True -> attend
            attn_mask = ~(q_mask[..., None] | k_mask[:, None])
        else: 
            attn_mask = None

        # (b, a, nodes)
        logits = self.pointer(q, k, v, logit_key, attn_mask=attn_mask)
        return logits


class MultiAgentAttentionPointer(AttentionPointer):

    def __init__(self, params: MacsimParams):
        super(MultiAgentAttentionPointer, self).__init__(params, check_nan=False)
        

    def forward(self, embs: MatNetEncoderOutput, td: TensorDict, env: MultiAgentEnvironment):

        # (bs, a, emb)
        q, q_mask = self.context_embedding(embs, td)
        # (bs, heads, nodes, key_dim) | (bs, heads, nodes, key_dim)  |  (bs, nodes, emb_dim)
        k, v, logit_key, k_mask = self.kvl_emb(embs, td)
        if self.use_decoder_attn_mask:
            # in F.scaled_dot_product_attention: True -> attend
            attn_mask = ~(q_mask[..., None] | k_mask[:, None])
            attn_mask[..., env.wait_op_idx] = True
        else: 
            attn_mask = None
        # (b, a, nodes)
        logits = self.pointer(q, k, v, logit_key, attn_mask=attn_mask) 
        return logits

