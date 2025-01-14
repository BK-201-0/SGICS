import torch
import torch.nn as nn
from ..utils.constants import INF
import math, copy
import torch.nn.functional as F

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, config, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.d_model = d_model
        # self.linears = self.clones(nn.Linear(d_model, d_model), 4)
        self.linears = self.clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.config = config

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # mask must be four dimension
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        residual = key

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        U = torch.randn(self.d_model, nbatches * query.size(3)).view(nbatches, self.h, -1, self.d_k)
        U = U.to(self.config['device'])
        attn_weight = torch.matmul(query, U)
        attn_weight = torch.matmul(attn_weight, key.transpose(-1, -2))
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask == 0, -INF)
        attn_weight = self.dropout(F.softmax(attn_weight, dim=-1))
        context = torch.matmul(attn_weight.transpose(-1,-2), value)
        context = context.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        context = self.dropout(context)
        output = context + residual
        return output


    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -INF)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

