import torch
import torch.nn as nn

from .utils import clones, scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    Note:
        WARN: dimension is [B, S, D]
        B is batch size
        S is sequence size
        D is hidden size
        In PyTorch, they wanna use [S, B, D], (TF users use [B, S, D])
    """

    def __init__(self, num_head: int,
                 d_model: int,
                 dropout_rate: float = 0.1,
                 num_linear: int = 4):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_head == 0,\
            "multihead attention accept d_model({}) % num_head({}) == 0"\
            .format(d_model, num_head)
        self.d_model = d_model
        self.d_k = d_model // num_head
        self.num_head = num_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.attn = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None):
        if mask is not None:
            mask = mask.unsqueeze
        B = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [linear(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
             for linear, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # this drop out is Attention Dropout for PyTorch
        x, self.attn = scaled_dot_product_attention(query, key, value, mask,
                                                    dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(B, -1, self.d_model)

        return self.linears[-1](x)
