import torch.nn as nn
import torch.nn.functional as F


class PointwiseFeedForward(nn.Module):
    """
    Note:
        WARN: dimension is [B, S, D]
        B is batch size
        S is sequence size
        D is hidden size
        In PyTorch, they wanna use [S, B, D], (TF users use [B, S, D])
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        super(PointwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
