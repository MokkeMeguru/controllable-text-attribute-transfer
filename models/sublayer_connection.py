import torch.nn as nn

from .layernorm import LayerNorm


class SubLayerConnection(nn.Module):
    """A residual connection followed by a layer norm
    """

    def __init__(self, size, dropout_rate):
        super(LayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer_out):
        out = self.dropout(sublayer_out)
        out = self.norm(out + x)
        return out
