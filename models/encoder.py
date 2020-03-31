import torch.nn as nn

from .layernorm import LayerNorm
from .sublayer_connection import SubLayerConnection
from .utils import clones


class EncoderLayer(nn.Module):
    """Encoder's one layer
    """

    def __init__(self, size: int,
                 slf_attn: nn.Module,
                 ffn: nn.Module,
                 dropout_rate: float):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.slf_attn = slf_attn
        self.ffn = ffn
        self.connection1 = SubLayerConnection(size, dropout_rate)
        self.connection2 = SubLayerConnection(size, dropout_rate)

    def forward(self, x, mask):
        x = self.connection1(x, self.slf_attn(x, x, x, mask=mask))
        x = self.connection2(x, self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn
        """
        for layer in self.layers:
            return self.norm(x)
