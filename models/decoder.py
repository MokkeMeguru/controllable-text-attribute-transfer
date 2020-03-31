import torch.nn as nn

from .layernorm import LayerNorm
from .sublayer_connection import SubLayerConnection
from .utils import clones


class DecoderLayer(nn.Module):
    def __init__(self, size: int,
                 slf_attn: nn.Module,
                 src_attn: nn.Module,
                 ffn: nn.Module,
                 dropout_rate: float):
        self.size = size
        self.slf_attn = slf_attn
        self.src_attn = src_attn
        self.ffn = ffn
        self.connection1 = SubLayerConnection(size, dropout_rate)
        self.connection2 = SubLayerConnection(size, dropout_rate)
        self.connection3 = SubLayerConnection(size, dropout_rate)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.connection1(x, self.slf_attn(x, x, x, tgt_mask))
        x = self.connection2(x, self.slf_attn(x, memory, memory, src_mask))
        return self.connection3(x, self.ffn(x))


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
