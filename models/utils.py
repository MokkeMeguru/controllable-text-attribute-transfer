import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def square_subsequent_mask(size):
    """subsequent mask
    ref. https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    If mask is invalid, PyTorch is invalid
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.bool().masked_fill(mask == 0, False)\
                      .masked_fill(mask == 1, True)
    return mask


def get_cuda(x: torch.Tensor):
    """torch tensor to cuda
    """
    return x.cuda()


def to_var(x: torch.Tensor, volatile: bool = False):
    """torch tensor to torch.autograd.Variable
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def padding_mask(src, pad_id: int = 0):
    src_mask = (src != pad_id).unsqueeze(-2)
    return src_mask


def look_ahead_mask(tgt, pad_id: int = 0):
    tgt_mask = (tgt != pad_id).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        square_subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def clones(module, N: int):
    """Produce N identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def scaled_dot_product_attention(query: torch.Tensor,
                                 key: torch.Tensor,
                                 value: torch.Tensor,
                                 mask: torch.Tensor = None,
                                 dropout: torch.nn.Dropout = None):
    """
    Note:
        this implementation uses Attention Dropout
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores_logits = scores / np.sqrt(d_k)
    if mask is not None:
        scores_logits = scores.masked_fill(mask == 0, -1e-9)
    attention_weights = F.softmax(scores_logits, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights
