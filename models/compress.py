from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = getLogger(__name__)


class AttentionScore(nn.Module):
    """Calculate Attention Score
    Note:
    ref.
    https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py#L54-L72
    if this layer is wrong, please open the issue on IBM's repo.
    """

    def __init__(self):
        super(AttentionScore, self).__init__()

    def forward(self,
                query: torch.Tensor,
                context: torch.Tensor,
                mask: torch.Tensor):
        batch_size, _, hidden_size = query.size()
        input_size = context.size(1)
        scores = torch.bmm(query, context.transpose(1, 2).contiguous())
        if mask is not None:
            mask = mask.eq(0).expand_as(scores)
            scores.data.masked_fill(mask, -float("inf"))
        probed_score = F.softmax(scores.view(-1, input_size), dim=-1)\
                        .view(batch_size, -1, input_size)
        return probed_score


class GRUCompress(nn.Module):
    def __init__(self, input_size, num_gru: int = 1,
                 correct_att: bool = True):
        super(GRUCompress, self).__init__()
        self.attn = AttentionScore()
        self.correct_att = correct_att
        self.input_size = input_size
        if self.correct_att:
            self.gru = nn.GRU(input_size, input_size,
                              num_gru, batch_first=True)
            self.linear_out = nn.Linear(input_size * 2, input_size)
        else:
            self.gru = nn.GRU(input_size, input_size,
                              num_gru, batch_first=True)

    def forward(self, x, src_mask):
        probed_score = self.attn(x, x, src_mask)

        # x [B, S, D] -> [B, S, D]
        if self.correct_att:
            # IBM's implementation
            batch_size = x.size(0)
            mix = torch.bmm(probed_score, x)
            combined = torch.cat((mix, x), dim=-1)
            x = torch.tanh(
                self.linear_out(combined).contiguous().view(
                    -1, 2 * self.input_size)).view(
                        batch_size, -1, self.input_size)

            x, _ = self.gru(x)
            x = torch.sigmoid(x)
            x = torch.sum(x, dim=1, keepdim=True)
        else:
            # controllable-text-attribute-transfer's implement
            x = torch.bmm(probed_score, x)
            x, _ = self.gru(x)
            x = torch.sigmoid(x)
            x = torch.sum(x, dim=1, keepdim=True)
        return x


class Compress(nn.Module):
    def __init__(self, max_seq_len, d_model, compress_type: str, num_gru: int = 1):
        super(Compress, self).__init__()
        if compress_type == "linear":
            self.compress = nn.Linear(max_seq_len, 1)
        elif compress_type == "sum":
            self.compress = lambda x: torch.sum(x, dim=-1, keepdim=True)
        elif compress_type == "gru":
            self.compress = GRUCompress(
                d_model,
                num_gru=num_gru)
        else:
            raise NotImplementedError()
        self.compress_type = compress_type

    def forward(self, x: torch.Tensor, src_mask):
        if self.compress_type == "gru":
            # x [B, S, D] -> [B, 1, D]
            x = self.compress(x, src_mask)
        else:
            # x [B, S, D] -> [B, D, S]
            x = x.contiguous().transpose(1, 2)
            x = self.compress(x)
            # x [B, D, 1] -> [B, 1, D]
            x = x.transpose(2, 1)
        return x


def test_compress():
    import logging
    logger.setLevel(logging.INFO)

    batch_size = 16
    max_seq_len = 32
    d_model = 128
    correct_output_dims = [batch_size, 1, d_model]
    x = torch.randn(batch_size, max_seq_len, d_model)
    # test 1
    logger.info("test_1: compress type linear")
    compress = Compress(max_seq_len, d_model,  "linear")
    y = compress(x, src_mask=None)
    assert list(y.size()) == correct_output_dims,\
        "shape check error get {} vs correct {}".format(
            list(y.size()), correct_output_dims)
    logger.info("test_1: pass")

    # test 2
    logger.info("test_2: compress type sum")
    compress = Compress(max_seq_len, d_model, "sum")
    y = compress(x, src_mask=None)
    assert list(y.size()) == correct_output_dims,\
        "shape check error get {} vs correct {}".format(
            list(y.size()), correct_output_dims)
    logger.info("test_2: pass")

    # test 3
    logger.info("test_3: compress type gru")
    compress = Compress(max_seq_len, d_model, "gru")
    y = compress(x, src_mask=None)
    assert list(y.size()) == correct_output_dims,\
        "shape check error get {} vs correct {}".format(
            list(y.size()), correct_output_dims)
    logger.info("test_3: pass")


if __name__ == '__main__':
    import logging
    logger.setLevel(logging.INFO)
    test_compress()
