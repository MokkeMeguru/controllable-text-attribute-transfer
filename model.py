import torch
import torch.nn as nn

from .models import (compress, decoder, embedding, encoder, generator,
                     multi_head_attention, pointwise_ff, positional_encoding,
                     utils)


class EncoderDecoder(nn.Module):
    def __init__(self, hparams):
        super(EncoderDecoder, self).__init__()
        basic_params = hparams["basic"]
        pos_encoding_params = hparams["positional_encoding"]
        encoder_params = hparams["encoder"]
        decoder_params = hparams["decoder"]
        compress_params = hparams["compress"]
        embedding_params = hparams["embedding"]

        def gen_slf_attn(num_head):
            return multi_head_attention.MultiHeadAttention(
                num_head, basic_params["d_model"])

        def gen_ffn(d_ff, dropout_rate):
            return pointwise_ff.PointwiseFeedForward(
                basic_params["d_model"], d_ff, dropout_rate)

        self.positional_encoding = positional_encoding.PositionalEncoding(
            basic_params["d_model"], basic_params["max_seq_len"],
            pos_encoding_params["dropout_rate"])
        encoder_layer = encoder.EncoderLayer(
            basic_params["d_model"],
            gen_slf_attn(encoder_params["num_head"]),
            gen_ffn(encoder_params["d_ff"], encoder_params["dropout_rate"]),
            encoder_params["dropout_rate"])
        self.encoder = encoder.Encoder(
            encoder_layer, encoder_params["num_layer"])
        decoder_layer = decoder.DecoderLayer(
            basic_params["d_model"],
            gen_slf_attn(decoder_params["num_head"]),
            gen_slf_attn(decoder_params["num_head"]),
            gen_ffn(decoder_params["d_ff"], decoder_params["dropout_rate"]),
            decoder_params["dropout_rate"])
        self.decoder = decoder.Decoder(
            decoder_layer,
            decoder_params["num_layer"])
        self.compressor = compress.Compress(
            basic_params["max_seq_len"], basic_params["d_model"],
            compress_params["type"])
        self.src_embed = nn.Sequential(
            embedding.Embedding(
                basic_params["d_model"], basic_params["d_model"]),
            positional_encoding.PositionalEncoding(
                basic_params["d_model"], basic_params["max_seq_len"],
                embedding_params["dropout_rate"]))
        self.tgt_embed = nn.Sequential(
            embedding.Embedding(
                basic_params["d_model"], basic_params["d_model"]),
            positional_encoding.PositionalEncoding(
                basic_params["d_model"], basic_params["max_seq_len"],
                embedding_params["dropout_rate"]))

        self.generator = generator.Generator(
            basic_params["d_model"], basic_params["vocab_size"])
        self.d_model = basic_params["d_model"]
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor):
        """
        Args:
            src: [B, S_s]
            tgt: [B, S_t]
            src_mask: [B, S_s]
            tgt_mask: [B, S_t]
        """
        latent = self.encoder(self.src_embed(src), src_mask)
        latent = self.sigmoid(latent)
        compressed_latent = self.compressor(latent)
        compressed_src_mask = utils.get_cuda(
            torch.ones(compressed_latent.size(0), 1, 1).long())
        logit = self.decoder(
            self.tgt_embed(tgt),
            compressed_latent, compressed_src_mask, tgt_mask)
        prob = self.generator(logit)
        return compressed_latent, prob
