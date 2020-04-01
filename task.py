import logging
import os
import time
from logging import getLogger
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra import utils as hutils
from matplotlib import pyplot as plt
from torch import optim

from omegaconf import DictConfig

from .early_stopping import EarlyStopping
from .model import EncoderDecoder
from .models import labelsmoothing, normopt
from .models import utils as mutils

fmt = "[%(asctime)s] %(levelname)s %(name)s :%(message)s"
logging.basicConfig(level=logging.DEBUG, format=fmt)

logger = getLogger(__name__)


def add_log(message: str):
    logger.info(message)


def add_output(output: str, path: Path):
    with Path.open("a", encoding="utf-8") as f:
        f.write(str(output) + "\n")


class Task:
    def __init__(self, hparams, save_folder: str):
        self.basic_params = hparams["basic"]
        self.utils_params = hparams["utils"]
        self.ae_model = None
        self.load_dataset()
        self.early_stopping = EarlyStopping(
            save_folder,
            self.utils_params["early_stopping"]["patience"],
            self.utils_params["early_stopping"]["delta"]
        )
        if self.utils_params["optimizer"]["ae"] == "Adam":
            ae_optimizer = torch.optim.Adam(
                self.ae_model.parameters(),
                lr=0.0,
                betas=(0.9, 0.98),
                eps=1e-9)
        else:
            raise NotImplementedError()
        self.ae_optimizer = normopt.NormOpt(
            self.basic_params["d_model"],
            self.utils_params["optimizer"]["ae"]["factor"],
            self.utils_params["optimizer"]["ae"]["warmup_step"],
            ae_optimizer)
        self.ae_criterion = mutils.get_cuda(
            labelsmoothing.LabelSmoothing(
                size=self.basic_params["vocab_size"],
                padding_idx=self.pad_idx,
                smoothing=self.utils_params["label_smoothing"]["smoothing"]))
        if not self.load_model():
            for p in self.ae_model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def load_dataset(self):
        self.pad_idx = 0

    def load_model(self):
        return False

    def train_step(self, batch, example: bool = False):
        src = batch.src.transpose(0, 1)
        tgt = batch.tgt.transpose(0, 1)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src_mask = mutils.padding_mask(src, pad_id=self.pad_idx)
        tgt_mask = mutils.look_ahead_mask(tgt, pad_id=self.pad_idx)
        n_tokens = (tgt_output != 0).data.sum().float()
        latent, out = self.ae_model.forward(
            src, tgt_input, src_mask, tgt_mask)

        loss_rec = self.ae_criterion(
            out.contiguous().view(-1, out.size(-1)),
            tgt.contiguous().view(-1)) / n_tokens.data
        if example:
            add_log("example reconstruct")
            add_log(self.vocab.decode(tgt_input[0].numpy().tolist()))
            generated = self.greedy_decode(latent)
            add_log(self.vocab.decode(generated.numpy().tolist()))
        return loss_rec

    def greedy_decode(self, latent):
        batch_size = latent.size(0)
        ys = mutils.get_cuda(
            torch.ones(batch_size, 1).fill_(self.bos_idx).long())
        src_mask = mutils.get_cuda(
            torch.ones(latent.size(0), 1, 1).long())
        for i in range(self.basic_params["max_seq_len"] - 1):
            out = self.ae_model.decode(
                latent,
                mutils.to_var(ys),
                mutils.to_var(src_mask),
                # TODO: something wrong
                mutils.to_var(
                    mutils.square_subsequent_mask(ys.size(1))))
            prob = self.ae_model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
        return ys[:, 1:]

    def train(self):
        for epoch in range(self.basic_params["epochs"]):
            self.ae_model.train()
            for batch in range(self.data_iterator["train"]):
                loss_rec = self.train_step(batch)
                self.ae_optimizer.optimizer.zero_grad()
                loss_rec.backward()
                self.ae_optimizer.step()

            self.ae_model.eval()
            val_rec_losses = []
            for batch in range(self.data_iterator["val"]):
                loss_rec = self.train_step(batch)
                val_rec_losses.append(loss_rec.data.numpy())
            val_rec_loss = np.mean(val_rec_losses).astype("float")
            add_log("epoch {:03d} | rec_loss {:5.4f}".format(
                epoch, val_rec_loss))
            if self.early_stopping(val_rec_loss, self.ae_model):
                break
