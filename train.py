import logging
import os
import time
from logging import getLogger
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra import utils
from matplotlib import pyplot as plt
from torch import optim

from omegaconf import DictConfig

from .model import EncoderDecoder

fmt = "[%(asctime)s] %(levelname)s %(name)s :%(message)s"
logging.basicConfig(level=logging.DEBUG, format=fmt)

logger = getLogger(__name__)


def add_log(message: str):
    logger.info(message)


def add_output(output: str, path: Path):
    with Path.open("a", encoding="utf-8") as f:
        f.write(str(output) + "\n")
