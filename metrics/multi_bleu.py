"""moses multi bleu score in python
"""
import logging
from pathlib import Path

import hydra
from hydra import utils
from omegaconf import DictConfig

from bleu import corpus_bleu


def file_exist(path: Path):
    """check file exists

    Args:
        path (Path): file path instance

    Returns:
        bool: if file is exist, return True
    """
    if path.exists() and path.is_file():
        return True
    else:
        return False

def calc_bleu(bleu_cfg: DictConfig):
    """calculate Bleu Score
    Args:
        bleu_cfg (DictConfig): hyp file, ref files, etc...
                              ref. ./config.yml
    Returns:
        bleu (List): [average bleu, 1-gram bleu, 2-gram bleu, ...]
    """
    data = []
    paths = [Path(utils.get_original_cwd()) /  Path(fname)
             for fname in [bleu_cfg.hyp] + list(bleu_cfg.refs)]
    n = None
    for path in paths:
        if not file_exist(path):
            raise Exception("File Not Found: {}".format(path))
        with path.open('r', encoding='utf-8') as f:
            data.append(f.readlines())
        if n is None:
            n = len(data[-1])
        elif n != len(data[-1]):
            raise Exception("Not Parallel: {} {}-{}".format(path, n, len(data[-1])))

    hyp_data = data[0]
    ref_data = list(map(list, zip(*data[1:])))

    result = corpus_bleu(hyp_data, ref_data, max_n=bleu_cfg.max_n)

    bleu = result[0]
    addition = result[1:]

    result_bleu = "{:.2f} ".format(bleu[0] * 100) + \
        "/".join(map(lambda score: "{:.1f}".format(score * 100.0), bleu[1:]))
    result_others = "(BP = {:.3f}, ratio = {:.3f}, hyp_len = {}, ref_len = {})".format(
        addition[0], addition[1], addition[2], addition[3])

    print(result_bleu + " " + result_others)
    logging.info("BLEU : {}".format(result_bleu))
    logging.info("ADDITION : {}".format(result_others))
    return bleu


@hydra.main(config_path="../conf/config.yaml")
def main(cfg: DictConfig = None):
    bleu_cfg = cfg.bleu
    return calc_bleu(bleu_cfg)

if __name__ == "__main__":
    main()
