from logging import getLogger
from pathlib import Path
from typing import List

logger = getLogger(__name__)


class Vocab:
    def __init__(self, vocab_path: Path, freq_min: int,
                 unk_token: str):
        """
        Args:
            vocab_path: vocabulary saved path
            freq_min  : min frequency of the acceptable vocab
        """
        self.stoi = {}
        self.itos = []

        with vocab_path.open("r", encoding="utf-8") as f:
            for line in f:
                base = line.strip().split("\t")
                if len(base) > 1:
                    if int(base[1]) < freq_min:
                        break
                word = base[0]
                self.stoi[word] = len(self.stoi)
                self.itos.append(word)
        self.vocab_size = len(self.itos)
        logger.info("load vocabulary: size {}".format(self.vocab_size))
        logger.info("vocab example1: {}".format(
            list(self.stoi.items())[0::100][:10]))
        logger.info("vocab example2: {}".format(self.itos[0::100][:10]))
        self.unk_token = unk_token
        self.unk = self.stoi[self.unk_token]

    def encode(self, sent: List[str]):
        return [self.stoi.get(word, self.unk) for word in sent]

    def encode_word(self, word: str, restrict: bool = False):
        if not restrict:
            return self.stoi.get(word, self.unk)
        else:
            try:
                return self.stoi[word]
            except Exception:
                raise KeyError(
                    "the word is not found in vocab: {}".format(word))

    def decode(self, sent_idxs: List[int]):
        return [self.itos[idx] for idx in sent_idxs]

    def decode_word(self, idx: int):
        try:
            return self.itos[idx]
        except Exception as inst:
            raise inst
