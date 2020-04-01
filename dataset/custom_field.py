import torch
from torchtext import data

from .vocab import Vocab


class CustomField(data.RawField):
    def __init__(self,
                 sequential: bool = True,
                 vocab: Vocab = None,
                 bos_token: str = None,
                 eos_token: str = None,
                 unk_token: str = "<unk>",
                 pad_token: str = "<pad>",
                 fix_length: int = None,
                 preprocessing=None,
                 postprocessing=None,
                 dtype=torch.long,
                 lower: bool = True,
                 tokenize=lambda x: x.split(),
                 pad_first: bool = False,
                 truncate_first: bool = False,
                 batch_first: bool = False,
                 is_target: bool = False,
                 use_vocab: bool = True):
        self.postprocessing = postprocessing
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.vocab = vocab
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.fix_length = fix_length
        self.preprocessing = preprocessing
        self.dtype = dtype
        self.lower = lower
        self.pad_first = pad_first
        self.tokenize = tokenize
        self.is_target = is_target
        self.truncate_first = truncate_first
        self.batch_first = batch_first

    def preprocess(self, sentence: str):
        if self.sequential and isinstance(sentence, str):
            sentence = self.tokenize(sentence)
        if self.preprocessing is not None:
            return self.preprocessing(sentence)
        else:
            return sentence

    def process(self, batch, device=None):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, minibatch):
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        elif self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self.fix_length + (
                self.bos_token, self.eos_token).count(None) - 2
        padded = []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x))
                    + ([] if self.bos_token is None else [self.bos_token])
                    + list(x[-max_len:]
                           if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token]))
            else:
                padded.append(
                    ([] if self.bos_token is None else [self.bos_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token])
                    + [self.pad_token] * max(0, max_len - len(x))
                )
        return padded

    def numericalize(self, arr, device=None):
        if self.use_vocab:
            if self.sequential:
                arr = [self.vocab.encode(sent) for sent in arr]
            else:
                arr = [self.vocab.encode(sent) for sent in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
            else:
                pass
        else:
            raise NotImplementedError()
        var = torch.tensor(arr, dtype=self.dtype, device=device)
        if self.sequential and not self.batch_first:
            var.t_()
        if self.sequential:
            var = var.contiguous()
        return var
