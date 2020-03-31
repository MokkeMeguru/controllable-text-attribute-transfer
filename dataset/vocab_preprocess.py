import logging
from pathlib import Path
from typing import List

from .tokenizer.base import Tokenizer


def build_vocab_file(extra_keywords: List[str],
                     vocab_file: Path,
                     source_files: List[Path],
                     reference_files: List[Path],
                     labeled_files: List[Path],
                     tokenizer: Tokenizer):
    """
    Args:
        extra_keywords : e.g. ['<pad>', '<bos>', '<eos>', '<unk>', '<mask>']
        vocab_file     : save path of vocab
        source_files   : source_files - one sentence per one line
        reference_files: reference_files - two sentence per one line splitted by \t
        labled_files   : labeled_files - one sentence \t label per one line
        tokenizer      : tokenizer for the sentence
    Returns:
        None (None)    :
    """
    word_to_id = {}
    for source_file in source_files:
        with source_file.open("r", encoding="utf-8") as f:
            for sentence in f:
                sentence = sentence.strip()
                word_list = tokenizer.tokenize(sentence)
                for word in word_list:
                    if word not in word_to_id:
                        word_to_id[word] = 0
                    word_to_id[word] += 1

    for reference_file in reference_files:
        with reference_files.open("r", encoding="utf-8") as f:
            for instance in f:
                instance = instance.strip()
                sent1, sent2 = instance.split("\t")
                for sent in [sent1, sent2]:
                    word_list = tokenizer.tokenize(sent)
                    for word in word_list:
                        if word not in word_to_id:
                            word_to_id[word] = 0
                        word_to_id[word] += 1

    for labeled_file in labeled_files:
        with labeled_file.open("r", encoding="utf-8") as f:
            for instance in f:
                sent, _ = instance.strip().split("\t")
                word_list = tokenizer.tokenize(sent)
                for word in word_list:
                    if word not in word_to_id:
                        word_to_id[word] += 1
    logging.info("Get word_dict success: {} words".format(len(word_to_id)))
    word_dict_list = sorted(word_to_id.items(),
                            key=lambda d: d[1], reverse=True)

    with vocab_file.open('w', encoding="utf-8") as f:
        for keyword in extra_keywords:
            f.write(keyword + "\n")
        for word, freq in word_dict_list:
            f.write("{}\t{}\n".format(word, freq))
    logging.info("Build dict is completed! saved at {}".format(vocab_file))
