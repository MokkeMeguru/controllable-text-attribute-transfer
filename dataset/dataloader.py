"""Load dataset example
simple implementation torchtext dataset with custom tokenizer
"""
from pathlib import Path

from torchtext import data

from .custom_field import CustomField
from .vocab import Vocab

if __name__ == '__main__':
    vocab = Vocab(Path("data/yelp/processed_files/word_to_id.txt"),
                  5, unk_token="<UNK>")
    text = CustomField(vocab=vocab, pad_token="<PAD>",
                       eos_token="<EOS>", bos_token="<BOS>", unk_token="<UNK>",
                       fix_length=18)
    label = data.Field(sequential=False, use_vocab=False)
    train = data.TabularDataset(path="data/yelp/sentiment.train",
                                format='tsv', fields=[('text', text),
                                                      ('label', label)])
    for idx, example in enumerate(train):
        if idx > 10:
            break
        print(example.text, ":", example.label)
    train_iter = data.BucketIterator(
        train, batch_size=3, sort_key=lambda x: len(x.txt), shuffle=True)
    for idx, batch in enumerate(train_iter):
        if idx > 2:
            break
        print(batch.text.transpose(0, 1))
        print(batch.text.transpose(0, 1).shape)
        for sent, label in zip(batch.text.transpose(0, 1), batch.label):
            print(' '.join([word for word in vocab.decode(sent.tolist())
                            if word[0] != "<"]),
                  ":", label.numpy())
