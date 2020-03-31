"""English sentence tokenizer using NLTK
"""
import nltk

from .base import Tokenizer


class NLTKTokenizer(Tokenizer):
    """tokenizer using nltk
    Attributes:
        lower: flag to transform word to lower it.
    """

    def __init__(self, lower: bool = True):
        super(NLTKTokenizer, self).__init__()
        self.lower = lower
        self.tokenizer = None

    def tokenize(self, sentence: str):
        """tokenize a sentence
        Args:
            sentence: a sentence
        Returns:
            words List[str]: list of word
        """
        words = nltk.word_tokenize(sentence)
        if self.lower:
            return [word.lower() for word in words]
        return words

    @property
    def lower(self):
        """getter of the lower
        """
        return self.lower
