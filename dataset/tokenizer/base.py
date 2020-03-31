from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def tokenize(self, sentence: str) -> List[str]:
        return []
