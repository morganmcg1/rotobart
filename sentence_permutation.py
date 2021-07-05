import random
from dataclasses import dataclass
from typing import Dict

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer


@dataclass
class SentencePermutationDataCollator:
    sentence_tokenizer: PunktSentenceTokenizer = nltk.data.load(
        "tokenizers/punkt/english.pickle"
    )
    sentence_separator: str = " "

    def __call__(self, example: Dict[str, str]) -> Dict[str, str]:
        sentences = self.sentence_tokenizer.tokenize(example["text"])
        random.shuffle(sentences)
        permutated_text = self.sentence_separator.join(sentences)
        example["permuated_text"] = permutated_text
        return example
