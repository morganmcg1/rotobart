import math
from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
from jax import random, ops

import nltk
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class SentenceTokenize:
    """Tokenize the document into sentences and add bos and eos tokens"""

    sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    eos: str = "<s>"
    bos: str = "</s>"

    def __call__(self, example: Dict[str, str]) -> Dict[str, str]:
        sentences = self.sentence_tokenizer.tokenize(example["text"])
        example["text"] = "".join([self.eos + sentence + self.bos for sentence in sentences])
        return example


@dataclass
class DataCollatorForSentencePermutation:
    tokenizer: PreTrainedTokenizerBase
    permutate_sentence_ratio: float = 1.0
    random_key = random.PRNGKey(0)

    def __post_init__(self):
        self.full_stop_index = self.tokenizer.eos_token_id

    def __call__(self, example):
        source = jnp.array(example["input_ids"])
        full_stops = source == self.full_stop_index

        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero()[0] + 2
        result = source.clone()

        num_sentences = jnp.size(sentence_ends, 0)
        num_to_permute = math.ceil((num_sentences * 2 * self.permutate_sentence_ratio) / 2.0)
        substitutions = random.permutation(self.random_key, num_sentences)[:num_to_permute]
        ordering = jnp.arange(0, num_sentences)
        ordering = ops.index_update(
            ordering, substitutions, substitutions[random.permutation(self.random_key, num_to_permute)]
        )

        index = 0
        for i in ordering:
            sentence = source[(sentence_ends[i - 1] if i > 0 else 0) : sentence_ends[i]]
            result = ops.index_update(result, ops.index[index : index + jnp.size(sentence, 0)], sentence)
            index += jnp.size(sentence, 0)

        example["input_ids"] = result

        return example
