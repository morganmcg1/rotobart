import sys

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from numpy.random import poisson, permutation
import nltk
from jax import random, ops

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import _collate_batch

nltk.download('punkt')

@dataclass
class DataCollatorForTextInfilling:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    poisson_lambda: float = 3.0
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError

    def __call__(self, examples: List[Union[List[int], jnp.ndarray, Dict[str, jnp.ndarray]]]
                 ) -> Dict[str, jnp.ndarray]:
        batch = {}
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples, (dict, BatchEncoding)):
            examples_ids = examples['input_ids']     
            if 'decoder_input_ids' in examples.keys():
                examples_dec = examples['decoder_input_ids']
            else:
                examples_dec = examples_ids
            

            #bs of one
            if type(examples_ids[0]) is int:
                examples_ids = [examples_ids]
            #bs of one
            if type(examples_dec[0]) is int:
                examples_dec = [examples_dec]

            
            batch["input_ids"] =  _collate_batch(examples_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            batch["decoder_input_ids"] = _collate_batch(examples_dec, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            batch["decoder_input_ids"] = batch["decoder_input_ids"].tolist()


        elif isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="jax", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch["input_ids"] =  _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            batch["decoder_input_ids"] =  _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of).tolist()

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch

    def mask_tokens(self,
                    inputs: jnp.ndarray,
                    special_tokens_mask: Optional[jnp.ndarray] = None
                    ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        inputs_copy = np.array(inputs)
        labels = np.array(inputs)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = jnp.array(special_tokens_mask, dtype=bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        # determine how many tokens we need to mask in total
        is_token = ~(labels == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.astype(float).sum() * self.mlm_probability))
        if num_to_mask == 0:
            return inputs, labels

        # generate a sufficient number of span lengths
        #poisson_distribution = poisson(lam=self.poisson_lambda,size=(num_to_mask,))
        lengths = poisson(lam=self.poisson_lambda,size=(num_to_mask,))
        while np.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = np.concatenate([lengths, poisson(lam=self.poisson_lambda,size=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = np.argmin(np.abs(np.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[:idx + 1]

        # select span start indices
        #print("IS TOKEN")
        #print(is_token)
        #print(sum(list(map(lambda x: 1 if(x) else 0, is_token[0]))))
        token_indices = np.argwhere(is_token==1)
        #print("TOKEN INDICES")
        #print(token_indices)
        span_starts = permutation(token_indices.shape[0])[:lengths.shape[0]]

        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        #print("MASKED INDICES")
        #print(masked_indices)
        mask = np.full_like(labels, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = labels.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask==True)] = False
        inputs_copy[np.where(mask==1)] = self.tokenizer.mask_token_id
        labels[np.where(mask == 0)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1),1, 1)
        new_inputs = np.full_like(labels, fill_value=self.tokenizer.pad_token_id)

        #splits = list(map(lambda x: x.reshape(-1),  np.split(inputs_copy, indices_or_sections=2, axis=0))
        for i, example in enumerate(np.split(inputs_copy, indices_or_sections=new_inputs.shape[0], axis=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0:new_example.shape[0]] = new_example
            
        #batching now fixed
        return new_inputs, labels


#Code below is by Matt Bui
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
        decoder_input_ids = example["input_ids"]

        full_stops = source == self.full_stop_index

        # Tokens that are full stops, where the previous token is not
        sentence_ends = (full_stops[1:] * ~full_stops[:-1]).nonzero()[0] + 2
        result = source.copy()

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

        example["input_ids"] = np.asarray(result).tolist()
        example['decoder_input_ids'] = decoder_input_ids
        return example

