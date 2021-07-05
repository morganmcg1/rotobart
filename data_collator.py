import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


import jax.numpy as jnp
import numpy as np
from numpy.random import poisson, permutation

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import _collate_batch

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
            examples = examples['input_ids']     
            batch["input_ids"] =  _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        elif isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="jax", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch["input_ids"] =  _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

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
            lengths = np.concatenate([lengths, poisson(rate=self.poisson_lambda,size=(num_to_mask,))])

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

        return jnp.array(new_inputs), jnp.array(labels)
