import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import nltk
import numpy as np
from numpy.random import permutation, poisson
from transformers.data.data_collator import _collate_batch
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

nltk.download("punkt")


@dataclass
class DataCollatorForTextInfilling:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    poisson_lambda: float = 3.0
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = {}
        if isinstance(examples, (dict, BatchEncoding)):
            examples_ids = examples["input_ids"]
            if "decoder_input_ids" in examples.keys():
                examples_dec = examples["decoder_input_ids"]
            else:
                examples_dec = examples_ids

            # bs of one
            if type(examples_ids[0]) is int:
                examples_ids = [examples_ids]
            # bs of one
            if type(examples_dec[0]) is int:
                examples_dec = [examples_dec]

            batch["input_ids"] = _collate_batch(
                examples_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
            )
            batch["decoder_input_ids"] = _collate_batch(
                examples_dec, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
            )
            batch["decoder_input_ids"] = batch["decoder_input_ids"].tolist()

        elif isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="jax", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch["input_ids"] = _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            batch["decoder_input_ids"] = _collate_batch(
                examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
            ).tolist()

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        batch["input_ids"], batch["labels"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )

        return batch

    def mask_tokens(self, inputs):
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
        lengths = poisson(lam=self.poisson_lambda, size=(num_to_mask,))
        while np.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = np.concatenate([lengths, poisson(lam=self.poisson_lambda, size=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = np.argmin(np.abs(np.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[: idx + 1]

        # select span start indices
        # print("IS TOKEN")
        # print(is_token)
        # print(sum(list(map(lambda x: 1 if(x) else 0, is_token[0]))))
        token_indices = np.argwhere(is_token == 1)
        # print("TOKEN INDICES")
        # print(token_indices)
        span_starts = permutation(token_indices.shape[0])[: lengths.shape[0]]

        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        # print("MASKED INDICES")
        # print(masked_indices)
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
        mask[np.where(special_tokens_mask == True)] = False
        inputs_copy[np.where(mask == 1)] = self.tokenizer.mask_token_id
        labels[np.where(mask == 0)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_inputs = np.full_like(labels, fill_value=self.tokenizer.pad_token_id)

        # splits = list(map(lambda x: x.reshape(-1),  np.split(inputs_copy, indices_or_sections=2, axis=0))
        for i, example in enumerate(np.split(inputs_copy, indices_or_sections=new_inputs.shape[0], axis=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0 : new_example.shape[0]] = new_example

        # batching now fixed
        return new_inputs.tolist(), labels.tolist()


# Code below is by Matt Bui
@dataclass
class SentenceTokenize:
    """Tokenize documents into sentences, add bos and eos tokens and split sentences into smaller chunks if too long."""

    sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    bos: str = "<s>"
    eos: str = "</s>"
    max_sentences = 256
    sentence_stride = 128
    max_characters = 100000

    def __call__(self, examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        is_batched = isinstance(examples["text"], list)
        if not is_batched:
            # raise ValueError("required batched=True in map() method")
            examples["text"] = [examples["text"]]

        texts = []
        print(f"len(examples['text'] : {len(examples["text"])}")
        for doc in examples["text"]:
            sentences = self.sentence_tokenizer.tokenize(doc)
            start_index = 0
            print(f"doc len: {len(doc)}")
            print(f"sent_tok len: {len(sentences)}")
            while start_index < len(sentences):
                sentence_span = sentences[start_index : min(len(sentences), start_index + self.max_sentences)]
                text = f"{self.eos}{self.bos}".join([sentence for sentence in sentence_span])

                # trim text by max characters
                if len(text) > self.max_characters:
                    text = text[: self.max_characters]
                texts.append(text)
                start_index += self.sentence_stride
        print(len(texts))
        print()      
        return {"text": texts}


@dataclass
class DataCollatorForSentencePermutation:
    tokenizer: PreTrainedTokenizerBase
    permutate_sentence_ratio: float = 1.0

    def __post_init__(self):
        self.full_stop_index = self.tokenizer.eos_token_id

    def __call__(self, example: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        source = example["input_ids"]

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

        example["decoder_input_ids"] = example["input_ids"]
        example["input_ids"] = result

        return example


@dataclass
class DataCollatorForDenoisingTasks:
    """Data collator used denoising language modeling task in BART.
    The implementation is based on
    https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/data/denoising_dataset.py.
    The default paramters is based on BART paper https://arxiv.org/abs/1910.13461.
    """

    tokenizer: PreTrainedTokenizerBase
    mask_ratio: float = 0.3
    poisson_lambda: float = 3.0
    permutate_sentence_ratio: float = 1.0
    pad_to_multiple_of: int = 16

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, np.ndarray]:
        """Batching, adding whole word mask and permutate sentences
        Args:
            examples (dict): list of examples each examples contains input_ids field
        """
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(examples, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="np")
        batch["decoder_input_ids"] = self.shift_tokens_right(batch["input_ids"])

        do_permutate = False
        if self.permutate_sentence_ratio > 0.0:
            batch["input_ids"] = self.permutate_sentences(batch["input_ids"])
            do_permutate = True

        if self.mask_ratio:
            batch["input_ids"], batch["labels"] = self.add_whole_word_mask(batch["input_ids"], do_permutate)

        return batch

    def shift_tokens_right(self, inputs):
        """Shift decoder input ids right: https://github.com/huggingface/transformers/issues/7961.
        Examples:
            <s>My dog is cute.</s><s>It loves to play in the park.</s><pad><pad>
            shift to -> </s><s>My dog is cute.</s><s>It loves to play in the park.<pad><pad>
        """

        shifted_inputs = np.roll(inputs, 1, axis=-1)

        # replace first token with eos token
        shifted_inputs[:, 0] = self.tokenizer.eos_token_id

        # when there's padding, the last eos tokens will not be rotate to first positon
        # we'll need to replace it with a padding token

        # replace eos tokens at the end of sequences with pad tokens
        end_with_eos = np.where(shifted_inputs[:, -1] == self.tokenizer.eos_token_id)
        shifted_inputs[end_with_eos, -1] = self.tokenizer.pad_token_id

        # find positions where where's the token is eos and its follwing token is a padding token
        last_eos_indices = np.where(
            (shifted_inputs[:, :-1] == self.tokenizer.eos_token_id)
            * (shifted_inputs[:, 1:] == self.tokenizer.pad_token_id)
        )

        # replace eos tokens with pad token
        shifted_inputs[last_eos_indices] = self.tokenizer.pad_token_id
        return shifted_inputs

    def permutate_sentences(self, inputs):
        results = inputs.copy()

        full_stops = inputs == self.tokenizer.eos_token_id

        sentence_ends = np.argwhere(full_stops[:, 1:] * ~full_stops[:, :-1])
        sentence_ends[:, 1] += 2
        num_sentences = np.unique(sentence_ends[:, 0], return_counts=True)[1]
        num_to_permute = np.ceil((num_sentences * 2 * self.permutate_sentence_ratio) / 2.0).astype(int)

        sentence_ends = np.split(sentence_ends[:, 1], np.unique(sentence_ends[:, 0], return_index=True)[1][1:])

        for i in range(inputs.shape[0]):
            substitutions = np.random.permutation(num_sentences[i])[: num_to_permute[i]]

            ordering = np.arange(0, num_sentences[i])
            ordering[substitutions] = substitutions[np.random.permutation(num_to_permute[i])]

            index = 0
            for j in ordering:
                sentence = inputs[i, (sentence_ends[i][j - 1] if j > 0 else 0) : sentence_ends[i][j]]
                results[i, index : index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def add_whole_word_mask(self, inputs, do_permutate):
        labels = inputs.copy()

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = np.array(special_tokens_mask, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token = ~(labels == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.astype(float).sum() * self.mask_ratio))
        if num_to_mask == 0:
            return inputs, labels

        # generate a sufficient number of span lengths
        lengths = poisson(lam=self.poisson_lambda, size=(num_to_mask,))
        while np.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = np.concatenate([lengths, poisson(lam=self.poisson_lambda, size=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = np.argmin(np.abs(np.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[: idx + 1]

        # select span start indices
        # print("IS TOKEN")
        # print(is_token)
        # print(sum(list(map(lambda x: 1 if(x) else 0, is_token[0]))))
        token_indices = np.argwhere(is_token == 1)
        # print("TOKEN INDICES")
        # print(token_indices)
        span_starts = permutation(token_indices.shape[0])[: lengths.shape[0]]

        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        # print("MASKED INDICES")
        # print(masked_indices)
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
        mask[np.where(special_tokens_mask)] = False
        inputs[np.where(mask)] = self.tokenizer.mask_token_id

        if not do_permutate:
            labels[np.where(mask)] = -100
        else:
            labels[np.where(special_tokens_mask)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_inputs = np.full_like(labels, fill_value=self.tokenizer.pad_token_id)

        # splits = list(map(lambda x: x.reshape(-1),  np.split(inputs_copy, indices_or_sections=2, axis=0))
        for i, example in enumerate(np.split(inputs, indices_or_sections=new_inputs.shape[0], axis=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0 : new_example.shape[0]] = new_example

        # batching now fixed
        return new_inputs, labels
