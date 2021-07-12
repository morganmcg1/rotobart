import math
from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp
import nltk
import numpy as np
from jax import ops, random
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from data_collator import DataCollatorForSentencePermutation, DataCollatorForTextInfilling, SentenceTokenize

example = {"text": " My dog is cute. It loves to play in the park. There are many parks in SF."}
sent_tok = SentenceTokenize()
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
permuate_sent = DataCollatorForSentencePermutation(tokenizer)
example = sent_tok(example)
print(example["text"])
out = permuate_sent(tokenizer(example["text"], add_special_tokens=False))
example["text"] = tokenizer.decode(out["input_ids"])
print(example["text"])
masking = DataCollatorForTextInfilling(tokenizer)
out = masking(out)
example["text"] = tokenizer.decode(out["input_ids"][0])
print(example["text"])
