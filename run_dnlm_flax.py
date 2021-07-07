#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import nltk

import flax
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    BatchEncoding,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    #BartTokenizerFast,
    DebertaV2Tokenizer,
    TensorType,
    TrainingArguments,
    is_tensorboard_available,
    set_seed,
)

## TODO: import from rotobart file
#from transformers.models.bart.configuration_bart import shift_tokens_right

# RotoBART imports
from modeling_flax_rotobart import *
from configuration_rotobart import *
from transformers import BartTokenizer
from data_collator import DataCollatorForSentencePermutation, DataCollatorForTextInfilling, SentenceTokenize

# MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_MASKED_LM_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    # model_type: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    # )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )
    encoder_layers: Optional[int] = field(
        default=2,
        metadata={
            "help": "Number of encoder layers"
        },
    )
    decoder_layers: Optional[int] = field(
        default=2,
        metadata={
            "help": "Number of decoder layers"
        },
    )
    encoder_ffn_dim: Optional[int] = field(
        default=256,
        metadata={
            "help": "Dimension of encoder feedforward network"
        },
    )
    decoder_ffn_dim: Optional[int] = field(
        default=256,
        metadata={
            "help": "Dimension of decoder feedforward network"
        },
    )
    decoder_ffn_dim: Optional[int] = field(default=256,
        metadata={"help": "Dimension of decoder feedforward network"})
    d_model: Optional[int] = field(default=1024,
        metadata={"help": "Dimension of model"})
    vocab_size: Optional[int] = field(default=50265,
        metadata={"help": "Vocab size"})
    max_position_embeddings: Optional[int] = field(default=1024,
        metadata={"help": "Max position embeddings"})
    encoder_layerdrop: Optional[float] = field(default=0.0,
        metadata={"help": "Max position embeddings"})
    decoder_layerdrop: Optional[float] = field(default=0.0,
        metadata={"help": "Max position embeddings"})
    use_bf16: bool = field(
      default=False, metadata={"help": "Train in bf16 or not"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_path: Optional[str] = field(
        default="./pile.py", metadata={"help": "Path to custom dataset file."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word masking in Chinese."},
    )
    validation_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input validation ref data file for whole word masking in Chinese."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization and masking. Sequences longer than this will be truncated. Default to the max input length of the model."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for span masked language modeling loss"}
    )
    mean_noise_span_length: float = field(
        default=3.0,
        metadata={"help": "Mean span length of masked tokens"},
    )
    shuffle_buffer_size: int = field(
        default=10000, metadata={"help": "The number of examples to pre-load for shuffling."}
    )
    num_train_steps: int = field(
      default=50000, metadata={"help": "The number of training steps."}
    )
    num_eval_samples: int = field(
      default=50000, metadata={"help": "The number of samples to be used for evaluation"}
    )
    use_wandb: bool = field(
      default=False, metadata={"help": "Use Weights & Biases for experiment tracking"}
    )
    testing: bool = field(
      default=False, metadata={"help": "If testing, only 1 train batch will be used"}
    )
    colab_tpu: bool = field(
      default=False, metadata={"help": "Whether you are training on a colab TPU"}
    )


def generate_batch_splits(samples_idx: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    num_samples = len(samples_idx)
    samples_to_remove = num_samples % batch_size

    if samples_to_remove != 0:
        samples_idx = samples_idx[:-samples_to_remove]
    sections_split = num_samples // batch_size
    batch_idx = np.split(samples_idx, sections_split)
    return batch_idx


def advance_iter_and_group_samples(train_iterator, num_samples, max_seq_length):
    """
    The training iterator is advanced so that after groupifying the samples,
    `num_samples` of length `max_seq_length` are returned.
    """
    num_total_tokens = max_seq_length * num_samples
    samples = defaultdict(list)

    i = 0
    while i < num_total_tokens:
        tokenized_samples = next(train_iterator)
        i += len(tokenized_samples["input_ids"])

        # concatenate tokenized samples to list
        samples = {k: samples[k] + tokenized_samples[k] for k in tokenized_samples.keys()}

    # Concatenated tokens are split to lists of length `max_seq_length`.
    # Note that remainedr of % max_seq_length are thrown away.
    def group_texts(examples):
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, num_total_tokens, max_seq_length)]
            for k, t in examples.items()
        }
        return result

    grouped_samples = group_texts(samples)

    # print(grouped_samples)
    # print(len(grouped_samples))

    return grouped_samples


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


if __name__ == "__main__":
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup Colab TPU
    if data_args.colab_tpu:
      print("Setting up colab TPU")
      import jax.tools.colab_tpu
      jax.tools.colab_tpu.setup_tpu()
      print(f"Colab TPU setup complete, jax.device_count: {jax.device_count()}")

    # TODO: Fix logger
    # # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     level="NOTSET",
    #     datefmt="[%X]",
    # )

    # TODO: Fix logger
    # Log on each process the small summary:
    # logger = logging.getLogger(__name__)
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #     + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    # )

    # Set the verbosity to info of the Transformers logger (on main process only):
    #logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load Datasets
    # Train Dataset - Stream The Pile dataset
    print('Loading train data')
    train_dataset = load_dataset(
      data_args.dataset_path, 
      split="train",
      cache_dir=model_args.cache_dir,
      streaming=True)

    print('Loading eval data')
    # Test Dataset - Stream The Pile dataset
    eval_dataset = load_dataset(
      data_args.dataset_path, 
      split="validation",
      streaming=True,
      cache_dir=model_args.cache_dir,
    )

    # Shuffle the training dataset
    shuffled_train_dataset = train_dataset.shuffle(
      buffer_size=data_args.shuffle_buffer_size,
      seed=training_args.seed
    )

    # Sentence Tokenization
    # Used for Sentence Permutation
    sent_tok = SentenceTokenize()
    # Batching is not yet supported. Not sure if necessary?
    sent_tokenized_train_dataset = shuffled_train_dataset.map(
        sent_tok
    )
    sent_tokenized_eval_dataset = eval_dataset.map(
        sent_tok
    ) 

    # Do Tokenization
    # Load tokenizer
    # tokenizer = BartTokenizerFast.from_pretrained(
    #     model_args.tokenizer_name, cache_dir=model_args.cache_dir
    # )

    tokenizer = DebertaV2Tokenizer.from_pretrained(
      model_args.tokenizer_name, 
      unk_token="<unk>",
      sep_token="<sep>",
      pad_token="<pad>",
      cls_token="",
      mask_token="<mask>",
      eos_token="<s>",
      bos_token="</s>"
    )

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    text_column_name = "text" 

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], 
          return_attention_mask=False,
          truncation=True,
          max_length=max_seq_length,
          padding='max_length'
          )

    tokenized_train_dataset = sent_tokenized_train_dataset.map(
        tokenize_function,
        batched=True
    )

    tokenized_eval_dataset = sent_tokenized_eval_dataset.map(
        tokenize_function,
        batched=True
    )
    
    # Do Sentence Permutation
    permute_sent = DataCollatorForSentencePermutation(tokenizer)
    tokenized_train_dataset = tokenized_train_dataset.map(permute_sent)
    tokenized_eval_dataset = tokenized_eval_dataset.map(permute_sent)

    # Do Text Infilling
    masking_collator = DataCollatorForTextInfilling(tokenizer)
    tokenized_train_dataset = tokenized_train_dataset.map(masking_collator)
    tokenized_eval_dataset = tokenized_eval_dataset.map(masking_collator)

    print("Collate deployed successfully.")

    # Log to Weights and Biases 
    if data_args.use_wandb and jax.process_index() == 0:
      import wandb
      wandb.init(entity='wandb', project='hf-flax-rotobart', sync_tensorboard=True)
      wandb.config.update(training_args)  # optional, log your configs
      wandb.config.update(model_args)  # optional, log your configs
      wandb.config.update(data_args)   # optional, log your configs

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    # Load Model
    # TODO: Leverage AutoConfig
    config = RotoBARTConfig(
      encoder_layers=model_args.encoder_layers,
      encoder_ffn_dim=model_args.encoder_ffn_dim, 
      decoder_layers=model_args.decoder_layers, 
      decoder_ffn_dim=model_args.decoder_ffn_dim,
      d_model=model_args.d_model,
      vocab_size=model_args.vocab_size,
      max_position_embeddings=model_args.max_position_embeddings,
      encoder_layerdrop=model_args.encoder_layerdrop,
      decoder_layerdrop=model_args.decoder_layerdrop,
    )

    # TODO: Load model from config
    #model = FlaxAutoModelForMaskedLM.from_config(config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype))
    model = FlaxRotoBARTForConditionalGeneration(config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype))
    
    # Convert model to bf16
    if model_args.use_bf16:
        def to_bf16(t):
            return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t)
        model.params = to_bf16(model.params)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()

    # define number steps per stream epoch
    num_train_steps = data_args.num_train_steps

    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=training_args.learning_rate, transition_steps=training_args.warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=training_args.learning_rate,
        end_value=0,
        transition_steps=num_train_steps - training_args.warmup_steps,
    )
    linear_decay_lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[training_args.warmup_steps]
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxBERT-like models.
    # For other models, one should correct the layer norm parameter naming
    # accordingly.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )

    # Setup train state
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)

    def loss_fn(logits, labels):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        loss = optax.softmax_cross_entropy(shift_logits, onehot(shift_labels, shift_logits.shape[-1]))
        return loss.mean()

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels)
            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        new_state = state.apply_gradients(grads=grad)

        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
        )

        return new_state, metrics, new_dropout_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")

        logits = model(**batch, params=params, train=False)[0]

        # compute loss, ignore padded input tokens
        label_mask = jnp.where(labels > 0, 1.0, 0.0)
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask

        # compute accuracy
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels) * label_mask

        # summarize metrics
        metrics = {"loss": loss.sum(), "accuracy": accuracy.sum(), "normalizer": label_mask.sum()}
        metrics = jax.lax.psum(metrics, axis_name="batch")

        return metrics

    p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)

    train_time = 0
    train_start = time.time()
    train_metrics = []
    eval_metrics = []

    training_iter = iter(tokenized_train_dataset)
    eval_iter = iter(tokenized_eval_dataset)

    def data_collator(examples):
      batch = tokenizer.pad(examples, return_tensors=TensorType.NUMPY)
      print(batch['input_ids'].shape)
      return batch

    print(f'Getting {data_args.num_eval_samples} eval samples')
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    eval_samples = advance_iter_and_group_samples(eval_iter, data_args.num_eval_samples, max_seq_length)

    print('Start training')
    steps = tqdm(range(num_train_steps), desc="Training...", position=0)
    for step in range(num_train_steps):
        # ======================== Training ================================
        try:
            if (data_args.testing and step == 0) or not data_args.testing:
              samples = advance_iter_and_group_samples(training_iter, train_batch_size, max_seq_length)
            
        except StopIteration:
            # Once the end of the dataset stream is reached, the training iterator
            # is reinitialized and reshuffled and a new eval dataset is randomely chosen.
            shuffle_seed += 1
            tokenized_datasets.set_epoch(shuffle_seed)

            training_iter = iter(tokenized_train_dataset)

            eval_dataset = advance_iter_and_group_samples(eval_iter, data_args.num_eval_samples, max_seq_length)
            samples = advance_iter_and_group_samples(training_iter, train_batch_size, max_seq_length)

        # process input samples
        model_inputs = data_collator(samples)

        # Model forward
        model_inputs = shard(model_inputs.data)
        # model_inputs = shard(samples.data)
        state, train_metric, dropout_rngs = p_train_step(state, model_inputs, dropout_rngs)

        train_metrics.append(train_metric)

        if step % training_args.logging_steps == 0 and step > 0:
            steps.write(
                f"Step... ({step} | Loss: {train_metric['loss'].mean()}, Learning Rate: {train_metric['learning_rate'].mean()})"
            )
            train_time += time.time() - train_start
            if has_tensorboard and jax.process_index() == 0:
                write_train_metric(summary_writer, train_metrics, train_time, step)
            train_metrics = []

        # ======================== Evaluating ==============================
        if step % training_args.eval_steps == 0 and step > 0:
            eval_samples_idx = jnp.arange(data_args.num_eval_samples)
            eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size)

            for i, batch_idx in enumerate(tqdm(eval_batch_idx, desc="Evaluating ...", position=1)):
                # process input samples
                batch_eval_samples = {k: [v[idx] for idx in batch_idx] for k, v in eval_samples.items()}
                model_inputs = data_collator(batch_eval_samples)

                # Model forward
                model_inputs = shard(model_inputs.data)
                metrics = p_eval_step(state.params, model_inputs)
                eval_metrics.append(metrics)

            # normalize eval metrics
            eval_metrics = get_metrics(eval_metrics)
            eval_metrics = jax.tree_map(jnp.sum, eval_metrics)
            eval_normalizer = eval_metrics.pop("normalizer")
            eval_metrics = jax.tree_map(lambda x: x / eval_normalizer, eval_metrics)

            # Update progress bar
            steps.desc = f"Step... ({step + 1}/{num_train_steps} | Loss: {eval_metrics['loss']}, Acc: {eval_metrics['accuracy']})"

            if has_tensorboard and jax.process_index() == 0:
                write_eval_metric(summary_writer, eval_metrics, step)
            eval_metrics = []

            # save checkpoint after each epoch and push checkpoint to the hub
            if jax.process_index() == 0 and training_args.save_strategy=="epoch":
                params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
                model.save_pretrained(
                    training_args.output_dir,
                    params=params,
                    push_to_hub=training_args.push_to_hub,
                    commit_message=f"Saving weights and logs of step {step+1}",
                )


        # save checkpoint on steps and push checkpoint to the hub
        if (training_args.save_steps % (step+1)) == 0 and training_args.save_strategy=="steps":
            params = jax.device_get(jax.tree_map(lambda x: x[0], state.params))
            model.save_pretrained(
                training_args.output_dir,
                params=params,
                push_to_hub=training_args.push_to_hub,
                commit_message=f"Saving weights and logs of step {step+1}"
            )

        # update tqdm bar
        steps.update(1)
