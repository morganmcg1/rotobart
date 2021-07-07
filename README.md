# RotoBART

## ToDos:
- Ensure fixed sequence lengths (padding), to avoid jax recompilation
- Test on our TPU VM
- Check we're using all TPU capacity
- Define hyperparamters to train on
- ~~Model parallel ?
- ~~Train 128k sentencepiece tokenizer on the Pile
- ~~add sentence permutation to train script
- ~~add text infill to train script
- ~~Add checkpointing



## Running the script

```
python run_dnlm_flax.py \
  --output_dir rotobart_output \
  --overwrite_output_dir \
  --model_name_or_path rotobart \
  --tokenizer_name facebook/bart-large-cnn \
  --shuffle_buffer_size 100_000 \
  --do_train --do_eval \
  --max_seq_length 1024 \
  --encoder_layers 2 \
  --decoder_layers 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --num_train_steps 1000 \
  --logging_steps 8 \
  --eval_steps 1000 \
  --save_steps 1000 \  # how often to checkpoint
  --num_eval_samples 100 \
  --warmup_steps 30 \
  --learning_rate 1e-4 \
  --use_wandb \  # Turn on weights and biases logging
  --testing \    # 1 batch training
  --colab_tpu \  # if running on colab TPU
  --use_bf16     # convert model to bf16
```
