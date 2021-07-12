# RotoBART

## ToDos:
- Ensure fixed sequence lengths (padding), to avoid jax recompilation
- Test on our TPU VM
- Check we're using all TPU capacity
- Define hyperparamters to train on
- ~~Model parallel ?~~
- ~~Train 128k sentencepiece tokenizer on the Pile~~
- ~~add sentence permutation to train script~~
- ~~add text infill to train script~~
- ~~Add checkpointing~~



## Running the script

### Script arguemnts

Available model config arguments from script:
```
encoder_layers
encoder_ffn_dim
decoder_layers
decoder_ffn_dim
d_model
vocab_size
max_position_embeddings
encoder_layerdrop
decoder_layerdrop
```

`testing` : only uses 1 batch, for testing the script
`adafactor`: will enable adafactor, removing the command will revert to Adam
`grad_accum`: what value for gradient accumulation to use, default is 4

```
python rotobart/run_dnlm_flax.py \
  --output_dir rotobart_output \
  --overwrite_output_dir \
  --dataset_path rotobart/pile.py \
  --model_name_or_path rotobart \
  --tokenizer_name ./rotobart/vocab-2/the_pile.model \
  --shuffle_buffer_size 100_000 \
  --do_train --do_eval \
  --max_seq_length 1024 \
  --encoder_layers 2 \
  --decoder_layers 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --logging_steps 8 \
  --num_train_steps 1000 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --num_eval_samples 100 \
  --warmup_steps 30 \
  --learning_rate 1e-4 \
  --use_wandb \
  --testing \
  --colab_tpu \
  --use_bf16 \
  --adafactor
```
