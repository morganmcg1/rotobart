from modeling_flax_rotobart import *
from configuration_rotobart import *
from transformers import BartTokenizer

config = RotoBARTConfig(encoder_layers=2, \
    encoder_ffn_dim=256, decoder_layers=2, decoder_ffn_dim=256)

model = FlaxRotoBARTModel(config=config)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

inputs = tokenizer(["Some text"], max_length=1024, return_tensors='jax')


model.encode(**inputs)