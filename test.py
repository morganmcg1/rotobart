from modeling_flax_rotobart import *
from configuration_rotobart import *
from transformers import BartTokenizer
from data_collator import DataCollatorForTextInfilling

config = RotoBARTConfig(encoder_layers=2, \
    encoder_ffn_dim=256, decoder_layers=2, decoder_ffn_dim=256)

#model = FlaxRotoBARTModel(config=config)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
special_tokens_dict = {'additional_special_tokens': ['[MASK]']}
tokenizer.add_special_tokens(special_tokens_dict)


lorem_ispum = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Nec feugiat nisl pretium fusce id. Odio ut enim blandit volutpat maecenas volutpat. Tincidunt dui ut ornare lectus sit amet est placerat. Non tellus orci ac auctor augue. Gravida quis blandit turpis cursus in. Pharetra vel turpis nunc eget lorem. Sit amet cursus sit amet dictum sit amet justo. Ipsum consequat nisl vel pretium lectus quam. At in tellus integer feugiat scelerisque varius morbi. Risus nec feugiat in fermentum. In ante metus dictum at tempor commodo ullamcorper a lacus. Id neque aliquam vestibulum morbi blandit cursus risus at. Elementum pulvinar etiam non quam lacus suspendisse faucibus interdum posuere. Integer feugiat scelerisque varius morbi enim nunc faucibus a pellentesque. Sit amet cursus sit amet. Cursus mattis molestie a iaculis at erat pellentesque. Sed sed risus pretium quam vulputate dignissim suspendisse. Eu feugiat pretium nibh ipsum consequat nisl vel pretium.

Quis auctor elit sed vulputate mi sit amet mauris commodo. Tempus urna et pharetra pharetra massa massa ultricies mi. Lobortis elementum nibh tellus molestie nunc non. Pellentesque eu tincidunt tortor aliquam nulla. Amet mattis vulputate enim nulla aliquet. Volutpat commodo sed egestas egestas fringilla phasellus faucibus. Orci sagittis eu volutpat odio facilisis mauris sit amet. Id ornare arcu odio ut sem nulla pharetra. Iaculis nunc sed augue lacus viverra vitae congue. Tincidunt eget nullam non nisi est sit.

Molestie ac feugiat sed lectus vestibulum mattis. Ut sem nulla pharetra diam sit amet. Varius sit amet mattis vulputate enim nulla aliquet. Bibendum arcu vitae elementum curabitur vitae nunc sed velit dignissim. Sit amet luctus venenatis lectus magna fringilla. Tellus rutrum tellus pellentesque eu tincidunt tortor aliquam nulla. Vel pharetra vel turpis nunc eget lorem dolor sed viverra. Hendrerit gravida rutrum quisque non tellus orci ac. Netus et malesuada fames ac turpis. Nibh nisl condimentum id venenatis a condimentum vitae sapien. Eu tincidunt tortor aliquam nulla facilisi. Pharetra massa massa ultricies mi quis hendrerit dolor. Nisl nisi scelerisque eu ultrices vitae auctor eu augue.
"""

inputs = tokenizer([lorem_ispum, lorem_ispum, lorem_ispum+lorem_ispum], max_length=1024, padding=True, truncation=True)


collator = DataCollatorForTextInfilling(tokenizer)


print(collator(inputs['input_ids']))
#print(inputs)
#model.encode(**inputs)