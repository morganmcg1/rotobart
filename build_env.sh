python3 -m venv jax_env
source ~/jax_env/bin/activate
pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install transformers datasets einops flax aiohttp jsonlines zstandard wandb SentencePiece nltk prefetch_generator torch tensorboard tensorflow
