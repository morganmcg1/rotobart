import jax.numpy as jnp
import numpy as np
from einops import rearrange, repeat

# Taken from Ben Wang's mesh-transformer-jax implementation
# https://github.com/kingoflolz/mesh-transformer-jax


def fixed_pos_embedding(x, seq_dim=1, seq_len=None, position_ids=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    if position_ids is None:
        position_ids = np.arange(seq_len)

    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = jnp.einsum("i... , j -> i... j", position_ids, inv_freq)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d j -> ... (d j)")


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(lambda t: repeat(t[:, offset : x.shape[1] + offset, :], "b n d -> b n () (d j)", j=2), sincos)
    return (x * cos) + (rotate_every_two(x) * sin)
