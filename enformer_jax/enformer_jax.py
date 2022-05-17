from functools import partial

import jmp
import jax
import haiku as hk

from jax import random
from jax import nn
import jax.numpy as np

from haiku import initializers
from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# base class

class EnformerBase(hk.Module):
    def __init__(
        self,
        *,
        dim = 1536,
        depth = 11,
        heads = 8,
        output_heads = dict(human = 5313, mouse= 1643),
        target_length = 896,
        attn_dim_key = 64,
        dropout_rate = 0.4,
        attn_dropout = 0.05,
        pos_dropout = 0.01,
        num_downsamples = 7,
        dim_divisible_by = 128,
    ):
        super().__init__()

    def __call__(self, x):
        return x

def Enformer(
    mixed_precision = False,
    mixed_precision_policy = dict(params = 'float32', compute = 'float16', output = 'float32'),
    **kwargs
):
    @hk.transform
    def inner(seq):
        if mixed_precision:
            serialized_policy = ','.join([f'{k}={v}' for k, v in mixed_precision_policy.items()])
            policy = jmp.get_policy(serialized_policy)
            hk.mixed_precision.set_policy(EnformerBase, policy)

        return EnformerBase(**kwargs)(seq)
    return inner
