import time
from functools import reduce
from typing import Callable, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array

import tyro

from mwe import get_matrices
from jax_multidot import multi_dot as numpy_port_multi_dot

SEED = 0 # for matrix entries

def reduce_dot(xs: list[Array]) -> Array:
    return reduce(jnp.dot, xs)


def multi_dot(xs: list[Array]) -> Array:
    return jnp.linalg.multi_dot(xs)


def loop_dot(xs: list[Array]) -> Array:
    product = xs[0]
    for x in xs[1:]:
        product = jnp.dot(product, x)
    return product


def profile(func: Callable[[list[Array]], Array], xs: list[Array]) -> tuple[float, float]:
    
    jitted_func = jax.jit(func)

    first_jitted_start = time.time()
    _ = jitted_func(xs).block_until_ready()
    first_jitted_time = time.time() - first_jitted_start
    
    second_jitted_start = time.time()
    _ = jitted_func(xs).block_until_ready()
    second_jitted_time = time.time() - second_jitted_start

    unjitted_start = time.time()
    _ = func(xs).block_until_ready()
    unjitted_time = time.time() - unjitted_start
    
    return first_jitted_time, second_jitted_time, unjitted_time

def main(num: int, width: int, seed: int = SEED, mode: Literal['fixed_width', 'random_width'] = 'fixed_width'):

    key = jax.random.key(seed)
    key, key_sizes = jax.random.split(key)

    if mode == 'fixed_width':
        widths = num * [width]
    elif mode == 'random_width':
        min_width = width // 2
        max_width = width

        widths = jax.random.randint(key_sizes, (num, ), min_width, max_width).tolist()
    else:
        raise ValueError(f"Unrecognised mode '{mode}'")
    
    key, key_matrices = jax.random.split(key)
    xs = get_matrices(key_matrices, widths)

    first_jitted_time, second_jitted_time, not_jitted_time = profile(loop_dot, xs)
    print(f"loop_dot:\n\t{first_jitted_time=}\n\t{second_jitted_time=}\n\t{not_jitted_time=}")

    first_jitted_time, second_jitted_time, not_jitted_time = profile(reduce_dot, xs)
    print(f"reduce_dot:\n\t{first_jitted_time=}\n\t{second_jitted_time=}\n\t{not_jitted_time=}")

    first_jitted_time, second_jitted_time, not_jitted_time = profile(numpy_port_multi_dot, xs)
    print(f"numpy_port_multi_dot:\n\t{first_jitted_time=}\n\t{second_jitted_time=}\n\t{not_jitted_time=}")

    first_jitted_time, second_jitted_time, not_jitted_time = profile(multi_dot, xs)
    print(f"multi_dot:\n\t{first_jitted_time=}\n\t{second_jitted_time=}\n\t{not_jitted_time=}")

if __name__ == "__main__":
    tyro.cli(main)
