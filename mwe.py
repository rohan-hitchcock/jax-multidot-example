import time

import jax
import jax.numpy as jnp

import numpy as np

N = 12    # jax's multidot scales as O(2^N) while numpy's does not

WIDTH = 50 # multiply N matrices of shape (WIDTH, WIDTH)
SEED = 0 # for matrix entries

def get_matrices(key, widths: list[int]):
    shapes = zip(widths, widths[1:])
    keys = jax.random.split(key, num=len(widths) - 1)
    return [jax.random.normal(k, shape) for k, shape in zip(keys, shapes)]

if __name__ == "__main__":
    key = jax.random.key(SEED)

    widths = N * [WIDTH]
    # alt: 
    # widths = [38, 49, 26, 32, 29, 28, 49, 46, 41, 46, 49, 42]
    # 
    # alt:
    # key, key_widths = jax.random.split(key)
    # widths = jax.random.randint(key_widths, (N,), WIDTH // 2, WIDTH).tolist()

    xs = get_matrices(key, widths)

    # time jax's multi_dot
    jax_start = time.time()
    jnp.linalg.multi_dot(xs).block_until_ready()
    jax_time = time.time() - jax_start

    # time numpy's multi_dot
    xs = [np.asarray(x) for x in xs] 
    numpy_start = time.time()
    np.linalg.multi_dot(xs)
    numpy_time = time.time() - numpy_start

    print(f"{jax_time=} s")
    print(f"{numpy_time=} s")
