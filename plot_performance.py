import time
from typing import Literal, Optional

import numpy as np

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import tyro

from mwe import get_matrices

SEED = 0 # for matrix entries

WIDTH = 50


def get_jax_timing_data(xs):
    times = []
    for this_num in range(2, len(xs) + 1):

        this_xs = xs[:this_num]

        start = time.time()
        jnp.linalg.multi_dot(this_xs).block_until_ready()
        total_time = time.time() - start

        times.append(total_time)

    return times


def get_numpy_timing_data(xs):

    xs = np.asarray(xs)

    times = []
    for this_num in range(2, len(xs) + 1):

        this_xs = xs[:this_num]

        start = time.time()
        np.linalg.multi_dot(this_xs)
        total_time = time.time() - start

        times.append(total_time)

    return np.array(times)

def main(
        num: int, 
        seed: int = SEED, 
        width: int = WIDTH, 
        mode: Literal['jax', 'numpy', 'both'] = 'both', 
        yscale: Literal['log', 'linear'] = 'log', 
        save_plot: Optional[str] = None
    ): 

    assert num >= 2

    key = jax.random.key(seed)

    widths = num * [width]
    key, key_matrices = jax.random.split(key)
    xs = get_matrices(key_matrices, widths)

    fig, ax = plt.subplots()
    
    if mode == 'jax' or mode == 'both':
        print("Timing jax...")
        jax_times = get_jax_timing_data(xs)
        ax.plot(np.arange(len(jax_times)) + 2, jax_times, color='b', label="jax")

    if mode == 'numpy' or mode == 'both':
        print("Timing numpy...")
        numpy_times = get_numpy_timing_data(xs)
        ax.plot(np.arange(len(numpy_times)) + 2, numpy_times, color='g', label='numpy')

    ax.set_xlabel('Number of arrays')
    ax.set_ylabel('Execution time (s)')
    ax.set_yscale(yscale)
    ax.legend()

    fig.tight_layout()
    if save_plot is None:
        plt.show()
    else:
        plt.savefig(save_plot)
    plt.close(fig)
    

if __name__ == "__main__":
    tyro.cli(main)
