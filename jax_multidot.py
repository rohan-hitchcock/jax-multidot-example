import jax
import jax.numpy as jnp

import tyro

from mwe import get_matrices

SEED = 0

import numpy as np

def multi_dot(arrays):
    """ A working version of multi_dot for jax, copied straight from numpy.linalg.multidot"""    
    n = len(arrays)
    # optimization only makes sense for len(arrays) > 2
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    elif n == 2:
        return jnp.dot(arrays[0], arrays[1])

    # save original ndim to reshape the result array into the proper form later
    ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
    # Explicitly convert vectors to 2D arrays to keep the logic of the internal
    # _multi_dot_* functions as simple as possible.
    if arrays[0].ndim == 1:
        arrays[0] = jnp.atleast_2d(arrays[0])
    if arrays[-1].ndim == 1:
        arrays[-1] = jnp.atleast_2d(arrays[-1]).T

    # _multi_dot_three is much faster than _multi_dot_matrix_chain_order
    if n == 3:
        result = _multi_dot_three(arrays[0], arrays[1], arrays[2])
    else:
        order = _multi_dot_matrix_chain_order(arrays)
        result = _multi_dot(arrays, order, 0, n - 1)

    # return proper shape
    if ndim_first == 1 and ndim_last == 1:
        return result[0, 0]  # scalar
    elif ndim_first == 1 or ndim_last == 1:
        return result.ravel()  # 1-D
    else:
        return result


def _multi_dot_three(A, B, C):
    """ From numpy.linalg.multidot"""    
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    # cost1 = cost((AB)C) = a0*a1b0*b1c0 + a0*b1c0*c1
    cost1 = a0 * b1c0 * (a1b0 + c1)
    # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
    cost2 = a1b0 * c1 * (a0 + b1c0)

    if cost1 < cost2:
        return jnp.dot(jnp.dot(A, B), C)
    else:
        return jnp.dot(A, jnp.dot(B, C))


def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    """ From numpy.linalg.multidot"""    

    # Storing m and p as numpy arrays here rather than jax arrays. They depend 
    # only on the shapes of `arrays` and we can compute them at trace time. 
    # They then go on to control the structure of the computation graph in 
    # _multi_dot

    n = len(arrays)
    # p stores the dimensions of the matrices
    # Example for p: A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
    p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    # m is a matrix of costs of the subproblems
    # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
    m = np.zeros((n, n), dtype=np.double)
    # s is the actual ordering
    # s[i, j] is the value of k at which we split the product A_i..A_j
    s = np.empty((n, n), dtype=np.intp)

    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i, j] = jnp.inf
            for k in range(i, j):
                q = m[i, k] + m[k+1, j] + p[i]*p[k+1]*p[j+1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # Note that Cormen uses 1-based index

    return (s, m) if return_costs else s


def _multi_dot(arrays, order, i, j):
    """ From numpy.linalg.multidot"""    
    if i == j:
        return arrays[i]
    else:
        return jnp.dot(
            _multi_dot(arrays, order, i, order[i, j]),
            _multi_dot(arrays, order, order[i, j] + 1, j)
        )
    

def main(seed: int = SEED, width: int = 50, n: int = 20):
    key = jax.random.key(seed)
    
    # widths = n * [width]
    key, key_widths = jax.random.split(key)
    widths = jax.random.randint(key_widths, (n,), width // 2, width).tolist()
    
    xs = get_matrices(key, widths)

    result = multi_dot(xs)

    print(result)

    jitted_multidot = jax.jit(multi_dot)
    
    result = jitted_multidot(xs)
    print(result)


if __name__ == "__main__":
    tyro.cli(main)
