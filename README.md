Demonstrates an issue in `jax.numpy.linalg.multi_dot`, where this function is `O(2^N)` in the number of arrays being multiplied. This is much worse than the peformance of `numpy.linalg.multi_dot`. 

Components:
- `mwe.py`: A minimum working example of the issue.
- `plot_performance.py`: Plots the performance of numpy vs jax.
- `jax_multidot.py`: A jittable version of `multi_dot` for `jax`, basically copied from `numpy.linalg.multi_dot`.
- `compare_alternatives.py`: Compares various options for doing `multi_dot` in jax.

See https://github.com/jax-ml/jax/issues/25051
