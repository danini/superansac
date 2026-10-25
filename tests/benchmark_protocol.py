"""Shared benchmark protocol: iteration budgets, worker count, per-problem records and plot locations.

Main tables : every method runs its own adaptive termination within [TABLE_MIN_ITERS, TABLE_MAX_ITERS].
Sweeps      : the same configuration with min_iterations in SWEEP_MIN_ITERS (max_iterations = TABLE_MAX_ITERS),
              so the min_iterations = TABLE_MIN_ITERS point of every curve is exactly the table configuration.
              Methods without a minimum (OpenCV) sweep their iteration cap over SWEEP_CAPS instead.
"""
import os

TABLE_MIN_ITERS, TABLE_MAX_ITERS = 1000, 5000
SWEEP_MIN_ITERS = [0, 25, 50, 100, 250, 500, 1000, 2500, 5000]
SWEEP_CAPS = [25, 50, 100, 250, 500, 1000, 2500, 5000]


def default_workers():
    """Few enough worker processes that the per-call timings are not inflated by contention."""
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def per_problem_record(results, methods):
    """Per-problem errors, aligned across methods, for paired (cluster) bootstrap confidence intervals.

    `keys` identify the independent unit (image pair / scene); repeated runs or several matchers on the
    same pair share a key, so the bootstrap can resample whole clusters.
    """
    return {
        "keys": [str(r.get("scene", i)) for i, r in enumerate(results)],
        "errors": {m: [float(r[m][0]) for r in results] for m in methods},
    }


def plots_dir(data_dir):
    """Per-script exploration plots go to <data_dir>/plots (not tracked); README figures come from
    tests/make_readme_figures.py."""
    path = os.path.join(data_dir, "plots")
    os.makedirs(path, exist_ok=True)
    return path

