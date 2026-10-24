"""Small statistics helpers. Every bootstrap here resamples whole runs."""
from __future__ import annotations

import numpy as np

N_BOOT = 10_000


def boot_mean_ci(x, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05):
    """Mean of run-level values with a percentile bootstrap CI (resampling runs)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return x.mean(), *np.quantile(means, [alpha / 2, 1 - alpha / 2])


def boot_diff_ci(a, b, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05):
    """mean(a) - mean(b), resampling runs within each group independently."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float)
    b = b[~np.isnan(b)]
    rng = np.random.default_rng(seed)
    d = (rng.choice(a, size=(n_boot, len(a))).mean(axis=1) - rng.choice(b, size=(n_boot, len(b))).mean(axis=1))
    return a.mean() - b.mean(), *np.quantile(d, [alpha / 2, 1 - alpha / 2])
