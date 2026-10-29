"""Baseline data generators that use no behavioral knowledge and no human data.

These answer the question a skeptical reviewer should ask first: does the behavioral engine do any better
than the ordinary statistical simulation a researcher could write in ten lines (the DeclareDesign / fabricatr /
simr style: a latent normal outcome, a stated effect size, equal-probability response categories)?

* ``naive_null``   : every response category is equally likely under a latent normal with no condition effect.
* ``naive_prior``  : the same, with a researcher-stated standardized effect on the focal cell.

Binary outcomes are Bernoulli(0.5) shifted by the same latent mechanism. Outcomes are independent across DVs.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import stats

from .metrics import DV


def naive_dataset(n_per_cell: Dict[str, int], dvs: List[DV], focal_cell: str, d: float, seed: int) -> Dict[str, np.ndarray]:
    """Return {"cell": ..., dv: ...} with a latent-normal ordinal DGP."""
    rng = np.random.default_rng(seed)
    cells: List[str] = []
    for cell, n in n_per_cell.items():
        cells += [cell] * n
    cells_arr = np.array(cells)
    out: Dict[str, np.ndarray] = {"cell": cells_arr}
    for dv in dvs:
        k = len(dv.levels)
        z = rng.standard_normal(len(cells_arr)) + np.where(cells_arr == focal_cell, d, 0.0)
        # equal-probability categories under the null: thresholds at the k-quantiles of N(0,1)
        thr = stats.norm.ppf(np.arange(1, k) / k)
        idx = np.searchsorted(thr, z)
        out[dv.name] = np.array(dv.levels, dtype=float)[idx]
    return out
