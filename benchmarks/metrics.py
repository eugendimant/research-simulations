"""Benchmark metrics for comparing a synthetic dataset with a human dataset.

Three purposes are kept apart because they need different evidence:

* pipeline rehearsal   : does the synthetic table support the analysis code (schema, ranges, finite estimates)?
* design sensitivity   : how do variances and implied minimum detectable effects compare?
* behavioral forecast  : do marginals, focal effects, correlations and test decisions match the human data?

Only numpy and scipy are used. Every function is deterministic given its inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


def _arr(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def hedges_g(treat: Sequence[float], ctrl: Sequence[float]) -> Dict[str, float]:
    """Standardized mean difference (Hedges g) with a normal-approximation 95% CI."""
    a, b = _arr(treat), _arr(ctrl)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"g": float("nan"), "lo": float("nan"), "hi": float("nan"), "diff": float("nan")}
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return {"g": 0.0, "lo": 0.0, "hi": 0.0, "diff": float(a.mean() - b.mean())}
    d = (a.mean() - b.mean()) / sp
    j = 1 - 3 / (4 * (na + nb) - 9)
    g = d * j
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb)))
    return {"g": float(g), "lo": float(g - 1.96 * se), "hi": float(g + 1.96 * se), "diff": float(a.mean() - b.mean())}


def welch_p(treat: Sequence[float], ctrl: Sequence[float]) -> float:
    a, b = _arr(treat), _arr(ctrl)
    if len(a) < 2 or len(b) < 2 or (a.var() == 0 and b.var() == 0):
        return 1.0
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def ols_2x2(y: Sequence[float], tone: Sequence[int], eth: Sequence[int]) -> Dict[str, float]:
    """OLS of y on tone, eth and tone x eth (0/1 coding). Returns coefficients and two-sided p values."""
    y = _arr(y)
    t, e = _arr(tone), _arr(eth)
    X = np.column_stack([np.ones_like(y), t, e, t * e])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = float(resid @ resid / dof)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    tv = beta / se
    p = 2 * stats.t.sf(np.abs(tv), dof)
    names = ["intercept", "tone", "eth", "tone_x_eth"]
    out: Dict[str, float] = {}
    for i, n in enumerate(names):
        out[f"b_{n}"] = float(beta[i])
        out[f"p_{n}"] = float(p[i])
    return out


def ordinal_wasserstein(a: Sequence[float], b: Sequence[float], lo: float, hi: float) -> float:
    """1-Wasserstein distance between two samples, divided by the response range (0 = identical)."""
    rng = float(hi - lo)
    return float(stats.wasserstein_distance(_arr(a), _arr(b)) / rng) if rng > 0 else 0.0


def total_variation(a: Sequence[float], b: Sequence[float], levels: Sequence[float]) -> float:
    """Total variation distance between category proportions."""
    a, b = _arr(a), _arr(b)
    pa = np.array([(a == l).mean() for l in levels])
    pb = np.array([(b == l).mean() for l in levels])
    return float(0.5 * np.abs(pa - pb).sum())


def spearman_matrix(cols: List[np.ndarray]) -> np.ndarray:
    k = len(cols)
    m = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            if np.std(cols[i]) == 0 or np.std(cols[j]) == 0:
                r = 0.0
            else:
                r = float(stats.spearmanr(cols[i], cols[j]).statistic)
            m[i, j] = m[j, i] = r
    return m


def mean_abs_offdiag_diff(m1: np.ndarray, m2: np.ndarray) -> float:
    k = m1.shape[0]
    iu = np.triu_indices(k, 1)
    return float(np.mean(np.abs(m1[iu] - m2[iu])))


@dataclass
class DV:
    name: str
    lo: float
    hi: float
    levels: List[float]


def summarize(table: Dict[str, np.ndarray], dvs: List[DV], focal: np.ndarray, tone: np.ndarray, eth: np.ndarray) -> Dict:
    """Statistics of one dataset. `focal` is a 0/1 array marking the focal cell (negative tone x Latino)."""
    focal = focal.astype(bool)
    out: Dict = {"n": int(len(focal)), "n_focal": int(focal.sum()), "dv": {}}
    cols = []
    for dv in dvs:
        y = table[dv.name].astype(float)
        cols.append(y)
        g = hedges_g(y[focal], y[~focal])
        reg = ols_2x2(y, tone, eth)
        out["dv"][dv.name] = {
            "mean": float(y.mean()),
            "sd": float(y.std(ddof=1)),
            "share_floor": float((y == dv.lo).mean()),
            "share_ceiling": float((y == dv.hi).mean()),
            "props": {str(l): float((y == l).mean()) for l in dv.levels},
            "focal_g": g["g"], "focal_lo": g["lo"], "focal_hi": g["hi"], "focal_diff": g["diff"],
            "focal_p": welch_p(y[focal], y[~focal]),
            "reg": reg,
            # raw-unit minimum detectable effect for the focal contrast at 80% power, alpha .05
            "mde_raw": float(2.8 * y.std(ddof=1) * np.sqrt(1 / max(focal.sum(), 1) + 1 / max((~focal).sum(), 1))),
        }
    out["spearman"] = spearman_matrix(cols).tolist()
    return out


def compare_to_human(human: Dict, synth: Dict, dvs: List[DV], table_h: Dict[str, np.ndarray], table_s: Dict[str, np.ndarray]) -> Dict:
    """One synthetic run against the human data. Returns flat per-DV metrics."""
    res: Dict = {"dv": {}}
    for dv in dvs:
        h, s = human["dv"][dv.name], synth["dv"][dv.name]
        sd_h = h["sd"] if h["sd"] > 0 else 1.0
        sign_h = np.sign(h["focal_g"])
        res["dv"][dv.name] = {
            "mean_err_sd": (s["mean"] - h["mean"]) / sd_h,
            "sd_ratio": s["sd"] / sd_h,
            "wasserstein": ordinal_wasserstein(table_s[dv.name], table_h[dv.name], dv.lo, dv.hi),
            "tv": total_variation(table_s[dv.name], table_h[dv.name], dv.levels),
            "focal_g_err": s["focal_g"] - h["focal_g"],
            "focal_sign_agree": float(np.sign(s["focal_g"]) == sign_h) if sign_h != 0 else float("nan"),
            "focal_ci_covers_human": float(s["focal_lo"] <= h["focal_g"] <= s["focal_hi"]),
            "focal_decision_agree": float((s["focal_p"] < 0.05) == (h["focal_p"] < 0.05)),
            "focal_signif_and_same_sign": float((s["focal_p"] < 0.05) and np.sign(s["focal_g"]) == sign_h),
            "interaction_decision_agree": float((s["reg"]["p_tone_x_eth"] < 0.05) == (h["reg"]["p_tone_x_eth"] < 0.05)),
            "mde_ratio": s["mde_raw"] / h["mde_raw"] if h["mde_raw"] > 0 else float("nan"),
        }
    res["corr_mad"] = mean_abs_offdiag_diff(np.array(human["spearman"]), np.array(synth["spearman"]))
    return res


def aggregate(runs: List[Dict], dvs: List[DV]) -> Dict:
    """Mean and standard deviation over seeds of every per-DV metric."""
    out: Dict = {"n_runs": len(runs), "dv": {}}
    for dv in dvs:
        keys = runs[0]["dv"][dv.name].keys()
        out["dv"][dv.name] = {}
        for k in keys:
            v = np.array([r["dv"][dv.name][k] for r in runs], dtype=float)
            out["dv"][dv.name][k] = {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0}
    c = np.array([r["corr_mad"] for r in runs], dtype=float)
    out["corr_mad"] = {"mean": float(c.mean()), "sd": float(c.std(ddof=1)) if len(c) > 1 else 0.0}
    return out
