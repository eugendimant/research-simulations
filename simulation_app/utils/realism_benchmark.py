"""
Realism Benchmark — measure simulated data against real human data.
===================================================================

The only defensible way to claim synthetic data "looks like real data" is to
measure both the same way and show the numbers. This module computes a fixed
profile of statistics that are known to separate real survey data from generated
data, and scores a simulated dataset against reference profiles computed from
openly licensed REAL datasets (see `reference_profiles.json` and
`docs/EMPIRICAL_PROVENANCE.md` for what those datasets are and their licences).

The profile covers four families:

1. MARGINAL SHAPE      per-item mean, SD, skew, excess kurtosis, and the share of
                       responses at the floor, the ceiling and the midpoint.
2. SCALE USE           how many of the available response options each respondent
                       actually uses, endpoint preference, and round-number heaping
                       on wide scales. Generated data is usually too smooth here.
3. CARELESS STRUCTURE  longest run of identical answers (long string), the share of
                       fully straight-lined respondents, and within-person SD.
                       Real data has a careless tail; clean data is a giveaway.
4. COVARIANCE          mean inter-item correlation and Cronbach's alpha. Generated
                       data is often either too independent or too uniformly
                       correlated.

Nothing here is specific to this simulator: `compute_profile` runs on any wide
table of item responses, which is what makes the comparison meaningful.

Pure standard library plus numpy if present; falls back to plain Python so the
module can never break an import chain.
"""
from __future__ import annotations

__version__ = "1.0.0"

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "reference_profiles.json")


# =============================================================================
# SMALL STATS HELPERS (no hard numpy dependency)
# =============================================================================

def _mean(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else 0.0


def _sd(xs: Sequence[float], ddof: int = 1) -> float:
    xs = [float(x) for x in xs]
    n = len(xs)
    if n <= ddof:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def _skew(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs]
    n = len(xs)
    s = _sd(xs, ddof=0)
    if n < 3 or s == 0:
        return 0.0
    m = _mean(xs)
    return sum(((x - m) / s) ** 3 for x in xs) / n


def _excess_kurtosis(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs]
    n = len(xs)
    s = _sd(xs, ddof=0)
    if n < 4 or s == 0:
        return 0.0
    m = _mean(xs)
    return sum(((x - m) / s) ** 4 for x in xs) / n - 3.0


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = [float(x) for x in a[:n]], [float(x) for x in b[:n]]
    ma, mb = _mean(a), _mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def _longest_run(row: Sequence[float]) -> int:
    best = run = 0
    prev = None
    for v in row:
        if prev is not None and v == prev:
            run += 1
        else:
            run = 1
        prev = v
        best = max(best, run)
    return best


def cronbach_alpha(columns: Sequence[Sequence[float]]) -> float:
    """Standard Cronbach's alpha over k item columns of equal length."""
    k = len(columns)
    if k < 2:
        return 0.0
    n = min(len(c) for c in columns)
    if n < 3:
        return 0.0
    cols = [[float(v) for v in c[:n]] for c in columns]
    item_var = sum(_sd(c) ** 2 for c in cols)
    totals = [sum(cols[j][i] for j in range(k)) for i in range(n)]
    total_var = _sd(totals) ** 2
    if total_var <= 0:
        return 0.0
    return (k / (k - 1.0)) * (1.0 - item_var / total_var)


# =============================================================================
# PROFILE
# =============================================================================

@dataclass
class Profile:
    """A comparable fingerprint of one block of item responses."""
    label: str = ""
    n_respondents: int = 0
    n_items: int = 0
    scale_min: float = 0.0
    scale_max: float = 0.0
    # marginal shape (averaged over items)
    item_mean: float = 0.0
    item_sd: float = 0.0
    item_skew: float = 0.0
    item_excess_kurtosis: float = 0.0
    floor_share: float = 0.0
    ceiling_share: float = 0.0
    midpoint_share: float = 0.0
    # scale use
    options_used_mean: float = 0.0        # distinct options per respondent
    endpoint_share: float = 0.0           # responses at either endpoint
    round_number_share: float = 0.0       # wide scales only
    # careless structure
    long_string_mean: float = 0.0
    long_string_p95: float = 0.0
    straightlined_share: float = 0.0
    within_person_sd_mean: float = 0.0
    # covariance
    mean_interitem_r: float = 0.0
    alpha: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _as_columns(data: Any, item_columns: Sequence[str]) -> List[List[float]]:
    """Accept a pandas DataFrame or a dict of column -> values, LISTWISE.

    Rows with a missing or non-numeric value in any requested column are dropped
    from ALL columns. Dropping per column instead would shift respondents against
    each other, which silently corrupts every row-wise metric here — long string,
    within-person SD, straight-lining and the inter-item correlations. A simulated
    block with 10% missingness measured 0.5% straight-liners under per-column
    dropping and 11.5% under listwise dropping; only the second is the truth.
    """
    raw: List[List[Any]] = []
    for c in item_columns:
        try:
            series = data[c]
        except Exception:
            return []
        raw.append(list(series.tolist()) if hasattr(series, "tolist") else list(series))
    if not raw:
        return []
    n = min(len(c) for c in raw)
    cols: List[List[float]] = [[] for _ in raw]
    for i in range(n):
        row: List[float] = []
        for c in raw:
            try:
                f = float(c[i])
            except (TypeError, ValueError):
                row = []
                break
            if f != f:            # NaN
                row = []
                break
            row.append(f)
        if not row:
            continue              # listwise deletion: skip this respondent entirely
        for j, f in enumerate(row):
            cols[j].append(f)
    if not cols or not cols[0]:
        return []
    return cols


def compute_profile(
    data: Any,
    item_columns: Sequence[str],
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
    label: str = "",
) -> Optional[Profile]:
    """Compute the comparable profile of one block of items.

    `data` may be a pandas DataFrame or a dict of column -> sequence. Rows with
    non-numeric or missing values in a column are dropped from that column.
    Returns None when there is not enough data to compute anything stable.
    """
    cols = _as_columns(data, item_columns)
    if not cols:
        return None
    n = len(cols[0])
    k = len(cols)
    if n < 5:
        return None

    observed = [v for c in cols for v in c]
    lo = float(scale_min) if scale_min is not None else min(observed)
    hi = float(scale_max) if scale_max is not None else max(observed)
    if hi <= lo:
        return None
    mid = (lo + hi) / 2.0
    span = hi - lo

    p = Profile(label=label, n_respondents=n, n_items=k, scale_min=lo, scale_max=hi)
    p.item_mean = _mean([_mean(c) for c in cols])
    p.item_sd = _mean([_sd(c) for c in cols])
    p.item_skew = _mean([_skew(c) for c in cols])
    p.item_excess_kurtosis = _mean([_excess_kurtosis(c) for c in cols])
    total = len(observed)
    p.floor_share = sum(1 for v in observed if abs(v - lo) < 1e-9) / total
    p.ceiling_share = sum(1 for v in observed if abs(v - hi) < 1e-9) / total
    p.midpoint_share = sum(1 for v in observed if abs(v - mid) < 1e-9) / total
    p.endpoint_share = p.floor_share + p.ceiling_share

    # round-number heaping only means something on a wide scale
    if span >= 20:
        step = span / 20.0
        p.round_number_share = sum(
            1 for v in observed
            if abs((v - lo) / step - round((v - lo) / step)) < 1e-6
        ) / total

    rows = [[cols[j][i] for j in range(k)] for i in range(n)]
    if k >= 2:
        runs = [_longest_run(r) for r in rows]
        runs_sorted = sorted(runs)
        p.long_string_mean = _mean(runs)
        p.long_string_p95 = float(runs_sorted[min(len(runs_sorted) - 1,
                                                  int(0.95 * len(runs_sorted)))])
        p.straightlined_share = sum(1 for r in runs if r == k) / n
        p.within_person_sd_mean = _mean([_sd(r) for r in rows])
        p.options_used_mean = _mean([len(set(r)) for r in rows])
        rs = [_pearson(cols[a], cols[b]) for a in range(k) for b in range(a + 1, k)]
        p.mean_interitem_r = _mean(rs) if rs else 0.0
        p.alpha = cronbach_alpha(cols)
    else:
        p.options_used_mean = 1.0
    return p


# =============================================================================
# COMPARISON
# =============================================================================

#: Metrics compared, with the tolerance inside which simulated data is considered
#: indistinguishable from the reference on that metric. Tolerances are absolute and
#: were chosen as roughly the spread BETWEEN real reference datasets, so "within
#: tolerance" means "as close to this real dataset as two real datasets are to
#: each other" rather than an arbitrary pass mark.
COMPARED_METRICS: Dict[str, float] = {
    "item_sd": 0.25,
    "item_skew": 0.35,
    "item_excess_kurtosis": 0.60,
    "floor_share": 0.06,
    "ceiling_share": 0.06,
    "midpoint_share": 0.06,
    "endpoint_share": 0.08,
    "options_used_mean": 0.80,
    "long_string_mean": 0.80,
    "straightlined_share": 0.04,
    "within_person_sd_mean": 0.25,
    "mean_interitem_r": 0.12,
    "alpha": 0.12,
}


@dataclass
class Comparison:
    reference_label: str = ""
    per_metric: Dict[str, Dict[str, float]] = field(default_factory=dict)
    n_within_tolerance: int = 0
    n_compared: int = 0

    @property
    def score(self) -> float:
        """Share of compared metrics within tolerance, in [0, 1]."""
        return (self.n_within_tolerance / self.n_compared) if self.n_compared else 0.0

    def failures(self) -> List[str]:
        return [m for m, d in self.per_metric.items() if not d.get("within")]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reference_label": self.reference_label,
            "score": round(self.score, 4),
            "n_within_tolerance": self.n_within_tolerance,
            "n_compared": self.n_compared,
            "per_metric": self.per_metric,
            "failures": self.failures(),
        }


def compare(
    simulated: Profile,
    reference: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
    tolerance_multiplier: float = 1.0,
) -> Comparison:
    """Compare a simulated profile to a reference profile, metric by metric.

    A reference may carry its own per-metric `tolerance` map, derived from the
    spread between real blocks of the same instrument; that is preferred over the
    module defaults. `tolerance_multiplier` widens every tolerance, which the
    caller does when the reference's scale length does not match the data's.
    """
    mets = dict(metrics or COMPARED_METRICS)
    ref_tol = reference.get("tolerance") or {}
    for m, t in ref_tol.items():
        if m in mets or m in simulated.to_dict():
            mets[m] = float(t)
    out = Comparison(reference_label=str(reference.get("label", "reference")))
    sim = simulated.to_dict()
    for metric, tol in mets.items():
        tol = float(tol) * max(1.0, float(tolerance_multiplier))
        if metric not in reference or reference.get(metric) is None:
            continue
        if metric not in sim:
            continue
        ref_v, sim_v = float(reference[metric]), float(sim[metric])
        delta = sim_v - ref_v
        within = abs(delta) <= tol
        out.per_metric[metric] = {
            "simulated": round(sim_v, 4),
            "reference": round(ref_v, 4),
            "delta": round(delta, 4),
            "tolerance": tol,
            "within": bool(within),
        }
        out.n_compared += 1
        out.n_within_tolerance += int(within)
    return out


# =============================================================================
# REFERENCE PROFILES
# =============================================================================

def load_reference_profiles(path: Optional[str] = None) -> Dict[str, Any]:
    """Reference profiles computed from real, openly licensed datasets.

    Returns {} when the file is absent, which disables benchmarking rather than
    inventing a reference to compare against.
    """
    p = path or _REFERENCE_PATH
    try:
        with open(p, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def reference_for_scale(
    scale_points: int,
    profiles: Optional[Dict[str, Any]] = None,
    direction_aligned: bool = True,
) -> Optional[Dict[str, Any]]:
    """Pick the reference whose scale length and keying match the data.

    `direction_aligned` says whether the block being checked has all its items
    keyed the same way. It matters a great deal: in the real reference data,
    aligning the reverse-keyed items moves the mean inter-item correlation from
    .096 to .363 and Cronbach's alpha from .02 to .73. Comparing an all-positive
    simulated block against a mixed-keying reference would be comparing it to a
    number no real instrument reports.
    """
    payload = profiles if profiles is not None else load_reference_profiles()
    refs = payload.get("profiles") or []
    if not refs:
        return None
    keyed = [r for r in refs if bool(r.get("direction_aligned", True)) == bool(direction_aligned)]
    pool = keyed or refs
    exact = [r for r in pool if int(r.get("scale_points", 0)) == int(scale_points)]
    pool = exact or pool
    return min(pool, key=lambda r: abs(int(r.get("scale_points", 0)) - int(scale_points)))


def benchmark(
    data: Any,
    item_columns: Sequence[str],
    scale_min: float,
    scale_max: float,
    label: str = "simulated",
    direction_aligned: bool = True,
) -> Dict[str, Any]:
    """Profile a simulated block and compare it to the matching real reference.

    Returns a dict with the profile, the comparison and a short verdict. When no
    reference is available the profile is still returned, with
    `comparison: None` and a verdict saying so — never a fabricated score.
    """
    prof = compute_profile(data, item_columns, scale_min, scale_max, label=label)
    if prof is None:
        return {"profile": None, "comparison": None,
                "verdict": "not enough data to profile"}
    scale_points = int(round(scale_max - scale_min + 1))
    ref = reference_for_scale(scale_points, direction_aligned=direction_aligned)
    if not ref:
        return {"profile": prof.to_dict(), "comparison": None,
                "verdict": "no real-data reference available for this scale length"}
    ref_points = int(ref.get("scale_points", scale_points) or scale_points)
    approximate = ref_points != scale_points
    # A reference from a different scale length still constrains shape, but less
    # tightly: floor and ceiling shares in particular depend on how many options
    # a respondent had. Widen the tolerances rather than pretending to a match.
    cmp_ = compare(prof, ref, tolerance_multiplier=1.6 if approximate else 1.0)
    if cmp_.n_compared == 0:
        verdict = "reference profile has no comparable metrics"
    elif cmp_.score >= 0.85:
        verdict = "indistinguishable from the reference on most metrics"
    elif cmp_.score >= 0.60:
        verdict = "broadly realistic, with specific departures listed in failures"
    else:
        verdict = "distinguishable from real data on most metrics"
    if approximate:
        verdict += (f" (reference is a {ref_points}-point scale, data is "
                    f"{scale_points}-point; tolerances widened accordingly)")
    return {"profile": prof.to_dict(), "comparison": cmp_.to_dict(),
            "approximate_reference": approximate, "verdict": verdict}
