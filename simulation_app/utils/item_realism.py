"""
Item Realism — give multi-item blocks the reliability real scales actually have.
==============================================================================

THE FINDING THIS MODULE EXISTS FOR
----------------------------------
Benchmarked against real human item-level data (psych::bfi, 2436 complete cases,
25 IPIP items on a 6-point agreement scale — see `reference_profiles.json`), the
simulator's Likert blocks matched 7 of 13 distributional metrics. The six that
failed all said the same thing:

    metric                simulated   real    
    mean inter-item r        0.60     0.36
    Cronbach's alpha         0.88     0.73
    within-person SD         0.72     1.47
    distinct options used    2.49     3.16
    share at scale floor     0.02     0.12

Simulated participants are too internally consistent. Every item in a block is
nearly the same draw from one latent trait, so a block is almost a single number
repeated with a nudge. Real respondents are much noisier item to item: alpha of a
good published scale is around .7-.8, not .88, and a real person's five answers to
five items of the same construct spread about twice as widely as the simulator's.
That gap is the single most detectable signature of synthetic survey data, and it
is visible in any correlation matrix a user runs.

WHAT THIS MODULE DOES
---------------------
It decomposes a block into the part shared across items (the latent trait, which
carries the treatment effect and the persona structure) and the part unique to
each item (item-specific variance plus measurement error), then scales the unique
part up until the block's internal consistency matches real data.

    x_ij = item_mean_j + common_i + unique_ij
                          ^^^^^^^^   ^^^^^^^^
                          untouched  scaled up

Because the common part is untouched, every participant keeps their rank on the
latent trait: condition effects, persona correlations and cross-scale structure
all survive.

THE HONEST TRADE-OFF
--------------------
Lower reliability attenuates the effect size observed on the composite score.
That is not a bug, it is measurement error, and it is why real studies need more
participants than a power calculation on true scores suggests. It cannot be
avoided while also fixing reliability: scaling the common and unique parts by the
same factor leaves the inter-item correlation exactly where it was, so there is no
transform that lowers alpha and leaves the observed composite d untouched.

So the module offers the choice, and defaults to protecting the user's
specification: with `preserve_effect=True` (the default) the BETWEEN-CONDITION
component of the common part is scaled up by exactly the factor that cancels the
attenuation, so the user still observes the effect size they asked for while the
data gains realistic reliability. With `preserve_effect=False` the attenuation is
left in place, which is what you want when the question is "how much power would
this design really have?".
"""
from __future__ import annotations

__version__ = "1.0.0"

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


#: Target mean inter-item correlation, from the real-data reference profile
#: (psych::bfi, five 5-item blocks: observed range .24 to .47, mean .363).
#: Blocks already at or below this are left alone — this module only ever makes
#: data LESS artificially consistent, never more.
DEFAULT_TARGET_R = 0.363

#: Never scale the unique component by more than this. A block whose simulated
#: items are almost perfectly correlated would otherwise need an enormous factor,
#: and pushing that far turns the block into noise. Capping means such a block
#: improves without being destroyed, and `decouple_block` reports the cap was hit.
MAX_UNIQUE_SCALE = 2.60


@dataclass
class DecoupleReport:
    """What the transform did, so it can be logged and tested."""
    applied: bool = False
    reason: str = ""
    n_items: int = 0
    n_rows: int = 0
    r_before: float = 0.0
    r_target: float = 0.0
    unique_scale: float = 1.0
    between_condition_scale: float = 1.0
    hit_scale_cap: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def _var(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    ma = sum(a[:n]) / n
    mb = sum(b[:n]) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def mean_interitem_r(columns: Sequence[Sequence[float]]) -> float:
    k = len(columns)
    if k < 2:
        return 0.0
    rs = [_pearson(columns[a], columns[b]) for a in range(k) for b in range(a + 1, k)]
    return sum(rs) / len(rs) if rs else 0.0


def decouple_block(
    columns: Sequence[Sequence[float]],
    scale_min: float,
    scale_max: float,
    condition_labels: Optional[Sequence[Any]] = None,
    target_r: float = DEFAULT_TARGET_R,
    preserve_effect: bool = True,
    integer_scale: bool = True,
) -> Tuple[List[List[float]], DecoupleReport]:
    """Raise a block's item-specific variance until its reliability is realistic.

    `columns` is one sequence per item, all the same length, ordered by respondent.
    Returns the transformed columns and a report. The input is returned unchanged,
    with `applied=False` and a reason, whenever the transform does not apply:
    fewer than 3 items, fewer than 20 respondents, a degenerate scale, or a block
    whose inter-item correlation is already at or below the real-data target.
    """
    rep = DecoupleReport(r_target=float(target_r))
    k = len(columns)
    rep.n_items = k
    if k < 3:
        rep.reason = "fewer than 3 items: reliability is not estimable"
        return [list(c) for c in columns], rep
    n = min(len(c) for c in columns)
    rep.n_rows = n
    if n < 20:
        rep.reason = "fewer than 20 respondents: correlations are unstable"
        return [list(c) for c in columns], rep
    span = float(scale_max) - float(scale_min)
    if span <= 0:
        rep.reason = "degenerate scale range"
        return [list(c) for c in columns], rep

    cols = [[float(v) for v in c[:n]] for c in columns]
    r_before = mean_interitem_r(cols)
    rep.r_before = r_before
    if r_before <= target_r:
        rep.reason = (f"inter-item r {r_before:.3f} already at or below the "
                      f"real-data target {target_r:.3f}")
        return [list(c) for c in cols], rep

    item_means = [sum(c) / n for c in cols]
    # common_i: the respondent's shared position on the construct
    common = [sum(cols[j][i] - item_means[j] for j in range(k)) / k for i in range(n)]
    # unique_ij: everything item-specific, including measurement error
    unique = [[cols[j][i] - item_means[j] - common[i] for i in range(n)] for j in range(k)]

    v_common = _var(common)
    v_unique = sum(_var(u) for u in unique) / k
    if v_common <= 1e-9 or v_unique <= 1e-12:
        rep.reason = "no usable variance to redistribute"
        return [list(c) for c in cols], rep

    # An analytic scale factor from the variance decomposition overshoots: the
    # row-mean estimate of the common part absorbs a share of the item-specific
    # noise, and real items do not load equally. So the factor is SOLVED against
    # the metric that actually matters — the measured mean inter-item correlation
    # after rounding and clipping. It is monotone decreasing in the scale factor,
    # so a bisection converges, and the measured value is what the benchmark
    # compares against.
    def _r_at(scale: float) -> float:
        trial = []
        for j in range(k):
            trial.append([
                min(scale_max, max(scale_min,
                    item_means[j] + common[i] + unique[j][i] * scale))
                for i in range(n)
            ])
        if integer_scale:
            trial = [[float(round(v)) for v in c] for c in trial]
        return mean_interitem_r(trial)

    lo, hi = 1.0, MAX_UNIQUE_SCALE
    if _r_at(hi) > target_r:
        s = hi                      # cannot decouple far enough within the cap
        rep.hit_scale_cap = True
    else:
        for _ in range(28):
            mid = (lo + hi) / 2.0
            if _r_at(mid) > target_r:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2.0
    if s <= 1.0001:
        rep.reason = "block already has at least the real level of item-specific variance"
        return [list(c) for c in cols], rep
    v_unique_target = v_unique * s * s
    rep.unique_scale = s

    # Pre-compensate the treatment effect, so the user still observes the effect
    # size they specified once measurement error has been added.
    b_scale = 1.0
    between = [0.0] * n
    within = list(common)
    if preserve_effect and condition_labels is not None and len(condition_labels) >= n:
        groups: Dict[Any, List[int]] = {}
        for i in range(n):
            groups.setdefault(condition_labels[i], []).append(i)
        if len(groups) >= 2 and all(len(ix) >= 2 for ix in groups.values()):
            grand = sum(common) / n
            for lab, ix in groups.items():
                gm = sum(common[i] for i in ix) / len(ix)
                for i in ix:
                    between[i] = gm - grand
                    within[i] = common[i] - gm
            # Composite score SD within condition goes from
            # sqrt(v_within + v_unique/k) to sqrt(v_within + v_unique_target/k);
            # scaling the between-condition component by that ratio keeps the
            # observed Cohen's d on the composite exactly where it was.
            v_within = _var(within)
            before = v_within + v_unique / k
            after = v_within + (v_unique * s * s) / k
            if before > 1e-12:
                b_scale = math.sqrt(after / before)
            for i in range(n):
                within[i] += grand      # fold the grand mean back into `within`
        else:
            between = [0.0] * n
            within = list(common)
    rep.between_condition_scale = b_scale

    out: List[List[float]] = []
    for j in range(k):
        col: List[float] = []
        for i in range(n):
            v = (item_means[j] + between[i] * b_scale + within[i] + unique[j][i] * s)
            v = min(scale_max, max(scale_min, v))
            col.append(round(v) if integer_scale else v)
        out.append(col)
    rep.applied = True
    rep.reason = (f"inter-item r {r_before:.3f} -> target {target_r:.3f}: "
                  f"item-specific variance scaled by {s:.2f}")
    return out, rep


def realism_gap(columns: Sequence[Sequence[float]], target_r: float = DEFAULT_TARGET_R) -> float:
    """How far a block's internal consistency is above real-data levels."""
    r = mean_interitem_r([[float(v) for v in c] for c in columns])
    return max(0.0, r - float(target_r))


# =============================================================================
# IDENTICAL-ANSWER RESPONDENTS
# =============================================================================
# In the real reference data, 5.2% of respondents give the SAME answer to all
# five direction-aligned items of a block (7.1% for agreeableness, 3.9% at the
# lowest). These are not careless responders — once items are aligned to the
# construct direction, a careless respondent who ticked one column no longer
# produces a constant row. They are people whose position on the construct is at
# or near the ceiling, who genuinely answer "6, 6, 6, 6, 6".
#
# A continuous latent trait plus noise almost never produces five identical
# integers: the simulator's share was 0.2%. That 25-fold shortfall is visible in
# any careless-responding screen a user runs over simulated data, and it makes
# the data look cleaner than any real sample.

#: Share of respondents giving an identical answer to every item in a block,
#: from the real reference (five 5-item blocks, direction-aligned: 0.039-0.071).
DEFAULT_STRAIGHTLINE_SHARE = 0.052


def straightlined_share(columns: Sequence[Sequence[float]]) -> float:
    k = len(columns)
    if k < 2:
        return 0.0
    n = min(len(c) for c in columns)
    if n == 0:
        return 0.0
    same = 0
    for i in range(n):
        first = columns[0][i]
        if all(columns[j][i] == first for j in range(1, k)):
            same += 1
    return same / n


def match_straightlining(
    columns: Sequence[Sequence[float]],
    scale_min: float,
    scale_max: float,
    target_share: float = DEFAULT_STRAIGHTLINE_SHARE,
    rng: Optional[random.Random] = None,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Bring the share of identical-answer respondents up to the real level.

    The respondents converted are the ones already closest to answering
    identically: smallest within-block spread first, ties broken by distance from
    the nearest scale endpoint, since that is where real constant rows come from.
    Each selected respondent's block collapses to their own rounded block mean, so
    their rank on the construct is unchanged and no treatment effect moves.

    Works in BOTH directions. Raising the share was the original problem: a
    continuous latent trait plus noise almost never produces identical integers.
    Lowering it became a problem once the marginal-shape pass landed, because a
    flat marginal produces constant rows readily — on an 8-item block the engine
    was delivering 3.7% where real mixed-keyed 8-item blocks sit at 0.30%, which
    is a 12x tell in the opposite direction. Excess constant rows are broken by
    nudging ONE item of the row by a single scale point, chosen toward the block
    interior so the row stays plausible and the item's marginal barely moves.

    Never converts more than a quarter of the sample however high `target_share`
    is, and never touches a row it does not have to.
    """
    k = len(columns)
    if k < 2:
        return [list(c) for c in columns], {"applied": False, "reason": "fewer than 2 items"}
    n = min(len(c) for c in columns)
    if n < 20:
        return [list(c) for c in columns], {"applied": False, "reason": "too few respondents"}
    cols = [[float(v) for v in c[:n]] for c in columns]
    current = straightlined_share(cols)
    target = max(0.0, min(0.25, float(target_share)))
    if current > target:
        excess = int(round((current - target) * n))
        if excess <= 0:
            return cols, {"applied": False, "reason": f"already at {current:.3f}",
                          "share_before": round(current, 4)}
        _rng = rng or random.Random(0x57BA)
        constant_rows = [i for i in range(n)
                         if len({cols[j][i] for j in range(k)}) == 1]
        _rng.shuffle(constant_rows)
        broken = 0
        for i in constant_rows[:excess]:
            j = _rng.randrange(k)
            v = cols[j][i]
            # Step toward the middle of the scale when at an endpoint, otherwise
            # either way: a row of 7s becoming 7,7,6,7 reads as a real respondent,
            # a row of 7s becoming 7,7,8,7 is off-scale.
            if v <= float(scale_min):
                cols[j][i] = v + 1.0
            elif v >= float(scale_max):
                cols[j][i] = v - 1.0
            else:
                cols[j][i] = v + (1.0 if _rng.random() < 0.5 else -1.0)
            broken += 1
        return cols, {
            "applied": broken > 0,
            "reason": f"broke {broken} excess identical rows",
            "share_before": round(current, 4),
            "share_after": round(straightlined_share(cols), 4),
            "target_share": round(target, 4),
            "n_broken": broken,
        }
    if current == target:
        return cols, {"applied": False, "reason": f"already at {current:.3f}",
                      "share_before": round(current, 4)}
    need = int(round((target - current) * n))
    if need <= 0:
        return cols, {"applied": False, "reason": "rounding leaves nothing to convert",
                      "share_before": round(current, 4)}

    mid = (float(scale_min) + float(scale_max)) / 2.0
    cand = []
    for i in range(n):
        row = [cols[j][i] for j in range(k)]
        if len(set(row)) == 1:
            continue                      # already constant
        spread = max(row) - min(row)
        m = sum(row) / k
        endpoint_dist = min(abs(m - float(scale_min)), abs(m - float(scale_max)))
        cand.append((spread, endpoint_dist, i))
    cand.sort()
    converted = 0
    for _spread, _dist, i in cand[:need]:
        v = round(sum(cols[j][i] for j in range(k)) / k)
        v = min(float(scale_max), max(float(scale_min), float(v)))
        for j in range(k):
            cols[j][i] = v
        converted += 1
    return cols, {
        "applied": converted > 0,
        "reason": f"converted {converted} respondents to identical answers",
        "share_before": round(current, 4),
        "share_after": round(straightlined_share(cols), 4),
        "target_share": round(target, 4),
        "n_converted": converted,
    }


def target_r_from_alpha(alpha: float, n_items: int) -> float:
    """Mean inter-item correlation implied by a target Cronbach's alpha.

    Inverts the standardised alpha formula, alpha = k*r / (1 + (k-1)*r). Lets the
    engine honour the reliability a user specified for a scale instead of a global
    constant: alpha .75 over 5 items implies r = .375, which is within a hair of
    the .363 observed in the real reference data.
    """
    k = int(n_items)
    a = float(alpha)
    if k < 2:
        return DEFAULT_TARGET_R
    denom = k - a * (k - 1)
    if abs(denom) < 1e-9:
        return DEFAULT_TARGET_R
    r = a / denom
    return float(min(0.95, max(0.05, r)))


# =============================================================================
# MARGINAL SHAPE — maximum-entropy discrete distributions
# =============================================================================
# Real Likert items are not discretised normals. Measured across 17 blocks of four
# published instruments (48,431 respondents, 5-point and 9-point), they are FLAT:
# mean excess kurtosis -0.51 on 5-point attitude scales, with 34% of all responses
# sitting on an endpoint. A discretised normal puts almost nothing on the
# endpoints and is peaked where real data is flat.
#
# The distribution that is flat-to-U-shaped on a bounded support subject only to a
# mean and a variance is the maximum-entropy one, p_k proportional to
# exp(a*k + b*k^2). So rather than inventing a shape, we impose the two moments we
# measured and let the least-committed distribution consistent with them supply
# the rest. That is both principled and, as it happens, what the data looks like.
#
# The measured constraint that travels is dispersion as a FRACTION OF THE SCALE
# SPAN: 0.295 with an SD of 0.019 across 5-point and 9-point instruments alike
# (registry entry item.likert.any.item_sd_fraction_of_span). Endpoint occupancy
# itself does NOT travel across scale lengths (34% on 5-point, 49% on 9-point), so
# it stays scale-specific and is not used here.

#: Measured item SD as a fraction of the scale span. See the registry entry for
#: the derivation, the contributing blocks and the file hashes.
DEFAULT_SD_FRACTION_OF_SPAN = 0.295


def maxent_discrete(lo: int, hi: int, mean: float, sd: float,
                    iters: int = 60) -> List[float]:
    """Probabilities over {lo..hi} with the given mean and SD, otherwise flattest.

    Solved by nested bisection: the mean is monotone in the linear coefficient and
    the variance is monotone in the quadratic one, so each is bracketed and halved
    rather than needing a Jacobian. Returns a uniform distribution if the targets
    are unreachable on this support, which is the right degenerate answer.
    """
    ks = list(range(int(lo), int(hi) + 1))
    n = len(ks)
    if n < 2:
        return [1.0] * n
    mid = 0.5 * (lo + hi)
    span = float(hi - lo)
    # Work on a centred, scaled axis so the exponents stay well conditioned.
    zs = [(k - mid) / span for k in ks]
    z_mean = (float(mean) - mid) / span
    z_sd = float(sd) / span

    def probs(a: float, b: float) -> List[float]:
        ex = [a * z + b * z * z for z in zs]
        m = max(ex)
        w = [math.exp(e - m) for e in ex]
        s = sum(w)
        return [x / s for x in w]

    def moments(a: float, b: float) -> Tuple[float, float]:
        p = probs(a, b)
        mu = sum(pi * z for pi, z in zip(p, zs))
        var = sum(pi * (z - mu) ** 2 for pi, z in zip(p, zs))
        return mu, math.sqrt(max(var, 0.0))

    def solve_a(b: float) -> float:
        lo_a, hi_a = -60.0, 60.0
        for _ in range(iters):
            a = 0.5 * (lo_a + hi_a)
            mu, _ = moments(a, b)
            if mu < z_mean:
                lo_a = a
            else:
                hi_a = a
        return 0.5 * (lo_a + hi_a)

    lo_b, hi_b = -60.0, 60.0
    for _ in range(iters):
        b = 0.5 * (lo_b + hi_b)
        _, s = moments(solve_a(b), b)
        if s < z_sd:
            lo_b = b          # more positive b pushes mass to the edges
        else:
            hi_b = b
    b = 0.5 * (lo_b + hi_b)
    return probs(solve_a(b), b)


def _quantile_targets(probs: Sequence[float], lo: int, n: int) -> List[float]:
    """n sorted values drawn deterministically from a discrete distribution.

    Deterministic rather than sampled, so the transform adds no variance of its own
    and two runs of the same design differ only where the engine made them differ.
    """
    out: List[float] = []
    cum = 0.0
    k = 0
    for i in range(n):
        q = (i + 0.5) / n
        while k < len(probs) - 1 and cum + probs[k] < q:
            cum += probs[k]
            k += 1
        out.append(float(lo + k))
    return out


def _rank_map(values: Sequence[float], targets: Sequence[float],
              rng: Optional[random.Random] = None) -> List[float]:
    """Rank-transport `values` onto sorted `targets`, breaking ties at random.

    The random tie-break is not a detail. Likert values take 5 to 9 distinct levels,
    so most of a column is tied, and any deterministic tie-break orders the tied
    respondents by whatever the fallback key is. Breaking ties by row index orders
    them by position in the frame — which, when the frame is built condition by
    condition, is the condition itself. That hands every low target to the first
    condition and every high target to the second and MANUFACTURES an effect out of
    a column that had none: measured at d = 0.06 before and d = 1.65 after, on data
    generated with no manipulation at all. Ties must be broken by noise that is
    independent of the design.
    """
    r = rng or random.Random(0xA17E)
    jitter = [r.random() for _ in values]
    order = sorted(range(len(values)), key=lambda i: (values[i], jitter[i]))
    out = [0.0] * len(values)
    for rank, idx in enumerate(order):
        out[idx] = float(targets[rank])
    return out


@dataclass
class DispersionReport:
    adjusted_items: int = 0
    sd_before: float = 0.0
    sd_after: float = 0.0
    target_sd: float = 0.0
    endpoint_before: float = 0.0
    endpoint_after: float = 0.0
    skipped: str = ""


def match_item_dispersion(
    columns: Sequence[Sequence[float]],
    scale_min: int,
    scale_max: int,
    condition_labels: Optional[Sequence[Any]] = None,
    sd_fraction: float = DEFAULT_SD_FRACTION_OF_SPAN,
    rng: Optional[random.Random] = None,
) -> Tuple[List[List[int]], DispersionReport]:
    """Give each item the measured marginal shape without moving its mean.

    Each item column is rank-transported onto a maximum-entropy distribution with
    the item's OWN current mean and a target SD. Rank transport means every
    participant keeps their position in the column, so condition differences,
    persona correlations and inter-item structure all survive; only the marginal
    changes.

    `preserve the effect`: the target is the dispersion a single condition should
    show, so the pooled target is widened by whatever between-condition variance
    the column already carries. Without that, a column split by a strong
    manipulation would be squeezed back toward a single-group spread and the
    manipulation would be partly transported away.
    """
    rep = DispersionReport()
    cols = [list(map(float, c)) for c in columns]

    def _untouched():
        """Hand the input straight back on any skip.

        Not rounded: a column that was skipped BECAUSE it holds NaN cannot be cast
        to int, and the caller must not write back a skipped block anyway — it
        checks `rep.skipped` first.
        """
        return [list(c) for c in columns], rep

    if not cols or not cols[0]:
        rep.skipped = "empty"
        return _untouched()
    n = len(cols[0])
    span = float(scale_max) - float(scale_min)
    if span <= 0 or n < 20:
        rep.skipped = "degenerate scale or too few rows"
        return _untouched()
    if any(len(c) != n for c in cols):
        rep.skipped = "ragged columns"
        return _untouched()
    if any(any(v != v for v in c) for c in cols):
        rep.skipped = "missing values present"
        return _untouched()

    target_within = sd_fraction * span
    rep.target_sd = target_within

    def _sd(xs: Sequence[float]) -> float:
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))

    def _endpoint_share(cs: Sequence[Sequence[float]]) -> float:
        tot = sum(len(c) for c in cs) or 1
        hits = sum(1 for c in cs for v in c
                   if abs(v - scale_min) < 1e-9 or abs(v - scale_max) < 1e-9)
        return hits / tot

    _rng = rng or random.Random(0xA17E)
    rep.sd_before = sum(_sd(c) for c in cols) / len(cols)
    rep.endpoint_before = _endpoint_share(cols)

    groups: Optional[Dict[Any, List[int]]] = None
    if condition_labels is not None and len(condition_labels) == n:
        groups = {}
        for i, lab in enumerate(condition_labels):
            groups.setdefault(lab, []).append(i)
        if len(groups) < 2:
            groups = None

    out: List[List[int]] = []
    for col in cols:
        mean = sum(col) / n
        target = target_within
        if groups:
            # Between-condition variance already in this column, added back so the
            # manipulation is not squeezed out along with the excess concentration.
            between = sum(len(ix) * ((sum(col[i] for i in ix) / len(ix)) - mean) ** 2
                          for ix in groups.values()) / n
            target = math.sqrt(target_within ** 2 + between)
        target = max(0.05 * span, min(0.6 * span, target))
        probs = maxent_discrete(int(scale_min), int(scale_max), mean, target)
        targets = _quantile_targets(probs, int(scale_min), n)
        out.append([int(round(v)) for v in _rank_map(col, targets, _rng)])
        rep.adjusted_items += 1

    rep.sd_after = sum(_sd([float(v) for v in c]) for c in out) / len(out)
    rep.endpoint_after = _endpoint_share([[float(v) for v in c] for c in out])
    return out, rep
