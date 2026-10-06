"""
Empirical Marginals — distribution SHAPES that make synthetic data pass for real.
================================================================================

The mean of a simulated variable is the easy part. What gives synthetic survey and
experiment data away is the *shape* of its marginal distribution:

* Real dictator-game giving is not a bell curve around 28%. It is a spike of exact
  zeros (~36%), a spike at exactly half (~17%), a smaller spike at everything, and
  round-number heaping in between. A Gaussian with mean 0.28 produces almost no
  exact zeros and no spike at half, and is distinguishable from real data by eye.
* Real slider and open-numeric answers heap on multiples of 5 and 10. Continuous
  draws rounded to integers do not.
* Real Likert data has characteristic endpoint and midpoint masses that differ by
  scale length and by construct.

This module supplies those shapes, and applies them by RANK TRANSPORT: the engine's
own value for each participant decides that participant's *rank*, and the rank is
mapped onto the empirical marginal. Condition effects, persona correlations and any
other structure the engine built are rank-preserving, so they survive the transform
intact while the marginal becomes the empirical one.

Every marginal here is gated on verification: `marginal_for()` returns None unless
the shape is backed by a provenance record, so an unsourced shape never silently
reshapes a user's data. See `empirical_registry.py`.
"""
from __future__ import annotations

__version__ = "1.0.0"

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import random

try:
    from .empirical_registry import (
        status_of, provenance_of, VERIFIED, CORRECTED, PARTIAL,
    )
except Exception:  # pragma: no cover
    try:
        from empirical_registry import (  # type: ignore
            status_of, provenance_of, VERIFIED, CORRECTED, PARTIAL,
        )
    except Exception:
        def status_of(kind: str, key: str) -> str:  # type: ignore
            return "unverified"

        def provenance_of(kind: str, key: str):  # type: ignore
            return None
        VERIFIED, CORRECTED, PARTIAL = "verified", "corrected", "partial"

_SOURCED = {VERIFIED, CORRECTED, PARTIAL}


# =============================================================================
# ROUND-NUMBER HEAPING
# =============================================================================
# Humans answering a wide numeric or slider question overwhelmingly pick multiples
# of 5 and 10. The strength of the pull is a property of the response format, not
# of the construct, so it is applied to any numeric DV with a wide range.

def heaping_grid(scale_min: float, scale_max: float) -> List[float]:
    """Round-number grid a respondent is likely to pick on this scale."""
    span = float(scale_max) - float(scale_min)
    if span <= 0:
        return []
    if span >= 50:
        step = span / 20.0          # 0-100 -> every 5
    elif span >= 20:
        step = span / 10.0
    else:
        return []                    # short scales are already discrete
    grid = []
    x = float(scale_min)
    while x <= scale_max + 1e-9:
        grid.append(round(x, 6))
        x += step
    return grid


def snap_to_round(
    value: float,
    scale_min: float,
    scale_max: float,
    strength: float,
    rng: random.Random,
) -> float:
    """Pull `value` onto a round number with probability `strength`.

    `strength` is the proportion of respondents who answer in round numbers.
    Values already on the grid are left alone. The pull is to the NEAREST grid
    point, so the transform is monotone and cannot reorder participants by more
    than one grid cell — rank structure, and therefore treatment effects, survive.
    """
    grid = heaping_grid(scale_min, scale_max)
    if not grid or strength <= 0:
        return value
    if rng.random() >= strength:
        return value
    nearest = min(grid, key=lambda g: abs(g - value))
    # Coarser heaping for the stronger heapers: a tenth of them go to a multiple
    # of ten rather than of five.
    if rng.random() < 0.35:
        span = scale_max - scale_min
        coarse = [g for g in grid if abs(((g - scale_min) / span) * 10.0 -
                                        round(((g - scale_min) / span) * 10.0)) < 1e-6]
        if coarse:
            nearest = min(coarse, key=lambda g: abs(g - value))
    return float(nearest)


# =============================================================================
# POINT-MASS MIXTURE
# =============================================================================

@dataclass
class PointMassMixture:
    """A marginal on [0, 1] with exact point masses plus a decaying remainder.

    This is the canonical shape of allocation data in economic games:
      * `point_masses` — the spikes the literature reports (e.g. 0.0 at 36%,
        0.5 at 17%, 1.0 at 5%).
      * the remaining probability is spread over a round-number grid with an
        exponential decay, because allocations cluster on round fractions and
        smaller gifts are more common than larger ones.

    The decay rate is SOLVED so that the mixture's mean equals the published mean.
    That makes the published mean and the published point masses hold
    simultaneously, instead of one being sacrificed to the other.
    """
    key: str                                  # knowledge-base key, for provenance
    mean: float                               # published mean proportion
    point_masses: Tuple[Tuple[float, float], ...] = ()
    grid_step: float = 0.05
    #: Proportion of the remainder that lands exactly on the round grid rather than
    #: being jittered off it. Real data is heaped but not perfectly so.
    grid_adherence: float = 0.80
    source: str = ""
    notes: str = ""
    _lambda: Optional[float] = field(default=None, repr=False)

    # ---- construction -------------------------------------------------------

    def _grid(self) -> List[float]:
        """Round-number support for the remainder, excluding the point masses."""
        pts = {round(p, 6) for p, _ in self.point_masses}
        out = []
        steps = int(round(1.0 / self.grid_step))
        for i in range(0, steps + 1):
            x = round(i * self.grid_step, 6)
            if x not in pts:
                out.append(x)
        return out

    def _remainder_mass(self) -> float:
        return max(0.0, 1.0 - sum(p for _, p in self.point_masses))

    def _required_remainder_mean(self) -> Optional[float]:
        """Mean the remainder must have for the mixture to hit `self.mean`."""
        rm = self._remainder_mass()
        if rm <= 1e-9:
            return None
        fixed = sum(v * p for v, p in self.point_masses)
        return (self.mean - fixed) / rm

    def _weights(self, lam: float) -> List[float]:
        grid = self._grid()
        w = [math.exp(-lam * x) for x in grid]
        s = sum(w)
        return [x / s for x in w] if s > 0 else [1.0 / len(grid)] * len(grid)

    def _mean_at(self, lam: float) -> float:
        grid = self._grid()
        return sum(x * w for x, w in zip(grid, self._weights(lam)))

    def solve_lambda(self) -> float:
        """Find the decay rate that makes the mixture reproduce `self.mean`.

        Monotone in lam (larger lam -> more mass on small allocations -> smaller
        mean), so a bisection is exact to within tolerance and cannot fail to
        converge. Returns 0.0 when the remainder is empty or the target is
        unreachable, in which case the mixture matches the point masses and the
        mean is as close as the point masses allow.
        """
        if self._lambda is not None:
            return self._lambda
        target = self._required_remainder_mean()
        if target is None:
            self._lambda = 0.0
            return 0.0
        lo, hi = -40.0, 40.0         # lam < 0 shifts mass UP, for high-mean games
        if self._mean_at(lo) < target or self._mean_at(hi) > target:
            # Target outside the reachable range: clamp to the closest end.
            self._lambda = lo if self._mean_at(lo) < target else hi
            return self._lambda
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if self._mean_at(mid) > target:
                lo = mid
            else:
                hi = mid
        self._lambda = (lo + hi) / 2.0
        return self._lambda

    # ---- sampling -----------------------------------------------------------

    def sample(self, rng: random.Random) -> float:
        u = rng.random()
        cum = 0.0
        for v, p in self.point_masses:
            cum += p
            if u < cum:
                return float(v)
        grid = self._grid()
        weights = self._weights(self.solve_lambda())
        r = rng.random()
        c = 0.0
        pick = grid[-1]
        for x, w in zip(grid, weights):
            c += w
            if r < c:
                pick = x
                break
        if rng.random() > self.grid_adherence:
            # Off-grid answer: jitter within half a grid step, staying in [0, 1].
            pick = min(1.0, max(0.0, pick + rng.uniform(-self.grid_step / 2.0,
                                                        self.grid_step / 2.0)))
        return float(pick)

    def sample_sorted(self, n: int, rng: random.Random) -> List[float]:
        """n draws, ascending — the target sequence for rank transport."""
        return sorted(self.sample(rng) for _ in range(max(0, int(n))))

    def theoretical_mean(self) -> float:
        fixed = sum(v * p for v, p in self.point_masses)
        rm = self._remainder_mass()
        return fixed + rm * self._mean_at(self.solve_lambda())

    def mass_at(self, value: float) -> float:
        for v, p in self.point_masses:
            if abs(v - value) < 1e-9:
                return p
        return 0.0


# =============================================================================
# GAME MARGINALS
# =============================================================================
# Populated from the verification pass (see empirical_marginals_data.py). A game is
# only reshaped when its marginal carries a sourced provenance record, so an
# unverified shape never touches user data.

GAME_MARGINALS: Dict[str, PointMassMixture] = {}


def register_marginal(mix: PointMassMixture) -> None:
    GAME_MARGINALS[mix.key] = mix


def marginal_for(game_key: str) -> Optional[PointMassMixture]:
    """Verified marginal for a knowledge-base game key, or None."""
    mix = GAME_MARGINALS.get(game_key)
    if mix is None:
        return None
    if status_of("game", game_key) not in _SOURCED:
        return None
    return mix


def available_marginals() -> List[Dict[str, Any]]:
    """Audit view: every registered marginal, whether it is active, and why."""
    out = []
    for key, mix in sorted(GAME_MARGINALS.items()):
        st = status_of("game", key)
        out.append({
            "key": key,
            "status": st,
            "active": st in _SOURCED,
            "published_mean": mix.mean,
            "reproduced_mean": round(mix.theoretical_mean(), 4),
            "point_masses": list(mix.point_masses),
            "source": mix.source,
        })
    return out


# =============================================================================
# RANK TRANSPORT
# =============================================================================

def rank_transport(
    values: Sequence[float],
    targets: Sequence[float],
) -> List[float]:
    """Map `values` onto the sorted `targets` by rank.

    The participant with the k-th smallest value receives the k-th smallest target.
    Ties are broken stably, so the transform is deterministic given the inputs.
    Requires len(targets) == len(values).
    """
    n = len(values)
    if n == 0 or len(targets) != n:
        return list(values)
    order = sorted(range(n), key=lambda i: (values[i], i))
    out = [0.0] * n
    for rank, idx in enumerate(order):
        out[idx] = float(targets[rank])
    return out


def reshape_allocation_column(
    values: Sequence[float],
    game_key: str,
    scale_min: float,
    scale_max: float,
    rng: random.Random,
) -> Optional[List[float]]:
    """Reshape a column of allocation values onto the verified game marginal.

    Returns None when the game has no verified marginal, when the column is too
    short for a stable marginal, or when the scale is not an allocation scale —
    in every one of those cases the caller must leave the data untouched.
    """
    mix = marginal_for(game_key)
    if mix is None:
        return None
    n = len(values)
    if n < 20:                      # too few rows for point masses to mean anything
        return None
    span = float(scale_max) - float(scale_min)
    if span <= 0:
        return None
    proportions = mix.sample_sorted(n, rng)
    targets = [float(scale_min) + p * span for p in proportions]
    return rank_transport(values, targets)
