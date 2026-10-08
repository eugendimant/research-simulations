"""Within-subjects and mixed (repeated-measures) designs.

Every other part of the simulator draws ONE condition per participant. This module adds the two
designs in which a participant answers the same measures more than once:

* ``within``  - every participant experiences every level of the within factor(s);
* ``mixed``   - between-subject factor(s) crossed with within-subject factor(s)
                (treatment/control x pre/post, or Time 1..3).

How it works (and why it is built this way)
-------------------------------------------
The between-subjects engine is left untouched. A repeated-measures run asks it for ONE pass over
ONE set of participants in which every measure is present once per within-level
(``Attitude_occ1``, ``Attitude_occ2`` ...), with the cross-measure correlation matrix set so that the
same construct is correlated across occasions. That single pass already gives, for free and
consistently for every column: one persona, one set of demographics, one attention check, one
careless-responding style per person, the scale-reliability targets of every block, the
open-ended text and the quality flags. This module then

1. builds the occasion-by-occasion correlation structure (``within_correlation``, default 0.5),
2. tops the realised correlation up to the target when the engine's own coupling falls short,
3. builds the requested effects into the finished answers as WITHIN-PERSON shifts, in units of the
   measure's own SD, so that the requested d is d_av (see ``EFFECT DEFINITION``),
4. assigns a presentation order (counterbalancing) and adds a small, documented order drift,
5. applies attrition (a participant who drops out loses the LATER conditions),
6. names the columns ``<DV>_<condition>_<i>`` / ``<DV>_<condition>_mean`` and can write a long file.

EFFECT DEFINITION
-----------------
The ``cohens_d`` of an effect on a within factor is **d_av**: the mean difference divided by the
average of the two conditions' SDs. That is the quantity that is comparable to a between-subjects d
(Lakens 2013). The paired d_z (mean difference / SD of the differences) is reported next to it; the two
are linked by ``d_z = d_av / sqrt(2 (1 - r))`` with r the within-person correlation, so a larger r gives
a larger d_z and a more powerful paired test for the same d_av.

Effects on a between factor of a mixed design are ordinary between-group d's. By default they are
present at every level of the within factor except a baseline first level ("Pre", "T1", "Baseline",
"Wave 1"), where randomised groups do not differ; a spec may carry an ``at`` mapping to say where it
applies, and an ``interaction`` such as Treatment x Time is written as a between-factor effect that is
restricted to one within-level (``at={"Time": "Post"}``).

Order and fatigue
-----------------
Counterbalancing (``random``, ``latin_square``, ``full``, ``fixed``) is recorded in the ``Order``
column. ``order_effect_d`` (default -0.05 SD per later presentation position, i.e. mild fatigue; set
``order_effects=False`` to switch it off) is a modest drift of the kind described for repeated ratings
(practice and fatigue effects are usually small relative to the manipulation; Greenwald 1976, Poulton
& Edwards 1979). It is recalled from that literature, not fitted to data; it exists so that ``Order``
and ``Position_*`` can be used as covariates.
"""
from __future__ import annotations

import itertools
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "DESIGN_TYPES", "ORDER_SCHEMES", "DEFAULT_WITHIN_CORRELATION", "DEFAULT_ORDER_EFFECT_D",
    "DesignSpec", "normalize_design", "build_orders", "generate_repeated", "build_long_format",
    "observed_contrasts", "design_from_metadata", "wide_columns_for", "suggest_design_from_blocks",
    "suggest_design_from_qsf",
]

DESIGN_TYPES = ("between", "within", "mixed")
ORDER_SCHEMES = ("random", "latin_square", "full", "fixed")
CORRELATION_STRUCTURES = ("exchangeable", "ar1")
DEFAULT_WITHIN_CORRELATION = 0.5
DEFAULT_ORDER_EFFECT_D = -0.05
ALL_PARTICIPANTS_LABEL = "All participants"

# the engine's cross-scale coupling attenuates a correlation between single items more than one between
# composites (measured at N = 1,500: a latent 0.5 became 0.48 on a 4-item scale and 0.35 on one item)
_SINGLE_ITEM_COUPLING = 0.72
_TWO_ITEM_COUPLING = 0.88
_COMPOSITE_COUPLING = 0.97

_BASELINE_LABEL_RE = re.compile(
    r"\b(?:pre|pretest|before|baseline|t0|t1|time ?[01]|wave ?[01]|week ?0|session ?1)\b")
# scale types whose finished answers are not a plain rating that can be moved by a constant
_UNSHIFTABLE_TYPES = frozenset({"rank_order", "ranking", "constant_sum", "best_worst", "paired_comparison",
                                "hot_spot", "heatmap"})


# --------------------------------------------------------------------------------------
# design description
# --------------------------------------------------------------------------------------
def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "").strip().lower())


def _slug(label: Any) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_")
    return s or "level"


@dataclass
class DesignSpec:
    """Normalised description of a repeated-measures design."""

    type: str = "between"
    within_factors: List[Dict[str, Any]] = field(default_factory=list)
    between_factors: List[Dict[str, Any]] = field(default_factory=list)
    cells: List[Dict[str, Any]] = field(default_factory=list)       # label, slug, levels{factor: level}
    order: str = "random"
    fixed_order: Optional[List[str]] = None
    within_correlation: float = DEFAULT_WITHIN_CORRELATION
    correlation_structure: str = "exchangeable"
    order_effects: bool = True
    order_effect_d: float = DEFAULT_ORDER_EFFECT_D
    long_format: bool = True
    simple_effects: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def is_repeated(self) -> bool:
        return self.type in ("within", "mixed") and len(self.cells) >= 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "within_factors": [dict(f) for f in self.within_factors],
            "between_factors": [dict(f) for f in self.between_factors],
            "cells": [dict(c) for c in self.cells],
            "order": self.order,
            "fixed_order": list(self.fixed_order) if self.fixed_order else None,
            "within_correlation": float(self.within_correlation),
            "correlation_structure": self.correlation_structure,
            "order_effects": bool(self.order_effects),
            "order_effect_d": float(self.order_effect_d),
            "long_format": bool(self.long_format),
            "simple_effects": [dict(s) for s in self.simple_effects],
            "notes": list(self.notes),
        }


def _clean_levels(levels: Any) -> List[str]:
    if isinstance(levels, str):
        levels = [x for x in re.split(r"[,;\n]", levels)]
    out: List[str] = []
    for lv in levels or []:
        s = re.sub(r"\s+", " ", str(lv).replace("\xa0", " ")).strip()
        if s and s not in out:
            out.append(s)
    return out


def _match_levels(label: str, factors: Sequence[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Level of every factor named inside a condition label (whole words, longest level wins)."""
    text = " " + _norm(re.sub(r"[_\-/|x×:,]+", " ", str(label))) + " "
    # the "x" separator must not eat levels such as "x"; levels are matched first on the raw text
    raw = " " + _norm(label) + " "
    found: Dict[str, str] = {}
    for factor in factors:
        best: Optional[Tuple[int, str]] = None
        for lv in factor.get("levels") or []:
            n = _norm(lv)
            if not n:
                continue
            pat = r"(?<![^\W_])" + re.escape(n) + r"(?![^\W_])"
            if re.search(pat, raw) or re.search(pat, text):
                if best is None or len(n) > best[0]:
                    best = (len(n), lv)
        if best is None:
            return None
        found[str(factor["name"])] = best[1]
    return found


def _make_cells(within_factors: List[Dict[str, Any]], labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    if labels:
        for lab in labels:
            lv = _match_levels(lab, within_factors)
            if lv is None:
                return []
            cells.append({"label": str(lab), "levels": lv})
    else:
        names = [f["name"] for f in within_factors]
        for combo in itertools.product(*[f["levels"] for f in within_factors]):
            lab = combo[0] if len(combo) == 1 else " x ".join(combo)
            cells.append({"label": lab, "levels": dict(zip(names, combo))})
    used: Dict[str, int] = {}
    for c in cells:
        base = _slug(c["label"])
        used[base] = used.get(base, 0) + 1
        c["slug"] = base if used[base] == 1 else f"{base}_{used[base]}"
    return cells


def normalize_design(
    design: Any,
    factors: Optional[List[Dict[str, Any]]] = None,
    conditions: Optional[List[str]] = None,
) -> Optional[DesignSpec]:
    """Return a :class:`DesignSpec` for ``within``/``mixed`` designs, else ``None`` (between-subjects).

    ``design`` is a dict (or just the string ``"within"``/``"mixed"``):

    ``type``                   "between" | "within" | "mixed"
    ``within_factors``         [{"name": "Time", "levels": ["Pre", "Post"]}]; for ``within`` it is optional
                               (the engine's ``factors``/``conditions`` are then the within structure);
                               required for ``mixed`` (the engine's conditions are the between groups)
    ``order``                  "random" (default) | "latin_square" | "full" | "fixed"
    ``fixed_order``            list of cell labels, for ``order="fixed"``
    ``within_correlation``     r of the same measure across conditions (default 0.5)
    ``correlation_structure``  "exchangeable" (default) | "ar1" (r**|lag|, for time points)
    ``order_effects``          True (default) / False; ``order_effect_d`` default -0.05 SD per position
    ``simple_effects``         extra effects restricted to one part of the design, see ``generate_repeated``
    ``long_format``            True (default): also build the participant x condition table
    """
    if design is None:
        return None
    if isinstance(design, str):
        design = {"type": design}
    if not isinstance(design, dict):
        return None
    dtype = str(design.get("type") or design.get("design_type") or "between").strip().lower()
    dtype = {"within-subjects": "within", "repeated": "within", "repeated_measures": "within",
             "between-subjects": "between"}.get(dtype, dtype)
    if dtype not in DESIGN_TYPES:
        raise ValueError(f"design type must be one of {DESIGN_TYPES}, not {dtype!r}")
    if dtype == "between":
        return None

    conditions = [str(c) for c in (conditions or [])]
    factors = [dict(f) for f in (factors or []) if isinstance(f, dict)]
    spec = DesignSpec(type=dtype)

    raw_within = design.get("within_factors")
    within_factors: List[Dict[str, Any]] = []
    if raw_within:
        for wf in raw_within:
            if isinstance(wf, dict):
                name = str(wf.get("name") or "Condition").strip() or "Condition"
                levels = _clean_levels(wf.get("levels"))
                if len(levels) >= 2:
                    within_factors.append({"name": name, "levels": levels})
    if dtype == "mixed":
        if not within_factors:
            raise ValueError("A mixed design needs at least one within-subject factor with two or more levels "
                             "(design['within_factors'] = [{'name': 'Time', 'levels': ['Pre', 'Post']}]).")
        spec.within_factors = within_factors
        spec.between_factors = [dict(f) for f in factors]
        spec.cells = _make_cells(within_factors)
    else:
        if within_factors:
            spec.within_factors = within_factors
            spec.cells = _make_cells(within_factors)
        else:
            spec.cells = []
            real = [f for f in factors if _clean_levels(f.get("levels"))]
            if len(real) > 1:
                wf = [{"name": str(f["name"]), "levels": _clean_levels(f["levels"])} for f in real]
                spec.cells = _make_cells(wf, conditions or None)
                if spec.cells:
                    spec.within_factors = wf
            if not spec.cells:
                levels = _clean_levels(conditions)
                spec.within_factors = [{"name": str((real[0]["name"] if len(real) == 1 else "Condition")),
                                        "levels": levels}]
                spec.cells = _make_cells(spec.within_factors)
                if len(real) > 1:
                    spec.notes.append("The condition labels could not be read as a crossing of the factors, so the "
                                      "conditions are treated as the levels of one within-subject factor.")
    if len(spec.cells) < 2:
        raise ValueError("A within-subjects design needs at least two conditions that every participant sees.")

    order = str(design.get("order") or design.get("counterbalancing") or "random").strip().lower().replace(" ", "_")
    order = {"latin": "latin_square", "latin-square": "latin_square", "none": "fixed", "no": "fixed"}.get(order, order)
    spec.order = order if order in ORDER_SCHEMES else "random"
    fo = design.get("fixed_order")
    if fo:
        spec.fixed_order = [str(x) for x in fo]
    try:
        r = float(design.get("within_correlation", DEFAULT_WITHIN_CORRELATION))
    except (TypeError, ValueError):
        r = DEFAULT_WITHIN_CORRELATION
    spec.within_correlation = float(min(0.95, max(0.0, r if math.isfinite(r) else DEFAULT_WITHIN_CORRELATION)))
    cs = str(design.get("correlation_structure") or "exchangeable").strip().lower()
    spec.correlation_structure = cs if cs in CORRELATION_STRUCTURES else "exchangeable"
    spec.order_effects = bool(design.get("order_effects", True))
    try:
        od = float(design.get("order_effect_d", DEFAULT_ORDER_EFFECT_D))
    except (TypeError, ValueError):
        od = DEFAULT_ORDER_EFFECT_D
    spec.order_effect_d = float(min(0.5, max(-0.5, od if math.isfinite(od) else DEFAULT_ORDER_EFFECT_D)))
    spec.long_format = bool(design.get("long_format", True))
    spec.simple_effects = [dict(s) for s in (design.get("simple_effects") or []) if isinstance(s, dict)]
    return spec


def design_from_metadata(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The ``design`` block of a finished run's metadata when it is a within/mixed design, else None."""
    d = (metadata or {}).get("design") if isinstance(metadata, dict) else None
    if isinstance(d, dict) and d.get("type") in ("within", "mixed") and d.get("cells"):
        return d
    return None


# --------------------------------------------------------------------------------------
# counterbalancing
# --------------------------------------------------------------------------------------
def _williams_rows(k: int) -> List[List[int]]:
    """Rows of a balanced Latin square (each condition once per position, each ordered pair once)."""
    first = [0]
    lo, hi = 1, k - 1
    take_low = True
    while len(first) < k:
        if take_low:
            first.append(lo)
            lo += 1
        else:
            first.append(hi)
            hi -= 1
        take_low = not take_low
    rows = [[(x + s) % k for x in first] for s in range(k)]
    if k % 2:
        rows += [list(reversed(r)) for r in rows]
    return rows


def build_orders(k: int, n: int, scheme: str, rng: np.random.RandomState,
                 fixed: Optional[Sequence[int]] = None,
                 groups: Optional[np.ndarray] = None) -> Tuple[np.ndarray, str]:
    """Presentation order of the ``k`` conditions for ``n`` participants.

    Returns ``(orders, effective_scheme)``; ``orders[i]`` lists condition indices in the order person i
    saw them. Orders are balanced within each between-group (``groups``) so a mixed design is
    counterbalanced inside every group.
    """
    scheme = scheme if scheme in ORDER_SCHEMES else "random"
    eff = scheme
    if scheme == "fixed" or k < 2:
        base = list(fixed) if fixed is not None and sorted(fixed) == list(range(k)) else list(range(k))
        return np.tile(np.array(base, dtype=int), (n, 1)), "fixed"
    if scheme == "full" and k > 5:
        eff = "latin_square"
    if scheme == "random":
        out = np.zeros((n, k), dtype=int)
        for i in range(n):
            out[i] = rng.permutation(k)
        return out, "random"
    pool = ([list(p) for p in itertools.permutations(range(k))] if eff == "full" else _williams_rows(k))
    out = np.zeros((n, k), dtype=int)
    gidx = np.zeros(n, dtype=int) if groups is None else np.asarray(groups, dtype=int)
    for g in np.unique(gidx):
        members = np.where(gidx == g)[0]
        m = len(members)
        reps = int(math.ceil(m / len(pool)))
        take = np.tile(np.arange(len(pool)), reps)[:m]
        take = rng.permutation(take)
        for pos, who in enumerate(members):
            out[who] = pool[int(take[pos])]
    return out, eff


# --------------------------------------------------------------------------------------
# effects: cell means in units of the measure's SD
# --------------------------------------------------------------------------------------
def _is_baseline_first_level(spec: DesignSpec) -> bool:
    if not spec.cells:
        return False
    first = _norm(re.sub(r"[_\-/.:]+", " ", str(spec.cells[0]["label"])))
    return bool(_BASELINE_LABEL_RE.search(first))


def _scope_ok(at: Any, cell_levels: Dict[str, str], group_levels: Dict[str, str]) -> bool:
    if not at:
        return True
    if not isinstance(at, dict):
        return True
    for key, want in at.items():
        wants = want if isinstance(want, (list, tuple, set)) else [want]
        wants_n = {_norm(w) for w in wants}
        have = None
        for src in (cell_levels, group_levels):
            for k2, v2 in src.items():
                if _norm(k2) == _norm(key):
                    have = v2
        if have is None:
            continue
        if _norm(have) not in wants_n:
            return False
    return True


def _spec_value(spec: Any, key: str, default: Any = None) -> Any:
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)


def _solve_cell_offsets(
    engine: Any, design: DesignSpec, variable: str, specs: List[Any],
    group_labels: List[str], group_levels: List[Dict[str, str]],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Least-squares cell means (G x K, in SD units) that satisfy every spec that names ``variable``.

    Each spec is a set of linear contrasts between cells; the minimum-norm solution splits a lone
    contrast symmetrically (+d/2, -d/2) exactly like the between-subjects route, effects on different
    factors add, and several specs on one factor are combined by least squares (so A-B = .5 and
    B-C = .5 give A-C = 1).
    """
    from . import enhanced_simulation_engine as E

    G, K = len(group_labels), len(design.cells)
    rows: List[np.ndarray] = []
    rhs: List[float] = []
    report: List[Dict[str, Any]] = []
    within_names = {_norm(f["name"]): f for f in design.within_factors}
    skip_baseline = design.type == "mixed" and _is_baseline_first_level(design)

    for sp in specs:
        high = str(_spec_value(sp, "level_high", ""))
        low = str(_spec_value(sp, "level_low", ""))
        try:
            d = abs(float(_spec_value(sp, "cohens_d", 0.5)))
        except (TypeError, ValueError):
            d = 0.5
        if not math.isfinite(d):
            d = 0.5
        d = min(d, 3.0)
        if str(_spec_value(sp, "direction", "positive")).lower().strip() == "negative":
            d = -d
        at = _spec_value(sp, "at", None)
        factor_n = _norm(_spec_value(sp, "factor", ""))
        wf = within_names.get(factor_n)
        if wf is None:
            for f in design.within_factors:
                lv = {_norm(x) for x in f["levels"]}
                if _norm(high) in lv and _norm(low) in lv:
                    wf = f
                    break
        n_rows = 0
        if wf is not None:
            for g in range(G):
                for ci, c in enumerate(design.cells):
                    if _norm(c["levels"].get(wf["name"])) != _norm(high):
                        continue
                    if not _scope_ok(at, c["levels"], group_levels[g]):
                        continue
                    partner = None
                    for cj, c2 in enumerate(design.cells):
                        if _norm(c2["levels"].get(wf["name"])) != _norm(low):
                            continue
                        if all(_norm(c2["levels"].get(k)) == _norm(v) for k, v in c["levels"].items()
                               if k != wf["name"]):
                            partner = cj
                            break
                    if partner is None:
                        continue
                    row = np.zeros(G * K)
                    row[g * K + ci] += 1.0
                    row[g * K + partner] -= 1.0
                    rows.append(row)
                    rhs.append(d)
                    n_rows += 1
        else:
            hi_g, lo_g = [], []
            for g, lab in enumerate(group_labels):
                side = E._spec_side(sp, E._label_norm(lab))
                if side > 0:
                    hi_g.append(g)
                elif side < 0:
                    lo_g.append(g)
            if hi_g and lo_g:
                for ci, c in enumerate(design.cells):
                    if at is None and skip_baseline and ci == 0:
                        continue
                    if not _scope_ok(at, c["levels"], {}):
                        continue
                    row = np.zeros(G * K)
                    for g in hi_g:
                        row[g * K + ci] += 1.0 / len(hi_g)
                    for g in lo_g:
                        row[g * K + ci] -= 1.0 / len(lo_g)
                    rows.append(row)
                    rhs.append(d)
                    n_rows += 1
        report.append({
            "variable": variable, "factor": str(_spec_value(sp, "factor", "")),
            "level_high": high, "level_low": low, "intended_d": float(abs(d)) * (1 if d >= 0 else -1),
            "scope": at, "kind": "within" if wf is not None else "between",
            "matched": bool(n_rows), "status": "applied" if n_rows else "levels_not_found",
        })
    if not rows:
        return np.zeros((G, K)), report
    sol = np.linalg.lstsq(np.vstack(rows), np.asarray(rhs), rcond=None)[0]
    return sol.reshape(G, K), report


def _inferred_cell_offsets(engine: Any, design: DesignSpec, variable: str,
                           group_labels: List[str], n_items: int, smin: int, smax: int) -> np.ndarray:
    """Literature/keyword effects read from the condition names, in SD units; reference level = zero."""
    G, K = len(group_labels), len(design.cells)
    out = np.zeros((G, K))
    if not getattr(engine, "auto_effects", True):
        return out
    try:
        engine._scale_effect_meta = dict(getattr(engine, "_scale_effect_meta", {}) or {})
        engine._scale_effect_meta[str(variable)] = (int(n_items), int(smin), int(smax))
        unit = engine._EFFECT_D_TO_NORMALIZED * engine._explicit_effect_scale(variable)
        if not unit:
            return out
    except Exception:  # noqa: BLE001 - inference is a heuristic; its failure means "no inferred effect"
        logger.debug("inferred within effect unavailable", exc_info=True)
        return out
    saved = list(getattr(engine, "effect_sizes", []) or [])
    try:
        engine.effect_sizes = []
        within_eff = []
        for c in design.cells:
            try:
                within_eff.append(float(engine._inferred_effect_value(c["label"], variable)) / unit)
            except Exception:  # noqa: BLE001
                within_eff.append(0.0)
        # the reference arm is the zero point: keyword residuals on a control / baseline label are not an effect
        try:
            ref = [i for i, c in enumerate(design.cells) if engine._is_control_arm(c["label"])]
        except Exception:  # noqa: BLE001
            ref = []
        if ref and design.type == "within":
            within_eff = [v - float(np.mean([within_eff[i] for i in ref])) for v in within_eff]
        out += np.array(within_eff)[None, :]
        if design.type == "mixed":
            skip0 = _is_baseline_first_level(design)
            for g, lab in enumerate(group_labels):
                try:
                    dv = float(engine._inferred_effect_value(lab, variable)) / unit
                except Exception:  # noqa: BLE001
                    dv = 0.0
                for ci in range(K):
                    if not (skip0 and ci == 0):
                        out[g, ci] += dv
    finally:
        engine.effect_sizes = saved
    # cap the total like the between route does (an implausible stack of keywords never exceeds d = 1.2)
    return np.clip(out, -1.2, 1.2)


# --------------------------------------------------------------------------------------
# moving finished answers
# --------------------------------------------------------------------------------------
def _num(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Writable float matrix of ``cols`` (NaN where missing)."""
    return np.array(df[list(cols)].apply(pd.to_numeric, errors="coerce"), dtype=float, copy=True)


def _composite(X: np.ndarray, rev: np.ndarray, lo: float, hi: float) -> np.ndarray:
    flipped = np.where(rev[None, :], (lo + hi) - X, X)
    with np.errstate(all="ignore"):
        return np.nanmean(flipped, axis=1)


def _shift_block(X: np.ndarray, rev: np.ndarray, delta: np.ndarray, lo: float, hi: float,
                 rng: np.random.RandomState) -> np.ndarray:
    """Move every person's composite by ``delta`` (scale points) by whole-point item changes.

    The shift is shared by the items (an item keyed against the construct moves the other way), the
    number of one-point steps is rounded stochastically so the expectation is exact, and steps that
    would leave the scale are given to the items that still have room.
    """
    n, k = X.shape
    valid = ~np.isnan(X)
    keff = valid.sum(1).astype(float)
    out = X.copy()
    active = (np.abs(delta) > 1e-12) & (keff > 0)
    if not active.any():
        return out
    sign_item = np.where(rev, -1.0, 1.0)[None, :]
    want = np.abs(delta) * keff
    base = np.floor(want)
    frac = want - base
    units = base.astype(int)
    # Rounding the fractional step counts: pick exactly round(sum of fractions) people (more likely the larger
    # the fraction) instead of tossing a coin per person, so the realised mean shift has no rounding noise.
    for sgn in (1.0, -1.0):
        grp = np.where(active & (np.sign(delta) == sgn) & (frac > 1e-12))[0]
        if len(grp):
            m = int(round(float(frac[grp].sum())))
            if m > 0:
                pick = rng.choice(grp, size=min(m, len(grp)), replace=False, p=frac[grp] / frac[grp].sum())
                units[pick] += 1
    units[~active] = 0
    direction = np.sign(delta)
    prio = rng.random_sample((n, k))
    prio[~valid] = 2.0
    step = direction[:, None] * sign_item                 # +-1 per item
    remaining = units.copy()
    for _ in range(8):
        if not (remaining > 0).any():
            break
        room = np.where(step > 0, hi - out, out - lo)
        room = np.where(valid, room, 0.0)
        eligible = room >= 1
        order = np.argsort(np.where(eligible, prio, 3.0), axis=1)
        rank = np.empty_like(order)
        rank[np.arange(n)[:, None], order] = np.arange(k)[None, :]
        n_el = eligible.sum(1)
        give_each = np.minimum(remaining, n_el)
        add = (eligible & (rank < give_each[:, None])).astype(float)
        out = out + add * step
        remaining = remaining - add.sum(1).astype(int)
        # persons with no room at all stay put
        remaining[n_el == 0] = 0
    return np.where(valid, np.clip(out, lo, hi), np.nan)


# --------------------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------------------
def _coupling(n_items: int) -> float:
    if n_items <= 1:
        return _SINGLE_ITEM_COUPLING
    if n_items == 2:
        return _TWO_ITEM_COUPLING
    return _COMPOSITE_COUPLING


def _build_latent_matrix(n_dv: int, k: int, r_dv: np.ndarray, rdv: np.ndarray, structure: str) -> np.ndarray:
    """Correlation of the (occasion-major, DV-minor) block of replicated measures."""
    size = n_dv * k
    R = np.eye(size)
    for j in range(k):
        for j2 in range(k):
            for a in range(n_dv):
                for b in range(n_dv):
                    row, col = j * n_dv + a, j2 * n_dv + b
                    if j == j2:
                        R[row, col] = 1.0 if a == b else rdv[a, b]
                    else:
                        lag = abs(j - j2)
                        rho_a = r_dv[a] ** (lag if structure == "ar1" else 1)
                        rho_b = r_dv[b] ** (lag if structure == "ar1" else 1)
                        R[row, col] = rho_a if a == b else rdv[a, b] * math.sqrt(rho_a * rho_b)
    R = (R + R.T) / 2.0
    w, V = np.linalg.eigh(R)
    if w.min() < 1e-3:
        w = np.clip(w, 1e-3, None)
        R = V @ np.diag(w) @ V.T
        dd = np.sqrt(np.diag(R))
        R = R / np.outer(dd, dd)
    np.fill_diagonal(R, 1.0)
    return R


def _scale_prefixes(E: Any, scales: List[Dict[str, Any]]) -> List[Optional[str]]:
    """Column prefix the engine will use for each scale (cleaned, de-duplicated), None when skipped."""
    used: set = set()
    out: List[Optional[str]] = []
    for sc in scales:
        raw = str(sc.get("name", "")).strip()
        if not raw:
            out.append(None)
            continue
        var = str(sc.get("variable_name", "")).strip()
        name = E._clean_column_name(var) if var else E._clean_column_name(raw)
        base, suffix = name, 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        out.append(name)
    return out


def wide_columns_for(metadata: Dict[str, Any], dv: str) -> Dict[str, Dict[str, Any]]:
    """{cell label: {"items": [...], "mean": col or None}} for one measure of a repeated-measures run."""
    wide = ((metadata or {}).get("design") or {}).get("wide_columns") or {}
    return wide.get(dv) or {}


def generate_repeated(engine: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate a within/mixed dataset for ``engine`` (an ``EnhancedSimulationEngine`` built with a design)."""
    from . import enhanced_simulation_engine as E

    spec: DesignSpec = engine.design_spec
    n = engine.sample_size
    cells = spec.cells
    K = len(cells)
    scales = [dict(s) for s in engine.scales]
    prefixes = _scale_prefixes(E, scales)
    live = [(a, sc, p) for a, (sc, p) in enumerate(zip(scales, prefixes)) if p]
    n_dv = len(live)
    ctor = dict(engine._ctor_kwargs)
    mixed = spec.type == "mixed"
    group_labels = list(engine.conditions) if mixed else [ALL_PARTICIPANTS_LABEL]
    group_levels: List[Dict[str, str]] = []
    for lab in group_labels:
        gl = _match_levels(lab, spec.between_factors) if (mixed and spec.between_factors) else None
        group_levels.append(gl or {})

    # ---- 1. replicate each measure once per within-cell, with a correlation matrix across occasions
    r_target = spec.within_correlation
    r_in = np.array([min(0.95, r_target / _coupling(int(sc.get("num_items", 1) or 1))) for _, sc, _ in live])
    if n_dv:
        if engine.correlation_matrix is not None and np.asarray(engine.correlation_matrix).shape == (n_dv, n_dv):
            rdv = np.asarray(engine.correlation_matrix, dtype=float)
        else:
            try:
                rdv = np.asarray(E.infer_correlation_matrix([s for _, s, _ in live])[0], dtype=float)
                if rdv.shape != (n_dv, n_dv):
                    rdv = np.eye(n_dv)
            except Exception:  # noqa: BLE001 - fall back to independent measures
                rdv = np.eye(n_dv)
    else:
        rdv = np.eye(0)
    latent = _build_latent_matrix(n_dv, K, r_in, rdv, spec.correlation_structure) if n_dv else None

    replicas: List[Dict[str, Any]] = []
    for j in range(K):
        for a, sc, pfx in live:
            rep = dict(sc)
            rep["name"] = rep["variable_name"] = f"{pfx}_occ{j + 1}"
            items = int(float(sc.get("num_items", 1) or 1))
            rel = sc.get("reliability")
            if rel is None and items >= 3:
                rep["reliability"] = float(np.random.RandomState(
                    E._stable_int_hash(f"{pfx}|target_alpha") & 0x7FFFFFFF).uniform(0.80, 0.90))
            replicas.append(rep)

    kw = dict(ctor)
    ctx = dict(ctor.get("study_context") or {})
    try:
        merged = list(dict.fromkeys(list(engine.detected_domains) + list(ctx.get("persona_domains") or [])))
        if merged:
            ctx["persona_domains"] = merged
    except Exception:  # noqa: BLE001
        pass
    kw.update(
        conditions=list(group_labels) if mixed else [ALL_PARTICIPANTS_LABEL],
        factors=[dict(f) for f in spec.between_factors] if mixed else [],
        scales=replicas, effect_sizes=[], auto_effects=False, dropout_rate=0.0,
        correlation_matrix=latent, study_context=ctx, design=None,
        condition_allocation=(ctor.get("condition_allocation") if mixed else None),
    )
    inner = type(engine)(**kw)
    # the pre-flight health check, the watchdog and the progress bar all talk to the OUTER engine's objects
    if getattr(engine, "llm_generator", None) is not None:
        inner.llm_generator = engine.llm_generator
    inner.allow_template_fallback = engine.allow_template_fallback
    inner.progress_callback = engine.progress_callback
    df_in, md_in = inner.generate()
    df = df_in.copy()

    # ---- 2. person-level design variables
    rng = np.random.RandomState((engine.seed + 0x57A1) % (2 ** 31))
    gcol = df["CONDITION"].astype(str).to_numpy()
    gidx = np.array([group_labels.index(g) if g in group_labels else 0 for g in gcol], dtype=int)
    fixed_idx = None
    if spec.fixed_order:
        lab_to_i = {c["label"]: i for i, c in enumerate(cells)}
        fixed_idx = [lab_to_i[x] for x in spec.fixed_order if x in lab_to_i]
    orders, eff_order = build_orders(K, n, spec.order, rng, fixed=fixed_idx, groups=gidx)
    pos = np.zeros((n, K), dtype=int)          # 1-based presentation position of cell c for person i
    for i in range(n):
        pos[i, orders[i]] = np.arange(1, K + 1)
    centre = (K + 1) / 2.0

    flag_sl = (pd.to_numeric(df.get("Flag_StraightLine", 0), errors="coerce").fillna(0).to_numpy() > 0)
    # a person who gives one identical answer to every item in half or more of the blocks is a straight-liner too
    n_const = np.zeros(n)
    n_blocks = 0
    for _a, sc_, pfx_ in live:
        k_items = int(float(sc_.get("num_items", 1) or 1))
        if k_items < 3:
            continue
        for j in range(K):
            cols_ = [f"{pfx_}_occ{j + 1}_{i + 1}" for i in range(k_items)]
            if all(c in df.columns for c in cols_):
                blk_ = _num(df, cols_)
                with np.errstate(all="ignore"):
                    n_const += (np.nanmax(blk_, axis=1) == np.nanmin(blk_, axis=1)).astype(float)
                n_blocks += 1
    mostly_constant = (n_const >= max(2.0, n_blocks / 2.0)) if n_blocks else np.zeros(n, dtype=bool)
    careless = flag_sl | mostly_constant
    careful = ~careless

    # ---- 2b. careless responding is a property of the PERSON: a straight-liner straight-lines every block (before the
    #      correlation is measured, because repeating the same answer is itself a source of correlation)
    forced = 0
    careless_idx = np.where(careless)[0]
    for a, sc, pfx in live:
        items = int(float(sc.get("num_items", 1) or 1))
        icols = [[f"{pfx}_occ{j + 1}_{i + 1}" for i in range(items)] for j in range(K)]
        if items < 3 or len(careless_idx) == 0 or any(c not in df.columns for blk in icols for c in blk):
            continue
        lo, hi = float(sc.get("scale_min", 1)), float(sc.get("scale_max", sc.get("scale_points", 7)))
        rev = np.array([(i + 1) in {int(x) for x in (sc.get("reverse_items") or []) if str(x).lstrip("-").isdigit()}
                        for i in range(items)])
        Xs = [_num(df, icols[j]) for j in range(K)]
        changed = False
        for i in careless_idx:
            rows = [Xs[j][i] for j in range(K)]
            const = [r[np.isfinite(r)][0] for r in rows
                     if np.isfinite(r).any() and np.nanmax(r) == np.nanmin(r)]
            if not const:
                continue
            for j in range(K):
                r = rows[j]
                if np.isfinite(r).any() and np.nanmax(r) != np.nanmin(r):
                    Xs[j][i] = np.where(np.isfinite(r), const[0], np.nan)
                    forced += 1
                    changed = True
        if changed:
            for j in range(K):
                for i2, c in enumerate(icols[j]):
                    df[c] = Xs[j][:, i2]
                mc = f"{pfx}_occ{j + 1}_mean"
                if mc in df.columns:
                    df[mc] = np.round(_composite(Xs[j], rev, lo, hi), 2)

    # ---- 3. per-measure effects, order drift and correlation top-up
    user_specs = list(engine.effect_sizes or []) + [s for s in spec.simple_effects]
    effects_log: List[Dict[str, Any]] = []
    spec_rows: List[Dict[str, Any]] = []
    coupling_log: List[Dict[str, Any]] = []
    wide_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    shift_rng = np.random.RandomState((engine.seed + 0x5E1F) % (2 ** 31))

    for a, sc, pfx in live:
        items = int(float(sc.get("num_items", 1) or 1))
        lo, hi = float(sc.get("scale_min", 1)), float(sc.get("scale_max", sc.get("scale_points", 7)))
        rev_set = set()
        for x in sc.get("reverse_items") or []:
            try:
                rev_set.add(int(x))
            except (TypeError, ValueError):
                pass
        rev = np.array([(i + 1) in rev_set for i in range(items)])
        icols = [[f"{pfx}_occ{j + 1}_{i + 1}" for i in range(items)] for j in range(K)]
        mean_cols = [f"{pfx}_occ{j + 1}_mean" for j in range(K)]
        wide_map[pfx] = {
            cells[j]["label"]: {
                "items": [f"{pfx}_{cells[j]['slug']}_{i + 1}" for i in range(items)],
                "mean": (f"{pfx}_{cells[j]['slug']}_mean" if items > 1 and mean_cols[j] in df.columns else None),
                "inner_items": icols[j], "inner_mean": mean_cols[j],
            } for j in range(K)
        }
        stype = str(sc.get("type", "")).lower()
        if (stype in _UNSHIFTABLE_TYPES or hi <= lo
                or any(c not in df.columns for blk in icols for c in blk)):
            continue
        Xs = [_num(df, icols[j]) for j in range(K)]
        comps = np.column_stack([_composite(Xs[j], rev, lo, hi) for j in range(K)])
        resid = np.full_like(comps, np.nan)
        sd_j = np.zeros(K)
        for j in range(K):
            col = comps[:, j].copy()
            for g in range(len(group_labels)):
                m = (gidx == g) & careful & np.isfinite(col)
                if m.any():
                    col[gidx == g] = col[gidx == g] - np.nanmean(col[m])
            resid[:, j] = col
            m_all = np.isfinite(col)        # the SD an analyst sees: everyone, careless responders included
            sd_j[j] = float(np.nanstd(col[m_all], ddof=1)) if m_all.sum() > 2 else 0.0
        sd_bar = float(np.mean(sd_j[sd_j > 0])) if (sd_j > 0).any() else 0.0

        # effect offsets (SD units)
        specs_a = [s for s in user_specs if engine._spec_applies_to_variable(_spec_value(s, "variable", ""), pfx)]
        if specs_a:
            mu, rows = _solve_cell_offsets(engine, spec, pfx, specs_a, group_labels, group_levels)
            spec_rows.extend(rows)
            source = "user"
        else:
            mu = _inferred_cell_offsets(engine, spec, pfx, group_labels, items, int(lo), int(hi))
            source = "inferred" if np.any(mu != 0) else "none"
        for g, glab in enumerate(group_labels):
            for j in range(K):
                effects_log.append({"variable": pfx, "group": glab, "cell": cells[j]["label"],
                                    "offset_d": round(float(mu[g, j]), 4), "source": source})
        order_term = ((pos - centre) * spec.order_effect_d) if spec.order_effects else np.zeros((n, K))
        base_shift = np.zeros((n, K))
        for j in range(K):
            base_shift[:, j] = sd_bar * (mu[gidx, j] + order_term[:, j])
        # Careless responders (straight-liners) carry no construct information, so the effect is built into the
        # careful respondents and scaled up by the share of careless ones: the requested d_av is a statement about
        # the whole sample, as it is for the between-subjects route.
        if careful.any() and (~careful).any():
            base_shift *= float(min(1.25, n / max(1, int(careful.sum()))))
        base_shift[~careful, :] = 0.0

        # correlation top-up with a person-level draw shared by the occasions
        s_i = np.random.RandomState((engine.seed + 0xC0DE + 97 * a) % (2 ** 31)).standard_normal(n)
        s_i = (s_i - s_i.mean()) / (s_i.std() or 1.0)
        e = np.nan_to_num(np.where(sd_j[None, :] > 0, resid / np.where(sd_j > 0, sd_j, 1.0)[None, :], 0.0))
        w = 0.0                     # share of the person-level shift moved onto a draw shared by the conditions
        w_down = 0.0                # share replaced by an independent draw per condition (lowers the correlation)
        eps_ij = np.random.RandomState((engine.seed + 0xE55 + 89 * a) % (2 ** 31)).standard_normal((n, K))
        corr = np.zeros(K)          # per-cell correction for steps lost to the scale ends
        applied: Optional[List[np.ndarray]] = None
        r_real = float("nan")
        r_before = float("nan")
        for attempt in range(6):
            delta = base_shift + corr[None, :]
            delta[~careful, :] = 0.0
            if w > 0 or w_down > 0:
                for j in range(K):
                    if w > 0:
                        delta[:, j] += sd_j[j] * ((math.sqrt(1.0 - w) - 1.0) * e[:, j] + math.sqrt(w) * s_i)
                    if w_down > 0:
                        delta[:, j] += sd_j[j] * ((math.sqrt(1.0 - w_down) - 1.0) * e[:, j] + math.sqrt(w_down) * eps_ij[:, j])
                delta[~careful, :] = 0.0
            new_blocks = [_shift_block(Xs[j], rev, delta[:, j], lo, hi,
                                       np.random.RandomState(shift_rng.randint(0, 2 ** 31 - 1)))
                          for j in range(K)]
            applied = new_blocks
            comps_new = np.column_stack([_composite(new_blocks[j], rev, lo, hi) for j in range(K)])
            settled = True
            # (a) steps the scale ends swallowed: give the cell back what it lost (one scalar per cell, so
            #     the individual differences and the sampling variation of the contrast are untouched)
            for j in range(K):
                m = careful & np.isfinite(comps_new[:, j]) & np.isfinite(comps[:, j])
                if m.sum() > 5:
                    gap = float(np.mean(base_shift[m, j])) - float(np.mean(comps_new[m, j] - comps[m, j]))
                    if abs(gap) > 0.01 * max(sd_bar, 1e-9):
                        corr[j] = float(np.clip(corr[j] + gap, -1.5 * sd_bar, 1.5 * sd_bar))
                        settled = False
            # (b) correlation across conditions, as the analyst sees it (everyone with all conditions answered)
            ok = np.all(np.isfinite(comps_new), axis=1)
            if ok.sum() > 10 and K >= 2:
                res_new = comps_new.copy()
                for j in range(K):
                    for g in range(len(group_labels)):
                        m = (gidx == g) & ok
                        if m.any():
                            res_new[gidx == g, j] -= np.mean(comps_new[m, j])
                cm = np.corrcoef(res_new[ok].T)
                off = cm[np.triu_indices(K, 1)]
                r_real = float(np.nanmean(off)) if spec.correlation_structure == "exchangeable" else float(
                    np.nanmean([cm[i, i + 1] for i in range(K - 1)]))
            if attempt == 0:
                r_before = r_real
            if math.isfinite(r_real) and spec.correlation_structure == "exchangeable":
                if r_real < r_target - 0.01:
                    if w_down > 0:
                        w_down = max(0.0, w_down - (r_target - r_real) / max(r_real, 1e-6))
                    else:
                        w = float(min(0.9, w + (r_target - r_real) / max(1e-6, 1.0 - r_real)))
                    settled = False
                elif r_real > r_target + 0.02 and w == 0.0:
                    w_down = float(min(0.9, w_down + (r_real - r_target) / max(r_real, 1e-6)))
                    settled = False
            if settled:
                break
        for j in range(K):
            blk = applied[j] if applied is not None else Xs[j]
            for i2, c in enumerate(icols[j]):
                df[c] = blk[:, i2]
            if items > 1 and mean_cols[j] in df.columns:
                df[mean_cols[j]] = np.round(_composite(blk, rev, lo, hi), 2)
        coupling_log.append({"variable": pfx, "target_r": round(r_target, 3),
                             "engine_r": None if not math.isfinite(r_before) else round(r_before, 3),
                             "final_r": None if not math.isfinite(r_real) else round(r_real, 3),
                             "topup_weight": round(float(w), 3), "lowering_weight": round(float(w_down), 3)})

    # ---- 5. attrition: a dropout loses the LATER conditions
    drop_rng = np.random.RandomState((engine.seed + 0xD209) % (2 ** 31))
    completed = np.full(n, K, dtype=int)
    n_drop = int(round(float(engine.dropout_rate) * n))
    if n_drop > 0 and K >= 2:
        who = drop_rng.choice(n, size=min(n, n_drop), replace=False)
        for i in who:
            completed[i] = int(drop_rng.randint(1, K))          # finished 1..K-1 conditions
    cell_cols: Dict[int, List[str]] = {j: [] for j in range(K)}
    for c in df.columns:
        m = re.match(r"^(.*)_occ(\d+)(?:_.*)?$", str(c))
        if m and m.group(1) in {p for _, _, p in live}:
            cell_cols[int(m.group(2)) - 1].append(c)
    for i in np.where(completed < K)[0]:
        for j in range(K):
            if pos[i, j] > completed[i]:
                df.loc[df.index[i], cell_cols[j]] = np.nan
    if (completed < K).any():
        for col in ("Completion_Time_Seconds", "Total_Scale_RT_ms"):
            if col in df.columns:
                frac = completed / float(K)
                df[col] = np.where(completed < K, np.round(pd.to_numeric(df[col], errors="coerce") * frac, 1),
                                   df[col])

    # ---- 6. rename to <DV>_<condition>_<i> and add the design columns
    rename: Dict[str, str] = {}
    for c in df.columns:
        m = re.match(r"^(.*)_occ(\d+)((?:_.*)?)$", str(c))
        if m and m.group(1) in {p for _, _, p in live}:
            rename[c] = f"{m.group(1)}_{cells[int(m.group(2)) - 1]['slug']}{m.group(3)}"
    df = df.rename(columns=rename)
    if not mixed:
        df["CONDITION"] = ALL_PARTICIPANTS_LABEL
    order_labels = [">".join(cells[j]["label"] for j in orders[i]) for i in range(n)]
    ins = list(df.columns).index("CONDITION") + 1
    df.insert(ins, "Order", order_labels)
    for j in range(K):
        df.insert(ins + 1 + j, f"Position_{cells[j]['slug']}", pos[:, j])
    df.insert(ins + 1 + K, "Conditions_Completed", completed)

    # final names in the map; drop internals
    for pfx, per in wide_map.items():
        for lab, d in per.items():
            d.pop("inner_items", None)
            d.pop("inner_mean", None)

    md = _compose_metadata(engine, inner, md_in, df, spec, group_labels, wide_map, rename, effects_log, spec_rows,
                           coupling_log, eff_order, completed, careless, forced, live, r_target)
    # the outer engine adopts the inner run's bookkeeping so the exports/explainer see the wide columns
    engine.column_info = [(rename.get(c, c), d) for c, d in (inner.column_info or [])]
    engine.validation_log = list(getattr(inner, "validation_log", []) or []) + [
        f"Within-subjects design: {K} conditions per participant, order '{eff_order}'."]
    engine._scale_generation_log = md.get("scale_generation_log", [])
    engine._within_result = {"df": df, "metadata": md}
    return df, md


def _compose_metadata(engine, inner, md_in, df, spec, group_labels, wide_map, rename, effects_log, spec_rows,
                      coupling_log, eff_order, completed, careless, forced, live, r_target):
    md = dict(md_in)
    K = len(spec.cells)
    md["conditions"] = list(engine.conditions)
    md["factors"] = [dict(f) for f in engine.factors]
    md["scales"] = [dict(s) for s in engine.scales]
    md["sample_size"] = int(engine.sample_size)
    md["column_descriptions"] = {rename.get(k, k): v for k, v in (md.get("column_descriptions") or {}).items()}
    gl = []
    for entry in md.get("scale_generation_log", []) or []:
        cols = [rename.get(c, c) for c in entry.get("columns_generated", [])]
        e2 = dict(entry)
        e2["columns_generated"] = cols
        m = re.match(r"^(.*)_occ(\d+)$", str(entry.get("name", "")))
        if m:
            cell = spec.cells[int(m.group(2)) - 1]
            e2["name"] = f"{m.group(1)} ({cell['label']})"
        gl.append(e2)
    md["scale_generation_log"] = gl
    cd = md["column_descriptions"]
    cd["Order"] = "Presentation order of the within-subject conditions for this participant (first > last)"
    for j, c in enumerate(spec.cells):
        cd[f"Position_{c['slug']}"] = f"Presentation position (1 = first) of the condition '{c['label']}'"
    cd["Conditions_Completed"] = "Number of within-subject conditions the participant finished before leaving (attrition)"
    cd["CONDITION"] = ("Between-subjects group" if spec.type == "mixed"
                       else "All participants see every condition; the conditions are in the column names")
    md["cross_dv_correlation"] = {"enabled": True, "note": "latent coupling across measures and across conditions "
                                  "(see design.within_correlation)"}
    design = spec.to_dict()
    design.update({
        "order_effective": eff_order,
        "wide_columns": wide_map,
        "long_format_file": "Simulated_Data_Long.csv" if spec.long_format else None,
        "effect_definition": ("cohens_d on a within factor is d_av = mean difference / mean of the two conditions' "
                              "SDs; d_z is reported next to it"),
        "group_labels": list(group_labels),
        "coupling": coupling_log,
        "attrition": {"rate_configured": float(engine.dropout_rate), "dropped": int((completed < len(spec.cells)).sum()),
                      "pattern": "a participant who drops out loses the conditions presented after the last one finished"},
        "careless": {"n_flagged_straight_line": int(careless.sum()), "blocks_made_consistent": int(forced)},
        "order_effect": {"enabled": bool(spec.order_effects), "d_per_position": float(spec.order_effect_d)},
    })
    md["design"] = design
    md["design_type"] = spec.type
    md["effect_sizes_configured"] = [
        {"variable": _spec_value(s, "variable"), "factor": _spec_value(s, "factor"),
         "level_high": _spec_value(s, "level_high"), "level_low": _spec_value(s, "level_low"),
         "cohens_d": _spec_value(s, "cohens_d"), "direction": _spec_value(s, "direction", "positive"),
         "at": _spec_value(s, "at", None)}
        for s in list(engine.effect_sizes or []) + list(spec.simple_effects or [])]
    obs = observed_contrasts(df, md)
    md["effect_sizes_observed"] = obs
    md["effect_sizes_applied"] = {
        "design": spec.type, "definition": design["effect_definition"],
        "inferred_effects_enabled": bool(getattr(engine, "auto_effects", True)),
        "specs": spec_rows, "cell_offsets_d": effects_log,
        "contrasts": [{"variable": o["variable"], "condition_1": o["condition_1"], "condition_2": o["condition_2"],
                       "kind": o.get("kind", "within"), "observed_d_av": o.get("d_av"),
                       "observed_d_z": o.get("d_z"), "n_pairs": o.get("n")} for o in obs],
        "note": ("Each within contrast is condition_1 minus condition_2 for the same participants. The requested "
                 "d is d_av; the observed values here are for this sample and vary with sampling."),
    }
    md["missing_data"] = dict(md.get("missing_data") or {}, dropout_rate=float(engine.dropout_rate),
                              dropout_count=int((completed < K).sum()))
    return md


# --------------------------------------------------------------------------------------
# observed contrasts and long format
# --------------------------------------------------------------------------------------
def _col_for(metadata: Dict[str, Any], dv: str, label: str) -> Optional[str]:
    info = wide_columns_for(metadata, dv).get(label) or {}
    if info.get("mean"):
        return info["mean"]
    items = info.get("items") or []
    return items[0] if len(items) == 1 else None


def observed_contrasts(df: pd.DataFrame, metadata: Dict[str, Any], max_pairs: int = 6) -> List[Dict[str, Any]]:
    """Observed within contrasts (d_av, d_z, paired r) per measure and pair of conditions, and, for a mixed
    design, the between-group d at each within-level."""
    d = design_from_metadata(metadata)
    if not d:
        return []
    labels = [c["label"] for c in d["cells"]]
    out: List[Dict[str, Any]] = []
    for dv in (d.get("wide_columns") or {}):
        cols = {lab: _col_for(metadata, dv, lab) for lab in labels}
        if any(v is None or v not in df.columns for v in cols.values()):
            continue
        n_pair = 0
        for a, b in itertools.combinations(labels, 2):
            if n_pair >= max_pairs:
                break
            xa = pd.to_numeric(df[cols[a]], errors="coerce").to_numpy(float)
            xb = pd.to_numeric(df[cols[b]], errors="coerce").to_numpy(float)
            ok = np.isfinite(xa) & np.isfinite(xb)
            if ok.sum() < 3:
                continue
            diff = xa[ok] - xb[ok]
            sa, sb = np.std(xa[ok], ddof=1), np.std(xb[ok], ddof=1)
            sdd = np.std(diff, ddof=1)
            avg = (sa + sb) / 2.0
            r = float(np.corrcoef(xa[ok], xb[ok])[0, 1]) if sa > 0 and sb > 0 else float("nan")
            out.append({
                "variable": dv, "kind": "within", "condition_1": a, "condition_2": b, "n": int(ok.sum()),
                "mean_1": round(float(xa[ok].mean()), 4), "mean_2": round(float(xb[ok].mean()), 4),
                "d_av": round(float(diff.mean() / avg), 4) if avg > 0 else None,
                "d_z": round(float(diff.mean() / sdd), 4) if sdd > 0 else None,
                "r": None if not math.isfinite(r) else round(r, 4),
            })
            n_pair += 1
    return out


def build_long_format(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    """Participant x condition rows: one row per participant and within-cell.

    Columns: the person-level variables, the within factor(s), ``Condition``, ``Position`` (presentation
    position), ``Order`` and, per measure, ``<DV>_<i>`` and ``<DV>_mean`` (the names without the cell).
    """
    d = design_from_metadata(metadata)
    if not d:
        raise ValueError("build_long_format needs the data of a within-subjects or mixed run")
    cells = d["cells"]
    wide_cols = d.get("wide_columns") or {}
    measure_cols = {c for per in wide_cols.values() for info in per.values()
                    for c in list(info.get("items") or []) + ([info["mean"]] if info.get("mean") else [])}
    pos_cols = {f"Position_{c['slug']}" for c in cells}
    person_cols = [c for c in df.columns if c not in measure_cols and c not in pos_cols
                   and c not in ("Conditions_Completed",)]
    try:  # participant-facing columns only, like the wide file (the rest is in Simulation_Diagnostics.csv)
        from .qualtrics_export import protected_columns, split_columns
        facing, _internal = split_columns(person_cols, protected_columns(metadata))
        person_cols = [c for c in person_cols if c == "PARTICIPANT_ID" or c in facing]
    except Exception:  # noqa: BLE001 - fall back to every person-level column
        logger.debug("long format: could not separate diagnostic columns", exc_info=True)
    pieces = []
    for cell in cells:
        part = df[person_cols].copy()
        part.insert(0, "Condition", cell["label"])
        for fname, lv in cell["levels"].items():
            part[str(fname)] = lv
        part["Position"] = df.get(f"Position_{cell['slug']}")
        part["Conditions_Completed"] = df.get("Conditions_Completed")
        for dv, per in wide_cols.items():
            info = per.get(cell["label"]) or {}
            for i, c in enumerate(info.get("items") or []):
                part[f"{dv}_{i + 1}"] = df[c].to_numpy() if c in df.columns else np.nan
            if info.get("mean") and info["mean"] in df.columns:
                part[f"{dv}_mean"] = df[info["mean"]].to_numpy()
        pieces.append(part)
    long_df = pd.concat(pieces, ignore_index=True)
    sort_cols = [c for c in ("PARTICIPANT_ID", "Position") if c in long_df.columns]
    if sort_cols:
        long_df = long_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return long_df


# --------------------------------------------------------------------------------------
# design suggestion from a QSF's block structure
# --------------------------------------------------------------------------------------
_REPEAT_TOKEN_RE = re.compile(
    r"(?:^|[\s_\-./:])(?:pre|post|pretest|posttest|before|after|baseline|follow ?up|fu|t ?\d|time ?\d|wave ?\d|"
    r"week ?\d|day ?\d|session ?\d|round ?\d|trial ?\d|block ?\d|phase ?\d)(?:$|[\s_\-./:\d])", re.I)


def _strip_repeat_tokens(text: str) -> str:
    s = re.sub(r"\b(?:pre|post|pretest|posttest|before|after|baseline|follow ?up|fu)\b", " ", str(text), flags=re.I)
    s = re.sub(r"\b(?:t|time|wave|week|day|session|round|trial|block|phase)[\s_\-]*\d+\b", " ", s, flags=re.I)
    s = re.sub(r"\d+", " ", s)
    return re.sub(r"[^a-z]+", " ", s.lower()).strip()


def suggest_design_from_blocks(blocks: Sequence[Dict[str, Any]], has_randomizer: bool = False) -> Dict[str, Any]:
    """Look for the same measures repeated across blocks that every participant sees.

    ``blocks`` is a list of ``{"name": str, "questions": [question text or name, ...]}`` in flow order
    (only blocks that are actually shown). The result is a SUGGESTION, never a decision::

        {"suggest": "within"|None, "confidence": "high"|"medium"|"low",
         "levels": ["Pre", "Post"], "blocks": [...], "reason": "..."}

    It fires when two or more blocks have the same number of questions and, after removing time words and
    digits, the same question wording or names, and the blocks are not randomized between participants.
    """
    result: Dict[str, Any] = {"suggest": None, "confidence": "low", "levels": [], "blocks": [], "reason": ""}
    if has_randomizer or not blocks:
        result["reason"] = "A randomizer assigns participants to one block, or there are no blocks."
        return result
    sigs: List[Tuple[str, Tuple[str, ...]]] = []
    for b in blocks:
        qs = tuple(sorted(filter(None, (_strip_repeat_tokens(q) for q in (b.get("questions") or [])))))
        sigs.append((str(b.get("name", "")), qs))
    groups: Dict[Tuple[str, ...], List[int]] = {}
    for i, (_, qs) in enumerate(sigs):
        if len(qs) >= 1:
            groups.setdefault(qs, []).append(i)
    best = max(groups.values(), key=len) if groups else []
    if len(best) < 2:
        result["reason"] = "No set of measures appears in more than one block."
        return result
    names = [sigs[i][0] for i in best]
    named_like_time = sum(1 for nme in names if _REPEAT_TOKEN_RE.search(" " + nme + " "))
    result["blocks"] = names
    result["suggest"] = "within"
    result["confidence"] = "high" if named_like_time >= 2 else "medium"
    levels = []
    for nme in names:
        t = re.sub(r"\s+", " ", nme.replace("_", " ")).strip()
        levels.append(t or f"Time {len(levels) + 1}")
    if len(set(levels)) < len(levels):
        levels = [f"Time {i + 1}" for i in range(len(names))]
    result["levels"] = levels
    result["reason"] = (f"{len(best)} blocks ({', '.join(names)}) ask the same {len(sigs[best[0]][1])} question(s) "
                        "and are not randomized, so every participant answers them all.")
    return result


_DESCRIPTIVE_TYPES = frozenset({"db", "descriptivetext", "descriptive text", "descriptive", "timing", "meta", "captcha"})


def suggest_design_from_qsf(result: Any) -> Dict[str, Any]:
    """Suggest within/mixed from a parsed QSF (a ``QSFPreviewResult``); never decides, only suggests.

    Fires when the same measures are asked in two or more blocks that EVERY participant sees (in a
    mixed design: every participant of every randomized group). Blocks that only some conditions see
    (the usual between-subjects layout: one stimulus block per condition) never count.
    """
    base: Dict[str, Any] = {"suggest": None, "confidence": "low", "levels": [], "blocks": [], "reason": ""}
    try:
        blocks = list(getattr(result, "blocks", None) or [])
        conditions = [str(c) for c in (getattr(result, "detected_conditions", None) or []) if str(c).strip()]
        cond_blocks = getattr(result, "condition_blocks", None) or {}
        items: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for b in blocks:
            sig = (str(getattr(b, "block_name", "")), tuple(str(getattr(q, "question_id", "")) for q in getattr(b, "questions", None) or []))
            if sig in seen_ids:
                continue  # the parser lists each block under its id and under its key in the survey
            seen_ids.add(sig)
            if str(getattr(b, "block_type", "Standard")).lower() == "trash" or getattr(b, "is_randomizer", False):
                continue
            qs = []
            for q in getattr(b, "questions", None) or []:
                if str(getattr(q, "question_type", "")).lower() in _DESCRIPTIVE_TYPES:
                    continue
                text = str(getattr(q, "question_text", "") or "") or str(getattr(q, "export_tag", "") or "")
                if text.strip():
                    qs.append(text)
            if qs:
                items.append({"id": getattr(b, "block_id", ""), "name": str(getattr(b, "block_name", "")), "questions": qs})
        if conditions and cond_blocks:
            seen_by_all = [it for it in items if all(it["id"] in set(cond_blocks.get(c) or []) for c in conditions)]
            items = seen_by_all
        elif conditions and not cond_blocks:
            items = [it for it in items if _REPEAT_TOKEN_RE.search(" " + it["name"] + " ")]
        out = suggest_design_from_blocks(items, has_randomizer=False)
        if out.get("suggest") and conditions:
            out["suggest"] = "mixed"
            out["between_groups"] = conditions
            out["reason"] += f" The randomized groups ({', '.join(conditions[:4])}) would be the between-subjects part."
        return out
    except Exception:  # noqa: BLE001 - a suggestion must never break the upload step
        logger.debug("design suggestion failed", exc_info=True)
        return base
