"""
Empirical Registry — provenance, verification tiers, and replication-adjusted effects.
=====================================================================================

WHY THIS MODULE EXISTS
----------------------
`scientific_knowledge_base.py` holds 484 calibration entries, each carrying a
citation, a 95% CI, k, N, tau and I-squared. Those fields *look* like they were
transcribed from papers. Most were not: they were written out in bulk, and the
numbers carry the fingerprint of generated values rather than extracted ones
(79% of meta-analytic N values are exact multiples of 1000; 98% of construct-norm
N values are multiples of 100; tau takes only 8 distinct values across 187
entries; 173/187 entries are labelled "replicated").

A simulator whose selling point is "grounded in the literature" must not present
unverified numbers as published fact. This module therefore adds the layer the
knowledge base never had:

1. PROVENANCE — a per-entry record of what was actually checked against a primary
   source, with DOI, URL, the quoted source text and the verification date.
   Entries with no record are reported as UNVERIFIED rather than silently trusted.
2. VERIFICATION TIERS — `status_of()` returns the tier for any knowledge-base key,
   and `confidence_weight()` turns the tier into a multiplier the engine can use to
   shrink the influence of anything that was never checked.
3. REPLICATION ADJUSTMENT — published effect sizes systematically exceed what
   multi-lab replications find. `adjust_effect()` applies that correction so the
   simulator targets the effect size a *new* study would observe, not the one the
   original paper reported. This is the behaviour the tool's users actually want,
   and it is opt-outable via `EffectPolicy`.
4. EMPIRICAL MARGINALS — verified distribution *shapes* (point masses, modes,
   round-number heaping) for paradigms where the shape, not the mean, is what
   makes real data recognisable. See `empirical_marginals.py` for the samplers.

AUDITABILITY
------------
`audit_table()` returns one row per knowledge-base entry with its tier and
provenance, so the app can show users exactly which numbers are sourced and which
are not. `coverage_summary()` gives the headline counts.

This module has no dependencies beyond the standard library and
`scientific_knowledge_base`, and it never raises on a missing entry.
"""
from __future__ import annotations

__version__ = "1.0.0"

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import random


# =============================================================================
# VERIFICATION TIERS
# =============================================================================

#: Checked against the primary source; the numbers below are the source's own.
VERIFIED = "verified"
#: The source exists and supports part of the entry; some fields are not sourced.
PARTIAL = "partial"
#: The citation is real and the claim is broadly consistent with it, but the exact
#: numbers in the entry were not confirmed against the source.
CITED_UNCHECKED = "cited_unchecked"
#: No verification record. The entry's precise values should be treated as a
#: plausible prior, not as published fact.
UNVERIFIED = "unverified"
#: Checked and found to be WRONG. The corrected value is in `Provenance.corrected`.
CORRECTED = "corrected"

_TIER_ORDER = (VERIFIED, CORRECTED, PARTIAL, CITED_UNCHECKED, UNVERIFIED)

#: How much weight the engine should give an entry's numbers, by tier.
#: Unverified entries are not discarded — they are the best prior available — but
#: their deviation from a neutral default is damped, so an invented number can
#: never drive a simulation as hard as a sourced one.
TIER_WEIGHT: Dict[str, float] = {
    VERIFIED: 1.00,
    CORRECTED: 1.00,
    PARTIAL: 0.85,
    CITED_UNCHECKED: 0.70,
    UNVERIFIED: 0.55,
}


@dataclass
class Provenance:
    """What was actually checked, by whom, against what."""
    status: str                      # one of the tier constants above
    source: str = ""                 # citation as it appears in the source
    doi: str = ""
    url: str = ""                    # the page that was fetched
    quote: str = ""                  # verbatim text supporting the numbers
    verified_on: str = ""            # ISO date of the check
    checked_fields: Tuple[str, ...] = ()   # which fields the quote actually supports
    corrected: Dict[str, Any] = field(default_factory=dict)  # field -> correct value
    note: str = ""

    def weight(self) -> float:
        return TIER_WEIGHT.get(self.status, TIER_WEIGHT[UNVERIFIED])


# =============================================================================
# PROVENANCE RECORDS
# =============================================================================
# Keys are "<kind>:<knowledge_base_key>" where kind is one of
# meta | game | norm | culture | rt | order, matching the knowledge-base tables.
#
# EVERY record below was produced by fetching the cited page during the
# verification pass dated in `verified_on`. Records are added only when a page
# was actually read; nothing here is filled in from recall. Entries the
# knowledge base contains but this table does not are reported as UNVERIFIED by
# `status_of()`, which is the honest default.

PROVENANCE: Dict[str, Provenance] = {}


def register(key: str, prov: Provenance) -> None:
    """Add or replace a provenance record."""
    PROVENANCE[key] = prov


def status_of(kind: str, key: str) -> str:
    """Verification tier for a knowledge-base entry. UNVERIFIED when unrecorded."""
    rec = PROVENANCE.get(f"{kind}:{key}")
    return rec.status if rec else UNVERIFIED


def provenance_of(kind: str, key: str) -> Optional[Provenance]:
    return PROVENANCE.get(f"{kind}:{key}")


def confidence_weight(kind: str, key: str) -> float:
    """Multiplier in (0, 1] for how hard an entry may push a simulation."""
    rec = PROVENANCE.get(f"{kind}:{key}")
    return rec.weight() if rec else TIER_WEIGHT[UNVERIFIED]


def corrected_value(kind: str, key: str, field_name: str, default: Any = None) -> Any:
    """Return the verified correction for a field, or `default` if none."""
    rec = PROVENANCE.get(f"{kind}:{key}")
    if rec and field_name in rec.corrected:
        return rec.corrected[field_name]
    return default


# =============================================================================
# REPLICATION ADJUSTMENT (publication-bias correction)
# =============================================================================

@dataclass
class EffectPolicy:
    """How literature-derived effect sizes are turned into simulation targets.

    `as_published`        — use the point estimate exactly as the entry states it.
    `replication_adjusted` (default) — shrink toward the effect a new, adequately
                            powered study would find. Published estimates are
                            inflated by selective reporting; a simulator that
                            reproduces published d values produces data that are
                            systematically *too clean* to pass for real.
    `heterogeneity_draw`  — when True, each simulation run draws its own effect
                            from the between-study distribution instead of using
                            the pooled mean, so two runs of the same design differ
                            the way two real labs differ.
    """
    mode: str = "replication_adjusted"   # as_published | replication_adjusted
    heterogeneity_draw: bool = True
    #: Multiplicative shrinkage applied in replication_adjusted mode. Set from the
    #: verified literature in `_SHRINKAGE`; see that record for provenance.
    shrinkage: Optional[float] = None
    #: Fallback between-study SD (on the d scale) when an entry reports no tau.
    default_tau: float = 0.0
    #: Hard floor so shrinkage can never erase a real effect entirely.
    min_retained: float = 0.35


#: Publication-bias shrinkage factor and default heterogeneity, with provenance.
#: Populated by the verification pass; until a record is registered the policy
#: falls back to NO shrinkage (1.0), because applying an unverified correction
#: would be the same mistake this module exists to prevent.
_SHRINKAGE: Dict[str, Any] = {
    "factor": None,          # ratio replication_d / published_d
    "default_tau": None,     # typical between-study SD on the d scale
    "provenance_key": "policy:publication_bias",
}


def set_shrinkage(factor: float, default_tau: float, prov: Provenance) -> None:
    """Install a verified shrinkage factor and its provenance."""
    _SHRINKAGE["factor"] = float(factor)
    _SHRINKAGE["default_tau"] = float(default_tau)
    register(_SHRINKAGE["provenance_key"], prov)


def shrinkage_factor() -> float:
    """Verified shrinkage factor, or 1.0 (no correction) when unverified."""
    f = _SHRINKAGE.get("factor")
    return float(f) if f else 1.0


def default_tau() -> float:
    t = _SHRINKAGE.get("default_tau")
    return float(t) if t else 0.0


def adjust_effect(
    effect_d: float,
    kind: str = "meta",
    key: str = "",
    policy: Optional[EffectPolicy] = None,
    rng: Optional[random.Random] = None,
    tau: Optional[float] = None,
) -> float:
    """Turn a literature effect size into the effect a fresh study would see.

    Order of operations:
      1. tier weighting  — an unverified entry's deviation from zero is damped.
      2. publication-bias shrinkage (replication_adjusted mode only).
      3. between-study heterogeneity draw, so runs vary like real labs do.

    The sign of `effect_d` is always preserved, and the magnitude never grows
    beyond the input magnitude under shrinkage.
    """
    pol = policy or EffectPolicy()
    d = float(effect_d)
    if d == 0.0:
        return 0.0

    # 1. Tier weighting: only applied when we actually know the key.
    if key:
        d *= confidence_weight(kind, key)

    # 2. Publication-bias shrinkage.
    if pol.mode == "replication_adjusted":
        f = pol.shrinkage if pol.shrinkage is not None else shrinkage_factor()
        f = max(pol.min_retained, min(1.0, float(f)))
        d *= f

    # 3. Heterogeneity draw.
    if pol.heterogeneity_draw:
        _tau = tau if tau is not None else (pol.default_tau or default_tau())
        if _tau and _tau > 0:
            _rng = rng or random.Random()
            drawn = _rng.gauss(d, _tau)
            # Keep the sign: a heterogeneity draw should not flip a real effect.
            d = drawn if (drawn * d) > 0 else d * 0.25
    return d


# =============================================================================
# AUDIT SURFACE
# =============================================================================

def _kb_tables() -> Dict[str, Dict[str, Any]]:
    """Load the knowledge-base tables, tolerating an absent module."""
    try:
        from . import scientific_knowledge_base as skb  # type: ignore
    except Exception:  # pragma: no cover - import styles differ across entrypoints
        try:
            import scientific_knowledge_base as skb  # type: ignore
        except Exception:
            return {}
    return {
        "meta": getattr(skb, "META_ANALYTIC_DB", {}) or {},
        "game": getattr(skb, "GAME_CALIBRATIONS", {}) or {},
        "norm": getattr(skb, "CONSTRUCT_NORMS", {}) or {},
        "culture": getattr(skb, "CULTURAL_ADJUSTMENTS", {}) or {},
        "rt": getattr(skb, "RESPONSE_TIME_NORMS", {}) or {},
        "order": getattr(skb, "ORDER_EFFECTS", {}) or {},
    }


def audit_table() -> List[Dict[str, Any]]:
    """One row per knowledge-base entry: kind, key, citation, tier, provenance.

    This is what the app shows users who ask "where does this number come from?".
    """
    rows: List[Dict[str, Any]] = []
    for kind, table in _kb_tables().items():
        for key, entry in table.items():
            prov = provenance_of(kind, key)
            rows.append({
                "kind": kind,
                "key": key,
                "citation": getattr(entry, "source", "") or "",
                "status": prov.status if prov else UNVERIFIED,
                "weight": prov.weight() if prov else TIER_WEIGHT[UNVERIFIED],
                "doi": prov.doi if prov else "",
                "url": prov.url if prov else "",
                "verified_on": prov.verified_on if prov else "",
                "quote": prov.quote if prov else "",
                "corrected": dict(prov.corrected) if prov else {},
                "note": prov.note if prov else "",
            })
    rows.sort(key=lambda r: (_TIER_ORDER.index(r["status"]) if r["status"] in _TIER_ORDER else 99,
                            r["kind"], r["key"]))
    return rows


def coverage_summary() -> Dict[str, Any]:
    """Headline verification counts, for the audit page and for tests."""
    rows = audit_table()
    by_status: Dict[str, int] = {t: 0 for t in _TIER_ORDER}
    by_kind: Dict[str, Dict[str, int]] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        k = by_kind.setdefault(r["kind"], {t: 0 for t in _TIER_ORDER})
        k[r["status"]] = k.get(r["status"], 0) + 1
    total = len(rows)
    sourced = by_status.get(VERIFIED, 0) + by_status.get(CORRECTED, 0) + by_status.get(PARTIAL, 0)
    return {
        "total_entries": total,
        "by_status": by_status,
        "by_kind": by_kind,
        "sourced_entries": sourced,
        "sourced_fraction": (sourced / total) if total else 0.0,
        "shrinkage_factor": shrinkage_factor(),
        "shrinkage_verified": bool(_SHRINKAGE.get("factor")),
        "default_tau": default_tau(),
    }


def honesty_notice() -> str:
    """One paragraph the app can show verbatim next to any literature claim."""
    s = coverage_summary()
    pct = 100.0 * s["sourced_fraction"]
    bias = ("No publication-bias correction is applied, because the correction "
            "factor itself has not been verified."
            if not s["shrinkage_verified"] else
            f"Literature effects are shrunk by {s['shrinkage_factor']:.2f} toward "
            "what a replication would find.")
    return (
        f"{s['sourced_entries']} of {s['total_entries']} calibration entries "
        f"({pct:.1f}%) carry a verification record. For the rest, the citation has "
        "NOT been checked against the source and the exact value has not been "
        "transcribed from the paper, so it should be read as a documented prior "
        "rather than a published fact; where the simulator consults one it damps "
        f"how hard that number may push the data. {bias} The provenance table "
        "gives the tier of every number behind a run."
    )


# Import the verified records. Kept in a separate module so the verification pass
# can grow without touching this logic, and so a syntax error in the data file can
# never take down the engine (see CLAUDE.md "Import Resilience").
try:
    from .empirical_provenance_data import install as _install_provenance  # type: ignore
    _install_provenance(register, set_shrinkage, Provenance,
                        VERIFIED, PARTIAL, CITED_UNCHECKED, CORRECTED, UNVERIFIED)
except Exception:  # pragma: no cover - registry degrades to "everything unverified"
    pass
