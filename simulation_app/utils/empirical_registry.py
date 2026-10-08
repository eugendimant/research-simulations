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
import json as _json
import os as _os
import random


# =============================================================================
# VERIFICATION TIERS
# =============================================================================

#: Computed by us from a dataset we hold, with the script, the file and its
#: SHA-256 recorded. This is the ONLY tier that needs no publication behind it,
#: because the evidence is the data itself and the derivation is reproducible.
MEASURED = "measured"
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

# ── Recall tiers ─────────────────────────────────────────────────────────────
# These record a DIFFERENT kind of evidence from the tiers above: a model that has
# read a great deal of this literature was asked, entry by entry, whether it
# recognises the citation and whether the stored number matches the published or
# meta-analytic estimate. That is a judgement from memory, not a source check, and
# it is kept in its own tier band so the two can never be confused. A recall tier
# NEVER grants `may_set_magnitude` and always weighs less than `CITED_UNCHECKED`.
#
#: Citation recognised and the value is consistent with the recalled literature.
RECALL_CONSISTENT = "recall_consistent"
#: Recall says a field was wrong; the entry has been changed and the old value is
#: in `Provenance.corrected` under "<field>_was".
RECALL_CORRECTED = "recall_corrected"
#: The citation is plausible but the number could not be judged from memory.
RECALL_UNCERTAIN = "recall_uncertain"
#: The citation could not be placed at all. Weaker than having no record, because
#: a specific-looking citation nobody can place is itself a warning sign.
UNRECOGNIZED = "unrecognized"

_TIER_ORDER = (MEASURED, VERIFIED, CORRECTED, PARTIAL, CITED_UNCHECKED,
               RECALL_CONSISTENT, RECALL_CORRECTED, RECALL_UNCERTAIN,
               UNVERIFIED, UNRECOGNIZED)

#: Tiers that rest on recall rather than on a source or a dataset.
RECALL_TIERS = (RECALL_CONSISTENT, RECALL_CORRECTED, RECALL_UNCERTAIN, UNRECOGNIZED)

#: How much weight the engine should give an entry's numbers, by tier.
#: Unverified entries are not discarded — they are the best prior available — but
#: their deviation from a neutral default is damped, so an invented number can
#: never drive a simulation as hard as a sourced one.
TIER_WEIGHT: Dict[str, float] = {
    MEASURED: 1.00,
    VERIFIED: 1.00,
    CORRECTED: 1.00,
    PARTIAL: 0.85,
    CITED_UNCHECKED: 0.70,
    RECALL_CONSISTENT: 0.68,
    RECALL_CORRECTED: 0.68,
    RECALL_UNCERTAIN: 0.58,
    UNVERIFIED: 0.55,
    UNRECOGNIZED: 0.45,
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


def register_recall(key: str, tier: str, note: str, *,
                    source: str = "", checked_fields: Tuple[str, ...] = (),
                    corrected: Optional[Dict[str, Any]] = None,
                    audited_on: str = "") -> bool:
    """Record a recall-based audit verdict for a knowledge-base entry.

    Separate from `register` so the invariants of this evidence class are enforced
    in one place and cannot be bypassed by a data file:

    * the tier must be one of `RECALL_TIERS` — a recall pass can never write
      `VERIFIED`, `MEASURED` or `CORRECTED`, which are reserved for a source that
      was actually read or a dataset we hold;
    * a verdict with no explanatory note is not a verdict, so it is dropped;
    * `doi`, `url` and `quote` are left empty by construction, because nothing was
      fetched or quoted.

    Returns True when the record was installed.
    """
    if tier not in RECALL_TIERS or not str(note).strip():
        return False
    existing = PROVENANCE.get(key)
    if existing is not None and existing.status not in RECALL_TIERS:
        # A measured, verified, corrected, partial or cited record was produced by
        # looking at something. A recall verdict must never replace it, however
        # much later it arrives.
        return False
    register(key, Provenance(
        status=tier,
        source=source,
        doi="", url="", quote="",
        verified_on="",          # nothing was verified; see `note`
        checked_fields=tuple(checked_fields),
        corrected=dict(corrected or {}),
        note=f"RECALL, NOT SOURCE-VERIFIED ({audited_on or 'undated'}): {note}",
    ))
    return True


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
#: Two ways in. `set_shrinkage` takes a SOURCE-VERIFIED figure (MEASURED, VERIFIED
#: or CORRECTED provenance). `set_recalled_shrinkage` takes a figure recalled from
#: the replication literature and installs it in the recall band, so the tier says
#: "recalled, unchecked" for as long as nobody has read the sources. With neither
#: called the policy falls back to NO shrinkage (1.0).
_SHRINKAGE: Dict[str, Any] = {
    "factor": None,          # ratio replication_d / published_d
    "default_tau": None,     # typical between-study SD on the d scale
    "provenance_key": "policy:publication_bias",
    "by_evidence": {},       # evidence type -> ratio, for the audit surface only
}


class ProvenanceTooWeak(ValueError):
    """Raised when a value would be installed on evidence that does not support it."""


def set_shrinkage(factor: float, default_tau: float, prov: Provenance,
                  tau_scale: str = "d") -> None:
    """Install a publication-bias shrinkage factor and its provenance.

    This constant multiplies EVERY literature effect the simulator produces, so it
    is the one number in the system where a wrong value does the most damage while
    being the hardest to notice. Three guards, all deliberate:

    * the provenance must be MEASURED, VERIFIED or CORRECTED — a factor may not
      enter on a citation nobody read, and certainly not on a search summary;
    * a publication-derived record must carry the verbatim source sentence, so a
      tier claim can never outrun its evidence;
    * tau must declare its scale. tau is reported on d, on Fisher's z and on log
      odds and is not comparable across them, and I-squared is a proportion of
      observed variance rather than a magnitude at all. Applying one as the other
      would silently miscalibrate every effect in the system.
    """
    if prov.status not in (MEASURED, VERIFIED, CORRECTED):
        raise ProvenanceTooWeak(
            f"shrinkage needs {MEASURED}/{VERIFIED}/{CORRECTED} provenance, "
            f"got {prov.status!r}")
    if prov.status in (VERIFIED, CORRECTED) and not prov.quote.strip():
        raise ProvenanceTooWeak("a verified shrinkage factor must carry the source quote")
    if tau_scale != "d":
        raise ProvenanceTooWeak(
            f"default_tau must be on the d scale, got {tau_scale!r}; convert it "
            "at the call site rather than here, where the scale would be lost")
    _SHRINKAGE["factor"] = float(factor)
    _SHRINKAGE["default_tau"] = float(default_tau)
    register(_SHRINKAGE["provenance_key"], prov)


def set_recalled_shrinkage(factor: float, default_tau: float, note: str, *,
                           by_evidence: Optional[Dict[str, float]] = None,
                           source: str = "", audited_on: str = "",
                           corrected: Optional[Dict[str, Any]] = None) -> bool:
    """Install a publication-bias shrinkage figure that rests on RECALL.

    The recall-band counterpart of `set_shrinkage`. It is honest about what it is:
    the record is written through `register_recall` as `RECALL_UNCERTAIN` (the
    figure is a judgement across sources whose estimates differ by definition, so
    it is the weakest recall tier that still carries a note), with no DOI, URL or
    quote. `shrinkage_verified()` stays False, `coverage_summary()` reports the
    tier, and `honesty_notice()` says the factor is recalled and unchecked.

    Unlike `set_shrinkage` it is refused when a source-verified record is already
    installed, so a recall figure can never overwrite a checked one.
    """
    existing = PROVENANCE.get(_SHRINKAGE["provenance_key"])
    if existing is not None and existing.status not in RECALL_TIERS:
        return False
    if not (0.0 < float(factor) <= 1.0) or float(default_tau) < 0.0:
        return False
    if not register_recall(_SHRINKAGE["provenance_key"], RECALL_UNCERTAIN, note,
                           source=source, checked_fields=("shrinkage", "default_tau"),
                           audited_on=audited_on, corrected=corrected):
        return False
    _SHRINKAGE["factor"] = float(factor)
    _SHRINKAGE["default_tau"] = float(default_tau)
    _SHRINKAGE["by_evidence"] = {str(k): float(v) for k, v in (by_evidence or {}).items()}
    return True


def shrinkage_verified() -> bool:
    """True only when the installed factor rests on a source or a dataset."""
    rec = PROVENANCE.get(_SHRINKAGE["provenance_key"])
    return bool(_SHRINKAGE.get("factor")) and rec is not None and rec.status not in RECALL_TIERS


def shrinkage_tier() -> str:
    """Tier of the installed shrinkage figure; UNVERIFIED when none is installed."""
    rec = PROVENANCE.get(_SHRINKAGE["provenance_key"])
    return rec.status if (rec and _SHRINKAGE.get("factor")) else UNVERIFIED


def shrinkage_factor() -> float:
    """Installed shrinkage factor (source-verified or recalled; see shrinkage_tier), or 1.0 when none."""
    f = _SHRINKAGE.get("factor")
    return float(f) if f else 1.0


def default_tau() -> float:
    t = _SHRINKAGE.get("default_tau")
    return float(t) if t else 0.0


def policy_factor(policy: Optional["EffectPolicy"] = None) -> float:
    """The multiplicative shrinkage `adjust_effect` applies under `policy`.

    1.0 in as_published mode. Exposed so callers can report the factor they used
    without re-deriving the clamp.
    """
    pol = policy or EffectPolicy()
    if pol.mode != "replication_adjusted":
        return 1.0
    f = pol.shrinkage if pol.shrinkage is not None else shrinkage_factor()
    return max(pol.min_retained, min(1.0, float(f)))


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

    The sign of `effect_d` is always preserved, the magnitude never grows beyond
    the input magnitude under shrinkage (the heterogeneity draw can), and in
    replication_adjusted mode the result never falls below
    `min_retained * |effect_d|`.
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

    # 4. Floor. Neither the shrinkage nor the draw may erase a real effect: the
    #    result keeps at least `min_retained` of the input magnitude.
    if pol.mode == "replication_adjusted" and pol.min_retained > 0:
        floor = abs(float(effect_d)) * float(pol.min_retained)
        if abs(d) < floor:
            d = floor if effect_d > 0 else -floor
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
    # Recall-audited entries are counted separately and are deliberately NOT added
    # to `sourced_entries`: a judgement from memory is not a source check, and the
    # honesty notice must not be able to claim otherwise.
    recall_audited = sum(by_status.get(t, 0) for t in RECALL_TIERS)
    return {
        "total_entries": total,
        "by_status": by_status,
        "by_kind": by_kind,
        "sourced_entries": sourced,
        "sourced_fraction": (sourced / total) if total else 0.0,
        "recall_audited_entries": recall_audited,
        "recall_consistent_entries": by_status.get(RECALL_CONSISTENT, 0),
        "recall_corrected_entries": by_status.get(RECALL_CORRECTED, 0),
        "recall_uncertain_entries": by_status.get(RECALL_UNCERTAIN, 0),
        "unrecognized_entries": by_status.get(UNRECOGNIZED, 0),
        "shrinkage_factor": shrinkage_factor(),
        "shrinkage_verified": shrinkage_verified(),
        "shrinkage_active": bool(_SHRINKAGE.get("factor")),
        "shrinkage_tier": shrinkage_tier(),
        "default_tau": default_tau(),
    }


def honesty_notice() -> str:
    """One paragraph the app can show verbatim next to any literature claim."""
    s = coverage_summary()
    pct = 100.0 * s["sourced_fraction"]
    if s["shrinkage_verified"]:
        bias = (f"Literature effects are shrunk by {s['shrinkage_factor']:.2f} toward "
                "what a replication would find.")
    elif s["shrinkage_active"]:
        bias = (f"Effects the tool infers from the literature are shrunk by "
                f"{s['shrinkage_factor']:.2f} toward what a replication would find, "
                "because published estimates are inflated by selective reporting. That "
                "factor is recalled from the replication literature and has NOT been "
                "checked against the sources; it never applies to an effect you specify.")
    else:
        bias = ("No publication-bias correction is applied, because no correction "
                "factor has been installed.")
    recall = ""
    if s.get("recall_audited_entries"):
        recall = (
            f" A further {s['recall_audited_entries']} entries have been audited "
            "against recalled knowledge of the literature rather than against a "
            f"source: {s['recall_consistent_entries']} were consistent with what is "
            f"known, {s['recall_corrected_entries']} were corrected, "
            f"{s['recall_uncertain_entries']} could not be judged from memory and "
            f"{s['unrecognized_entries']} "
            f"{'carries' if s['unrecognized_entries'] == 1 else 'carry'} a citation "
            "that could not be placed. "
            "Recall is not verification, so those entries remain barred from setting "
            "a magnitude on their own.")
    return (
        f"{s['sourced_entries']} of {s['total_entries']} calibration entries "
        f"({pct:.1f}%) carry a verification record. For the rest, the citation has "
        "NOT been checked against the source and the exact value has not been "
        "transcribed from the paper, so it should be read as a documented prior "
        "rather than a published fact; where the simulator consults one it damps "
        f"how hard that number may push the data.{recall} {bias} The provenance "
        "table gives the tier of every number behind a run."
    )


# =============================================================================
# REGISTRY STORE — applicability-guarded entries loaded from JSON
# =============================================================================
# The provenance machinery above annotates the EXISTING knowledge base. This part
# is the store for new, first-class entries, each of which declares the designs it
# is allowed to speak about.
#
# Why applicability is a hard guard rather than advice: nearly every misuse of a
# published number in this codebase has the same shape — a value that is true
# under the conditions it was measured under, applied where those conditions do
# not hold. A pooled ultimatum rejection rate applied flat when rejection is
# conditional on offer size. A public-goods contribution rate applied to every
# round when contributions decay with repetition. A same-keyed block's
# straight-lining rate (3.3%) applied to a mixed-keyed block (0.3%). The guard
# makes those lookups DECLINE instead of returning a plausible wrong number, and
# a declined lookup leaves the engine's existing behaviour untouched.

_REGISTRY_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "registry")


@dataclass(frozen=True)
class Applicability:
    """The conditions under which an entry may be consulted.

    A field left as None means "this entry does not care about that dimension".
    Anything stated must be satisfied by the design, or the entry does not fire.
    """
    scale_points: Optional[Tuple[int, ...]] = None
    items_per_block: Optional[Tuple[int, int]] = None      # inclusive bounds
    keying: Optional[str] = None                           # mixed | same | any
    design: Optional[str] = None                           # between | within | any
    population: Optional[str] = None
    repetition: Optional[str] = None                       # one_shot | repeated | any

    #: Number of stated conditions. Used to break ties toward the narrower entry,
    #: so a 5-point-specific benchmark wins over an any-scale one.
    def specificity(self) -> int:
        return sum(1 for v in (self.scale_points, self.items_per_block, self.keying,
                               self.design, self.population, self.repetition)
                   if v is not None and v != "any")

    def accepts(self, sig: Any) -> bool:
        """True when `sig` (a DesignSignature, or anything with those attributes)
        satisfies every stated condition. An unknown signature value never
        satisfies a stated condition — silence is not agreement."""
        def _get(name):
            return getattr(sig, name, None) if not isinstance(sig, dict) else sig.get(name)

        if self.scale_points is not None:
            sp = _get("scale_points")
            if sp is None or int(sp) not in self.scale_points:
                return False
        if self.items_per_block is not None:
            k = _get("items_per_block")
            lo, hi = self.items_per_block
            if k is None or not (lo <= int(k) <= hi):
                return False
        for name, want in (("keying", self.keying), ("design", self.design),
                           ("population", self.population), ("repetition", self.repetition)):
            if want in (None, "any"):
                continue
            got = _get(name)
            if got is None or str(got) == "unknown" or str(got) != want:
                return False
        return True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Applicability":
        d = d or {}
        sp = d.get("scale_points")
        ipb = d.get("items_per_block")
        return cls(
            scale_points=tuple(int(x) for x in sp) if sp else None,
            items_per_block=(int(ipb[0]), int(ipb[1])) if ipb else None,
            keying=d.get("keying"), design=d.get("design"),
            population=d.get("population"), repetition=d.get("repetition"),
        )


#: Scales a value may be expressed on. Stated explicitly on every entry, with no
#: default, because an effect size silently read on the wrong scale is the most
#: expensive mistake available here.
EFFECT_SCALES = ("d", "fisher_z", "log_odds", "proportion", "raw_points", "rate", "count")


@dataclass(frozen=True)
class RegistryEntry:
    entry_id: str
    quantity: str
    value: float
    effect_scale: str
    applicability: Applicability
    tier: str = UNVERIFIED
    dispersion: Optional[float] = None
    dispersion_kind: str = "none"
    construct: str = ""
    paradigm: Optional[str] = None
    k_studies: Optional[int] = None
    n_participants: Optional[int] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    caveats: Tuple[str, ...] = ()

    def weight(self) -> float:
        return TIER_WEIGHT.get(self.tier, TIER_WEIGHT[UNVERIFIED])

    @property
    def may_set_magnitude(self) -> bool:
        """Only measured or primary-source-verified evidence may set a number.

        Everything else supplies sign and rank order; the engine keeps whatever it
        was already doing for the magnitude. This is what stops the 484 asserted
        entries from quietly becoming the simulator's ground truth.
        """
        return self.tier in (MEASURED, VERIFIED, CORRECTED)


def _parse_entry(d: Dict[str, Any]) -> Optional[RegistryEntry]:
    try:
        scale = str(d["effect_scale"])
        if scale not in EFFECT_SCALES:
            return None
        tier = str(d.get("tier", UNVERIFIED)).lower().replace("t0_", "").replace(
            "t1_", "").replace("t2_", "").replace("t3_", "")
        if tier not in _TIER_ORDER:
            tier = UNVERIFIED
        prov = dict(d.get("provenance") or {})
        # A tier claim may not outrun its evidence.
        if tier == MEASURED and not (prov.get("script") and prov.get("sources")):
            tier = UNVERIFIED
        if tier in (VERIFIED, CORRECTED) and not str(prov.get("quote", "")).strip():
            tier = CITED_UNCHECKED
        return RegistryEntry(
            entry_id=str(d["entry_id"]),
            quantity=str(d["quantity"]),
            value=float(d["value"]),
            effect_scale=scale,
            applicability=Applicability.from_dict(d.get("applicability")),
            tier=tier,
            dispersion=(float(d["dispersion"]) if d.get("dispersion") is not None else None),
            dispersion_kind=str(d.get("dispersion_kind", "none")),
            construct=str(d.get("construct") or ""),
            paradigm=d.get("paradigm"),
            k_studies=d.get("k_studies"),
            n_participants=d.get("n_participants"),
            provenance=prov,
            caveats=tuple(d.get("caveats") or ()),
        )
    except Exception:
        return None


_ENTRIES: Optional[Dict[str, RegistryEntry]] = None


def entries() -> Dict[str, RegistryEntry]:
    """All registry entries, loaded once from `utils/registry/*.json`.

    A malformed file is skipped rather than raised: a bad calibration file must
    never take the app down (CLAUDE.md, "Import Resilience").
    """
    global _ENTRIES
    if _ENTRIES is not None:
        return _ENTRIES
    loaded: Dict[str, RegistryEntry] = {}
    try:
        names = sorted(n for n in _os.listdir(_REGISTRY_DIR) if n.endswith(".json"))
    except Exception:
        names = []
    for name in names:
        try:
            with open(_os.path.join(_REGISTRY_DIR, name), "r", encoding="utf-8") as fh:
                payload = _json.load(fh)
        except Exception:
            continue
        for raw in (payload.get("entries") or []):
            ent = _parse_entry(raw)
            if ent is not None:
                loaded[ent.entry_id] = ent
    _ENTRIES = loaded
    return _ENTRIES


def reload_entries() -> None:
    """Drop the cache; for tests and for a hot calibration edit."""
    global _ENTRIES
    _ENTRIES = None


@dataclass(frozen=True)
class Lookup:
    """A registry hit, with everything the caller needs to decide and to log."""
    entry: RegistryEntry
    value: float

    @property
    def entry_id(self) -> str:
        return self.entry.entry_id

    @property
    def tier(self) -> str:
        return self.entry.tier


def lookup(entry_id: str, signature: Any = None,
           require_magnitude: bool = True) -> Optional[Lookup]:
    """Return the entry for `entry_id` if the design satisfies its applicability.

    Returns None — never a guess — when the entry is absent, when the design does
    not satisfy its conditions, or when `require_magnitude` is set and the entry's
    tier is not allowed to set a number. The caller then keeps whatever it was
    already doing.
    """
    ent = entries().get(str(entry_id))
    if ent is None:
        return None
    if require_magnitude and not ent.may_set_magnitude:
        return None
    if signature is not None and not ent.applicability.accepts(signature):
        return None
    return Lookup(entry=ent, value=ent.value)


def lookup_best(prefix: str, metric: str, signature: Any = None,
                require_magnitude: bool = True) -> Optional[Lookup]:
    """Highest-tier entry matching `<prefix>*.<metric>` that the design satisfies.

    Ties break toward the NARROWER applicability, so a 5-point-specific benchmark
    beats an any-scale one rather than the other way round.
    """
    cands = []
    for ent in entries().values():
        if not ent.entry_id.startswith(prefix) or not ent.entry_id.endswith("." + metric):
            continue
        if require_magnitude and not ent.may_set_magnitude:
            continue
        if signature is not None and not ent.applicability.accepts(signature):
            continue
        cands.append(ent)
    if not cands:
        return None
    cands.sort(key=lambda e: (_TIER_ORDER.index(e.tier) if e.tier in _TIER_ORDER else 99,
                              -e.applicability.specificity()))
    return Lookup(entry=cands[0], value=cands[0].value)


# =============================================================================
# RUN LEDGER
# =============================================================================
# Every registry value that actually changed a number is recorded here, so the
# honesty notice can report the real mix for THIS run rather than the mix across
# the whole table, and so the export metadata can carry it.

@dataclass
class RunLedger:
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, entry_id: str, tier: str, value: float, applied_to: str,
               note: str = "") -> None:
        self.rows.append({"entry_id": entry_id, "tier": tier, "value": float(value),
                          "applied_to": applied_to, "note": note})

    def record_lookup(self, hit: "Lookup", applied_to: str, note: str = "") -> None:
        self.record(hit.entry_id, hit.tier, hit.value, applied_to, note)

    def summary(self) -> Dict[str, Any]:
        by_tier: Dict[str, int] = {}
        for r in self.rows:
            by_tier[r["tier"]] = by_tier.get(r["tier"], 0) + 1
        evidenced = sum(by_tier.get(t, 0) for t in (MEASURED, VERIFIED, CORRECTED))
        return {
            "applied": len(self.rows),
            "by_tier": by_tier,
            "evidenced": evidenced,
            "asserted": len(self.rows) - evidenced,
            "entry_ids": [r["entry_id"] for r in self.rows],
        }

    def notice(self) -> str:
        """One line the app can show about THIS run. Empty when nothing applied."""
        s = self.summary()
        if not s["applied"]:
            return ""
        if s["asserted"] == 0:
            return (f"All {s['applied']} calibration(s) used in this run come from "
                    "measured data or a checked primary source.")
        return (f"{s['evidenced']} of {s['applied']} calibration(s) used in this run "
                f"come from measured data or a checked source; {s['asserted']} rest on "
                "literature assertions that have not been verified.")


# Import the verified records. Kept in a separate module so the verification pass
# can grow without touching this logic, and so a syntax error in the data file can
# never take down the engine (see CLAUDE.md "Import Resilience").
try:
    from .empirical_provenance_data import install as _install_provenance  # type: ignore
    _install_provenance(register, set_shrinkage, Provenance,
                        VERIFIED, PARTIAL, CITED_UNCHECKED, CORRECTED, UNVERIFIED)
except Exception:  # pragma: no cover - registry degrades to "everything unverified"
    pass

# The shrinkage policy rests on recall, so it is installed through the recall band.
# Guarded like the provenance import: without it the policy is simply 1.0 (off).
try:
    from .empirical_provenance_data import RECALLED_SHRINKAGE as _RS  # type: ignore
    set_recalled_shrinkage(_RS["factor"], _RS["default_tau"], _RS["note"],
                           by_evidence=_RS.get("by_evidence"), source=_RS.get("source", ""),
                           audited_on="2026-10-08", corrected=_RS.get("corrected"))
except Exception:  # pragma: no cover
    pass


# =============================================================================
# RECALL AUDIT RECORDS
# =============================================================================
#: The recall audit's data file, relative to the registry directory.
RECALL_AUDIT_FILE = "recall_audit.json"


def load_recall_audit(path: str = "") -> int:
    """Install the recall-audit verdicts. Returns the number of records added.

    The file is data only: a dict with `audited_on` and a `records` map from
    "<kind>:<key>" to {tier, note, source, checked_fields, corrected}. Every record
    goes through `register_recall`, so a file cannot promote an entry past the
    recall band however it is written, and a malformed file installs nothing rather
    than taking the engine down (see CLAUDE.md "Import Resilience").
    """
    target = path or _os.path.join(_REGISTRY_DIR, RECALL_AUDIT_FILE)
    try:
        with open(target, "r", encoding="utf-8") as fh:
            blob = _json.load(fh)
    except Exception:
        return 0
    audited_on = str(blob.get("audited_on", ""))
    records = blob.get("records") or {}
    if not isinstance(records, dict):
        return 0
    added = 0
    for key, rec in records.items():
        if not isinstance(rec, dict):
            continue
        ok = register_recall(
            str(key),
            str(rec.get("tier", "")),
            str(rec.get("note", "")),
            source=str(rec.get("source", "")),
            checked_fields=tuple(rec.get("checked_fields") or ()),
            corrected=rec.get("corrected") or {},
            audited_on=str(rec.get("audited_on") or audited_on),
        )
        added += 1 if ok else 0
    return added


try:  # pragma: no cover - exercised through the registry's own tests
    RECALL_AUDIT_COUNT = load_recall_audit()
except Exception:
    RECALL_AUDIT_COUNT = 0

#: Recall records for the paradigms added in v1.3.0.5 (see `paradigm_coverage`). Same format and
#: same `register_recall` gate as the main audit; a separate file keeps the audit of the original
#: entries untouched.
RECALL_COVERAGE_FILE = "recall_coverage_v1305.json"
try:  # pragma: no cover - exercised through the coverage tests
    RECALL_AUDIT_COUNT += load_recall_audit(_os.path.join(_REGISTRY_DIR, RECALL_COVERAGE_FILE))
except Exception:
    pass

#: v1.3.0.6 recall records written after search-summary corroboration (relabels two entries that
#: cite no meta-analysis and scopes the ostracism magnitude). Loaded last so it supersedes the
#: earlier recall verdicts for those keys; still gated by `register_recall`.
RECALL_EVIDENCE_FILE = "recall_evidence_v1306.json"
try:  # pragma: no cover - exercised through tests/test_evidence_v1306.py
    RECALL_AUDIT_COUNT += load_recall_audit(_os.path.join(_REGISTRY_DIR, RECALL_EVIDENCE_FILE))
except Exception:
    pass
