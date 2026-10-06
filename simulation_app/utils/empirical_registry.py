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

_TIER_ORDER = (MEASURED, VERIFIED, CORRECTED, PARTIAL, CITED_UNCHECKED, UNVERIFIED)

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
