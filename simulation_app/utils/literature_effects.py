"""
Literature Effects — match a condition to a published effect, by meaning.
========================================================================

The engine detects condition effects with hand-written keyword rules. They are
good, but they are a closed set: an uploaded study whose conditions are called
`descriptive_norm_high` / `descriptive_norm_low`, or `mortality_salience` /
`control`, matches nothing, and the simulated experiment then has NO treatment
effect at all. That is the worst possible failure for a simulator — the user's
design silently becomes a null design.

This module is the last-resort layer underneath those rules. It matches the
condition label and the dependent variable against the 187 entries of
`META_ANALYTIC_DB` on content (construct, paradigm, domain and key tokens), and
returns the published effect size for the best match, already passed through
`empirical_registry.adjust_effect()` so that:

  * an entry whose numbers were never verified against the source pushes the data
    less hard than one that was,
  * the effect is shrunk toward what a replication would find rather than what the
    original paper reported, and
  * each run draws its own effect from the between-study distribution, so two runs
    of the same design differ the way two real labs differ.

It returns None whenever the match is not confident. Guessing an effect for a
condition we do not recognise would be worse than returning nothing, because the
caller's own fallback is explicit and inspectable.
"""
from __future__ import annotations

__version__ = "1.1.0"

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .construct_matcher import tokenize, _stem  # type: ignore
except Exception:  # pragma: no cover
    try:
        from construct_matcher import tokenize, _stem  # type: ignore
    except Exception:
        import re as _re

        def tokenize(text: str) -> List[str]:  # type: ignore
            return [t for t in _re.split(r"[^a-z0-9]+", str(text or "").lower()) if len(t) > 2]

        def _stem(tok: str) -> str:  # type: ignore
            return tok

try:
    from .empirical_registry import (  # type: ignore
        EffectPolicy, adjust_effect, status_of, provenance_of, policy_factor,
    )
except Exception:  # pragma: no cover
    try:
        from empirical_registry import (  # type: ignore
            EffectPolicy, adjust_effect, status_of, provenance_of, policy_factor,
        )
    except Exception:
        EffectPolicy = None  # type: ignore

        def adjust_effect(effect_d: float, **_kw: Any) -> float:  # type: ignore
            return float(effect_d)

        def status_of(kind: str, key: str) -> str:  # type: ignore
            return "unverified"

        def provenance_of(kind: str, key: str):  # type: ignore
            return None

        def policy_factor(policy: Any = None) -> float:  # type: ignore
            return 1.0


try:
    from . import paradigm_coverage as _pcov  # type: ignore
except Exception:  # pragma: no cover
    try:
        import paradigm_coverage as _pcov  # type: ignore
    except Exception:
        _pcov = None  # type: ignore


@dataclass
class EffectMatch:
    """A matched published effect, with everything needed to audit it."""
    key: str
    effect_d: float              # after tier weighting, shrinkage and the draw
    published_d: float           # the entry's own point estimate
    score: float
    source: str
    construct: str
    paradigm: str
    status: str
    matched_tokens: Tuple[str, ...]
    #: Set when the match came from a curated condition-label rule (not a content match).
    rule: str = ""
    #: The engine flips the sign of a polarity-aware match for symptom-type DVs.
    polarity_aware: bool = False
    shrinkage: float = 1.0       # publication-bias factor the policy applied

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "effect_d": round(self.effect_d, 4),
            "published_d": round(self.published_d, 4),
            "score": round(self.score, 3),
            "source": self.source,
            "construct": self.construct,
            "paradigm": self.paradigm,
            "verification": self.status,
            "shrinkage_factor": round(self.shrinkage, 4),
            "matched_tokens": list(self.matched_tokens),
            **({"rule": self.rule} if self.rule else {}),
        }


class _EffectIndex:
    """IDF index over META_ANALYTIC_DB, built once from whatever the table holds."""

    def __init__(self, db: Dict[str, Any]) -> None:
        self.entries = db
        self.doc_tokens: Dict[str, set] = {}
        df: Dict[str, int] = {}
        # Entries added by `paradigm_coverage` are reached only through its curated label
        # rules: a content match on their wording would let them join (and shift the
        # document frequencies of) the open-ended ranking that the older entries rely on.
        _rule_only = getattr(_pcov, "RULE_ONLY_KEYS", frozenset()) if _pcov is not None else frozenset()
        for key, e in db.items():
            if key in _rule_only:
                continue
            toks = set()
            for src in (key,
                        getattr(e, "construct", "") or "",
                        getattr(e, "paradigm", "") or "",
                        getattr(e, "domain", "") or ""):
                toks.update(_stem(t) for t in tokenize(src))
            # moderator level names carry real signal ("ingroup", "outgroup", "high")
            for mod in (getattr(e, "moderators", {}) or {}):
                toks.update(_stem(t) for t in tokenize(mod))
            self.doc_tokens[key] = toks
            for t in toks:
                df[t] = df.get(t, 0) + 1
        self.df = dict(df)
        n_docs = max(1, len(self.doc_tokens))
        self.n_docs = n_docs
        self.idf = {t: math.log((n_docs + 1) / (c + 0.5)) for t, c in df.items()}

    def rank(self, tokens: Sequence[str]) -> List[Tuple[str, float, Tuple[str, ...]]]:
        q = {_stem(t) for t in tokens}
        if not q:
            return []
        out = []
        for key, toks in self.doc_tokens.items():
            shared = q & toks
            if not shared:
                continue
            raw = sum(self.idf.get(t, 1.0) for t in shared)
            out.append((key, raw / (math.sqrt(len(toks)) or 1.0), tuple(sorted(shared))))
        out.sort(key=lambda r: (-r[1], r[0]))
        return out


@lru_cache(maxsize=1)
def _index() -> Optional[_EffectIndex]:
    try:
        from . import scientific_knowledge_base as skb  # type: ignore
    except Exception:  # pragma: no cover
        try:
            import scientific_knowledge_base as skb  # type: ignore
        except Exception:
            return None
    db = getattr(skb, "META_ANALYTIC_DB", {}) or {}
    return _EffectIndex(db) if db else None


#: Minimum match score for a MULTI-token match. Deliberately strict: a wrong
#: effect is worse than none, because the caller has an explicit fallback and this
#: layer does not.
MATCH_THRESHOLD = 1.8

#: A single shared word is never enough. Tested against real condition labels, the
#: one-word route matched "mortality_salience" to base-rate neglect (on "salience")
#: and "ostracism/need_threat" to stereotype threat (on "threat"): rare-but-generic
#: psychology words are frequent enough to produce confident nonsense. Two shared
#: content words is the smallest gate that did not produce a wrong match in testing,
#: and a missed match merely hands control back to the caller's own rules.
MIN_SHARED_TOKENS = 2


#: At least one shared word must come from the CONDITION label, not only from the
#: dependent variable. Without this, "scarcity_cue" measured by "willingness_to_pay"
#: matched a price-anchoring meta-analysis on "willing"+"pay" while the word that
#: actually names the manipulation matched something else entirely. The condition
#: names the manipulation; the DV cannot identify it alone.
REQUIRE_CONDITION_TOKEN = True


import re as _re_q

#: v1.3.0.6 -- words that mark a label as the plain / reference form of something ("Generic brand",
#: "Standard price", "Basic plan"). A label carrying one names the absence of a manipulation, so it
#: cannot select a paradigm: "Generic brand" used to match brand extension on the word "brand" and
#: gave the comparison arm of "Premium vs Generic brand" a made-up effect of its own.
QUALIFIER_WORDS = frozenset({
    "generic", "standard", "basic", "regular", "plain", "ordinary", "normal", "typical", "common",
    "usual", "unbranded", "nobrand", "stock", "baseline", "usualcare",
})

#: "default" is a paradigm word ("Default option", "Opt-out default") but a qualifier next to a thing
#: ("Default brand", "default plan"): only the second use is vetoed.
_QUALIFIER_PHRASE = _re_q.compile(r"(?<![a-z0-9])default[\s_-]+(?:brand|product|plan|tier|version|supplier|provider)s?(?![a-z0-9])")

#: Words that occur in many labels without naming a manipulation. A label token from this set (or a
#: qualifier) does not count as the paradigm's distinctive token.
GENERIC_LABEL_TOKENS = frozenset({
    "control", "brand", "consumer", "product", "group", "test", "treatment", "condition", "behavior",
    "behaviour", "psychology", "social", "website", "original", "effective", "outcome", "attitude",
    "study", "experiment", "task", "version", "type", "level", "high", "low", "choice", "option",
    "health", "mental", "ment",
})

#: A distinctive label token occurs in at most this many paradigms of the table.
MAX_DISTINCTIVE_DF = 3


def _has_qualifier(condition: str) -> bool:
    """Whether a whole word of the label is a plain / reference qualifier (see QUALIFIER_WORDS)."""
    words = {w.lower() for w in tokenize(condition)}
    return bool(words & QUALIFIER_WORDS) or bool(_QUALIFIER_PHRASE.search(str(condition).lower()))


def _is_confident(idx: "_EffectIndex", shared: Tuple[str, ...], score: float,
                  thr: float, condition_tokens: Optional[set] = None) -> bool:
    """Whether a match is specific enough to override the caller's own fallback."""
    if len(shared) < MIN_SHARED_TOKENS or score < thr:
        return False
    if REQUIRE_CONDITION_TOKEN and condition_tokens is not None:
        from_label = set(shared) & condition_tokens
        if not from_label:
            return False
        # v1.3.0.6: the label must share a DISTINCTIVE token with the paradigm -- rare in the table and
        # not a generic or qualifier word. "brand" (5 paradigms) or "default" next to "brand" is not
        # enough to say which paradigm a label belongs to.
        generic = {_stem(w) for w in GENERIC_LABEL_TOKENS | QUALIFIER_WORDS}
        distinctive = [t for t in from_label if t not in generic and idx.df.get(t, 99) <= MAX_DISTINCTIVE_DF]
        if not distinctive:
            return False
    return True


def _rule_lookup(condition: str, policy: Optional[Any] = None,
                 rng: Optional[random.Random] = None) -> Optional[EffectMatch]:
    """Match a condition LABEL against the curated paradigm phrases, or None.

    Stricter than the content ranking: a whole-word phrase from the label, no
    negator or direction-reversing word, and one unambiguous entry.
    """
    if _pcov is None:
        return None
    try:
        rule = _pcov.match_label_rule(condition)
    except Exception:
        return None
    if rule is None:
        return None
    try:
        from . import scientific_knowledge_base as skb  # type: ignore
    except Exception:  # pragma: no cover
        try:
            import scientific_knowledge_base as skb  # type: ignore
        except Exception:
            return None
    entry = (getattr(skb, "META_ANALYTIC_DB", {}) or {}).get(rule.key)
    if entry is None:
        return None
    published = float(getattr(entry, "effect_d", 0.0) or 0.0) * rule.sign
    if abs(published) < 0.02:
        return None
    tau = float(getattr(entry, "heterogeneity_tau", 0.0) or 0.0)
    adjusted = adjust_effect(published, kind="meta", key=rule.key, policy=policy, rng=rng, tau=tau)
    return EffectMatch(
        key=rule.key, effect_d=adjusted, published_d=published, score=0.0,
        source=getattr(entry, "source", "") or "",
        construct=getattr(entry, "construct", "") or "",
        paradigm=getattr(entry, "paradigm", "") or "",
        status=status_of("meta", rule.key), matched_tokens=(),
        rule="label_phrase", polarity_aware=bool(rule.polarity_aware),
    )


def lookup_curated(condition: str, policy: Optional[Any] = None,
                   rng: Optional[random.Random] = None) -> Optional[EffectMatch]:
    """Match for a paradigm ADDED by `paradigm_coverage`, from the condition label alone.

    The engine consults this before its keyword rules, because the keyword rules know no
    more about these paradigms than a generic valence. Older entries are not eligible here:
    for them the keyword rules stay first, exactly as before.
    """
    _hit = _rule_lookup(condition, policy=policy, rng=rng)
    if _hit is None or _pcov is None or _hit.key not in getattr(_pcov, "RULE_ONLY_KEYS", ()):
        return None
    return _hit


def lookup(
    condition: str = "",
    variable: str = "",
    study_context: str = "",
    policy: Optional[Any] = None,
    rng: Optional[random.Random] = None,
    threshold: Optional[float] = None,
) -> Optional[EffectMatch]:
    """Published effect for a condition/DV pair, or None when unsure.

    `condition` carries most of the signal (it names the manipulation);
    `variable` and `study_context` break ties. A baseline-distribution entry —
    one whose effect_d is 0, like the dictator-game giving distribution — is never
    returned, since it describes a marginal rather than a contrast.
    """
    idx = _index()
    if idx is None:
        return None
    # Curated label phrases first -- except in economic-game designs, whose effects belong to the
    # game models; there the older content ranking runs alone, exactly as before.
    _hit = None
    if not (_pcov is not None and _pcov.is_economic_game_text(f"{condition} {variable} {study_context}")):
        _hit = _rule_lookup(condition, policy=policy, rng=rng)
    if _hit is not None:
        return _hit
    if _has_qualifier(condition):
        return None      # a plain / reference label ("Generic brand") names no manipulation
    thr = MATCH_THRESHOLD if threshold is None else float(threshold)
    cond_tokens = {_stem(t) for t in tokenize(condition)}
    tokens = tokenize(condition) * 2 + tokenize(variable) + tokenize(study_context)
    if not tokens:
        return None
    ranked = idx.rank(tokens)
    for key, score, shared in ranked:
        if score < thr:
            return None          # ranked by score: nothing below can qualify
        if not _is_confident(idx, shared, score, thr, cond_tokens):
            continue
        entry = idx.entries[key]
        published = float(getattr(entry, "effect_d", 0.0) or 0.0)
        if abs(published) < 0.02:
            continue           # a marginal, not a contrast
        status = status_of("meta", key)
        # An entry that reports no tau gets the policy's default, not "no draw".
        tau = float(getattr(entry, "heterogeneity_tau", 0.0) or 0.0) or None
        adjusted = adjust_effect(published, kind="meta", key=key,
                                 policy=policy, rng=rng, tau=tau)
        return EffectMatch(
            key=key, effect_d=adjusted, published_d=published, score=score,
            source=getattr(entry, "source", "") or "",
            construct=getattr(entry, "construct", "") or "",
            paradigm=getattr(entry, "paradigm", "") or "",
            status=status, matched_tokens=shared,
            shrinkage=policy_factor(policy),
        )
    return None


def explain(condition: str = "", variable: str = "", top_k: int = 5) -> List[Dict[str, Any]]:
    """Ranked candidates with scores — for the audit page and for debugging."""
    idx = _index()
    if idx is None:
        return []
    _cond = {_stem(t) for t in tokenize(condition)}
    ranked = idx.rank(tokenize(condition) * 2 + tokenize(variable))[: max(1, int(top_k))]
    out = []
    for key, score, shared in ranked:
        _ok = _is_confident(idx, shared, score, MATCH_THRESHOLD, _cond)
        e = idx.entries[key]
        out.append({
            "key": key,
            "score": round(score, 3),
            "published_d": getattr(e, "effect_d", 0.0),
            "source": getattr(e, "source", ""),
            "verification": status_of("meta", key),
            "matched_tokens": list(shared),
            "document_frequency": {t: idx.df.get(t, 0) for t in shared},
            "would_be_used": _ok,
        })
    return out
