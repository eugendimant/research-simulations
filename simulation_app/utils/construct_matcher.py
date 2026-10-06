"""
Construct Matcher — map an arbitrary survey item to a published construct norm.
==============================================================================

`scientific_knowledge_base.CONSTRUCT_NORMS` holds 201 published means and SDs, but
the engine could only reach about 40 of them, through a hand-written substring map
(`'brand_loyal' -> 'brand_loyalty'`). An uploaded survey whose variable is called
`PSS4_1`, `stress_total` or `How stressed have you felt this week?` matched nothing,
so the published norm went unused and the item fell back to a generic prior.

This module matches on CONTENT instead of on an exact substring:

* the variable name, the question text, the scale name and the item wording are all
  searched, with underscores and camelCase split into words;
* each norm contributes its construct key, its scale name and an alias list, and
  matching is by token overlap with inverse-document-frequency weighting, so a rare
  token like "loneliness" counts far more than a common one like "scale";
* instrument acronyms (PSS, UCLA, SWLS, PHQ, GAD, UWES, …) are matched exactly,
  because an acronym in a variable name is the strongest possible signal;
* a match must clear a score threshold, and `match()` returns the score, so the
  caller can require a confident match before applying a norm.

The matcher is data-driven: it builds its index from whatever is in CONSTRUCT_NORMS
at import time, so adding a norm makes it reachable with no code change.
"""
from __future__ import annotations

__version__ = "1.0.0"

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_WORD = re.compile(r"[^a-z0-9]+")
_DIGIT_TAIL = re.compile(r"\d+$")

#: Tokens that carry no construct information and must never drive a match.
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "my",
    "your", "you", "i", "me", "is", "are", "was", "do", "does", "did", "how",
    "what", "which", "please", "rate", "scale", "item", "items", "question",
    "questions", "survey", "study", "total", "score", "mean", "sum", "avg",
    "average", "response", "answer", "select", "choose", "indicate", "following",
    "statement", "statements", "agree", "disagree", "much", "very", "feel",
    "felt", "think", "about", "that", "this", "these", "those", "have", "has",
    "been", "be", "at", "as", "it", "its", "not", "no", "yes", "r", "rev",
    "reverse", "reversed", "subscale", "short", "form", "version", "revised",
}

#: Instrument acronyms worth matching exactly when they appear in a variable name.
#: Derived from the scale names in CONSTRUCT_NORMS at build time; this list only
#: supplies spellings the scale name itself does not contain.
_EXTRA_ACRONYMS: Dict[str, Tuple[str, ...]] = {
    "pss": ("perceived_stress",),
    "swls": ("life_satisfaction",),
    "ucla": ("loneliness",),
    "phq": ("depression",),
    "gad": ("generalized_anxiety",),
    "rses": ("self_esteem",),
    "bfi": ("extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"),
    "ipip": ("extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"),
    "mbi": ("emotional_exhaustion", "depersonalization", "personal_accomplishment"),
    "uwes": ("work_engagement",),
    "panas": ("positive_affect", "negative_affect"),
    "iri": ("empathic_concern", "perspective_taking", "personal_distress", "fantasy"),
    "sdo": ("social_dominance_orientation",),
    "rwa": ("right_wing_authoritarianism",),
    "mfq": ("moral_foundations_care", "moral_foundations_fairness",
            "moral_foundations_loyalty", "moral_foundations_authority",
            "moral_foundations_purity"),
    "nfc": ("need_for_cognition",),
    "crt": ("cognitive_reflection",),
    "maas": ("mindful_attention",),
    "bis": ("impulsivity",),
    "sd3": ("machiavellianism", "narcissism", "psychopathy"),
}


def tokenize(text: str) -> List[str]:
    """Split a variable name or question into informative lowercase tokens."""
    if not text:
        return []
    s = _CAMEL.sub(" ", str(text))
    s = _NON_WORD.sub(" ", s.lower())
    out = []
    for tok in s.split():
        tok = _DIGIT_TAIL.sub("", tok)        # PSS4_1 -> pss
        if len(tok) < 2 or tok in _STOP or tok.isdigit():
            continue
        out.append(tok)
    return out


def _stem(tok: str) -> str:
    """Crude suffix strip so 'stressed'/'stress' and 'loneliness'/'lonely' meet."""
    for suf in ("iness", "ness", "ility", "ional", "ingly", "ing", "edly",
                "ed", "es", "s", "ly", "ion", "ity", "al"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


@dataclass
class ConstructMatch:
    key: str
    score: float
    matched_tokens: Tuple[str, ...]
    via_acronym: bool = False
    scale_name: str = ""


class _Index:
    """Inverse-document-frequency index over the construct-norm table."""

    def __init__(self, norms: Dict[str, Any]) -> None:
        self.doc_tokens: Dict[str, set] = {}
        self.scale_names: Dict[str, str] = {}
        self.acronyms: Dict[str, List[str]] = {}
        df: Dict[str, int] = {}
        for key, norm in norms.items():
            scale_name = getattr(norm, "scale_name", "") or ""
            construct = getattr(norm, "construct", "") or key
            toks = set()
            for src in (key, construct, scale_name):
                toks.update(_stem(t) for t in tokenize(src))
            self.doc_tokens[key] = toks
            self.scale_names[key] = scale_name
            for t in toks:
                df[t] = df.get(t, 0) + 1
            # an all-caps instrument name in the scale field is an acronym
            for word in re.findall(r"\b[A-Z][A-Z0-9\-]{1,9}\b", scale_name):
                a = word.lower().replace("-", "")
                self.acronyms.setdefault(a, []).append(key)
        # Resolve the hand-listed acronyms against whatever keys the table
        # actually uses: CONSTRUCT_NORMS names its entries
        # `big_five_extraversion`, `positive_affect_panas`, so an exact-key lookup
        # would silently drop most of this list.
        for acro, wanted in _EXTRA_ACRONYMS.items():
            for want in wanted:
                if want in norms:
                    self.acronyms.setdefault(acro, []).append(want)
                    continue
                for key in norms:
                    if want in key:
                        self.acronyms.setdefault(acro, []).append(key)
        self.acronym_tokens = set(self.acronyms)
        n_docs = max(1, len(self.doc_tokens))
        self.idf = {t: math.log((n_docs + 1) / (c + 0.5)) for t, c in df.items()}

    def score(self, query_tokens: Sequence[str]) -> List[ConstructMatch]:
        q = {_stem(t) for t in query_tokens}
        if not q:
            return []
        results: List[ConstructMatch] = []
        for key, toks in self.doc_tokens.items():
            shared = q & toks
            if not shared:
                continue
            raw = sum(self.idf.get(t, 1.0) for t in shared)
            # normalise by the norm's own token count so a short, precise construct
            # key is not beaten by a long scale name that shares one common word
            norm_len = math.sqrt(len(toks)) or 1.0
            results.append(ConstructMatch(
                key=key,
                score=raw / norm_len,
                matched_tokens=tuple(sorted(shared)),
                scale_name=self.scale_names.get(key, ""),
            ))
        results.sort(key=lambda m: (-m.score, m.key))
        return results


@lru_cache(maxsize=1)
def _index() -> Optional[_Index]:
    try:
        from . import scientific_knowledge_base as skb  # type: ignore
    except Exception:  # pragma: no cover
        try:
            import scientific_knowledge_base as skb  # type: ignore
        except Exception:
            return None
    norms = getattr(skb, "CONSTRUCT_NORMS", {}) or {}
    return _Index(norms) if norms else None


#: Minimum score for a match to be used. Set so that a single shared common word
#: never qualifies, but one distinctive construct word (or an acronym) does.
MATCH_THRESHOLD = 1.4


def match(
    variable_name: str = "",
    question_text: str = "",
    item_text: str = "",
    scale_hint: str = "",
    threshold: Optional[float] = None,
) -> Optional[ConstructMatch]:
    """Best construct-norm key for an item, or None when nothing matches well.

    An instrument acronym in the variable name wins outright: `PSS4_1` is a
    Perceived Stress Scale item regardless of what the question text says. When a
    single acronym maps to several subscales (BFI, MBI, MFQ), the acronym narrows
    the candidate set and the remaining text decides between them.
    """
    idx = _index()
    if idx is None:
        return None
    thr = MATCH_THRESHOLD if threshold is None else float(threshold)

    name_tokens = tokenize(variable_name)
    all_tokens = (name_tokens + tokenize(question_text) +
                  tokenize(item_text) + tokenize(scale_hint))
    if not all_tokens:
        return None

    # 1. acronym route
    for tok in name_tokens:
        cands = idx.acronyms.get(tok)
        if not cands:
            continue
        cands = list(dict.fromkeys(cands))
        if len(cands) == 1:
            return ConstructMatch(key=cands[0], score=99.0,
                                  matched_tokens=(tok,), via_acronym=True,
                                  scale_name=idx.scale_names.get(cands[0], ""))
        ranked = [m for m in idx.score(all_tokens) if m.key in cands]
        # Subscale disambiguation: an abbreviated cue ("extra" for extraversion,
        # "exh" for emotional_exhaustion) shares no whole token with the norm, so
        # token overlap alone cannot separate BFI/MBI/MFQ subscales. Prefix
        # containment can, and a prefix hit outranks a token hit here because the
        # acronym has already fixed the instrument.
        _prefix_hits = []
        for _cand in cands:
            _cand_toks = idx.doc_tokens.get(_cand, set())
            for _qt in all_tokens:
                if len(_qt) < 3 or _qt in getattr(idx, "acronym_tokens", ()):
                    continue  # the acronym matches every candidate; it cannot choose
                if any(_ct.startswith(_qt) or _qt.startswith(_ct)
                       for _ct in _cand_toks if len(_ct) >= 3):
                    _prefix_hits.append((len(_qt), _cand, _qt))
                    break
        if _prefix_hits:
            _prefix_hits.sort(reverse=True)
            _, _best_key, _cue = _prefix_hits[0]
            return ConstructMatch(key=_best_key, score=99.0,
                                  matched_tokens=(_cue,), via_acronym=True,
                                  scale_name=idx.scale_names.get(_best_key, ""))
        if ranked:
            best = ranked[0]
            return ConstructMatch(key=best.key, score=max(best.score, thr),
                                  matched_tokens=best.matched_tokens,
                                  via_acronym=True, scale_name=best.scale_name)
        return ConstructMatch(key=cands[0], score=thr,
                              matched_tokens=(tok,), via_acronym=True,
                              scale_name=idx.scale_names.get(cands[0], ""))

    # 2. content route
    ranked = idx.score(all_tokens)
    if not ranked:
        return None
    best = ranked[0]
    if best.score < thr:
        return None
    # require a clear winner: an ambiguous tie is no match at all
    if len(ranked) > 1 and ranked[1].score > best.score * 0.92:
        if not set(best.matched_tokens) - set(ranked[1].matched_tokens):
            return None
    return best


def match_all(
    variable_name: str = "",
    question_text: str = "",
    top_k: int = 5,
) -> List[ConstructMatch]:
    """Ranked candidates, for diagnostics and for the audit page."""
    idx = _index()
    if idx is None:
        return []
    toks = tokenize(variable_name) + tokenize(question_text)
    return idx.score(toks)[: max(1, int(top_k))]


def coverage() -> Dict[str, Any]:
    """How many norms the matcher can reach, for tests and the audit page."""
    idx = _index()
    if idx is None:
        return {"norms_indexed": 0, "acronyms_indexed": 0}
    return {
        "norms_indexed": len(idx.doc_tokens),
        "acronyms_indexed": len(idx.acronyms),
        "distinct_tokens": len(idx.idf),
    }
