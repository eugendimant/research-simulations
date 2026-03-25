"""
HBS Stylometric Voice Fingerprint Engine
==========================================

Per-participant writing identity system. Each simulated participant receives
a unique stylometric fingerprint (sentence length, vocabulary richness,
typo patterns, filler words, capitalization style) that constrains all
their open-ended responses to sound like the same real person.

Based on: Pennebaker & King (1999) LIWC, Denscombe (2008) response lengths,
Zipf's law for vocabulary naturalness.

Architecture: DANEEL's cognitive realism pillar — the hardest human signal
to fake is sounding like the same person across all OE responses.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["StylometricFingerprint", "HBSStylometricEngine"]

# ---------------------------------------------------------------------------
# QWERTY adjacency map — loaded from data file with hardcoded fallback
# ---------------------------------------------------------------------------

_QWERTY_ADJACENCY: Dict[str, List[str]] = {}

def _load_qwerty_adjacency() -> Dict[str, List[str]]:
    """Load QWERTY adjacency from data file; fall back to hardcoded subset."""
    global _QWERTY_ADJACENCY
    if _QWERTY_ADJACENCY:
        return _QWERTY_ADJACENCY

    _data_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "hbs_qwerty_adjacency.json"
    )
    try:
        with open(_data_path, "r", encoding="utf-8") as f:
            _QWERTY_ADJACENCY = json.load(f)
            return _QWERTY_ADJACENCY
    except Exception:
        pass

    # Hardcoded fallback for common pairs
    _QWERTY_ADJACENCY = {
        "a": ["s", "q", "z"], "b": ["v", "n", "g"], "c": ["x", "v", "d"],
        "d": ["s", "f", "e"], "e": ["w", "r", "d"], "f": ["d", "g", "r"],
        "g": ["f", "h", "t"], "h": ["g", "j", "y"], "i": ["u", "o", "k"],
        "j": ["h", "k", "u"], "k": ["j", "l", "i"], "l": ["k", "o", "p"],
        "m": ["n", "k"], "n": ["b", "m", "h"], "o": ["i", "p", "l"],
        "p": ["o", "l"], "q": ["w", "a"], "r": ["e", "t", "f"],
        "s": ["a", "d", "w"], "t": ["r", "y", "g"], "u": ["y", "i", "j"],
        "v": ["c", "b", "f"], "w": ["q", "e", "s"], "x": ["z", "c", "s"],
        "y": ["t", "u", "h"], "z": ["a", "x"],
    }
    return _QWERTY_ADJACENCY


# ---------------------------------------------------------------------------
# Demographic prior tables
# ---------------------------------------------------------------------------

# Education -> stylometric parameter ranges
_EDUCATION_PRIORS = {
    "<HS": {
        "mean_sentence_len": (8, 12), "punct_rate": (1.0, 3.0),
        "vocab_richness": (0.40, 0.55), "filler_rate": (0.04, 0.07),
        "typo_rate": (0.03, 0.05), "mean_word_length": (3.5, 4.2),
        "contraction_rate": (0.7, 0.95), "hedge_rate": (0.01, 0.03),
    },
    "HS": {
        "mean_sentence_len": (10, 15), "punct_rate": (2.0, 5.0),
        "vocab_richness": (0.50, 0.65), "filler_rate": (0.03, 0.05),
        "typo_rate": (0.02, 0.04), "mean_word_length": (3.8, 4.5),
        "contraction_rate": (0.5, 0.85), "hedge_rate": (0.02, 0.04),
    },
    "Some college": {
        "mean_sentence_len": (12, 16), "punct_rate": (4.0, 7.0),
        "vocab_richness": (0.60, 0.72), "filler_rate": (0.02, 0.04),
        "typo_rate": (0.01, 0.03), "mean_word_length": (4.0, 4.8),
        "contraction_rate": (0.4, 0.75), "hedge_rate": (0.03, 0.05),
    },
    "BA": {
        "mean_sentence_len": (14, 18), "punct_rate": (5.0, 9.0),
        "vocab_richness": (0.68, 0.80), "filler_rate": (0.01, 0.02),
        "typo_rate": (0.005, 0.015), "mean_word_length": (4.2, 5.0),
        "contraction_rate": (0.3, 0.6), "hedge_rate": (0.03, 0.06),
    },
    "Graduate": {
        "mean_sentence_len": (16, 22), "punct_rate": (7.0, 12.0),
        "vocab_richness": (0.75, 0.90), "filler_rate": (0.005, 0.01),
        "typo_rate": (0.002, 0.01), "mean_word_length": (4.5, 5.5),
        "contraction_rate": (0.2, 0.5), "hedge_rate": (0.04, 0.07),
    },
}

# Filler words pool
_FILLER_WORDS = [
    "like", "you know", "I mean", "basically", "honestly",
    "kind of", "sort of", "actually", "literally", "um",
]

# Hedge words pool
_HEDGE_WORDS = [
    "I think", "maybe", "probably", "it seems like", "I guess",
    "in my opinion", "I feel like", "perhaps", "might be",
]

# Simple word substitutions (complex -> simple)
_SIMPLIFICATIONS = {
    "utilize": "use", "demonstrate": "show", "approximately": "about",
    "sufficient": "enough", "purchase": "buy", "commence": "start",
    "terminate": "end", "endeavor": "try", "facilitate": "help",
    "implement": "do", "indicate": "show", "additional": "more",
    "subsequently": "then", "consequently": "so", "nevertheless": "still",
    "furthermore": "also", "regarding": "about", "numerous": "many",
    "substantial": "big", "beneficial": "good", "detrimental": "bad",
    "comprehend": "get", "investigate": "look into", "accumulate": "build up",
}


# ---------------------------------------------------------------------------
# StylometricFingerprint dataclass
# ---------------------------------------------------------------------------

@dataclass
class StylometricFingerprint:
    """Per-participant writing identity built from demographics + persona."""
    vocab_richness: float = 0.65
    mean_word_length: float = 4.3
    mean_sentence_length: float = 14.0
    sentence_length_variance: float = 3.0
    capitalization: str = "standard"  # "standard" | "all_lower" | "inconsistent"
    punctuation_rate: float = 5.0  # marks per 100 words
    contraction_rate: float = 0.5  # 0.0-1.0
    filler_word_rate: float = 0.02  # per 100 words
    hedge_word_rate: float = 0.03
    typo_keys: List[str] = field(default_factory=list)  # QWERTY pairs
    apostrophe_omission: bool = False
    typo_rate: float = 0.01  # fraction of words that get typos


# ---------------------------------------------------------------------------
# HBSStylometricEngine
# ---------------------------------------------------------------------------

class HBSStylometricEngine:
    """
    Builds and applies per-participant stylometric fingerprints.

    Usage:
        engine = HBSStylometricEngine()
        fp = engine.build_fingerprint(participant_state)
        modified_text = engine.apply_fingerprint(original_text, fp)
    """

    def __init__(self) -> None:
        self._adjacency = _load_qwerty_adjacency()

    # ------------------------------------------------------------------
    # Build fingerprint from participant demographics
    # ------------------------------------------------------------------

    def build_fingerprint(
        self,
        participant_state: Any,
        rng: Optional[random.Random] = None,
    ) -> StylometricFingerprint:
        """
        Create a stylometric fingerprint based on participant demographics
        and response style. Parameters are drawn from education-level priors
        with persona-based modifiers.
        """
        rng = rng or random.Random()

        # Get education level
        edu = getattr(participant_state, "education_level", "HS")
        priors = _EDUCATION_PRIORS.get(edu, _EDUCATION_PRIORS["HS"])

        def _draw(key: str) -> float:
            lo, hi = priors[key]
            return rng.uniform(lo, hi)

        fp = StylometricFingerprint(
            vocab_richness=_draw("vocab_richness"),
            mean_word_length=_draw("mean_word_length"),
            mean_sentence_length=_draw("mean_sentence_len"),
            sentence_length_variance=rng.uniform(2.0, 5.0),
            punctuation_rate=_draw("punct_rate"),
            contraction_rate=_draw("contraction_rate"),
            filler_word_rate=_draw("filler_rate"),
            hedge_word_rate=_draw("hedge_rate"),
            typo_rate=_draw("typo_rate"),
        )

        # Response style modifiers
        style = getattr(participant_state, "response_style", "")
        if style == "satisficer":
            fp.mean_sentence_length *= 0.75
            fp.vocab_richness *= 0.85
            fp.filler_word_rate *= 1.5
        elif style == "deliberate_responder":
            fp.mean_sentence_length *= 1.2
            fp.vocab_richness *= 1.1
            fp.filler_word_rate *= 0.5
        elif style == "careless_responder":
            fp.typo_rate *= 2.0
            fp.capitalization = rng.choice(["all_lower", "inconsistent"])
            fp.apostrophe_omission = rng.random() < 0.6
        elif style == "extreme_responder":
            fp.punctuation_rate *= 1.3  # More exclamation-heavy
        elif style == "social_desirability":
            fp.hedge_word_rate *= 1.5

        # Capitalization — education-correlated
        if fp.capitalization == "standard":
            if edu in ("<HS", "HS"):
                _cap_roll = rng.random()
                if _cap_roll < 0.15:
                    fp.capitalization = "all_lower"
                elif _cap_roll < 0.25:
                    fp.capitalization = "inconsistent"

        # Apostrophe omission — education-correlated
        if not fp.apostrophe_omission:
            _apo_prob = {"<HS": 0.35, "HS": 0.20, "Some college": 0.10, "BA": 0.05, "Graduate": 0.02}
            fp.apostrophe_omission = rng.random() < _apo_prob.get(edu, 0.10)

        # Select 2-3 QWERTY typo key pairs for this participant
        _all_keys = list(self._adjacency.keys())
        if _all_keys:
            _n_typo_keys = rng.randint(2, 4)
            fp.typo_keys = rng.sample(_all_keys, min(_n_typo_keys, len(_all_keys)))

        # Clamp values
        fp.vocab_richness = max(0.3, min(0.95, fp.vocab_richness))
        fp.mean_sentence_length = max(5, min(30, fp.mean_sentence_length))
        fp.typo_rate = max(0.0, min(0.08, fp.typo_rate))
        fp.filler_word_rate = max(0.0, min(0.10, fp.filler_word_rate))
        fp.hedge_word_rate = max(0.0, min(0.10, fp.hedge_word_rate))

        return fp

    # ------------------------------------------------------------------
    # Apply fingerprint to text
    # ------------------------------------------------------------------

    def apply_fingerprint(
        self,
        text: str,
        fingerprint: StylometricFingerprint,
        rng: Optional[random.Random] = None,
    ) -> str:
        """
        Post-process generated OE response to match the participant's
        stylometric fingerprint. Applied to EVERY OE response.
        """
        if not text or len(text.strip()) < 5:
            return text

        rng = rng or random.Random()
        result = text

        # 1. Apply capitalization style
        result = self._apply_capitalization(result, fingerprint, rng)

        # 2. Handle contractions
        result = self._apply_contractions(result, fingerprint, rng)

        # 3. Apply apostrophe omission
        if fingerprint.apostrophe_omission:
            result = self._remove_apostrophes(result)

        # 4. Inject filler words
        result = self._inject_fillers(result, fingerprint, rng)

        # 5. Inject hedge words
        result = self._inject_hedges(result, fingerprint, rng)

        # 6. Inject typos
        result = self._inject_typos(result, fingerprint, rng)

        # 7. Vocabulary simplification for low vocab_richness
        if fingerprint.vocab_richness < 0.55:
            result = self._simplify_vocabulary(result)

        return result

    # ------------------------------------------------------------------
    # Extract fingerprint from text (bootstrap from first OE response)
    # ------------------------------------------------------------------

    def extract_fingerprint_from_text(self, text: str) -> StylometricFingerprint:
        """Analyze existing text to extract stylometric features."""
        fp = StylometricFingerprint()

        if not text or len(text.strip()) < 10:
            return fp

        words = text.split()
        word_count = len(words)
        if word_count == 0:
            return fp

        # Vocab richness (type-token ratio)
        unique_words = set(w.lower().strip(".,!?;:'\"") for w in words)
        fp.vocab_richness = len(unique_words) / max(word_count, 1)

        # Mean word length
        fp.mean_word_length = sum(len(w.strip(".,!?;:'\"")) for w in words) / max(word_count, 1)

        # Sentence count and length
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            lens = [len(s.split()) for s in sentences]
            fp.mean_sentence_length = sum(lens) / len(lens)
            if len(lens) > 1:
                mean_l = fp.mean_sentence_length
                fp.sentence_length_variance = math.sqrt(
                    sum((l - mean_l) ** 2 for l in lens) / len(lens)
                )

        # Punctuation rate
        punct_count = sum(1 for c in text if c in ".,!?;:—-")
        fp.punctuation_rate = (punct_count / max(word_count, 1)) * 100

        # Capitalization style
        if text == text.lower():
            fp.capitalization = "all_lower"
        elif not text[0].isupper():
            fp.capitalization = "inconsistent"

        return fp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_capitalization(
        self, text: str, fp: StylometricFingerprint, rng: random.Random,
    ) -> str:
        """Apply capitalization style to text."""
        if fp.capitalization == "all_lower":
            return text.lower()
        elif fp.capitalization == "inconsistent":
            # Randomly lowercase some sentence starts
            sentences = re.split(r'(?<=[.!?])\s+', text)
            result_parts = []
            for s in sentences:
                if s and rng.random() < 0.4:
                    result_parts.append(s[0].lower() + s[1:] if len(s) > 1 else s.lower())
                else:
                    result_parts.append(s)
            return " ".join(result_parts)
        return text

    def _apply_contractions(
        self, text: str, fp: StylometricFingerprint, rng: random.Random,
    ) -> str:
        """Apply contraction preferences based on fingerprint."""
        _expansions = {
            "do not": "don't", "does not": "doesn't", "did not": "didn't",
            "is not": "isn't", "are not": "aren't", "was not": "wasn't",
            "were not": "weren't", "have not": "haven't", "has not": "hasn't",
            "had not": "hadn't", "will not": "won't", "would not": "wouldn't",
            "could not": "couldn't", "should not": "shouldn't",
            "cannot": "can't", "can not": "can't",
            "I am": "I'm", "I have": "I've", "I will": "I'll",
            "I would": "I'd", "it is": "it's", "it has": "it's",
            "that is": "that's", "there is": "there's",
        }
        _contractions = {v: k for k, v in _expansions.items()}

        if fp.contraction_rate > 0.5:
            # Prefer contractions — expand formal forms
            for expanded, contracted in _expansions.items():
                if rng.random() < fp.contraction_rate:
                    text = re.sub(
                        re.escape(expanded), contracted, text, flags=re.IGNORECASE,
                    )
        else:
            # Prefer formal — expand contractions
            for contracted, expanded in _contractions.items():
                if rng.random() < (1.0 - fp.contraction_rate):
                    text = re.sub(
                        re.escape(contracted), expanded, text, flags=re.IGNORECASE,
                    )
        return text

    def _remove_apostrophes(self, text: str) -> str:
        """Remove apostrophes from contractions: don't -> dont."""
        return re.sub(r"(\w)'(\w)", r"\1\2", text)

    def _inject_fillers(
        self, text: str, fp: StylometricFingerprint, rng: random.Random,
    ) -> str:
        """Inject filler words at sentence boundaries."""
        if fp.filler_word_rate <= 0.005:
            return text

        words = text.split()
        word_count = len(words)
        if word_count < 8:
            return text

        # Calculate how many fillers to inject
        n_fillers = max(0, int(word_count * fp.filler_word_rate))
        if n_fillers == 0 and rng.random() < fp.filler_word_rate * 20:
            n_fillers = 1  # Small chance of at least one filler

        if n_fillers == 0:
            return text

        # Find sentence boundaries (after periods, at commas)
        _insertion_points = []
        for i, w in enumerate(words):
            if i > 0 and (words[i - 1].endswith((",", ".", "!", "?", ";")) or i == 0):
                _insertion_points.append(i)
        # Also add a few mid-sentence points
        for i in range(3, len(words) - 2, max(4, len(words) // 4)):
            if i not in _insertion_points:
                _insertion_points.append(i)

        if not _insertion_points:
            return text

        _insertion_points.sort()
        _chosen = rng.sample(
            _insertion_points, min(n_fillers, len(_insertion_points)),
        )
        _chosen.sort(reverse=True)  # Insert from end to preserve indices

        for idx in _chosen:
            filler = rng.choice(_FILLER_WORDS)
            if idx < len(words):
                words.insert(idx, filler + ",")

        return " ".join(words)

    def _inject_hedges(
        self, text: str, fp: StylometricFingerprint, rng: random.Random,
    ) -> str:
        """Inject hedge words for hedge-prone profiles."""
        if fp.hedge_word_rate <= 0.01:
            return text

        # Only inject at the start of the text or at sentence starts
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return text

        modified = []
        for i, s in enumerate(sentences):
            if s and rng.random() < fp.hedge_word_rate * 10:
                hedge = rng.choice(_HEDGE_WORDS)
                # Don't double-hedge
                if not any(h in s.lower()[:30] for h in [hw.lower() for hw in _HEDGE_WORDS]):
                    if s[0].isupper():
                        s = hedge + " " + s[0].lower() + s[1:]
                    else:
                        s = hedge + " " + s
            modified.append(s)

        return " ".join(modified)

    def _inject_typos(
        self, text: str, fp: StylometricFingerprint, rng: random.Random,
    ) -> str:
        """Inject QWERTY-adjacency typos at the participant's typo rate."""
        if fp.typo_rate <= 0.001 or not fp.typo_keys:
            return text

        words = text.split()
        if len(words) < 5:
            return text

        n_typos = max(0, int(len(words) * fp.typo_rate))
        if n_typos == 0 and rng.random() < fp.typo_rate * 20:
            n_typos = 1

        if n_typos == 0:
            return text

        # Pick random words to typo-ify
        _eligible = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
        if not _eligible:
            return text

        _chosen = rng.sample(_eligible, min(n_typos, len(_eligible)))

        for idx in _chosen:
            word = words[idx]
            # Find a character in this word that matches a typo key
            _typo_chars = [
                (ci, c.lower())
                for ci, c in enumerate(word)
                if c.lower() in fp.typo_keys and c.lower() in self._adjacency
            ]
            if _typo_chars:
                ci, c = rng.choice(_typo_chars)
                _adj = self._adjacency.get(c, [])
                if _adj:
                    replacement = rng.choice(_adj)
                    # Preserve case
                    if word[ci].isupper():
                        replacement = replacement.upper()
                    word = word[:ci] + replacement + word[ci + 1:]
                    words[idx] = word

        return " ".join(words)

    def _simplify_vocabulary(self, text: str) -> str:
        """Replace complex words with simpler alternatives."""
        result = text
        for complex_word, simple_word in _SIMPLIFICATIONS.items():
            # Case-insensitive replacement, preserving original case
            pattern = re.compile(re.escape(complex_word), re.IGNORECASE)
            result = pattern.sub(simple_word, result)
        return result
