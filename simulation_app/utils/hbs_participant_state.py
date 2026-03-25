"""
HBS Participant State — Central data model for the Human Behavior Simulator.

Part of the HBS architecture: this module defines the ParticipantState
dataclass that every HBS module reads from and writes to. ParticipantState
is the single source of truth for a simulated participant's identity,
demographics, behavioral parameters, stylometric fingerprint, survey
progress, and validation outcomes.

The HBSParticipantFactory produces census-weighted participant batches
with persona-demographic coupling, ensuring that simulated samples
approximate the demographic composition of US adults while maintaining
internally consistent behavioral profiles (Krosnick 1999; Podsakoff
et al. 2003).

References:
    - Krosnick, J. A. (1999). Survey research. Annual Review of Psychology.
    - Paulhus, D. L. (2002). Socially desirable responding.
    - Woods, C. M. (2006). Careless responding to reverse-worded items.
    - Greenleaf, E. A. (1992). Measuring extreme response style.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["ParticipantState", "HBSParticipantFactory"]


# ---------------------------------------------------------------------------
# Census-weighted distribution tables
# ---------------------------------------------------------------------------

_AGE_BRACKETS: List[Tuple[Tuple[int, int], float]] = [
    ((18, 29), 0.22),
    ((30, 44), 0.25),
    ((45, 59), 0.25),
    ((60, 85), 0.28),
]

_EDUCATION_LEVELS: List[Tuple[str, float]] = [
    ("Less than high school", 0.10),
    ("High school diploma", 0.27),
    ("Some college", 0.29),
    ("Bachelor's degree", 0.22),
    ("Graduate degree", 0.12),
]

_INCOME_BRACKETS: List[Tuple[str, float]] = [
    ("<$30k", 0.20),
    ("$30k-$60k", 0.25),
    ("$60k-$100k", 0.28),
    ("$100k-$150k", 0.16),
    ("$150k+", 0.11),
]

_GENDER: List[Tuple[str, float]] = [
    ("Male", 0.49),
    ("Female", 0.50),
    ("Non-binary", 0.01),
]

# 7-point party identification scale (ANES convention)
_PARTY_ID_7PT: List[Tuple[str, float]] = [
    ("Strong Democrat", 0.16),
    ("Weak Democrat", 0.10),
    ("Lean Democrat", 0.10),
    ("Independent", 0.13),
    ("Lean Republican", 0.09),
    ("Weak Republican", 0.11),
    ("Strong Republican", 0.16),
    ("Other", 0.05),
]

# Krosnick-taxonomy response styles with base rates
_RESPONSE_STYLES: List[Tuple[str, float]] = [
    ("engaged_responder", 0.30),
    ("satisficer", 0.20),
    ("moderate_responder", 0.10),
    ("deliberate_responder", 0.10),
    ("social_desirability", 0.10),
    ("extreme_responder", 0.08),
    ("acquiescent_responder", 0.07),
    ("careless_responder", 0.05),
]

# Top-10 US states by adult population (approximate weights); remainder
# share the leftover mass equally.
_TOP_STATES: List[Tuple[str, float]] = [
    ("CA", 0.118),
    ("TX", 0.088),
    ("FL", 0.068),
    ("NY", 0.060),
    ("PA", 0.040),
    ("IL", 0.039),
    ("OH", 0.036),
    ("GA", 0.033),
    ("NC", 0.032),
    ("MI", 0.031),
]

_OTHER_STATES: List[str] = [
    "AL", "AK", "AZ", "AR", "CO", "CT", "DE", "DC", "HI", "ID",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "ND", "OK", "OR",
    "RI", "SC", "SD", "TN", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

_REGION_MAP: Dict[str, str] = {
    # Northeast
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast",
    "NH": "Northeast", "RI": "Northeast", "VT": "Northeast",
    "NJ": "Northeast", "NY": "Northeast", "PA": "Northeast",
    # South
    "DE": "South", "FL": "South", "GA": "South", "MD": "South",
    "NC": "South", "SC": "South", "VA": "South", "DC": "South",
    "WV": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South",
    "TX": "South",
    # Midwest
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    # West
    "AK": "West", "AZ": "West", "CA": "West", "CO": "West",
    "HI": "West", "ID": "West", "MT": "West", "NV": "West",
    "NM": "West", "OR": "West", "UT": "West", "WA": "West",
    "WY": "West",
}

# Rough zip-code prefix ranges by state (first 3 digits).  Used to
# generate plausible 5-digit zip codes.  Each state maps to a list of
# (low, high) prefix ranges.
_STATE_ZIP_PREFIXES: Dict[str, List[Tuple[int, int]]] = {
    "AL": [(350, 369)], "AK": [(995, 999)], "AZ": [(850, 865)],
    "AR": [(716, 729)], "CA": [(900, 961)], "CO": [(800, 816)],
    "CT": [(60, 69)], "DE": [(197, 199)], "DC": [(200, 205)],
    "FL": [(320, 349)], "GA": [(300, 319)], "HI": [(967, 968)],
    "ID": [(832, 838)], "IL": [(600, 629)], "IN": [(460, 479)],
    "IA": [(500, 528)], "KS": [(660, 679)], "KY": [(400, 427)],
    "LA": [(700, 714)], "ME": [(39, 49)], "MD": [(206, 219)],
    "MA": [(10, 27)], "MI": [(480, 499)], "MN": [(550, 567)],
    "MS": [(386, 397)], "MO": [(630, 658)], "MT": [(590, 599)],
    "NE": [(680, 693)], "NV": [(889, 898)], "NH": [(30, 38)],
    "NJ": [(70, 89)], "NM": [(870, 884)], "NY": [(100, 149)],
    "NC": [(270, 289)], "ND": [(580, 588)], "OH": [(430, 458)],
    "OK": [(730, 749)], "OR": [(970, 979)], "PA": [(150, 196)],
    "RI": [(28, 29)], "SC": [(290, 299)], "SD": [(570, 577)],
    "TN": [(370, 385)], "TX": [(750, 799)], "UT": [(840, 847)],
    "VT": [(50, 59)], "VA": [(220, 246)], "WA": [(980, 994)],
    "WV": [(247, 268)], "WI": [(530, 549)], "WY": [(820, 831)],
}

# Religious affiliation (Pew 2021 approximate)
_RELIGIOUS_AFFILIATIONS: List[Tuple[str, float]] = [
    ("Protestant", 0.40),
    ("Catholic", 0.21),
    ("Unaffiliated", 0.26),
    ("Jewish", 0.02),
    ("Muslim", 0.01),
    ("Mormon", 0.02),
    ("Buddhist", 0.01),
    ("Hindu", 0.01),
    ("Other", 0.06),
]

# News consumption categories
_NEWS_CONSUMPTION: List[Tuple[str, float]] = [
    ("Heavy (daily, multiple sources)", 0.18),
    ("Regular (daily, 1-2 sources)", 0.27),
    ("Moderate (few times a week)", 0.25),
    ("Light (once a week or less)", 0.20),
    ("Minimal (rarely/never)", 0.10),
]

# Capitalization style options
_CAPITALIZATION_STYLES: List[str] = [
    "standard",       # Normal sentence case
    "all_lower",      # everything lowercase
    "all_caps",       # EVERYTHING CAPS
    "inconsistent",   # Random mix
]

# Race/ethnicity (US Census 2020 approximate)
_RACE_ETHNICITY: List[Tuple[str, float]] = [
    ("White", 0.58),
    ("Hispanic/Latino", 0.19),
    ("Black", 0.12),
    ("Asian", 0.06),
    ("Multiracial", 0.03),
    ("Other", 0.02),
]


# ---------------------------------------------------------------------------
# Helper: clipped normal draw
# ---------------------------------------------------------------------------

def _clipped_normal(rng: random.Random, mu: float, sigma: float,
                    lo: float, hi: float) -> float:
    """Draw from N(mu, sigma) clipped to [lo, hi]."""
    val = rng.gauss(mu, sigma)
    return max(lo, min(hi, val))


def _beta_draw(rng: random.Random, a: float = 2.0, b: float = 5.0) -> float:
    """Draw from Beta(a, b) using the stdlib random.betavariate."""
    return rng.betavariate(a, b)


def _weighted_choice(rng: random.Random,
                     options: List[Tuple[str, float]]) -> str:
    """Weighted random selection; weights need not sum to exactly 1.0."""
    labels = [o[0] for o in options]
    weights = [o[1] for o in options]
    return rng.choices(labels, weights=weights, k=1)[0]


def _weighted_choice_idx(rng: random.Random,
                         options: List[Tuple[Any, float]]) -> int:
    """Return the INDEX of a weighted random selection."""
    weights = [o[1] for o in options]
    return rng.choices(range(len(options)), weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# ParticipantState dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParticipantState:
    """Central data model for a single simulated participant.

    HBS architecture: ParticipantState is the shared mutable record
    that every HBS pipeline stage reads and writes.  Fields are grouped
    into identity, demographics, behavioral parameters, stylometric
    fingerprint, survey state, behavioral flags, and validation scores.

    Behavioral parameters are drawn ONCE at creation and are treated as
    immutable for the lifetime of the participant.  Survey state fields
    accumulate as the participant progresses through questions.
    """

    # --- Identity -----------------------------------------------------------
    participant_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    persona_id: Optional[str] = None
    response_style: str = "engaged_responder"

    # --- Demographics (census-weighted US adult) ----------------------------
    age: int = 35
    birth_year: int = 1991
    education_level: str = "Some college"
    income_bracket: str = "$30k-$60k"
    gender: str = "Female"
    race_ethnicity: str = "White"
    party_id: str = "Independent"          # 7-point ANES label
    ideology: float = 0.0                  # -3.0 (far left) to +3.0 (far right)
    state: str = "CA"
    zip_code: str = "90210"
    region: str = "West"
    religious_affiliation: str = "Unaffiliated"
    news_consumption: str = "Moderate (few times a week)"

    # --- Behavioral params (drawn ONCE, immutable) --------------------------
    typing_speed_wpm: float = 55.0         # N(55,15) clipped [20,110]
    reading_speed_wpm: float = 238.0       # N(238,60) clipped [100,450]
    attention_level: float = 0.5           # Beta(2,5)
    target_completion_seconds: float = 840.0  # N(840,120) clipped [480,1500]

    # --- Stylometric fingerprint --------------------------------------------
    vocab_richness: float = 0.6            # 0..1 type-token ratio proxy
    mean_sentence_len: float = 14.0        # words per sentence
    punct_rate: float = 0.05               # punctuation marks per word
    filler_word_rate: float = 0.02         # "like", "um", "you know" rate
    capitalization_style: str = "standard"
    typo_signature: float = 0.01           # probability of per-character typo

    # --- Survey state (mutable, accumulated) --------------------------------
    prior_responses: Dict[str, Any] = field(default_factory=dict)
    questions_seen: List[str] = field(default_factory=list)
    voice_memory: List[str] = field(default_factory=list)
    established_political_lean: Optional[float] = None

    # --- Behavioral flags ---------------------------------------------------
    is_straight_liner: bool = False
    is_satisficer: bool = False
    is_careless: bool = False
    failed_attention_checks: int = 0

    # --- Validation scores --------------------------------------------------
    tipi_sd: Optional[float] = None
    completion_time_seconds: Optional[float] = None
    validation_passed: bool = True
    validation_flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HBSParticipantFactory
# ---------------------------------------------------------------------------

class HBSParticipantFactory:
    """Factory for creating census-weighted participant batches.

    HBS architecture: the factory is the sole entry-point for
    constructing ParticipantState instances.  It enforces demographic
    plausibility, persona-demographic coupling, and consistent
    within-participant behavioral parameters.

    Usage::

        factory = HBSParticipantFactory(seed=42)
        participants = factory.create_batch(200, conditions=["control", "treatment"])
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # ---- public API -------------------------------------------------------

    def create_batch(
        self,
        n: int,
        conditions: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> List[ParticipantState]:
        """Create *n* participants with census-weighted demographics.

        Args:
            n: Number of participants to create.
            conditions: Experimental condition labels.  Participants are
                assigned to conditions via round-robin so each condition
                gets an approximately equal share.
            domain: Optional study domain (e.g. ``"political"``,
                ``"consumer"``).  Currently reserved for future
                domain-specific persona adjustments.

        Returns:
            List of fully initialised ``ParticipantState`` instances.
        """
        if n <= 0:
            return []

        conditions = conditions or ["control"]
        participants: List[ParticipantState] = []

        for i in range(n):
            condition_label = conditions[i % len(conditions)]
            p = self._create_one(condition_label, domain)
            participants.append(p)

        return participants

    # ---- internals --------------------------------------------------------

    def _create_one(
        self,
        condition: str,
        domain: Optional[str],
    ) -> ParticipantState:
        """Build a single ParticipantState with coupled demographics."""
        rng = self._rng

        # --- Demographics ---------------------------------------------------
        age_bracket_idx = _weighted_choice_idx(rng, _AGE_BRACKETS)
        age_lo, age_hi = _AGE_BRACKETS[age_bracket_idx][0]
        age = rng.randint(age_lo, age_hi)
        birth_year = 2026 - age

        education = _weighted_choice(rng, _EDUCATION_LEVELS)
        income = _weighted_choice(rng, _INCOME_BRACKETS)
        gender = _weighted_choice(rng, _GENDER)
        race_ethnicity = _weighted_choice(rng, _RACE_ETHNICITY)
        party_id_raw = _weighted_choice(rng, _PARTY_ID_7PT)
        # Map "Other" to "Independent" per spec
        party_id = "Independent" if party_id_raw == "Other" else party_id_raw
        ideology = self._ideology_from_party(party_id)
        state = self._sample_state()
        region = _REGION_MAP.get(state, "West")
        zip_code = self._random_zip(state)
        religious = _weighted_choice(rng, _RELIGIOUS_AFFILIATIONS)
        news = _weighted_choice(rng, _NEWS_CONSUMPTION)

        # --- Response style (with persona-demographic coupling) -------------
        response_style = self._sample_response_style(age, education, party_id)

        # --- Behavioral parameters (drawn once) ----------------------------
        typing_speed = _clipped_normal(rng, 55.0, 15.0, 20.0, 110.0)
        reading_speed = _clipped_normal(rng, 238.0, 60.0, 100.0, 450.0)
        attention = _beta_draw(rng, 2.0, 5.0)
        target_completion = _clipped_normal(rng, 840.0, 120.0, 480.0, 1500.0)

        # --- Stylometric fingerprint ----------------------------------------
        vocab_richness = self._stylometric_vocab(education, response_style)
        mean_sentence_len = _clipped_normal(rng, 14.0, 5.0, 4.0, 35.0)
        punct_rate = max(0.0, rng.gauss(0.05, 0.02))
        filler_word_rate = max(0.0, rng.gauss(0.02, 0.01))
        cap_style = self._sample_capitalization(response_style)
        typo_sig = self._sample_typo_rate(response_style, age)

        # --- Behavioral flags -----------------------------------------------
        is_careless = response_style == "careless_responder"
        is_satisficer = response_style in ("satisficer", "careless_responder")
        is_straight_liner = is_satisficer and rng.random() < 0.35

        # --- Assemble -------------------------------------------------------
        return ParticipantState(
            participant_id=str(uuid.uuid4()),
            persona_id=None,
            response_style=response_style,
            age=age,
            birth_year=birth_year,
            education_level=education,
            income_bracket=income,
            gender=gender,
            race_ethnicity=race_ethnicity,
            party_id=party_id,
            ideology=round(ideology, 2),
            state=state,
            zip_code=zip_code,
            region=region,
            religious_affiliation=religious,
            news_consumption=news,
            typing_speed_wpm=round(typing_speed, 1),
            reading_speed_wpm=round(reading_speed, 1),
            attention_level=round(attention, 3),
            target_completion_seconds=round(target_completion, 0),
            vocab_richness=round(vocab_richness, 3),
            mean_sentence_len=round(mean_sentence_len, 1),
            punct_rate=round(punct_rate, 4),
            filler_word_rate=round(filler_word_rate, 4),
            capitalization_style=cap_style,
            typo_signature=round(typo_sig, 4),
            is_straight_liner=is_straight_liner,
            is_satisficer=is_satisficer,
            is_careless=is_careless,
        )

    # ---- demographic helpers -----------------------------------------------

    def _sample_state(self) -> str:
        """Sample a US state weighted by adult population."""
        rng = self._rng
        top_mass = sum(w for _, w in _TOP_STATES)
        remaining_mass = 1.0 - top_mass
        per_other = remaining_mass / len(_OTHER_STATES) if _OTHER_STATES else 0

        all_states: List[Tuple[str, float]] = list(_TOP_STATES)
        for s in _OTHER_STATES:
            all_states.append((s, per_other))

        return _weighted_choice(rng, all_states)

    def _ideology_from_party(self, party_id: str) -> float:
        """Derive a continuous ideology score from 7-point party ID.

        Returns a value in [-3.0, +3.0] with noise, following the
        conventional mapping where negative = liberal, positive =
        conservative.
        """
        rng = self._rng
        _MAP = {
            "Strong Democrat": -2.5,
            "Weak Democrat": -1.5,
            "Lean Democrat": -0.8,
            "Independent": 0.0,
            "Lean Republican": 0.8,
            "Weak Republican": 1.5,
            "Strong Republican": 2.5,
        }
        base = _MAP.get(party_id, 0.0)
        noise = rng.gauss(0, 0.4)
        return max(-3.0, min(3.0, base + noise))

    def _random_zip(self, state: str) -> str:
        """Generate a plausible 5-digit zip code for the given state."""
        rng = self._rng
        prefixes = _STATE_ZIP_PREFIXES.get(state)
        if not prefixes:
            # Fallback: generic zip
            return f"{rng.randint(10000, 99999)}"
        lo, hi = rng.choice(prefixes)
        prefix = rng.randint(lo, hi)
        suffix = rng.randint(0, 99)
        return f"{prefix:03d}{suffix:02d}"

    # ---- response style with persona-demographic coupling ------------------

    def _sample_response_style(
        self, age: int, education: str, party_id: str,
    ) -> str:
        """Sample response style with persona-demographic coupling.

        Coupling rules (soft biases, not deterministic):
        - engaged_responder correlates with higher education
        - extreme_responder correlates with stronger party ID
        - careless_responder correlates with younger age
        """
        rng = self._rng

        # Start with base weights
        weights: Dict[str, float] = {s: w for s, w in _RESPONSE_STYLES}

        # Education coupling: higher education -> more engaged/deliberate
        high_ed = education in ("Bachelor's degree", "Graduate degree")
        low_ed = education in ("Less than high school",)
        if high_ed:
            weights["engaged_responder"] *= 1.4
            weights["deliberate_responder"] *= 1.3
            weights["careless_responder"] *= 0.6
        elif low_ed:
            weights["engaged_responder"] *= 0.7
            weights["satisficer"] *= 1.3

        # Party strength coupling: stronger partisanship -> more extreme
        strong_partisan = party_id in ("Strong Democrat", "Strong Republican")
        if strong_partisan:
            weights["extreme_responder"] *= 1.5
            weights["moderate_responder"] *= 0.6

        # Age coupling: younger -> more careless
        if age < 25:
            weights["careless_responder"] *= 1.6
            weights["deliberate_responder"] *= 0.7
        elif age > 55:
            weights["deliberate_responder"] *= 1.3
            weights["careless_responder"] *= 0.7

        options = [(k, v) for k, v in weights.items()]
        return _weighted_choice(rng, options)

    # ---- stylometric helpers -----------------------------------------------

    def _stylometric_vocab(self, education: str, style: str) -> float:
        """Vocab richness correlated with education and response style."""
        rng = self._rng
        base = 0.55
        if education in ("Bachelor's degree", "Graduate degree"):
            base += 0.15
        elif education == "Less than high school":
            base -= 0.15

        if style in ("engaged_responder", "deliberate_responder"):
            base += 0.08
        elif style == "careless_responder":
            base -= 0.12

        return max(0.1, min(1.0, base + rng.gauss(0, 0.08)))

    def _sample_capitalization(self, style: str) -> str:
        """Capitalization style; careless participants deviate more."""
        rng = self._rng
        if style == "careless_responder":
            return rng.choices(
                _CAPITALIZATION_STYLES,
                weights=[0.30, 0.40, 0.10, 0.20],
                k=1,
            )[0]
        # Most participants use standard capitalization
        return rng.choices(
            _CAPITALIZATION_STYLES,
            weights=[0.85, 0.08, 0.02, 0.05],
            k=1,
        )[0]

    def _sample_typo_rate(self, style: str, age: int) -> float:
        """Per-character typo probability; varies by style and age."""
        rng = self._rng
        base = 0.008
        if style == "careless_responder":
            base = 0.025
        elif style in ("deliberate_responder", "engaged_responder"):
            base = 0.004

        # Younger and older participants make slightly more typos
        if age < 25 or age > 65:
            base *= 1.3

        return max(0.0, base + rng.gauss(0, 0.003))
