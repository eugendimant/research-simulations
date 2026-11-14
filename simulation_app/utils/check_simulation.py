"""Comprehension, manipulation and named attention checks.

Version: 1.3.1.4

Before v1.3.1.4 these three column families were stamped onto the finished
dataframe by ``app.py`` *after* ``EnhancedSimulationEngine.generate()`` returned.
That placement caused a cluster of defects: the manipulation check used a
hard-coded 1-7 scale with a +1.5 shift applied to every arm whose label did not
contain the substring "control", so a study with arms such as "Low scarcity" /
"High scarcity" got no condition difference at all and a fixed 3-7 range; the
columns were invisible to the codebook; they never received missing data; and
they were drawn from fixed literal seeds rather than the run seed, so they were
uncorrelated with the participant's attention, carelessness or outcome scores.

This module holds the spec normalisation and the response model. The engine owns
the call site (``EnhancedSimulationEngine._generate_checks``), so the columns are
produced alongside every other column and inherit the seed, the traits, the
codebook and the missing-data pass.

Public API:

``normalize_check_name``
    The single name-to-column-name rule. ``app.py`` used to normalise with
    ``re.sub(r'[^\\w]', '_', name)`` when creating the column and with
    ``name.replace(' ', '_')`` when reading it back, so any name containing
    punctuation reported a realised pass rate of 0.0.

``ManipulationCheckSpec`` / ``normalize_manipulation_specs``
``ComprehensionCheckSpec`` / ``normalize_comprehension_specs``
``AttentionCheckSpec`` / ``normalize_attention_specs``
``generate_manipulation_values`` / ``generate_comprehension_values``
``calibrate_probability_scale``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__version__ = "1.3.1.4"

__all__ = [
    "AttentionCheckSpec",
    "ComprehensionCheckSpec",
    "ManipulationCheckSpec",
    "calibrate_probability_scale",
    "condition_intensity_order",
    "generate_comprehension_values",
    "generate_manipulation_values",
    "normalize_attention_specs",
    "normalize_check_name",
    "normalize_comprehension_specs",
    "normalize_manipulation_specs",
    "resolve_existing_column",
]


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

_NON_WORD = re.compile(r"[^\w]+")


def normalize_check_name(name: Any, fallback: str = "Check") -> str:
    """Turn a user-entered check name into a stable column name.

    One rule, used both when the column is written and when it is read back.
    """
    cleaned = _NON_WORD.sub("_", str(name or "")).strip("_")
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        cleaned = f"V_{cleaned}"
    return cleaned


def resolve_existing_column(name: str, columns: Sequence[str]) -> Optional[str]:
    """Find the column the engine already generates for a DV called ``name``.

    A single-item DV named ``MC_Scarce`` is written as ``MC_Scarce_1`` and a
    multi-item one adds ``MC_Scarce_mean``; the old collision guard tested only
    the bare name, missed, and appended a second fabricated column beside the
    correctly scaled one.
    """
    base = normalize_check_name(name)
    cols = list(columns)
    for candidate in (f"{base}_mean", base, f"{base}_1"):
        if candidate in cols:
            return candidate
    return None


# --------------------------------------------------------------------------
# Word-boundary matching (CLAUDE.md: never substring ``in``)
# --------------------------------------------------------------------------

_WORD_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _word_hit(word: str, text: str) -> bool:
    """True when ``word`` appears in ``text`` delimited by non-word characters.

    ``\\b`` is deliberately not used: labels such as ``80%``, ``$10`` or
    ``Treatment (high)`` must still match.
    """
    key = word.lower()
    pattern = _WORD_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(r"(?<!\w)" + re.escape(key) + r"(?!\w)", re.IGNORECASE)
        _WORD_CACHE[key] = pattern
    return bool(pattern.search(str(text or "")))


_REFERENCE_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _reference_hit(word: str, text: str) -> bool:
    """``_word_hit`` with hyphens treated as part of the word.

    The general rule allows a hyphen as a boundary so ``Treatment (high)`` and
    ``80%`` match, but that made the condition "Self-control training" read as a
    control arm. A compound word is not the bare word.
    """
    key = word.lower()
    pattern = _REFERENCE_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(r"(?<![\w-])" + re.escape(key) + r"(?![\w-])", re.IGNORECASE)
        _REFERENCE_CACHE[key] = pattern
    return bool(pattern.search(str(text or "")))


_REFERENCE_WORDS = (
    "control", "baseline", "neutral", "none", "no treatment", "no manipulation",
    "untreated", "placebo", "standard", "status quo",
)

# Ordinal cues, lowest intensity first. A label matching several takes the last.
_ORDINAL_LADDER = (
    ("none", "no", "zero", "absent", "never"),
    ("very low", "minimal", "lowest"),
    ("low", "weak", "small", "little", "mild", "few"),
    ("moderate", "medium", "mid", "some", "average"),
    ("high", "strong", "large", "much", "many", "intense"),
    ("very high", "extreme", "highest", "maximum", "max"),
)


def _numeric_cue(label: str) -> Optional[float]:
    """Pull a magnitude out of a label such as ``80%``, ``$10`` or ``Round 3``."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(label or ""))
    if not matches:
        return None
    try:
        return float(matches[0])
    except (TypeError, ValueError):
        return None


def condition_intensity_order(conditions: Sequence[str]) -> Tuple[List[float], str]:
    """Rank conditions on the dimension a manipulation check would measure.

    Returns ``(ranks, basis)`` where ``ranks`` are in ``[0, 1]`` in the order the
    conditions were given and ``basis`` names the evidence used, so the caller can
    warn the analyst when the ordering was merely inferred.

    Resolution order: explicit numbers in the labels, then ordinal words, then a
    reference ("control"/"baseline") arm pinned low with the rest spread over the
    declared order. The last of those is arbitrary for nominal arms such as
    "Ingroup"/"Outgroup", which is exactly why it is reported.
    """
    labels = [str(c) for c in conditions]
    k = len(labels)
    if k == 0:
        return [], "empty"
    if k == 1:
        return [0.5], "single"

    def _spread(scores: Sequence[float]) -> List[float]:
        lo, hi = min(scores), max(scores)
        if hi - lo <= 0:
            return [0.5] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    numeric = [_numeric_cue(lbl) for lbl in labels]
    if all(v is not None for v in numeric) and len(set(numeric)) > 1:
        return _spread([float(v) for v in numeric]), "numeric labels"

    ordinal: List[Optional[float]] = []
    for lbl in labels:
        rank: Optional[float] = None
        for level, words in enumerate(_ORDINAL_LADDER):
            if any(_word_hit(w, lbl) for w in words):
                rank = float(level)
        ordinal.append(rank)
    if all(v is not None for v in ordinal) and len(set(ordinal)) > 1:
        return _spread([float(v) for v in ordinal]), "ordinal labels"

    is_reference = [any(_reference_hit(w, lbl) for w in _REFERENCE_WORDS) for lbl in labels]
    if any(is_reference) and not all(is_reference):
        others = [i for i, ref in enumerate(is_reference) if not ref]
        ranks = [0.0] * k
        if len(others) == 1:
            ranks[others[0]] = 1.0
        else:
            for pos, idx in enumerate(others):
                ranks[idx] = 0.4 + 0.6 * (pos / (len(others) - 1))
        return ranks, "reference arm"

    return [i / (k - 1) for i in range(k)], "declared order"


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------

def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class ManipulationCheckSpec:
    """One manipulation check.

    ``high_conditions`` is the analyst's statement of which arms should score
    high. When it is empty the ordering is inferred (see
    :func:`condition_intensity_order`) and ``order_basis`` records how.
    """

    name: str
    column: str
    scale_min: int = 1
    scale_max: int = 7
    effect_size: float = 2.0
    high_conditions: List[str] = field(default_factory=list)
    low_conditions: List[str] = field(default_factory=list)
    question_text: str = ""
    targets: Dict[str, float] = field(default_factory=dict)
    order_basis: str = ""
    bound_column: Optional[str] = None

    @property
    def midpoint(self) -> float:
        return (float(self.scale_min) + float(self.scale_max)) / 2.0

    @property
    def sd(self) -> float:
        """Within-condition SD. A 1-7 scale gives 1.2, the historical value."""
        return max(0.25, (float(self.scale_max) - float(self.scale_min)) / 5.0)


@dataclass
class ComprehensionCheckSpec:
    name: str
    column: str
    correct_answer: str = ""
    options: List[str] = field(default_factory=list)
    pass_rate: float = 0.90
    question_text: str = ""
    correct_by_condition: Dict[str, str] = field(default_factory=dict)
    coded_only: bool = False


@dataclass
class AttentionCheckSpec:
    name: str
    column: str


def normalize_manipulation_specs(
    raw_specs: Sequence[Any],
    conditions: Sequence[str],
    scale_specs: Sequence[Dict[str, Any]] = (),
    existing_columns: Sequence[str] = (),
    default_scale_points: int = 7,
) -> Tuple[List[ManipulationCheckSpec], List[str]]:
    """Normalise manipulation-check entries and compute their per-condition targets.

    Accepts either a bare name (the pre-v1.3.1.4 format) or a dict carrying
    ``scale_min`` / ``scale_max`` / ``scale_points`` / ``effect_size`` /
    ``high_conditions``. Returns the specs plus warnings for the analyst.
    """
    specs: List[ManipulationCheckSpec] = []
    warnings: List[str] = []
    seen: set = set()
    cond_labels = [str(c) for c in conditions]
    by_scale = {normalize_check_name(s.get("name", "")): s for s in (scale_specs or [])
                if isinstance(s, dict)}

    for raw in raw_specs or []:
        if isinstance(raw, dict):
            name = str(raw.get("name", "") or "").strip()
        else:
            name = str(raw or "").strip()
            raw = {"name": name}
        if not name:
            warnings.append("A manipulation check with no name was skipped.")
            continue
        column = normalize_check_name(name, fallback="ManipCheck")
        if column in seen:
            warnings.append(
                f"Manipulation check '{name}' duplicates an earlier entry and was skipped."
            )
            continue
        seen.add(column)

        # A manipulation check that is also a confirmed DV keeps the DV's own
        # column and scale; only the pass/fail flag is added.
        bound = resolve_existing_column(name, existing_columns)
        linked_scale = by_scale.get(column)

        if linked_scale is not None:
            points = _as_int(linked_scale.get("scale_points"), default_scale_points)
            s_min = _as_int(linked_scale.get("scale_min", 1), 1)
            s_max = _as_int(linked_scale.get("scale_max", points), points)
        else:
            points = _as_int(raw.get("scale_points"), default_scale_points)
            s_min = _as_int(raw.get("scale_min", 1), 1)
            s_max = _as_int(raw.get("scale_max", points), points)
        if s_max <= s_min:
            warnings.append(
                f"Manipulation check '{name}' had an inconsistent range "
                f"({s_min}-{s_max}); using 1-{max(2, default_scale_points)}."
            )
            s_min, s_max = 1, max(2, default_scale_points)

        # A manipulation check is built to be unambiguous, so the default separation
        # is large (d = 2.0, i.e. the extreme arms sit one within-condition SD either
        # side of the midpoint, giving a ~84% pass rate in a two-arm design).
        effect = _as_float(raw.get("effect_size"), 2.0)
        effect = float(np.clip(effect, 0.0, 4.0))

        highs = [str(c).strip() for c in (raw.get("high_conditions") or []) if str(c).strip()]
        lows = [str(c).strip() for c in (raw.get("low_conditions") or []) if str(c).strip()]
        unknown = [c for c in highs + lows if c not in cond_labels]
        if unknown:
            warnings.append(
                f"Manipulation check '{name}' names condition(s) "
                f"{', '.join(sorted(set(unknown)))} that are not in this design; they were ignored."
            )
            highs = [c for c in highs if c in cond_labels]
            lows = [c for c in lows if c in cond_labels]

        spec = ManipulationCheckSpec(
            name=name,
            column=column,
            scale_min=s_min,
            scale_max=s_max,
            effect_size=effect,
            high_conditions=highs,
            low_conditions=lows,
            question_text=str(raw.get("question_text", "") or ""),
            bound_column=bound,
        )

        if highs:
            lows_eff = lows or [c for c in cond_labels if c not in highs]
            if not lows_eff:
                warnings.append(
                    f"Manipulation check '{name}' marks every condition as high, so it "
                    "cannot show a condition difference."
                )
            ranks = {c: (1.0 if c in highs else 0.0) for c in cond_labels}
            spec.order_basis = "analyst-specified"
        else:
            rank_list, basis = condition_intensity_order(cond_labels)
            ranks = dict(zip(cond_labels, rank_list))
            spec.order_basis = basis
            if basis == "declared order" and len(cond_labels) > 1:
                warnings.append(
                    f"Manipulation check '{name}': the condition labels carry no "
                    "high/low cue, so the direction was taken from the order the "
                    f"conditions are listed ({cond_labels[0]} lowest). Set "
                    "'Higher in' on the Design page to state it explicitly."
                )

        span = effect * spec.sd
        spec.targets = {
            c: float(spec.midpoint + (ranks.get(c, 0.5) - 0.5) * span) for c in cond_labels
        }
        specs.append(spec)

    return specs, warnings


def normalize_comprehension_specs(
    raw_specs: Sequence[Any],
    conditions: Sequence[str] = (),
) -> Tuple[List[ComprehensionCheckSpec], List[str]]:
    """Normalise comprehension-check entries.

    A check with no answer options is generated as a 1/0 coded column rather
    than as text: the old code emitted the literal string ``"Wrong_<answer>"``,
    which is not an answer any participant could have given.
    """
    specs: List[ComprehensionCheckSpec] = []
    warnings: List[str] = []
    seen: set = set()
    cond_labels = [str(c) for c in conditions]

    for raw in raw_specs or []:
        if not isinstance(raw, dict):
            raw = {"name": str(raw or ""), "correct_answer": ""}
        name = str(raw.get("name", "") or "").strip()
        if not name:
            warnings.append("A comprehension check with no name was skipped.")
            continue
        column = normalize_check_name(name, fallback="CompCheck")
        if column in seen:
            warnings.append(
                f"Comprehension check '{name}' duplicates an earlier entry and was skipped."
            )
            continue

        by_condition_raw = raw.get("correct_by_condition") or {}
        by_condition = {
            str(k): str(v).strip()
            for k, v in by_condition_raw.items()
            if str(k) in cond_labels and str(v).strip()
        }
        correct = str(raw.get("correct_answer", "") or "").strip()
        if not correct and by_condition:
            correct = next(iter(by_condition.values()))
        if not correct:
            warnings.append(
                f"Comprehension check '{name}' has no correct answer, so no column "
                "was generated. Enter the correct answer on the Design page."
            )
            continue
        seen.add(column)

        options = [o.strip() for o in str(raw.get("answer_options", "") or "").split(";") if o.strip()]
        if not options and isinstance(raw.get("options"), (list, tuple)):
            options = [str(o).strip() for o in raw["options"] if str(o).strip()]
        all_correct = set(by_condition.values()) | {correct}
        coded_only = False
        if options:
            missing = [c for c in sorted(all_correct) if c not in options]
            if missing:
                options = options + missing
        else:
            coded_only = True
            warnings.append(
                f"Comprehension check '{name}' has no answer options, so it was "
                "generated as a 1/0 correct/incorrect column. Enter the options "
                "(semicolon-separated) on the Design page to get labelled answers."
            )

        rate = _as_float(raw.get("pass_rate"), 0.90)
        if rate > 1.0:
            rate = rate / 100.0
        rate = float(np.clip(rate, 0.0, 1.0))

        specs.append(ComprehensionCheckSpec(
            name=name,
            column=column,
            correct_answer=correct,
            options=options,
            pass_rate=rate,
            question_text=str(raw.get("question_text", "") or ""),
            correct_by_condition=by_condition,
            coded_only=coded_only,
        ))

    return specs, warnings


def normalize_attention_specs(raw_specs: Sequence[Any]) -> Tuple[List[AttentionCheckSpec], List[str]]:
    specs: List[AttentionCheckSpec] = []
    warnings: List[str] = []
    seen: set = set()
    for idx, raw in enumerate(raw_specs or []):
        name = str(raw.get("name", "") if isinstance(raw, dict) else raw or "").strip()
        if not name:
            warnings.append("An attention check with no name was skipped.")
            continue
        column = normalize_check_name(name, fallback=f"AttnCheck_{idx + 1}")
        if column in seen:
            warnings.append(f"Attention check '{name}' duplicates an earlier entry and was skipped.")
            continue
        seen.add(column)
        specs.append(AttentionCheckSpec(name=name, column=column))
    return specs, warnings


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def calibrate_probability_scale(weights: Sequence[float], target_mean: float) -> float:
    """Solve ``scale`` so that ``mean(clip(scale * w, 0, 1)) == target_mean``.

    The configured pass rate is a property of the SAMPLE, not of an attentive
    participant: the old code multiplied the configured rate by 0.6 for
    inattentive participants, so the realised rate was always below what the
    analyst asked for. Bisection is needed because the cap at 1 makes the mean
    non-linear in the scale (the same reason ``_calibrate_attention_fail_scale``
    uses it).
    """
    w = np.asarray(list(weights), dtype=float)
    target = float(np.clip(target_mean, 0.0, 1.0))
    if w.size == 0 or float(w.sum()) <= 0.0:
        return 0.0
    if target <= 0.0:
        return 0.0
    if target >= 1.0:
        return 1e4
    lo, hi = 0.0, 1e4
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if float(np.minimum(1.0, w * mid).mean()) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _engagement(traits: Dict[str, Any]) -> float:
    try:
        value = float(traits.get("attention_level", 0.85))
    except (TypeError, ValueError):
        return 0.85
    if not np.isfinite(value):
        return 0.85
    return float(np.clip(value, 0.0, 1.0))


def generate_manipulation_values(
    spec: ManipulationCheckSpec,
    conditions: Sequence[str],
    all_traits: Sequence[Dict[str, Any]],
    dv_z: Sequence[float],
    seed: int,
) -> Tuple[List[float], List[int]]:
    """Draw the manipulation-check answers and their pass flags.

    The answer is the participant's condition target, pulled toward the scale
    midpoint in proportion to inattention, nudged by the participant's own
    standardised outcome score (real manipulation checks correlate with the DV),
    and rounded onto the configured scale.

    ``pass`` is nearest-target classification: the participant passes when their
    answer is at least as close to their own condition's target as to any other
    condition's. For two arms that is the midpoint rule; for three or more it
    stays meaningful, which a midpoint rule does not.
    """
    n = len(conditions)
    values: List[float] = []
    passes: List[int] = []
    targets = spec.targets or {}
    default_target = spec.midpoint
    target_items = sorted(targets.items())
    sd = spec.sd

    for i in range(n):
        cond = str(conditions[i])
        own_target = float(targets.get(cond, default_target))
        traits = all_traits[i] if i < len(all_traits) else {}
        eng = _engagement(traits if isinstance(traits, dict) else {})
        rng = np.random.RandomState((int(seed) + i * 7919) % (2 ** 31))

        pull = 0.45 * (1.0 - eng)
        mu = spec.midpoint + (own_target - spec.midpoint) * (1.0 - pull)
        try:
            z = float(dv_z[i])
        except (IndexError, TypeError, ValueError):
            z = 0.0
        if not np.isfinite(z):
            z = 0.0
        mu += 0.25 * sd * float(np.clip(z, -3.0, 3.0))
        sd_i = sd * (1.0 + 0.6 * (1.0 - eng))

        raw = rng.normal(mu, sd_i)
        value = int(np.clip(round(raw), spec.scale_min, spec.scale_max))
        values.append(value)

        if target_items:
            own_gap = abs(value - own_target)
            passed = all(own_gap <= abs(value - other) + 1e-9
                         for label, other in target_items if label != cond)
        else:
            passed = True
        passes.append(1 if passed else 0)

    return values, passes


def generate_comprehension_values(
    spec: ComprehensionCheckSpec,
    conditions: Sequence[str],
    all_traits: Sequence[Dict[str, Any]],
    seed: int,
) -> Tuple[List[Any], List[int]]:
    """Draw comprehension-check answers calibrated to the configured pass rate.

    Attentive participants are likelier to be correct, but the SAMPLE pass rate
    equals the configured one (see :func:`calibrate_probability_scale`).
    """
    n = len(conditions)
    weights = []
    for i in range(n):
        traits = all_traits[i] if i < len(all_traits) else {}
        eng = _engagement(traits if isinstance(traits, dict) else {})
        weights.append(0.45 + 0.55 * eng)
    scale = calibrate_probability_scale(weights, spec.pass_rate)

    answers: List[Any] = []
    passes: List[int] = []
    for i in range(n):
        rng = np.random.RandomState((int(seed) + i * 6367) % (2 ** 31))
        p_correct = float(np.clip(weights[i] * scale, 0.0, 1.0))
        correct = spec.correct_by_condition.get(str(conditions[i]), spec.correct_answer)
        if rng.random() < p_correct:
            answers.append(1 if spec.coded_only else correct)
            passes.append(1)
            continue
        passes.append(0)
        if spec.coded_only:
            answers.append(0)
            continue
        wrong = [o for o in spec.options if o != correct]
        if wrong:
            answers.append(str(wrong[int(rng.randint(0, len(wrong)))]))
        else:
            # Only reachable when every option equals the correct answer.
            answers.append(correct)
            passes[-1] = 1
    return answers, passes
