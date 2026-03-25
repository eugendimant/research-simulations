"""
Human Behavior Simulator (HBS) — Distributional Error Calibrator

Ensures simulated participants get knowledge/attention questions wrong at
empirically-documented rates, stratified by education level and response style.

Calibration data is loaded from data/hbs_error_calibration.json at construction
time, with a hardcoded minimal fallback table if the file is missing.
"""

import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Union

__all__ = ["HBSErrorCalibrator"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded minimal calibration table (fallback when JSON file is missing)
# ---------------------------------------------------------------------------
_MINIMAL_CALIBRATION_TABLE: Dict[str, Any] = {
    "bat_and_ball": {
        "patterns": [
            r"bat\s+and\s+(?:a\s+)?ball\s+cost",
            r"bat\s+costs?\s+\$?1(?:\.00)?\s+more\s+than\s+the\s+ball",
        ],
        "correct_answer": "0.05",
        "correct_rate_by_education": {
            "no_degree": 0.20,
            "HS": 0.33,
            "BA": 0.48,
            "MA": 0.56,
            "PhD": 0.64,
        },
        "wrong_answers": {
            "0.10": 0.80,
            "other_numeric": 0.15,
            "i_dont_know": 0.05,
        },
    },
    "lily_pad": {
        "patterns": [
            r"patch\s+of\s+lily\s+pads",
            r"lily\s+pad(?:s)?\s+doubl(?:es?|ing)\s+in\s+size",
        ],
        "correct_answer": "47",
        "correct_rate_by_education": {
            "no_degree": 0.22,
            "HS": 0.35,
            "BA": 0.50,
            "MA": 0.58,
            "PhD": 0.65,
        },
        "wrong_answers": {
            "24": 0.70,
            "other_numeric": 0.20,
            "i_dont_know": 0.10,
        },
    },
    "widget_machines": {
        "patterns": [
            r"machines?\s+(?:take|need)?\s*5\s+minutes?\s+to\s+make\s+5\s+widgets?",
            r"widget(?:s)?\s+(?:machine|factory)",
        ],
        "correct_answer": "5",
        "correct_rate_by_education": {
            "no_degree": 0.28,
            "HS": 0.40,
            "BA": 0.55,
            "MA": 0.62,
            "PhD": 0.70,
        },
        "wrong_answers": {
            "100": 0.75,
            "other_numeric": 0.15,
            "i_dont_know": 0.10,
        },
    },
    "attention_check_simple": {
        "patterns": [
            r"please\s+select\s+(?:strongly\s+)?(?:agree|disagree|option|answer)",
            r"to\s+show\s+you\s+are\s+paying\s+attention",
            r"quality\s+control\s+question",
        ],
        "correct_answer": None,
        "correct_rate_by_education": {
            "no_degree": 0.82,
            "HS": 0.88,
            "BA": 0.92,
            "MA": 0.95,
            "PhD": 0.97,
        },
        "wrong_answers": {
            "random_int_1_7": 1.0,
        },
    },
    "attention_check_reverse": {
        "patterns": [
            r"do\s+not\s+answer\s+this\s+question",
            r"leave\s+this\s+(?:question\s+)?blank",
            r"skip\s+this\s+(?:question|item)",
        ],
        "correct_answer": "",
        "correct_rate_by_education": {
            "no_degree": 0.70,
            "HS": 0.78,
            "BA": 0.85,
            "MA": 0.90,
            "PhD": 0.94,
        },
        "wrong_answers": {
            "random_int_1_7": 1.0,
        },
    },
    "population_estimate": {
        "patterns": [
            r"how\s+many\s+people\s+(?:live|are)\s+in\s+the\s+(?:united\s+states|US|U\.S\.)",
            r"population\s+of\s+the\s+(?:united\s+states|US|U\.S\.)",
        ],
        "correct_answer": "330000000",
        "correct_rate_by_education": {
            "no_degree": 0.15,
            "HS": 0.30,
            "BA": 0.52,
            "MA": 0.60,
            "PhD": 0.68,
        },
        "wrong_answers": {
            "millions": 0.50,
            "other_numeric": 0.35,
            "i_dont_know": 0.15,
        },
    },
}


class HBSErrorCalibrator:
    """Distributional human-error calibrator for the Human Behavior Simulator.

    Ensures simulated participants get knowledge / attention-check questions
    wrong at empirically-documented rates, stratified by education level and
    response style.
    """

    # Recognized education levels in ascending order of accuracy
    _EDUCATION_LEVELS = ("no_degree", "HS", "BA", "MA", "PhD")
    _DEFAULT_EDUCATION = "HS"
    _DEFAULT_PERSONA = "engaged_responder"

    # ------------------------------------------------------------------
    # I-don't-know and millions answer pools
    # ------------------------------------------------------------------
    _IDK_ANSWERS: List[str] = ["I don't know", "idk", "not sure", "no idea"]
    _MILLIONS_ANSWERS: List[str] = ["millions", "a million", "like a million"]

    def __init__(self) -> None:
        """Load calibration table from JSON file or fall back to hardcoded table."""
        self._calibration_table: Dict[str, Any] = {}
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}

        json_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "hbs_error_calibration.json",
        )

        loaded = False
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and data:
                    self._calibration_table = data
                    loaded = True
                    logger.info(
                        "Loaded HBS error calibration from %s (%d entries)",
                        json_path,
                        len(data),
                    )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed to load calibration file %s: %s — using fallback table",
                    json_path,
                    exc,
                )

        if not loaded:
            self._calibration_table = _MINIMAL_CALIBRATION_TABLE
            logger.info(
                "Using hardcoded minimal calibration table (%d entries)",
                len(self._calibration_table),
            )

        # Pre-compile all regex patterns once
        for key, entry in self._calibration_table.items():
            patterns_raw = entry.get("patterns", [])
            compiled: List[re.Pattern] = []
            for pat in patterns_raw:
                try:
                    compiled.append(re.compile(pat, re.IGNORECASE))
                except re.error as exc:
                    logger.warning(
                        "Invalid regex in calibration key '%s': %s — skipping pattern",
                        key,
                        exc,
                    )
            self._compiled_patterns[key] = compiled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_question(
        self, question_text: str, question_id: str = ""
    ) -> Optional[str]:
        """Match *question_text* against calibration table patterns.

        Returns the calibration key (e.g. ``"bat_and_ball"``) or ``None`` if
        no pattern matches.
        """
        if not question_text:
            return None

        text = question_text.strip()

        for key, patterns in self._compiled_patterns.items():
            for pat in patterns:
                if pat.search(text):
                    logger.debug(
                        "Question '%s' (id=%s) matched calibration key '%s'",
                        text[:60],
                        question_id,
                        key,
                    )
                    return key

        return None

    def should_answer_correctly(
        self,
        calibration_key: str,
        education_level: str,
        persona_id: str = "",
        rng: Optional[random.Random] = None,
    ) -> bool:
        """Return ``True`` if this participant should answer correctly.

        Uses a probabilistic draw from the education-stratified correct rate
        stored in the calibration table.  If the entry uses
        ``correct_rate_by_persona`` instead of ``correct_rate_by_education``,
        the persona-keyed rate is used.
        """
        rng = rng or random.Random()
        entry = self._calibration_table.get(calibration_key)
        if entry is None:
            logger.warning(
                "Unknown calibration key '%s' — defaulting to correct", calibration_key
            )
            return True

        # Try persona-keyed rates first
        persona_rates = entry.get("correct_rate_by_persona")
        if persona_rates is not None:
            pid = persona_id or self._DEFAULT_PERSONA
            rate = persona_rates.get(pid)
            if rate is None:
                # Fall through to education rates if persona not listed
                rate = self._resolve_education_rate(entry, education_level)
        else:
            rate = self._resolve_education_rate(entry, education_level)

        if rate is None:
            # Absolute fallback — assume 50 %
            rate = 0.50
            logger.warning(
                "No rate found for key '%s', edu='%s', persona='%s' — using 0.50",
                calibration_key,
                education_level,
                persona_id,
            )

        return rng.random() < rate

    def get_wrong_answer(
        self,
        calibration_key: str,
        correct_answer: str = "",
        rng: Optional[random.Random] = None,
    ) -> str:
        """Return a calibrated wrong answer drawn from the empirical error distribution.

        Special answer-pattern tokens:
        - ``"random_int_X_Y"`` — random int in [X, Y]
        - ``"i_dont_know"``    — random "idk"-style phrase
        - ``"millions"``       — random "millions"-style phrase
        - ``"other_numeric"``  — random plausible numeric value
        - anything else        — returned as a literal string
        """
        rng = rng or random.Random()
        entry = self._calibration_table.get(calibration_key)

        if entry is None:
            logger.warning(
                "Unknown calibration key '%s' — returning generic wrong answer",
                calibration_key,
            )
            return "I don't know"

        wrong_answers: Dict[str, float] = entry.get("wrong_answers", {})
        if not wrong_answers:
            return "I don't know"

        # Weighted draw from the wrong-answer distribution
        answers = list(wrong_answers.keys())
        weights = [wrong_answers[a] for a in answers]
        chosen = rng.choices(answers, weights=weights, k=1)[0]

        return self._resolve_answer_token(chosen, correct_answer, rng)

    def calibrate_response(
        self,
        question_text: str,
        question_id: str,
        participant_education: str,
        participant_persona: str,
        original_answer: str = "",
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """High-level calibration: classify, decide correctness, pick answer.

        Returns a dict with keys:
            ``should_calibrate`` (bool) — whether the question matched a pattern
            ``calibration_key``  (str | None)
            ``correct``          (bool) — whether participant answers correctly
            ``answer``           (str | None) — wrong answer if incorrect, else None
            ``source``           (str) — ``"hbs_error_calibrator"``
        """
        rng = rng or random.Random()

        key = self.classify_question(question_text, question_id)
        if key is None:
            return {
                "should_calibrate": False,
                "calibration_key": None,
                "correct": True,
                "answer": None,
                "source": "hbs_error_calibrator",
            }

        entry = self._calibration_table.get(key, {})
        correct_answer = entry.get("correct_answer", original_answer) or original_answer

        is_correct = self.should_answer_correctly(
            key, participant_education, participant_persona, rng
        )

        if is_correct:
            return {
                "should_calibrate": True,
                "calibration_key": key,
                "correct": True,
                "answer": None,
                "source": "hbs_error_calibrator",
            }

        wrong = self.get_wrong_answer(key, correct_answer, rng)
        return {
            "should_calibrate": True,
            "calibration_key": key,
            "correct": False,
            "answer": wrong,
            "source": "hbs_error_calibrator",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_education_rate(
        self, entry: Dict[str, Any], education_level: str
    ) -> Optional[float]:
        """Look up the correct-rate for *education_level* with graceful fallback."""
        edu_rates = entry.get("correct_rate_by_education")
        if edu_rates is None:
            return None

        edu = education_level if education_level in edu_rates else self._DEFAULT_EDUCATION
        if edu not in edu_rates:
            # If even the default is missing, pick the first available
            if edu_rates:
                edu = next(iter(edu_rates))
            else:
                return None

        rate = edu_rates[edu]
        try:
            return float(rate)
        except (TypeError, ValueError):
            return None

    def _resolve_answer_token(
        self, token: str, correct_answer: str, rng: random.Random
    ) -> str:
        """Expand special answer-pattern tokens into concrete strings."""

        # random_int_X_Y
        m = re.match(r"^random_int_(\d+)_(\d+)$", token)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            return str(rng.randint(lo, hi))

        if token == "i_dont_know":
            return rng.choice(self._IDK_ANSWERS)

        if token == "millions":
            return rng.choice(self._MILLIONS_ANSWERS)

        if token == "other_numeric":
            return self._generate_other_numeric(correct_answer, rng)

        # Literal answer string
        return token

    @staticmethod
    def _generate_other_numeric(correct_answer: str, rng: random.Random) -> str:
        """Generate a plausible but wrong numeric value near *correct_answer*."""
        try:
            correct_val = float(correct_answer)
        except (TypeError, ValueError):
            # No parseable correct answer — return a small random number
            return str(rng.randint(1, 100))

        if correct_val == 0:
            return str(rng.randint(1, 10))

        # Perturb by 1.5x–5x in either direction, avoiding the correct value
        multiplier = rng.choice([0.2, 0.5, 2.0, 3.0, 5.0, 10.0])
        wrong_val = correct_val * multiplier

        # Format: match the precision of the correct answer
        if "." in correct_answer:
            decimal_places = len(correct_answer.split(".")[-1])
            return f"{wrong_val:.{decimal_places}f}"

        return str(int(round(wrong_val)))
