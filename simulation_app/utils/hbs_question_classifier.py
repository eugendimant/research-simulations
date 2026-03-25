"""
HBS Question Type Classifier

Classifies survey questions into categories that determine how the
Human Behavior Simulator generates responses. Detection follows a
priority order so that attention checks and cognitive reflection items
are caught before falling through to more generic categories.
"""

import re
import logging
from typing import Dict, List, Optional

__all__ = ["HBSQuestionClassifier"]

logger = logging.getLogger(__name__)


class HBSQuestionClassifier:
    """Classifies survey questions into response-generation categories."""

    def __init__(self) -> None:
        self._compile_patterns()

    # ------------------------------------------------------------------
    # Pattern compilation
    # ------------------------------------------------------------------

    def _compile_patterns(self) -> None:
        """Pre-compile all regex patterns used for classification."""

        _I = re.IGNORECASE

        # --- attention_check ---
        self._attention_instructed = re.compile(
            r"please\s+select|choose\s+.*option|select\s+.*below", _I
        )
        self._attention_trap = re.compile(
            r"if\s+you\s+are\s+reading|quality\s+check|attention\s+check|attention", _I
        )
        self._attention_directed = re.compile(
            r"to\s+show\s+you\s+are\s+paying\s+attention|directed\s+response", _I
        )

        # --- cognitive_reflection ---
        self._crt = re.compile(
            r"bat\s+and\s+(a\s+)?ball|lily\s+pad|machines?\s*.*widgets?"
            r"|costs?\s*.*more\s+than", _I
        )

        # --- knowledge_factual ---
        self._knowledge_numeric = re.compile(
            r"how\s+many|how\s+much\s+is|what\s+is\s+\d|calculate|sum\s+of"
            r"|total\s+of|percentage\s+of", _I
        )
        self._knowledge_verbal = re.compile(
            r"what\s+does\s+.*\s+mean|define\s+the\s+term|synonym\s+of"
            r"|antonym\s+of|which\s+word", _I
        )
        self._knowledge_trivia = re.compile(
            r"what\s+is\s+the\b|what\s+year|who\s+is\s+the|who\s+was\s+the"
            r"|capital\s+of|tallest|largest|smallest|longest|shortest"
            r"|which\s+country|which\s+planet|which\s+element", _I
        )
        self._math_hint = re.compile(
            r"calculate|sum\s+of|total\s+of|how\s+much\s+is|percentage"
            r"|\d+\s*[\+\-\*/]\s*\d+", _I
        )
        self._counting_hint = re.compile(
            r"how\s+many|count\s+the\s+number|number\s+of\s+times", _I
        )

        # --- behavioral_game ---
        self._game_dictator = re.compile(
            r"divide\s+.*between|split\s+.*between|allocate\s+.*to\s+the\s+other"
            r"|how\s+much\s+.*give\s+to\s+the\s+other|dictator", _I
        )
        self._game_trust = re.compile(
            r"send\s+.*to\s+(player|person|participant|the\s+other)"
            r"|trust\s+game|amount\s+to\s+send|tripled", _I
        )
        self._game_ultimatum = re.compile(
            r"ultimatum|propose\s+.*split|accept\s+or\s+reject"
            r"|reject\s+the\s+offer|offer\s+.*to\s+the\s+other", _I
        )
        self._game_public_goods = re.compile(
            r"public\s+goods|contribute\s+to\s+(the\s+)?(common|group|public)\s+(pool|fund|pot)"
            r"|group\s+project\s+contribution", _I
        )
        self._game_generic = re.compile(
            r"endowed\s+with|matched\s+with\s+(a\s+)?(partner|player|participant)"
            r"|allocate|contribute", _I
        )

        # --- visual ---
        self._visual = re.compile(
            r"look\s+at\s+the\s+image|which\s+is\s+larger|count\s+the"
            r"|picture|photograph|diagram|chart\s+below|graph\s+below"
            r"|image\s+below|figure\s+below", _I
        )

        # --- temporal ---
        self._temporal = re.compile(
            r"what\s+day\s+is|what\s+time\s+is|current\s+date|today'?s?\s+date"
            r"|what\s+year\s+is\s+it|what\s+month\s+is", _I
        )

        # --- demographic ---
        self._demo_age = re.compile(r"\byour\s+age\b|how\s+old\s+are\s+you", _I)
        self._demo_gender = re.compile(r"\bgender\b|\bsex\b|identify\s+as", _I)
        self._demo_education = re.compile(
            r"\beducation\b|highest\s+degree|level\s+of\s+schooling", _I
        )
        self._demo_income = re.compile(
            r"\bincome\b|household\s+earnings|annual\s+salary|socioeconomic", _I
        )
        self._demo_location = re.compile(
            r"\bzip\s*code\b|\bstate\b.*live|country\s+of\s+residence"
            r"|where\s+.*\b(born|live|reside)\b", _I
        )
        self._demo_political = re.compile(
            r"political\s+(affiliation|party|orientation|leaning)"
            r"|democrat|republican|liberal\s+or\s+conservative"
            r"|left.right\s+scale", _I
        )
        self._demo_generic = re.compile(
            r"race|ethnicity|\bmarital\b|employment\s+status"
            r"|first\s+language|native\s+language|religious|religion", _I
        )

        # --- likert_scale ---
        self._likert_agreement = re.compile(
            r"strongly\s+agree|agree.*disagree|level\s+of\s+agreement", _I
        )
        self._likert_frequency = re.compile(
            r"how\s+often|frequency|never.*always|rarely.*frequently", _I
        )
        self._likert_satisfaction = re.compile(
            r"how\s+satisfied|satisfaction|very\s+dissatisfied.*very\s+satisfied", _I
        )
        self._likert_importance = re.compile(
            r"how\s+important|importance|not\s+at\s+all\s+important"
            r".*extremely\s+important", _I
        )
        self._likert_generic = re.compile(
            r"how\s+much\s+do\s+you|to\s+what\s+extent|on\s+a\s+scale"
            r"|rate\s+your|rate\s+the|from\s+1\s+to|from\s+0\s+to", _I
        )

        # --- open_ended ---
        self._open_ended_cues = re.compile(
            r"\bdescribe\b|\bexplain\b|tell\s+us|in\s+your\s+own\s+words"
            r"|please\s+elaborate|write\s+about|share\s+your\s+(thoughts|experience)"
            r"|open.ended|free\s+response", _I
        )
        self._oe_factual = re.compile(
            r"what\s+is\s+the|name\s+the|list\s+the|identify\s+the", _I
        )
        self._oe_creative = re.compile(
            r"imagine|create|design|invent|come\s+up\s+with|write\s+a\s+story", _I
        )
        self._oe_experience = re.compile(
            r"your\s+experience|a\s+time\s+when|have\s+you\s+ever"
            r"|personal\s+example", _I
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        question_text: str,
        question_type: str = "",
        choices: Optional[List[str]] = None,
        question_id: str = "",
    ) -> Dict[str, object]:
        """Classify a survey question and return a structured result dict.

        Parameters
        ----------
        question_text : str
            The full text of the question.
        question_type : str
            Qualtrics-style question type code (e.g. ``"MC"``, ``"TE"``).
        choices : list[str] | None
            Answer options, if any.
        question_id : str
            Optional identifier (used only for logging).

        Returns
        -------
        dict with keys: category, subcategory, tool_hint,
            requires_knowledge, has_correct_answer, sensitivity, confidence.
        """
        if not question_text:
            return self._result("unknown", "none", confidence=0.0)

        text = question_text.strip()
        qt = (question_type or "").strip()
        has_choices = bool(choices and len(choices) > 0)

        # Priority-ordered detection chain
        for detector in (
            self._detect_attention_check,
            self._detect_cognitive_reflection,
            self._detect_knowledge_factual,
            self._detect_behavioral_game,
            self._detect_visual,
            self._detect_temporal,
            self._detect_demographic,
            self._detect_likert_scale,
            self._detect_open_ended,
        ):
            result = detector(text, qt, has_choices)
            if result is not None:
                return result

        # Fallback: if choices exist it is generic multiple-choice
        if has_choices:
            return self._result("multiple_choice", "general", confidence=0.5)

        return self._result("unknown", "none", confidence=0.0)

    # ------------------------------------------------------------------
    # Detectors (private, priority-ordered)
    # ------------------------------------------------------------------

    def _detect_attention_check(
        self, text: str, qt: str, has_choices: bool
    ) -> Optional[Dict[str, object]]:
        if self._attention_directed.search(text):
            return self._result(
                "attention_check", "directed",
                has_correct_answer=True, confidence=0.95,
            )
        if self._attention_trap.search(text):
            return self._result(
                "attention_check", "trap",
                has_correct_answer=True, confidence=0.90,
            )
        if self._attention_instructed.search(text) and self._looks_like_attention(text):
            return self._result(
                "attention_check", "instructed",
                has_correct_answer=True, confidence=0.85,
            )
        return None

    def _detect_cognitive_reflection(
        self, text: str, qt: str, has_choices: bool
    ) -> Optional[Dict[str, object]]:
        if self._crt.search(text):
            return self._result(
                "cognitive_reflection", "crt",
                tool_hint="compute_arithmetic",
                requires_knowledge=True,
                has_correct_answer=True,
                confidence=0.95,
            )
        return None

    def _detect_knowledge_factual(
        self, text: str, qt: str, has_choices: bool
    ) -> Optional[Dict[str, object]]:
        if self._math_hint.search(text):
            return self._result(
                "knowledge_factual", "numeric",
                tool_hint="compute_arithmetic",
                requires_knowledge=True,
                has_correct_answer=True,
                confidence=0.90,
            )
        if self._counting_hint.search(text):
            return self._result(
                "knowledge_factual", "numeric",
                tool_hint="analyze_text",
                requires_knowledge=True,
                has_correct_answer=True,
                confidence=0.85,
            )
        if self._knowledge_verbal.search(text):
            return self._result(
                "knowledge_factual", "verbal",
                tool_hint="lookup_factual",
                requires_knowledge=True,
                has_correct_answer=True,
                confidence=0.85,
            )
        if self._knowledge_trivia.search(text):
            return self._result(
                "knowledge_factual", "trivia",
                tool_hint="lookup_factual",
                requires_knowledge=True,
                has_correct_answer=True,
                confidence=0.80,
            )
        if self._knowledge_numeric.search(text):
            return self._result(
                "knowledge_factual", "numeric",
                tool_hint="compute_arithmetic",
                requires_knowledge=True,
                has_correct_answer=True,
                confidence=0.80,
            )
        return None

    def _detect_behavioral_game(
        self, text: str, qt: str, has_choices: bool
    ) -> Optional[Dict[str, object]]:
        if self._game_dictator.search(text):
            return self._result(
                "behavioral_game", "dictator", confidence=0.90,
            )
        if self._game_trust.search(text):
            return self._result(
                "behavioral_game", "trust", confidence=0.90,
            )
        if self._game_ultimatum.search(text):
            return self._result(
                "behavioral_game", "ultimatum", confidence=0.90,
            )
        if self._game_public_goods.search(text):
            return self._result(
                "behavioral_game", "public_goods", confidence=0.90,
            )
        if self._game_generic.search(text):
            return self._result(
                "behavioral_game", "other", confidence=0.70,
            )
        return None

    def _detect_visual(
        self, text: str, qt: str, has_choices: bool
    ) -> Optional[Dict[str, object]]:
        if self._visual.search(text):
            return self._result(
                "visual", "image",
                tool_hint="analyze_image",
                requires_knowledge=True,
                confidence=0.85,
            )
        return None

    def _detect_temporal(
        self, text: str, qt: str, has_choices: bool
    ) -> Optional[Dict[str, object]]:
        if self._temporal.search(text):
            return self._result(
                "temporal", "current_time",
                tool_hint="get_current_datetime",
                requires_knowledge=True,
                has_correct_answer=True,
                confidence=0.90,
            )
        return None

    def _detect_demographic(
        self, text: str, qt: str, has_choices: bool
    ) -> Optional[Dict[str, object]]:
        for pattern, subcat in (
            (self._demo_age, "age"),
            (self._demo_gender, "gender"),
            (self._demo_education, "education"),
            (self._demo_income, "income"),
            (self._demo_location, "location"),
            (self._demo_political, "political"),
            (self._demo_generic, "other"),
        ):
            if pattern.search(text):
                sensitivity = "medium" if subcat in ("income", "political") else "low"
                return self._result(
                    "demographic", subcat,
                    sensitivity=sensitivity,
                    confidence=0.85,
                )
        return None

    def _detect_likert_scale(
        self, text: str, qt: str, has_choices: bool
    ) -> Optional[Dict[str, object]]:
        for pattern, subcat in (
            (self._likert_agreement, "agreement"),
            (self._likert_frequency, "frequency"),
            (self._likert_satisfaction, "satisfaction"),
            (self._likert_importance, "importance"),
            (self._likert_generic, "agreement"),
        ):
            if pattern.search(text):
                return self._result(
                    "likert_scale", subcat, confidence=0.85,
                )
        return None

    def _detect_open_ended(
        self, text: str, qt: str, has_choices: bool
    ) -> Optional[Dict[str, object]]:
        is_te = qt.upper() in ("TE", "TEXTENTRY")
        has_oe_cue = self._open_ended_cues.search(text)

        if is_te or (has_oe_cue and not has_choices):
            subcat = "opinion"
            tool_hint: Optional[str] = None
            requires_knowledge = False
            has_correct = False

            if self._oe_factual.search(text):
                subcat = "factual"
                tool_hint = "lookup_factual"
                requires_knowledge = True
                has_correct = True
            elif self._oe_creative.search(text):
                subcat = "creative"
            elif self._oe_experience.search(text):
                subcat = "experience"

            confidence = 0.90 if is_te else 0.75
            return self._result(
                "open_ended", subcat,
                tool_hint=tool_hint,
                requires_knowledge=requires_knowledge,
                has_correct_answer=has_correct,
                confidence=confidence,
            )
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_attention(text: str) -> bool:
        """Heuristic: short text with an explicit instruction is likely an
        attention check rather than a regular MC question."""
        lower = text.lower()
        cues = (
            "to show you are paying attention",
            "carefully read",
            "please select the following",
            "do not select",
            "select strongly agree",
            "this is an attention",
        )
        return any(c in lower for c in cues) or len(text.split()) < 15

    @staticmethod
    def _result(
        category: str,
        subcategory: str,
        *,
        tool_hint: Optional[str] = None,
        requires_knowledge: bool = False,
        has_correct_answer: bool = False,
        sensitivity: str = "low",
        confidence: float = 0.80,
    ) -> Dict[str, object]:
        return {
            "category": category,
            "subcategory": subcategory,
            "tool_hint": tool_hint,
            "requires_knowledge": requires_knowledge,
            "has_correct_answer": has_correct_answer,
            "sensitivity": sensitivity,
            "confidence": confidence,
        }
