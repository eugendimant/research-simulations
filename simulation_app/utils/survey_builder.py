"""
Conversational Survey Builder - Natural Language Study Specification Parser

Version: 1.2.8
Allows users to describe their experiment in words instead of uploading a QSF file.
Parses natural language descriptions into structured survey specifications that
feed directly into the EnhancedSimulationEngine.

This module handles:
- Parsing condition/group descriptions from free text
- Extracting scale/DV specifications from measure descriptions
- Detecting factorial designs from condition structure
- Identifying open-ended questions from descriptions
- Domain detection from study descriptions
- Converting all parsed data into the same inferred_design format used by QSF upload
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

__version__ = "1.3.7"


# ─── Common scale anchors used in behavioral science ───────────────────────────
COMMON_SCALE_ANCHORS = {
    "likert_7": {"min": 1, "max": 7, "label": "1-7 Likert"},
    "likert_5": {"min": 1, "max": 5, "label": "1-5 Likert"},
    "likert_9": {"min": 1, "max": 9, "label": "1-9 Likert"},
    "likert_10": {"min": 1, "max": 10, "label": "1-10 Likert"},
    "slider": {"min": 0, "max": 100, "label": "0-100 Slider"},
    "binary": {"min": 0, "max": 1, "label": "Yes/No (0-1)"},
    "percentage": {"min": 0, "max": 100, "label": "0-100 Percentage"},
    "wtp": {"min": 0, "max": 1000, "label": "Willingness to Pay ($)"},
}

# ─── Known scale names in behavioral science ───────────────────────────────────
KNOWN_SCALES = {
    # Attitudes & Satisfaction
    "satisfaction": {"items": 5, "min": 1, "max": 7, "label": "Satisfaction Scale"},
    "attitude": {"items": 4, "min": 1, "max": 7, "label": "Attitude Scale"},
    "trust": {"items": 5, "min": 1, "max": 7, "label": "Trust Scale"},
    "credibility": {"items": 4, "min": 1, "max": 7, "label": "Credibility Scale"},
    "purchase intention": {"items": 3, "min": 1, "max": 7, "label": "Purchase Intention"},
    "willingness to pay": {"items": 1, "min": 0, "max": 100, "label": "WTP"},
    "behavioral intention": {"items": 3, "min": 1, "max": 7, "label": "Behavioral Intention"},
    "risk perception": {"items": 4, "min": 1, "max": 7, "label": "Risk Perception"},
    "perceived fairness": {"items": 3, "min": 1, "max": 7, "label": "Perceived Fairness"},
    "moral judgment": {"items": 4, "min": 1, "max": 7, "label": "Moral Judgment"},
    "emotional response": {"items": 6, "min": 1, "max": 7, "label": "Emotional Response"},
    "self-efficacy": {"items": 5, "min": 1, "max": 7, "label": "Self-Efficacy"},
    "perceived quality": {"items": 4, "min": 1, "max": 7, "label": "Perceived Quality"},
    "brand attitude": {"items": 4, "min": 1, "max": 7, "label": "Brand Attitude"},

    # Wellbeing & Personality
    "well-being": {"items": 5, "min": 1, "max": 7, "label": "Subjective Well-Being"},
    "life satisfaction": {"items": 5, "min": 1, "max": 7, "label": "Life Satisfaction (SWLS)"},
    "anxiety": {"items": 7, "min": 1, "max": 4, "label": "Anxiety (GAD-7 style)"},
    "stress": {"items": 10, "min": 1, "max": 5, "label": "Perceived Stress Scale"},
    "self-esteem": {"items": 10, "min": 1, "max": 4, "label": "Rosenberg Self-Esteem"},
    "motivation": {"items": 4, "min": 1, "max": 7, "label": "Motivation Scale"},
    "empathy": {"items": 5, "min": 1, "max": 7, "label": "Empathy Scale"},

    # Social Psychology
    "social distance": {"items": 3, "min": 1, "max": 7, "label": "Social Distance"},
    "perceived warmth": {"items": 3, "min": 1, "max": 7, "label": "Perceived Warmth"},
    "perceived competence": {"items": 3, "min": 1, "max": 7, "label": "Perceived Competence"},
    "identification": {"items": 4, "min": 1, "max": 7, "label": "Group Identification"},
    "prejudice": {"items": 5, "min": 1, "max": 7, "label": "Prejudice Scale"},
    "norm perception": {"items": 3, "min": 1, "max": 7, "label": "Social Norm Perception"},

    # Consumer & Marketing
    "ad effectiveness": {"items": 4, "min": 1, "max": 7, "label": "Ad Effectiveness"},
    "product evaluation": {"items": 5, "min": 1, "max": 7, "label": "Product Evaluation"},
    "perceived value": {"items": 3, "min": 1, "max": 7, "label": "Perceived Value"},
    "loyalty": {"items": 4, "min": 1, "max": 7, "label": "Loyalty Scale"},

    # Technology
    "technology acceptance": {"items": 6, "min": 1, "max": 7, "label": "TAM Scale"},
    "perceived usefulness": {"items": 4, "min": 1, "max": 7, "label": "Perceived Usefulness"},
    "ease of use": {"items": 4, "min": 1, "max": 7, "label": "Ease of Use"},
    "privacy concern": {"items": 5, "min": 1, "max": 7, "label": "Privacy Concern"},

    # Validated Instruments (by abbreviation and full name)
    "bfi-10": {"items": 10, "min": 1, "max": 5, "label": "Big Five Inventory-10"},
    "big five inventory": {"items": 10, "min": 1, "max": 5, "label": "Big Five Inventory-10"},
    "panas": {"items": 20, "min": 1, "max": 5, "label": "Positive and Negative Affect Schedule"},
    "positive and negative affect": {"items": 20, "min": 1, "max": 5, "label": "PANAS"},
    "tipi": {"items": 10, "min": 1, "max": 7, "label": "Ten-Item Personality Inventory"},
    "ten-item personality": {"items": 10, "min": 1, "max": 7, "label": "TIPI"},
    "swls": {"items": 5, "min": 1, "max": 7, "label": "Satisfaction with Life Scale"},
    "satisfaction with life": {"items": 5, "min": 1, "max": 7, "label": "SWLS"},
    "gad-7": {"items": 7, "min": 0, "max": 3, "label": "Generalized Anxiety Disorder-7"},
    "generalized anxiety disorder": {"items": 7, "min": 0, "max": 3, "label": "GAD-7"},
    "phq-9": {"items": 9, "min": 0, "max": 3, "label": "Patient Health Questionnaire-9"},
    "patient health questionnaire": {"items": 9, "min": 0, "max": 3, "label": "PHQ-9"},
    "rosenberg self-esteem": {"items": 10, "min": 1, "max": 4, "label": "Rosenberg Self-Esteem Scale"},
    "rse": {"items": 10, "min": 1, "max": 4, "label": "Rosenberg Self-Esteem Scale"},
    "ses": {"items": 3, "min": 1, "max": 10, "label": "Socioeconomic Status Scale"},
    "socioeconomic status": {"items": 3, "min": 1, "max": 10, "label": "Socioeconomic Status Scale"},
    "need for cognition": {"items": 18, "min": 1, "max": 5, "label": "Need for Cognition Scale"},
    "nfc": {"items": 18, "min": 1, "max": 5, "label": "Need for Cognition Scale"},
    "moral foundations": {"items": 20, "min": 1, "max": 6, "label": "Moral Foundations Questionnaire"},
    "mfq": {"items": 20, "min": 1, "max": 6, "label": "Moral Foundations Questionnaire"},
    "sense of belonging": {"items": 8, "min": 1, "max": 7, "label": "Sense of Belonging Scale"},
    "belonging": {"items": 8, "min": 1, "max": 7, "label": "Sense of Belonging Scale"},

    # Additional validated instruments
    "ces-d": {"items": 20, "min": 0, "max": 3, "label": "Center for Epidemiological Studies Depression (CES-D)"},
    "cesd": {"items": 20, "min": 0, "max": 3, "label": "Center for Epidemiological Studies Depression (CES-D)"},
    "system usability": {"items": 10, "min": 1, "max": 5, "label": "System Usability Scale (SUS)"},
    "sus": {"items": 10, "min": 1, "max": 5, "label": "System Usability Scale (SUS)"},
    "state-trait anxiety": {"items": 20, "min": 1, "max": 4, "label": "State-Trait Anxiety Inventory (STAI)"},
    "stai": {"items": 20, "min": 1, "max": 4, "label": "State-Trait Anxiety Inventory (STAI)"},
    "social dominance orientation": {"items": 16, "min": 1, "max": 7, "label": "Social Dominance Orientation (SDO)"},
    "sdo": {"items": 16, "min": 1, "max": 7, "label": "Social Dominance Orientation (SDO)"},
    "right-wing authoritarianism": {"items": 15, "min": 1, "max": 7, "label": "Right-Wing Authoritarianism (RWA)"},
    "rwa": {"items": 15, "min": 1, "max": 7, "label": "Right-Wing Authoritarianism (RWA)"},
    "ucla loneliness": {"items": 20, "min": 1, "max": 4, "label": "UCLA Loneliness Scale"},
    "loneliness": {"items": 20, "min": 1, "max": 4, "label": "UCLA Loneliness Scale"},

    # Iteration 6 additions
    "iri": {"items": 28, "min": 1, "max": 5, "label": "Interpersonal Reactivity Index (IRI)"},
    "interpersonal reactivity index": {"items": 28, "min": 1, "max": 5, "label": "Interpersonal Reactivity Index (IRI)"},
    "big five": {"items": 10, "min": 1, "max": 5, "label": "Big Five Inventory-10"},
    "tam": {"items": 12, "min": 1, "max": 7, "label": "Technology Acceptance Model (TAM)"},
    "technology acceptance model": {"items": 12, "min": 1, "max": 7, "label": "Technology Acceptance Model (TAM)"},
    "net promoter score": {"items": 1, "min": 0, "max": 10, "label": "Net Promoter Score (NPS)"},
    "nps": {"items": 1, "min": 0, "max": 10, "label": "Net Promoter Score (NPS)"},
    "nasa-tlx": {"items": 6, "min": 0, "max": 100, "label": "NASA Task Load Index (NASA-TLX)"},
    "nasa task load index": {"items": 6, "min": 0, "max": 100, "label": "NASA Task Load Index (NASA-TLX)"},
    "tlx": {"items": 6, "min": 0, "max": 100, "label": "NASA Task Load Index (NASA-TLX)"},
    "csq-8": {"items": 8, "min": 1, "max": 4, "label": "Client Satisfaction Questionnaire (CSQ-8)"},
    "csq": {"items": 8, "min": 1, "max": 4, "label": "Client Satisfaction Questionnaire (CSQ-8)"},
    "client satisfaction questionnaire": {"items": 8, "min": 1, "max": 4, "label": "Client Satisfaction Questionnaire (CSQ-8)"},
    "imc": {"items": 1, "min": 0, "max": 1, "label": "Instructional Manipulation Check (IMC)"},
    "instructional manipulation check": {"items": 1, "min": 0, "max": 1, "label": "Instructional Manipulation Check (IMC)"},
}

# ─── Scale abbreviation aliases → canonical KNOWN_SCALES key ──────────────────
SCALE_ABBREVIATIONS: Dict[str, str] = {
    "bfi": "bfi-10",
    "bfi10": "bfi-10",
    "bfi-10": "bfi-10",
    "panas": "panas",
    "ses": "ses",
    "tipi": "tipi",
    "swls": "swls",
    "gad7": "gad-7",
    "gad-7": "gad-7",
    "phq9": "phq-9",
    "phq-9": "phq-9",
    "rse": "rse",
    "nfc": "need for cognition",
    "mfq": "mfq",
    "pss": "stress",
    "cesd": "cesd",
    "ces-d": "ces-d",
    "sus": "sus",
    "stai": "stai",
    "sdo": "sdo",
    "rwa": "rwa",
    # Iteration 6 additions
    "iri": "iri",
    "tam": "tam",
    "nps": "nps",
    "net promoter score": "nps",
    "tlx": "tlx",
    "nasa-tlx": "nasa-tlx",
    "csq": "csq",
    "csq-8": "csq-8",
    "imc": "imc",
}

# ─── Builder domain → Persona library domain mapping ────────────────────────
# Maps display-friendly domain names (used in builder UI) to the internal
# domain keys used by PersonaLibrary.detect_domains() and get_personas_for_domains()
BUILDER_DOMAIN_TO_PERSONA_DOMAIN: Dict[str, List[str]] = {
    "consumer behavior": ["consumer_behavior", "marketing"],
    "social psychology": ["social_psychology"],
    "behavioral economics": ["behavioral_economics", "economic_games"],
    "organizational behavior": ["organizational_behavior"],
    "health psychology": ["health_psychology"],
    "political psychology": ["political_psychology"],
    "cognitive psychology": ["cognitive_psychology"],
    "developmental psychology": ["developmental_psychology"],
    "environmental psychology": ["environmental"],
    "educational psychology": ["educational_psychology"],
    "moral psychology": ["deontology_utilitarianism"],
    "technology & hci": ["ai", "technology"],
    "communication": ["accuracy_misinformation"],
    "food & nutrition": ["consumer_behavior", "health_psychology"],
    "prosocial behavior": ["charitable_giving", "social_psychology"],
    "emotion research": ["emotions"],
}

# All available builder domains (for dropdown display)
AVAILABLE_DOMAINS: List[str] = sorted(BUILDER_DOMAIN_TO_PERSONA_DOMAIN.keys())


@dataclass
class ParsedCondition:
    """A condition extracted from user description."""
    name: str
    description: str = ""
    is_control: bool = False


@dataclass
class ParsedScale:
    """A scale/DV extracted from user description."""
    name: str
    variable_name: str = ""
    num_items: int = 1
    scale_min: int = 1
    scale_max: int = 7
    scale_type: str = "likert"  # likert, slider, numeric, binary
    description: str = ""
    reverse_items: List[int] = field(default_factory=list)
    item_labels: List[str] = field(default_factory=list)


@dataclass
class ParsedOpenEnded:
    """An open-ended question extracted from user description."""
    variable_name: str
    question_text: str
    context_type: str = "general"


@dataclass
class ParsedDesign:
    """Complete parsed study design from conversational input."""
    conditions: List[ParsedCondition] = field(default_factory=list)
    scales: List[ParsedScale] = field(default_factory=list)
    open_ended: List[ParsedOpenEnded] = field(default_factory=list)
    factors: List[Dict[str, Any]] = field(default_factory=list)
    design_type: str = "between"  # between, within, mixed
    sample_size: int = 100
    research_domain: str = ""
    study_title: str = ""
    study_description: str = ""
    attention_checks: List[str] = field(default_factory=list)
    manipulation_checks: List[str] = field(default_factory=list)
    participant_characteristics: str = ""  # Free-text description of expected participants


class SurveyDescriptionParser:
    """
    Parses natural language descriptions of experiments into structured
    survey specifications compatible with the simulation engine.
    """

    def __init__(self) -> None:
        self._condition_keywords = [
            "control", "treatment", "experimental", "condition", "group",
            "manipulation", "intervention", "placebo", "baseline",
            "high", "low", "present", "absent", "positive", "negative",
        ]
        self._scale_keywords = [
            "likert", "scale", "measure", "rate", "rating", "slider",
            "score", "index", "items", "questionnaire", "inventory",
            "subscale", "dimension", "factor", "battery",
        ]
        self._oe_keywords = [
            "open-ended", "open ended", "free text", "free-text",
            "write", "describe", "explain", "text response",
            "essay", "narrative", "qualitative", "verbatim",
        ]

    def parse_conditions(self, text: str) -> Tuple[List[ParsedCondition], List[str]]:
        """
        Extract experimental conditions from a natural language description.

        Returns a tuple of (conditions, warnings) where warnings contains any
        non-fatal messages for the user (e.g., numbered-implicit patterns).

        Handles formats like:
        - "Control group and treatment group"
        - "3 conditions: low, medium, high"
        - "2x2 design with trust (high/low) and risk (high/low)"
        - "Condition 1: AI-generated, Condition 2: Human-written, Condition 3: No information"
        - "Trust (high, low) and Risk (high, low)" → 4 crossed conditions
        - "High/Low Trust x High/Low Risk" → slash-separated factorial
        - "Control vs Treatment" → vs-separated conditions
        - "3 conditions" → warning about underspecification
        - "3 (Annotation: AI vs Human vs None) × 2 (Product: Hedonic vs Utilitarian)" → 6 crossed
        """
        conditions: List[ParsedCondition] = []
        warnings: List[str] = []
        text_clean = text.strip()

        if not text_clean:
            return conditions, warnings

        # ── Strip trailing design metadata ────────────────────────────────
        # Remove ", between-subjects, random assignment." and similar suffixes
        text_clean = re.sub(
            r',?\s*(?:between[- ]subjects?|within[- ]subjects?|mixed[- ]design|'
            r'random(?:ized)?\s+assignment|randomization|repeated[- ]measures)'
            r'[\s.,;]*$',
            '', text_clean, flags=re.IGNORECASE
        ).strip().rstrip('.,;')

        # ── Pre-check: numbered implicit like "3 conditions" without names ──
        implicit_match = re.search(
            r'^(\d+)\s+(?:conditions?|groups?)\s*$',
            text_clean.strip(), re.IGNORECASE
        )
        if implicit_match:
            n = int(implicit_match.group(1))
            warnings.append(
                f"You specified {n} conditions but did not name them. "
                f"Please provide explicit condition names (e.g., "
                f"'Control, Treatment A, Treatment B') for better simulation results."
            )
            return conditions, warnings

        # Pattern 1: Numbered conditions "Condition 1: Name, Condition 2: Name"
        numbered = re.findall(
            r'(?:condition|group|cond\.?)\s*(\d+)\s*[:\-–]\s*([^,\n;]+)',
            text_clean, re.IGNORECASE
        )
        if numbered:
            for _, name in numbered:
                name = name.strip().rstrip('.')
                if name:
                    conditions.append(ParsedCondition(
                        name=name,
                        is_control=self._is_control_condition(name),
                    ))
            return conditions, warnings

        # ── Pattern 1b: Labeled parenthetical factorial ───────────────────
        # Format: "N (Name: level1 vs level2 vs level3) × N (Name: level1 vs level2)"
        # or: "N (Name: level1, level2, level3) x N (Name: level1, level2)"
        labeled_paren = re.findall(
            r'(\d+)\s*\(\s*([^:)]+?)\s*:\s*([^)]+)\s*\)',
            text_clean
        )
        if len(labeled_paren) >= 2:
            factors: List[Dict[str, Any]] = []
            for count_str, factor_name, levels_str in labeled_paren:
                factor_name = factor_name.strip()
                expected_count = int(count_str)
                levels = self._parse_list_items(levels_str)
                if len(levels) >= 2 and factor_name:
                    factors.append({"name": factor_name, "levels": levels})
            if len(factors) >= 2:
                crossed = self._cross_factors(factors)
                for combo_name in crossed:
                    conditions.append(ParsedCondition(
                        name=combo_name,
                        is_control=False,
                    ))
                return conditions, warnings

        # Pattern 2: Factorial "AxB" or "A x B" patterns (supports 2-5+ factors)
        factorial_match = re.search(r'(\d+)(?:\s*[x×X]\s*\d+)+', text_clean)
        factorial_dims: list = []
        if factorial_match:
            factorial_dims = re.findall(r'\d+', factorial_match.group())
        if factorial_match and len(factorial_dims) >= 2:
            dims = [int(d) for d in factorial_dims]
            # Extract factor levels from surrounding text
            factors = self._extract_factorial_factors(text_clean, factorial_match, dims=dims)
            if factors:
                crossed = self._cross_factors(factors)
                for combo_name in crossed:
                    conditions.append(ParsedCondition(
                        name=combo_name,
                        is_control=False,
                    ))
                return conditions, warnings

        # ── Pattern 2a+: Non-adjacent factorial with × between parenthetical groups
        # "3 (...) × 2 (...)" where × separates groups with text between digits
        nonadj_factorial = re.search(
            r'(\d+)\s*\([^)]+\)\s*[x×X]\s*(\d+)\s*\([^)]+\)',
            text_clean
        )
        if nonadj_factorial and not conditions:
            # Already handled by Pattern 1b above if labeled, but handle
            # unlabeled parenthetical: "3 (AI, Human, None) × 2 (Hedonic, Utilitarian)"
            unlabeled_paren = re.findall(
                r'(\d+)\s*\(\s*([^):]+)\s*\)',
                text_clean
            )
            if len(unlabeled_paren) >= 2:
                factors = []
                for count_str, levels_str in unlabeled_paren:
                    levels = self._parse_list_items(levels_str)
                    if len(levels) >= 2:
                        factors.append({"name": f"Factor_{len(factors)+1}", "levels": levels})
                if len(factors) >= 2:
                    crossed = self._cross_factors(factors)
                    for combo_name in crossed:
                        conditions.append(ParsedCondition(
                            name=combo_name,
                            is_control=False,
                        ))
                    return conditions, warnings

        # Pattern 2b: Parenthetical levels – "Trust (high, low) and Risk (high, low)"
        paren_factors = re.findall(
            r'(\w[\w\s]*?)\s*\(\s*([^)]+)\s*\)',
            text_clean, re.IGNORECASE
        )
        if len(paren_factors) >= 2:
            factors: List[Dict[str, Any]] = []
            for factor_name, levels_str in paren_factors:
                factor_name = factor_name.strip()
                # Skip noise words that got captured
                if factor_name.lower() in ('the', 'and', 'with', 'for', 'or', 'a', 'an'):
                    continue
                # If factor_name is just a digit, the real name may be inside the parens
                # e.g. "3 (Annotation: AI vs Human vs None)" — handled by Pattern 1b
                if re.match(r'^\d+$', factor_name):
                    # Check if levels_str has a colon (label:levels format)
                    if ':' in levels_str:
                        parts = levels_str.split(':', 1)
                        factor_name = parts[0].strip()
                        levels_str = parts[1].strip()
                    else:
                        continue  # Skip bare-digit factor names
                levels = self._parse_list_items(levels_str)
                if len(levels) >= 2 and len(factor_name) > 1:
                    factors.append({"name": factor_name, "levels": levels})
            if len(factors) >= 2:
                crossed = self._cross_factors(factors)
                for combo_name in crossed:
                    conditions.append(ParsedCondition(
                        name=combo_name,
                        is_control=False,
                    ))
                return conditions, warnings

        # Pattern 2c: Slash-separated factorial – "High/Low Trust × High/Low Risk"
        # Also handles multi-word levels: "Very High/Very Low Trust"
        slash_factors = re.findall(
            r'([\w]+(?:\s+\w+)?\s*/\s*[\w]+(?:\s+\w+)?(?:\s*/\s*[\w]+(?:\s+\w+)?)*)\s+([\w]+(?:\s+[\w]+)?)',
            text_clean
        )
        if len(slash_factors) >= 2:
            factors = []
            for levels_slash, factor_name in slash_factors:
                factor_name = factor_name.strip()
                levels = [l.strip() for l in levels_slash.split('/') if l.strip()]
                if len(levels) >= 2 and factor_name.lower() not in (
                    'x', 'and', 'by', 'the', 'with', 'for',
                ):
                    factors.append({"name": factor_name, "levels": levels})
            if len(factors) >= 2:
                crossed = self._cross_factors(factors)
                for combo_name in crossed:
                    conditions.append(ParsedCondition(
                        name=combo_name,
                        is_control=False,
                    ))
                return conditions, warnings

        # Pattern 2d: "vs" pattern – "Control vs Treatment" or "A vs B vs C"
        vs_match = re.split(r'\s+vs\.?\s+', text_clean, flags=re.IGNORECASE)
        if len(vs_match) >= 2:
            all_valid = all(
                m.strip() and len(m.strip()) < 100 for m in vs_match
            )
            if all_valid:
                for name in vs_match:
                    name = name.strip().rstrip('.')
                    if name:
                        conditions.append(ParsedCondition(
                            name=name,
                            is_control=self._is_control_condition(name),
                        ))
                return conditions, warnings

        # Pattern 3: Explicit list with commas, semicolons, or "and"
        # "control, treatment A, treatment B"
        list_patterns = [
            r'(?:conditions?|groups?)\s*(?:are|include|:|=)\s*(.+)',
            r'(?:between|across)\s+(?:the\s+)?(?:following\s+)?(?:conditions?|groups?)\s*[:\-]?\s*(.+)',
        ]
        for pattern in list_patterns:
            match = re.search(pattern, text_clean, re.IGNORECASE)
            if match:
                items_text = match.group(1)
                parsed = self._parse_list_items(items_text)
                if len(parsed) >= 2:
                    for name in parsed:
                        conditions.append(ParsedCondition(
                            name=name,
                            is_control=self._is_control_condition(name),
                        ))
                    return conditions, warnings

        # Pattern 4: Quoted items
        quoted = re.findall(r'["""\']([^"""\']+)["""\']', text_clean)
        if len(quoted) >= 2:
            for name in quoted:
                conditions.append(ParsedCondition(
                    name=name.strip(),
                    is_control=self._is_control_condition(name),
                ))
            return conditions, warnings

        # Pattern 5: Line-by-line conditions
        lines = [l.strip() for l in text_clean.split('\n') if l.strip()]
        if len(lines) >= 2:
            for line in lines:
                # Remove numbering like "1.", "1)", "- "
                clean = re.sub(r'^[\d]+[.)]\s*|^[-•]\s*', '', line).strip()
                if clean and len(clean) < 80:
                    conditions.append(ParsedCondition(
                        name=clean,
                        is_control=self._is_control_condition(clean),
                    ))
            if len(conditions) >= 2:
                return conditions, warnings
            conditions.clear()

        # Pattern 6: Simple comma/and separated
        parsed = self._parse_list_items(text_clean)
        if len(parsed) >= 2:
            for name in parsed:
                conditions.append(ParsedCondition(
                    name=name,
                    is_control=self._is_control_condition(name),
                ))
            return conditions, warnings

        # ── Warn if non-empty input produced 0 or 1 condition ──
        if not warnings:
            n = len(conditions)
            if n <= 1:
                warnings.append(
                    f"Only {n} condition detected from your input. "
                    f"At least 2 are needed for an experiment."
                )

        return conditions, warnings

    def parse_scales(self, text: str) -> List[ParsedScale]:
        """
        Extract scale/DV specifications from a natural language description.

        Handles formats like:
        - "7-point Likert scale measuring satisfaction (5 items)"
        - "Trust scale (1-7, 4 items), Purchase intention (1-7, 3 items)"
        - "We measure attitude on a 5-point scale and willingness to pay in dollars"
        - "Slider from 0 to 100 for perceived risk"
        """
        scales: List[ParsedScale] = []
        text_clean = text.strip()

        if not text_clean:
            return scales

        # Split by common delimiters for multiple scales
        segments = self._split_scale_segments(text_clean)

        for segment in segments:
            parsed_scale = self._parse_single_scale(segment)
            if parsed_scale:
                scales.append(parsed_scale)

        # If no structured parsing worked, try known scale name matching
        if not scales:
            scales = self._match_known_scales(text_clean)

        return scales

    def parse_open_ended(self, text: str) -> List[ParsedOpenEnded]:
        """
        Extract open-ended questions from a description.

        Handles formats like:
        - "We ask participants to explain their reasoning"
        - "Open-ended: Why did you choose this option?"
        - "Free text question about their experience"
        """
        questions: List[ParsedOpenEnded] = []
        text_clean = text.strip()

        if not text_clean:
            return questions

        # Split by lines or semicolons (NOT on "?" which cuts questions in half)
        segments = re.split(r'[\n;]', text_clean)

        for i, segment in enumerate(segments):
            segment = segment.strip()
            if not segment:
                continue

            # Extract the question text
            q_text = segment
            # Remove numbered prefixes: "1.", "2)", "a.", etc.
            q_text = re.sub(r'^\s*\d+[.)]\s*', '', q_text).strip()
            q_text = re.sub(r'^\s*[a-zA-Z][.)]\s*', '', q_text).strip()
            # Remove common textual prefixes
            q_text = re.sub(
                r'^(?:open[- ]?ended|free[- ]?text|qualitative|question)\s*[:\-]?\s*',
                '', q_text, flags=re.IGNORECASE
            ).strip()
            # Remove surrounding quotes
            if len(q_text) >= 2 and q_text[0] in ('"', '\u201c') and q_text[-1] in ('"', '\u201d'):
                q_text = q_text[1:-1].strip()

            if not q_text or len(q_text) < 5:
                continue

            # Generate variable name
            var_name = self._generate_var_name(q_text, prefix="OE", index=i + 1)

            # Detect context type
            context = self._detect_oe_context(q_text)

            questions.append(ParsedOpenEnded(
                variable_name=var_name,
                question_text=q_text,
                context_type=context,
            ))

        return questions

    def parse_design_type(self, text: str) -> str:
        """Detect whether the design is between, within, or mixed."""
        text_lower = text.lower()
        if "within" in text_lower and "between" in text_lower:
            return "mixed"
        if "within" in text_lower or "repeated" in text_lower:
            return "within"
        return "between"

    def parse_sample_size(self, text: str) -> int:
        """Extract sample size from a description."""
        # Pattern: "N = 200" or "200 participants" or "sample of 200"
        patterns = [
            r'[nN]\s*=\s*(\d+)',
            r'(\d+)\s*(?:participants?|subjects?|respondents?|people)',
            r'(?:sample|total)\s*(?:size|of|:)?\s*(?:of\s+)?(\d+)',
            r'(?:need|want|require|collect)\s+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                n = int(match.group(1))
                if 5 <= n <= 100000:
                    return n
        return 100  # Default

    def detect_factorial_structure(
        self, conditions: List[ParsedCondition]
    ) -> List[Dict[str, Any]]:
        """
        Detect factorial structure from condition names.
        E.g., ["High_Trust_High_Risk", "High_Trust_Low_Risk", ...] → 2 factors
        """
        if len(conditions) < 4:
            return []

        names = [c.name for c in conditions]
        # Try common separators
        for sep in ['_', ' x ', ' × ', ' - ', ' + ', ' & ']:
            parts = [n.split(sep) for n in names]
            part_counts = [len(p) for p in parts]
            if len(set(part_counts)) == 1 and part_counts[0] >= 2:
                n_factors = part_counts[0]
                factors = []
                for fi in range(n_factors):
                    levels = list(set(p[fi].strip() for p in parts))
                    if len(levels) >= 2:
                        factors.append({
                            "name": f"Factor_{fi + 1}",
                            "levels": sorted(levels),
                        })
                if len(factors) >= 2:
                    return factors
        return []

    def detect_research_domain(
        self,
        title: str,
        description: str,
        conditions_text: str = "",
        scales_text: str = "",
    ) -> str:
        """
        Detect research domain from study title, description, condition names,
        and scale names.

        Args:
            title: Study title.
            description: Study description text.
            conditions_text: Space-separated condition names for additional
                keyword matching (improves detection when title/description
                is sparse).
            scales_text: Space-separated scale/DV names for additional keyword
                matching.

        Returns:
            Best-matching research domain string.
        """
        combined = f"{title} {description} {conditions_text} {scales_text}".lower()

        domain_keywords = {
            "consumer behavior": ["consumer", "purchase", "buying", "shopping", "brand", "product", "advertising", "marketing"],
            "social psychology": ["social", "group", "conformity", "persuasion", "attitude", "stereotype", "prejudice", "discrimination"],
            "behavioral economics": ["economic", "decision", "framing", "loss aversion", "nudge", "incentive", "willingness to pay", "wtp"],
            "organizational behavior": ["organization", "workplace", "employee", "leadership", "team", "management", "job satisfaction"],
            "health psychology": ["health", "medical", "patient", "wellness", "illness", "treatment", "therapy", "medication", "symptom"],
            "political psychology": ["political", "voting", "election", "ideology", "partisan", "democrat", "republican", "policy"],
            "cognitive psychology": ["cognitive", "memory", "attention", "perception", "reasoning", "judgment", "heuristic", "bias"],
            "developmental psychology": ["child", "development", "adolescent", "aging", "lifespan", "parenting"],
            "environmental psychology": ["environment", "climate", "sustainability", "green", "recycling", "carbon", "pollution"],
            "educational psychology": ["education", "learning", "student", "teacher", "academic", "classroom", "instruction"],
            "moral psychology": ["moral", "ethical", "fairness", "justice", "virtue", "dilemma", "right", "wrong"],
            "technology & hci": ["technology", "ai", "artificial intelligence", "robot", "app", "digital", "online", "interface", "ux"],
            "communication": ["media", "news", "framing", "message", "communication", "misinformation", "fake news"],
            "food & nutrition": ["food", "eating", "diet", "nutrition", "meal", "restaurant", "taste", "organic"],
            "prosocial behavior": ["cooperation", "altruism", "helping", "charity", "donation", "volunteer", "prosocial"],
            "emotion research": ["emotion", "affect", "mood", "happiness", "anger", "sadness", "fear", "disgust", "anxiety"],
        }

        best_domain = "social psychology"  # Default
        best_score = 0

        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_domain = domain

        return best_domain

    def build_inferred_design(self, parsed: ParsedDesign) -> Dict[str, Any]:
        """
        Convert a ParsedDesign into the same inferred_design dict format
        used by the QSF upload pathway.
        """
        # Build conditions list
        conditions = [c.name for c in parsed.conditions]

        # Build scales list in engine format (with variable name deduplication)
        scales = []
        _used_var_names: set = set()
        for s in parsed.scales:
            var_name = s.variable_name or self._to_variable_name(s.name)
            # Deduplicate variable names to prevent column collisions
            base_var = var_name
            _suffix = 2
            while var_name in _used_var_names:
                var_name = f"{base_var}_{_suffix}"
                _suffix += 1
            _used_var_names.add(var_name)
            items = [f"{var_name}_{i+1}" for i in range(s.num_items)]
            scale_dict = {
                "name": s.name,
                "variable_name": var_name,
                "items": items,
                "num_items": s.num_items,
                "scale_min": s.scale_min,
                "scale_max": s.scale_max,
                "scale_points": s.scale_max - s.scale_min + 1 if s.scale_type != "numeric" else None,
                "type": s.scale_type,
                "reverse_items": s.reverse_items,
                "reliability": 0.85,
                "detected_from_qsf": False,
                "description": s.description,
            }
            if s.item_labels:
                scale_dict["item_names"] = s.item_labels
            scales.append(scale_dict)

        # Build open-ended list in engine format (with variable name deduplication)
        open_ended = []
        for oe in parsed.open_ended:
            oe_var = oe.variable_name
            # Deduplicate against scale variable names and other OE variables
            base_oe = oe_var
            _oe_suffix = 2
            while oe_var in _used_var_names:
                oe_var = f"{base_oe}_{_oe_suffix}"
                _oe_suffix += 1
            _used_var_names.add(oe_var)
            open_ended.append({
                "variable_name": oe_var,
                "name": oe.question_text[:50],
                "question_text": oe.question_text,
                "source_type": "conversational_builder",
                "force_response": True,
                "context_type": oe.context_type,
                "block_name": "",  # Required by engine visibility check
            })

        # Build factors
        factors = parsed.factors if parsed.factors else []

        # Compute default equal condition allocation
        n_conds = len(conditions)
        if n_conds >= 2:
            pct = round(100.0 / n_conds, 1)
            condition_allocation = {c: pct for c in conditions}
        else:
            condition_allocation = {}

        return {
            "conditions": conditions,
            "factors": factors,
            "scales": scales,
            "open_ended_questions": open_ended,
            "design_type": parsed.design_type,
            "sample_size": parsed.sample_size,
            "condition_allocation": condition_allocation,
            "randomization_level": "participant",
            "condition_visibility_map": {},
            "study_context": {
                "domain": parsed.research_domain,
                "study_domain": parsed.research_domain,  # Engine reads this key
                "title": parsed.study_title,
                "description": parsed.study_description,
                "source": "conversational_builder",
                "survey_name": parsed.study_title,
                "participant_characteristics": parsed.participant_characteristics,
                "persona_domains": BUILDER_DOMAIN_TO_PERSONA_DOMAIN.get(
                    parsed.research_domain.lower().strip(),
                    [parsed.research_domain.replace(" ", "_")],
                ),
            },
        }

    def generate_feedback(self, parsed: ParsedDesign) -> List[str]:
        """
        Generate smart, actionable suggestions based on the parsed design.

        Checks for common design issues and returns a list of human-readable
        suggestion strings that help researchers improve their study setup.

        Args:
            parsed: A fully populated ParsedDesign instance.

        Returns:
            List of suggestion strings (may be empty if design looks good).
        """
        suggestions: List[str] = []

        # ── Check for generic condition names ─────────────────────────────
        if parsed.conditions:
            generic_pattern = re.compile(
                r'^(?:condition|group|cond\.?)\s*\d+$', re.IGNORECASE
            )
            generic_count = sum(
                1 for c in parsed.conditions if generic_pattern.match(c.name.strip())
            )
            if generic_count == len(parsed.conditions) and generic_count >= 2:
                suggestions.append(
                    "Your conditions have generic names (e.g., 'Condition 1'). "
                    "Consider using descriptive names that reflect the manipulation "
                    "(e.g., 'High Trust', 'AI-Generated') for clearer simulation output."
                )

        # ── Check for single scale — suggest mediators / manipulation checks ─
        if len(parsed.scales) == 1:
            suggestions.append(
                f"You have only one measure ('{parsed.scales[0].name}'). "
                f"Consider adding a mediator variable to test underlying mechanisms, "
                f"or a manipulation check to verify your conditions worked as intended."
            )

        # ── Check for missing open-ended questions ────────────────────────
        if not parsed.open_ended:
            suggestions.append(
                "Your design has no open-ended questions. Adding at least one "
                "qualitative measure (e.g., 'Why did you make this choice?') "
                "can provide richer insights and help interpret quantitative patterns."
            )

        # ── Check per-cell sample size ────────────────────────────────────
        n_conditions = max(len(parsed.conditions), 1)
        per_cell = parsed.sample_size // n_conditions
        if per_cell < 20 and len(parsed.conditions) >= 2:
            suggestions.append(
                f"With {parsed.sample_size} participants across "
                f"{n_conditions} conditions (~{per_cell} per cell), "
                f"statistical power may be low. Consider increasing to at least "
                f"{n_conditions * 20} participants (20 per cell) for more "
                f"reliable effect detection."
            )

        # ── Check for mixed scale ranges ──────────────────────────────────
        if len(parsed.scales) >= 2:
            ranges = [
                (s.scale_max - s.scale_min) for s in parsed.scales
            ]
            if max(ranges) > 10 * min(ranges) and min(ranges) > 0:
                narrow = [
                    s.name for s in parsed.scales
                    if (s.scale_max - s.scale_min) == min(ranges)
                ]
                wide = [
                    s.name for s in parsed.scales
                    if (s.scale_max - s.scale_min) == max(ranges)
                ]
                suggestions.append(
                    f"Your scales have very different ranges "
                    f"(e.g., '{narrow[0]}' uses {min(ranges)+1} points vs "
                    f"'{wide[0]}' uses {max(ranges)+1} points). "
                    f"This is fine, but remember to standardize (z-score) "
                    f"before comparing effect sizes across measures."
                )

        # ── Warn about large factorial designs ────────────────────────────
        if len(parsed.conditions) > 8:
            suggestions.append(
                f"Your design has {len(parsed.conditions)} conditions. "
                f"With large factorial designs, ensure your sample size is "
                f"sufficient for each cell. A minimum of 20-30 per cell is "
                f"recommended, meaning N >= {len(parsed.conditions) * 25} "
                f"for adequate power."
            )

        return suggestions

    def suggest_additional_measures(
        self,
        domain: str,
        existing_scales: List[ParsedScale],
    ) -> List[Dict[str, str]]:
        """
        Suggest relevant scales based on the detected research domain that
        are not already included in the design.

        Args:
            domain: The detected research domain (e.g., 'consumer behavior').
            existing_scales: The scales already present in the design.

        Returns:
            Up to 3 dicts with keys 'name', 'description', and 'why'.
        """
        # Map domains to recommended scales with reasons
        domain_scale_suggestions: Dict[str, List[Dict[str, str]]] = {
            "consumer behavior": [
                {"name": "Purchase Intention", "description": "3-item, 1-7 Likert scale measuring likelihood to buy", "why": "Core consumer outcome; captures behavioral intent beyond attitudes."},
                {"name": "Brand Attitude", "description": "4-item, 1-7 Likert scale measuring overall brand evaluation", "why": "Mediates between ad exposure and purchase; useful for process analysis."},
                {"name": "Perceived Quality", "description": "4-item, 1-7 Likert scale measuring product quality perception", "why": "Helps distinguish between affective and cognitive evaluation routes."},
                {"name": "Net Promoter Score (NPS)", "description": "1-item, 0-10 numeric scale measuring recommendation likelihood", "why": "Industry-standard metric; easy to compare with real-world benchmarks."},
            ],
            "social psychology": [
                {"name": "Social Distance", "description": "3-item, 1-7 Likert scale measuring psychological closeness", "why": "Classic social psych measure; captures subtle intergroup attitudes."},
                {"name": "Perceived Warmth", "description": "3-item, 1-7 Likert scale (Stereotype Content Model)", "why": "One of two fundamental dimensions of social perception."},
                {"name": "Perceived Competence", "description": "3-item, 1-7 Likert scale (Stereotype Content Model)", "why": "Complements warmth; together they predict distinct behavioral outcomes."},
                {"name": "Empathy Scale", "description": "5-item, 1-7 Likert scale measuring empathic concern", "why": "Key mediator in prosocial behavior and intergroup relations."},
            ],
            "behavioral economics": [
                {"name": "Risk Perception", "description": "4-item, 1-7 Likert scale measuring perceived risk", "why": "Central to decision-making under uncertainty; common mediator."},
                {"name": "Need for Cognition (NFC)", "description": "18-item, 1-5 Likert scale measuring cognitive motivation", "why": "Important moderator of framing effects and cognitive biases."},
                {"name": "Willingness to Pay", "description": "Numeric input, 0-100 dollars", "why": "Direct behavioral measure with high ecological validity."},
                {"name": "Behavioral Intention", "description": "3-item, 1-7 Likert scale measuring intent to act", "why": "Bridges the attitude-behavior gap in economic decision models."},
            ],
            "organizational behavior": [
                {"name": "Job Satisfaction", "description": "5-item, 1-7 Likert scale", "why": "Foundational OB outcome; strongly tied to turnover and performance."},
                {"name": "Trust Scale", "description": "5-item, 1-7 Likert scale measuring interpersonal/organizational trust", "why": "Key mediator in leadership and team dynamics research."},
                {"name": "Motivation Scale", "description": "4-item, 1-7 Likert scale measuring intrinsic/extrinsic motivation", "why": "Helps distinguish between motivational mechanisms in interventions."},
            ],
            "health psychology": [
                {"name": "GAD-7", "description": "7-item, 0-3 anxiety screening measure", "why": "Validated clinical instrument; enables comparison with clinical populations."},
                {"name": "PHQ-9", "description": "9-item, 0-3 depression screening measure", "why": "Gold-standard brief depression measure; excellent psychometric properties."},
                {"name": "Self-Efficacy", "description": "5-item, 1-7 Likert scale measuring perceived capability", "why": "Strong predictor of health behavior change across domains."},
                {"name": "Perceived Stress Scale", "description": "10-item, 1-5 scale measuring stress appraisal", "why": "Most widely used stress measure; great for pre-post designs."},
            ],
            "political psychology": [
                {"name": "Social Dominance Orientation (SDO)", "description": "16-item, 1-7 Likert scale", "why": "Predicts attitudes toward social hierarchy and intergroup relations."},
                {"name": "Moral Foundations Questionnaire", "description": "20-item, 1-6 scale measuring moral values", "why": "Explains political ideology differences through moral reasoning."},
                {"name": "Right-Wing Authoritarianism (RWA)", "description": "15-item, 1-7 Likert scale", "why": "Complements SDO; together they explain most variance in prejudice."},
            ],
            "cognitive psychology": [
                {"name": "Need for Cognition (NFC)", "description": "18-item, 1-5 Likert scale", "why": "Measures individual differences in cognitive engagement."},
                {"name": "NASA-TLX", "description": "6-item, 0-100 slider measuring cognitive workload", "why": "Standard measure of task difficulty and mental effort."},
                {"name": "Self-Efficacy", "description": "5-item, 1-7 Likert scale", "why": "Captures confidence in cognitive abilities; moderates performance."},
            ],
            "technology & hci": [
                {"name": "Technology Acceptance Model (TAM)", "description": "12-item, 1-7 Likert scale", "why": "Standard measure for technology adoption research."},
                {"name": "System Usability Scale (SUS)", "description": "10-item, 1-5 Likert scale", "why": "Industry-standard usability benchmark; easy to interpret."},
                {"name": "NASA-TLX", "description": "6-item, 0-100 slider measuring cognitive workload", "why": "Captures task load in human-computer interaction studies."},
                {"name": "Privacy Concern", "description": "5-item, 1-7 Likert scale", "why": "Critical in studies involving data collection or AI systems."},
            ],
            "moral psychology": [
                {"name": "Moral Foundations Questionnaire", "description": "20-item, 1-6 scale", "why": "Maps the five moral foundations; central to the field."},
                {"name": "Moral Judgment", "description": "4-item, 1-7 Likert scale", "why": "Measures moral evaluation of specific scenarios or agents."},
                {"name": "Empathy Scale", "description": "5-item, 1-7 Likert measuring empathic concern", "why": "Key mediator between moral reasoning and prosocial action."},
            ],
            "communication": [
                {"name": "Credibility Scale", "description": "4-item, 1-7 Likert scale measuring source credibility", "why": "Central to persuasion and media effects research."},
                {"name": "Ad Effectiveness", "description": "4-item, 1-7 Likert scale", "why": "Measures message impact on attitudes and behavioral intent."},
                {"name": "Trust Scale", "description": "5-item, 1-7 Likert measuring media/source trust", "why": "Key outcome in misinformation and news credibility studies."},
            ],
            "emotion research": [
                {"name": "PANAS", "description": "20-item, 1-5 Positive and Negative Affect Schedule", "why": "Gold-standard measure of affective states; captures both valence dimensions."},
                {"name": "Emotional Response", "description": "6-item, 1-7 Likert scale", "why": "Brief measure suitable for within-subjects emotion inductions."},
                {"name": "Anxiety (GAD-7)", "description": "7-item, 0-3 scale", "why": "Captures anxiety as trait or state; useful for emotion regulation studies."},
            ],
            "environmental psychology": [
                {"name": "Risk Perception", "description": "4-item, 1-7 Likert scale", "why": "Central to climate risk and environmental hazard research."},
                {"name": "Behavioral Intention", "description": "3-item, 1-7 Likert scale", "why": "Predicts pro-environmental action; bridges attitude-behavior gap."},
                {"name": "Moral Judgment", "description": "4-item, 1-7 Likert scale", "why": "Captures moral dimensions of environmental decision-making."},
            ],
            "food & nutrition": [
                {"name": "Perceived Quality", "description": "4-item, 1-7 Likert scale", "why": "Measures food quality perceptions across conditions."},
                {"name": "Purchase Intention", "description": "3-item, 1-7 Likert scale", "why": "Captures consumer intent for food products."},
                {"name": "Willingness to Pay", "description": "Numeric input, 0-100 dollars", "why": "Direct behavioral measure; highly relevant for food pricing studies."},
            ],
            "prosocial behavior": [
                {"name": "Empathy Scale", "description": "5-item, 1-7 Likert scale", "why": "Primary motivational mechanism for prosocial behavior."},
                {"name": "Interpersonal Reactivity Index (IRI)", "description": "28-item, 1-5 scale", "why": "Multi-dimensional empathy measure; captures cognitive and affective components."},
                {"name": "Moral Judgment", "description": "4-item, 1-7 Likert scale", "why": "Links moral reasoning to helping and cooperation."},
            ],
            "educational psychology": [
                {"name": "Self-Efficacy", "description": "5-item, 1-7 Likert scale", "why": "Strongest predictor of academic performance and persistence."},
                {"name": "Motivation Scale", "description": "4-item, 1-7 Likert scale", "why": "Captures intrinsic vs extrinsic motivation for learning."},
                {"name": "Satisfaction Scale", "description": "5-item, 1-7 Likert measuring learning satisfaction", "why": "Common outcome measure in educational intervention studies."},
            ],
            "developmental psychology": [
                {"name": "Self-Esteem (RSE)", "description": "10-item, 1-4 Rosenberg Self-Esteem Scale", "why": "Most widely used self-esteem measure across age groups."},
                {"name": "Well-Being Scale", "description": "5-item, 1-7 Likert scale", "why": "Captures subjective well-being across developmental stages."},
                {"name": "Life Satisfaction (SWLS)", "description": "5-item, 1-7 Satisfaction with Life Scale", "why": "Validated for adolescents and adults; strong test-retest reliability."},
            ],
        }

        existing_names_lower = {s.name.lower().strip() for s in existing_scales}
        # Also normalize against known labels to catch abbreviation matches
        existing_labels_lower: set = set()
        for s in existing_scales:
            existing_labels_lower.add(s.name.lower().strip())
            if s.description:
                existing_labels_lower.add(s.description.lower().strip())

        domain_lower = domain.lower().strip()
        candidates = domain_scale_suggestions.get(domain_lower, [])

        # Fallback: if exact match fails, try partial match on domain key
        if not candidates:
            for key, val in domain_scale_suggestions.items():
                if key in domain_lower or domain_lower in key:
                    candidates = val
                    break

        # Filter out already-included scales
        suggestions: List[Dict[str, str]] = []
        for candidate in candidates:
            candidate_name_lower = candidate["name"].lower().strip()
            # Check if this scale (or something very similar) is already present
            already_present = False
            for existing in existing_names_lower | existing_labels_lower:
                # Check substring overlap in both directions
                if (candidate_name_lower in existing
                        or existing in candidate_name_lower):
                    already_present = True
                    break
            if not already_present:
                suggestions.append(candidate)
            if len(suggestions) >= 3:
                break

        return suggestions

    def validate_full_design(self, parsed: ParsedDesign) -> Dict[str, List[str]]:
        """
        Validate a complete parsed design and return errors and warnings.

        Checks:
        - At least 2 conditions
        - At least 1 scale
        - Sample size >= 10
        - Scale ranges are valid (min < max)
        - No duplicate condition names
        - No duplicate scale names
        - Total items across all scales not exceeding 100

        Returns:
            Dict with 'errors' (fatal) and 'warnings' (non-fatal) lists of strings.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # ── Condition checks ─────────────────────────────────────────────
        if len(parsed.conditions) < 2:
            errors.append(
                f"At least 2 conditions are required, but only "
                f"{len(parsed.conditions)} provided."
            )

        # Duplicate condition names (case-insensitive)
        cond_names_lower = [c.name.lower().strip() for c in parsed.conditions]
        seen_conds: set = set()
        for cn in cond_names_lower:
            if cn in seen_conds:
                errors.append(f"Duplicate condition name: '{cn}'.")
            seen_conds.add(cn)

        # Warn if no control condition detected
        if parsed.conditions and not any(c.is_control for c in parsed.conditions):
            warnings.append(
                "No control/baseline condition detected. Consider adding one "
                "for clearer experimental comparisons."
            )

        # Warn about very long condition names (e.g., from large factorials)
        for c in parsed.conditions:
            if len(c.name) > 100:
                warnings.append(
                    f"Condition name '{c.name[:50]}...' is very long ({len(c.name)} chars). "
                    f"This may cause display issues in reports and CSV headers."
                )
                break  # Only warn once

        # ── Scale checks ─────────────────────────────────────────────────
        if len(parsed.scales) < 1:
            errors.append(
                "At least 1 dependent variable (scale) is required."
            )

        # Duplicate scale names (case-insensitive)
        scale_names_lower = [s.name.lower().strip() for s in parsed.scales]
        seen_scales: set = set()
        for sn in scale_names_lower:
            if sn in seen_scales:
                errors.append(f"Duplicate scale name: '{sn}'.")
            seen_scales.add(sn)

        # Scale range validity
        for s in parsed.scales:
            if s.scale_min >= s.scale_max:
                errors.append(
                    f"Scale '{s.name}' has invalid range: "
                    f"min ({s.scale_min}) must be less than max ({s.scale_max})."
                )
            if s.num_items < 1:
                errors.append(
                    f"Scale '{s.name}' must have at least 1 item "
                    f"(got {s.num_items})."
                )
            # Warn about very large scales
            if s.num_items > 30:
                warnings.append(
                    f"Scale '{s.name}' has {s.num_items} items, which is "
                    f"unusually high. Verify this is correct."
                )
            # Warn about reverse-coded items outside item range
            for ri in s.reverse_items:
                if ri < 1 or ri > s.num_items:
                    warnings.append(
                        f"Scale '{s.name}': reverse-coded item {ri} is "
                        f"outside the item range (1-{s.num_items})."
                    )

        # ── Sample size check ────────────────────────────────────────────
        if parsed.sample_size < 10:
            errors.append(
                f"Sample size must be at least 10, but got "
                f"{parsed.sample_size}."
            )
        elif parsed.sample_size < 30:
            warnings.append(
                f"Sample size of {parsed.sample_size} is very small. "
                f"Consider at least 30 per condition for reliable results."
            )

        # ── Per-cell sample check ────────────────────────────────────────
        n_conditions = max(len(parsed.conditions), 1)
        per_cell = parsed.sample_size // n_conditions
        if per_cell < 5 and len(parsed.conditions) >= 2:
            warnings.append(
                f"With {parsed.sample_size} total participants across "
                f"{n_conditions} conditions, each cell gets ~{per_cell} "
                f"participants. Consider increasing sample size."
            )
        elif per_cell < 20 and n_conditions >= 6:
            warnings.append(
                f"Your factorial design has {n_conditions} cells with only "
                f"~{per_cell} participants per cell. For medium effect sizes "
                f"(d=0.5), consider at least 20 per cell (N={n_conditions * 20})."
            )

        # ── Factorial balance check ───────────────────────────────────────
        if parsed.factors and len(parsed.factors) >= 2:
            expected_cells = 1
            for f in parsed.factors:
                expected_cells *= len(f.get("levels", [])) if isinstance(f, dict) else 1
            if expected_cells != n_conditions and expected_cells > 1:
                warnings.append(
                    f"Factorial design expects {expected_cells} crossed conditions "
                    f"but {n_conditions} were found. The design may be incomplete."
                )

        # ── Total survey length check ────────────────────────────────────
        total_items = sum(s.num_items for s in parsed.scales)
        if total_items > 100:
            warnings.append(
                f"Your survey contains {total_items} total items across "
                f"{len(parsed.scales)} scales, which may be too long. "
                f"Surveys exceeding 100 items often suffer from participant "
                f"fatigue, lower completion rates, and reduced data quality. "
                f"Consider shortening scales or using brief validated "
                f"alternatives where possible."
            )

        # ── Variable name validity ────────────────────────────────────────
        seen_vars: set = set()
        for s in parsed.scales:
            vn = s.variable_name
            if vn and vn in seen_vars:
                warnings.append(
                    f"Duplicate variable name '{vn}' across scales. "
                    f"These will be auto-suffixed to avoid collisions."
                )
            if vn:
                seen_vars.add(vn)
        for oe in parsed.open_ended:
            vn = oe.variable_name
            if vn and vn in seen_vars:
                warnings.append(
                    f"Open-ended variable '{vn}' conflicts with a scale variable. "
                    f"Consider renaming for clarity."
                )
            if vn:
                seen_vars.add(vn)

        return {"errors": errors, "warnings": warnings}

    @staticmethod
    def generate_example_descriptions() -> List[Dict[str, str]]:
        """
        Return 3-5 example study descriptions that users can click to
        auto-fill the survey builder for inspiration and guidance.

        Each example includes:
        - title: Short study title
        - conditions: Natural-language condition description
        - scales: Natural-language scale/DV description
        - open_ended: Natural-language open-ended question description
        - domain: Research domain
        """
        return [
            {
                "title": "AI vs Human Content Credibility",
                "description": (
                    "This study examines how disclosure of AI authorship affects "
                    "perceived credibility of news articles. Participants read an article "
                    "and are told it was written by AI, a human journalist, or given no "
                    "authorship information. We measure credibility, trust, and willingness "
                    "to share the article on social media."
                ),
                "conditions": (
                    "Condition 1: AI-generated article, "
                    "Condition 2: Human-written article, "
                    "Condition 3: No authorship information"
                ),
                "scales": (
                    "Credibility scale (4 items, 1-7); "
                    "Trust scale (5 items, 1-7); "
                    "Willingness to share (slider 0-100)"
                ),
                "open_ended": (
                    "Why did you rate the article's credibility the way you did?"
                ),
                "domain": "technology & hci",
            },
            {
                "title": "Moral Framing and Donation Behavior",
                "description": (
                    "We investigate how moral framing (care vs fairness) and trust "
                    "in the charity (high vs low) affect donation amounts. Participants "
                    "read a charity appeal framed around either caring for others or "
                    "fairness/justice, from an organization with either high or low "
                    "trustworthiness ratings."
                ),
                "conditions": (
                    "Trust (high, low) and Moral Frame (care, fairness)"
                ),
                "scales": (
                    "Moral Foundations Questionnaire (MFQ); "
                    "Donation amount in dollars (WTP, 0-100); "
                    "Empathy scale (5 items, 1-7)"
                ),
                "open_ended": (
                    "Describe your reasoning when deciding how much to donate."
                ),
                "domain": "moral psychology",
            },
            {
                "title": "Social Media and Well-Being",
                "description": (
                    "A field experiment examining the causal effects of social media "
                    "usage reduction on psychological well-being. Participants are "
                    "randomly assigned to continue normal usage, reduce usage to 30 "
                    "minutes per day, or abstain completely for two weeks."
                ),
                "conditions": "Control vs Reduced Usage vs Complete Abstinence",
                "scales": (
                    "SWLS (Satisfaction with Life Scale); "
                    "PANAS; "
                    "PHQ-9; "
                    "GAD-7"
                ),
                "open_ended": (
                    "How did the social media intervention affect your "
                    "daily routine?"
                ),
                "domain": "health psychology",
            },
            {
                "title": "Brand Authenticity in Green Marketing",
                "description": (
                    "This study tests whether brand origin (local vs global) and the "
                    "presence of green environmental claims interact to influence "
                    "consumer perceptions and purchase intentions. We hypothesize that "
                    "local brands benefit more from green claims than global brands."
                ),
                "conditions": (
                    "Brand Origin (local, global) and "
                    "Green Claim (present, absent)"
                ),
                "scales": (
                    "Brand attitude (4 items, 1-7); "
                    "Purchase intention (3 items, 1-7); "
                    "Perceived quality (4 items, 1-7, "
                    "1=strongly disagree, 7=strongly agree)"
                ),
                "open_ended": (
                    "What factors influenced your evaluation of this brand?"
                ),
                "domain": "consumer behavior",
            },
            {
                "title": "Personality and Risk-Taking Under Uncertainty",
                "description": (
                    "Investigating how personality traits moderate risk-taking behavior "
                    "under varying levels of uncertainty. Participants complete personality "
                    "measures and then face financial decision scenarios with different "
                    "levels of outcome uncertainty."
                ),
                "conditions": (
                    "High uncertainty vs Low uncertainty vs Ambiguous"
                ),
                "scales": (
                    "BFI-10; "
                    "Risk perception (4 items, 1-7); "
                    "Need for Cognition (NFC); "
                    "Behavioral intention (3 items, 1-7, "
                    "items 2 and 3 are reverse-coded)"
                ),
                "open_ended": (
                    "Explain why you chose the option you did in "
                    "the decision task."
                ),
                "domain": "behavioral economics",
            },
            {
                "title": "Product Annotation and Consumer Perception",
                "description": (
                    "A 3x2 between-subjects experiment examining how product "
                    "description source (AI-generated, human-curated, or no source info) "
                    "and product type (hedonic vs utilitarian) affect perceived quality, "
                    "purchase intention, and ad credibility."
                ),
                "conditions": (
                    "3 (Annotation: AI-generated vs Human-curated vs No source) "
                    "× 2 (Product type: Hedonic vs Utilitarian), "
                    "between-subjects, random assignment"
                ),
                "scales": (
                    "Perceived Quality (PQ): 3 items (7-point Likert; "
                    "1=extremely low quality, 7=extremely high quality)\n\n"
                    "Purchase Intention (PI): 3 items (7-point Likert; "
                    "1=strongly disagree, 7=strongly agree)\n\n"
                    "Ad Credibility (AC): 3 items (7-point Likert; "
                    "1=not at all credible, 7=extremely credible)"
                ),
                "open_ended": (
                    "What influenced your perception of the product?\n"
                    "Describe your overall impression of the product."
                ),
                "domain": "consumer behavior",
            },
            {
                "title": "Framing, Source, and Time Pressure",
                "description": (
                    "A 2x2x2 factorial experiment testing how message framing "
                    "(gain vs loss), information source (expert vs peer), and time "
                    "pressure (immediate vs delayed response) interact to influence "
                    "risk perception and decision confidence."
                ),
                "conditions": (
                    "2 (Frame: Gain vs Loss) "
                    "× 2 (Source: Expert vs Peer) "
                    "× 2 (Time: Immediate vs Delayed)"
                ),
                "scales": (
                    "Risk perception (4 items, 1-7); "
                    "Decision confidence (3 items, 1-7); "
                    "Willingness to act (slider 0-100)"
                ),
                "open_ended": (
                    "Explain why you made the decision you did."
                ),
                "domain": "behavioral economics",
            },
        ]

    def generate_smart_description(
        self,
        title: str,
        conditions: List[ParsedCondition],
        scales: List[ParsedScale],
        design_type: str = "between",
    ) -> str:
        """
        Auto-generate a rich study description from title, conditions, and scales.
        Helps users who leave the description field empty.
        """
        parts = []

        # Design type intro
        n_conds = len(conditions)
        cond_names = [c.name for c in conditions]
        is_factorial = any(" × " in cn for cn in cond_names)

        if is_factorial:
            n_factors = len(cond_names[0].split(" × ")) if cond_names else 0
            parts.append(
                f"This is a {n_factors}-factor {design_type}-subjects experiment "
                f"with {n_conds} conditions."
            )
        elif n_conds >= 2:
            parts.append(
                f"This is a {design_type}-subjects experiment comparing "
                f"{n_conds} conditions: {', '.join(cond_names[:6])}."
            )

        # Scales summary
        if scales:
            scale_names = [s.name for s in scales]
            parts.append(
                f"The dependent variables include {', '.join(scale_names[:5])}"
                + (f" and {len(scale_names) - 5} more" if len(scale_names) > 5 else "")
                + "."
            )

        # Domain-relevant filler based on title keywords
        title_lower = title.lower()
        if any(w in title_lower for w in ["ai", "artificial", "algorithm"]):
            parts.append("The study investigates perceptions of AI-generated content.")
        elif any(w in title_lower for w in ["brand", "product", "consumer"]):
            parts.append("The study examines consumer attitudes and purchase behavior.")
        elif any(w in title_lower for w in ["moral", "ethical", "dilemma"]):
            parts.append("The study explores moral reasoning and ethical judgments.")
        elif any(w in title_lower for w in ["health", "well-being", "wellness"]):
            parts.append("The study examines health-related attitudes and behaviors.")
        elif any(w in title_lower for w in ["social", "group", "identity"]):
            parts.append("The study investigates social psychological processes.")

        return " ".join(parts) if parts else f"A {design_type}-subjects experiment studying {title}."

    @staticmethod
    def get_persona_domain_keys(builder_domain: str) -> List[str]:
        """
        Map a builder display domain to persona library internal domain keys.

        Args:
            builder_domain: The human-readable domain name from the builder UI
                (e.g., 'consumer behavior').

        Returns:
            List of persona library domain keys (e.g., ['consumer_behavior', 'marketing']).
        """
        return BUILDER_DOMAIN_TO_PERSONA_DOMAIN.get(
            builder_domain.lower().strip(),
            [builder_domain.lower().strip().replace(" ", "_")],
        )

    # ─── Private helper methods ────────────────────────────────────────────────

    def _is_control_condition(self, name: str) -> bool:
        """Check if a condition name indicates a control group."""
        control_words = ["control", "baseline", "placebo", "no treatment", "no intervention", "neutral"]
        name_lower = name.lower().strip()
        return any(w in name_lower for w in control_words)

    def _extract_factorial_factors(
        self, text: str, factorial_match: re.Match,
        dims: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract factor names and levels from text surrounding a factorial pattern.

        Args:
            text: The full input text
            factorial_match: The regex match object for the factorial pattern
            dims: Pre-extracted dimension list (e.g., [2, 3, 2] for 2x3x2).
                  If None, falls back to extracting from match groups.
        """
        factors: List[Dict[str, Any]] = []
        if dims is None:
            # Fallback: extract all digits from the matched factorial pattern
            dims = [int(d) for d in re.findall(r'\d+', factorial_match.group())]
            if len(dims) < 2:
                return []

        noise_words = {
            'the', 'and', 'with', 'for', 'or', 'a', 'an', 'in', 'on',
            'by', 'to', 'of', 'x', 'design', 'between', 'subjects',
        }

        # Try to find factor descriptions near the factorial notation
        # Pattern: "Factor (level1, level2)" — strict parenthetical
        text_after = text[factorial_match.end():]
        text_before = text[:factorial_match.start()]
        search_text = f"{text_before} {text_after}"

        # Use strict pattern: word(s) followed by parenthetical levels
        paren_matches = re.findall(
            r'([A-Z][\w\s]*?)\s*\(\s*([^)]+)\s*\)',
            search_text
        )
        for name, levels_str in paren_matches:
            name = name.strip().rstrip(':')
            # Filter noise words and overly long captures
            if (len(name) > 1 and len(name) < 40
                    and name.lower() not in noise_words):
                levels = self._parse_list_items(levels_str)
                if len(levels) >= 2:
                    factors.append({"name": name, "levels": levels})

        # Only keep factors matching the expected number of dimensions
        if len(factors) > len(dims):
            factors = factors[:len(dims)]

        # Verify level counts match expected dimensions
        if len(factors) == len(dims):
            valid = all(len(f["levels"]) == d for f, d in zip(factors, dims))
            if not valid:
                # Log warning about level count mismatch
                warnings = getattr(self, '_warnings', [])
                warnings.append(f"Factor level counts don't match expected {dims} dimensions - some conditions may be missing")
                factors = factors[:len(dims)]

        # If we couldn't extract names, use generic Factor_1, Factor_2
        if len(factors) < len(dims):
            factors = []
            for i, dim in enumerate(dims):
                factors.append({
                    "name": f"Factor_{i+1}",
                    "levels": [f"Level_{j+1}" for j in range(dim)],
                })

        return factors

    def _cross_factors(self, factors: List[Dict[str, Any]]) -> List[str]:
        """Generate crossed condition names from factors with factor name prefixes.

        Uses ' × ' as separator for crossed conditions to make them readable
        and recognizable by the simulation engine's factorial detection.
        """
        if not factors:
            return []

        sep = " × "

        result = []
        for level in factors[0]["levels"]:
            result.append(f"{level}")

        for factor in factors[1:]:
            new_result = []
            for existing in result:
                for level in factor["levels"]:
                    new_result.append(f"{existing}{sep}{level}")
            result = new_result

        # If crossed names are ambiguous (e.g., "High × High"), prefix with factor names
        if len(factors) >= 2:
            level_sets = [set(lv.lower() for lv in f["levels"]) for f in factors]
            has_overlap = False
            for i in range(len(level_sets)):
                for j in range(i + 1, len(level_sets)):
                    if level_sets[i] & level_sets[j]:
                        has_overlap = True
                        break

            if has_overlap:
                # Regenerate with factor name prefixes
                result = []
                for level in factors[0]["levels"]:
                    fname = factors[0]["name"]
                    result.append(f"{fname}: {level}")
                for factor in factors[1:]:
                    new_result = []
                    fname = factor["name"]
                    for existing in result:
                        for level in factor["levels"]:
                            new_result.append(f"{existing}{sep}{fname}: {level}")
                    result = new_result

        return result

    def _parse_list_items(self, text: str) -> List[str]:
        """Parse a comma/and/semicolon-separated list of items.

        Handles commas inside parentheses by not splitting on them.
        """
        # Normalize separators
        text = re.sub(r'\s*[;]\s*', ', ', text)
        text = re.sub(r',?\s+and\s+', ', ', text, flags=re.IGNORECASE)
        text = re.sub(r',?\s+or\s+', ', ', text, flags=re.IGNORECASE)
        text = re.sub(r',?\s+vs\.?\s+', ', ', text, flags=re.IGNORECASE)

        # Split on commas, but not commas inside parentheses
        items: List[str] = []
        depth = 0
        current = ""
        for ch in text:
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth = max(0, depth - 1)
                current += ch
            elif ch == ',' and depth == 0:
                items.append(current.strip().rstrip('.'))
                current = ""
            else:
                current += ch
        if current.strip():
            items.append(current.strip().rstrip('.'))

        items = [item for item in items if item and len(item) < 100]

        # Strip leading articles and trailing "group" from each item
        cleaned: List[str] = []
        for item in items:
            # Remove leading articles
            item = re.sub(r'^(?:the|a|an)\s+', '', item, flags=re.IGNORECASE).strip()
            # Remove trailing "group" (e.g., "control group" → "control")
            item = re.sub(r'\s+group$', '', item, flags=re.IGNORECASE).strip()
            if item:
                cleaned.append(item)
        return cleaned

    def _split_scale_segments(self, text: str) -> List[str]:
        """Split text into segments, each describing one scale.

        Splitting priority:
        1. Paragraph breaks (double newlines) — most reliable for multi-line input
        2. Numbered items ("1.", "2.") — common structured format
        3. Bullet points ("- ", "• ") — common list format
        4. Lines starting with "Name:" pattern — colon-prefixed scale specs
        5. Semicolons (only if no parentheses contain semicolons)
        6. Individual non-empty lines
        """
        # ── 1. Paragraph breaks (double newlines) ────────────────────────
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) >= 2:
            # Merge continuation lines within each paragraph
            merged = []
            for p in paragraphs:
                p = p.strip()
                if p:
                    # Skip section headers like "Manipulation checks:" or "Covariates (optional):"
                    # that don't contain item/scale specs
                    if re.match(r'^[\w\s/]+:\s*$', p):
                        continue
                    merged.append(p)
            if len(merged) >= 2:
                return merged

        # ── 2. Numbered items "1.", "2." ──────────────────────────────────
        numbered = re.split(r'\n\s*\d+[.)]\s*', text)
        if len(numbered) >= 2:
            return [s.strip() for s in numbered if s.strip()]

        # ── 3. Bullet points ──────────────────────────────────────────────
        bulleted = re.split(r'\n\s*[-•]\s*', text)
        if len(bulleted) >= 2:
            return [s.strip() for s in bulleted if s.strip()]

        # ── 4. Lines starting with "Name:" pattern ────────────────────────
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if len(lines) >= 2:
            # Check if most lines look like "Scale Name: spec"
            colon_lines = [l for l in lines if re.match(r'^[A-Z][\w\s/()-]+:', l)]
            if len(colon_lines) >= 2:
                # Re-split: each "Name:" line starts a new segment, continuation
                # lines are appended to the previous segment
                segments: List[str] = []
                for l in lines:
                    if re.match(r'^[A-Z][\w\s/()-]+:', l):
                        segments.append(l)
                    elif segments:
                        segments[-1] += ' ' + l
                    else:
                        segments.append(l)
                return [s.strip() for s in segments if s.strip()]

        # ── 5. Semicolons (only at top level, not inside parentheses) ─────
        # Count semicolons outside parentheses
        _depth = 0
        _top_level_semis = 0
        for ch in text:
            if ch == '(':
                _depth += 1
            elif ch == ')':
                _depth = max(0, _depth - 1)
            elif ch == ';' and _depth == 0:
                _top_level_semis += 1
        if _top_level_semis >= 1:
            # Split only on top-level semicolons
            parts: List[str] = []
            current = []
            _depth = 0
            for ch in text:
                if ch == '(':
                    _depth += 1
                    current.append(ch)
                elif ch == ')':
                    _depth = max(0, _depth - 1)
                    current.append(ch)
                elif ch == ';' and _depth == 0:
                    parts.append(''.join(current).strip())
                    current = []
                else:
                    current.append(ch)
            if current:
                parts.append(''.join(current).strip())
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                return parts

        # ── 6. Individual non-empty lines ─────────────────────────────────
        if len(lines) >= 2:
            return lines

        # Single segment
        return [text]

    def _parse_single_scale(self, text: str) -> Optional[ParsedScale]:
        """
        Parse a single scale description into a ParsedScale.

        Supports:
        - Scale abbreviations (BFI-10, PANAS, GAD-7, PHQ-9, etc.)
        - Reverse-coded items ("items 3 and 5 are reverse-coded")
        - Scale anchors ("1=strongly disagree, 7=strongly agree")
        - Standard numeric specs (N-point, N items, range)
        """
        if not text or len(text) < 3:
            return None

        text_lower = text.lower()
        name = ""
        num_items = 1
        scale_min = 1
        scale_max = 7
        scale_type = "likert"
        reverse_items: List[int] = []
        item_labels: List[str] = []

        # ── Check abbreviation lookup first ──────────────────────────────
        abbrev_matched = False
        for abbrev, canonical_key in SCALE_ABBREVIATIONS.items():
            # Match the abbreviation as a whole token (case-insensitive)
            if re.search(r'(?<![a-zA-Z])' + re.escape(abbrev) + r'(?![a-zA-Z])', text_lower):
                if canonical_key in KNOWN_SCALES:
                    spec = KNOWN_SCALES[canonical_key]
                    name = spec["label"]
                    num_items = spec["items"]
                    scale_min = spec["min"]
                    scale_max = spec["max"]
                    abbrev_matched = True
                    break

        # ── Extract scale anchors: "1=strongly disagree, 7=strongly agree" ──
        anchor_matches = re.findall(
            r'(\d+)\s*=\s*([^,;]+?)(?:\s*[,;]|\s*$)',
            text
        )
        if anchor_matches:
            for val_str, label in anchor_matches:
                val = int(val_str)
                item_labels.append(f"{val}={label.strip()}")
            # Use anchor endpoints to infer scale range
            anchor_vals = [int(m[0]) for m in anchor_matches]
            if len(anchor_vals) >= 2:
                scale_min = min(anchor_vals)
                scale_max = max(anchor_vals)

        # ── Extract reverse-coded / reverse-scored items ──────────────────
        rev_match = re.search(
            r'(?:items?\s+)?([\d,\s]+(?:\s+and\s+\d+)?)\s+'
            r'(?:are|is)\s+reverse[- ]?(?:coded|scored)',
            text_lower
        )
        if rev_match:
            rev_text = rev_match.group(1)
            reverse_items = [
                int(n) for n in re.findall(r'\d+', rev_text)
            ]
        # Also handle "reverse-coded items: 3, 5, 8" / "reverse-scored items: 3, 5, 8"
        rev_match2 = re.search(
            r'reverse[- ]?(?:coded|scored)\s+items?\s*[:\-]?\s*([\d,\s]+(?:\s+and\s+\d+)?)',
            text_lower
        )
        if rev_match2 and not reverse_items:
            rev_text = rev_match2.group(1)
            reverse_items = [
                int(n) for n in re.findall(r'\d+', rev_text)
            ]

        # ── Standard extraction (only override if abbreviation didn't set) ──
        if not abbrev_matched:
            # Extract scale range: "1-7", "0-100", "1 to 5"
            range_match = re.search(
                r'(?:from\s+)?(\d+)\s*(?:to|-|–)\s*(\d+)(?:\s*(?:scale|point|range))?',
                text
            )
            if range_match and not anchor_matches:
                scale_min = int(range_match.group(1))
                scale_max = int(range_match.group(2))

            # Extract N-point scale: "7-point", "5 point"
            # Only apply if no explicit range was already found (e.g., "0-100")
            npoint_match = re.search(r'(\d+)\s*[-–]?\s*point', text_lower)
            if npoint_match and not anchor_matches and not range_match:
                scale_max = int(npoint_match.group(1))
                scale_min = 1

            # Extract number of items: "5 items", "(3 items)", "3-item"
            items_match = re.search(r'(\d+)\s*[-–]?\s*items?', text_lower)
            if items_match:
                num_items = int(items_match.group(1))
                num_items = max(1, min(num_items, 50))

        # Detect scale type
        if 'slider' in text_lower or 'visual analog' in text_lower or 'vas' in text_lower:
            scale_type = "slider"
            if scale_min == 1 and scale_max == 7:
                scale_min = 0
                scale_max = 100
        elif 'binary' in text_lower or 'yes/no' in text_lower or 'yes or no' in text_lower:
            scale_type = "binary"
            scale_min = 0
            scale_max = 1
            num_items = 1
        elif any(kw in text_lower for kw in [
            'numeric', 'dollar', 'wtp', 'willingness to pay',
            'amount', 'price', 'cost', 'payment',
        ]):
            scale_type = "numeric"
            if scale_min == 1 and scale_max == 7:
                # Default numeric range; use 0-500 for dollar/WTP, 0-100 otherwise
                scale_min = 0
                if any(kw in text_lower for kw in ['dollar', 'wtp', 'willingness to pay', 'price', 'payment', 'cost']):
                    scale_max = 500
                else:
                    scale_max = 100
        elif any(kw in text_lower for kw in ['frequency', 'never', 'rarely', 'sometimes', 'often', 'always']):
            # Frequency-type scale (never/rarely/sometimes/often/always)
            if scale_min == 1 and scale_max == 7:
                scale_min = 1
                scale_max = 5
                if not item_labels:
                    item_labels = ["1=Never", "2=Rarely", "3=Sometimes", "4=Often", "5=Always"]

        # Try to extract scale name (only if abbreviation didn't set it)
        if not abbrev_matched:
            name_text = ""

            # ── Priority 1: "Name (Abbreviation): spec" format ──────────
            # Common academic format: "Perceived Quality (PQ): 3 items ..."
            leading_name = re.match(
                r'^([A-Z][\w\s]+?)\s*'         # Name starting with capital
                r'(?:\([^)]{1,12}\)\s*)?'       # Optional short abbreviation
                r'(?:\([^)]*(?:manipulation|covariate|check)[^)]*\)\s*)?'  # Optional role tag
                r':\s',                         # Colon delimiter
                text
            )
            if leading_name:
                name_text = leading_name.group(1).strip()
            else:
                # Also try "Name / Alt Name: spec" or "Name: spec"
                leading_simple = re.match(
                    r'^([A-Z][\w\s/]+?)\s*:\s',
                    text
                )
                if leading_simple:
                    candidate = leading_simple.group(1).strip()
                    # Only use if it looks like a name (not just "items" or numbers)
                    if len(candidate) >= 3 and not re.match(r'^\d', candidate):
                        name_text = candidate

            # ── Priority 2: Generic cleanup (fallback) ──────────────────
            if not name_text:
                name_text = text
                name_text = re.sub(r'\d+\s*[-–]?\s*point\s*', '', name_text, flags=re.IGNORECASE)
                name_text = re.sub(r'\(?\d+\s*[-–]?\s*items?\)?', '', name_text, flags=re.IGNORECASE)
                name_text = re.sub(r'(?:from\s+)?\d+\s*(?:to|-|–)\s*\d+', '', name_text)
                name_text = re.sub(r'(?:likert|scale|slider|measure|rating|binary|yes\s*/\s*no|yes\s+or\s+no)\s*', '', name_text, flags=re.IGNORECASE)
                # Remove anchor specs from name
                name_text = re.sub(r'\d+\s*=\s*[^,;]+', '', name_text)
                # Remove reverse-coded/reverse-scored mention from name
                name_text = re.sub(r'(?:items?\s+)?[\d,\s]+(?:and\s+\d+\s+)?(?:are|is)\s+reverse[- ]?(?:coded|scored)', '', name_text, flags=re.IGNORECASE)
                name_text = re.sub(r'reverse[- ]?(?:coded|scored)\s+items?\s*[:\-]?\s*[\d,\s]+', '', name_text, flags=re.IGNORECASE)
                # Remove quoted item text
                name_text = re.sub(r'"[^"]*"', '', name_text)
                name_text = re.sub(r'[,;:()]', ' ', name_text)
                name_text = name_text.strip()

            # Strip common noise words from beginning and end of name
            _noise_words = {
                'with', 'and', 'the', 'a', 'an', 'of', 'for', 'in', 'on',
                'to', 'by', 'from', 'using', 'about', 'measuring', 'measured',
                'we', 'our', 'my', 'is', 'are', 'was', 'were', 'that', 'this',
                'it', 'its', 'or', 'but', 'yes', 'no',
            }
            if name_text:
                # Strip noise words from both ends iteratively
                words = name_text.split()
                while words and words[0].lower() in _noise_words:
                    words.pop(0)
                while words and words[-1].lower() in _noise_words:
                    words.pop()
                name_text = ' '.join(words).strip()

            # Generate a descriptive fallback if name is empty or a noise word
            if not name_text or name_text.lower() in _noise_words:
                # Build a descriptive name from the scale_type and range
                if scale_type == "binary":
                    name = "Binary Response"
                elif scale_type == "slider":
                    name = f"Slider ({scale_min}-{scale_max})"
                elif scale_type == "numeric":
                    name = f"Numeric ({scale_min}-{scale_max})"
                else:
                    # Likert fallback: use N-Point Likert
                    n_points = scale_max - scale_min + 1
                    if n_points >= 2:
                        name = f"{n_points}-Point Likert"
                    else:
                        name = f"Likert {scale_min}-{scale_max}"
            else:
                name = name_text.strip()
                if len(name) > 60:
                    name = name[:57] + "..."

            # Check against known scales for better defaults
            for known_name, known_spec in KNOWN_SCALES.items():
                if known_name in text_lower:
                    name = name or known_name.title()
                    if num_items == 1 and known_spec["items"] > 1:
                        num_items = known_spec["items"]
                    if scale_min == 1 and scale_max == 7:
                        scale_min = known_spec["min"]
                        scale_max = known_spec["max"]
                    break

        if not name:
            return None

        var_name = self._to_variable_name(name)

        return ParsedScale(
            name=name.strip(),
            variable_name=var_name,
            num_items=num_items,
            scale_min=scale_min,
            scale_max=scale_max,
            scale_type=scale_type,
            description=text.strip(),
            reverse_items=reverse_items,
            item_labels=item_labels,
        )

    def _match_known_scales(self, text: str) -> List[ParsedScale]:
        """Try to match known scale names and abbreviations in the text."""
        scales = []
        text_lower = text.lower()
        # Track by canonical label to prevent duplicates when both abbreviation
        # and full name match the same underlying instrument
        matched_labels: set = set()
        # Keys that should be skipped (aliases of already-matched instruments)
        covered_keys: set = set()

        # First try abbreviations for precise matching
        matched_canonical_keys: set = set()
        for abbrev, canonical_key in SCALE_ABBREVIATIONS.items():
            if re.search(r'(?<![a-zA-Z])' + re.escape(abbrev) + r'(?![a-zA-Z])', text_lower):
                if canonical_key in KNOWN_SCALES:
                    spec = KNOWN_SCALES[canonical_key]
                    label = spec["label"]
                    if label not in matched_labels:
                        var_name = self._to_variable_name(label)
                        scales.append(ParsedScale(
                            name=label,
                            variable_name=var_name,
                            num_items=spec["items"],
                            scale_min=spec["min"],
                            scale_max=spec["max"],
                            scale_type="likert",
                            description=label,
                        ))
                        matched_labels.add(label)
                        matched_canonical_keys.add(canonical_key)

        # Find alias keys for matched abbreviations: entries with same
        # (items, min, max) whose labels cross-reference each other
        for key in list(matched_canonical_keys):
            if key in KNOWN_SCALES:
                spec = KNOWN_SCALES[key]
                fingerprint = (spec["items"], spec["min"], spec["max"])
                covered_keys.add(key)
                for other_key, other_spec in KNOWN_SCALES.items():
                    if other_key == key:
                        continue
                    other_fp = (other_spec["items"], other_spec["min"], other_spec["max"])
                    if fingerprint == other_fp:
                        # Same structure -- check if labels cross-reference
                        if (key in other_spec["label"].lower()
                                or other_key in spec["label"].lower()):
                            covered_keys.add(other_key)

        # Then try full known scale names, longest first so that shorter
        # substring keys (e.g. "anxiety") are covered when a longer key
        # (e.g. "generalized anxiety disorder") matches first
        sorted_known = sorted(
            KNOWN_SCALES.items(), key=lambda x: len(x[0]), reverse=True
        )
        for known_name, spec in sorted_known:
            if known_name in text_lower:
                # Mark shorter keys that are substrings of this matched key
                # as covered, regardless of whether we add this scale
                for other_name in KNOWN_SCALES:
                    if other_name != known_name and other_name in known_name:
                        covered_keys.add(other_name)

                label = spec["label"]
                if label not in matched_labels and known_name not in covered_keys:
                    var_name = self._to_variable_name(known_name)
                    scales.append(ParsedScale(
                        name=known_name.title(),
                        variable_name=var_name,
                        num_items=spec["items"],
                        scale_min=spec["min"],
                        scale_max=spec["max"],
                        scale_type="likert",
                        description=label,
                    ))
                    matched_labels.add(label)

        return scales

    def _to_variable_name(self, name: str) -> str:
        """Convert a display name to a valid variable name."""
        var = re.sub(r'[^a-zA-Z0-9\s]', '', name)
        var = re.sub(r'\s+', '_', var.strip())
        # Ensure no leading digit (invalid identifier)
        if var and var[0].isdigit():
            var = 'v_' + var
        var = var[:30]  # Max length
        if not var:
            var = "var"
        return var

    def _generate_var_name(self, text: str, prefix: str = "Q", index: int = 1) -> str:
        """Generate a variable name from question text."""
        # Stop words to skip when building variable names
        _stop_words = {
            'the', 'a', 'an', 'of', 'for', 'in', 'on', 'to', 'by', 'from',
            'with', 'and', 'or', 'but', 'your', 'you', 'this', 'that', 'what',
            'how', 'why', 'did', 'was', 'were', 'are', 'is', 'has', 'have',
            'please', 'would', 'could', 'about', 'any', 'anything',
        }
        # Extract meaningful words
        words = re.findall(r'[a-zA-Z]+', text)
        key_words = [w.lower() for w in words if len(w) > 2 and w.lower() not in _stop_words]
        if key_words:
            # Take up to 3 meaningful words for context
            return '_'.join(key_words[:3])[:30]
        return f"{prefix}_{index}"

    def _detect_oe_context(self, text: str) -> str:
        """Detect the context type of an open-ended question."""
        text_lower = text.lower()
        if any(w in text_lower for w in ['explain', 'why', 'reason', 'because']):
            return "explanation"
        if any(w in text_lower for w in ['feel', 'emotion', 'experience', 'reaction']):
            return "experience"
        if any(w in text_lower for w in ['think', 'opinion', 'believe', 'view']):
            return "opinion"
        if any(w in text_lower for w in ['suggest', 'recommend', 'improve', 'advice']):
            return "recommendation"
        if any(w in text_lower for w in ['describe', 'tell', 'share', 'narrative']):
            return "narrative"
        return "general"
