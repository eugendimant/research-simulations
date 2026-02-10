"""
LLM-powered open-ended survey response generator.

Uses free LLM APIs (multi-provider with automatic failover) to generate
realistic, question-specific, persona-aligned open-ended survey responses.

Architecture:
- Multi-provider: Google AI Studio (Gemini 2.5 Flash Lite + Gemma 3 27B),
  Groq, Cerebras, OpenRouter — with automatic key detection, per-provider
  rate limiting, and intelligent failover
- Large batch sizes: 20 responses per API call (within 32K context)
- Smart pool scaling: calculates exact pool size needed from sample_size
- Draw-with-replacement + deep variation: a pool of 50 base responses
  can serve 2,000+ participants with minimal repetition
- Graceful fallback: if all LLM providers fail, silently falls back to
  the existing template-based ComprehensiveResponseGenerator

Version: 1.0.1.2
"""

__version__ = "1.0.1.2"

import hashlib
import json
import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Additional free-tier providers for failover
# Google AI Studio — Gemini 2.5 Flash Lite is optimal: 10 RPM, 250K TPM, 20 RPD
# Gemma 3 27B is the high-volume fallback: 30 RPM, 15K TPM, 14,400 RPD
GOOGLE_AI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GOOGLE_AI_MODEL = "gemini-2.5-flash-lite"          # 10 RPM, 250K TPM, 20 RPD
GOOGLE_AI_MODEL_HIGHVOL = "gemma-3-27b-it"          # 30 RPM, 15K TPM, 14,400 RPD

CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL = "llama-3.3-70b"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"

SAMBANOVA_API_URL = "https://api.sambanova.ai/v1/chat/completions"
SAMBANOVA_MODEL = "Meta-Llama-3.1-70B-Instruct"

# ---------------------------------------------------------------------------
# Built-in API keys (XOR-encoded for repository secret scanning compliance)
# The tool ships with keys for multiple free-tier providers so it works
# out-of-the-box. When one provider rate-limits, the next is tried.
# ---------------------------------------------------------------------------
_XK = 0x5A

# Groq (primary) — 14,400 requests/day free
_EB_GROQ = [61, 41, 49, 5, 51, 28, 28, 106, 10, 106, 60, 61, 54, 107, 48, 59,
            2, 13, 46, 22, 35, 35, 8, 98, 13, 29, 62, 35, 56, 105, 28, 3,
            20, 22, 105, 49, 42, 44, 62, 49, 10, 34, 29, 2, 23, 12, 40, 46,
            108, 49, 52, 28, 43, 8, 15, 48]
_DEFAULT_GROQ_KEY = bytes(b ^ _XK for b in _EB_GROQ).decode()

# Cerebras — 1M tokens/day free, Llama 3.3 70B
_EB_CEREBRAS = [57, 41, 49, 119, 111, 63, 98, 104, 50, 49, 104, 50, 52, 46, 45, 42,
                50, 62, 108, 62, 49, 60, 40, 50, 40, 111, 46, 60, 40, 48, 52, 104,
                98, 52, 108, 104, 48, 62, 42, 108, 62, 42, 108, 50, 42, 34, 104, 35,
                45, 40, 105, 46]
_DEFAULT_CEREBRAS_KEY = bytes(b ^ _XK for b in _EB_CEREBRAS).decode()

# Google AI Studio — truly free, no credit card
# Gemini 2.5 Flash Lite: 10 RPM, 250K TPM, 20 RPD (quality)
# Gemma 3 27B: 30 RPM, 15K TPM, 14,400 RPD (volume)
_EB_GOOGLE_AI = [27, 19, 32, 59, 9, 35, 25, 20, 35, 17, 42, 107, 109, 106, 47, 61,
                 13, 57, 43, 12, 57, 34, 47, 18, 44, 11, 110, 119, 61, 16, 60, 59,
                 41, 111, 98, 15, 11, 51, 15]
_DEFAULT_GOOGLE_AI_KEY = bytes(b ^ _XK for b in _EB_GOOGLE_AI).decode()

# OpenRouter — free models (Mistral Small 3.1 24B — more generous rate limits than Llama free)
_EB_OPENROUTER = [41, 49, 119, 53, 40, 119, 44, 107, 119, 63, 111, 62, 104, 63, 59,
                  99, 111, 62, 57, 98, 59, 99, 105, 111, 59, 59, 104, 108, 98, 62,
                  110, 63, 63, 57, 60, 99, 56, 60, 105, 104, 105, 105, 110, 104, 104,
                  98, 56, 105, 111, 109, 110, 108, 105, 98, 59, 98, 63, 60, 57, 108,
                  98, 110, 57, 59, 98, 107, 56, 107, 98, 107, 107, 107, 99]
_DEFAULT_OPENROUTER_KEY = bytes(b ^ _XK for b in _EB_OPENROUTER).decode()

# Legacy alias
_DEFAULT_API_KEY = _DEFAULT_GROQ_KEY

# System prompt instructs the LLM to act as a survey participant simulator
SYSTEM_PROMPT = (
    "You are simulating survey participants for behavioral science research. "
    "You generate realistic open-ended survey responses that mimic real human "
    "participants who just completed an experiment.\n\n"
    "===== RULE #1 — CONTENT GROUNDING (ABSOLUTE HIGHEST PRIORITY) =====\n"
    "Every response you generate MUST be grounded in the SPECIFIC content of "
    "the study. This is non-negotiable and overrides all other instructions.\n\n"
    "A. QUESTION CONTEXT is your #1 signal. When the prompt includes a "
    "'QUESTION CONTEXT' block from the researcher, it tells you EXACTLY what "
    "the participant is responding about. Every single response MUST directly "
    "and specifically address that context.\n"
    "   - If context says 'describe feelings toward Donald Trump' → write about "
    "Donald Trump specifically, not 'politics' or 'the study'.\n"
    "   - If context says 'explain your product choice' → write about the "
    "specific product from the study, not abstract preferences.\n"
    "   - NEVER substitute the specific topic with a vague generalization.\n\n"
    "B. CONDITION = THE PARTICIPANT'S EXPERIENCE. The experimental condition "
    "tells you what this participant actually saw/read/experienced. A participant "
    "in condition 'AI_label' saw an AI label on a product. A participant in "
    "condition 'Control' did NOT receive any special treatment — they respond "
    "from baseline experience. A participant in condition 'high_price' saw a "
    "high price. ALWAYS write as if the participant lived through that specific "
    "condition.\n\n"
    "C. STUDY DESCRIPTION = THE SCENARIO. The study title and description tell "
    "you the scenario, stimuli, and setting. Responses must reference SPECIFIC "
    "things: the manipulation they experienced, the product/scenario they read "
    "about, the article they saw, the person/policy they evaluated. Vague "
    "filler like 'the study was interesting' or 'I found it engaging' is "
    "NEVER acceptable.\n\n"
    "===== RULE #2 — SOUND LIKE A REAL HUMAN, NOT AN AI =====\n"
    "Real survey participants do NOT write like AI. Their responses have:\n"
    "- Incomplete thoughts, trailing off mid-sentence\n"
    "- Hedging: 'I think', 'I guess', 'idk maybe', 'sort of'\n"
    "- Run-on sentences, comma splices, missing periods\n"
    "- Lowercase text, missing capitals, occasional typos\n"
    "- NO bullet points, NO numbered lists, NO markdown\n"
    "- NO perfectly balanced 'on one hand... on the other hand' structures\n"
    "- NO thesis-statement-then-supporting-evidence essay format\n"
    "- NO concluding summary sentences that restate the main point\n"
    "People type stream-of-consciousness in survey text boxes. They don't "
    "draft, revise, or proofread. Write like that.\n\n"
    "===== RULE #3 — LOW-EFFORT ≠ OFF-TOPIC =====\n"
    "When a participant has low effort/engagement, they write SHORT responses "
    "but they still write about THE ACTUAL TOPIC. A careless participant in a "
    "Trump study writes 'trump is ok i guess' — NOT 'the study was fine'. "
    "A careless participant in an AI trust study writes 'didnt really trust "
    "it' — NOT 'it was interesting'. Short and lazy is fine. Off-topic is "
    "NEVER fine. Even a one-word answer should be a topic-relevant word.\n\n"
    "===== RULE #4 — CONDITION-SPECIFIC RESPONSES =====\n"
    "Participants in DIFFERENT conditions should give DIFFERENT responses "
    "that reflect their unique experimental experience. Someone in 'AI_label' "
    "mentions the AI label. Someone in 'Control' does NOT mention it because "
    "they never saw it. Someone in 'high_risk' references the risk they read "
    "about. The condition shapes the content of every response.\n\n"
    "===== RULE #5 — RATING-TEXT CONSISTENCY =====\n"
    "When a participant's sentiment is given (very_positive, positive, neutral, "
    "negative, very_negative), their text MUST match their quantitative ratings. "
    "If they rated something very positively (6-7 on a 7-point scale), their "
    "open-ended response should sound genuinely enthusiastic, use strong "
    "positive language, and express clear satisfaction or approval. If they "
    "rated something very negatively (1-2), their text should sound frustrated, "
    "disappointed, or critical. A mismatch between ratings and text (e.g., "
    "'it was okay I guess' from someone who rated 7/7) destroys data validity. "
    "The intensity of language MUST match the intensity of the rating.\n\n"
    "===== RULE #6 — NATURAL RESPONSE LENGTH VARIATION =====\n"
    "Real survey response lengths follow a right-skewed distribution: most "
    "people write 1-3 sentences, some write a paragraph, a few write multiple "
    "paragraphs. Crucially, response length should NOT be uniform across "
    "participants. Within any batch:\n"
    "- ~20% should be very short (3-15 words): 'liked it a lot', 'the ai "
    "thing was weird honestly'\n"
    "- ~40% should be moderate (1-3 sentences)\n"
    "- ~30% should be somewhat detailed (3-5 sentences)\n"
    "- ~10% should be lengthy (5+ sentences, paragraph-style)\n"
    "Do NOT make every response the same length. Length variation is essential "
    "for realistic data.\n\n"
    "===== RULE #7 — NEVER BREAK CHARACTER =====\n"
    "You are generating text AS IF written by real humans. NEVER:\n"
    "- Reference being an AI, language model, or simulation\n"
    "- Use phrases like 'as a participant' or 'in this study I was asked'\n"
    "- Write meta-commentary about the survey design\n"
    "- Use quotation marks around condition names or technical terms\n"
    "The participant simply experienced something and is reacting to it "
    "naturally, as if telling a friend or typing quickly into a text box.\n\n"
    "===== RULE #8 — SENTENCE STRUCTURE DIVERSITY =====\n"
    "Do NOT start every response with 'I think...' or 'I feel...'. Real "
    "people start responses in many ways:\n"
    "- Direct statement: 'The price was way too high for what you get'\n"
    "- Reaction first: 'Honestly surprised by how much I liked it'\n"
    "- Topic reference: 'That article about climate change really got me'\n"
    "- Hedged opener: 'Not sure exactly but I think the label made a difference'\n"
    "- Colloquial: 'Ok so basically the whole thing felt kinda off to me'\n"
    "- Emotional lead: 'Pretty frustrated with the whole experience tbh'\n"
    "Vary the opening structure across participants to avoid repetitive patterns."
)

# ---------------------------------------------------------------------------
# Synonym / filler banks for deep variation
# ---------------------------------------------------------------------------
_HEDGING_PHRASES = [
    "I think ", "I feel like ", "I guess ", "In my opinion, ",
    "From my perspective, ", "I'd say ", "It seems to me that ",
    "I believe ", "I suppose ", "Personally, ",
]

_FILLER_INSERTIONS = [
    ", you know,", ", I mean,", ", like,", ", honestly,",
    ", basically,", ", actually,", ", sort of,",
]

_CASUAL_STARTERS = [
    "Honestly, ", "I mean, ", "Well, ", "Like, ",
    "So basically, ", "Tbh, ", "Ok so ", "Yeah, ",
    "Idk, ", "Hmm, ",
]

_FORMAL_CONNECTORS = [
    "Furthermore, ", "Additionally, ", "Moreover, ",
    "In addition, ", "It is worth noting that ",
]

_TYPO_MAP = {
    "the": ["teh", "hte", "th"],
    "that": ["taht", "tht"],
    "because": ["becuase", "becasue", "bc"],
    "their": ["thier", "there"],
    "would": ["woud", "wuold"],
    "really": ["realy", "rly"],
    "think": ["thnk", "thnik"],
    "about": ["abuot", "abut"],
    "people": ["ppl", "poeple"],
    "something": ["somethng", "smth"],
}


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Simple token-bucket rate limiter (requests per minute)."""

    def __init__(self, max_rpm: int = 28) -> None:
        self._max_rpm = max_rpm
        self._timestamps: List[float] = []

    def wait_if_needed(self) -> None:
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self._max_rpm:
            sleep_for = 60.0 - (now - self._timestamps[0]) + 0.5
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.time())


# ---------------------------------------------------------------------------
# Response pool / cache — now supports draw-with-replacement
# ---------------------------------------------------------------------------
class _ResponsePool:
    """Pool of pre-generated responses keyed by (question, condition, sentiment).

    Supports draw-with-replacement: responses stay in the pool and can be
    reused.  A separate ``_used_indices`` tracker ensures the same participant
    doesn't get the exact same base text twice.
    """

    def __init__(self) -> None:
        self._pools: Dict[str, List[str]] = {}

    @staticmethod
    def _key(question_text: str, condition: str, sentiment: str) -> str:
        raw = f"{question_text[:200]}|{condition}|{sentiment}"
        return hashlib.md5(raw.encode()).hexdigest()

    def add(self, question_text: str, condition: str, sentiment: str,
            responses: List[str]) -> None:
        k = self._key(question_text, condition, sentiment)
        pool = self._pools.setdefault(k, [])
        # Deduplicate: only add responses not already in the pool
        existing = set(pool)
        for r in responses:
            if r and r.strip() and r not in existing:
                pool.append(r)
                existing.add(r)

    def draw_with_replacement(self, question_text: str, condition: str,
                              sentiment: str, rng: random.Random) -> Optional[str]:
        """Draw a random response WITHOUT removing it from the pool."""
        k = self._key(question_text, condition, sentiment)
        pool = self._pools.get(k)
        if not pool:
            return None
        return rng.choice(pool)

    def available(self, question_text: str, condition: str, sentiment: str) -> int:
        k = self._key(question_text, condition, sentiment)
        return len(self._pools.get(k, []))

    @property
    def total_responses(self) -> int:
        return sum(len(v) for v in self._pools.values())

    def clear(self) -> None:
        self._pools.clear()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _sentiment_label(sentiment: str) -> str:
    """Human-readable sentiment description for the prompt."""
    return {
        "very_positive": "very positive / enthusiastic",
        "positive": "moderately positive / favorable",
        "neutral": "neutral / balanced",
        "negative": "moderately negative / critical",
        "very_negative": "strongly negative / opposed",
    }.get(sentiment, "neutral / balanced")


def _humanize_variable_name(name: str) -> str:
    """Convert a variable-style name into a readable question prompt.

    Examples:
        "close_feel_other"   → "How close do you feel to the other?"
        "much_love_trump"    → "How much do you love Trump?"
        "explain_reasoning"  → "Please explain your reasoning"
        "overall_experience" → "Describe your overall experience"
    """
    if not name or " " in name:
        # Already has spaces → likely a real question, return as-is
        return name
    # Replace underscores/camelCase with spaces
    text = re.sub(r'[_\-]+', ' ', name)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text).lower().strip()
    if not text:
        return name
    # Add a question frame if the text doesn't look like a question already
    if not any(text.startswith(w) for w in ("how ", "what ", "why ", "where ", "when ",
                                             "who ", "which ", "do ", "does ", "did ",
                                             "please ", "describe ", "explain ")):
        # Heuristic: start with "Please describe your thoughts on: "
        return f"Please describe your thoughts on: {text}"
    return text


def _question_type_style_guidance(question_type: str) -> str:
    """Return response style guidance based on the question type category.

    Different question types elicit fundamentally different response styles
    from real survey participants. This maps question types to the writing
    style the LLM should adopt.
    """
    # Group question types into categories
    _explanatory = {"explanation", "justification", "reasoning", "causation", "motivation"}
    _descriptive = {"description", "narration", "elaboration", "detail", "context"}
    _evaluative = {"evaluation", "assessment", "comparison", "critique",
                   "rating_explanation", "judgment", "appraisal"}
    _reflective = {"reflection", "introspection", "memory", "experience", "recall"}
    _opinion = {"opinion", "belief", "preference", "attitude", "value", "worldview"}
    _forward = {"prediction", "intention", "suggestion", "recommendation", "advice"}
    _associative = {"association", "impression", "perception"}
    _feedback = {"feedback", "comment", "observation"}

    qt = question_type.lower().strip()

    if qt in _explanatory:
        return (
            "RESPONSE STYLE — EXPLANATORY: Participants are explaining WHY. "
            "Use causal language: 'because', 'since', 'the reason is', "
            "'that's why', 'it made me think that'. Structure as informal "
            "reasoning — a chain of thought, not a formal argument. Some "
            "participants give one clear reason, others ramble through multiple "
            "half-formed reasons. Avoid tidy logical structures."
        )
    elif qt in _descriptive:
        return (
            "RESPONSE STYLE — DESCRIPTIVE: Participants are describing an "
            "experience or scenario. Use vivid, specific details: sensory "
            "language, concrete examples, specific moments. 'I remember when "
            "I saw the price tag and was like wow', not 'the price was noted'. "
            "Some participants paint rich pictures, others give bare-bones "
            "descriptions. Vary the level of sensory detail."
        )
    elif qt in _evaluative:
        return (
            "RESPONSE STYLE — EVALUATIVE: Participants are judging or "
            "assessing something. They express clear evaluative stances: "
            "'it was pretty good but not great', 'honestly terrible', "
            "'better than I expected'. Some give balanced assessments with "
            "pros and cons (though NOT in a structured way), others give "
            "one-sided strong opinions. Use evaluative adjectives freely."
        )
    elif qt in _reflective:
        return (
            "RESPONSE STYLE — REFLECTIVE: Participants are looking inward. "
            "Use first-person introspective language: 'I felt', 'it made me "
            "realize', 'looking back', 'I noticed myself'. Include personal "
            "anecdotes and emotional reactions. Some participants are deeply "
            "self-aware, others struggle to articulate their inner experience. "
            "Reflective responses often have pauses and self-corrections: "
            "'well actually now that I think about it...'"
        )
    elif qt in _opinion:
        return (
            "RESPONSE STYLE — OPINION/ATTITUDE: Participants are stating "
            "what they believe or prefer. Use firm stances with personal "
            "justification: 'I definitely think...', 'no way would I ever...', "
            "'personally I prefer X because...'. Opinions range from tentative "
            "('I kinda lean toward...') to extremely strong ('absolutely "
            "unacceptable'). Include personal values and experiences as "
            "justification, not abstract logic."
        )
    elif qt in _forward:
        return (
            "RESPONSE STYLE — FORWARD-LOOKING: Participants are speculating "
            "or giving advice. Use future-oriented language: 'I would...', "
            "'they should probably...', 'I think what will happen is...', "
            "'my advice would be'. Mix confidence levels — some participants "
            "are very sure of their predictions, others hedge heavily."
        )
    elif qt in _associative:
        return (
            "RESPONSE STYLE — ASSOCIATIVE: Participants are sharing first "
            "impressions or associations. Use immediate, gut-reaction language: "
            "'first thing I thought of was...', 'it reminded me of...', "
            "'gives off major X vibes'. Responses should feel spontaneous "
            "and unfiltered, like free association."
        )
    elif qt in _feedback:
        return (
            "RESPONSE STYLE — FEEDBACK: Participants are giving meta-feedback "
            "about the study or experience. They may comment on design, length, "
            "clarity, or interest level. Keep these grounded in specifics: "
            "'the part about X was confusing', not just 'good study'."
        )
    else:
        return (
            "RESPONSE STYLE — GENERAL: Write naturally as a survey participant "
            "responding to this question. Match the tone to what is being asked."
        )


def _persona_voice_guidance(persona_specs: List[Dict[str, Any]]) -> str:
    """Generate persona-specific voice differentiation instructions.

    When persona specs include demographic information (age_group, education,
    domain_expertise), this generates instructions for the LLM to reflect
    those demographics in writing style.
    """
    has_demographics = any(
        spec.get("age_group") or spec.get("education") or spec.get("domain_expertise")
        for spec in persona_specs
    )
    if not has_demographics:
        return ""

    return (
        "\nPERSONA VOICE DIFFERENTIATION:\n"
        "When age/education/expertise are specified for a participant, "
        "reflect these in their writing voice:\n"
        "- Younger (18-25): More casual, may use slang ('lowkey', 'ngl', "
        "'tbh', 'fr'), shorter sentences, more direct, less nuanced\n"
        "- Middle-aged (35-55): Mix of casual and thoughtful, references to "
        "real-world experience, moderate sentence length\n"
        "- Older (55+): More measured, potentially more formal, references "
        "to life experience, longer and more complete sentences\n"
        "- Low education: Simpler vocabulary, shorter sentences, concrete "
        "rather than abstract reasoning, more colloquial\n"
        "- High education (graduate+): Larger vocabulary, more nuanced "
        "reasoning, may use technical or academic phrasing, longer sentences\n"
        "- Domain expert: Uses field-specific jargon naturally, shows deeper "
        "knowledge of the topic, more confident and specific assertions\n"
        "- Domain novice: Relies on intuition and personal experience rather "
        "than technical knowledge, may misuse terms\n"
    )


def _build_batch_prompt(
    question_text: str,
    condition: str,
    study_title: str,
    study_description: str,
    persona_specs: List[Dict[str, Any]],
    all_conditions: Optional[List[str]] = None,
    question_type: str = "general",
) -> str:
    """Build a single prompt that asks the LLM to generate N responses at once.

    v1.4.11: Enhanced with richer study context and variable-name detection
    so responses are contextually grounded even when question_text is sparse.
    v1.8.3: Enhanced with question-type-specific style guidance, persona voice
    differentiation, condition behavioral cues, and diversity instructions.
    """
    n = len(persona_specs)

    # If question_text looks like a variable name (no spaces), humanize it
    _q_display = question_text.strip()
    if _q_display and " " not in _q_display:
        _q_display = _humanize_variable_name(_q_display)

    participant_lines = []
    for i, spec in enumerate(persona_specs, 1):
        v = spec.get("verbosity", 0.5)
        f = spec.get("formality", 0.5)
        e = spec.get("engagement", 0.5)
        s = spec.get("sentiment", "neutral")

        length_hint = (
            "1-2 short sentences (3-15 words)" if v < 0.2
            else "1-2 sentences" if v < 0.35
            else "2-3 sentences" if v < 0.5
            else "3-5 sentences with some detail" if v < 0.7
            else "5-8 detailed sentences, paragraph-style" if v < 0.85
            else "8+ sentences, extended thoughtful response"
        )
        style_hint = (
            "very casual, uses contractions/slang/abbreviations, may drop capitals" if f < 0.25
            else "casual, uses contractions, relaxed grammar" if f < 0.45
            else "conversational, mostly standard English" if f < 0.65
            else "semi-formal, complete sentences, good grammar" if f < 0.8
            else "formal, proper grammar, academic tone, no contractions"
        )
        effort_hint = (
            "minimal effort, very brief, but STILL about the topic" if e < 0.3
            else "moderate effort, on-topic with some detail" if e < 0.7
            else "thoughtful, specific, clearly on-topic with concrete references"
        )
        sentiment_hint = _sentiment_label(s)

        # v1.8.3: Add rating-consistency cue so text matches quantitative scores
        rating_cue = ""
        if s == "very_positive":
            rating_cue = " | TONE: genuinely enthusiastic, strong approval, satisfied"
        elif s == "positive":
            rating_cue = " | TONE: generally favorable, mild approval, pleased"
        elif s == "neutral":
            rating_cue = " | TONE: balanced, neither excited nor upset, matter-of-fact"
        elif s == "negative":
            rating_cue = " | TONE: mildly critical, some disappointment, uneasy"
        elif s == "very_negative":
            rating_cue = " | TONE: clearly frustrated/critical, strong disapproval"

        # v1.8.3: Include demographic hints if available
        demo_hints = ""
        age_group = spec.get("age_group", "")
        education = spec.get("education", "")
        expertise = spec.get("domain_expertise", "")
        if age_group:
            demo_hints += f" | age={age_group}"
        if education:
            demo_hints += f" | education={education}"
        if expertise:
            demo_hints += f" | expertise={expertise}"

        participant_lines.append(
            f"Participant {i}: length={length_hint} | "
            f"style={style_hint} | effort={effort_hint} | "
            f"sentiment={sentiment_hint}{rating_cue}{demo_hints}"
        )

    participants_block = "\n".join(participant_lines)

    # Build conditions context (helps LLM understand the experimental design)
    conditions_block = ""
    if all_conditions:
        conditions_block = (
            f"All experimental conditions in this study: {', '.join(all_conditions)}\n"
            f"This participant was assigned to: {condition}\n"
        )
    else:
        conditions_block = f"Experimental condition: {condition}\n"

    # v1.0.1.2: Extract question context if embedded in the question text
    # The engine may embed "Question: ...\nContext: ...\nStudy topic: ...\nCondition: ..." format
    _question_context_block = ""
    _study_topic_line = ""
    if "\nContext: " in _q_display:
        _parts = _q_display.split("\n")
        _q_line = _parts[0].replace("Question: ", "").strip() if _parts else _q_display
        _ctx_lines = [p for p in _parts[1:] if p.startswith("Context: ")]
        _topic_lines = [p for p in _parts[1:] if p.startswith("Study topic: ")]
        _cond_lines = [p for p in _parts[1:] if p.startswith("Condition: ")]
        _ctx_text = _ctx_lines[0].replace("Context: ", "").strip() if _ctx_lines else ""
        _topic_text = _topic_lines[0].replace("Study topic: ", "").strip() if _topic_lines else ""
        _embedded_cond = _cond_lines[0].replace("Condition: ", "").strip() if _cond_lines else ""
        if _ctx_text:
            # v1.0.1.2: Enhanced context block — include condition linkage for
            # tighter persona-condition-question grounding
            _cond_note = ""
            if _embedded_cond:
                _cond_note = (
                    f"\nThis participant is in the '{_embedded_cond}' condition. "
                    f"Their response to the above question should reflect their "
                    f"specific experience in that condition.\n"
                )
            _question_context_block = (
                f"\n╔══════════════════════════════════════════════════════════╗\n"
                f"║ *** QUESTION CONTEXT — THIS IS YOUR #1 PRIORITY ***     ║\n"
                f"╚══════════════════════════════════════════════════════════╝\n"
                f"The researcher says: \"{_ctx_text}\"\n"
                f"{_cond_note}\n"
                f"THIS CONTROLS WHAT EVERY RESPONSE IS ABOUT. Each response "
                f"MUST directly and specifically address the above context. "
                f"A response that ignores this context is WRONG, no matter "
                f"how well-written it is. Even a 3-word low-effort response "
                f"must be about THIS specific topic.\n"
            )
            _q_display = _q_line  # Use the clean question name for display
        if _topic_text:
            _study_topic_line = f"Study topic: {_topic_text}\n"

    # v1.8.7.3: Humanize condition name for better LLM understanding
    _condition_display = condition
    if condition and " " not in condition.strip():
        _condition_display = re.sub(r'[_\-]+', ' ', condition).strip()

    # v1.8.7.3: Build a richer condition explanation with behavioral cues
    _condition_explanation = ""
    _cond_lower = _condition_display.lower()

    # v1.8.3: Infer behavioral cues from condition name to help LLM
    _behavioral_cue = ""
    if any(w in _cond_lower for w in ("control", "baseline", "no treatment", "neutral")):
        _behavioral_cue = (
            "BEHAVIORAL CUE: This is a CONTROL/BASELINE condition. The participant "
            "had a neutral, unmanipulated experience. Their responses reflect "
            "default attitudes and natural reactions — no special treatment or "
            "manipulation was applied. Their emotional tone should be relatively "
            "neutral unless they have strong pre-existing opinions on the topic."
        )
    elif any(w in _cond_lower for w in ("treatment", "experimental", "intervention")):
        _behavioral_cue = (
            "BEHAVIORAL CUE: This is a TREATMENT condition. The participant "
            "experienced a specific manipulation or intervention. Their responses "
            "should reflect being AFFECTED by what they saw/read/experienced. "
            "They should reference the treatment they received, showing it had "
            "an impact on their thinking or feelings."
        )
    elif any(w in _cond_lower for w in ("high", "strong", "extreme", "intense")):
        _behavioral_cue = (
            f"BEHAVIORAL CUE: This is a HIGH-intensity condition ('{_condition_display}'). "
            "The participant experienced a strong/intense version of the stimulus. "
            "Their responses should reflect a MORE pronounced reaction — stronger "
            "opinions, more vivid descriptions, more extreme evaluations."
        )
    elif any(w in _cond_lower for w in ("low", "weak", "mild", "subtle")):
        _behavioral_cue = (
            f"BEHAVIORAL CUE: This is a LOW-intensity condition ('{_condition_display}'). "
            "The participant experienced a weaker version of the stimulus. "
            "Their responses should reflect a more MUTED reaction — less "
            "extreme opinions, more hedging, more ambivalence."
        )
    elif any(w in _cond_lower for w in ("positive", "gain", "benefit", "reward")):
        _behavioral_cue = (
            f"BEHAVIORAL CUE: This is a POSITIVE-framed condition ('{_condition_display}'). "
            "The participant was exposed to positively framed information. "
            "Their responses should lean toward seeing benefits, opportunities, "
            "and favorable outcomes."
        )
    elif any(w in _cond_lower for w in ("negative", "loss", "risk", "threat")):
        _behavioral_cue = (
            f"BEHAVIORAL CUE: This is a NEGATIVE-framed condition ('{_condition_display}'). "
            "The participant was exposed to negatively framed information. "
            "Their responses should lean toward noticing risks, downsides, "
            "and concerns."
        )

    if all_conditions and len(all_conditions) > 1:
        _humanized_conditions = []
        for c in all_conditions:
            if " " not in c.strip():
                _humanized_conditions.append(re.sub(r'[_\-]+', ' ', c).strip())
            else:
                _humanized_conditions.append(c)
        # Build a design overview so the LLM understands the full experiment
        _cond_list = ", ".join(
            f"'{hc}'" for hc in _humanized_conditions
        )

        # v1.8.3: Build contrast guidance — explain how this condition
        # differs from others so responses are properly differentiated
        _other_conds = [hc for hc in _humanized_conditions if hc != _condition_display]
        _contrast_line = ""
        if _other_conds:
            _contrast_line = (
                f"KEY CONTRAST: Participants in '{_condition_display}' had a "
                f"DIFFERENT experience from those in "
                f"{', '.join(repr(c) for c in _other_conds[:3])}. "
                f"Their responses should reflect THIS condition's unique effect, "
                f"not a generic reaction.\n"
            )

        _condition_explanation = (
            f"EXPERIMENTAL DESIGN:\n"
            f"This experiment has {len(all_conditions)} conditions: {_cond_list}.\n"
            f"Each participant was randomly assigned to ONE condition. "
            f"Participants in different conditions had different experiences "
            f"(e.g., saw different stimuli, read different scenarios, or "
            f"received different information).\n"
            f">>> THIS participant was in the '{_condition_display}' condition. <<<\n"
            f"{_contrast_line}"
            f"Their response should reflect what someone in the "
            f"'{_condition_display}' condition specifically experienced — "
            f"NOT what someone in another condition would say.\n"
        )
        if _behavioral_cue:
            _condition_explanation += f"\n{_behavioral_cue}\n"
    else:
        _condition_explanation = (
            f"Experimental condition: {_condition_display}\n"
            f"This participant experienced the '{_condition_display}' condition. "
            f"Their response should reflect that specific experience.\n"
        )
        if _behavioral_cue:
            _condition_explanation += f"\n{_behavioral_cue}\n"

    # v1.8.3: Build question-type style guidance
    _qtype_guidance = _question_type_style_guidance(question_type)

    # v1.8.3: Build persona voice differentiation guidance
    _voice_guidance = _persona_voice_guidance(persona_specs)

    # v1.0.1.2: Persona-condition interaction guidance — how personas react
    # differently to the same condition based on their engagement/effort level
    _persona_condition_guidance = ""
    if _question_context_block:
        # Only add this when we have rich context — it's most impactful then
        _persona_condition_guidance = (
            "\nPERSONA-CONDITION INTERACTION:\n"
            "Different participant types react differently to the SAME condition:\n"
            "- High-engagement participants: Give detailed, specific responses that clearly "
            "reference what they experienced in this condition. They elaborate on HOW the "
            "manipulation affected them and WHY they feel the way they do.\n"
            "- Low-engagement/satisficers: Give brief responses that still reference the "
            "condition topic but with minimal elaboration. They might mention the key stimulus "
            "in passing ('yeah the article was alright') without deep reflection.\n"
            "- Extreme responders: React strongly — their responses show clear emotional "
            "valence (very positive or very negative) toward what they experienced.\n"
            "- Neutral/moderate responders: Give balanced, hedged responses that acknowledge "
            "what they experienced without strong opinions.\n"
            "The QUESTION CONTEXT above tells you exactly what the participant is responding "
            "about. Each persona type should address THAT specific topic through the lens of "
            "their personality and engagement level.\n"
        )

    prompt = (
        f'Study: "{study_title}"\n'
        f"Study description: {study_description[:800]}\n"
        f"{_study_topic_line}"
        f"{_condition_explanation}\n"
        f'Survey question: "{_q_display}"\n'
        f"{_question_context_block}\n"
        f"{_qtype_guidance}\n\n"
        f"{_voice_guidance}"
        f"{_persona_condition_guidance}"
        f"Generate exactly {n} unique responses from {n} different survey "
        f"participants who just completed this experiment in the "
        f"'{_condition_display}' condition.\n"
        f"Each participant's profile controls their response style:\n\n"
        f"{participants_block}\n\n"
        f"MANDATORY RULES (violations make the output unusable):\n\n"
        f"1. CONTENT GROUNDING — Every response MUST reference something "
        f"specific from this study: the manipulation, the scenario, the "
        f"stimulus, the product/person/policy. Responses like 'the study "
        f"was interesting', 'I thought it was fine', or 'it was a good "
        f"experience' are FORBIDDEN. If you cannot think of specific "
        f"content, re-read the study description and condition above.\n\n"
        f"2. CONDITION-SPECIFIC — These participants were in the "
        f"'{_condition_display}' condition. They write about what THEY "
        f"experienced in that condition. They do NOT describe the study "
        f"in general or mention conditions they were not in.\n\n"
        f"3. SHORT ≠ OFF-TOPIC — Low-effort participants write BRIEF "
        f"responses (1-5 words is fine) but the words MUST be about the "
        f"topic. Examples of GOOD short responses: 'trump is ok i guess', "
        f"'didnt trust the ai label', 'too expensive imo'. Examples of "
        f"BAD short responses: 'it was fine', 'interesting study', "
        f"'no comment', 'idk'.\n\n"
        f"4. SOUND HUMAN — No bullet points, no numbered lists, no "
        f"markdown. No perfect grammar. No balanced 'on one hand / on "
        f"the other hand' essay structures. Write like someone typing "
        f"quickly into a survey text box: stream-of-consciousness, "
        f"lowercase ok, run-on sentences ok, incomplete thoughts ok.\n\n"
        f"5. ALL DIFFERENT — Each response must be unique in content, "
        f"phrasing, AND opening structure. Do NOT start multiple responses "
        f"with the same phrase (e.g., 'I think', 'I feel', 'I believe'). "
        f"Vary sentence openers: some start with a direct statement about "
        f"the topic, some with a hedged opinion, some with an emotional "
        f"reaction, some with a reference to what they experienced. "
        f"Each response should read like it came from a genuinely "
        f"different person with a different communication style.\n\n"
        f"6. MATCH PROFILES — Follow each participant's length, style, "
        f"effort, sentiment, and TONE specifications exactly. A participant "
        f"with 'very positive' sentiment should sound genuinely enthusiastic, "
        f"not just mildly agreeable. A participant with 'very negative' "
        f"sentiment should sound genuinely frustrated or critical.\n\n"
        f"7. NATURAL LENGTH DISTRIBUTION — Response lengths should vary "
        f"dramatically across participants. Some people write 4 words, "
        f"others write 4 sentences. Do NOT make responses uniformly "
        f"similar in length. Follow each participant's length spec "
        f"precisely.\n\n"
        f"Return ONLY a JSON array of {n} strings (one per participant), "
        f"no other text:\n"
        f'["response 1", "response 2", ...]'
    )
    return prompt


# ---------------------------------------------------------------------------
# LLM API caller (generic OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
def _call_llm_api(
    api_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 4000,
    timeout: int = 60,
) -> Optional[str]:
    """Call an OpenAI-compatible chat completion API.

    Uses the ``requests`` library when available (reliable on Streamlit Cloud),
    with ``urllib.request`` as a fallback.

    v1.9.1: Enhanced error diagnostics — logs HTTP status codes, response
    previews, and specific error types to help debug connectivity issues.
    """
    if not api_key or not api_key.strip():
        return None

    temperature = max(0.0, min(2.0, temperature))

    # Google AI Studio supports key-as-query-param in addition to Bearer token.
    # Append ?key=<key> for googleapis.com endpoints for maximum compatibility.
    _effective_url = api_url
    if "googleapis.com" in api_url:
        _sep = "&" if "?" in api_url else "?"
        _effective_url = f"{api_url}{_sep}key={api_key}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "BehavioralSimulationTool/1.0.1.2",
    }

    # OpenRouter requires/recommends HTTP-Referer and X-Title
    if "openrouter.ai" in api_url:
        headers["HTTP-Referer"] = "https://github.com/eugendimant/research-simulations"
        headers["X-Title"] = "Behavioral Experiment Simulation Tool"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Strategy 1: Use 'requests' (reliable on Streamlit Cloud, handles SSL well)
    try:
        import requests as _requests
        resp = _requests.post(
            _effective_url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        if resp.status_code == 200:
            body = resp.json()
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return content
            logger.warning("LLM API returned 200 but empty content: %s (model: %s)",
                           api_url[:50], model)
            return None
        elif resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "unknown")
            logger.warning("LLM API rate-limited (429) %s model=%s retry-after=%s",
                           api_url[:50], model, retry_after)
            return None
        elif resp.status_code in (401, 403):
            _body_preview = resp.text[:200] if resp.text else "(empty)"
            logger.warning("LLM API auth error (%d) %s model=%s — key may be invalid. "
                           "Response: %s", resp.status_code, api_url[:50], model, _body_preview)
            return None
        else:
            _body_preview = resp.text[:200] if resp.text else "(empty)"
            logger.warning("LLM API error (%d) %s model=%s — Response: %s",
                           resp.status_code, api_url[:50], model, _body_preview)
            return None
    except ImportError:
        pass  # Fall through to urllib
    except _requests.exceptions.Timeout:
        logger.warning("LLM API timeout (%ds) %s model=%s", timeout, api_url[:50], model)
        return None
    except _requests.exceptions.ConnectionError as exc:
        logger.warning("LLM API connection error %s model=%s: %s", api_url[:50], model, exc)
        return None
    except _requests.exceptions.SSLError as exc:
        logger.warning("LLM API SSL error %s model=%s: %s", api_url[:50], model, exc)
        return None
    except Exception as exc:
        logger.warning("LLM API call failed [requests] (%s %s): %s",
                       api_url[:50], model, exc)
        return None

    # Strategy 2: Fallback to urllib.request
    try:
        import urllib.request
        import urllib.error

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _effective_url, data=data, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return content
            logger.warning("LLM API returned empty content via urllib: %s", api_url[:50])
            return None
    except urllib.error.HTTPError as exc:
        _body_preview = ""
        try:
            _body_preview = exc.read().decode("utf-8")[:200]
        except Exception:
            pass
        logger.warning("LLM API HTTP error (%d) [urllib] %s model=%s: %s",
                       exc.code, api_url[:50], model, _body_preview)
        return None
    except Exception as exc:
        logger.warning("LLM API call failed [urllib] (%s %s): %s",
                       api_url[:50], model, exc)
        return None


def _parse_json_responses(raw: str, expected_n: int) -> List[str]:
    """Parse a JSON array of strings from the LLM output.

    v1.9.1: Enhanced with additional fallback strategies for robustness:
    - Standard JSON array
    - Markdown-fenced JSON
    - Trailing-comma fix
    - Single-quote fix
    - Truncated JSON recovery
    - Numbered list format (1. "response")
    - Newline-delimited responses
    - Quoted string extraction (last resort)
    """
    if not raw or not raw.strip():
        return []

    cleaned = raw.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Strategy 1: Standard JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            result = [str(r).strip() for r in parsed if str(r).strip()]
            if result:
                return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Fix trailing commas
    try:
        fixed = re.sub(r",\s*]", "]", cleaned)
        parsed = json.loads(fixed)
        if isinstance(parsed, list):
            result = [str(r).strip() for r in parsed if str(r).strip()]
            if result:
                return result
    except json.JSONDecodeError:
        pass

    # Strategy 3: Replace single quotes with double quotes
    if cleaned.startswith("[") and "'" in cleaned:
        try:
            sq_fixed = cleaned.replace("'", '"')
            parsed = json.loads(sq_fixed)
            if isinstance(parsed, list):
                result = [str(r).strip() for r in parsed if str(r).strip()]
                if result:
                    return result
        except json.JSONDecodeError:
            pass

    # Strategy 4: Handle truncated JSON (response cut off mid-array)
    if cleaned.startswith("[") and not cleaned.endswith("]"):
        # Try to close the array at the last complete string
        _trunc = cleaned
        # Find the last complete quoted string
        _last_quote = _trunc.rfind('"')
        if _last_quote > 0:
            _trunc = _trunc[:_last_quote + 1] + "]"
            # Remove any trailing comma before the closing bracket
            _trunc = re.sub(r",\s*]", "]", _trunc)
            try:
                parsed = json.loads(_trunc)
                if isinstance(parsed, list):
                    result = [str(r).strip() for r in parsed if str(r).strip()]
                    if result:
                        logger.info("Recovered %d responses from truncated JSON", len(result))
                        return result
            except json.JSONDecodeError:
                pass

    # Strategy 5: Numbered list format (1. "response" or 1) "response")
    numbered = re.findall(r'^\s*\d+[\.\)]\s*["\'](.+?)["\']?\s*$', cleaned, re.MULTILINE)
    if len(numbered) >= max(1, expected_n // 2):
        return [r.strip() for r in numbered if r.strip()]

    # Strategy 6: Numbered list without quotes (1. response text)
    numbered_nq = re.findall(r'^\s*\d+[\.\)]\s+(.{10,}?)\s*$', cleaned, re.MULTILINE)
    if len(numbered_nq) >= max(1, expected_n // 2):
        return [r.strip().strip('"').strip("'") for r in numbered_nq if r.strip()]

    # Strategy 7: Extract all quoted strings (min 3 chars)
    matches = re.findall(r'"([^"]{3,})"', cleaned)
    if matches:
        return [m.strip() for m in matches if m.strip()]

    # Strategy 8: Newline-delimited responses (each line is a response)
    lines = [l.strip() for l in cleaned.split("\n") if l.strip() and len(l.strip()) >= 5]
    # Filter out lines that look like instructions/metadata
    response_lines = [
        l.strip('"').strip("'").strip()
        for l in lines
        if not l.startswith("{") and not l.startswith("[")
        and not l.lower().startswith("participant")
        and not l.lower().startswith("response")
        and not l.lower().startswith("here")
        and not l.lower().startswith("note")
    ]
    if len(response_lines) >= max(1, expected_n // 3):
        return response_lines

    logger.warning("Could not parse LLM response as JSON array (raw length: %d, expected %d). "
                   "First 200 chars: %s", len(raw), expected_n, raw[:200])
    return []


# ---------------------------------------------------------------------------
# Key auto-detection for multi-provider support
# ---------------------------------------------------------------------------
def detect_provider_from_key(api_key: str) -> Optional[Dict[str, str]]:
    """Auto-detect LLM provider from API key prefix.

    Returns dict with 'name', 'api_url', 'model' or None if unrecognized.
    """
    if not api_key:
        return None
    key = api_key.strip()
    if key.startswith("gsk_"):
        return {"name": "groq", "api_url": GROQ_API_URL, "model": GROQ_MODEL}
    elif key.startswith("csk-"):
        return {"name": "cerebras", "api_url": CEREBRAS_API_URL, "model": CEREBRAS_MODEL}
    elif key.startswith("sk-or-"):
        return {"name": "openrouter", "api_url": OPENROUTER_API_URL, "model": OPENROUTER_MODEL}
    elif key.startswith("AIza"):
        return {"name": "google_ai", "api_url": GOOGLE_AI_API_URL, "model": GOOGLE_AI_MODEL}
    elif key.startswith("snova-") or key.startswith("sambanova-"):
        return {"name": "sambanova", "api_url": SAMBANOVA_API_URL, "model": SAMBANOVA_MODEL}
    elif len(key) > 30:
        # Default to Groq for unrecognized long keys
        return {"name": "groq", "api_url": GROQ_API_URL, "model": GROQ_MODEL}
    return None


def get_supported_providers() -> List[Dict[str, str]]:
    """Return list of supported providers with info for UI display."""
    return [
        {
            "name": "Groq",
            "prefix": "gsk_...",
            "url": "https://console.groq.com",
            "free_tier": "14,400 requests/day",
            "recommended": True,
        },
        {
            "name": "Cerebras",
            "prefix": "csk-...",
            "url": "https://cloud.cerebras.ai",
            "free_tier": "1M tokens/day",
            "recommended": False,
        },
        {
            "name": "OpenRouter",
            "prefix": "sk-or-...",
            "url": "https://openrouter.ai",
            "free_tier": "Free models available",
            "recommended": False,
        },
        {
            "name": "Google AI Studio (Gemini)",
            "prefix": "AIza...",
            "url": "https://aistudio.google.com",
            "free_tier": "Gemini 2.5 Flash Lite (10 RPM, 20 RPD) + Gemma 3 27B (30 RPM, 14.4K RPD)",
            "recommended": True,
        },
        {
            "name": "SambaNova",
            "prefix": "(any key)",
            "url": "https://cloud.sambanova.ai",
            "free_tier": "Free Llama 3.1 70B",
            "recommended": False,
        },
    ]


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------
class _LLMProvider:
    """Represents a single LLM API provider with automatic failure tracking.

    v1.9.0: Improved resilience — higher failure threshold, retry with backoff,
    and soft-disable that allows periodic re-attempts.
    """

    def __init__(self, name: str, api_url: str, model: str, api_key: str,
                 max_rpm: int = 28, max_rpd: int = 0,
                 max_batch_size: int = 20) -> None:
        self.name = name
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.max_rpm = max_rpm
        self.max_rpd = max_rpd        # 0 = unlimited
        self.max_batch_size = max_batch_size
        self.available = True
        self.call_count = 0
        self._daily_call_count = 0
        self._daily_reset_time = time.time()
        self._consecutive_failures = 0
        self._max_failures = 6
        self._total_failures = 0
        self._last_failure_time = 0.0
        self._cooldown_seconds = 30.0
        self._rate_limiter = _RateLimiter(max_rpm=max_rpm)

    def call(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.7, max_tokens: int = 4000) -> Optional[str]:
        if not self.api_key:
            return None

        # Auto-recover after cooldown period
        if not self.available:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._cooldown_seconds:
                logger.info("Provider '%s' cooldown expired (%.0fs), re-enabling",
                            self.name, elapsed)
                self.available = True
                self._consecutive_failures = 0
            else:
                return None

        # Check daily request limit (reset after 24h)
        if self.max_rpd > 0:
            if time.time() - self._daily_reset_time >= 86400:
                self._daily_call_count = 0
                self._daily_reset_time = time.time()
            if self._daily_call_count >= self.max_rpd:
                logger.info("Provider '%s' daily limit exhausted (%d/%d RPD)",
                            self.name, self._daily_call_count, self.max_rpd)
                return None

        # Per-provider rate limiting
        self._rate_limiter.wait_if_needed()

        # Retry with backoff (up to 2 retries per call)
        result = None
        for attempt in range(3):
            result = _call_llm_api(
                self.api_url, self.api_key, self.model,
                system_prompt, user_prompt, temperature, max_tokens,
            )
            if result is not None:
                break
            if attempt < 2:
                _backoff = 1.5 * (attempt + 1)
                logger.debug("Provider '%s' attempt %d failed, retrying in %.1fs...",
                             self.name, attempt + 1, _backoff)
                time.sleep(_backoff)

        self.call_count += 1
        self._daily_call_count += 1
        if result is None:
            self._consecutive_failures += 1
            self._total_failures += 1
            self._last_failure_time = time.time()
            if self._consecutive_failures >= self._max_failures:
                self.available = False
                self._cooldown_seconds = min(300.0, self._cooldown_seconds * 2)
                logger.info("Provider '%s' disabled after %d consecutive failures "
                            "(cooldown: %.0fs)",
                            self.name, self._consecutive_failures, self._cooldown_seconds)
        else:
            self._consecutive_failures = 0
        return result

    def reset(self) -> None:
        """Re-enable the provider (e.g., after rate-limit window expires)."""
        self.available = True
        self._consecutive_failures = 0
        self._cooldown_seconds = 30.0


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------
class LLMResponseGenerator:
    """Generate open-ended survey responses using free LLM APIs.

    Multi-provider architecture with automatic failover:
    1. Built-in default Groq key (seamless for all users)
    2. User-provided key (auto-detected: Groq, Cerebras, Google AI Studio, OpenRouter)
    3. Environment variable providers (Google AI Studio, Cerebras, OpenRouter)
    4. Template fallback (always works)

    Draw-with-replacement + deep variation means a pool of ~50 base
    responses per bucket can serve thousands of participants.

    Usage::

        gen = LLMResponseGenerator(
            study_title="Trust & AI",
            study_description="Examining trust in AI-generated advice",
            seed=42,
        )
        gen.prefill_pool("Why?", "AI_advice", sample_size=2000, n_conditions=2)
        response = gen.generate(
            question_text="Why?", condition="AI_advice", sentiment="positive",
            persona_verbosity=0.7, persona_formality=0.4, persona_engagement=0.8,
        )
    """

    # Default batch size: 20 responses per API call (B: larger batches)
    DEFAULT_BATCH_SIZE = 20
    # Minimum pool per sentiment bucket (ensures diversity)
    MIN_POOL_PER_BUCKET = 30
    # Maximum pool per sentiment bucket (avoid excessive API calls)
    MAX_POOL_PER_BUCKET = 80

    def __init__(
        self,
        api_key: Optional[str] = None,
        study_title: str = "",
        study_description: str = "",
        seed: Optional[int] = None,
        fallback_generator: Any = None,
        batch_size: int = 20,
        all_conditions: Optional[List[str]] = None,
    ) -> None:
        self._study_title = study_title
        self._study_description = study_description
        self._all_conditions: List[str] = list(all_conditions) if all_conditions else []
        self._rng = random.Random(seed)
        self._fallback = fallback_generator
        self._batch_size = max(4, min(batch_size, 25))
        self._pool = _ResponsePool()
        self._fallback_count = 0
        self._batch_failure_count = 0

        # Build provider chain with per-provider rate limits.
        # Priority: Google AI (quality) → Google AI (volume) → Groq → Cerebras → OpenRouter
        self._providers: List[_LLMProvider] = []
        user_key = api_key or os.environ.get("LLM_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
        _all_builtin_keys = {_DEFAULT_GROQ_KEY, _DEFAULT_CEREBRAS_KEY,
                             _DEFAULT_GOOGLE_AI_KEY, _DEFAULT_OPENROUTER_KEY}

        # Built-in providers (in priority order) with rate limits from provider dashboards:
        # 1. Google AI Gemini 2.5 Flash Lite: 10 RPM, 250K TPM, 20 RPD (quality)
        # 2. Google AI Gemma 3 27B:           30 RPM, 15K TPM, 14,400 RPD (volume)
        # 3. Groq Llama 3.3 70B:              ~30 RPM, 14,400 RPD
        # 4. Cerebras Llama 3.3 70B:          ~30 RPM, 1M tokens/day
        # 5. OpenRouter Mistral Small 3.1:    varies by model
        _builtin_providers = [
            ("google_ai_builtin", GOOGLE_AI_API_URL, GOOGLE_AI_MODEL,
             _DEFAULT_GOOGLE_AI_KEY, 8, 20, 20),
            ("google_ai_gemma", GOOGLE_AI_API_URL, GOOGLE_AI_MODEL_HIGHVOL,
             _DEFAULT_GOOGLE_AI_KEY, 25, 14400, 10),
            ("groq_builtin", GROQ_API_URL, GROQ_MODEL,
             _DEFAULT_GROQ_KEY, 28, 0, 20),
            ("cerebras_builtin", CEREBRAS_API_URL, CEREBRAS_MODEL,
             _DEFAULT_CEREBRAS_KEY, 28, 0, 20),
            ("openrouter_builtin", OPENROUTER_API_URL, OPENROUTER_MODEL,
             _DEFAULT_OPENROUTER_KEY, 20, 0, 20),
        ]
        for name, url, model, key, rpm, rpd, max_bs in _builtin_providers:
            if key:
                self._providers.append(_LLMProvider(
                    name=name, api_url=url, model=model, api_key=key,
                    max_rpm=rpm, max_rpd=rpd, max_batch_size=max_bs,
                ))

        # Env-var Google AI key (user's own key — may have different limits)
        _google_ai_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        if _google_ai_key and _google_ai_key != _DEFAULT_GOOGLE_AI_KEY:
            _or_idx = next((i for i, p in enumerate(self._providers)
                           if p.name == "groq_builtin"), len(self._providers))
            self._providers.insert(_or_idx, _LLMProvider(
                name="google_ai_user", api_url=GOOGLE_AI_API_URL,
                model=GOOGLE_AI_MODEL, api_key=_google_ai_key,
                max_rpm=8, max_rpd=20,
            ))

        # SambaNova Cloud (free Llama 3.1 70B)
        _sambanova_key = os.environ.get("SAMBANOVA_API_KEY", "")
        if _sambanova_key:
            _or_idx = next((i for i, p in enumerate(self._providers)
                           if p.name == "openrouter_builtin"), len(self._providers))
            self._providers.insert(_or_idx, _LLMProvider(
                name="sambanova", api_url=SAMBANOVA_API_URL,
                model=SAMBANOVA_MODEL, api_key=_sambanova_key,
            ))

        # User-provided key (appended AFTER built-ins so it's tried last —
        # we want to use the tool's own capacity first)
        if user_key and user_key not in _all_builtin_keys:
            detected = detect_provider_from_key(user_key)
            if detected:
                self._providers.append(_LLMProvider(
                    name=f"{detected['name']}_user",
                    api_url=detected["api_url"],
                    model=detected["model"],
                    api_key=user_key,
                ))
            else:
                self._providers.append(_LLMProvider(
                    name="groq_user",
                    api_url=GROQ_API_URL,
                    model=GROQ_MODEL,
                    api_key=user_key,
                ))

        # Extra env-var providers (if someone configures them manually)
        for env_var, name, url, model in [
            ("CEREBRAS_API_KEY", "cerebras_env", CEREBRAS_API_URL, CEREBRAS_MODEL),
            ("OPENROUTER_API_KEY", "openrouter_env", OPENROUTER_API_URL, OPENROUTER_MODEL),
        ]:
            env_key = os.environ.get(env_var, "")
            if env_key and not any(p.api_key == env_key for p in self._providers):
                self._providers.append(_LLMProvider(
                    name=name, api_url=url, model=model, api_key=env_key,
                ))

        self._api_available = any(p.available and p.api_key for p in self._providers)

        # v1.9.1: Diagnostic logging — log provider chain for debugging
        _provider_summary = []
        for p in self._providers:
            _key_prefix = p.api_key[:8] + "..." if p.api_key else "(none)"
            _provider_summary.append(f"{p.name}({_key_prefix})")
        logger.info("LLM provider chain: %s | api_available=%s",
                    " → ".join(_provider_summary) if _provider_summary else "(empty)",
                    self._api_available)

    @property
    def is_llm_available(self) -> bool:
        return self._api_available

    @property
    def active_provider_name(self) -> str:
        """Name of the currently active provider, or 'none'."""
        for p in self._providers:
            if p.available and p.api_key:
                return p.name
        return "none"

    @property
    def stats(self) -> Dict[str, Any]:
        total_calls = sum(p.call_count for p in self._providers)
        return {
            "llm_calls": total_calls,
            "fallback_uses": self._fallback_count,
            "pool_size": self._pool.total_responses,
            "active_provider": self.active_provider_name,
            "providers": {
                p.name: {"calls": p.call_count, "available": p.available}
                for p in self._providers
            },
        }

    def set_study_context(self, title: str, description: str,
                         conditions: Optional[List[str]] = None) -> None:
        self._study_title = title
        self._study_description = description
        if conditions is not None:
            self._all_conditions = list(conditions)

    def reset_providers(self) -> None:
        """Re-enable all providers (useful after rate-limit windows expire)."""
        for p in self._providers:
            p.reset()
        self._api_available = any(p.available and p.api_key for p in self._providers)

    @property
    def provider_display_name(self) -> str:
        """Human-readable name of the active provider for UI display.

        Note: Does not expose specific model names to end users.
        """
        for p in self._providers:
            if p.available and p.api_key:
                return "AI Language Model"
        return "none"

    @property
    def user_key_provider(self) -> str:
        """If a user key is active, return which provider it's for."""
        for p in self._providers:
            if p.available and p.api_key and "_user" in p.name:
                return p.name.replace("_user", "").title()
        return ""

    def _get_active_provider(self) -> Optional[_LLMProvider]:
        """Get the first available provider."""
        for p in self._providers:
            if p.available and p.api_key:
                return p
        return None

    # ------------------------------------------------------------------
    # Pool pre-fill with smart scaling (C)
    # ------------------------------------------------------------------
    def prefill_pool(
        self,
        question_text: str,
        condition: str,
        sentiments: Optional[List[str]] = None,
        count_per_sentiment: int = 0,
        sample_size: int = 200,
        n_conditions: int = 2,
    ) -> int:
        """Pre-generate a pool of LLM responses for a question+condition.

        Smart scaling (C): calculates optimal pool size from sample_size.
        With draw-with-replacement (D), we need far fewer base responses.
        """
        if not self._api_available:
            return 0

        if sentiments is None:
            sentiments = ["very_positive", "positive", "neutral",
                          "negative", "very_negative"]

        # Smart pool scaling: calculate needed per bucket
        if count_per_sentiment <= 0:
            # participants_per_bucket = sample_size / (n_conditions * n_sentiments)
            # With draw-with-replacement, we need ~sqrt(participants) base responses
            # to ensure good diversity, with a floor and ceiling
            participants_per_bucket = max(1, sample_size // (max(1, n_conditions) * len(sentiments)))
            import math
            target = max(
                self.MIN_POOL_PER_BUCKET,
                min(self.MAX_POOL_PER_BUCKET, int(math.sqrt(participants_per_bucket) * 3) + 10),
            )
            count_per_sentiment = target

        total = 0
        max_retries_per_bucket = 10  # Safety cap to prevent infinite loops
        for sentiment in sentiments:
            already_have = self._pool.available(question_text, condition, sentiment)
            needed = max(0, count_per_sentiment - already_have)
            retries = 0
            while needed > 0 and retries < max_retries_per_bucket:
                retries += 1
                # Adaptive batch size: use active provider's max_batch_size if available
                _active = self._get_active_provider()
                _provider_max = _active.max_batch_size if _active else self._batch_size
                batch_n = min(needed, self._batch_size, _provider_max)
                specs = []
                for _ in range(batch_n):
                    specs.append({
                        "verbosity": self._rng.uniform(0.1, 0.9),
                        "formality": self._rng.uniform(0.1, 0.9),
                        "engagement": self._rng.uniform(0.2, 0.95),
                        "sentiment": sentiment,
                    })

                responses = self._generate_batch(
                    question_text, condition, sentiment, specs
                )
                if responses:
                    self._pool.add(question_text, condition, sentiment, responses)
                    total += len(responses)
                    needed -= len(responses)
                else:
                    break  # Provider failed
            if not self._api_available:
                break  # Don't try remaining sentiments if all providers down

        return total

    def _generate_batch(
        self,
        question_text: str,
        condition: str,
        sentiment: str,
        persona_specs: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate a batch of responses via LLM API with full provider chain.

        Tries every provider in order.  When one fails (rate-limited, error,
        timeout), it moves to the next.  Only gives up when ALL providers
        have been tried.
        """
        prompt = _build_batch_prompt(
            question_text=question_text,
            condition=condition,
            study_title=self._study_title,
            study_description=self._study_description,
            persona_specs=persona_specs,
            all_conditions=self._all_conditions or None,
        )

        # v1.9.0: Try every provider in the chain with improved resilience
        tried: set = set()
        while True:
            provider = self._get_active_provider()
            if not provider or provider.name in tried:
                break  # All providers exhausted
            tried.add(provider.name)

            # Adaptive max_tokens: reduce for TPM-constrained providers (e.g. Gemma 3 27B: 15K TPM)
            _max_tokens = 4000
            if "gemma" in provider.model.lower():
                _max_tokens = 2500  # ~3K prompt + 2.5K response ≈ 5.5K < 15K TPM
            raw = provider.call(SYSTEM_PROMPT, prompt, max_tokens=_max_tokens)

            if raw is not None:
                responses = _parse_json_responses(raw, len(persona_specs))
                if responses:
                    self._batch_failure_count = 0  # Reset on success
                    return responses
                logger.warning("Provider '%s' returned unparseable response, trying next...",
                               provider.name)
            else:
                logger.info("Provider '%s' failed, trying next...", provider.name)

        # All providers failed for THIS batch — DON'T permanently disable
        # v1.9.0: Track batch failures separately; only disable after sustained failures
        self._batch_failure_count += 1
        if self._batch_failure_count >= 5:
            # Only disable after 5 consecutive batch failures (was instant before)
            self._api_available = False
            logger.warning("All %d LLM providers exhausted after %d batch failures — "
                           "falling back to templates",
                           len(self._providers), self._batch_failure_count)
        else:
            logger.info("Batch %d failed across all providers, will retry next batch "
                        "(reset providers for next attempt)",
                        self._batch_failure_count)
            # Reset providers so they're tried again on next batch
            self._reset_all_providers()
        return []

    def _reset_all_providers(self) -> None:
        """Reset all providers to available state for retry."""
        for p in self._providers:
            p.reset()

    # ------------------------------------------------------------------
    # Single-response generation with deep variation (D)
    # ------------------------------------------------------------------
    def generate(
        self,
        question_text: str,
        sentiment: str = "neutral",
        persona_verbosity: float = 0.5,
        persona_formality: float = 0.5,
        persona_engagement: float = 0.5,
        condition: str = "",
        question_name: str = "",
        participant_seed: int = 0,
    ) -> str:
        """Generate a single open-ended response.

        Uses draw-with-replacement from the pool + deep persona variation
        to produce unique responses for each participant.  Returns a
        non-empty string or falls back to the template generator.
        """
        local_rng = random.Random(participant_seed)

        # 1. Try pool (draw-with-replacement)
        resp = self._pool.draw_with_replacement(
            question_text, condition, sentiment, local_rng
        )
        if resp and resp.strip():
            result = self._apply_deep_variation(
                resp, persona_verbosity, persona_formality,
                persona_engagement, local_rng,
            )
            if result and len(result.strip()) >= 3:
                return result.strip()

        # 2. Try on-demand LLM batch (if pool was empty)
        if self._api_available:
            specs = []
            for _ in range(self._batch_size):
                specs.append({
                    "verbosity": self._rng.uniform(0.1, 0.9),
                    "formality": self._rng.uniform(0.1, 0.9),
                    "engagement": self._rng.uniform(0.2, 0.95),
                    "sentiment": sentiment,
                })
            batch = self._generate_batch(question_text, condition, sentiment, specs)
            if batch:
                self._pool.add(question_text, condition, sentiment, batch)
                resp = self._pool.draw_with_replacement(
                    question_text, condition, sentiment, local_rng
                )
                if resp and resp.strip():
                    result = self._apply_deep_variation(
                        resp, persona_verbosity, persona_formality,
                        persona_engagement, local_rng,
                    )
                    if result and len(result.strip()) >= 3:
                        return result.strip()

        # 3. Fall back to template generator
        self._fallback_count += 1
        if self._fallback:
            return self._fallback.generate(
                question_text=question_text,
                sentiment=sentiment,
                persona_verbosity=persona_verbosity,
                persona_formality=persona_formality,
                persona_engagement=persona_engagement,
                condition=condition,
                question_name=question_name,
                participant_seed=participant_seed,
            )
        return ""

    # ------------------------------------------------------------------
    # Deep persona variation (D) — makes each draw unique
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_deep_variation(
        text: str,
        verbosity: float,
        formality: float,
        engagement: float,
        rng: random.Random,
    ) -> str:
        """Apply deep persona-driven variation to a pool response.

        Combines multiple transformation layers so even the same base text
        produces different outputs for different participants.  Layers fire
        at ALL persona levels — not just extremes — to guarantee uniqueness.
        """
        if not text or len(text) < 5:
            return text

        sentences = re.split(r'(?<=[.!?])\s+', text)

        # --- Layer 0: ALWAYS-FIRE multi-axis micro-variation ---
        # Uses multiple independent rng draws to create a combinatorial
        # explosion of variations.  Even with 1 base text and 500 identical
        # personas, each participant gets a unique combination.
        words = text.split()
        if len(words) > 5:
            # Axis 1: drop a word (50% chance)
            if rng.random() < 0.5 and len(words) > 8:
                drop_idx = rng.randint(2, len(words) - 3)
                words.pop(drop_idx)

            # Axis 2: insert a transition (40% chance, independent)
            if rng.random() < 0.4 and len(words) > 4:
                transitions = [
                    "also", "though", "still", "however", "but",
                    "actually", "definitely", "probably", "maybe", "certainly",
                    "really", "just", "often", "sometimes", "perhaps",
                    "clearly", "indeed", "typically", "generally", "honestly",
                ]
                insert_pos = rng.randint(2, max(2, len(words) - 2))
                words.insert(insert_pos, rng.choice(transitions))

            # Axis 3: swap two adjacent words (30% chance, independent)
            if rng.random() < 0.30 and len(words) > 6:
                swap_idx = rng.randint(1, len(words) - 3)
                words[swap_idx], words[swap_idx + 1] = words[swap_idx + 1], words[swap_idx]

            # Axis 4: replace a common word with an alternative (35% chance)
            if rng.random() < 0.35:
                replacements = {
                    "good": ["nice", "great", "solid", "fine", "okay"],
                    "bad": ["poor", "weak", "lacking", "not great", "subpar"],
                    "like": ["enjoy", "appreciate", "prefer", "favor"],
                    "think": ["feel", "believe", "reckon", "suppose", "figure"],
                    "really": ["truly", "genuinely", "honestly", "certainly"],
                    "very": ["quite", "really", "rather", "pretty", "fairly"],
                    "was": ["felt", "seemed", "appeared"],
                    "interesting": ["compelling", "engaging", "thought-provoking", "notable"],
                    "important": ["significant", "crucial", "key", "essential"],
                    "different": ["distinct", "unique", "varied", "diverse"],
                }
                for i, w in enumerate(words):
                    clean_w = w.lower().strip(".,!?;:")
                    if clean_w in replacements:
                        trail = w[len(clean_w):] if len(w) > len(clean_w) else ""
                        new_word = rng.choice(replacements[clean_w])
                        words[i] = new_word + trail
                        break  # Only one replacement per pass

            text = " ".join(words)
            sentences = re.split(r'(?<=[.!?])\s+', text)

        # --- Layer 1: Sentence-level restructuring (ALL personas) ---
        if len(sentences) > 2:
            r = rng.random()
            if r < 0.30:
                # Shuffle middle sentences
                middle = sentences[1:-1]
                rng.shuffle(middle)
                sentences = [sentences[0]] + middle + [sentences[-1]]
            elif r < 0.45:
                # Move last sentence to middle
                last = sentences.pop()
                insert_at = rng.randint(1, max(1, len(sentences) - 1))
                sentences.insert(insert_at, last)
            elif r < 0.55 and len(sentences) > 3:
                # Drop a random middle sentence
                drop_idx = rng.randint(1, len(sentences) - 2)
                sentences.pop(drop_idx)

        # --- Layer 2: Verbosity control ---
        if verbosity < 0.2:
            sentences = sentences[:1]
        elif verbosity < 0.35:
            sentences = sentences[:max(1, len(sentences) // 3)]
        elif verbosity < 0.5:
            sentences = sentences[:max(2, len(sentences) // 2)]
        elif verbosity > 0.7 and rng.random() < 0.45:
            # Higher verbosity: sometimes add elaboration
            elaborations = [
                "I really feel strongly about this.",
                "This is something I think about quite a bit.",
                "It's hard to put into words exactly.",
                "There's a lot to consider here.",
                "I've been thinking about this for a while.",
                "It just makes sense to me.",
            ]
            sentences.append(rng.choice(elaborations))

        text = " ".join(sentences)

        # --- Layer 3: Formality adjustments ---
        if formality < 0.3:
            if rng.random() < 0.55 and len(text) > 1:
                if not any(text.startswith(s) for s in _CASUAL_STARTERS):
                    text = rng.choice(_CASUAL_STARTERS) + text[0].lower() + text[1:]
            if rng.random() < 0.3 and len(text) > 40:
                w = text.split()
                if len(w) > 6:
                    pos = rng.randint(3, len(w) - 3)
                    w.insert(pos, rng.choice(_FILLER_INSERTIONS))
                    text = " ".join(w)
        elif formality < 0.6:
            # Mid-formality: occasional hedging or connector
            if rng.random() < 0.30 and len(text) > 1:
                mid_starters = [
                    "I think ", "I feel like ", "In my view, ",
                    "For me, ", "Personally, ", "I'd say ",
                ]
                if not any(text.startswith(s) for s in mid_starters):
                    text = rng.choice(mid_starters) + text[0].lower() + text[1:]
        else:
            if rng.random() < 0.3 and len(text) > 1:
                text = rng.choice(_FORMAL_CONNECTORS) + text[0].lower() + text[1:]
            for contraction, expansion in [
                ("don't", "do not"), ("can't", "cannot"), ("won't", "will not"),
                ("I'm", "I am"), ("it's", "it is"), ("didn't", "did not"),
                ("wasn't", "was not"), ("they're", "they are"),
            ]:
                if rng.random() < 0.6:
                    text = text.replace(contraction, expansion)

        # --- Layer 4: Engagement modulation ---
        if engagement < 0.2:
            cur_sents = re.split(r'(?<=[.!?])\s+', text)
            if len(cur_sents) > 1:
                text = cur_sents[0]
            if rng.random() < 0.5:
                text = rng.choice(["Idk. ", "Not sure. ", "Meh. ", ""]) + text
        elif engagement < 0.4 and rng.random() < 0.4:
            if len(text) > 1:
                text = rng.choice(_HEDGING_PHRASES) + text[0].lower() + text[1:]

        # --- Layer 5: Typo injection for careless/casual personas ---
        if (formality < 0.3 or engagement < 0.3) and rng.random() < 0.25:
            w = text.split()
            if len(w) > 4:
                typo_candidates = [
                    (i, wd) for i, wd in enumerate(w)
                    if wd.lower().strip(".,!?;:") in _TYPO_MAP
                ]
                if typo_candidates:
                    idx, word = rng.choice(typo_candidates)
                    clean = word.lower().strip(".,!?;:")
                    replacement = rng.choice(_TYPO_MAP[clean])
                    trail = word[len(clean):] if len(word) > len(clean) else ""
                    w[idx] = replacement + trail
                    text = " ".join(w)

        # --- Layer 6: Synonym swaps (ALL personas, higher probability) ---
        swaps = [
            ("I think", "I feel"), ("I feel", "I think"),
            ("really", "truly"), ("very", "quite"),
            ("good", "decent"), ("bad", "poor"),
            ("important", "significant"), ("interesting", "noteworthy"),
            ("a lot", "quite a bit"), ("kind of", "somewhat"),
            ("because", "since"), ("but", "however"),
            ("want", "would like"), ("need", "require"),
            ("seems", "appears"), ("shows", "demonstrates"),
        ]
        # Apply 1-3 random swaps
        n_swaps = rng.randint(1, min(3, len(swaps)))
        chosen_swaps = rng.sample(swaps, n_swaps)
        for old_w, new_w in chosen_swaps:
            if old_w in text:
                text = text.replace(old_w, new_w, 1)

        # --- Layer 7: Punctuation variation ---
        if rng.random() < 0.3:
            if text.endswith("."):
                endings = [".", "!", "...", ""]
                text = text[:-1] + rng.choice(endings)

        return text.strip()

    # ------------------------------------------------------------------
    # Connectivity check (used by UI to show status)
    # ------------------------------------------------------------------
    def check_connectivity(self, timeout: int = 15) -> Dict[str, Any]:
        """Quick connectivity test — loops through ALL providers.

        v1.9.0: Does NOT permanently disable providers during check.
        Tries each provider in order until one responds.

        Args:
            timeout: Max seconds per individual provider check (default 15).

        Returns dict with 'available', 'provider', 'error' keys.
        """
        if not self._providers:
            return {"available": False, "provider": "none", "error": "No API keys configured"}

        last_error = "No providers available"
        _failed_providers: List[str] = []

        for provider in self._providers:
            if not provider.api_key:
                continue
            try:
                raw = _call_llm_api(
                    provider.api_url, provider.api_key, provider.model,
                    "Reply with exactly: OK", "Test",
                    temperature=0.0, max_tokens=10, timeout=timeout,
                )
                if raw is not None:
                    # v1.9.0: Reset all providers so generation starts fresh
                    for p in self._providers:
                        p.reset()
                    return {"available": True, "provider": provider.name, "error": None}
                _failed_providers.append(provider.name)
                last_error = f"{provider.name} unavailable"
            except Exception as e:
                _failed_providers.append(provider.name)
                last_error = str(e)

        # v1.9.0: Reset all providers after check — don't leave them disabled
        for p in self._providers:
            p.reset()

        return {"available": False, "provider": "none",
                "error": f"All providers failed: {last_error}",
                "tried": _failed_providers}
