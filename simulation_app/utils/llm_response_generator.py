"""
LLM-powered open-ended survey response generator.

Uses free LLM APIs (Groq primary, with template fallback) to generate
realistic, question-specific, persona-aligned open-ended survey responses.

Key design decisions:
- Batch-in-prompt: generates 10-15 responses per API call (not 1 per participant)
- Response pool: LLM-generated responses are cached and distributed to participants
- Graceful fallback: if the LLM is unavailable, falls back to the existing
  template-based ComprehensiveResponseGenerator silently
- Persona-aware prompts: verbosity, formality, engagement, sentiment all
  influence the generated text

Version: 1.4.7
"""

__version__ = "1.4.7"

import hashlib
import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# System prompt instructs the LLM to act as a survey participant simulator
SYSTEM_PROMPT = (
    "You are simulating survey participants for behavioral science research. "
    "You generate realistic open-ended survey responses that mimic real human "
    "participants.  Each response should sound natural, contain typical human "
    "imperfections (hedging, incomplete thoughts, colloquialisms where appropriate), "
    "and vary in quality and detail based on the participant profile provided.\n\n"
    "IMPORTANT: Responses must be realistic survey responses, NOT polished essays.  "
    "Real participants write informally, sometimes go off-topic slightly, and vary "
    "significantly in effort and detail.  Careless participants give very short, "
    "generic answers.  Engaged participants give thoughtful, specific answers."
)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Simple token-bucket rate limiter (requests per minute)."""

    def __init__(self, max_rpm: int = 25):
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
# Response pool / cache
# ---------------------------------------------------------------------------
class _ResponsePool:
    """Pool of pre-generated responses keyed by (question, condition, sentiment)."""

    def __init__(self) -> None:
        self._pools: Dict[str, List[str]] = {}

    @staticmethod
    def _key(question_text: str, condition: str, sentiment: str) -> str:
        raw = f"{question_text[:200]}|{condition}|{sentiment}"
        return hashlib.md5(raw.encode()).hexdigest()

    def add(self, question_text: str, condition: str, sentiment: str,
            responses: List[str]) -> None:
        k = self._key(question_text, condition, sentiment)
        self._pools.setdefault(k, []).extend(responses)

    def draw(self, question_text: str, condition: str, sentiment: str,
             rng: random.Random) -> Optional[str]:
        k = self._key(question_text, condition, sentiment)
        pool = self._pools.get(k)
        if not pool:
            return None
        idx = rng.randint(0, len(pool) - 1)
        return pool.pop(idx)

    def available(self, question_text: str, condition: str, sentiment: str) -> int:
        k = self._key(question_text, condition, sentiment)
        return len(self._pools.get(k, []))

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


def _build_batch_prompt(
    question_text: str,
    condition: str,
    study_title: str,
    study_description: str,
    persona_specs: List[Dict[str, Any]],
) -> str:
    """Build a single prompt that asks the LLM to generate N responses at once."""
    n = len(persona_specs)

    participant_lines = []
    for i, spec in enumerate(persona_specs, 1):
        v = spec.get("verbosity", 0.5)
        f = spec.get("formality", 0.5)
        e = spec.get("engagement", 0.5)
        s = spec.get("sentiment", "neutral")

        # Translate numeric traits to plain-English instructions
        length_hint = (
            "1-2 short sentences" if v < 0.3
            else "2-4 sentences" if v < 0.7
            else "4-8 detailed sentences"
        )
        style_hint = (
            "very casual, uses contractions and slang" if f < 0.3
            else "conversational, mostly standard English" if f < 0.7
            else "formal, proper grammar, academic tone"
        )
        effort_hint = (
            "minimal effort, vague, possibly off-topic" if e < 0.3
            else "moderate effort, mostly on-topic" if e < 0.7
            else "thoughtful, specific, clearly on-topic"
        )
        sentiment_hint = _sentiment_label(s)

        participant_lines.append(
            f"Participant {i}: length={length_hint} | "
            f"style={style_hint} | effort={effort_hint} | "
            f"sentiment={sentiment_hint}"
        )

    participants_block = "\n".join(participant_lines)

    prompt = (
        f'Study: "{study_title}"\n'
        f"Context: {study_description[:300]}\n"
        f"Experimental condition: {condition}\n\n"
        f'Survey question: "{question_text}"\n\n'
        f"Generate exactly {n} unique responses from {n} different participants.\n"
        f"Each participant's profile controls their response style:\n\n"
        f"{participants_block}\n\n"
        f"Rules:\n"
        f"- Each response MUST be different from every other response.\n"
        f"- Responses should reference the specific experimental context "
        f"and condition when relevant.\n"
        f"- Do NOT use bullet points, numbered lists, or markdown formatting "
        f"inside responses — just plain text as a survey participant would write.\n"
        f"- Match each participant's length, style, effort, and sentiment exactly.\n\n"
        f"Return ONLY a JSON array of {n} strings (one per participant), "
        f"no other text:\n"
        f'["response 1", "response 2", ...]'
    )
    return prompt


# ---------------------------------------------------------------------------
# LLM API caller
# ---------------------------------------------------------------------------
def _call_groq(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 3000,
) -> Optional[str]:
    """Call the Groq chat completion API. Returns the response text or None."""
    try:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(
            GROQ_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]

    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.warning("Groq API HTTP %s: %s", e.code, error_body)
        return None
    except Exception as exc:
        logger.warning("Groq API call failed: %s", exc)
        return None


def _parse_json_responses(raw: str, expected_n: int) -> List[str]:
    """Parse a JSON array of strings from the LLM output.

    Handles common LLM quirks: markdown code fences, trailing commas, etc.
    """
    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Try standard JSON parse first
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(r).strip() for r in parsed if str(r).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: try to fix trailing commas
    try:
        fixed = re.sub(r",\s*]", "]", cleaned)
        parsed = json.loads(fixed)
        if isinstance(parsed, list):
            return [str(r).strip() for r in parsed if str(r).strip()]
    except json.JSONDecodeError:
        pass

    # Last resort: extract quoted strings
    matches = re.findall(r'"([^"]{10,})"', cleaned)
    if matches:
        return [m.strip() for m in matches]

    logger.warning("Could not parse LLM response as JSON array")
    return []


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------
class LLMResponseGenerator:
    """Generate open-ended survey responses using a free LLM API.

    Designed to be used alongside (and as an upgrade to) the existing
    ``ComprehensiveResponseGenerator`` template system.  When the LLM is
    available, it produces question-specific, persona-aligned responses.
    When it's not, it silently falls back to the template system.

    Usage::

        gen = LLMResponseGenerator(
            api_key="gsk_...",
            study_title="Trust & AI",
            study_description="Examining trust in AI-generated advice",
            seed=42,
        )
        # Pre-generate a pool of responses for a question
        gen.prefill_pool(
            question_text="Why did you choose this option?",
            condition="AI_advice",
            sentiments=["positive", "neutral", "negative"],
            count_per_sentiment=15,
        )
        # Draw individual responses for each participant
        response = gen.generate(
            question_text="Why did you choose this option?",
            condition="AI_advice",
            sentiment="positive",
            persona_verbosity=0.7,
            persona_formality=0.4,
            persona_engagement=0.8,
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        study_title: str = "",
        study_description: str = "",
        seed: Optional[int] = None,
        fallback_generator: Any = None,
        batch_size: int = 12,
    ) -> None:
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._study_title = study_title
        self._study_description = study_description
        self._rng = random.Random(seed)
        self._fallback = fallback_generator
        self._batch_size = max(4, min(batch_size, 20))
        self._pool = _ResponsePool()
        self._rate_limiter = _RateLimiter(max_rpm=25)
        self._api_available = bool(self._api_key)
        self._call_count = 0
        self._fallback_count = 0

    @property
    def is_llm_available(self) -> bool:
        return self._api_available

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "llm_calls": self._call_count,
            "fallback_uses": self._fallback_count,
        }

    def set_study_context(self, title: str, description: str) -> None:
        self._study_title = title
        self._study_description = description

    # ------------------------------------------------------------------
    # Pool pre-fill: generate batches of responses BEFORE the per-participant loop
    # ------------------------------------------------------------------
    def prefill_pool(
        self,
        question_text: str,
        condition: str,
        sentiments: Optional[List[str]] = None,
        count_per_sentiment: int = 15,
    ) -> int:
        """Pre-generate a pool of LLM responses for a question+condition.

        Returns the total number of responses generated.
        """
        if not self._api_available:
            return 0

        if sentiments is None:
            sentiments = ["very_positive", "positive", "neutral",
                          "negative", "very_negative"]

        total = 0
        for sentiment in sentiments:
            needed = count_per_sentiment
            while needed > 0:
                batch_n = min(needed, self._batch_size)
                # Create diverse persona specs for this batch
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
                    # LLM failed — stop trying for this sentiment
                    break

        return total

    def _generate_batch(
        self,
        question_text: str,
        condition: str,
        sentiment: str,
        persona_specs: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate a batch of responses via the LLM API."""
        if not self._api_available:
            return []

        prompt = _build_batch_prompt(
            question_text=question_text,
            condition=condition,
            study_title=self._study_title,
            study_description=self._study_description,
            persona_specs=persona_specs,
        )

        self._rate_limiter.wait_if_needed()
        raw = _call_groq(self._api_key, SYSTEM_PROMPT, prompt)
        self._call_count += 1

        if raw is None:
            self._api_available = False  # Disable LLM for rest of session
            logger.warning("LLM API unavailable — disabling for this session")
            return []

        responses = _parse_json_responses(raw, len(persona_specs))
        if not responses:
            logger.warning("LLM returned unparseable response — falling back")
            return []

        return responses

    # ------------------------------------------------------------------
    # Single-response generation (draws from pool or falls back)
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

        Tries to draw from the pre-filled pool first.  If the pool is
        empty, generates a small on-demand batch.  If the LLM is
        unavailable, falls back to the template generator.
        """
        local_rng = random.Random(participant_seed)

        # 1. Try pool
        resp = self._pool.draw(question_text, condition, sentiment, local_rng)
        if resp:
            return self._apply_persona_variation(
                resp, persona_verbosity, persona_formality, persona_engagement,
                local_rng,
            )

        # 2. Try on-demand LLM batch
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
                resp = self._pool.draw(question_text, condition, sentiment, local_rng)
                if resp:
                    return self._apply_persona_variation(
                        resp, persona_verbosity, persona_formality,
                        persona_engagement, local_rng,
                    )

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
    # Post-generation persona variation
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_persona_variation(
        text: str,
        verbosity: float,
        formality: float,
        engagement: float,
        rng: random.Random,
    ) -> str:
        """Apply light persona-driven variation to a pool response.

        This adds uniqueness without changing the core content.
        """
        # Trim for low-verbosity personas
        if verbosity < 0.25:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 2:
                text = " ".join(sentences[:2])
        elif verbosity < 0.4:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 3:
                text = " ".join(sentences[:3])

        # Casual variations for low-formality
        if formality < 0.3 and rng.random() < 0.5:
            casual_starters = [
                "Honestly, ", "I mean, ", "Well, ", "Like, ",
                "So basically, ", "Tbh, ",
            ]
            if not any(text.startswith(s) for s in casual_starters):
                text = rng.choice(casual_starters) + text[0].lower() + text[1:]

        # Low engagement: sometimes truncate or add hedging
        if engagement < 0.2 and rng.random() < 0.4:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 1:
                text = sentences[0]

        return text.strip()
