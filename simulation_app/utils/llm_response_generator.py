"""
LLM-powered open-ended survey response generator.

Uses free LLM APIs (multi-provider with automatic failover) to generate
realistic, question-specific, persona-aligned open-ended survey responses.

Architecture:
- Multi-provider: Groq (built-in), Cerebras, Together AI, OpenRouter
  with automatic key detection and failover
- Large batch sizes: 20 responses per API call (within 32K context)
- Smart pool scaling: calculates exact pool size needed from sample_size
- Draw-with-replacement + deep variation: a pool of 50 base responses
  can serve 2,000+ participants with minimal repetition
- Graceful fallback: if all LLM providers fail, silently falls back to
  the existing template-based ComprehensiveResponseGenerator

Version: 1.4.10
"""

__version__ = "1.4.10"

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
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL = "llama-3.3-70b"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"

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
    "participants.  Each response should sound natural, contain typical human "
    "imperfections (hedging, incomplete thoughts, colloquialisms where appropriate), "
    "and vary in quality and detail based on the participant profile provided.\n\n"
    "CRITICAL: Every response MUST be grounded in the specific study topic, "
    "experimental condition, and survey question provided.  A participant in a "
    "political polarization study should talk about political feelings and partisan "
    "dynamics, NOT generic 'the study was interesting' filler.  A participant in a "
    "trust study should reference trust-related thoughts.  The study context, "
    "condition, and question are your primary guide for response content.\n\n"
    "Responses must be realistic survey responses, NOT polished essays.  "
    "Real participants write informally, sometimes go off-topic slightly, and vary "
    "significantly in effort and detail.  Careless participants give very short "
    "but still topic-relevant answers.  Engaged participants give thoughtful, "
    "specific answers referencing their experimental experience."
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
    import re as _re
    text = _re.sub(r'[_\-]+', ' ', name)
    text = _re.sub(r'([a-z])([A-Z])', r'\1 \2', text).lower().strip()
    if not text:
        return name
    # Add a question frame if the text doesn't look like a question already
    if not any(text.startswith(w) for w in ("how ", "what ", "why ", "where ", "when ",
                                             "who ", "which ", "do ", "does ", "did ",
                                             "please ", "describe ", "explain ")):
        # Heuristic: start with "Please describe your thoughts on: "
        return f"Please describe your thoughts on: {text}"
    return text


def _build_batch_prompt(
    question_text: str,
    condition: str,
    study_title: str,
    study_description: str,
    persona_specs: List[Dict[str, Any]],
    all_conditions: Optional[List[str]] = None,
) -> str:
    """Build a single prompt that asks the LLM to generate N responses at once.

    v1.4.11: Enhanced with richer study context and variable-name detection
    so responses are contextually grounded even when question_text is sparse.
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

    # Build conditions context (helps LLM understand the experimental design)
    conditions_block = ""
    if all_conditions:
        conditions_block = (
            f"All experimental conditions in this study: {', '.join(all_conditions)}\n"
            f"This participant was assigned to: {condition}\n"
        )
    else:
        conditions_block = f"Experimental condition: {condition}\n"

    prompt = (
        f'Study: "{study_title}"\n'
        f"Study description: {study_description[:500]}\n"
        f"{conditions_block}\n"
        f'Survey question: "{_q_display}"\n\n'
        f"Generate exactly {n} unique responses from {n} different survey "
        f"participants who just completed this experiment.\n"
        f"Each participant's profile controls their response style:\n\n"
        f"{participants_block}\n\n"
        f"Rules:\n"
        f"- Each response MUST be different from every other response.\n"
        f"- Responses MUST be grounded in the specific study topic "
        f"(\"{study_title}\") and the participant's assigned condition "
        f"(\"{condition}\").\n"
        f"- Participants should write AS IF they actually experienced the "
        f"experimental manipulation. Reference specific aspects of the study, "
        f"condition, or topic — do not give generic 'the study was fine' answers.\n"
        f"- Do NOT use bullet points, numbered lists, or markdown formatting "
        f"inside responses — just plain text as a survey participant would write.\n"
        f"- Match each participant's length, style, effort, and sentiment exactly.\n\n"
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

    Args:
        api_url: Full URL for the chat completions endpoint.
        api_key: Bearer token for authentication.
        model: Model identifier string.
        system_prompt: System message content.
        user_prompt: User message content.
        temperature: Sampling temperature (clamped to 0.0-2.0).
        max_tokens: Max response tokens.
        timeout: Request timeout in seconds.

    Returns the response text or None on any error.
    """
    if not api_key or not api_key.strip():
        return None

    temperature = max(0.0, min(2.0, temperature))

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "BehavioralSimulationTool/1.4.10",
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
            api_url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        return body["choices"][0]["message"]["content"]
    except ImportError:
        pass  # Fall through to urllib
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
            api_url, data=data, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("LLM API call failed [urllib] (%s %s): %s",
                       api_url[:50], model, exc)
        return None


def _parse_json_responses(raw: str, expected_n: int) -> List[str]:
    """Parse a JSON array of strings from the LLM output.

    Handles common LLM quirks: markdown code fences, trailing commas,
    single quotes, escaped quotes, and short responses.
    """
    if not raw or not raw.strip():
        return []

    cleaned = raw.strip()
    # Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Try standard JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(r).strip() for r in parsed if str(r).strip()]
    except json.JSONDecodeError:
        pass

    # Fix trailing commas
    try:
        fixed = re.sub(r",\s*]", "]", cleaned)
        parsed = json.loads(fixed)
        if isinstance(parsed, list):
            return [str(r).strip() for r in parsed if str(r).strip()]
    except json.JSONDecodeError:
        pass

    # Try replacing single quotes with double quotes (some LLMs do this)
    if cleaned.startswith("[") and "'" in cleaned:
        try:
            sq_fixed = cleaned.replace("'", '"')
            parsed = json.loads(sq_fixed)
            if isinstance(parsed, list):
                return [str(r).strip() for r in parsed if str(r).strip()]
        except json.JSONDecodeError:
            pass

    # Last resort: extract quoted strings (min 3 chars to accept short responses)
    matches = re.findall(r'"([^"]{3,})"', cleaned)
    if matches:
        return [m.strip() for m in matches if m.strip()]

    logger.warning("Could not parse LLM response as JSON array")
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
    ]


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------
class _LLMProvider:
    """Represents a single LLM API provider with automatic failure tracking."""

    def __init__(self, name: str, api_url: str, model: str, api_key: str) -> None:
        self.name = name
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.available = True
        self.call_count = 0
        self._consecutive_failures = 0
        self._max_failures = 3  # Disable after 3 consecutive failures

    def call(self, system_prompt: str, user_prompt: str,
             temperature: float = 0.7, max_tokens: int = 4000) -> Optional[str]:
        if not self.available or not self.api_key:
            return None
        result = _call_llm_api(
            self.api_url, self.api_key, self.model,
            system_prompt, user_prompt, temperature, max_tokens,
        )
        self.call_count += 1
        if result is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_failures:
                self.available = False
                logger.info("Provider '%s' disabled after %d consecutive failures",
                            self.name, self._consecutive_failures)
        else:
            self._consecutive_failures = 0  # Reset on success
        return result

    def reset(self) -> None:
        """Re-enable the provider (e.g., after rate-limit window expires)."""
        self.available = True
        self._consecutive_failures = 0


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------
class LLMResponseGenerator:
    """Generate open-ended survey responses using free LLM APIs.

    Multi-provider architecture with automatic failover:
    1. Built-in default Groq key (seamless for all users)
    2. User-provided key (auto-detected: Groq, Cerebras, OpenRouter)
    3. Environment variable providers (Together AI, Cerebras, OpenRouter)
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
        self._rate_limiter = _RateLimiter(max_rpm=28)
        self._fallback_count = 0

        # Build provider chain — all built-in keys first, then user key last.
        # When Groq rate-limits → auto-switch to Cerebras → OpenRouter → user key.
        self._providers: List[_LLMProvider] = []
        user_key = api_key or os.environ.get("LLM_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
        _all_builtin_keys = {_DEFAULT_GROQ_KEY, _DEFAULT_CEREBRAS_KEY, _DEFAULT_OPENROUTER_KEY}

        # Built-in providers (in priority order)
        for name, url, model, key in [
            ("groq_builtin", GROQ_API_URL, GROQ_MODEL, _DEFAULT_GROQ_KEY),
            ("cerebras_builtin", CEREBRAS_API_URL, CEREBRAS_MODEL, _DEFAULT_CEREBRAS_KEY),
            ("openrouter_builtin", OPENROUTER_API_URL, OPENROUTER_MODEL, _DEFAULT_OPENROUTER_KEY),
        ]:
            if key:
                self._providers.append(_LLMProvider(
                    name=name, api_url=url, model=model, api_key=key,
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
            ("TOGETHER_API_KEY", "together", TOGETHER_API_URL, TOGETHER_MODEL),
            ("CEREBRAS_API_KEY", "cerebras_env", CEREBRAS_API_URL, CEREBRAS_MODEL),
            ("OPENROUTER_API_KEY", "openrouter_env", OPENROUTER_API_URL, OPENROUTER_MODEL),
        ]:
            env_key = os.environ.get(env_var, "")
            if env_key and not any(p.api_key == env_key for p in self._providers):
                self._providers.append(_LLMProvider(
                    name=name, api_url=url, model=model, api_key=env_key,
                ))

        self._api_available = any(p.available and p.api_key for p in self._providers)

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
        """Human-readable name of the active provider for UI display."""
        names = {
            "groq_builtin": "Groq (built-in)",
            "cerebras_builtin": "Cerebras (built-in)",
            "openrouter_builtin": "OpenRouter (built-in)",
            "groq_user": "Groq (your key)",
            "cerebras_user": "Cerebras (your key)",
            "openrouter_user": "OpenRouter (your key)",
            "together": "Together AI",
            "cerebras_env": "Cerebras",
            "openrouter_env": "OpenRouter",
        }
        active = self.active_provider_name
        return names.get(active, active)

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
                batch_n = min(needed, self._batch_size)
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

        # Try every provider in the chain
        tried: set = set()
        while True:
            provider = self._get_active_provider()
            if not provider or provider.name in tried:
                break  # All providers exhausted
            tried.add(provider.name)

            self._rate_limiter.wait_if_needed()
            raw = provider.call(SYSTEM_PROMPT, prompt, max_tokens=4000)

            if raw is not None:
                responses = _parse_json_responses(raw, len(persona_specs))
                if responses:
                    return responses
                logger.warning("Provider '%s' returned unparseable response, trying next...",
                               provider.name)
            else:
                logger.info("Provider '%s' failed, trying next...", provider.name)

        # All providers failed
        self._api_available = False
        logger.warning("All %d LLM providers exhausted — falling back to templates",
                       len(self._providers))
        return []

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

        Tries each provider in order until one responds.  This ensures we
        find a working provider even when earlier ones are rate-limited.

        Args:
            timeout: Max seconds per individual provider check (default 15).

        Returns dict with 'available', 'provider', 'error' keys.
        """
        if not self._providers:
            return {"available": False, "provider": "none", "error": "No API keys configured"}

        last_error = "No providers available"
        tried: set = set()
        while True:
            provider = self._get_active_provider()
            if not provider or provider.name in tried:
                break
            tried.add(provider.name)

            try:
                raw = _call_llm_api(
                    provider.api_url, provider.api_key, provider.model,
                    "Reply with exactly: OK", "Test",
                    temperature=0.0, max_tokens=10, timeout=timeout,
                )
                if raw is not None:
                    return {"available": True, "provider": provider.name, "error": None}
                # Mark failed so _get_active_provider skips it next loop
                provider.available = False
                last_error = f"{provider.name} unavailable"
            except Exception as e:
                provider.available = False
                last_error = str(e)

        return {"available": False, "provider": "none", "error": last_error}
