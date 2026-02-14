"""Offline Text Realizer — template-grammar NLG for human-like open-ended responses.

No LLM required.  Uses:
  - Template grammars with persona-conditioned phrasing
  - Synonym sets for vocabulary diversity
  - Controlled noise (typos, fillers, hedges)
  - Anti-repetition constraints

Produces diverse, persona-consistent text without any network calls.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

import numpy as np

from ..persona import Persona


# ---------------------------------------------------------------------------
# Synonym banks
# ---------------------------------------------------------------------------

HEDGES = [
    "I think", "In my opinion", "I feel like", "Honestly", "To be honest",
    "I'd say", "From my perspective", "The way I see it", "Personally",
    "If you ask me", "I believe", "It seems to me",
]

FILLERS = [
    "um", "well", "like", "you know", "I mean", "basically",
    "sort of", "kind of", "actually", "honestly",
]

INTENSIFIERS = [
    "really", "very", "extremely", "incredibly", "absolutely",
    "definitely", "completely", "totally", "strongly",
]

DIMINISHERS = [
    "somewhat", "a bit", "slightly", "kind of", "sort of",
    "a little", "not really", "barely", "marginally",
]

POSITIVE_STEMS = [
    "I agree with this approach",
    "This seems like a good idea",
    "I support this direction",
    "I feel positively about this",
    "This resonates with my values",
    "I think this is the right choice",
    "I'm in favor of this",
    "This aligns with what I believe",
]

NEGATIVE_STEMS = [
    "I disagree with this approach",
    "This doesn't seem right to me",
    "I have concerns about this",
    "I feel negatively about this",
    "This goes against my values",
    "I don't think this is the right choice",
    "I'm against this",
    "This doesn't align with what I believe",
]

NEUTRAL_STEMS = [
    "I have mixed feelings about this",
    "I can see both sides",
    "It's hard to say definitively",
    "I'm somewhat undecided",
    "There are pros and cons",
    "I see merit in different perspectives",
]

ELABORATION_TEMPLATES = [
    "because {reason}",
    "since {reason}",
    "given that {reason}",
    "considering {reason}",
    "the reason being {reason}",
]

REASON_BANKS: Dict[str, List[str]] = {
    "prosocial": [
        "it helps others", "it promotes fairness", "everyone benefits",
        "it's the right thing to do", "cooperation is important",
        "we should think about the group", "sharing creates trust",
    ],
    "selfish": [
        "it benefits me the most", "I need to look out for myself",
        "the rational choice is clear", "maximizing my outcome matters",
        "others would do the same", "it's just common sense",
    ],
    "fairness": [
        "fairness matters to me", "equal treatment is important",
        "everyone deserves their share", "inequity bothers me",
        "I value equal outcomes", "nobody should be left behind",
    ],
    "strategic": [
        "it's the smart play", "strategically it makes sense",
        "thinking ahead leads here", "the logic is clear",
        "given what others will likely do", "game theory suggests this",
    ],
    "normative": [
        "it's what society expects", "most people would agree",
        "the social norm is clear", "it's the accepted behavior",
        "this is what's considered appropriate", "cultural values guide this",
    ],
    "default": [
        "it makes sense to me", "I've thought about it carefully",
        "based on my experience", "this reflects my values",
        "it aligns with my worldview", "I feel strongly about this",
    ],
}

# Careless/low-effort templates — ALL must reference {topic}
# Real careless participants still write about the topic (Meade & Craig 2012)
CARELESS_TEMPLATES = [
    "{topic} idk", "idk about {topic}", "{topic} is fine i guess",
    "{topic} ok", "whatever about {topic}", "{topic} sure",
    "{topic} doesnt matter", "i dont care about {topic}", "{topic} is ok i guess",
    "not sure about {topic}", "{topic} whatever", "eh {topic}",
]

# Game-specific reason banks for economic games
GAME_REASON_BANKS: Dict[str, List[str]] = {
    "dictator": [
        "I wanted to be fair in splitting the money",
        "keeping more felt like the right call for me",
        "I thought about what would be fair to both of us",
        "I gave what I felt comfortable with",
        "splitting money is about more than just maximizing",
        "I considered what the other person might need",
    ],
    "trust": [
        "trusting someone is risky but can pay off",
        "I wanted to show good faith",
        "you have to give trust to get trust back",
        "I was cautious because I don't know the other person",
        "sending more could lead to a better outcome for both",
        "I held back because the other person might not return anything",
    ],
    "public_goods": [
        "contributing helps the whole group",
        "I didn't want to be the one who free rides",
        "if everyone contributes we all benefit more",
        "I contributed less because others might not pull their weight",
        "group outcomes depend on everyone doing their part",
        "I balanced what's good for me with what's good for the group",
    ],
    "ultimatum": [
        "the offer seemed fair enough to accept",
        "rejecting means nobody gets anything",
        "I wouldn't accept an unfair split on principle",
        "fairness matters more than getting something rather than nothing",
        "I tried to make an offer that would be accepted",
        "low offers feel insulting even if some money is better than none",
    ],
    "prisoners_dilemma": [
        "cooperating is better if the other person does too",
        "I couldn't be sure the other person wouldn't defect",
        "the temptation to defect is strong but cooperation is better long term",
        "I chose based on what I thought the other person would do",
        "mutual cooperation is the best outcome for everyone",
    ],
}


# ---------------------------------------------------------------------------
# Core realizer
# ---------------------------------------------------------------------------

class OfflineTextRealizer:
    """Generate persona-conditioned open-ended text without any LLM.

    Uses template grammars + synonym sets + controlled noise.
    """

    def __init__(self, anti_repetition_window: int = 10) -> None:
        self._recent: List[str] = []
        self._window = anti_repetition_window

    def generate(
        self,
        prompt_spec: Dict[str, Any],
        persona: Persona,
        rng: np.random.Generator,
    ) -> str:
        """Generate an open-ended text response.

        Parameters
        ----------
        prompt_spec : dict
            Must contain:
              - question_text: the question being answered
              - context: experimental context
              - condition: condition label
            Optional:
              - topic: extracted topic words
              - sentiment: target sentiment (positive/negative/neutral)
        persona : Persona
            The simulated participant.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        str
            Generated response text.
        """
        question = prompt_spec.get("question_text", "")
        context = prompt_spec.get("context", "")
        condition = prompt_spec.get("condition", "")
        topic = prompt_spec.get("topic", self._extract_topic(question, context))

        # Persona traits
        verbosity = float(persona.params.get("verbosity", 0.5))
        formality = float(persona.params.get("formality", 0.5))
        engagement = float(persona.params.get("engagement", 0.5))
        prosoc = float(persona.params.get("prosociality", 0.0))
        extremity = float(persona.params.get("extremity", 0.5))

        # Determine engagement level
        if engagement < 0.2:
            return self._careless_response(topic, rng)

        # Determine sentiment from prosociality and context
        if prosoc > 0.5:
            sentiment = "positive"
        elif prosoc < -0.5:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Build response
        parts: List[str] = []

        # Opening hedge (for less formal respondents)
        if formality < 0.4 and rng.random() < 0.6:
            parts.append(rng.choice(HEDGES))

        # Main stem
        if sentiment == "positive":
            stem = str(rng.choice(POSITIVE_STEMS))
        elif sentiment == "negative":
            stem = str(rng.choice(NEGATIVE_STEMS))
        else:
            stem = str(rng.choice(NEUTRAL_STEMS))

        # Add topic reference
        if topic:
            stem = stem.replace("this approach", f"the approach to {topic}")
            stem = stem.replace("this direction", f"the direction on {topic}")
            stem = stem.replace("this", f"{topic}" if rng.random() < 0.3 else "this")

        # Add intensifier/diminisher based on extremity
        if extremity > 0.7 and rng.random() < 0.5:
            intensifier = str(rng.choice(INTENSIFIERS))
            stem = stem.replace("I think", f"I {intensifier} think")
            stem = stem.replace("I feel", f"I {intensifier} feel")
        elif extremity < 0.3 and rng.random() < 0.5:
            diminisher = str(rng.choice(DIMINISHERS))
            stem = stem.replace("I agree", f"I {diminisher} agree")
            stem = stem.replace("I disagree", f"I {diminisher} disagree")

        parts.append(stem)

        # Elaboration (based on verbosity)
        if verbosity > 0.4:
            game_name = prompt_spec.get("game_name", "")
            reason_bank = self._select_reason_bank(persona, game_name=game_name)
            reason = str(rng.choice(reason_bank))
            template = str(rng.choice(ELABORATION_TEMPLATES))
            parts.append(template.format(reason=reason))

        # Additional sentence for verbose respondents
        if verbosity > 0.7 and rng.random() < 0.6:
            extra = self._extra_sentence(sentiment, topic, persona, rng)
            parts.append(extra)

        # Add filler for informal respondents
        if formality < 0.3 and rng.random() < 0.4:
            filler = str(rng.choice(FILLERS))
            parts[0] = f"{filler}, {parts[0]}"

        response = ". ".join(parts)
        if not response.endswith("."):
            response += "."

        # Apply noise (typos for low-attention respondents)
        attention = float(persona.params.get("attention", 0.5))
        if attention < 0.3:
            response = self._add_typos(response, rng, rate=0.03)

        # Anti-repetition check
        response = self._anti_repeat(response, rng)
        self._recent.append(response)
        if len(self._recent) > self._window:
            self._recent.pop(0)

        return response

    def _careless_response(self, topic: str, rng: np.random.Generator) -> str:
        # All careless templates require a topic — never use bare "this"
        effective_topic = topic or "the questions asked"
        template = str(rng.choice(CARELESS_TEMPLATES))
        return template.format(topic=effective_topic)

    def _select_reason_bank(
        self, persona: Persona, game_name: str = "",
    ) -> List[str]:
        """Select reason bank based on persona traits and game context."""
        # Game-specific banks take priority when available
        if game_name and game_name in GAME_REASON_BANKS:
            return GAME_REASON_BANKS[game_name]

        prosoc = float(persona.params.get("prosociality", 0.0))
        norm_w = float(persona.params.get("norm_weight", 0.0))
        depth = float(persona.params.get("strategic_depth", 0.0))

        if prosoc > 0.8:
            return REASON_BANKS["prosocial"]
        if prosoc < -0.5:
            return REASON_BANKS["selfish"]
        if norm_w > 0.5:
            return REASON_BANKS["normative"]
        if depth > 2.0:
            return REASON_BANKS["strategic"]
        if abs(prosoc) > 0.3:
            return REASON_BANKS["fairness"]
        return REASON_BANKS["default"]

    def _extra_sentence(
        self,
        sentiment: str,
        topic: str,
        persona: Persona,
        rng: np.random.Generator,
    ) -> str:
        topic_ref = f"about {topic}" if topic else ""
        if sentiment == "positive":
            options = [
                f"I've had positive experiences {topic_ref}",
                f"My experience {topic_ref} has been largely good",
                f"Overall I'm optimistic {topic_ref}",
            ]
        elif sentiment == "negative":
            options = [
                f"I've had negative experiences {topic_ref}",
                f"My experience {topic_ref} has been disappointing",
                f"I remain skeptical {topic_ref}",
            ]
        else:
            options = [
                f"I can see arguments on both sides {topic_ref}",
                f"It really depends on the specifics {topic_ref}",
                f"More information would help me decide {topic_ref}",
            ]
        return str(rng.choice(options))

    @staticmethod
    def _extract_topic(question: str, context: str) -> str:
        """Extract topic words from question/context."""
        stop = {
            "this", "that", "about", "what", "your", "please", "describe",
            "explain", "question", "context", "study", "topic", "condition",
            "think", "feel", "have", "some", "with", "from", "very", "really",
            "would", "could", "should", "much", "many", "more", "than",
            "the", "and", "for", "are", "was", "were", "how", "why",
            "you", "they", "them", "their", "will", "been", "being",
            "opinion", "important", "believe", "agree", "disagree",
            "following", "answer", "response", "rate", "scale",
            "degree", "extent", "which", "whether", "does", "did",
        }
        text = f"{question} {context}".lower()
        words = re.findall(r"\b[a-z]{3,}\b", text)
        content = [w for w in words if w not in stop]
        return " ".join(content[:4]) if content else ""

    @staticmethod
    def _add_typos(text: str, rng: np.random.Generator, rate: float = 0.02) -> str:
        """Introduce random typos at the given rate."""
        chars = list(text)
        for i in range(len(chars)):
            if chars[i].isalpha() and rng.random() < rate:
                if rng.random() < 0.5:
                    # Swap adjacent characters
                    if i + 1 < len(chars):
                        chars[i], chars[i + 1] = chars[i + 1], chars[i]
                else:
                    # Drop a character
                    chars[i] = ""
        return "".join(chars)

    def _anti_repeat(self, response: str, rng: np.random.Generator) -> str:
        """Ensure response isn't too similar to recent ones."""
        if not self._recent:
            return response
        # Simple check: if first 20 chars match any recent, regenerate opening
        prefix = response[:20].lower()
        for recent in self._recent[-5:]:
            if recent[:20].lower() == prefix:
                # Swap the opening phrase
                new_hedge = str(rng.choice(HEDGES))
                if ". " in response:
                    _, rest = response.split(". ", 1)
                    return f"{new_hedge}, {rest}"
        return response
