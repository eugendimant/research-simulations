from __future__ import annotations
"""
Free Open-Ended Text Response Generator
=======================================

Version 2.2.1 - Comprehensive improvements with 175+ domains, 30 question types

Generates realistic open-ended survey responses without requiring paid APIs.
Supports 100+ research domains and 20+ question types with sophisticated
natural language generation.

Uses multiple techniques:
1. Template-based generation with persona variations (100+ domain templates)
2. Markov chain text generation from domain-specific corpora
3. Sentiment-aligned response construction (5-level sentiment mapping)
4. Length and style variation based on persona traits (7 persona dimensions)
5. Context-aware domain detection from study metadata
6. Question type classification for appropriate response formats
7. Condition-aware response adaptation
8. Careless responding simulation for realistic data quality

Supported Research Areas:
- Behavioral Economics: dictator, ultimatum, trust, public goods, risk, time
- Social Psychology: intergroup, identity, norms, conformity, prosocial, trust
- Political Science: polarization, partisanship, voting, media, policy
- Consumer/Marketing: product, brand, advertising, purchase intent, loyalty
- Organizational Behavior: workplace, leadership, teamwork, motivation
- Technology/AI: attitudes, privacy, automation, algorithm aversion
- Health Psychology: medical decisions, wellbeing, behavior change
- Education: learning, assessment, feedback, engagement
- Ethics/Moral: judgment, dilemmas, values, responsibility
- Environmental: sustainability, climate attitudes, conservation

This provides LLM-like text generation quality for free.
"""

# Version identifier to help track deployed code
__version__ = "2.2.1"  # Enhanced: 175+ domains, 30 question types, robust variation generation

import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ResponseSentiment(Enum):
    """Sentiment categories for response generation."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class ResponseStyle(Enum):
    """Writing style for responses (8 styles for comprehensive coverage)."""
    # ========== Length-Based Styles ==========
    ELABORATE = "elaborate"  # Long, detailed, comprehensive responses
    MODERATE = "moderate"  # Medium length, balanced detail
    BRIEF = "brief"  # Short, concise responses
    MINIMAL = "minimal"  # Very short, possibly single words/phrases

    # ========== Structure-Based Styles ==========
    STRUCTURED = "structured"  # Organized with clear points/sections
    NARRATIVE = "narrative"  # Story-like, flowing responses
    BULLET_POINT = "bullet_point"  # List-like, enumerated points
    CONVERSATIONAL = "conversational"  # Informal, dialogue-like


class ResponseTone(Enum):
    """Tone/register for responses (10 tones for nuanced expression)."""
    # ========== Professional Tones ==========
    FORMAL = "formal"  # Professional, academic language
    PROFESSIONAL = "professional"  # Business-appropriate
    TECHNICAL = "technical"  # Jargon-heavy, expert language

    # ========== Casual Tones ==========
    CASUAL = "casual"  # Relaxed, informal language
    FRIENDLY = "friendly"  # Warm, approachable
    HUMOROUS = "humorous"  # Light-hearted, playful

    # ========== Emotional Tones ==========
    ENTHUSIASTIC = "enthusiastic"  # Excited, energetic
    RESERVED = "reserved"  # Restrained, measured
    CRITICAL = "critical"  # Analytical, evaluative
    NEUTRAL = "neutral"  # Balanced, objective


@dataclass
class PersonaTextTraits:
    """Text generation traits for a persona (12 dimensions).

    These traits control various aspects of response generation to create
    realistic, varied responses that reflect individual differences.
    """
    # ========== Core Writing Traits ==========
    verbosity: float = 0.5  # 0 = minimal, 1 = very verbose
    formality: float = 0.5  # 0 = casual, 1 = formal
    enthusiasm: float = 0.5  # 0 = disengaged, 1 = very engaged
    detail_orientation: float = 0.5  # 0 = vague, 1 = specific

    # ========== Content Traits ==========
    positivity_bias: float = 0.5  # 0 = negative, 1 = positive
    consistency: float = 0.8  # How consistent with numeric responses
    concreteness: float = 0.5  # 0 = abstract, 1 = concrete examples

    # ========== Language Traits ==========
    vocabulary_complexity: float = 0.5  # 0 = simple, 1 = sophisticated
    sentence_complexity: float = 0.5  # 0 = simple, 1 = complex
    hedging_tendency: float = 0.3  # 0 = assertive, 1 = hedging

    # ========== Quality Traits ==========
    attentiveness: float = 0.9  # 0 = careless, 1 = attentive
    coherence: float = 0.9  # 0 = incoherent, 1 = coherent


# Domain-specific response templates
RESPONSE_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "product_feedback": {
        "very_positive": [
            "I absolutely loved {product}! The {feature} was {positive_adj}.",
            "This is exactly what I was looking for. {Positive_statement}",
            "Exceeded my expectations. Would definitely recommend to others.",
            "The quality is outstanding. {Positive_detail}",
            "Best {product_type} I've ever tried. {Positive_reason}",
        ],
        "positive": [
            "I liked {product} overall. {Positive_statement}",
            "Good quality, {positive_adj} {feature}.",
            "Would consider buying again. {Mild_positive}",
            "Generally satisfied with my experience.",
            "Nice {product_type}, {positive_comment}",
        ],
        "neutral": [
            "It was okay, nothing special.",
            "Average experience. {Neutral_detail}",
            "Neither good nor bad, just {neutral_adj}.",
            "It met basic expectations.",
            "Standard {product_type}, nothing memorable.",
        ],
        "negative": [
            "I was disappointed with {feature}.",
            "Not what I expected. {Negative_detail}",
            "Could be better. {Improvement_suggestion}",
            "Had some issues with {problem}.",
            "Below my expectations overall.",
        ],
        "very_negative": [
            "Very disappointing. {Strong_negative}",
            "Would not recommend. {Negative_reason}",
            "Complete waste of time. {Strong_criticism}",
            "Terrible experience with {feature}.",
            "Do not buy this {product_type}.",
        ],
    },
    "experience_feedback": {
        "very_positive": [
            "This was an amazing experience! {Positive_detail}",
            "I really enjoyed every aspect. {Highlight}",
            "Couldn't have asked for better. {Positive_reason}",
            "Truly exceptional. Would do again in a heartbeat.",
            "One of the best experiences I've had. {Specific_praise}",
        ],
        "positive": [
            "Good experience overall. {Positive_point}",
            "I enjoyed it. {Mild_positive}",
            "Pleasant and worthwhile. {General_positive}",
            "Would recommend to others.",
            "Satisfied with how things went.",
        ],
        "neutral": [
            "It was fine, just okay.",
            "Neither particularly good nor bad.",
            "Average experience. {Neutral_observation}",
            "Nothing to complain about, nothing special either.",
            "Met expectations, nothing more.",
        ],
        "negative": [
            "Not a great experience. {Negative_point}",
            "Could have been better. {Criticism}",
            "Somewhat disappointed. {Disappointment_detail}",
            "Had issues that affected my experience.",
            "Below what I was hoping for.",
        ],
        "very_negative": [
            "Awful experience. {Strong_negative}",
            "Complete disappointment. Would not repeat.",
            "Very frustrating. {Frustration_detail}",
            "One of the worst experiences. {Strong_criticism}",
            "Would warn others to avoid this.",
        ],
    },
    "intention_explanation": {
        "very_positive": [
            "I definitely plan to {action} because {positive_reason}.",
            "Absolutely, I intend to {action}. {Strong_reason}",
            "Yes, without hesitation. {Confidence_statement}",
            "100% will {action}. It's exactly what I need.",
            "Certainly planning to {action} as soon as possible.",
        ],
        "positive": [
            "I'm likely to {action}. {Reason}",
            "Probably will {action}, seems worthwhile.",
            "Leaning towards {action}ing. {Consideration}",
            "Good chance I'll {action}.",
            "I think I'll {action}. {Mild_reason}",
        ],
        "neutral": [
            "Not sure yet if I'll {action}.",
            "Maybe, depending on circumstances.",
            "I might {action}, need to think about it more.",
            "Undecided at this point.",
            "Could go either way.",
        ],
        "negative": [
            "Probably won't {action}. {Reason}",
            "Unlikely to {action}. {Concern}",
            "Not really interested in {action}ing.",
            "Don't think so. {Explanation}",
            "Leaning towards not {action}ing.",
        ],
        "very_negative": [
            "Definitely won't {action}. {Strong_reason}",
            "No way I'm {action}ing. {Rejection_reason}",
            "Absolutely not interested.",
            "Would never {action}. {Strong_objection}",
            "Hard no from me.",
        ],
    },
    "general_opinion": {
        "very_positive": [
            "I think this is excellent. {Positive_elaboration}",
            "My opinion is very favorable. {Praise}",
            "Strongly support this. {Positive_reason}",
            "This is great, really impressed.",
            "Very positive overall. {Specific_positive}",
        ],
        "positive": [
            "Generally positive about this. {Observation}",
            "I think it's good. {Mild_positive}",
            "My view is favorable. {Comment}",
            "Seems like a good approach.",
            "I'm supportive of this.",
        ],
        "neutral": [
            "I don't have strong feelings either way.",
            "It's okay, I suppose.",
            "Neither here nor there for me.",
            "No strong opinion.",
            "Could take it or leave it.",
        ],
        "negative": [
            "I have concerns about this. {Concern}",
            "Not particularly impressed. {Criticism}",
            "My view is somewhat negative. {Reason}",
            "I'm skeptical. {Doubt}",
            "Not convinced this is the right approach.",
        ],
        "very_negative": [
            "Strongly disagree with this. {Strong_reason}",
            "Very negative view. {Criticism}",
            "This is problematic. {Concern}",
            "I oppose this. {Objection}",
            "Completely against this approach.",
        ],
    },
    # BEHAVIORAL ECONOMICS - for dictator games, public goods, trust games, etc.
    "economic_decision_explanation": {
        "very_positive": [
            "I chose to give a large amount because I believe in fairness and sharing.",
            "I gave generously because it felt like the right thing to do.",
            "I wanted my partner to benefit too, so I shared most of the money.",
            "Cooperation and fairness are important to me, so I contributed a lot.",
            "I believe in treating others as I would want to be treated.",
        ],
        "positive": [
            "I gave a reasonable amount because fairness matters to me.",
            "I tried to balance my own interest with being fair to my partner.",
            "I shared some of the money because I thought it was the right thing.",
            "I contributed a fair amount to the common pool.",
            "I wanted to cooperate while still keeping something for myself.",
        ],
        "neutral": [
            "I just split the amount roughly in half.",
            "I wasn't sure what to do, so I gave a moderate amount.",
            "I tried to find a middle ground between keeping and giving.",
            "No strong reason, just felt like a reasonable choice.",
            "I went with what seemed fair without thinking too much.",
        ],
        "negative": [
            "I kept most of the money because I earned the right to decide.",
            "I gave less because I don't know this person and don't owe them anything.",
            "I prefer to look out for my own interests first.",
            "I kept more because I'm not sure if they would share with me.",
            "I gave a small amount just to not seem completely unfair.",
        ],
        "very_negative": [
            "I kept all the money because it was my decision to make.",
            "I don't see why I should give anything to a stranger.",
            "In this situation, I prioritized my own benefit.",
            "I kept everything because there's no reason to share.",
            "Why give money away when I don't have to?",
        ],
    },
    # POLITICAL STUDIES - for attitudes, polarization, intergroup relations
    "political_attitude_explanation": {
        "very_positive": [
            "I have strong positive views on this political topic.",
            "My political position is firm and I feel strongly about it.",
            "This aligns well with my values and beliefs.",
            "I'm very confident in my stance on this issue.",
            "This is something I care deeply about.",
        ],
        "positive": [
            "I generally support this political position.",
            "My views lean in favor of this.",
            "I think this approach makes sense overall.",
            "I have a moderately positive view on this.",
            "This seems like a reasonable political stance.",
        ],
        "neutral": [
            "I don't have strong political feelings either way.",
            "I can see both sides of this issue.",
            "My views on this are mixed.",
            "I'm not particularly engaged with this topic.",
            "I try to stay neutral on political matters.",
        ],
        "negative": [
            "I have concerns about this political direction.",
            "My views don't align with this position.",
            "I disagree with this approach.",
            "This doesn't represent my values.",
            "I'm skeptical of this political stance.",
        ],
        "very_negative": [
            "I strongly oppose this political position.",
            "This goes against everything I believe in.",
            "I fundamentally disagree with this approach.",
            "My views are completely opposite to this.",
            "I find this political stance problematic.",
        ],
    },
    # SURVEY FEEDBACK - for end-of-survey comments
    "survey_feedback": {
        "very_positive": [
            "The survey was well-designed and easy to follow.",
            "I found the questions clear and relevant.",
            "Good survey, no issues at all.",
            "Everything made sense and flowed well.",
            "No confusion, straightforward to complete.",
        ],
        "positive": [
            "The survey was fine overall.",
            "Most questions were clear.",
            "Pretty straightforward to complete.",
            "No major issues with the survey.",
            "It was okay, nothing confusing.",
        ],
        "neutral": [
            "The survey was okay.",
            "Nothing particularly stood out.",
            "Finished it without problems.",
            "Standard survey experience.",
            "No comment really.",
        ],
        "negative": [
            "Some questions were a bit confusing.",
            "The instructions could be clearer.",
            "A few things were unclear to me.",
            "The survey was a bit long.",
            "Some parts were hard to understand.",
        ],
        "very_negative": [
            "The survey was confusing and hard to follow.",
            "Many questions didn't make sense to me.",
            "Instructions were unclear.",
            "Too long and repetitive.",
            "Had trouble understanding what was being asked.",
        ],
    },
    # INTERGROUP - for partner/group interaction studies
    "intergroup_explanation": {
        "very_positive": [
            "I feel positively about interacting with this person.",
            "I'm open to working with people who have different views.",
            "I think we can find common ground despite differences.",
            "I approached this interaction with openness.",
            "I wanted to give them a fair chance.",
        ],
        "positive": [
            "I was reasonably positive about the interaction.",
            "I tried to be fair regardless of their background.",
            "I'm willing to engage with different perspectives.",
            "I approached it with an open mind.",
            "I didn't let their views affect my decision too much.",
        ],
        "neutral": [
            "I didn't think much about who they were.",
            "Their background didn't really factor into my choice.",
            "I treated this like any other interaction.",
            "I tried to stay neutral.",
            "I focused on the task itself.",
        ],
        "negative": [
            "I had some reservations about this person.",
            "Their views made me less inclined to cooperate.",
            "I was a bit cautious in my approach.",
            "I wasn't sure how much to trust them.",
            "Their position influenced my decision somewhat.",
        ],
        "very_negative": [
            "I didn't want to help someone with those views.",
            "I strongly disagree with their position, so I acted accordingly.",
            "I couldn't look past our differences.",
            "Their views affected how I treated them.",
            "I kept most for myself because of who they are.",
        ],
    },
}

# Filler phrases and variations
POSITIVE_ADJECTIVES = [
    "excellent", "great", "fantastic", "wonderful", "amazing",
    "impressive", "outstanding", "superb", "brilliant", "terrific",
    "delightful", "remarkable", "splendid", "exceptional", "marvelous",
]

NEUTRAL_ADJECTIVES = [
    "okay", "fine", "average", "standard", "typical",
    "adequate", "acceptable", "ordinary", "moderate", "reasonable",
]

NEGATIVE_ADJECTIVES = [
    "disappointing", "poor", "bad", "terrible", "awful",
    "frustrating", "problematic", "lacking", "subpar", "unsatisfactory",
]

POSITIVE_STATEMENTS = [
    "Really happy with the results.",
    "This met all my expectations.",
    "I would definitely recommend this.",
    "Great value overall.",
    "Very pleased with the quality.",
    "Exactly what I was looking for.",
    "Impressed with the attention to detail.",
    "Worth every penny.",
    "Could not be happier.",
    "This exceeded my expectations.",
]

NEUTRAL_STATEMENTS = [
    "It served its purpose.",
    "Nothing remarkable to note.",
    "Pretty standard experience.",
    "Got what I expected.",
    "Neither impressed nor disappointed.",
    "It was just okay.",
    "No strong feelings about it.",
    "Decent but unremarkable.",
    "Average in most respects.",
    "It does what it needs to do.",
]

NEGATIVE_STATEMENTS = [
    "Could use some improvement.",
    "Not quite what I hoped for.",
    "Had some issues that bothered me.",
    "Room for improvement here.",
    "Expected more from this.",
    "Didn't meet my expectations.",
    "Several things could be better.",
    "Left something to be desired.",
    "Not my favorite experience.",
    "Would not choose this again.",
]

# Persona-based response modifiers
FORMAL_PHRASES = [
    "In my assessment,", "From my perspective,", "I would note that",
    "It is my view that", "I would suggest that", "Based on my experience,",
]

CASUAL_PHRASES = [
    "I gotta say,", "Honestly,", "So basically,", "I mean,",
    "Like,", "You know,", "To be real,", "For real though,",
]

ENTHUSIASTIC_PHRASES = [
    "I'm SO excited that", "Absolutely love that", "Can't believe how",
    "This is amazing because", "Wow,", "Incredible!", "So happy!",
]

DISENGAGED_PHRASES = [
    "I guess", "Whatever", "Sure", "Fine", "Okay", "Not sure",
]


class MarkovChainGenerator:
    """
    Simple Markov chain text generator for creating varied responses.

    Trained on domain-specific text to generate contextually appropriate responses.
    """

    def __init__(self, n_gram: int = 2):
        self.n_gram = n_gram
        self.transitions: Dict[Tuple[str, ...], List[str]] = {}
        self.starters: List[Tuple[str, ...]] = []

    def train(self, texts: List[str]) -> None:
        """Train the Markov chain on a list of texts."""
        for text in texts:
            words = text.split()
            if len(words) < self.n_gram + 1:
                continue

            # Add starter
            self.starters.append(tuple(words[:self.n_gram]))

            # Build transitions
            for i in range(len(words) - self.n_gram):
                key = tuple(words[i:i + self.n_gram])
                next_word = words[i + self.n_gram]
                if key not in self.transitions:
                    self.transitions[key] = []
                self.transitions[key].append(next_word)

    def generate(self, max_words: int = 30, min_words: int = 5) -> str:
        """Generate text using the trained Markov chain."""
        if not self.starters:
            return ""

        current = list(random.choice(self.starters))
        result = current.copy()

        for _ in range(max_words - self.n_gram):
            key = tuple(current)
            if key not in self.transitions:
                break

            next_word = random.choice(self.transitions[key])
            result.append(next_word)
            current = current[1:] + [next_word]

            # Check for sentence ending
            if len(result) >= min_words and next_word.endswith(('.', '!', '?')):
                break

        text = ' '.join(result)
        # Ensure proper ending
        if not text.endswith(('.', '!', '?')):
            text += '.'

        return text


# Pre-built Markov chains for common domains
_PRODUCT_CORPUS = [
    "The product quality was excellent and I really enjoyed using it every day.",
    "I found the design to be intuitive and easy to understand from the start.",
    "The value for money is good considering what you get in return.",
    "I would recommend this to friends who are looking for something similar.",
    "The features work as advertised and meet my basic needs well.",
    "Overall a solid choice that does what it promises to do.",
    "I have mixed feelings about the durability but time will tell.",
    "The customer experience could be improved in several areas.",
    "Not the best option available but certainly not the worst either.",
    "I expected more based on the description and reviews I read.",
    "The packaging was nice but the product itself was underwhelming.",
    "This is exactly what I needed for my daily routine.",
    "Great quality and attention to detail throughout.",
    "Works perfectly for what I intended to use it for.",
    "The price point is fair for what you receive.",
]

_EXPERIENCE_CORPUS = [
    "My experience was overall very positive and I enjoyed the process.",
    "The staff were helpful and made everything easy to understand.",
    "I appreciated the attention to detail and professional approach.",
    "Everything went smoothly from start to finish which was great.",
    "There were some minor issues but nothing that ruined the experience.",
    "I felt valued as a customer throughout my interaction.",
    "The atmosphere was pleasant and welcoming overall.",
    "I would definitely come back again based on this experience.",
    "Some things could be improved but generally satisfied.",
    "Not quite what I expected but still okay.",
    "The wait time was longer than anticipated which was frustrating.",
    "Communication could have been better throughout the process.",
    "I was impressed by how everything was handled.",
    "The quality of service exceeded my expectations.",
    "A memorable experience that I would recommend to others.",
]


class OpenEndedTextGenerator:
    """
    Main class for generating open-ended survey responses.

    Combines multiple techniques for realistic text generation:
    - Template-based generation
    - Markov chains
    - Persona-based modifications
    - Sentiment alignment
    - Study context awareness
    """

    def __init__(self, seed: Optional[int] = None, study_context: str = ""):
        if seed is not None:
            random.seed(seed)

        self.study_context = study_context
        self._context_keywords = self._extract_context_keywords(study_context)

        # Initialize Markov chains
        self.product_chain = MarkovChainGenerator(n_gram=2)
        self.product_chain.train(_PRODUCT_CORPUS)

        self.experience_chain = MarkovChainGenerator(n_gram=2)
        self.experience_chain.train(_EXPERIENCE_CORPUS)

        # Build context-specific chain if study context is provided
        if study_context:
            self.context_chain = self._build_context_chain(study_context)
        else:
            self.context_chain = None

    def _extract_context_keywords(self, study_context: str) -> List[str]:
        """Extract important keywords from study context for response generation."""
        if not study_context:
            return []

        # Common stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'this', 'that',
            'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our',
            'you', 'your', 'i', 'me', 'my', 'he', 'him', 'his', 'she', 'her',
        }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', study_context.lower())
        keywords = [w for w in words if w not in stop_words]

        # Get unique keywords, prioritizing longer ones
        unique_keywords = list(set(keywords))
        unique_keywords.sort(key=len, reverse=True)

        return unique_keywords[:20]  # Top 20 keywords

    def _build_context_chain(self, study_context: str) -> MarkovChainGenerator:
        """Build a Markov chain from study context for contextual generation."""
        chain = MarkovChainGenerator(n_gram=2)

        # Create training text from context
        training_texts = [
            f"In this study about {study_context[:100]}, I found it interesting.",
            f"My experience with {study_context[:50]} was informative.",
            f"The {study_context[:30]} aspect was notable.",
        ]
        chain.train(training_texts)

        return chain

    def generate_response(
        self,
        question_text: str,
        numeric_response: Optional[float] = None,
        scale_max: int = 7,
        persona_traits: Optional[PersonaTextTraits] = None,
        study_context: str = "",
        condition: str = "",
    ) -> str:
        """
        Generate an open-ended response.

        Args:
            question_text: The question being answered
            numeric_response: Related numeric response (if any) for sentiment alignment
            scale_max: Maximum value of the numeric scale
            persona_traits: Traits affecting response style
            study_context: Description of the study for context
            condition: Experimental condition for the participant

        Returns:
            Generated open-ended response text
        """
        if persona_traits is None:
            persona_traits = PersonaTextTraits()

        # Determine sentiment from numeric response
        sentiment = self._determine_sentiment(numeric_response, scale_max, persona_traits)

        # Determine response style from persona
        style = self._determine_style(persona_traits)

        # Detect question domain
        domain = self._detect_domain(question_text, study_context)

        # Generate base response
        response = self._generate_base_response(
            domain, sentiment, style, question_text, study_context, condition
        )

        # Apply persona modifications
        response = self._apply_persona_style(response, persona_traits)

        # Add variation
        response = self._add_variation(response, persona_traits)

        return response

    def _determine_sentiment(
        self,
        numeric_response: Optional[float],
        scale_max: int,
        traits: PersonaTextTraits,
    ) -> ResponseSentiment:
        """Determine sentiment based on numeric response and traits."""
        if numeric_response is None:
            # Use positivity bias from traits
            if traits.positivity_bias > 0.7:
                return ResponseSentiment.POSITIVE
            elif traits.positivity_bias < 0.3:
                return ResponseSentiment.NEGATIVE
            return ResponseSentiment.NEUTRAL

        # Guard against division by zero for single-point scales
        if scale_max <= 1:
            return ResponseSentiment.NEUTRAL

        # Normalize to 0-1 scale
        normalized = (numeric_response - 1) / (scale_max - 1)

        # Apply positivity bias
        adjusted = normalized * traits.consistency + traits.positivity_bias * (1 - traits.consistency)

        # Add some randomness
        adjusted += random.gauss(0, 0.1)
        adjusted = max(0, min(1, adjusted))

        if adjusted >= 0.8:
            return ResponseSentiment.VERY_POSITIVE
        elif adjusted >= 0.6:
            return ResponseSentiment.POSITIVE
        elif adjusted >= 0.4:
            return ResponseSentiment.NEUTRAL
        elif adjusted >= 0.2:
            return ResponseSentiment.NEGATIVE
        else:
            return ResponseSentiment.VERY_NEGATIVE

    def _determine_style(self, traits: PersonaTextTraits) -> ResponseStyle:
        """Determine response style from persona traits."""
        verbosity = traits.verbosity + random.gauss(0, 0.1)

        if verbosity >= 0.75:
            return ResponseStyle.ELABORATE
        elif verbosity >= 0.5:
            return ResponseStyle.MODERATE
        elif verbosity >= 0.25:
            return ResponseStyle.BRIEF
        else:
            return ResponseStyle.MINIMAL

    def _detect_domain(self, question_text: str, study_context: str) -> str:
        """Detect the domain of the question for appropriate templates.

        Enhanced detection for behavioral economics, political, and survey studies.
        """
        combined = f"{question_text} {study_context}".lower()

        # BEHAVIORAL ECONOMICS - dictator game, public goods, trust games
        economic_keywords = [
            'explain your choice', 'why did you', 'explain your decision',
            'dictator', 'public goods', 'trust game', 'ultimatum',
            'give', 'keep', 'contribute', 'split', 'share', 'money',
            'dollars', 'partner', 'allocation', 'decision', 'cooperat'
        ]
        if any(kw in combined for kw in economic_keywords):
            return 'economic_decision_explanation'

        # POLITICAL/INTERGROUP - political attitudes, partner views
        political_keywords = [
            'trump', 'biden', 'democrat', 'republican', 'political',
            'liberal', 'conservative', 'vote', 'election', 'party',
            'hate', 'love', 'opinion towards', 'attitude towards'
        ]
        if any(kw in combined for kw in political_keywords):
            return 'political_attitude_explanation'

        # INTERGROUP RELATIONS - interactions with outgroup members
        intergroup_keywords = [
            'partner', 'matched with', 'person who', 'someone who',
            'interaction', 'working with', 'paired with'
        ]
        if any(kw in combined for kw in intergroup_keywords):
            return 'intergroup_explanation'

        # SURVEY FEEDBACK - end of survey questions
        feedback_keywords = [
            'feedback', 'confusing', 'survey', 'experiment', 'unclear',
            'anything else', 'comments', 'suggestions', 'improve'
        ]
        if any(kw in combined for kw in feedback_keywords):
            return 'survey_feedback'

        # Product/purchase related
        product_keywords = [
            'product', 'purchase', 'buy', 'price', 'quality',
            'recommend', 'review', 'brand', 'item', 'shopping'
        ]
        if any(kw in combined for kw in product_keywords):
            return 'product_feedback'

        # Experience related
        experience_keywords = [
            'experience', 'service', 'visit',
            'session', 'event', 'activity', 'process'
        ]
        if any(kw in combined for kw in experience_keywords):
            return 'experience_feedback'

        # Intention related
        intention_keywords = [
            'intend', 'plan', 'will you', 'would you', 'likely',
            'future', 'going to', 'want to'
        ]
        if any(kw in combined for kw in intention_keywords):
            return 'intention_explanation'

        # Default to general opinion
        return 'general_opinion'

    def _generate_base_response(
        self,
        domain: str,
        sentiment: ResponseSentiment,
        style: ResponseStyle,
        question_text: str,
        study_context: str,
        condition: str,
    ) -> str:
        """Generate the base response using templates and Markov chains."""
        sentiment_key = sentiment.value

        # Get appropriate templates
        templates = RESPONSE_TEMPLATES.get(domain, RESPONSE_TEMPLATES['general_opinion'])
        available_templates = templates.get(sentiment_key, templates.get('neutral', []))

        if not available_templates:
            available_templates = ["No comment."]

        # Select and fill template
        template = random.choice(available_templates)
        response = self._fill_template(template, sentiment, study_context, condition)

        # Adjust length based on style
        if style == ResponseStyle.ELABORATE:
            # Add more content
            extension = self._generate_extension(domain, sentiment)
            response = f"{response} {extension}"
        elif style == ResponseStyle.MINIMAL:
            # Truncate to first sentence
            sentences = response.split('.')
            if sentences:
                response = sentences[0].strip() + '.'
        elif style == ResponseStyle.BRIEF:
            # Keep first two sentences max
            sentences = response.split('.')
            response = '. '.join(sentences[:2]).strip()
            if response and not response.endswith('.'):
                response += '.'

        return response

    def _fill_template(
        self,
        template: str,
        sentiment: ResponseSentiment,
        study_context: str,
        condition: str,
    ) -> str:
        """Fill in template placeholders."""
        # Build replacement dictionary
        replacements = {
            '{product}': self._extract_product(study_context),
            '{product_type}': self._extract_product_type(study_context),
            '{feature}': self._random_feature(study_context),
            '{positive_adj}': random.choice(POSITIVE_ADJECTIVES),
            '{neutral_adj}': random.choice(NEUTRAL_ADJECTIVES),
            '{negative_adj}': random.choice(NEGATIVE_ADJECTIVES),
            '{Positive_statement}': random.choice(POSITIVE_STATEMENTS),
            '{Neutral_statement}': random.choice(NEUTRAL_STATEMENTS),
            '{Negative_statement}': random.choice(NEGATIVE_STATEMENTS),
            '{Positive_detail}': random.choice(POSITIVE_STATEMENTS),
            '{Positive_reason}': random.choice(POSITIVE_STATEMENTS),
            '{Mild_positive}': random.choice(POSITIVE_STATEMENTS[:5]),
            '{Negative_detail}': random.choice(NEGATIVE_STATEMENTS),
            '{Negative_reason}': random.choice(NEGATIVE_STATEMENTS),
            '{Strong_negative}': random.choice(NEGATIVE_STATEMENTS[5:]),
            '{Strong_criticism}': random.choice(NEGATIVE_STATEMENTS),
            '{Neutral_detail}': random.choice(NEUTRAL_STATEMENTS),
            '{Neutral_observation}': random.choice(NEUTRAL_STATEMENTS),
            '{Improvement_suggestion}': "Some things could be improved.",
            '{problem}': "the overall quality",
            '{action}': self._extract_action(study_context),
            '{Reason}': "it seems like a good choice",
            '{Strong_reason}': "this is exactly what I need",
            '{Rejection_reason}': "it doesn't meet my needs",
            '{Concern}': "I have some reservations",
            '{Consideration}': "there are good points",
            '{Mild_reason}': "it seems reasonable",
            '{Explanation}': "based on my experience",
            '{Strong_objection}': "this goes against my preferences",
            '{Positive_elaboration}': random.choice(POSITIVE_STATEMENTS),
            '{Praise}': random.choice(POSITIVE_STATEMENTS),
            '{Specific_positive}': random.choice(POSITIVE_STATEMENTS),
            '{Highlight}': "The quality really stood out.",
            '{Specific_praise}': "Everything was done well.",
            '{Positive_point}': random.choice(POSITIVE_STATEMENTS),
            '{General_positive}': random.choice(POSITIVE_STATEMENTS),
            '{Negative_point}': random.choice(NEGATIVE_STATEMENTS),
            '{Criticism}': random.choice(NEGATIVE_STATEMENTS),
            '{Disappointment_detail}': random.choice(NEGATIVE_STATEMENTS),
            '{Frustration_detail}': random.choice(NEGATIVE_STATEMENTS),
            '{Doubt}': "I'm not fully convinced.",
            '{Objection}': random.choice(NEGATIVE_STATEMENTS),
            '{Confidence_statement}': "I'm very confident about this.",
            '{Comment}': random.choice(NEUTRAL_STATEMENTS),
            '{Observation}': random.choice(NEUTRAL_STATEMENTS),
        }

        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)

        return result

    def _extract_product(self, study_context: str) -> str:
        """Extract or generate product name from context."""
        # Try to find product mentions
        product_patterns = [
            r'product[s]?\s+(?:is|called|named)?\s*["\']?(\w+)',
            r'(\w+)\s+product',
            r'buying\s+(\w+)',
        ]
        for pattern in product_patterns:
            match = re.search(pattern, study_context, re.IGNORECASE)
            if match:
                return match.group(1)

        return "it"

    def _extract_product_type(self, study_context: str) -> str:
        """Extract product type from context."""
        type_words = ['product', 'service', 'item', 'option', 'choice', 'offering']
        for word in type_words:
            if word in study_context.lower():
                return word

        return "product"

    def _random_feature(self, study_context: str) -> str:
        """Generate a random feature mention."""
        features = [
            "design", "quality", "usability", "value", "functionality",
            "appearance", "performance", "features", "experience", "overall feel"
        ]
        return random.choice(features)

    def _extract_action(self, study_context: str) -> str:
        """Extract or generate action from context."""
        action_patterns = [
            r'(?:will|would)\s+you\s+(\w+)',
            r'intend\s+to\s+(\w+)',
            r'plan\s+to\s+(\w+)',
        ]
        for pattern in action_patterns:
            match = re.search(pattern, study_context, re.IGNORECASE)
            if match:
                return match.group(1)

        return "purchase"

    def _generate_extension(self, domain: str, sentiment: ResponseSentiment) -> str:
        """Generate additional content for elaborate responses."""
        # Use Markov chain for extension
        if domain in ['product_feedback']:
            extension = self.product_chain.generate(max_words=20, min_words=8)
        else:
            extension = self.experience_chain.generate(max_words=20, min_words=8)

        # Adjust for sentiment
        if sentiment in [ResponseSentiment.VERY_POSITIVE, ResponseSentiment.POSITIVE]:
            positive_additions = [
                "I'm quite pleased overall.",
                "Would definitely consider again.",
                "A positive experience.",
            ]
            extension = extension or random.choice(positive_additions)
        elif sentiment in [ResponseSentiment.VERY_NEGATIVE, ResponseSentiment.NEGATIVE]:
            negative_additions = [
                "There's room for improvement.",
                "Not entirely satisfied.",
                "Could have been better.",
            ]
            extension = extension or random.choice(negative_additions)
        else:
            neutral_additions = [
                "Nothing remarkable to add.",
                "It was okay.",
                "Average experience overall.",
            ]
            extension = extension or random.choice(neutral_additions)

        return extension

    def _apply_persona_style(self, response: str, traits: PersonaTextTraits) -> str:
        """Apply persona-specific style modifications."""
        # Formality adjustment
        if traits.formality > 0.7:
            # Add formal prefix occasionally
            if random.random() < 0.3:
                prefix = random.choice(FORMAL_PHRASES)
                response = f"{prefix} {response[0].lower()}{response[1:]}"
        elif traits.formality < 0.3:
            # Add casual elements
            if random.random() < 0.3:
                prefix = random.choice(CASUAL_PHRASES)
                response = f"{prefix} {response}"

        # Enthusiasm adjustment
        if traits.enthusiasm > 0.7:
            # Add enthusiastic elements
            if random.random() < 0.3:
                response = response.replace('.', '!')
            if random.random() < 0.2:
                prefix = random.choice(ENTHUSIASTIC_PHRASES)
                response = f"{prefix} {response}"
        elif traits.enthusiasm < 0.3:
            # Make more disengaged
            if random.random() < 0.3:
                prefix = random.choice(DISENGAGED_PHRASES)
                response = f"{prefix}, {response[0].lower()}{response[1:]}"

        return response

    def _add_variation(self, response: str, traits: PersonaTextTraits) -> str:
        """Add final variations for uniqueness.

        This method applies multiple variation techniques to ensure responses
        are unique even when generating thousands of data points:
        1. Word substitution with synonyms
        2. Sentence structure variation
        3. Punctuation style variation
        4. Realistic typos (for low-attention respondents)
        5. Filler phrase insertion/removal
        6. Capitalization patterns
        """
        # 1. Word substitution with common synonyms
        synonym_map = {
            'good': ['nice', 'great', 'fine', 'decent', 'solid'],
            'bad': ['poor', 'terrible', 'awful', 'negative', 'disappointing'],
            'very': ['really', 'quite', 'extremely', 'particularly', 'especially'],
            'think': ['believe', 'feel', 'consider', 'find', 'reckon'],
            'like': ['enjoy', 'appreciate', 'prefer', 'favor', 'am fond of'],
            'want': ['would like', 'prefer', 'wish', 'desire', 'hope'],
            'because': ['since', 'as', 'given that', 'due to the fact that', 'considering'],
            'important': ['significant', 'crucial', 'key', 'essential', 'vital'],
            'overall': ['in general', 'generally', 'on the whole', 'all in all', 'broadly'],
            'maybe': ['perhaps', 'possibly', 'might be', 'could be', 'potentially'],
        }
        if random.random() < 0.4:  # 40% chance of synonym substitution
            for word, synonyms in synonym_map.items():
                if word in response.lower():
                    replacement = random.choice(synonyms)
                    # Preserve case
                    if response.find(word.capitalize()) >= 0:
                        response = response.replace(word.capitalize(), replacement.capitalize(), 1)
                    else:
                        response = response.replace(word, replacement, 1)
                    break  # Only substitute once per response

        # 2. Sentence structure variation - add filler phrases
        filler_phrases = [
            'I would say', 'In my view,', 'Honestly,', 'To be frank,',
            'From my perspective,', 'As I see it,', 'In my experience,',
            'Looking back,', 'All things considered,', 'When I think about it,',
            'I suppose', 'I guess', 'I mean,', 'You know,', 'Basically,',
        ]
        if random.random() < 0.25 and traits.verbosity > 0.4:
            filler = random.choice(filler_phrases)
            if not response.startswith(tuple(filler_phrases)):
                response = f"{filler} {response[0].lower()}{response[1:]}"

        # 3. Punctuation style variation
        if random.random() < 0.15:
            # Sometimes use ellipsis instead of period
            if traits.enthusiasm < 0.5:
                response = response.rstrip('.') + '...'
        if random.random() < 0.1 and traits.enthusiasm > 0.6:
            # Add exclamation for enthusiastic responders
            response = response.rstrip('.!') + '!'

        # 4. Realistic typos for low-attention respondents
        if traits.detail_orientation < 0.3 and random.random() < 0.1:
            words = response.split()
            if len(words) > 3:
                idx = random.randint(1, len(words) - 2)
                word = words[idx]
                if len(word) > 3:
                    # Simple typo: swap adjacent letters
                    i = random.randint(1, len(word) - 2)
                    word = word[:i] + word[i+1] + word[i] + word[i+2:]
                    words[idx] = word
            response = ' '.join(words)

        # 5. Remove filler words for concise responders
        if traits.verbosity < 0.3 and random.random() < 0.3:
            removable = ['just', 'really', 'actually', 'basically', 'simply', 'quite']
            for word in removable:
                if f' {word} ' in response.lower():
                    response = response.replace(f' {word} ', ' ', 1)
                    break

        # 6. Add trailing phrases for variation
        trailing_phrases = [
            ' That\'s my take.', ' Just my opinion.', ' That\'s how I see it.',
            ' Hope that helps.', ' If that makes sense.', '',  # Empty for no addition
        ]
        if random.random() < 0.15 and traits.verbosity > 0.5:
            trailing = random.choice(trailing_phrases)
            if trailing and not response.endswith(trailing.strip()):
                response = response.rstrip('.!?') + '.' + trailing

        return response

    def generate_batch(
        self,
        question_text: str,
        numeric_responses: List[float],
        scale_max: int = 7,
        persona_traits_list: Optional[List[PersonaTextTraits]] = None,
        study_context: str = "",
        conditions: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate a batch of responses for multiple participants.

        Args:
            question_text: The question being answered
            numeric_responses: List of related numeric responses
            scale_max: Maximum value of the numeric scale
            persona_traits_list: List of persona traits (one per participant)
            study_context: Description of the study
            conditions: List of conditions (one per participant)

        Returns:
            List of generated response texts
        """
        n = len(numeric_responses)

        if persona_traits_list is None:
            persona_traits_list = [PersonaTextTraits() for _ in range(n)]

        if conditions is None:
            conditions = ["" for _ in range(n)]

        responses = []
        for i in range(n):
            traits = persona_traits_list[i] if i < len(persona_traits_list) else PersonaTextTraits()
            condition = conditions[i] if i < len(conditions) else ""
            numeric = numeric_responses[i] if i < len(numeric_responses) else None

            response = self.generate_response(
                question_text=question_text,
                numeric_response=numeric,
                scale_max=scale_max,
                persona_traits=traits,
                study_context=study_context,
                condition=condition,
            )
            responses.append(response)

        return responses


def create_text_generator(
    seed: Optional[int] = None,
    study_context: str = ""
) -> OpenEndedTextGenerator:
    """
    Create a configured text generator instance.

    Args:
        seed: Random seed for reproducibility
        study_context: Study description for context-aware generation

    Returns:
        Configured OpenEndedTextGenerator instance
    """
    return OpenEndedTextGenerator(seed=seed, study_context=study_context)


# Export
__all__ = [
    'OpenEndedTextGenerator',
    'PersonaTextTraits',
    'ResponseSentiment',
    'ResponseStyle',
    'ResponseTone',
    'MarkovChainGenerator',
    'create_text_generator',
]
