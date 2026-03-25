"""
Adaptive Behavioral Engine 2.0
==============================

Best-of-both-worlds engine that combines:
- ComprehensiveResponseGenerator (response_library.py): 8-archetype compositional
  system, 225+ domain banks, LIWC linguistic profiling, cross-response voice
  consistency, sentence-level uniqueness enforcement
- SocSim Adapter (socsim_adapter.py): Literature-calibrated economic game
  behavioral modeling with Fehr-Schmidt utility functions
- NEW narrative intent builders: creative_belief, personal_disclosure,
  creative_narrative, personal_story (Brotherton 2013, Pennebaker 1997,
  Green & Brock 2000)
- NEW variation phrase system for dedup diversity
- NEW narrative-specific archetype weighting

This engine is a WRAPPER — it delegates to ComprehensiveResponseGenerator
for all standard intents, and adds dedicated handling for narrative/creative
intents that previously fell through to the generic else clause.

The original Template Engine and Behavioral Engine remain untouched.

Scientific References:
- Brotherton et al. (2013): Conspiracy theory belief measurement
- Pennebaker (1997): Self-disclosure and emotional expression
- Green & Brock (2000): Narrative transportation theory
- Engel (2011): Dictator game meta-analysis
- Fehr & Schmidt (1999): Inequity aversion in economic games

Version: 2.0.0
"""

__version__ = "2.0.0"

import random
import re
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Narrative intent taxonomy
# ──────────────────────────────────────────────────────────────

_NARRATIVE_INTENTS = frozenset({
    "creative_belief",       # Conspiracy theories, paranormal, superstitions
    "personal_disclosure",   # Secrets, confessions, embarrassing admissions
    "creative_narrative",    # "Tell us your wildest/craziest/funniest X"
    "personal_story",        # "Describe a time when..." experiential prompts
    "hypothetical",          # "What if...", "Imagine..." scenario prompts
})

# ──────────────────────────────────────────────────────────────
# Narrative position builders — intent-specific position statements
# These fill the gap in _build_position() where creative/narrative
# intents previously fell to the generic else clause.
# ──────────────────────────────────────────────────────────────

_NARRATIVE_POSITIONS: Dict[str, Dict[str, List[str]]] = {
    "creative_belief": {
        "very_positive": [
            "I 100% believe {topic} and honestly more people should look into it",
            "yeah I think there's definitely something to {topic}, the evidence is there if you look",
            "I've always believed in {topic} and the more I read the more convinced I get",
            "call me crazy but {topic} makes total sense to me",
            "{topic} is real as far as I'm concerned and I have my reasons",
        ],
        "positive": [
            "I think there's probably some truth to {topic}",
            "I wouldn't say I'm a full believer but {topic} does make you wonder",
            "honestly {topic} has some compelling arguments behind it",
            "I'm somewhat open to {topic}, not gonna lie",
            "I lean toward believing {topic} at least partially",
        ],
        "neutral": [
            "I go back and forth on {topic} honestly",
            "I don't know what to think about {topic}, there's arguments both ways",
            "{topic} is one of those things where I'm genuinely 50/50",
            "I keep an open mind about {topic} but I'm not convinced either way",
            "it's hard to know what to believe when it comes to {topic}",
        ],
        "negative": [
            "I'm pretty skeptical about {topic} to be honest",
            "{topic} doesn't really hold up when you look at the evidence",
            "I used to be more open to {topic} but now I'm doubtful",
            "I think {topic} is mostly wishful thinking personally",
            "I don't buy {topic} but I get why some people do",
        ],
        "very_negative": [
            "{topic} is complete nonsense in my opinion",
            "I absolutely don't believe in {topic}, it's been debunked countless times",
            "honestly anyone who believes {topic} hasn't done their research",
            "{topic} is the kind of thing that sounds good until you actually think about it",
            "I have zero patience for {topic}, it's been disproven over and over",
        ],
    },
    "personal_disclosure": {
        "very_positive": [
            "ok so this is something I don't usually share but",
            "honestly this is pretty personal but I'll say it",
            "I've been wanting to get this off my chest for a while",
            "so there's something about me that most people don't know",
            "I trust this is anonymous so here goes",
        ],
        "positive": [
            "I don't mind sharing this actually",
            "this is something I've thought about opening up about",
            "it's not my deepest secret but it's personal",
            "I guess I can share this since it's anonymous",
            "ok so here's something about me",
        ],
        "neutral": [
            "I'm not sure how personal to get here but",
            "this is kind of hard to talk about honestly",
            "I don't usually open up about this kind of thing",
            "I'll share something but I'm keeping the details vague",
            "it's complicated but here's the gist",
        ],
        "negative": [
            "I'd rather not go too deep into this honestly",
            "this is uncomfortable to talk about but",
            "I'm not great at opening up about personal stuff",
            "I'll give a surface level answer because this is touchy",
            "this brings up feelings I'd rather not deal with",
        ],
        "very_negative": [
            "I really don't want to talk about this",
            "this is too personal honestly I'll keep it brief",
            "I'm going to be vague because this is painful",
            "I'd pass on this question if I could",
            "this hits too close to home but fine",
        ],
    },
    "creative_narrative": {
        "very_positive": [
            "oh man ok so this is a good one",
            "I have the perfect story for this actually",
            "so this one time",
            "lol ok this is actually hilarious let me tell you",
            "the best one I can think of is from last year",
        ],
        "positive": [
            "I've got a decent one for this",
            "so there was this time when",
            "here's one that comes to mind",
            "ok so my story for this would be",
            "I can think of a few but the best one is",
        ],
        "neutral": [
            "I'm trying to think of a good one",
            "nothing super dramatic comes to mind but",
            "I guess the closest thing would be",
            "let me think... ok so there was this one time",
            "I don't have anything amazing but",
        ],
        "negative": [
            "I don't really have a great story for this",
            "nothing really comes to mind honestly",
            "I'm not great at telling stories but",
            "this is hard to think of on the spot",
            "I'll try but I don't have much",
        ],
        "very_negative": [
            "I can't really think of anything",
            "I don't have a story for this sorry",
            "nothing comes to mind",
            "I'm drawing a blank on this one",
            "pass honestly, nothing interesting to share",
        ],
    },
    "personal_story": {
        "very_positive": [
            "so this happened to me a while back and it really stuck with me",
            "I remember this really clearly because it was a turning point",
            "ok so the experience that comes to mind is",
            "this is actually something I think about a lot",
            "the most vivid example I have is from",
        ],
        "positive": [
            "I've had a few experiences with this but one stands out",
            "so there was this situation where",
            "the experience I remember most is",
            "here's what happened to me personally",
            "one time that comes to mind is",
        ],
        "neutral": [
            "I've had some mixed experiences with this",
            "it's hard to pick just one example but",
            "my experience has been kind of all over the place",
            "I don't have one defining moment but generally",
            "I can think of a few things but nothing dramatic",
        ],
        "negative": [
            "I'd rather forget this experience honestly",
            "this wasn't great for me, basically what happened was",
            "I have a negative experience with this unfortunately",
            "so the thing that happened was kind of rough",
            "this is one of those memories I don't love revisiting",
        ],
        "very_negative": [
            "this brings up bad memories honestly",
            "the experience I had was genuinely terrible",
            "I don't like thinking about this but",
            "what happened was pretty awful and I still think about it",
            "this was one of the worst experiences of my life honestly",
        ],
    },
    "hypothetical": {
        "very_positive": [
            "I'd absolutely {topic} without hesitation",
            "in that scenario I'd go all in for sure",
            "honestly I think {topic} would be amazing",
            "if that happened I'd be thrilled, here's what I'd do",
            "oh I'd definitely {topic}, no question about it",
        ],
        "positive": [
            "I'd probably {topic} in that situation",
            "I think I'd go for it, seems like a good idea",
            "in that case I'd lean toward {topic}",
            "I'd be open to {topic} if the circumstances were right",
            "yeah I think I'd {topic} most likely",
        ],
        "neutral": [
            "that's a tough hypothetical honestly",
            "I'm not sure what I'd do in that situation",
            "it would really depend on the specifics",
            "I'd have to think about it more before deciding",
            "I could see myself going either way on {topic}",
        ],
        "negative": [
            "I probably wouldn't {topic} if I'm being honest",
            "in that situation I'd be hesitant",
            "I don't think {topic} would work out well",
            "I'd lean against it in that scenario",
            "that doesn't sound like something I'd do",
        ],
        "very_negative": [
            "absolutely not, I would never {topic}",
            "there's no way I'd {topic} in that scenario",
            "hard pass, that sounds terrible",
            "I'd refuse {topic} completely",
            "not a chance, here's why",
        ],
    },
}


# ──────────────────────────────────────────────────────────────
# Narrative corpus responses — complete standalone responses
# for each narrative domain, stratified by sentiment.
# Used when topic is unintelligible (variable names, etc.)
# ──────────────────────────────────────────────────────────────

_NARRATIVE_CORPUS: Dict[str, Dict[str, List[str]]] = {
    "creative_belief": {
        "positive": [
            "I think there's stuff out there the government doesn't want us to know about. Like not tinfoil hat stuff but real cover-ups that we'll find out about in 50 years.",
            "I'm pretty sure aliens have visited earth at some point. The universe is too big for us to be the only ones, and some of those sightings are hard to explain away.",
            "I believe in karma, like genuine cosmic karma. Every time I've done something selfish it comes back to bite me eventually.",
            "I've had experiences that science can't explain. Déjà vu that was too specific, dreams that came true. I don't care if people think I'm weird.",
            "I think Big Pharma suppresses certain cures because treating disease is more profitable than curing it. That's just business logic.",
            "I believe in ghosts honestly. I grew up in an old house and things would move on their own. You can't convince me otherwise.",
        ],
        "neutral": [
            "I don't know what to believe about most conspiracy theories. Some seem crazy but then some end up being true so who knows.",
            "I keep an open mind about paranormal stuff without fully committing either way. I've seen some weird things but also know our brains play tricks.",
            "I think the truth is probably somewhere in the middle for most of these theories. The official story isn't always complete but that doesn't mean the wildest version is right.",
            "I used to dismiss all of this stuff but as I've gotten older I'm less certain about what's real and what isn't.",
        ],
        "negative": [
            "I don't believe in conspiracy theories generally. Most of them fall apart when you look at the actual evidence.",
            "I'm a pretty skeptical person so supernatural stuff just doesn't work for me. I need proof.",
            "I used to be more into alternative theories but honestly most of them are just pattern recognition gone wrong.",
            "I think people believe in conspiracies because it's comforting to think someone is in control, even if it's a shadow government.",
        ],
    },
    "personal_disclosure": {
        "positive": [
            "Something most people don't know about me is that I almost dropped out of college my sophomore year. I was dealing with really bad anxiety and couldn't make it to classes. I got help eventually but it changed how I see mental health.",
            "I'll share that I didn't learn to swim until I was 22. I grew up too embarrassed to tell anyone and it limited what I could do socially for years.",
            "Honestly my biggest secret is that I'm the first person in my family to go to college and I feel like an impostor every single day.",
            "I never told anyone but I used to steal books from the library as a kid because we couldn't afford them. I've donated a lot of money to libraries since then to make up for it.",
        ],
        "neutral": [
            "I'm not great at sharing personal stuff but I guess something I don't talk about much is that my parents got divorced when I was young and it affected me more than I let on.",
            "The thing that comes to mind is that I switched careers completely at 30 and everyone thought I was crazy. It worked out but it was terrifying.",
            "I'll say that I deal with imposter syndrome more than people realize. I seem confident but internally I'm always second-guessing myself.",
        ],
        "negative": [
            "I'd rather not share too much but I'll say I went through a really difficult period a few years ago that I haven't fully processed yet.",
            "I'm going to keep this vague but I made a decision in my twenties that I still regret and think about regularly.",
            "Without going into details, I've experienced something that makes it hard for me to trust people easily. I'm working on it.",
        ],
    },
    "creative_narrative": {
        "positive": [
            "Ok so the wildest thing that ever happened to me was when I accidentally got on the wrong flight and ended up in Dallas instead of Denver. I didn't realize until we landed and I saw the signs. Had to rebook everything and it cost me a fortune but honestly it makes for a great story.",
            "This one time at work my boss accidentally sent a company-wide email that was meant for HR complaining about a specific employee. The whole office went silent and then everyone's phone started buzzing at once. Most chaotic day I've ever experienced.",
            "My funniest story is when I was trying to impress a date by cooking dinner and I set off the fire alarm in my entire apartment building. 200 people standing outside in the cold because I forgot about the garlic bread.",
            "The craziest coincidence I've experienced was running into my childhood best friend in an airport in Tokyo. We hadn't spoken in 15 years and were on the same layover.",
        ],
        "neutral": [
            "I don't have anything super wild but one time I found a wallet with $2000 in it and returned it. The guy gave me $20 as a thank you which felt kind of insulting honestly.",
            "The most interesting thing I can think of is that I once got stuck in an elevator for three hours with a complete stranger and we ended up becoming friends.",
            "I guess the most notable thing was when I witnessed a car accident and had to give a statement to the police. Nothing crazy but it was unsettling.",
        ],
        "negative": [
            "I can't think of anything that great honestly. My life is pretty routine.",
            "Nothing really comes to mind. I don't tend to get into crazy situations.",
            "I'm not the type of person who has wild stories. I keep things pretty low key.",
        ],
    },
    "personal_story": {
        "positive": [
            "The experience that stands out most is when I volunteered at a homeless shelter for the first time. This one guy I served food to started talking to me about his PhD in physics. It completely changed how I think about homelessness and what circumstances can do to anyone.",
            "What I remember most vividly is the moment I realized I wanted to change careers. I was sitting in a meeting pretending to care about quarterly projections and it hit me like a ton of bricks that I was wasting my life. I quit two months later.",
            "The most impactful experience I've had was studying abroad in college. Living in a completely different culture for six months taught me more about myself than the previous 20 years combined.",
        ],
        "neutral": [
            "My experience with this has been mixed. Some things went well, others didn't. I think the most relevant situation was when I had to make a big decision without much information and just went with my gut. It turned out ok but I'm not sure I'd do the same thing again.",
            "I've had a few relevant experiences but nothing that dramatically changed my perspective. It's more like a gradual accumulation of small moments that shaped how I think about it.",
        ],
        "negative": [
            "The experience I had was honestly not great. Without going into too much detail, it confirmed some of my worst fears about how things work.",
            "What happened to me was a learning experience but not a pleasant one. I came away from it more cynical than before.",
            "I don't love talking about this particular experience because it still bothers me. Let's just say it taught me to be more careful about who I trust.",
        ],
    },
    "hypothetical": {
        "positive": [
            "If I were in that situation I'd go for it without thinking twice. Life's too short to play it safe when the upside is that good.",
            "I'd take the opportunity in a heartbeat. The potential reward far outweighs the risk in my mind and I've always been someone who'd rather try and fail than wonder what if.",
            "In that scenario I think I'd surprise myself with how quickly I'd say yes. Sometimes your gut knows before your brain catches up.",
        ],
        "neutral": [
            "That's genuinely tough. Part of me would want to go for it but the practical side would probably hold me back. I'd probably end up overthinking it and then decide last minute.",
            "I honestly don't know what I'd do until I was actually in that situation. It's easy to say one thing hypothetically but reality is different.",
            "I'd probably go back and forth for a while and then make a snap decision based on how I felt in the moment. Not the most strategic approach but that's how I work.",
        ],
        "negative": [
            "There's no way I'd do that. The risk is too high and the downside is permanent. I've seen what happens to people who make impulsive decisions in situations like that.",
            "I'd pass on that without hesitation. It's not worth it when you consider everything that could go wrong, and I've learned the hard way to be more cautious.",
            "Hard no from me. I know my limits and that scenario is way outside my comfort zone for good reason.",
        ],
    },
}


# ──────────────────────────────────────────────────────────────
# Narrative concrete detail banks
# ──────────────────────────────────────────────────────────────

_NARRATIVE_DETAILS: Dict[str, Dict[str, List[str]]] = {
    "creative_belief": {
        "positive": [
            "I mean look at MKUltra, that was a real conspiracy that got declassified",
            "my grandmother always said she could feel things before they happened and I believe her",
            "there are documented government programs that were denied for decades",
            "I watched this documentary that laid it all out pretty convincingly",
            "the Snowden revelations proved that what people called conspiracy theories were true",
            "I've had personal experiences that can't be explained by coincidence",
            "certain pharmaceutical companies have literally been caught hiding studies",
            "the number of deathbed confessions from intelligence officials is telling",
        ],
        "negative": [
            "every conspiracy theory I've actually researched falls apart under scrutiny",
            "people see patterns because our brains are wired for it, not because the patterns are real",
            "Occam's razor explains most of these better than the conspiracy does",
            "I took a critical thinking course that basically inoculated me against this stuff",
            "the amount of coordination required for most conspiracies makes them practically impossible",
            "I used to believe some of this stuff until I looked at the actual data",
        ],
        "neutral": [
            "some conspiracy theories turned out to be true and some are obviously nonsense",
            "I think healthy skepticism is different from conspiracy thinking",
            "the line between conspiracy theory and legitimate concern isn't always clear",
            "my friends are split on this, some think I'm too credulous and others think I'm too skeptical",
        ],
    },
    "personal_disclosure": {
        "positive": [
            "it took me years to be comfortable sharing this",
            "my therapist helped me realize it's ok to talk about",
            "I've found that being honest about this actually helps other people open up too",
            "once I started being open about it my relationships got so much better",
        ],
        "negative": [
            "I still haven't told my family about this",
            "the last time I shared something personal it didn't go well",
            "I keep this to myself for a reason",
            "some things are better left unsaid honestly",
        ],
        "neutral": [
            "it's not something I bring up in normal conversation",
            "I've only told a couple of close friends about this",
            "I don't think about it that much but when I do it's complicated",
        ],
    },
    "creative_narrative": {
        "positive": [
            "everyone who was there still talks about it",
            "I couldn't make this up if I tried",
            "the timing was so perfect it was like a movie",
            "looking back it's hilarious but in the moment I was mortified",
            "my friends still bring this up at every gathering",
            "the look on everyone's face was priceless",
        ],
        "negative": [
            "it's not really a great story but it's all I've got",
            "I wish I had something more interesting to share",
            "most of my stories are pretty mundane to be honest",
        ],
        "neutral": [
            "it wasn't the craziest thing ever but it was definitely unusual",
            "looking back it's kind of funny but at the time it wasn't",
            "the story gets better every time I tell it honestly",
        ],
    },
    "personal_story": {
        "positive": [
            "it really changed how I see the world",
            "I think about this experience all the time",
            "it was one of those moments that shifts your perspective permanently",
            "the whole thing taught me more than any class ever could",
            "I'm grateful it happened even though it was hard at the time",
        ],
        "negative": [
            "I wish I could go back and handle it differently",
            "the whole experience left a bad taste in my mouth",
            "I learned from it but I wouldn't want to repeat it",
            "it confirmed some of my worst fears unfortunately",
        ],
        "neutral": [
            "it wasn't life-changing but it was definitely memorable",
            "I'm not sure what lesson to take from it honestly",
            "it just kind of happened and I moved on",
        ],
    },
    "hypothetical": {
        "positive": [
            "the upside is just too good to pass up",
            "I've been in similar situations and the bold move paid off",
            "my gut says go for it and I usually trust my gut",
            "the worst case scenario isn't even that bad honestly",
        ],
        "negative": [
            "I've seen how badly this can go wrong firsthand",
            "the risk-reward just doesn't add up for me",
            "my past experience with similar situations has been negative",
            "too many things could go wrong for my comfort",
        ],
        "neutral": [
            "it's one of those situations where there's no clear right answer",
            "I keep going back and forth on it honestly",
            "it depends on so many variables that it's hard to commit either way",
        ],
    },
}


# ──────────────────────────────────────────────────────────────
# Variation phrases for dedup diversity (from Template Engine)
# Adapted from persona_library.py variation system
# ──────────────────────────────────────────────────────────────

_VARIATION_PHRASES = {
    "time_phrases": [
        "At first,", "Initially,", "After thinking about it,",
        "Upon reflection,", "Looking at it now,", "Over time,",
        "As I've gotten older,", "Recently,", "Back then,",
        "In hindsight,", "After a while,", "Eventually,",
        "These days,", "Lately,", "When I first heard about it,",
    ],
    "personal_phrases": [
        "Personally,", "For me,", "In my experience,",
        "From my perspective,", "Speaking for myself,",
        "Based on what I've seen,", "From where I stand,",
        "In my own life,", "To be honest,", "If I'm being real,",
        "From my point of view,", "As someone who,",
        "Having gone through this,", "As far as I can tell,",
    ],
    "certainty_phrases": [
        "I'm fairly sure that", "I believe that", "I think",
        "I feel like", "It seems to me that", "I'm pretty confident that",
        "I'd say that", "My sense is that", "I suspect that",
        "I'm convinced that", "In my opinion,", "The way I see it,",
        "I'd argue that", "My gut says",
    ],
    "ending_phrases": [
        " That's my take on it.", " That's how I see it.",
        " That's just how I feel.", " Take it or leave it.",
        " But that's just me.", " For what it's worth.",
        " At least that's my perspective.", " But what do I know.",
        "", "", "", "",  # 33% chance of no ending — more natural
    ],
}


# ──────────────────────────────────────────────────────────────
# Narrative archetype weight overrides
# For narrative intents, bias toward story_first, emotional_burst,
# stream — away from reasoning_first, list_style
# ──────────────────────────────────────────────────────────────

_NARRATIVE_ARCHETYPE_WEIGHTS = {
    ('casual', 'low'):  {'direct_answer': 30, 'stream': 30, 'emotional_burst': 20, 'story_first': 15, 'rhetorical': 5},
    ('casual', 'mid'):  {'story_first': 30, 'stream': 25, 'emotional_burst': 20, 'direct_answer': 15, 'rhetorical': 10},
    ('casual', 'high'): {'story_first': 35, 'emotional_burst': 25, 'stream': 20, 'concession': 10, 'rhetorical': 10},
    ('moderate', 'low'):  {'direct_answer': 35, 'stream': 25, 'story_first': 20, 'emotional_burst': 15, 'rhetorical': 5},
    ('moderate', 'mid'):  {'story_first': 30, 'emotional_burst': 20, 'stream': 20, 'direct_answer': 15, 'concession': 10, 'rhetorical': 5},
    ('moderate', 'high'): {'story_first': 30, 'emotional_burst': 20, 'concession': 15, 'stream': 15, 'reasoning_first': 10, 'rhetorical': 10},
    ('formal', 'low'):  {'direct_answer': 35, 'story_first': 25, 'reasoning_first': 20, 'concession': 15, 'stream': 5},
    ('formal', 'mid'):  {'story_first': 30, 'reasoning_first': 20, 'concession': 20, 'emotional_burst': 15, 'direct_answer': 15},
    ('formal', 'high'): {'story_first': 30, 'reasoning_first': 20, 'emotional_burst': 15, 'concession': 15, 'rhetorical': 10, 'list_style': 10},
}


# ══════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════

class AdaptiveBehavioralEngineV2:
    """Adaptive Behavioral Engine 2.0 — best-of-both-worlds text generator.

    Wraps ComprehensiveResponseGenerator and extends it with:
    1. Narrative intent detection and dedicated position builders
    2. Narrative corpus responses for unintelligible topics
    3. Narrative concrete detail banks
    4. Variation phrase system for dedup diversity
    5. Narrative-specific archetype weighting
    6. SocSim adapter integration for economic games

    Drop-in replacement for ComprehensiveResponseGenerator — same
    generate() interface, same study_context API, enhanced output.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the engine.

        Args:
            seed: Random seed for reproducibility. Passed through to
                  ComprehensiveResponseGenerator.
        """
        self.seed = seed
        self._base_generator = None
        self._socsim_available = False
        self._init_errors: List[str] = []

        # Import and initialize the base ComprehensiveResponseGenerator
        try:
            from .response_library import ComprehensiveResponseGenerator
            self._base_generator = ComprehensiveResponseGenerator(seed=seed)
        except Exception as e:
            self._init_errors.append(f"ComprehensiveResponseGenerator init failed: {e}")
            logger.error("ABE2.0: Failed to initialize base generator: %s", e)

        # Track generated responses for uniqueness
        self._used_sentences: set = set()
        self._generation_count: int = 0

    # ── Public API ────────────────────────────────────────────

    def set_study_context(self, context: Dict[str, Any]) -> None:
        """Forward study context to the base generator."""
        if self._base_generator is not None:
            self._base_generator.set_study_context(context)

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
        behavioral_profile: Optional[Dict[str, Any]] = None,
        question_intent: str = "",
        question_context: str = "",
    ) -> str:
        """Generate an open-ended response with narrative-enhanced capabilities.

        Same interface as ComprehensiveResponseGenerator.generate() for
        drop-in compatibility. When the detected intent is narrative
        (creative_belief, personal_disclosure, etc.), uses dedicated
        narrative builders. Otherwise delegates to the base generator.

        Args:
            question_text: The question being answered
            sentiment: Overall sentiment (very_positive/positive/neutral/negative/very_negative)
            persona_verbosity: How verbose (0-1)
            persona_formality: How formal (0-1)
            persona_engagement: How engaged (0-1)
            condition: Experimental condition
            question_name: Unique question identifier
            participant_seed: Random seed for this participant
            behavioral_profile: Behavioral profile dict
            question_intent: Pre-computed intent from engine
            question_context: Raw user-provided context
        Returns:
            Generated response text
        """
        self._generation_count += 1

        # Determine if this is a narrative intent
        _effective_intent = question_intent or ""

        if _effective_intent in _NARRATIVE_INTENTS:
            # Use our dedicated narrative builder
            response = self._generate_narrative_response(
                question_text=question_text,
                sentiment=sentiment,
                intent=_effective_intent,
                persona_verbosity=persona_verbosity,
                persona_formality=persona_formality,
                persona_engagement=persona_engagement,
                condition=condition,
                question_name=question_name,
                participant_seed=participant_seed,
                behavioral_profile=behavioral_profile,
                question_context=question_context,
            )
            if response and response.strip() and len(response.strip()) >= 10:
                return response
            # Fall through to base generator if narrative builder returned empty

        # Delegate to base ComprehensiveResponseGenerator for all standard intents
        if self._base_generator is not None:
            try:
                response = self._base_generator.generate(
                    question_text=question_text,
                    sentiment=sentiment,
                    persona_verbosity=persona_verbosity,
                    persona_formality=persona_formality,
                    persona_engagement=persona_engagement,
                    condition=condition,
                    question_name=question_name,
                    participant_seed=participant_seed,
                    behavioral_profile=behavioral_profile,
                    question_intent=question_intent,
                    question_context=question_context,
                )
                if response and response.strip():
                    # Apply variation phrase for additional diversity
                    response = self._apply_variation_phrase(
                        response, participant_seed, persona_formality,
                    )
                    return response
            except Exception as e:
                logger.warning("ABE2.0: Base generator error: %s", e)

        # Last resort
        return self._last_resort_response(sentiment, participant_seed)

    # ── Narrative response builder ────────────────────────────

    def _generate_narrative_response(
        self,
        question_text: str,
        sentiment: str,
        intent: str,
        persona_verbosity: float,
        persona_formality: float,
        persona_engagement: float,
        condition: str,
        question_name: str,
        participant_seed: int,
        behavioral_profile: Optional[Dict[str, Any]],
        question_context: str,
    ) -> str:
        """Build a narrative-intent-specific response.

        This is the core value-add of ABE 2.0 — dedicated handling for
        creative_belief, personal_disclosure, creative_narrative, and
        personal_story intents that previously produced generic output.
        """
        # Create seeded RNG (same deterministic approach as base generator)
        combined_id = f"{question_name}|{question_text}"
        if combined_id:
            question_hash = sum(ord(c) * (i + 1) * 31 for i, c in enumerate(combined_id[:200]))
        else:
            question_hash = 0
        unique_seed = (participant_seed + question_hash) % (2**31)
        rng = random.Random(unique_seed)

        # Extract traits
        _traits = {}
        if behavioral_profile and isinstance(behavioral_profile, dict):
            _traits = behavioral_profile.get('trait_profile', {})
        _engagement = _traits.get('attention', persona_engagement)
        _verbosity = _traits.get('verbosity', persona_verbosity)
        _formality = _traits.get('formality', persona_formality)
        _extremity = _traits.get('extremity', 0.4)
        _is_straight_lined = behavioral_profile.get('straight_lined', False) if behavioral_profile else False

        # Ultra-short handler for disengaged participants
        if (_engagement < 0.2 or _is_straight_lined) and rng.random() < 0.6:
            return self._build_narrative_ultra_short(intent, sentiment, rng)
        if _engagement < 0.35 and _verbosity < 0.3 and rng.random() < 0.35:
            return self._build_narrative_ultra_short(intent, sentiment, rng)

        # Map sentiment for lookup (normalize very_positive/very_negative)
        _sent_key = sentiment
        if _sent_key == "very_positive":
            _sent_key = "very_positive" if rng.random() < 0.7 else "positive"
        elif _sent_key == "very_negative":
            _sent_key = "very_negative" if rng.random() < 0.7 else "negative"

        # Extract topic for template interpolation
        topic = self._extract_narrative_topic(
            question_text, question_context, intent,
        )

        # Select archetype with narrative weighting
        _arch = self._select_narrative_archetype(_formality, _engagement, rng)

        # Build the response using narrative-specific components
        response = self._assemble_narrative_response(
            intent=intent,
            sentiment=_sent_key,
            topic=topic,
            archetype=_arch,
            rng=rng,
            traits=_traits,
            behavioral_profile=behavioral_profile or {},
            condition=condition,
            formality=_formality,
            verbosity=_verbosity,
            engagement=_engagement,
            extremity=_extremity,
        )

        if not response or len(response.strip()) < 10:
            # Try corpus fallback
            response = self._get_narrative_corpus_response(intent, sentiment, rng)

        if not response:
            return ""

        # Post-processing pipeline (mirrors base generator)
        response = self._apply_narrative_polish(response, _formality, _engagement, rng)

        # Apply variation phrase for diversity
        response = self._apply_variation_phrase(response, participant_seed, _formality)

        # Ensure proper punctuation
        response = response.strip()
        if response and response[-1] not in '.!?':
            response += '.'
        # Capitalize first letter — but NOT for very disengaged participants
        # whose lowercase was deliberately set by _apply_narrative_polish
        if response and _engagement >= 0.35:
            response = response[0].upper() + response[1:]

        return response

    # ── Narrative assembly ────────────────────────────────────

    def _assemble_narrative_response(
        self,
        intent: str,
        sentiment: str,
        topic: str,
        archetype: str,
        rng: random.Random,
        traits: Dict[str, float],
        behavioral_profile: Dict[str, Any],
        condition: str,
        formality: float,
        verbosity: float,
        engagement: float,
        extremity: float,
    ) -> str:
        """Assemble a narrative response from components."""
        # Get position statement
        position = self._get_narrative_position(intent, sentiment, topic, rng)

        # Get detail/elaboration
        detail = self._get_narrative_detail(intent, sentiment, rng)

        # Assemble based on archetype
        if archetype == 'direct_answer':
            response = position
            if rng.random() < 0.3 and detail:
                response += ". " + detail

        elif archetype == 'story_first':
            # For narrative intents, the "position" IS often a story opener
            if intent in ("creative_narrative", "personal_story"):
                # Position already reads like a story start
                response = position
                if detail and rng.random() < 0.6:
                    response += ". " + detail
            else:
                # Start with detail (anecdote), then position (conclusion)
                if detail:
                    bridges = [" So basically, ", " That's why ", " Which is why ",
                               " Point being, ", " Anyway, "]
                    response = detail + rng.choice(bridges) + position
                else:
                    response = position

        elif archetype == 'emotional_burst':
            # For narrative: lead with feeling, then content
            _bursts_by_sent = {
                "very_positive": ["This genuinely excites me.", "I love this honestly.",
                                  "Ok I have strong feelings about this."],
                "positive": ["I feel good about this.", "This resonates with me.",
                             "I actually have thoughts on this."],
                "neutral": ["Hmm, interesting question.", "I'm not sure how I feel about this.",
                            "This is a mixed bag for me."],
                "negative": ["This is frustrating honestly.", "I don't love this topic.",
                             "Ugh, this one's tough."],
                "very_negative": ["This genuinely bothers me.", "I hate thinking about this.",
                                  "Ok this makes me angry."],
            }
            _burst_pool = _bursts_by_sent.get(sentiment, _bursts_by_sent["neutral"])
            response = rng.choice(_burst_pool) + " " + position

        elif archetype == 'stream':
            # Conversational stream-of-consciousness
            _trail = rng.choice(["", " you know?", " idk", " but yeah",
                                 " I guess", " so yeah", " or whatever"])
            if detail:
                response = position + ", " + detail.lower() + _trail
            else:
                response = position + _trail

        elif archetype == 'concession':
            # "I get why people X, but personally..."
            _conc_prefixes = [
                "I get why people have different views on this, but",
                "look I understand the other side, but",
                "I know not everyone agrees, but",
                "this is controversial but",
                "people might disagree but",
            ]
            response = rng.choice(_conc_prefixes) + " " + position

        elif archetype == 'rhetorical':
            # Rhetorical question → answer
            _rqs = {
                "creative_belief": [
                    "Do I actually believe in this stuff?",
                    "Is there any truth to it?",
                    "Am I the kind of person who buys into this?",
                ],
                "personal_disclosure": [
                    "How personal am I willing to get here?",
                    "Do I really want to share this?",
                    "Is this the right place to open up?",
                ],
                "creative_narrative": [
                    "What's the craziest thing I can think of?",
                    "Do I have a good story for this?",
                    "Where do I even start?",
                ],
                "personal_story": [
                    "What comes to mind first?",
                    "Which experience stands out the most?",
                    "How do I even pick one?",
                ],
                "hypothetical": [
                    "What would I actually do?",
                    "Am I being honest with myself here?",
                    "Could I really see myself doing that?",
                ],
            }
            _rq_pool = _rqs.get(intent, ["What do I think about this?"])
            _cap_pos = (position[0].upper() + position[1:]) if position else "I have my thoughts."
            response = rng.choice(_rq_pool) + " " + _cap_pos

        else:
            # Fallback: position + detail
            response = position
            if detail and rng.random() < 0.5:
                response += ". " + detail

        # Optional elaboration for verbose participants
        if verbosity > 0.55 and rng.random() < (verbosity * 0.4):
            _extra_detail = self._get_narrative_detail(intent, sentiment, rng)
            if _extra_detail and _extra_detail != detail:
                response += ". " + _extra_detail

        return response

    # ── Component builders ────────────────────────────────────

    def _get_narrative_position(
        self, intent: str, sentiment: str, topic: str, rng: random.Random,
    ) -> str:
        """Get a narrative-specific position statement."""
        bank = _NARRATIVE_POSITIONS.get(intent, {})
        sent_bank = bank.get(sentiment, bank.get("neutral", []))
        if not sent_bank:
            # Fallback to nearest sentiment
            for fallback_sent in ["neutral", "positive", "negative"]:
                sent_bank = bank.get(fallback_sent, [])
                if sent_bank:
                    break
        if not sent_bank:
            return f"I have thoughts about {topic}" if topic != "this" else "I have thoughts on this"

        position = rng.choice(sent_bank)
        # Interpolate topic
        if "{topic}" in position:
            position = position.replace("{topic}", topic)
        return position

    def _get_narrative_detail(
        self, intent: str, sentiment: str, rng: random.Random,
    ) -> str:
        """Get a narrative-specific concrete detail."""
        # Map sentiment to detail bank key
        if sentiment in ("very_positive", "positive"):
            _dk = "positive"
        elif sentiment in ("very_negative", "negative"):
            _dk = "negative"
        else:
            _dk = "neutral"

        bank = _NARRATIVE_DETAILS.get(intent, {})
        detail_bank = bank.get(_dk, [])
        if not detail_bank:
            # Try neutral fallback
            detail_bank = bank.get("neutral", [])
        if not detail_bank:
            return ""
        return rng.choice(detail_bank)

    def _get_narrative_corpus_response(
        self, intent: str, sentiment: str, rng: random.Random,
    ) -> str:
        """Get a complete standalone corpus response for narrative intent."""
        bank = _NARRATIVE_CORPUS.get(intent, {})
        # Map sentiment
        if sentiment in ("very_positive", "positive"):
            _sk = "positive"
        elif sentiment in ("very_negative", "negative"):
            _sk = "negative"
        else:
            _sk = "neutral"

        sent_bank = bank.get(_sk, [])
        if not sent_bank:
            # Try any available
            for k in ("neutral", "positive", "negative"):
                sent_bank = bank.get(k, [])
                if sent_bank:
                    break
        if not sent_bank:
            return ""
        return rng.choice(sent_bank)

    def _build_narrative_ultra_short(
        self, intent: str, sentiment: str, rng: random.Random,
    ) -> str:
        """Very short response for disengaged participants on narrative questions."""
        _pools = {
            "creative_belief": {
                "positive": ["yeah I believe it", "definitely real", "makes sense to me",
                             "I believe in that stuff", "sure why not"],
                "neutral": ["idk maybe", "could be true", "who knows", "not sure",
                            "50/50 on it"],
                "negative": ["nah", "don't believe it", "fake", "doubt it",
                             "not real", "nonsense"],
            },
            "personal_disclosure": {
                "positive": ["I have something but its personal", "rather not say",
                             "thats private", "I'll pass on this one"],
                "neutral": ["nothing comes to mind", "idk", "cant think of anything",
                            "pass"],
                "negative": ["no", "not sharing", "nope", "too personal"],
            },
            "creative_narrative": {
                "positive": ["I have a good one actually", "oh yeah definitely",
                             "lol yes"],
                "neutral": ["nothing crazy", "not really", "cant think of one",
                            "idk"],
                "negative": ["no", "nothing", "nope", "boring life lol"],
            },
            "personal_story": {
                "positive": ["yeah I remember something", "I have one",
                             "something like that happened to me"],
                "neutral": ["not really", "cant think of anything specific",
                            "maybe"],
                "negative": ["no", "dont have one", "nah", "nothing relevant"],
            },
            "hypothetical": {
                "positive": ["yeah I would", "definitely", "for sure"],
                "neutral": ["maybe", "depends", "not sure", "idk"],
                "negative": ["no way", "nope", "absolutely not", "hard pass"],
            },
        }

        # Map sentiment
        if sentiment in ("very_positive", "positive"):
            _sk = "positive"
        elif sentiment in ("very_negative", "negative"):
            _sk = "negative"
        else:
            _sk = "neutral"

        bank = _pools.get(intent, {})
        pool = bank.get(_sk, ["idk", "not sure", "no opinion"])
        return rng.choice(pool)

    # ── Archetype selection ───────────────────────────────────

    def _select_narrative_archetype(
        self, formality: float, engagement: float, rng: random.Random,
    ) -> str:
        """Select archetype with narrative-biased weights."""
        f_tier = 'casual' if formality < 0.35 else ('formal' if formality > 0.7 else 'moderate')
        e_tier = 'low' if engagement < 0.4 else ('high' if engagement > 0.7 else 'mid')
        weights = _NARRATIVE_ARCHETYPE_WEIGHTS.get(
            (f_tier, e_tier),
            _NARRATIVE_ARCHETYPE_WEIGHTS[('moderate', 'mid')],
        )
        archetypes = list(weights.keys())
        probs = list(weights.values())
        total = sum(probs)
        r = rng.random() * total
        cum = 0
        for arch, w in zip(archetypes, probs):
            cum += w
            if r <= cum:
                return arch
        return archetypes[-1]

    # ── Topic extraction ──────────────────────────────────────

    def _extract_narrative_topic(
        self, question_text: str, question_context: str, intent: str,
    ) -> str:
        """Extract a natural-language topic suitable for narrative templates."""
        _stop = {
            'this', 'that', 'about', 'what', 'your', 'please', 'describe',
            'explain', 'question', 'context', 'study', 'topic', 'condition',
            'think', 'feel', 'have', 'some', 'with', 'from', 'very', 'really',
            'tell', 'share', 'write', 'craziest', 'wildest', 'favorite',
            'most', 'biggest', 'worst', 'best', 'funniest', 'scariest',
        }

        _source = question_context or question_text or ""
        if not _source:
            return "this"

        # Try to extract the core noun phrase
        # "What is your craziest conspiracy theory?" → "conspiracy theory"
        # "Share a secret that only your family knows" → "a secret"
        _source_lower = _source.lower()

        # Pattern: "your X" or "a X" where X is the topic
        _patterns = [
            r'(?:your|a|an)\s+(?:craziest|wildest|favorite|biggest|most\s+\w+|worst|best|funniest|scariest|deepest|darkest)?\s*(.{3,40}?)(?:\?|$|\.|\bthat\b|\bwhen\b|\bwhere\b)',
            r'(?:believe\s+in|think\s+about|feel\s+about|opinion\s+on|views?\s+on|thoughts?\s+on)\s+(.{3,40}?)(?:\?|$|\.)',
            r'(?:describe|share|tell\s+us\s+about|write\s+about)\s+(?:your\s+)?(?:\w+\s+)?(.{3,40}?)(?:\?|$|\.)',
        ]

        for pattern in _patterns:
            m = re.search(pattern, _source_lower)
            if m:
                topic = m.group(1).strip()
                # Remove trailing stop words
                words = topic.split()
                while words and words[-1] in _stop:
                    words.pop()
                while words and words[0] in _stop:
                    words.pop(0)
                if words and len(' '.join(words)) >= 3:
                    return ' '.join(words)

        # Fallback: extract the longest non-stop word sequence
        words = re.findall(r'\b[a-zA-Z]{3,}\b', _source_lower)
        content_words = [w for w in words if w not in _stop]
        if content_words:
            # Take up to 3 content words
            return ' '.join(content_words[:3])

        # Intent-specific default topics
        _defaults = {
            "creative_belief": "conspiracy theories",
            "personal_disclosure": "something personal",
            "creative_narrative": "something memorable",
            "personal_story": "a personal experience",
            "hypothetical": "this scenario",
        }
        return _defaults.get(intent, "this")

    # ── Variation phrase system ────────────────────────────────

    def _apply_variation_phrase(
        self, response: str, participant_seed: int, formality: float,
    ) -> str:
        """Optionally prepend/append variation phrases for dedup diversity.

        Applied to ~25% of responses to add natural variety without
        making every response feel formulaic.
        """
        if not response or not response.strip():
            return response

        # Deterministic hash (not Python's session-random hash())
        _resp_hash = sum(ord(c) * (i + 1) * 17 for i, c in enumerate(response[:20]))
        rng = random.Random(participant_seed + _resp_hash % 10000)

        # Only apply to ~25% of responses
        if rng.random() > 0.25:
            return response

        # Very formal personas don't use casual variation phrases
        if formality > 0.8:
            return response

        # Pick a variation type
        _type = rng.choice(["time", "personal", "certainty", "ending"])

        if _type == "time" and rng.random() < 0.5:
            phrase = rng.choice(_VARIATION_PHRASES["time_phrases"])
            # Don't prepend if response already starts with a similar phrase
            if not response[:15].lower().startswith(phrase[:8].lower()):
                response = phrase + " " + response[0].lower() + response[1:]

        elif _type == "personal" and rng.random() < 0.4:
            phrase = rng.choice(_VARIATION_PHRASES["personal_phrases"])
            if not response[:15].lower().startswith(phrase[:8].lower()):
                response = phrase + " " + response[0].lower() + response[1:]

        elif _type == "certainty" and rng.random() < 0.4:
            phrase = rng.choice(_VARIATION_PHRASES["certainty_phrases"])
            # Prepend certainty phrase, replacing existing "I think" style openers
            _lower = response.lower()
            if not any(_lower.startswith(p.lower()) for p in _VARIATION_PHRASES["certainty_phrases"]):
                response = phrase + " " + response[0].lower() + response[1:]

        elif _type == "ending":
            phrase = rng.choice(_VARIATION_PHRASES["ending_phrases"])
            if phrase:  # Some entries are empty on purpose
                # Remove existing ending punctuation
                response = response.rstrip('.!? ')
                response += phrase

        return response

    # ── Polish and post-processing ────────────────────────────

    def _apply_narrative_polish(
        self, response: str, formality: float, engagement: float,
        rng: random.Random,
    ) -> str:
        """Apply natural language polish to narrative responses."""
        if not response:
            return response

        # Casual personas: add contractions and lowercase
        if formality < 0.4:
            _contractions = [
                ("I am", "I'm"), ("I have", "I've"), ("I would", "I'd"),
                ("I will", "I'll"), ("do not", "don't"), ("does not", "doesn't"),
                ("did not", "didn't"), ("cannot", "can't"), ("could not", "couldn't"),
                ("would not", "wouldn't"), ("should not", "shouldn't"),
                ("is not", "isn't"), ("are not", "aren't"), ("was not", "wasn't"),
                ("were not", "weren't"), ("has not", "hasn't"),
                ("it is", "it's"), ("that is", "that's"),
            ]
            for full, short in _contractions:
                if rng.random() < 0.7:
                    response = response.replace(full, short)
                    response = response.replace(full.capitalize(), short.capitalize())

        # Low engagement: occasionally drop capitalization
        if engagement < 0.35 and rng.random() < 0.4:
            response = response[0].lower() + response[1:] if response else response

        # High engagement: occasionally add emphasis
        if engagement > 0.7 and rng.random() < 0.2:
            _emphasis = [" honestly", " genuinely", " seriously", " truly"]
            # Find a comma or period to insert emphasis
            _comma_pos = response.find(',')
            if _comma_pos > 10 and _comma_pos < len(response) - 10:
                response = response[:_comma_pos] + rng.choice(_emphasis) + response[_comma_pos:]

        return response

    # ── Utility ───────────────────────────────────────────────

    def _last_resort_response(self, sentiment: str, seed: int) -> str:
        """Ultra-last-resort when everything else fails."""
        rng = random.Random(seed)
        _pool = [
            "I gave my honest answer based on how I feel about this.",
            "I thought about it and answered based on my personal experience.",
            "My response reflects my genuine views on the topic.",
            "I tried to be honest about how I see things.",
            "I answered based on what I've personally observed and experienced.",
        ]
        return rng.choice(_pool)

    # ── Passthrough properties ────────────────────────────────

    @property
    def study_context(self) -> Dict[str, Any]:
        """Access base generator's study context."""
        if self._base_generator is not None:
            return self._base_generator.study_context
        return {}

    def reset_uniqueness(self) -> None:
        """Reset uniqueness tracking for a new generation run."""
        self._used_sentences.clear()
        self._generation_count = 0
        if self._base_generator is not None and hasattr(self._base_generator, '_used_sentences'):
            self._base_generator._used_sentences.clear()
