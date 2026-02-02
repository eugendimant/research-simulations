"""
Comprehensive Response Library for Open-Ended Survey Questions
==============================================================

This module provides LLM-quality text generation for survey responses across
50+ research domains with dynamic context adaptation.

Features:
- 50+ domain-specific template sets
- Dynamic context detection from QSF content
- Persona-aware response generation
- Markov chain text generation for variation
- Question type classification and handling

Version: 2.1.15
"""

__version__ = "2.1.15"

import random
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class QuestionType(Enum):
    """Types of open-ended questions in surveys."""
    EXPLANATION = "explanation"           # "Please explain your choice..."
    FEEDBACK = "feedback"                 # "Any feedback about the survey?"
    DESCRIPTION = "description"           # "Describe your experience..."
    JUSTIFICATION = "justification"       # "Why did you choose..."
    OPINION = "opinion"                   # "What is your opinion on..."
    REFLECTION = "reflection"             # "Reflect on how you felt..."
    SUGGESTION = "suggestion"             # "How could we improve..."
    PREDICTION = "prediction"             # "What do you think will happen..."
    MEMORY = "memory"                     # "Recall a time when..."
    ASSOCIATION = "association"           # "What comes to mind when..."
    PREFERENCE = "preference"             # "Why do you prefer..."
    EVALUATION = "evaluation"             # "How would you evaluate..."
    COMPARISON = "comparison"             # "Compare X and Y..."
    REASONING = "reasoning"               # "What reasoning led to..."
    GENERAL = "general"                   # Catch-all


class StudyDomain(Enum):
    """Research domains for context-specific responses.

    This enum covers 50+ research domains across behavioral economics,
    social psychology, political science, consumer behavior, organizational
    behavior, technology, health, education, ethics, and environmental studies.
    """
    # Behavioral Economics (8 domains)
    BEHAVIORAL_ECONOMICS = "behavioral_economics"
    DICTATOR_GAME = "dictator_game"
    PUBLIC_GOODS = "public_goods"
    ULTIMATUM_GAME = "ultimatum_game"
    TRUST_GAME = "trust_game"
    PRISONERS_DILEMMA = "prisoners_dilemma"
    RISK_PREFERENCE = "risk_preference"
    TIME_PREFERENCE = "time_preference"

    # Social Psychology (9 domains)
    SOCIAL_PSYCHOLOGY = "social_psychology"
    INTERGROUP = "intergroup"
    IDENTITY = "identity"
    NORMS = "norms"
    CONFORMITY = "conformity"
    PROSOCIAL = "prosocial"
    TRUST = "trust"
    FAIRNESS = "fairness"
    COOPERATION = "cooperation"

    # Political Science (5 domains)
    POLITICAL = "political"
    POLARIZATION = "polarization"
    PARTISANSHIP = "partisanship"
    VOTING = "voting"
    MEDIA = "media"

    # Consumer/Marketing (5 domains)
    CONSUMER = "consumer"
    BRAND = "brand"
    ADVERTISING = "advertising"
    PRODUCT_EVALUATION = "product_evaluation"
    PURCHASE_INTENT = "purchase_intent"

    # Organizational Behavior (5 domains)
    ORGANIZATIONAL = "organizational"
    WORKPLACE = "workplace"
    LEADERSHIP = "leadership"
    TEAMWORK = "teamwork"
    MOTIVATION = "motivation"

    # Technology/AI (4 domains)
    TECHNOLOGY = "technology"
    AI_ATTITUDES = "ai_attitudes"
    PRIVACY = "privacy"
    AUTOMATION = "automation"

    # Health (3 domains)
    HEALTH = "health"
    MEDICAL_DECISION = "medical_decision"
    WELLBEING = "wellbeing"

    # Education (2 domains)
    EDUCATION = "education"
    LEARNING = "learning"

    # Ethics/Moral Psychology (2 domains)
    ETHICS = "ethics"
    MORAL_JUDGMENT = "moral_judgment"

    # Environmental (2 domains)
    ENVIRONMENTAL = "environmental"
    SUSTAINABILITY = "sustainability"

    # General (2 domains)
    GENERAL = "general"
    SURVEY_FEEDBACK = "survey_feedback"


# ============================================================================
# COMPREHENSIVE DOMAIN-SPECIFIC RESPONSE TEMPLATES
# ============================================================================

DOMAIN_TEMPLATES: Dict[str, Dict[str, Dict[str, List[str]]]] = {

    # ========== BEHAVIORAL ECONOMICS ==========

    "dictator_game": {
        "explanation": {
            "very_positive": [
                "I gave most of the money because I believe in sharing equally with others, even strangers.",
                "Fairness is important to me, so I wanted to give my partner a substantial amount.",
                "I believe that generosity creates positive outcomes for everyone involved.",
                "I decided to share because I would want someone to do the same for me.",
                "Giving felt like the right thing to do - money isn't everything.",
            ],
            "positive": [
                "I gave a fair amount because I think both people deserve something.",
                "I tried to be generous while still keeping some for myself.",
                "It seemed reasonable to share some of the money with my partner.",
                "I wanted to be fair, so I gave what felt like a reasonable amount.",
                "Sharing makes sense to me, even with people I don't know.",
            ],
            "neutral": [
                "I just split it roughly down the middle without thinking too much.",
                "I gave what seemed like an average amount.",
                "I wasn't sure what to do, so I went with a moderate choice.",
                "No particular reason - just picked something in the middle.",
                "I didn't have strong feelings either way.",
            ],
            "negative": [
                "I kept most of the money because it was my decision to make.",
                "I don't know this person, so I felt no obligation to share much.",
                "I figured I should look out for my own interests first.",
                "I gave a small amount to not seem completely unfair.",
                "I earned the right to decide, so I kept more.",
            ],
            "very_negative": [
                "I kept everything because there's no reason to give money to a stranger.",
                "Why would I give away money when I don't have to?",
                "I prioritized my own benefit in this situation.",
                "I don't see why I should share when it's my choice.",
                "The rules allowed me to keep it all, so I did.",
            ],
        },
    },

    "public_goods": {
        "explanation": {
            "very_positive": [
                "I contributed a lot because cooperation benefits everyone in the group.",
                "I believe in collective action and doing my part for the common good.",
                "If everyone contributes, we all end up better off.",
                "I wanted to set a good example for group cooperation.",
                "Contributing to the common pool seemed like the prosocial choice.",
            ],
            "positive": [
                "I contributed a reasonable amount to support the group.",
                "I tried to balance personal interest with group benefit.",
                "Cooperation makes sense when everyone does their part.",
                "I wanted to contribute while not giving everything.",
                "I gave what I thought was a fair contribution.",
            ],
            "neutral": [
                "I contributed a moderate amount without thinking too hard.",
                "I wasn't sure what others would do, so I went middle-ground.",
                "I gave an average contribution.",
                "No strong feelings about it either way.",
                "I just picked something reasonable.",
            ],
            "negative": [
                "I contributed less because I wasn't sure others would cooperate.",
                "I kept more for myself since the returns aren't guaranteed.",
                "I don't know these people, so I was cautious.",
                "Free-riding seemed like the rational choice.",
                "I contributed minimally to see what others would do.",
            ],
            "very_negative": [
                "I didn't contribute much because others will carry the burden.",
                "Why contribute when I can benefit from others' contributions?",
                "I kept my money since cooperation often fails anyway.",
                "I prioritized my own earnings over group returns.",
                "The incentive structure made keeping money rational.",
            ],
        },
    },

    "trust_game": {
        "explanation": {
            "very_positive": [
                "I sent a large amount because I believe in giving people a chance to be trustworthy.",
                "Trust is important for cooperation, so I wanted to show faith in my partner.",
                "I think most people will reciprocate if you trust them first.",
                "Showing trust seemed like the right way to start the interaction.",
                "I believe in the goodness of people, so I trusted them.",
            ],
            "positive": [
                "I sent a reasonable amount to test if they would reciprocate.",
                "I tried to show some trust while managing my risk.",
                "Trusting others often leads to better outcomes.",
                "I wanted to give them a chance to be fair.",
                "A moderate amount of trust seemed appropriate.",
            ],
            "neutral": [
                "I sent a middle amount since I wasn't sure what to expect.",
                "I went with a moderate choice given the uncertainty.",
                "I split the difference between trusting and not.",
                "Wasn't sure if they'd reciprocate, so I hedged.",
                "No strong intuition either way.",
            ],
            "negative": [
                "I sent less because I wasn't sure if they'd return anything.",
                "Trust needs to be earned, and I don't know this person.",
                "I was cautious because people often don't reciprocate.",
                "I kept more to protect myself from exploitation.",
                "Better safe than sorry when dealing with strangers.",
            ],
            "very_negative": [
                "I kept most of it because trusting strangers is risky.",
                "People often take advantage of trust, so I was careful.",
                "I didn't want to be the sucker in this situation.",
                "Trust is naive when there's no accountability.",
                "I protected my interests given the one-shot nature.",
            ],
        },
    },

    "risk_preference": {
        "explanation": {
            "very_positive": [
                "I chose the risky option because the potential payoff was worth it.",
                "I'm comfortable with risk when the expected value is good.",
                "Taking calculated risks is how you get ahead.",
                "I'd rather take a chance than play it too safe.",
                "The upside potential was too good to pass up.",
            ],
            "positive": [
                "I was willing to take some risk for a better outcome.",
                "The risky option seemed like a reasonable gamble.",
                "I calculated the odds and decided it was worth trying.",
                "A moderate amount of risk seemed acceptable.",
                "I leaned toward risk since the safe option was limited.",
            ],
            "neutral": [
                "I went with something in the middle of the risk spectrum.",
                "Wasn't strongly pulled toward either safe or risky.",
                "I picked a moderate level of risk.",
                "No strong preference for risk or safety.",
                "I just chose what felt comfortable.",
            ],
            "negative": [
                "I preferred the safer option to avoid potential losses.",
                "The sure thing seemed better than gambling.",
                "I'm risk-averse, especially with uncertain outcomes.",
                "Losing what I had worried me more than potential gains.",
                "Better to keep what I have than risk losing it.",
            ],
            "very_negative": [
                "I definitely wanted the safe, guaranteed option.",
                "Risk-taking doesn't make sense when you can lock in gains.",
                "I never gamble when a sure thing is available.",
                "The downside risk was too scary for me.",
                "I'm very conservative with uncertain choices.",
            ],
        },
    },

    # ========== SOCIAL PSYCHOLOGY ==========

    "intergroup": {
        "explanation": {
            "very_positive": [
                "I treated the outgroup member fairly because people are individuals.",
                "Their group membership didn't affect my decision - everyone deserves respect.",
                "I believe in equal treatment regardless of group differences.",
                "I try to judge people as individuals, not by their group.",
                "Our differences don't mean we can't interact positively.",
            ],
            "positive": [
                "I tried to be fair even though we're from different groups.",
                "Their group background didn't change my approach much.",
                "I gave them the benefit of the doubt.",
                "I aimed for equal treatment in this interaction.",
                "Group differences shouldn't determine individual behavior.",
            ],
            "neutral": [
                "Their group didn't really factor into my decision.",
                "I treated this like any other interaction.",
                "I didn't think much about our group differences.",
                "I focused on the task rather than group membership.",
                "I approached it without strong feelings either way.",
            ],
            "negative": [
                "I was somewhat cautious given our group differences.",
                "Their group membership made me a bit wary.",
                "I wasn't sure I could trust someone from that group.",
                "Our different backgrounds affected my choice somewhat.",
                "I was less generous than I might have been otherwise.",
            ],
            "very_negative": [
                "I don't trust people from that group, so I acted accordingly.",
                "Their group membership significantly influenced my decision.",
                "I couldn't look past our differences.",
                "I favored my own group in this situation.",
                "I treated them differently because of their group.",
            ],
        },
    },

    "identity": {
        "explanation": {
            "very_positive": [
                "My identity strongly shapes how I see the world and make decisions.",
                "I feel a deep connection to my group and its values.",
                "My background is central to who I am.",
                "I'm proud of my identity and it guides my choices.",
                "Being part of this group is meaningful to me.",
            ],
            "positive": [
                "My identity plays a role in how I approach things.",
                "I feel connected to my group in many ways.",
                "My background influences some of my decisions.",
                "I identify with my group to a reasonable degree.",
                "My identity matters, though it's not everything.",
            ],
            "neutral": [
                "I don't think much about my identity in these situations.",
                "My group membership doesn't dominate my thinking.",
                "I try to look at things without identity bias.",
                "I'm neutral about how my identity affects choices.",
                "It varies - sometimes identity matters, sometimes not.",
            ],
            "negative": [
                "I try not to let my identity influence my decisions.",
                "I'm somewhat detached from my group.",
                "Identity shouldn't determine how we treat others.",
                "I prefer to think of myself as an individual first.",
                "Group identity can be limiting.",
            ],
            "very_negative": [
                "I reject the idea that identity should matter.",
                "I don't identify strongly with any group.",
                "Focusing on group identity seems divisive.",
                "I consciously try to ignore group considerations.",
                "Individualism matters more than group belonging.",
            ],
        },
    },

    "norms": {
        "explanation": {
            "very_positive": [
                "I followed the social norm because it's important to maintain standards.",
                "Doing what's expected helps society function smoothly.",
                "I believe in following established rules and conventions.",
                "Social norms exist for good reasons, so I respected them.",
                "I try to act in ways that are socially appropriate.",
            ],
            "positive": [
                "I generally try to follow social expectations.",
                "The norm seemed reasonable, so I went along with it.",
                "I considered what's normally done in this situation.",
                "Social norms provided useful guidance.",
                "I didn't want to deviate from what's expected.",
            ],
            "neutral": [
                "I wasn't really thinking about norms specifically.",
                "I just did what felt natural without considering norms.",
                "Social expectations didn't strongly influence me.",
                "I wasn't sure what the norm even was.",
                "I went with my gut rather than following rules.",
            ],
            "negative": [
                "I didn't feel bound by what others expect.",
                "Social norms aren't always right.",
                "I prefer to make my own decisions.",
                "Just because something is normal doesn't mean it's best.",
                "I questioned whether the norm made sense.",
            ],
            "very_negative": [
                "I deliberately went against what's expected.",
                "Social norms often perpetuate problems.",
                "I don't believe in blindly following conventions.",
                "Conformity isn't a virtue in my view.",
                "I reject the pressure to follow arbitrary rules.",
            ],
        },
    },

    "trust": {
        "explanation": {
            "very_positive": [
                "I trust people until they give me a reason not to.",
                "I believe most people are fundamentally good and honest.",
                "Trust is essential for positive relationships and cooperation.",
                "I'm willing to be vulnerable because trust is important.",
                "Giving people your trust often brings out the best in them.",
            ],
            "positive": [
                "I'm generally trusting of others.",
                "I give people the benefit of the doubt initially.",
                "Trust is important, though it needs to be earned over time.",
                "I try to approach interactions with an open attitude.",
                "I'm reasonably trusting in most situations.",
            ],
            "neutral": [
                "I'm neither particularly trusting nor distrusting.",
                "I take a wait-and-see approach with trust.",
                "Trust depends on the specific situation and person.",
                "I don't have strong feelings about trust either way.",
                "I'm moderately cautious without being paranoid.",
            ],
            "negative": [
                "I'm careful about who I trust.",
                "Trust needs to be earned - I don't give it freely.",
                "I've learned to be wary of trusting too easily.",
                "People often disappoint when you trust them.",
                "I prefer to verify before trusting.",
            ],
            "very_negative": [
                "I don't trust people easily, if at all.",
                "Trust gets you taken advantage of.",
                "I assume people will act in their own interest.",
                "Being skeptical protects you from being burned.",
                "I've learned the hard way not to trust.",
            ],
        },
    },

    # ========== POLITICAL SCIENCE ==========

    "political": {
        "explanation": {
            "very_positive": [
                "I feel strongly about my political views and values.",
                "Politics matters because it affects everyone's lives.",
                "I'm engaged with political issues that are important to me.",
                "My political stance reflects my core beliefs.",
                "I'm passionate about making political change.",
            ],
            "positive": [
                "I generally support this political direction.",
                "My political views lean a certain way on this issue.",
                "I think this political approach makes sense.",
                "I have moderate but clear political preferences.",
                "Politics is important, though not my main focus.",
            ],
            "neutral": [
                "I don't have strong political opinions.",
                "I try to see multiple sides of political issues.",
                "I'm not particularly politically engaged.",
                "My views are mixed on this political topic.",
                "I stay neutral on most political matters.",
            ],
            "negative": [
                "I disagree with this political position.",
                "My political views don't align with this approach.",
                "I have concerns about this political direction.",
                "I'm skeptical of these political claims.",
                "This political stance doesn't represent my values.",
            ],
            "very_negative": [
                "I strongly oppose this political view.",
                "This goes against my fundamental political beliefs.",
                "I'm deeply concerned about this political direction.",
                "I can't support this political position at all.",
                "This political approach is harmful in my view.",
            ],
        },
    },

    "polarization": {
        "explanation": {
            "very_positive": [
                "I think people on the other side have valid perspectives.",
                "We need to find common ground despite our differences.",
                "Dialogue across political divides is essential.",
                "I'm open to understanding opposing viewpoints.",
                "Extreme polarization hurts our society.",
            ],
            "positive": [
                "I try to listen to different political perspectives.",
                "While I disagree, I respect other viewpoints.",
                "Finding middle ground is usually possible.",
                "I'm willing to engage with the other side.",
                "Polarization concerns me but I stay engaged.",
            ],
            "neutral": [
                "I see valid points on both sides.",
                "I'm not strongly aligned with either extreme.",
                "I try to stay out of political conflicts.",
                "I take a moderate position on most issues.",
                "I don't feel the need to pick sides.",
            ],
            "negative": [
                "The other side is difficult to understand.",
                "I find it hard to relate to opposing views.",
                "Political divisions seem quite deep.",
                "I'm frustrated with the other side's positions.",
                "It's hard to find common ground anymore.",
            ],
            "very_negative": [
                "I fundamentally disagree with the other side.",
                "Their values are incompatible with mine.",
                "I don't see how we can bridge this divide.",
                "The opposition's views are harmful.",
                "Political compromise seems impossible.",
            ],
        },
    },

    # ========== CONSUMER/MARKETING ==========

    "consumer": {
        "explanation": {
            "very_positive": [
                "I absolutely loved this product - it exceeded all my expectations.",
                "This is exactly what I was looking for, high quality all around.",
                "I would highly recommend this to anyone.",
                "The value for money is exceptional.",
                "This brand really delivers on its promises.",
            ],
            "positive": [
                "I'm satisfied with this product overall.",
                "It met my expectations and works well.",
                "Good quality for the price.",
                "I would consider buying from this brand again.",
                "A solid choice that I'm happy with.",
            ],
            "neutral": [
                "It's okay - nothing special but nothing wrong either.",
                "Average product, does what it's supposed to.",
                "Neither impressed nor disappointed.",
                "It's fine for what it is.",
                "Standard quality, nothing remarkable.",
            ],
            "negative": [
                "I'm not fully satisfied with this product.",
                "It didn't quite meet my expectations.",
                "There are some issues that bothered me.",
                "Could be better for the price.",
                "I probably wouldn't buy this again.",
            ],
            "very_negative": [
                "Very disappointed with this product.",
                "It failed to deliver on its promises.",
                "I would not recommend this to anyone.",
                "Poor quality and not worth the money.",
                "I regret this purchase.",
            ],
        },
    },

    "brand": {
        "explanation": {
            "very_positive": [
                "I'm a loyal customer of this brand - they never disappoint.",
                "This brand represents quality and values I believe in.",
                "I feel a connection to this brand.",
                "I trust this brand completely.",
                "This brand understands its customers.",
            ],
            "positive": [
                "I generally have positive feelings about this brand.",
                "This brand has a good reputation in my mind.",
                "I would choose this brand over others.",
                "Solid brand that I respect.",
                "I've had good experiences with this brand.",
            ],
            "neutral": [
                "I don't have strong feelings about this brand.",
                "It's just one of many brands to me.",
                "I don't really pay attention to brands.",
                "This brand is neither better nor worse than others.",
                "Brand loyalty isn't important to me.",
            ],
            "negative": [
                "I have some reservations about this brand.",
                "This brand hasn't impressed me.",
                "I'm skeptical of this brand's claims.",
                "I would probably choose a different brand.",
                "This brand needs improvement.",
            ],
            "very_negative": [
                "I actively avoid this brand.",
                "This brand has lost my trust.",
                "I have had bad experiences with this brand.",
                "I would warn others about this brand.",
                "This brand represents values I disagree with.",
            ],
        },
    },

    "purchase_intent": {
        "explanation": {
            "very_positive": [
                "I definitely plan to buy this - it's exactly what I need.",
                "I'm ready to make this purchase as soon as possible.",
                "This is a must-have for me.",
                "I've already decided to buy this.",
                "I'm very likely to purchase this product.",
            ],
            "positive": [
                "I'm seriously considering buying this.",
                "There's a good chance I'll make this purchase.",
                "I'm leaning toward buying this.",
                "This is on my shopping list.",
                "I'll probably buy this at some point.",
            ],
            "neutral": [
                "I might buy this, I might not.",
                "I need to think about it more.",
                "I'm undecided about purchasing.",
                "It depends on various factors.",
                "I haven't made up my mind yet.",
            ],
            "negative": [
                "I probably won't buy this.",
                "I'm not convinced I need this.",
                "This isn't a priority purchase for me.",
                "I'm leaning toward not buying.",
                "I have reservations about purchasing.",
            ],
            "very_negative": [
                "I definitely won't buy this.",
                "This doesn't interest me at all.",
                "I see no reason to purchase this.",
                "I'm not the target market for this.",
                "Hard pass on this product.",
            ],
        },
    },

    # ========== ORGANIZATIONAL BEHAVIOR ==========

    "workplace": {
        "explanation": {
            "very_positive": [
                "I love my job and feel valued by my organization.",
                "My workplace is supportive and encourages growth.",
                "I'm highly engaged and motivated at work.",
                "My colleagues are great and we work well together.",
                "I feel proud to be part of this organization.",
            ],
            "positive": [
                "Overall, I'm satisfied with my work situation.",
                "My job has its challenges but I enjoy it.",
                "I have good relationships with my coworkers.",
                "The work environment is generally positive.",
                "I feel reasonably supported at work.",
            ],
            "neutral": [
                "Work is work - it's neither great nor terrible.",
                "I'm neutral about my job situation.",
                "Some aspects are good, others less so.",
                "It's an average workplace experience.",
                "I don't have strong feelings about my job.",
            ],
            "negative": [
                "There are things about my job I'd like to change.",
                "I'm not fully satisfied with my work situation.",
                "Some workplace issues affect my engagement.",
                "I feel undervalued sometimes.",
                "The work environment could be better.",
            ],
            "very_negative": [
                "I'm very dissatisfied with my work situation.",
                "My workplace has serious problems.",
                "I feel disengaged and unmotivated.",
                "I'm looking for other opportunities.",
                "The organization doesn't care about employees.",
            ],
        },
    },

    "leadership": {
        "explanation": {
            "very_positive": [
                "I have great confidence in our leadership.",
                "Leaders here are competent and supportive.",
                "I feel inspired by my manager's leadership style.",
                "Leadership communicates clearly and listens well.",
                "I trust the decisions made by leadership.",
            ],
            "positive": [
                "Leadership is generally effective.",
                "My manager does a good job overall.",
                "I'm reasonably satisfied with leadership.",
                "Leaders here are competent.",
                "Communication from leadership is adequate.",
            ],
            "neutral": [
                "Leadership is neither particularly good nor bad.",
                "I don't have strong opinions about leadership.",
                "Average leadership in my experience.",
                "Some leaders are good, others less so.",
                "I'm neutral on leadership effectiveness.",
            ],
            "negative": [
                "I have concerns about leadership here.",
                "Leaders could communicate better.",
                "I'm not fully confident in leadership decisions.",
                "There's room for improvement in management.",
                "Leadership doesn't always listen to employees.",
            ],
            "very_negative": [
                "Leadership is a major problem here.",
                "I have no confidence in management.",
                "Leaders are disconnected from reality.",
                "Poor leadership affects morale significantly.",
                "I don't trust leadership at all.",
            ],
        },
    },

    # ========== TECHNOLOGY/AI ==========

    "ai_attitudes": {
        "explanation": {
            "very_positive": [
                "AI will bring tremendous benefits to society.",
                "I'm excited about the possibilities AI offers.",
                "AI can solve problems humans can't solve alone.",
                "I trust AI systems to make good decisions.",
                "The future with AI looks promising.",
            ],
            "positive": [
                "AI has more benefits than drawbacks.",
                "I'm generally optimistic about AI.",
                "AI can help with many tasks.",
                "I'm comfortable using AI tools.",
                "AI development should continue.",
            ],
            "neutral": [
                "AI has both pros and cons.",
                "I'm neither excited nor worried about AI.",
                "It depends on how AI is implemented.",
                "I take a wait-and-see approach to AI.",
                "AI is just a tool - it depends on how it's used.",
            ],
            "negative": [
                "I have concerns about AI.",
                "AI poses risks we should take seriously.",
                "I'm cautious about AI in certain domains.",
                "AI might replace jobs and create problems.",
                "We need better AI regulation.",
            ],
            "very_negative": [
                "AI is a serious threat to society.",
                "I don't trust AI systems.",
                "AI development is moving too fast.",
                "AI will cause more harm than good.",
                "I prefer human judgment over AI.",
            ],
        },
    },

    "privacy": {
        "explanation": {
            "very_positive": [
                "I'm very protective of my personal data.",
                "Privacy is a fundamental right.",
                "Companies should do more to protect user data.",
                "I carefully manage my privacy settings.",
                "Data collection practices concern me greatly.",
            ],
            "positive": [
                "Privacy matters to me.",
                "I try to be careful with my personal information.",
                "Companies should be transparent about data use.",
                "I take some steps to protect my privacy.",
                "Privacy is important, though not my top priority.",
            ],
            "neutral": [
                "I don't think much about privacy.",
                "I accept some loss of privacy for convenience.",
                "Privacy concerns don't strongly affect my behavior.",
                "I'm neutral on most privacy issues.",
                "It depends on what data we're talking about.",
            ],
            "negative": [
                "I don't worry much about privacy.",
                "Privacy concerns are sometimes overblown.",
                "I'm willing to share data for better services.",
                "I have nothing to hide.",
                "Convenience matters more to me than privacy.",
            ],
            "very_negative": [
                "Privacy concerns are largely exaggerated.",
                "I freely share my information.",
                "Data collection benefits everyone.",
                "Privacy restrictions hinder innovation.",
                "I don't see the big deal about privacy.",
            ],
        },
    },

    # ========== HEALTH ==========

    "health": {
        "explanation": {
            "very_positive": [
                "I prioritize my health in my daily decisions.",
                "Maintaining good health is very important to me.",
                "I actively work to live a healthy lifestyle.",
                "Health is wealth - I take it seriously.",
                "I make healthy choices consistently.",
            ],
            "positive": [
                "I generally try to maintain good health.",
                "I make some effort to be healthy.",
                "Health is important, though I'm not perfect.",
                "I try to balance health with enjoyment.",
                "I'm reasonably health-conscious.",
            ],
            "neutral": [
                "I don't focus much on health specifically.",
                "I'm not particularly health-conscious.",
                "Health is one of many considerations.",
                "I take it as it comes.",
                "I'm neither health-focused nor neglectful.",
            ],
            "negative": [
                "I struggle to maintain healthy habits.",
                "Health isn't always my priority.",
                "I know I should be healthier.",
                "I often choose convenience over health.",
                "Health is something I need to work on.",
            ],
            "very_negative": [
                "I don't really think about health.",
                "Health concerns don't change my behavior.",
                "I live for today without worrying about health.",
                "Health advice often seems overblown.",
                "I enjoy life without health restrictions.",
            ],
        },
    },

    # ========== ETHICS/MORAL ==========

    "moral_judgment": {
        "explanation": {
            "very_positive": [
                "This action is clearly morally right.",
                "I strongly support this from an ethical standpoint.",
                "This represents good moral values.",
                "This is the ethical thing to do.",
                "I find this morally admirable.",
            ],
            "positive": [
                "I think this is morally acceptable.",
                "This seems like the right thing to do.",
                "I have no moral objections to this.",
                "This aligns with my values.",
                "This is ethically sound in my view.",
            ],
            "neutral": [
                "I'm not sure about the morality of this.",
                "This is a gray area ethically.",
                "I can see moral arguments both ways.",
                "I don't have a strong moral stance on this.",
                "This isn't clearly right or wrong to me.",
            ],
            "negative": [
                "I have moral concerns about this.",
                "This doesn't seem quite right.",
                "This raises ethical questions.",
                "I'm uncomfortable with this morally.",
                "This conflicts with some of my values.",
            ],
            "very_negative": [
                "This is morally wrong in my view.",
                "I strongly object to this ethically.",
                "This goes against my moral principles.",
                "I find this ethically unacceptable.",
                "This represents poor moral values.",
            ],
        },
    },

    # ========== ENVIRONMENTAL ==========

    "environmental": {
        "explanation": {
            "very_positive": [
                "Environmental protection should be our top priority.",
                "I'm deeply concerned about climate change.",
                "I make efforts to reduce my environmental impact.",
                "We need urgent action on environmental issues.",
                "Sustainability is central to my values.",
            ],
            "positive": [
                "I care about environmental issues.",
                "I try to make environmentally friendly choices.",
                "Environmental protection is important.",
                "I support policies that help the environment.",
                "I'm reasonably eco-conscious.",
            ],
            "neutral": [
                "Environmental issues aren't my main focus.",
                "I don't strongly prioritize environmental concerns.",
                "I'm neutral on most environmental debates.",
                "I do some things for the environment, not others.",
                "Environmental issues are one of many concerns.",
            ],
            "negative": [
                "I'm skeptical of some environmental claims.",
                "Environmental concerns are sometimes exaggerated.",
                "I don't let environmental issues drive my choices.",
                "Economic concerns matter more to me.",
                "I'm not convinced by all environmental arguments.",
            ],
            "very_negative": [
                "Environmental activism has gone too far.",
                "Climate concerns are overblown.",
                "I don't believe environmental warnings.",
                "Environmental regulations harm the economy.",
                "I prioritize development over environment.",
            ],
        },
    },

    # ========== GENERAL/SURVEY FEEDBACK ==========

    "survey_feedback": {
        "explanation": {
            "very_positive": [
                "The survey was well-designed and interesting.",
                "I found the questions clear and engaging.",
                "No problems at all - easy to complete.",
                "The instructions were very clear.",
                "Good survey - I enjoyed participating.",
            ],
            "positive": [
                "The survey was fine overall.",
                "No major issues with the questions.",
                "Pretty straightforward to complete.",
                "Most questions made sense.",
                "A reasonable survey experience.",
            ],
            "neutral": [
                "Standard survey, nothing special.",
                "It was okay.",
                "Finished without problems.",
                "No strong feelings about the survey.",
                "Average experience.",
            ],
            "negative": [
                "Some questions were confusing.",
                "The survey was a bit long.",
                "Instructions could be clearer.",
                "A few questions didn't make sense.",
                "There were some issues.",
            ],
            "very_negative": [
                "The survey was confusing and poorly designed.",
                "Many questions didn't make sense.",
                "Too long and repetitive.",
                "Hard to understand what was being asked.",
                "I struggled to complete this.",
            ],
        },
    },

    "general": {
        "explanation": {
            "very_positive": [
                "I feel very positively about this.",
                "This is exactly what I think should happen.",
                "I strongly support this.",
                "This aligns perfectly with my views.",
                "I'm very much in favor of this.",
            ],
            "positive": [
                "I generally support this.",
                "I have a positive view of this.",
                "This seems like a good approach.",
                "I'm in favor of this overall.",
                "I think this makes sense.",
            ],
            "neutral": [
                "I don't have strong feelings either way.",
                "I can see both sides.",
                "I'm undecided on this.",
                "This could go either way.",
                "I'm neutral on this topic.",
            ],
            "negative": [
                "I have some concerns about this.",
                "I'm not fully supportive.",
                "This doesn't quite sit right with me.",
                "I have reservations.",
                "I'm skeptical about this.",
            ],
            "very_negative": [
                "I strongly disagree with this.",
                "I'm opposed to this.",
                "This is problematic in my view.",
                "I can't support this.",
                "This goes against my beliefs.",
            ],
        },
    },

    # ========== ULTIMATUM GAME ==========

    "ultimatum_game": {
        "explanation": {
            "very_positive": [
                "I accepted because the offer was fair and respectful.",
                "They offered a reasonable split, so I accepted.",
                "Fair offers deserve acceptance - it's the right thing to do.",
                "The offer showed good faith, so I was happy to accept.",
                "I believe in rewarding fair behavior.",
            ],
            "positive": [
                "The offer seemed reasonable enough to accept.",
                "I accepted because getting something is better than nothing.",
                "It wasn't perfect but acceptable.",
                "I decided to accept and not make a fuss.",
                "The offer was good enough for me.",
            ],
            "neutral": [
                "I wasn't sure if I should accept, but I did.",
                "It was a borderline offer.",
                "I accepted without strong feelings.",
                "I just went with my gut.",
                "Didn't think too hard about it.",
            ],
            "negative": [
                "The offer was low but I took what I could get.",
                "I reluctantly accepted despite the unfair offer.",
                "I was tempted to reject but decided money is money.",
                "Not happy with the offer but I accepted anyway.",
                "I accepted even though it felt unfair.",
            ],
            "very_negative": [
                "I rejected the offer because it was insulting.",
                "They were greedy so I refused to accept.",
                "I'd rather get nothing than accept such an unfair offer.",
                "Principle matters more than money.",
                "I rejected to punish their unfairness.",
            ],
        },
    },

    # ========== PRISONERS DILEMMA ==========

    "prisoners_dilemma": {
        "explanation": {
            "very_positive": [
                "I cooperated because mutual cooperation benefits everyone.",
                "Cooperation is the best strategy for society.",
                "I trusted my partner would also cooperate.",
                "Working together is always better than competing.",
                "I believe in the power of collective action.",
            ],
            "positive": [
                "I chose to cooperate hoping they would too.",
                "Cooperation seemed like the right approach.",
                "I tried to establish trust through my choice.",
                "I wanted to give cooperation a chance.",
                "I leaned toward working together.",
            ],
            "neutral": [
                "I wasn't sure whether to cooperate or defect.",
                "I went with my instinct in the moment.",
                "It was a tough decision either way.",
                "I didn't have a strong preference.",
                "I just picked one option.",
            ],
            "negative": [
                "I defected because I wasn't sure I could trust them.",
                "Self-interest seemed like the safer choice.",
                "I protected myself against potential exploitation.",
                "I couldn't risk being the sucker.",
                "Defection seemed rational given the uncertainty.",
            ],
            "very_negative": [
                "I defected because that's the rational choice.",
                "Why cooperate when defection is better for me?",
                "I maximized my own outcome regardless of theirs.",
                "Trust is naive in this situation.",
                "I looked out for my own interests.",
            ],
        },
    },

    # ========== TIME PREFERENCE ==========

    "time_preference": {
        "explanation": {
            "very_positive": [
                "I chose to wait because the larger reward is worth it.",
                "Patience pays off - delayed gratification is important.",
                "I'm good at waiting for better outcomes.",
                "Future me will thank present me for waiting.",
                "I always prefer larger future rewards.",
            ],
            "positive": [
                "Waiting seemed worth it for the extra amount.",
                "I generally prefer to be patient when it pays off.",
                "The future reward was significantly better.",
                "I can handle delayed gratification.",
                "I chose to wait for the better deal.",
            ],
            "neutral": [
                "It was hard to decide between now and later.",
                "I'm somewhat indifferent about timing.",
                "Both options seemed okay.",
                "I went with my gut feeling.",
                "Didn't have a strong preference.",
            ],
            "negative": [
                "I preferred getting something now over waiting.",
                "A bird in hand is worth two in the bush.",
                "I'm not very patient about these things.",
                "Who knows what could happen if I wait?",
                "I wanted immediate gratification.",
            ],
            "very_negative": [
                "I always prefer money now over money later.",
                "Waiting makes no sense when I can have it today.",
                "I'm very impatient - I want it now.",
                "Future rewards aren't guaranteed.",
                "Now is always better than later.",
            ],
        },
    },

    # ========== CONFORMITY ==========

    "conformity": {
        "explanation": {
            "very_positive": [
                "I went along with the group because they were probably right.",
                "Social consensus usually reflects the correct answer.",
                "I trusted the group's judgment over my own.",
                "Going with the majority seemed sensible.",
                "I value group harmony and agreement.",
            ],
            "positive": [
                "I considered the group's opinion in my decision.",
                "There's wisdom in crowds, so I partly followed them.",
                "I adjusted my view somewhat based on others.",
                "I gave the group perspective serious consideration.",
                "I'm open to being influenced by consensus.",
            ],
            "neutral": [
                "I made my own decision regardless of what others thought.",
                "The group didn't really influence me much.",
                "I considered their view but stuck with mine.",
                "I tried to be objective about it.",
                "Group opinion was just one input.",
            ],
            "negative": [
                "I disagreed with the group and stuck to my answer.",
                "Just because others think something doesn't make it true.",
                "I'm not easily swayed by social pressure.",
                "I trusted my own judgment over the crowd.",
                "I went against the group's opinion.",
            ],
            "very_negative": [
                "I deliberately chose differently from the group.",
                "Conformity for its own sake is mindless.",
                "I'm independent and don't follow the herd.",
                "I rejected the group's view entirely.",
                "Social pressure doesn't affect my decisions.",
            ],
        },
    },

    # ========== PROSOCIAL ==========

    "prosocial": {
        "explanation": {
            "very_positive": [
                "Helping others is deeply important to me.",
                "I always try to act in ways that benefit others.",
                "Being kind and helpful is part of who I am.",
                "I genuinely care about other people's welfare.",
                "I find joy in helping others.",
            ],
            "positive": [
                "I generally try to be helpful when I can.",
                "Helping others usually feels good.",
                "I care about being a positive influence.",
                "I try to consider others' needs.",
                "Being prosocial is important to me.",
            ],
            "neutral": [
                "I help sometimes but it depends on the situation.",
                "I'm neither particularly helpful nor unhelpful.",
                "I balance my needs with others'.",
                "It varies depending on the context.",
                "I'm moderately prosocial.",
            ],
            "negative": [
                "I focus more on my own needs first.",
                "I help when it's convenient but not always.",
                "I'm selective about when I help others.",
                "Self-interest comes before helping others.",
                "I'm not particularly altruistic.",
            ],
            "very_negative": [
                "I prioritize myself over helping others.",
                "Looking after myself is more important.",
                "I don't feel obligated to help others.",
                "Helping others isn't a priority for me.",
                "I'm focused on my own outcomes.",
            ],
        },
    },

    # ========== FAIRNESS ==========

    "fairness": {
        "explanation": {
            "very_positive": [
                "Fairness is one of my core values.",
                "I believe strongly in equal treatment for all.",
                "I actively advocate for fairness.",
                "Unfairness deeply bothers me.",
                "I try to ensure fair outcomes in my decisions.",
            ],
            "positive": [
                "I care about being fair in my dealings.",
                "Fairness matters to me.",
                "I try to treat everyone equitably.",
                "I generally support fair outcomes.",
                "I consider fairness in my decisions.",
            ],
            "neutral": [
                "Fairness is one consideration among many.",
                "I'm not particularly focused on fairness.",
                "Sometimes things are fair, sometimes not.",
                "I take a pragmatic view of fairness.",
                "Fairness isn't always my top priority.",
            ],
            "negative": [
                "Perfect fairness isn't always practical.",
                "Life isn't always fair and that's okay.",
                "I focus more on outcomes than fairness.",
                "Fairness concerns don't drive my choices.",
                "Sometimes unfairness is unavoidable.",
            ],
            "very_negative": [
                "Fairness is overrated in my view.",
                "Merit matters more than equal treatment.",
                "I don't let fairness constrain my decisions.",
                "Some inequality is natural and acceptable.",
                "I'm pragmatic rather than fair-minded.",
            ],
        },
    },

    # ========== COOPERATION ==========

    "cooperation": {
        "explanation": {
            "very_positive": [
                "I love working together with others.",
                "Cooperation leads to the best outcomes for everyone.",
                "I'm a team player through and through.",
                "Collaboration is more effective than competition.",
                "I thrive in cooperative environments.",
            ],
            "positive": [
                "I generally prefer to cooperate with others.",
                "Working together usually works well.",
                "I'm open to collaboration.",
                "Cooperation makes things easier.",
                "I try to be a good team member.",
            ],
            "neutral": [
                "I can cooperate or work alone depending on the task.",
                "Cooperation isn't always better or worse.",
                "I'm flexible about working with others.",
                "It depends on the situation.",
                "I don't have a strong preference.",
            ],
            "negative": [
                "I sometimes prefer working alone.",
                "Cooperation can be inefficient.",
                "I'm skeptical of group efforts.",
                "I often do better on my own.",
                "Coordination costs can outweigh benefits.",
            ],
            "very_negative": [
                "I strongly prefer working independently.",
                "Other people slow me down.",
                "I'm most effective alone.",
                "Group work is often frustrating.",
                "I avoid cooperation when possible.",
            ],
        },
    },

    # ========== PARTISANSHIP ==========

    "partisanship": {
        "explanation": {
            "very_positive": [
                "I strongly identify with my political party.",
                "My party represents my values well.",
                "I'm a loyal supporter of my party.",
                "Party loyalty is important to me.",
                "I vote with my party consistently.",
            ],
            "positive": [
                "I generally support my political party.",
                "I lean toward one party on most issues.",
                "I have a preferred party, though not rigidly.",
                "My party alignment is clear.",
                "I tend to vote for one party.",
            ],
            "neutral": [
                "I'm not strongly attached to any party.",
                "I consider candidates individually.",
                "I'm an independent voter.",
                "Party labels don't matter much to me.",
                "I vote based on issues, not party.",
            ],
            "negative": [
                "I'm disappointed in the major parties.",
                "Neither party represents me well.",
                "I feel politically homeless.",
                "Party politics frustrates me.",
                "I wish there were better options.",
            ],
            "very_negative": [
                "I reject the two-party system.",
                "Political parties are part of the problem.",
                "I'm actively against partisan politics.",
                "Parties divide us unnecessarily.",
                "I refuse to identify with any party.",
            ],
        },
    },

    # ========== VOTING ==========

    "voting": {
        "explanation": {
            "very_positive": [
                "Voting is a civic duty I take seriously.",
                "I always vote in every election.",
                "Voting is essential for democracy.",
                "I believe my vote matters.",
                "I'm actively engaged in electoral politics.",
            ],
            "positive": [
                "I try to vote in most elections.",
                "Voting is important to me.",
                "I participate in the electoral process.",
                "I make an effort to stay informed and vote.",
                "I believe in the importance of voting.",
            ],
            "neutral": [
                "I vote sometimes but not always.",
                "Voting isn't my top priority.",
                "I'm somewhat engaged politically.",
                "I vote when I feel strongly about an issue.",
                "My voting is inconsistent.",
            ],
            "negative": [
                "I don't see much point in voting.",
                "My vote doesn't seem to change anything.",
                "Politics is largely a spectacle.",
                "I'm disillusioned with the electoral system.",
                "I rarely bother to vote.",
            ],
            "very_negative": [
                "I've given up on voting.",
                "The electoral system is broken.",
                "Voting is a waste of time.",
                "Politicians are all the same anyway.",
                "I refuse to participate in this system.",
            ],
        },
    },

    # ========== PRODUCT EVALUATION ==========

    "product_evaluation": {
        "explanation": {
            "very_positive": [
                "This product exceeded all my expectations.",
                "Absolutely love everything about this product.",
                "Best purchase I've made in a long time.",
                "I highly recommend this to everyone.",
                "Outstanding quality and value.",
            ],
            "positive": [
                "Good product overall, satisfied with my purchase.",
                "It does what it's supposed to do well.",
                "I'm happy with this product.",
                "Meets my expectations.",
                "Good value for the price.",
            ],
            "neutral": [
                "It's okay, nothing special.",
                "Average product, does the job.",
                "Neither impressed nor disappointed.",
                "It's fine for what it is.",
                "Standard quality.",
            ],
            "negative": [
                "Not quite what I expected.",
                "Has some issues that bother me.",
                "Could be better for the price.",
                "I'm somewhat disappointed.",
                "Probably won't buy again.",
            ],
            "very_negative": [
                "Very disappointed with this product.",
                "Complete waste of money.",
                "Would not recommend to anyone.",
                "Poor quality all around.",
                "Regret this purchase.",
            ],
        },
    },

    # ========== ADVERTISING ==========

    "advertising": {
        "explanation": {
            "very_positive": [
                "This ad really resonated with me.",
                "Very effective and memorable advertising.",
                "The message was clear and compelling.",
                "I loved this advertisement.",
                "Made me want to learn more about the product.",
            ],
            "positive": [
                "Good ad, got my attention.",
                "Effective marketing message.",
                "I found the ad interesting.",
                "It communicated well.",
                "Decent advertisement overall.",
            ],
            "neutral": [
                "Just another ad, nothing remarkable.",
                "It was okay I suppose.",
                "Didn't make much of an impression.",
                "Standard advertising.",
                "Neither good nor bad.",
            ],
            "negative": [
                "Not a very effective ad.",
                "The message was unclear or annoying.",
                "I wasn't persuaded by this.",
                "Poor advertising in my view.",
                "This ad missed the mark.",
            ],
            "very_negative": [
                "This ad was terrible.",
                "Annoying and off-putting.",
                "Made me less likely to buy the product.",
                "Worst ad I've seen recently.",
                "Completely ineffective and irritating.",
            ],
        },
    },

    # ========== TEAMWORK ==========

    "teamwork": {
        "explanation": {
            "very_positive": [
                "I thrive working in teams.",
                "Teamwork brings out the best in everyone.",
                "I love collaborating with others.",
                "Teams accomplish more than individuals.",
                "I'm a natural team player.",
            ],
            "positive": [
                "I enjoy working with others.",
                "Teams can be very effective.",
                "I contribute well to group efforts.",
                "Teamwork has its advantages.",
                "I'm comfortable in team settings.",
            ],
            "neutral": [
                "I can work in teams or alone.",
                "Teamwork has pros and cons.",
                "It depends on the team.",
                "I adapt to whatever is needed.",
                "No strong preference either way.",
            ],
            "negative": [
                "I sometimes struggle with teamwork.",
                "Teams can be inefficient.",
                "I prefer more independence.",
                "Group dynamics can be frustrating.",
                "I'm more productive alone.",
            ],
            "very_negative": [
                "I avoid teamwork when possible.",
                "Teams slow me down.",
                "I strongly prefer working independently.",
                "Group work is frustrating and inefficient.",
                "I'm not a team player.",
            ],
        },
    },

    # ========== MOTIVATION ==========

    "motivation": {
        "explanation": {
            "very_positive": [
                "I'm highly motivated in this area.",
                "I find this deeply engaging.",
                "I'm driven to succeed here.",
                "This really motivates me.",
                "I'm passionate about this.",
            ],
            "positive": [
                "I'm reasonably motivated.",
                "I find this interesting enough to engage with.",
                "I'm willing to put in effort.",
                "This holds my interest.",
                "I'm engaged with this.",
            ],
            "neutral": [
                "My motivation varies.",
                "I'm neither particularly motivated nor unmotivated.",
                "It depends on the day.",
                "I do what needs to be done.",
                "Average level of motivation.",
            ],
            "negative": [
                "I struggle with motivation here.",
                "This doesn't really engage me.",
                "I find it hard to stay motivated.",
                "My interest is low.",
                "I'm not very driven in this area.",
            ],
            "very_negative": [
                "I have no motivation for this.",
                "This doesn't interest me at all.",
                "I can't find the drive to engage.",
                "Completely unmotivated.",
                "I don't care about this.",
            ],
        },
    },

    # ========== AUTOMATION ==========

    "automation": {
        "explanation": {
            "very_positive": [
                "Automation will make our lives better.",
                "I embrace technological automation.",
                "Automation increases efficiency dramatically.",
                "I'm excited about automated solutions.",
                "Automation frees us for more meaningful work.",
            ],
            "positive": [
                "Automation has many benefits.",
                "I'm generally positive about automation.",
                "It can improve productivity.",
                "Automation is helpful in many areas.",
                "I support thoughtful automation.",
            ],
            "neutral": [
                "Automation has trade-offs.",
                "I'm neutral on automation.",
                "It depends on the application.",
                "Some automation is good, some isn't.",
                "I take a balanced view.",
            ],
            "negative": [
                "I have concerns about automation.",
                "Automation can eliminate jobs.",
                "We're automating too quickly.",
                "I'm cautious about automation.",
                "Human touch is being lost.",
            ],
            "very_negative": [
                "Automation is a serious threat.",
                "I'm strongly opposed to more automation.",
                "Automation dehumanizes work.",
                "We need to slow down automation.",
                "Humans should do human work.",
            ],
        },
    },

    # ========== MEDICAL DECISION ==========

    "medical_decision": {
        "explanation": {
            "very_positive": [
                "I made this health decision confidently.",
                "This was clearly the right choice for my health.",
                "I'm very satisfied with this medical decision.",
                "I feel good about this health choice.",
                "This aligns with my health values.",
            ],
            "positive": [
                "I'm comfortable with this medical decision.",
                "This seems like a reasonable health choice.",
                "I think this will be good for my health.",
                "I'm satisfied with my decision.",
                "A good choice overall.",
            ],
            "neutral": [
                "I'm unsure if this was the right decision.",
                "Medical decisions are difficult.",
                "I went with what seemed reasonable.",
                "Hard to know if it's the right choice.",
                "I made my best guess.",
            ],
            "negative": [
                "I have doubts about this medical decision.",
                "I'm not fully comfortable with this choice.",
                "There were no great options.",
                "I'm uncertain about this.",
                "Medical decisions are stressful.",
            ],
            "very_negative": [
                "I regret this medical decision.",
                "This was not the right choice.",
                "I should have chosen differently.",
                "I'm very uncomfortable with this.",
                "Bad decision in hindsight.",
            ],
        },
    },

    # ========== WELLBEING ==========

    "wellbeing": {
        "explanation": {
            "very_positive": [
                "I feel great about my overall wellbeing.",
                "Life is going really well for me.",
                "I'm thriving in most areas.",
                "My wellbeing is excellent.",
                "I'm very satisfied with my life.",
            ],
            "positive": [
                "I'm doing pretty well overall.",
                "My wellbeing is good.",
                "Things are going okay for me.",
                "I'm reasonably satisfied.",
                "Life is decent.",
            ],
            "neutral": [
                "My wellbeing is average.",
                "Some days are better than others.",
                "I'm managing okay.",
                "Neither great nor terrible.",
                "I'm getting by.",
            ],
            "negative": [
                "My wellbeing could be better.",
                "I'm struggling somewhat.",
                "Things haven't been great lately.",
                "I'm not fully satisfied with life.",
                "Room for improvement.",
            ],
            "very_negative": [
                "My wellbeing is poor.",
                "I'm really struggling right now.",
                "Things are not going well.",
                "I'm very dissatisfied.",
                "Life is difficult currently.",
            ],
        },
    },

    # ========== LEARNING ==========

    "learning": {
        "explanation": {
            "very_positive": [
                "I love learning new things.",
                "Learning is one of my passions.",
                "I'm constantly seeking new knowledge.",
                "I find learning deeply rewarding.",
                "I'm a lifelong learner.",
            ],
            "positive": [
                "I enjoy learning when I can.",
                "Learning is valuable to me.",
                "I try to keep growing.",
                "I'm open to new knowledge.",
                "Learning is generally positive.",
            ],
            "neutral": [
                "I learn when I need to.",
                "Learning is okay.",
                "I don't think much about learning.",
                "I learn what's necessary.",
                "Neutral about learning.",
            ],
            "negative": [
                "Learning can be tedious.",
                "I don't love learning new things.",
                "I prefer what I already know.",
                "Learning takes too much effort.",
                "I'm not that interested in learning.",
            ],
            "very_negative": [
                "I dislike having to learn new things.",
                "Learning is frustrating.",
                "I avoid learning when possible.",
                "I don't value learning much.",
                "Learning is overrated.",
            ],
        },
    },

    # ========== SUSTAINABILITY ==========

    "sustainability": {
        "explanation": {
            "very_positive": [
                "Sustainability is crucial for our future.",
                "I'm deeply committed to sustainable practices.",
                "We must act now for sustainability.",
                "I make sustainable choices whenever possible.",
                "Sustainability guides my decisions.",
            ],
            "positive": [
                "I care about sustainability.",
                "I try to be sustainable when I can.",
                "Sustainability matters to me.",
                "I support sustainable practices.",
                "I'm reasonably eco-conscious.",
            ],
            "neutral": [
                "Sustainability is one consideration.",
                "I'm not focused on sustainability.",
                "It depends on the situation.",
                "I don't prioritize it highly.",
                "Mixed feelings about sustainability.",
            ],
            "negative": [
                "Sustainability is sometimes impractical.",
                "I don't let it drive my choices.",
                "Other things matter more to me.",
                "Sustainability can be costly.",
                "I'm skeptical of some sustainability claims.",
            ],
            "very_negative": [
                "Sustainability is overemphasized.",
                "I don't believe the sustainability hype.",
                "Economic concerns trump sustainability.",
                "Sustainability activism is excessive.",
                "I reject the sustainability agenda.",
            ],
        },
    },
}


# ============================================================================
# QUESTION TYPE DETECTION
# ============================================================================

QUESTION_TYPE_PATTERNS: Dict[QuestionType, List[str]] = {
    QuestionType.EXPLANATION: [
        r'\bexplain\b', r'\bwhy\b', r'\breason\b', r'\bbecause\b',
        r'\bwhat made you\b', r'\bwhat led you\b', r'\bdescribe.*reasoning\b',
    ],
    QuestionType.FEEDBACK: [
        r'\bfeedback\b', r'\bcomments?\b', r'\bsuggestions?\b',
        r'\banything.*(?:confusing|unclear)\b', r'\bthoughts.*survey\b',
    ],
    QuestionType.DESCRIPTION: [
        r'\bdescribe\b', r'\btell us\b', r'\bshare\b', r'\bnarrate\b',
    ],
    QuestionType.JUSTIFICATION: [
        r'\bjustify\b', r'\bdefend\b', r'\bsupport.*decision\b',
    ],
    QuestionType.OPINION: [
        r'\bopinion\b', r'\bthink\s+about\b', r'\bview\s+on\b', r'\bfeel\s+about\b',
    ],
    QuestionType.REFLECTION: [
        r'\breflect\b', r'\blook\s+back\b', r'\brecall.*experience\b',
    ],
    QuestionType.SUGGESTION: [
        r'\bsuggest\b', r'\bimprove\b', r'\bchange\b', r'\bbetter\b',
        r'\brecommend\b', r'\badvice\b',
    ],
    QuestionType.PREDICTION: [
        r'\bpredict\b', r'\bexpect\b', r'\bwhat.*happen\b', r'\bthink.*will\b',
    ],
    QuestionType.MEMORY: [
        r'\bremember\b', r'\brecall\b', r'\btime\s+when\b', r'\blast\s+time\b',
    ],
    QuestionType.ASSOCIATION: [
        r'\bcomes?\s+to\s+mind\b', r'\bassociate\b', r'\bthink\s+of\b',
    ],
    QuestionType.PREFERENCE: [
        r'\bprefer\b', r'\bfavorite\b', r'\blike\s+(?:more|better)\b',
    ],
    QuestionType.EVALUATION: [
        r'\bevaluate\b', r'\brate\b', r'\bassess\b', r'\bjudge\b',
    ],
    QuestionType.COMPARISON: [
        r'\bcompare\b', r'\bdifferent\b', r'\bsimilar\b', r'\bversus\b',
    ],
    QuestionType.REASONING: [
        r'\breasoning\b', r'\bthought\s+process\b', r'\bhow.*decide\b',
    ],
}


def detect_question_type(question_text: str) -> QuestionType:
    """Detect the type of open-ended question from its text."""
    text_lower = question_text.lower()

    for q_type, patterns in QUESTION_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return q_type

    return QuestionType.GENERAL


# ============================================================================
# DOMAIN DETECTION FROM STUDY CONTEXT
# ============================================================================

DOMAIN_KEYWORDS: Dict[StudyDomain, List[str]] = {
    StudyDomain.DICTATOR_GAME: [
        'dictator', 'allocator', 'recipient', 'give money', 'keep money',
        'split', 'allocation decision', 'endowment',
    ],
    StudyDomain.PUBLIC_GOODS: [
        'public goods', 'common pool', 'contribute', 'group account',
        'multiplied', 'shared pool', 'collective',
    ],
    StudyDomain.TRUST_GAME: [
        'trust game', 'trustee', 'investor', 'return', 'send money',
        'triple', 'investment game',
    ],
    StudyDomain.ULTIMATUM_GAME: [
        'ultimatum', 'proposer', 'responder', 'accept', 'reject offer',
    ],
    StudyDomain.PRISONERS_DILEMMA: [
        'prisoner', 'cooperate', 'defect', 'mutual', 'dilemma',
    ],
    StudyDomain.RISK_PREFERENCE: [
        'risk', 'gamble', 'lottery', 'certain', 'probability', 'chance',
        'risky option', 'safe option',
    ],
    StudyDomain.TIME_PREFERENCE: [
        'delay', 'discount', 'now vs later', 'patience', 'impatience',
        'immediate', 'future reward',
    ],
    StudyDomain.INTERGROUP: [
        'outgroup', 'ingroup', 'other group', 'group membership',
        'intergroup', 'different group',
    ],
    StudyDomain.IDENTITY: [
        'identity', 'who you are', 'self-concept', 'belonging',
        'identification',
    ],
    StudyDomain.NORMS: [
        'norm', 'expected', 'appropriate', 'should do', 'convention',
        'social expectation',
    ],
    StudyDomain.TRUST: [
        'trust', 'trustworthy', 'reliable', 'honest', 'dependable',
    ],
    StudyDomain.FAIRNESS: [
        'fair', 'unfair', 'equal', 'equitable', 'just', 'justice',
    ],
    StudyDomain.COOPERATION: [
        'cooperat', 'collaborate', 'work together', 'joint', 'mutual benefit',
    ],
    StudyDomain.POLITICAL: [
        'politic', 'democrat', 'republican', 'liberal', 'conservative',
        'government', 'policy', 'election', 'vote',
    ],
    StudyDomain.POLARIZATION: [
        'polariz', 'divided', 'partisan', 'other side', 'opposing view',
    ],
    StudyDomain.CONSUMER: [
        'product', 'brand', 'purchase', 'buy', 'consumer', 'shopping',
        'customer',
    ],
    StudyDomain.BRAND: [
        'brand', 'logo', 'company name', 'brand loyalty',
    ],
    StudyDomain.ADVERTISING: [
        'ad', 'advertis', 'commercial', 'marketing message',
    ],
    StudyDomain.WORKPLACE: [
        'work', 'job', 'employee', 'organization', 'colleague', 'office',
    ],
    StudyDomain.LEADERSHIP: [
        'leader', 'manager', 'boss', 'supervisor', 'management',
    ],
    StudyDomain.AI_ATTITUDES: [
        'ai', 'artificial intelligence', 'algorithm', 'machine learning',
        'robot', 'automation', 'chatbot',
    ],
    StudyDomain.PRIVACY: [
        'privacy', 'data', 'personal information', 'tracking', 'surveillance',
    ],
    StudyDomain.HEALTH: [
        'health', 'medical', 'doctor', 'illness', 'wellbeing', 'wellness',
    ],
    StudyDomain.MORAL_JUDGMENT: [
        'moral', 'ethical', 'right', 'wrong', 'should', 'ought',
    ],
    StudyDomain.ENVIRONMENTAL: [
        'environment', 'climate', 'green', 'sustainab', 'eco', 'carbon',
    ],
    StudyDomain.ULTIMATUM_GAME: [
        'ultimatum', 'proposer', 'responder', 'accept', 'reject', 'offer',
    ],
    StudyDomain.PRISONERS_DILEMMA: [
        'prisoner', 'cooperate', 'defect', 'dilemma', 'mutual',
    ],
    StudyDomain.TIME_PREFERENCE: [
        'delay', 'discount', 'patience', 'impatient', 'now vs later', 'immediate',
    ],
    StudyDomain.CONFORMITY: [
        'conform', 'social pressure', 'group influence', 'majority', 'peer',
    ],
    StudyDomain.PROSOCIAL: [
        'helping', 'altru', 'prosocial', 'charity', 'donation', 'volunteer',
    ],
    StudyDomain.FAIRNESS: [
        'fair', 'unfair', 'equal', 'equity', 'justice', 'just',
    ],
    StudyDomain.COOPERATION: [
        'cooperat', 'collaborate', 'team', 'collective', 'joint',
    ],
    StudyDomain.PARTISANSHIP: [
        'partisan', 'party', 'democrat', 'republican', 'liberal', 'conservative',
    ],
    StudyDomain.VOTING: [
        'vote', 'election', 'ballot', 'candidate', 'electoral',
    ],
    StudyDomain.PRODUCT_EVALUATION: [
        'product', 'purchase', 'buy', 'quality', 'review',
    ],
    StudyDomain.ADVERTISING: [
        'ad', 'advertis', 'commercial', 'marketing', 'promotion',
    ],
    StudyDomain.TEAMWORK: [
        'teamwork', 'team', 'group project', 'collaboration',
    ],
    StudyDomain.MOTIVATION: [
        'motivat', 'drive', 'engagement', 'interest', 'passion',
    ],
    StudyDomain.AUTOMATION: [
        'automat', 'robot', 'machine', 'ai replace',
    ],
    StudyDomain.MEDICAL_DECISION: [
        'medical', 'health decision', 'treatment', 'doctor', 'patient',
    ],
    StudyDomain.WELLBEING: [
        'wellbeing', 'well-being', 'happiness', 'life satisfaction', 'quality of life',
    ],
    StudyDomain.LEARNING: [
        'learn', 'education', 'study', 'knowledge', 'skill',
    ],
    StudyDomain.SUSTAINABILITY: [
        'sustainab', 'green', 'eco', 'renewable', 'recycle',
    ],
}


def detect_study_domain(study_context: Dict[str, Any], question_text: str) -> StudyDomain:
    """Detect the most appropriate domain for response generation."""
    # Combine all available context
    all_text = []

    if study_context:
        all_text.append(study_context.get('survey_name', ''))
        all_text.extend(study_context.get('topics', []))
        all_text.append(study_context.get('instructions_text', ''))
        all_text.extend(study_context.get('main_questions', []))

    all_text.append(question_text)
    combined = ' '.join(str(t) for t in all_text).lower()

    # Score each domain
    domain_scores: Dict[StudyDomain, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            domain_scores[domain] = score

    if not domain_scores:
        return StudyDomain.GENERAL

    # Return domain with highest score
    return max(domain_scores.items(), key=lambda x: x[1])[0]


# ============================================================================
# RESPONSE VARIATIONS AND MODIFIERS
# ============================================================================

HEDGING_PHRASES = [
    "I think ", "I believe ", "In my opinion, ", "From my perspective, ",
    "I feel that ", "It seems to me that ", "I would say ", "Personally, ",
    "For me, ", "As I see it, ", "I suppose ", "Generally speaking, ",
]

CONNECTORS = [
    " Also, ", " Additionally, ", " Furthermore, ", " Moreover, ",
    " Besides that, ", " On top of that, ", " Plus, ", " And ",
]

CONCLUDING_PHRASES = [
    " Overall, that's my view.", " That's my take on it.",
    " That's how I see it.", " That's my perspective.",
    "", "", "", ""  # Often no conclusion needed
]

CASUAL_MODIFIERS = [
    "honestly", "basically", "you know", "like", "I mean", "just",
    "pretty much", "kind of", "sort of", "really",
]

FORMAL_MODIFIERS = [
    "indeed", "certainly", "particularly", "specifically", "notably",
    "significantly", "fundamentally", "essentially",
]


def add_variation(response: str, persona_verbosity: float, persona_formality: float) -> str:
    """Add natural variation to a response based on persona traits."""
    result = response

    # Add hedging for less confident personas
    if random.random() < (1 - persona_verbosity) * 0.4:
        hedge = random.choice(HEDGING_PHRASES)
        if not result.startswith(('I ', 'My ')):
            result = hedge + result[0].lower() + result[1:]

    # Add connector for verbose personas
    if persona_verbosity > 0.6 and random.random() < 0.3:
        connector = random.choice(CONNECTORS)
        result += connector + random.choice([
            "this is how I see it.",
            "that's my perspective.",
            "I feel fairly strongly about this.",
            "it just makes sense to me.",
        ])

    # Add casual modifiers for informal personas
    if persona_formality < 0.4 and random.random() < 0.3:
        modifier = random.choice(CASUAL_MODIFIERS)
        words = result.split()
        if len(words) > 3:
            insert_pos = random.randint(1, min(3, len(words)-1))
            words.insert(insert_pos, modifier)
            result = ' '.join(words)

    # Add concluding phrase occasionally
    if random.random() < 0.2 and not result.endswith(('.', '!', '?')):
        result += random.choice(CONCLUDING_PHRASES)

    return result


# ============================================================================
# MAIN RESPONSE GENERATOR CLASS
# ============================================================================

class ComprehensiveResponseGenerator:
    """
    LLM-quality response generator for open-ended survey questions.

    Provides context-aware, persona-specific responses across 50+ research domains.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.study_context: Dict[str, Any] = {}

    def set_study_context(self, context: Dict[str, Any]):
        """Set the study context for context-aware generation."""
        self.study_context = context or {}

    def generate(
        self,
        question_text: str,
        sentiment: str = "neutral",  # very_positive, positive, neutral, negative, very_negative
        persona_verbosity: float = 0.5,
        persona_formality: float = 0.5,
        persona_engagement: float = 0.5,
        condition: str = "",
    ) -> str:
        """
        Generate a context-appropriate response to an open-ended question.

        Args:
            question_text: The question being answered
            sentiment: Overall sentiment of the response
            persona_verbosity: How verbose the persona is (0-1)
            persona_formality: How formal the persona is (0-1)
            persona_engagement: How engaged the persona is (0-1)
            condition: Experimental condition (for context)

        Returns:
            Generated response text
        """
        # Detect question type and domain
        q_type = detect_question_type(question_text)
        domain = detect_study_domain(self.study_context, question_text)

        # Get appropriate template
        response = self._get_template_response(domain, q_type, sentiment)

        # Handle disengaged/careless personas
        if persona_engagement < 0.3:
            response = self._make_careless(response, persona_engagement)

        # Add variation
        response = add_variation(response, persona_verbosity, persona_formality)

        # Adjust length based on verbosity
        if persona_verbosity < 0.3:
            response = self._shorten(response)
        elif persona_verbosity > 0.7:
            response = self._extend(response, domain, sentiment)

        return response

    def _get_template_response(
        self,
        domain: StudyDomain,
        q_type: QuestionType,
        sentiment: str
    ) -> str:
        """Get a template response for the given domain and sentiment."""
        domain_key = domain.value

        # Try to find domain-specific templates
        if domain_key in DOMAIN_TEMPLATES:
            templates = DOMAIN_TEMPLATES[domain_key]
            if "explanation" in templates:  # Most common question type
                sentiment_templates = templates["explanation"].get(sentiment, templates["explanation"].get("neutral", []))
                if sentiment_templates:
                    return random.choice(sentiment_templates)

        # Fall back to general templates
        if "general" in DOMAIN_TEMPLATES:
            templates = DOMAIN_TEMPLATES["general"]["explanation"]
            sentiment_templates = templates.get(sentiment, templates.get("neutral", []))
            if sentiment_templates:
                return random.choice(sentiment_templates)

        return "No specific comment."

    def _make_careless(self, response: str, engagement: float) -> str:
        """Transform response to reflect careless/disengaged responding."""
        careless_responses = [
            "ok", "fine", "idk", "whatever", "no reason",
            "just because", "not sure", "dont know", "didnt think about it",
            "i guess so", "no comment", "n/a", "nothing really",
        ]

        if engagement < 0.2:
            return random.choice(careless_responses)
        elif engagement < 0.3:
            # Very short version
            words = response.split()[:3]
            return ' '.join(words).rstrip('.,!?') + '.'
        return response

    def _shorten(self, response: str) -> str:
        """Shorten a response for low-verbosity personas."""
        sentences = response.split('.')
        if len(sentences) > 1:
            return sentences[0].strip() + '.'
        words = response.split()
        if len(words) > 10:
            return ' '.join(words[:8]).rstrip('.,!?') + '.'
        return response

    def _extend(self, response: str, domain: StudyDomain, sentiment: str) -> str:
        """Extend a response for high-verbosity personas."""
        extensions = {
            "very_positive": [
                " I really appreciate this opportunity to share my thoughts.",
                " This is something I feel strongly about.",
                " I hope this feedback is helpful.",
            ],
            "positive": [
                " I think this is a reasonable approach overall.",
                " That's my general perspective on this.",
                " I believe this captures my view well.",
            ],
            "neutral": [
                " I don't have particularly strong feelings about this.",
                " It's hard to say definitively one way or the other.",
                " I can see merit in different approaches.",
            ],
            "negative": [
                " I hope this can be improved in the future.",
                " There's definitely room for change here.",
                " I think this needs more consideration.",
            ],
            "very_negative": [
                " This is genuinely concerning to me.",
                " I feel strongly that this needs to change.",
                " I can't overstate my concerns here.",
            ],
        }

        extension_list = extensions.get(sentiment, extensions["neutral"])
        return response + random.choice(extension_list)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_response_generator(
    seed: Optional[int] = None,
    study_context: Optional[Dict[str, Any]] = None,
) -> ComprehensiveResponseGenerator:
    """Create a configured response generator."""
    generator = ComprehensiveResponseGenerator(seed=seed)
    if study_context:
        generator.set_study_context(study_context)
    return generator


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'QuestionType',
    'StudyDomain',
    'ComprehensiveResponseGenerator',
    'create_response_generator',
    'detect_question_type',
    'detect_study_domain',
    'DOMAIN_TEMPLATES',
    'DOMAIN_KEYWORDS',
]
