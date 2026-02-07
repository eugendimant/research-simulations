"""
Comprehensive Response Library for Open-Ended Survey Questions
==============================================================

This module provides LLM-quality text generation for survey responses across
225+ research domains with dynamic context adaptation. Generates responses
that are indistinguishable from real human participants.

Features:
- 225+ domain-specific template sets covering all major research areas
- 40 question type handlers for comprehensive question coverage
- Dynamic context detection from QSF content and survey structure
- Persona-aware response generation with 12 persona dimensions
- Markov chain text generation with domain-specific corpora
- Sophisticated sentiment mapping (5-level with nuanced variations)
- Condition-aware response adaptation
- Careless/inattentive responding simulation
- Cross-cultural response style variations
- Variation generation for 1000s of unique responses

Supported Research Domains (225+):
- Behavioral Economics (12): dictator, ultimatum, trust, public goods, risk, time...
- Social Psychology (15): intergroup, identity, norms, conformity, prosocial...
- Political Science (10): polarization, partisanship, voting, media, policy...
- Consumer/Marketing (10): product, brand, advertising, purchase intent, loyalty...
- Organizational Behavior (10): workplace, leadership, teamwork, motivation...
- Technology/AI (10): attitudes, privacy, automation, algorithm aversion...
- Health Psychology (10): medical decisions, wellbeing, behavior change...
- Education (8): learning, assessment, feedback, engagement, motivation...
- Ethics/Moral (8): judgment, dilemmas, values, responsibility...
- Environmental (8): sustainability, climate attitudes, conservation...
- Cognitive Psychology (8): decision-making, memory, attention, reasoning...
- Developmental (6): parenting, childhood, aging, life transitions...
- Clinical (6): anxiety, depression, coping, therapy attitudes...
- Economics (6): negotiation, bargaining, framing, anchoring...
- Communication (6): persuasion, media effects, social media, misinformation...
- Neuroscience (6): neuroeconomics, reward, impulse control, cognitive load...
- Sports Psychology (6): motivation, team dynamics, performance anxiety...
- Legal Psychology (6): jury, witness, procedural justice, compliance...
- Food/Nutrition (6): eating behavior, food choice, body image, diet...
- Human Factors (6): user experience, interface design, safety, workload...
- Cross-Cultural (5): values, acculturation, identity, global attitudes...
- Positive Psychology (5): gratitude, resilience, flourishing, satisfaction...
- Gender/Sexuality (4): roles, attitudes, LGBTQ experience...
- Relationships (4): attachment, romantic, social support...
- Financial Psychology (6): literacy, investment, debt, retirement, stress...
- Personality Psychology (6): big five, narcissism, dark triad, self-concept...
- Gaming/Entertainment (6): esports, gambling, streaming, virtual reality...
- Social Media Research (6): online identity, influencer, communities...
- Workplace Behavior (6): remote work, diversity, burnout, career...
- Decision Science (6): choice architecture, nudge, defaults, regret...
- Trust & Credibility (5): institutional, expert, source, science, media...
- Innovation & Creativity (5): entrepreneurship, idea generation, process...
- Risk & Safety (5): perception, attitudes, hazards, disaster preparedness...

Question Types (40): explanation, justification, reasoning, causation, motivation,
description, narration, elaboration, detail, context, evaluation, assessment,
comparison, critique, rating_explanation, judgment, appraisal, reflection,
introspection, memory, experience, recall, opinion, belief, preference, attitude,
value, worldview, prediction, intention, suggestion, recommendation, advice,
association, impression, perception, feedback, comment, observation, general

Version: 1.2.2 - New tab-based UI
"""

__version__ = "1.3.9"

import random
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class QuestionType(Enum):
    """Types of open-ended questions in surveys (40 types for comprehensive coverage).

    This extensive taxonomy covers all common open-ended question formats found
    in behavioral science, marketing, psychology, and social science research.
    Each type has specialized response generation templates.
    """
    # ========== EXPLANATORY QUESTIONS (Why/How) ==========
    EXPLANATION = "explanation"           # "Please explain your choice..."
    JUSTIFICATION = "justification"       # "Why did you choose..."
    REASONING = "reasoning"               # "What reasoning led to..."
    CAUSATION = "causation"               # "What caused you to..."
    MOTIVATION = "motivation"             # "What motivated your decision..."

    # ========== DESCRIPTIVE QUESTIONS ==========
    DESCRIPTION = "description"           # "Describe your experience..."
    NARRATION = "narration"               # "Tell us about a time..."
    ELABORATION = "elaboration"           # "Please elaborate on..."
    DETAIL = "detail"                     # "Provide more details about..."
    CONTEXT = "context"                   # "Provide context for..."

    # ========== EVALUATIVE QUESTIONS ==========
    EVALUATION = "evaluation"             # "How would you evaluate..."
    ASSESSMENT = "assessment"             # "Assess the quality of..."
    COMPARISON = "comparison"             # "Compare X and Y..."
    CRITIQUE = "critique"                 # "What are the pros and cons..."
    RATING_EXPLANATION = "rating_explanation"  # "Why did you give this rating..."
    JUDGMENT = "judgment"                 # "What is your judgment of..."
    APPRAISAL = "appraisal"               # "How do you appraise..."

    # ========== REFLECTIVE QUESTIONS ==========
    REFLECTION = "reflection"             # "Reflect on how you felt..."
    INTROSPECTION = "introspection"       # "What were you thinking when..."
    MEMORY = "memory"                     # "Recall a time when..."
    EXPERIENCE = "experience"             # "Share your experience with..."
    RECALL = "recall"                     # "What do you remember about..."

    # ========== OPINION/ATTITUDE QUESTIONS ==========
    OPINION = "opinion"                   # "What is your opinion on..."
    BELIEF = "belief"                     # "What do you believe about..."
    PREFERENCE = "preference"             # "Why do you prefer..."
    ATTITUDE = "attitude"                 # "How do you feel about..."
    VALUE = "value"                       # "What values guided your..."
    WORLDVIEW = "worldview"               # "How does this fit your worldview..."

    # ========== FORWARD-LOOKING QUESTIONS ==========
    PREDICTION = "prediction"             # "What do you think will happen..."
    INTENTION = "intention"               # "What do you plan to do..."
    SUGGESTION = "suggestion"             # "How could we improve..."
    RECOMMENDATION = "recommendation"     # "What would you recommend..."
    ADVICE = "advice"                     # "What advice would you give..."

    # ========== ASSOCIATIVE QUESTIONS ==========
    ASSOCIATION = "association"           # "What comes to mind when..."
    IMPRESSION = "impression"             # "What was your first impression..."
    PERCEPTION = "perception"             # "How do you perceive..."

    # ========== FEEDBACK QUESTIONS ==========
    FEEDBACK = "feedback"                 # "Any feedback about the survey?"
    COMMENT = "comment"                   # "Any additional comments..."
    OBSERVATION = "observation"           # "What did you observe..."

    # ========== CATCH-ALL ==========
    GENERAL = "general"                   # Catch-all for unclassified questions


class StudyDomain(Enum):
    """Research domains for context-specific responses.

    This enum covers 175+ research domains across 24 categories:
    - Behavioral Economics (12 domains)
    - Social Psychology (15 domains)
    - Political Science (10 domains)
    - Consumer/Marketing (10 domains)
    - Organizational Behavior (10 domains)
    - Technology/AI (10 domains)
    - Health Psychology (10 domains)
    - Education (8 domains)
    - Ethics/Moral Psychology (8 domains)
    - Environmental (8 domains)
    - Cognitive Psychology (8 domains)
    - Developmental Psychology (6 domains)
    - Clinical Psychology (6 domains)
    - Communication (6 domains)
    - Economics (6 domains)
    - General (4 domains)
    - Neuroscience/Neuroeconomics (6 domains)
    - Sports Psychology (6 domains)
    - Legal Psychology (6 domains)
    - Food/Nutrition Psychology (6 domains)
    - Human Factors/Ergonomics (6 domains)
    - Cross-Cultural (5 domains)
    - Positive Psychology (5 domains)
    - Gender/Sexuality (4 domains)
    - Relationship/Attachment (4 domains)
    """
    # ========== BEHAVIORAL ECONOMICS (12 domains) ==========
    BEHAVIORAL_ECONOMICS = "behavioral_economics"
    DICTATOR_GAME = "dictator_game"
    PUBLIC_GOODS = "public_goods"
    ULTIMATUM_GAME = "ultimatum_game"
    TRUST_GAME = "trust_game"
    PRISONERS_DILEMMA = "prisoners_dilemma"
    RISK_PREFERENCE = "risk_preference"
    TIME_PREFERENCE = "time_preference"
    LOSS_AVERSION = "loss_aversion"
    FRAMING_EFFECTS = "framing_effects"
    ANCHORING = "anchoring"
    SUNK_COST = "sunk_cost"

    # ========== SOCIAL PSYCHOLOGY (15 domains) ==========
    SOCIAL_PSYCHOLOGY = "social_psychology"
    INTERGROUP = "intergroup"
    IDENTITY = "identity"
    NORMS = "norms"
    CONFORMITY = "conformity"
    PROSOCIAL = "prosocial"
    TRUST = "trust"
    FAIRNESS = "fairness"
    COOPERATION = "cooperation"
    SOCIAL_INFLUENCE = "social_influence"
    ATTRIBUTION = "attribution"
    STEREOTYPE = "stereotype"
    PREJUDICE = "prejudice"
    SELF_ESTEEM = "self_esteem"
    EMPATHY = "empathy"

    # ========== POLITICAL SCIENCE (10 domains) ==========
    POLITICAL = "political"
    POLARIZATION = "polarization"
    PARTISANSHIP = "partisanship"
    VOTING = "voting"
    MEDIA = "media"
    POLICY_ATTITUDES = "policy_attitudes"
    CIVIC_ENGAGEMENT = "civic_engagement"
    POLITICAL_TRUST = "political_trust"
    IDEOLOGY = "ideology"
    MISINFORMATION = "misinformation"

    # ========== CONSUMER/MARKETING (10 domains) ==========
    CONSUMER = "consumer"
    BRAND = "brand"
    ADVERTISING = "advertising"
    PRODUCT_EVALUATION = "product_evaluation"
    PURCHASE_INTENT = "purchase_intent"
    BRAND_LOYALTY = "brand_loyalty"
    PRICE_PERCEPTION = "price_perception"
    SERVICE_QUALITY = "service_quality"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    WORD_OF_MOUTH = "word_of_mouth"

    # ========== ORGANIZATIONAL BEHAVIOR (10 domains) ==========
    ORGANIZATIONAL = "organizational"
    WORKPLACE = "workplace"
    LEADERSHIP = "leadership"
    TEAMWORK = "teamwork"
    MOTIVATION = "motivation"
    JOB_SATISFACTION = "job_satisfaction"
    ORGANIZATIONAL_COMMITMENT = "organizational_commitment"
    WORK_LIFE_BALANCE = "work_life_balance"
    EMPLOYEE_ENGAGEMENT = "employee_engagement"
    ORGANIZATIONAL_CULTURE = "organizational_culture"

    # ========== TECHNOLOGY/AI (10 domains) ==========
    TECHNOLOGY = "technology"
    AI_ATTITUDES = "ai_attitudes"
    PRIVACY = "privacy"
    AUTOMATION = "automation"
    ALGORITHM_AVERSION = "algorithm_aversion"
    TECHNOLOGY_ADOPTION = "technology_adoption"
    SOCIAL_MEDIA = "social_media"
    DIGITAL_WELLBEING = "digital_wellbeing"
    HUMAN_AI_INTERACTION = "human_ai_interaction"
    CYBERSECURITY = "cybersecurity"

    # ========== HEALTH PSYCHOLOGY (10 domains) ==========
    HEALTH = "health"
    MEDICAL_DECISION = "medical_decision"
    WELLBEING = "wellbeing"
    HEALTH_BEHAVIOR = "health_behavior"
    MENTAL_HEALTH = "mental_health"
    VACCINATION = "vaccination"
    PAIN_MANAGEMENT = "pain_management"
    HEALTH_ANXIETY = "health_anxiety"
    PATIENT_PROVIDER = "patient_provider"
    CHRONIC_ILLNESS = "chronic_illness"

    # ========== EDUCATION (8 domains) ==========
    EDUCATION = "education"
    LEARNING = "learning"
    ACADEMIC_MOTIVATION = "academic_motivation"
    TEACHING_EFFECTIVENESS = "teaching_effectiveness"
    ONLINE_LEARNING = "online_learning"
    EDUCATIONAL_TECHNOLOGY = "educational_technology"
    STUDENT_ENGAGEMENT = "student_engagement"
    ASSESSMENT_FEEDBACK = "assessment_feedback"

    # ========== ETHICS/MORAL PSYCHOLOGY (8 domains) ==========
    ETHICS = "ethics"
    MORAL_JUDGMENT = "moral_judgment"
    MORAL_DILEMMA = "moral_dilemma"
    ETHICAL_LEADERSHIP = "ethical_leadership"
    CORPORATE_ETHICS = "corporate_ethics"
    RESEARCH_ETHICS = "research_ethics"
    MORAL_EMOTIONS = "moral_emotions"
    VALUES = "values"

    # ========== ENVIRONMENTAL (8 domains) ==========
    ENVIRONMENTAL = "environmental"
    SUSTAINABILITY = "sustainability"
    CLIMATE_ATTITUDES = "climate_attitudes"
    PRO_ENVIRONMENTAL = "pro_environmental"
    GREEN_CONSUMPTION = "green_consumption"
    CONSERVATION = "conservation"
    ENERGY_BEHAVIOR = "energy_behavior"
    ENVIRONMENTAL_JUSTICE = "environmental_justice"

    # ========== COGNITIVE PSYCHOLOGY (8 domains) ==========
    COGNITIVE = "cognitive"
    DECISION_MAKING = "decision_making"
    MEMORY = "memory"
    ATTENTION = "attention"
    REASONING = "reasoning"
    PROBLEM_SOLVING = "problem_solving"
    COGNITIVE_BIAS = "cognitive_bias"
    METACOGNITION = "metacognition"

    # ========== DEVELOPMENTAL PSYCHOLOGY (6 domains) ==========
    DEVELOPMENTAL = "developmental"
    PARENTING = "parenting"
    CHILDHOOD = "childhood"
    AGING = "aging"
    LIFE_TRANSITIONS = "life_transitions"
    INTERGENERATIONAL = "intergenerational"

    # ========== CLINICAL PSYCHOLOGY (6 domains) ==========
    CLINICAL = "clinical"
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    COPING = "coping"
    THERAPY_ATTITUDES = "therapy_attitudes"
    STRESS = "stress"

    # ========== COMMUNICATION (6 domains) ==========
    COMMUNICATION = "communication"
    PERSUASION = "persuasion"
    MEDIA_EFFECTS = "media_effects"
    INTERPERSONAL = "interpersonal"
    PUBLIC_OPINION = "public_opinion"
    NARRATIVE = "narrative"

    # ========== ECONOMICS (6 domains) ==========
    ECONOMICS = "economics"
    NEGOTIATION = "negotiation"
    BARGAINING = "bargaining"
    FINANCIAL_DECISION = "financial_decision"
    SAVING_BEHAVIOR = "saving_behavior"
    ECONOMIC_EXPECTATIONS = "economic_expectations"

    # ========== GENERAL (4 domains) ==========
    GENERAL = "general"
    SURVEY_FEEDBACK = "survey_feedback"
    OPEN_ENDED = "open_ended"
    MISCELLANEOUS = "miscellaneous"

    # ========== NEUROSCIENCE/NEUROECONOMICS (6 domains) ==========
    NEUROECONOMICS = "neuroeconomics"
    REWARD_PROCESSING = "reward_processing"
    IMPULSE_CONTROL = "impulse_control"
    EMOTIONAL_REGULATION = "emotional_regulation"
    NEURAL_DECISION = "neural_decision"
    COGNITIVE_LOAD = "cognitive_load"

    # ========== SPORTS PSYCHOLOGY (6 domains) ==========
    SPORTS_PSYCHOLOGY = "sports_psychology"
    ATHLETIC_MOTIVATION = "athletic_motivation"
    TEAM_DYNAMICS = "team_dynamics"
    PERFORMANCE_ANXIETY = "performance_anxiety"
    COACH_ATHLETE = "coach_athlete"
    FAN_BEHAVIOR = "fan_behavior"

    # ========== LEGAL PSYCHOLOGY (6 domains) ==========
    LEGAL_PSYCHOLOGY = "legal_psychology"
    JURY_DECISION = "jury_decision"
    WITNESS_MEMORY = "witness_memory"
    PROCEDURAL_JUSTICE = "procedural_justice"
    CRIMINAL_JUSTICE = "criminal_justice"
    LEGAL_COMPLIANCE = "legal_compliance"

    # ========== FOOD/NUTRITION PSYCHOLOGY (6 domains) ==========
    FOOD_PSYCHOLOGY = "food_psychology"
    EATING_BEHAVIOR = "eating_behavior"
    FOOD_CHOICE = "food_choice"
    NUTRITION_KNOWLEDGE = "nutrition_knowledge"
    BODY_IMAGE = "body_image"
    DIET_ADHERENCE = "diet_adherence"

    # ========== HUMAN FACTORS/ERGONOMICS (6 domains) ==========
    HUMAN_FACTORS = "human_factors"
    USER_EXPERIENCE = "user_experience"
    INTERFACE_DESIGN = "interface_design"
    SAFETY_BEHAVIOR = "safety_behavior"
    WORKLOAD = "workload"
    HUMAN_ERROR = "human_error"

    # ========== CROSS-CULTURAL (5 domains) ==========
    CROSS_CULTURAL = "cross_cultural"
    CULTURAL_VALUES = "cultural_values"
    ACCULTURATION = "acculturation"
    CULTURAL_IDENTITY = "cultural_identity"
    GLOBAL_ATTITUDES = "global_attitudes"

    # ========== POSITIVE PSYCHOLOGY (5 domains) ==========
    POSITIVE_PSYCHOLOGY = "positive_psychology"
    GRATITUDE = "gratitude"
    RESILIENCE = "resilience"
    FLOURISHING = "flourishing"
    LIFE_SATISFACTION = "life_satisfaction"

    # ========== GENDER/SEXUALITY (4 domains) ==========
    GENDER_PSYCHOLOGY = "gender_psychology"
    GENDER_ROLES = "gender_roles"
    SEXUALITY_ATTITUDES = "sexuality_attitudes"
    LGBTQ_EXPERIENCE = "lgbtq_experience"

    # ========== RELATIONSHIP/ATTACHMENT (4 domains) ==========
    RELATIONSHIP = "relationship"
    ATTACHMENT = "attachment"
    ROMANTIC_RELATIONSHIPS = "romantic_relationships"
    SOCIAL_SUPPORT = "social_support"

    # ========== ADDITIONAL DOMAINS (50+ NEW - Expanding to 225+) ==========

    # Financial Psychology (6 domains)
    FINANCIAL_PSYCHOLOGY = "financial_psychology"
    FINANCIAL_LITERACY = "financial_literacy"
    INVESTMENT_BEHAVIOR = "investment_behavior"
    DEBT_ATTITUDES = "debt_attitudes"
    RETIREMENT_PLANNING = "retirement_planning"
    FINANCIAL_STRESS = "financial_stress"

    # Personality Psychology (6 domains)
    PERSONALITY = "personality"
    BIG_FIVE = "big_five"
    NARCISSISM = "narcissism"
    DARK_TRIAD = "dark_triad"
    TRAIT_ASSESSMENT = "trait_assessment"
    SELF_CONCEPT = "self_concept"

    # Gaming/Entertainment (6 domains)
    GAMING_PSYCHOLOGY = "gaming_psychology"
    ESPORTS = "esports"
    GAMBLING = "gambling"
    ENTERTAINMENT_MEDIA = "entertainment_media"
    STREAMING_BEHAVIOR = "streaming_behavior"
    VIRTUAL_REALITY = "virtual_reality"

    # Social Media Research (6 domains)
    SOCIAL_MEDIA_USE = "social_media_use"
    ONLINE_IDENTITY = "online_identity"
    DIGITAL_COMMUNICATION = "digital_communication"
    INFLUENCER_MARKETING = "influencer_marketing"
    ONLINE_COMMUNITIES = "online_communities"
    SOCIAL_COMPARISON = "social_comparison"

    # Workplace Behavior (6 domains)
    REMOTE_WORK = "remote_work"
    WORKPLACE_DIVERSITY = "workplace_diversity"
    BURNOUT = "burnout"
    CAREER_DEVELOPMENT = "career_development"
    WORKPLACE_CONFLICT = "workplace_conflict"
    ORGANIZATIONAL_JUSTICE = "organizational_justice"

    # Decision Science (6 domains)
    DECISION_SCIENCE = "decision_science"
    CHOICE_ARCHITECTURE = "choice_architecture"
    NUDGE = "nudge"
    DEFAULT_EFFECTS = "default_effects"
    INFORMATION_OVERLOAD = "information_overload"
    REGRET = "regret"

    # Trust & Credibility (5 domains)
    INSTITUTIONAL_TRUST = "institutional_trust"
    EXPERT_CREDIBILITY = "expert_credibility"
    SOURCE_CREDIBILITY = "source_credibility"
    SCIENCE_TRUST = "science_trust"
    MEDIA_TRUST = "media_trust"

    # Innovation & Creativity (5 domains)
    INNOVATION = "innovation"
    CREATIVITY = "creativity"
    ENTREPRENEURSHIP = "entrepreneurship"
    IDEA_GENERATION = "idea_generation"
    CREATIVE_PROCESS = "creative_process"

    # Risk & Safety (5 domains)
    RISK_PERCEPTION = "risk_perception"
    SAFETY_ATTITUDES = "safety_attitudes"
    HAZARD_PERCEPTION = "hazard_perception"
    DISASTER_PREPAREDNESS = "disaster_preparedness"
    RISK_COMMUNICATION = "risk_communication"

    # ========== NEW DOMAINS (v2.4.5) ==========

    # AI Alignment & Ethics (6 domains)
    AI_ALIGNMENT = "ai_alignment"
    AI_ETHICS = "ai_ethics"
    AI_SAFETY = "ai_safety"
    MACHINE_VALUES = "machine_values"
    AI_GOVERNANCE = "ai_governance"
    AI_TRANSPARENCY = "ai_transparency"

    # Climate Science & Action (6 domains)
    CLIMATE_ACTION = "climate_action"
    CLIMATE_COMMUNICATION = "climate_communication"
    CARBON_FOOTPRINT = "carbon_footprint"
    CLIMATE_ADAPTATION = "climate_adaptation"
    CLIMATE_JUSTICE = "climate_justice"
    RENEWABLE_ENERGY = "renewable_energy"

    # Health Disparities (6 domains)
    HEALTH_DISPARITIES = "health_disparities"
    HEALTHCARE_ACCESS = "healthcare_access"
    HEALTH_EQUITY = "health_equity"
    SOCIAL_DETERMINANTS = "social_determinants"
    HEALTH_LITERACY = "health_literacy"
    MEDICAL_MISTRUST = "medical_mistrust"

    # Genomics & Personalized Medicine (5 domains)
    GENOMICS = "genomics"
    GENETIC_TESTING = "genetic_testing"
    PERSONALIZED_MEDICINE = "personalized_medicine"
    GENE_THERAPY = "gene_therapy"
    BIOETHICS = "bioethics"

    # Digital Society (5 domains)
    DIGITAL_DIVIDE = "digital_divide"
    ONLINE_POLARIZATION = "online_polarization"
    ALGORITHMIC_FAIRNESS = "algorithmic_fairness"
    DATA_PRIVACY = "data_privacy"
    DIGITAL_LITERACY = "digital_literacy"

    # Future of Work (5 domains)
    AUTOMATION_ANXIETY = "automation_anxiety"
    GIG_ECONOMY = "gig_economy"
    SKILLS_OBSOLESCENCE = "skills_obsolescence"
    UNIVERSAL_BASIC_INCOME = "universal_basic_income"
    HUMAN_MACHINE_COLLABORATION = "human_machine_collaboration"

    # ========== ADVANCED SOCIAL SCIENCE DOMAINS (v1.1.0) ==========

    # Cognitive & Self-Regulation (4 domains)
    SELF_CONTROL = "self_control"
    EMOTION_REGULATION = "emotion_regulation"
    CONFIRMATION_BIAS = "confirmation_bias"
    MENTAL_EFFORT = "mental_effort"

    # Interpersonal & Group (3 domains)
    RELATIONSHIP_QUALITY = "relationship_quality"
    GROUP_IDENTITY = "group_identity"
    BEHAVIOR_INTENTION = "behavior_intention"

    # Scientific Reasoning (1 domain)
    SCIENTIFIC_REASONING = "scientific_reasoning"


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

    # ========== LOSS AVERSION ==========

    "loss_aversion": {
        "explanation": {
            "very_positive": [
                "I focused on potential gains more than losses.",
                "The upside potential outweighed the risk of loss.",
                "I'm comfortable accepting some risk for gains.",
                "Losses don't worry me as much as missing opportunities.",
                "I believe in taking calculated risks.",
            ],
            "positive": [
                "I tried to balance potential gains against losses.",
                "Some risk is acceptable for good returns.",
                "I considered both sides before deciding.",
                "I'm reasonably comfortable with risk.",
                "The potential benefit seemed worth the risk.",
            ],
            "neutral": [
                "I weighed gains and losses about equally.",
                "Neither gain nor loss dominated my thinking.",
                "I tried to be rational about the trade-offs.",
                "I didn't feel strongly either way.",
                "I just went with what seemed reasonable.",
            ],
            "negative": [
                "I was worried about potential losses.",
                "The fear of losing influenced my choice.",
                "I preferred to avoid potential downsides.",
                "Losing what I have concerns me more than gaining.",
                "I'm risk-averse when it comes to losses.",
            ],
            "very_negative": [
                "I strongly prefer avoiding losses over gaining.",
                "The potential loss was too scary to risk.",
                "I can't stand the thought of losing.",
                "Protecting what I have is my priority.",
                "I chose safety over potential gains.",
            ],
        },
    },

    # ========== FRAMING EFFECTS ==========

    "framing_effects": {
        "explanation": {
            "very_positive": [
                "The way it was presented made it seem very appealing.",
                "I responded positively to how it was framed.",
                "The positive framing influenced my favorable view.",
                "I appreciated how the information was presented.",
                "The description highlighted the benefits well.",
            ],
            "positive": [
                "The framing helped me see it positively.",
                "How it was described affected my perception.",
                "I noticed the presentation was favorable.",
                "The framing was reasonable and helpful.",
                "I responded to the way it was presented.",
            ],
            "neutral": [
                "I tried to look past how it was framed.",
                "The framing didn't particularly affect me.",
                "I focused on the substance, not presentation.",
                "I didn't pay much attention to the wording.",
                "The framing seemed neutral to me.",
            ],
            "negative": [
                "The way it was framed bothered me.",
                "I noticed the negative framing.",
                "The presentation made it seem worse.",
                "I was influenced by the negative description.",
                "The framing highlighted the downsides.",
            ],
            "very_negative": [
                "The negative framing strongly affected my view.",
                "I couldn't ignore how badly it was presented.",
                "The framing made me very concerned.",
                "I was put off by how it was described.",
                "The presentation was manipulatively negative.",
            ],
        },
    },

    # ========== ANCHORING ==========

    "anchoring": {
        "explanation": {
            "very_positive": [
                "The initial value seemed like a good starting point.",
                "I adjusted appropriately from the anchor.",
                "The reference point helped me make my judgment.",
                "I used the given information effectively.",
                "The anchor provided useful context.",
            ],
            "positive": [
                "I considered the initial value in my response.",
                "The starting point influenced my estimate.",
                "I adjusted from the given anchor.",
                "The reference was helpful for context.",
                "I factored in the initial information.",
            ],
            "neutral": [
                "I'm not sure how much the anchor affected me.",
                "I tried to think independently.",
                "The initial value was just one input.",
                "I made my own judgment.",
                "I didn't rely heavily on the given number.",
            ],
            "negative": [
                "I tried to ignore the initial anchor.",
                "The starting point seemed arbitrary.",
                "I didn't want to be biased by the anchor.",
                "I questioned the given reference value.",
                "The anchor seemed potentially misleading.",
            ],
            "very_negative": [
                "I rejected the anchor as manipulative.",
                "The initial value was clearly biased.",
                "I deliberately moved far from the anchor.",
                "I didn't trust the starting point.",
                "The anchor was irrelevant to my judgment.",
            ],
        },
    },

    # ========== SUNK COST ==========

    "sunk_cost": {
        "explanation": {
            "very_positive": [
                "I didn't let past investments bias my decision.",
                "I focused on future value, not past costs.",
                "What's spent is spent - I looked ahead.",
                "I made a rational choice regardless of sunk costs.",
                "Past investment shouldn't determine future choices.",
            ],
            "positive": [
                "I tried not to let sunk costs affect me.",
                "I focused more on future benefits than past costs.",
                "I considered both but prioritized going forward.",
                "I recognized the past investment but looked ahead.",
                "I made a reasonably forward-looking decision.",
            ],
            "neutral": [
                "I weighed past investment against future value.",
                "It was hard to ignore what I'd already invested.",
                "I considered both sunk costs and future returns.",
                "I tried to be balanced in my thinking.",
                "The decision was difficult either way.",
            ],
            "negative": [
                "I felt I should continue given my investment.",
                "Abandoning what I'd invested felt wasteful.",
                "I didn't want my past investment to be for nothing.",
                "The sunk cost influenced me somewhat.",
                "I felt committed due to what I'd already put in.",
            ],
            "very_negative": [
                "I couldn't abandon my investment.",
                "I had to continue given what I'd already spent.",
                "Quitting now would waste everything.",
                "My past investment strongly influenced my choice.",
                "I felt obligated to see it through.",
            ],
        },
    },

    # ========== SOCIAL INFLUENCE ==========

    "social_influence": {
        "explanation": {
            "very_positive": [
                "I was open to others' perspectives.",
                "Social input helped inform my decision.",
                "I value learning from what others think.",
                "Others' views provided useful information.",
                "I appreciate social guidance on decisions.",
            ],
            "positive": [
                "I considered what others thought.",
                "Social information was helpful.",
                "I factored in others' opinions.",
                "Knowing what others did influenced me.",
                "I'm reasonably open to social input.",
            ],
            "neutral": [
                "Others' views were one consideration.",
                "I made my own choice regardless.",
                "Social influence wasn't a big factor.",
                "I didn't pay much attention to others.",
                "I focused on my own judgment.",
            ],
            "negative": [
                "I try not to be swayed by others.",
                "Social pressure doesn't work on me.",
                "I prefer to think independently.",
                "Others' views didn't affect my choice.",
                "I went my own way.",
            ],
            "very_negative": [
                "I deliberately chose differently from others.",
                "I reject social conformity.",
                "Others' opinions are irrelevant to me.",
                "I'm proud of my independence.",
                "Social pressure actually pushes me away.",
            ],
        },
    },

    # ========== ATTRIBUTION ==========

    "attribution": {
        "explanation": {
            "very_positive": [
                "I attributed the outcome to positive factors.",
                "The success was due to skill and effort.",
                "Internal factors explain what happened.",
                "I take responsibility for good outcomes.",
                "The situation enabled this positive result.",
            ],
            "positive": [
                "I think the outcome was deserved.",
                "Both effort and circumstances contributed.",
                "There were good reasons for this result.",
                "I can explain why this happened.",
                "The causes seem reasonable.",
            ],
            "neutral": [
                "It's hard to know what caused this.",
                "Multiple factors were probably involved.",
                "I'm not sure about the attribution.",
                "The causes aren't entirely clear.",
                "It could be many things.",
            ],
            "negative": [
                "External factors explain this outcome.",
                "The situation was largely to blame.",
                "I don't think effort mattered much.",
                "Circumstances beyond control caused this.",
                "The attribution seems unfair.",
            ],
            "very_negative": [
                "This was entirely due to circumstances.",
                "I had no control over this outcome.",
                "External factors completely explain this.",
                "Blaming internal factors would be wrong.",
                "The situation was impossible.",
            ],
        },
    },

    # ========== STEREOTYPE ==========

    "stereotype": {
        "explanation": {
            "very_positive": [
                "I treat each person as an individual.",
                "Stereotypes don't influence my judgments.",
                "I actively resist categorical thinking.",
                "Each person is unique regardless of group.",
                "I reject stereotypical thinking.",
            ],
            "positive": [
                "I try to avoid stereotyping.",
                "I give individuals a fair chance.",
                "I don't assume based on group membership.",
                "I recognize stereotypes can be wrong.",
                "I aim for individual assessment.",
            ],
            "neutral": [
                "I have mixed views on this.",
                "Sometimes categories are useful, sometimes not.",
                "I try to be fair but it's difficult.",
                "I'm aware stereotypes exist.",
                "I don't have strong feelings either way.",
            ],
            "negative": [
                "Group patterns do contain some truth.",
                "Statistical differences exist between groups.",
                "I consider group information sometimes.",
                "Categories can be informative.",
                "I use generalizations when appropriate.",
            ],
            "very_negative": [
                "Group membership is informative.",
                "Stereotypes often reflect reality.",
                "I use categorical information in judgments.",
                "Group patterns guide my expectations.",
                "Generalizations are useful heuristics.",
            ],
        },
    },

    # ========== SELF ESTEEM ==========

    "self_esteem": {
        "explanation": {
            "very_positive": [
                "I feel confident in who I am.",
                "I have a positive view of myself.",
                "I'm comfortable with my identity.",
                "I believe in my own worth.",
                "I'm proud of who I've become.",
            ],
            "positive": [
                "I generally feel good about myself.",
                "I have reasonable self-confidence.",
                "I accept myself overall.",
                "My self-image is positive.",
                "I'm fairly comfortable with myself.",
            ],
            "neutral": [
                "I have mixed feelings about myself.",
                "Some days I feel better than others.",
                "My self-esteem varies.",
                "I'm neither particularly confident nor insecure.",
                "I don't think much about my self-worth.",
            ],
            "negative": [
                "I sometimes struggle with self-doubt.",
                "My confidence could be better.",
                "I'm often critical of myself.",
                "I have some insecurities.",
                "My self-esteem isn't great.",
            ],
            "very_negative": [
                "I often feel bad about myself.",
                "I struggle with low self-esteem.",
                "I'm very self-critical.",
                "I lack confidence in myself.",
                "I have a negative self-image.",
            ],
        },
    },

    # ========== EMPATHY ==========

    "empathy": {
        "explanation": {
            "very_positive": [
                "I deeply feel what others experience.",
                "Others' emotions strongly affect me.",
                "I'm highly attuned to others' feelings.",
                "Empathy comes naturally to me.",
                "I connect deeply with others' experiences.",
            ],
            "positive": [
                "I try to understand others' feelings.",
                "I'm reasonably empathetic.",
                "I can put myself in others' shoes.",
                "I care about how others feel.",
                "Empathy is important to me.",
            ],
            "neutral": [
                "I feel empathy sometimes.",
                "It depends on the situation.",
                "I'm moderately empathetic.",
                "I don't always feel others' emotions.",
                "My empathy is average.",
            ],
            "negative": [
                "I don't always connect emotionally.",
                "Empathy doesn't come easily to me.",
                "I'm more logical than emotional.",
                "I struggle to feel others' experiences.",
                "I keep emotional distance.",
            ],
            "very_negative": [
                "I rarely feel others' emotions.",
                "I'm not very empathetic.",
                "Others' feelings don't affect me much.",
                "I prefer rational to emotional thinking.",
                "Empathy isn't my strength.",
            ],
        },
    },

    # ========== POLICY ATTITUDES ==========

    "policy_attitudes": {
        "explanation": {
            "very_positive": [
                "I strongly support this policy.",
                "This policy aligns with my values.",
                "I believe this is the right approach.",
                "This policy would benefit society.",
                "I'm enthusiastic about this direction.",
            ],
            "positive": [
                "I generally support this policy.",
                "This seems like a reasonable approach.",
                "I'm favorable toward this policy.",
                "The benefits outweigh concerns.",
                "I think this policy makes sense.",
            ],
            "neutral": [
                "I'm undecided on this policy.",
                "There are valid arguments both ways.",
                "I don't have strong views on this.",
                "This policy has pros and cons.",
                "I'm neutral about this approach.",
            ],
            "negative": [
                "I have concerns about this policy.",
                "I'm skeptical this would work.",
                "I disagree with this approach.",
                "The downsides concern me.",
                "I'm not convinced this is right.",
            ],
            "very_negative": [
                "I strongly oppose this policy.",
                "This policy is misguided.",
                "I fundamentally disagree with this.",
                "This would cause more harm than good.",
                "I'm against this policy entirely.",
            ],
        },
    },

    # ========== MISINFORMATION ==========

    "misinformation": {
        "explanation": {
            "very_positive": [
                "I actively check facts and sources.",
                "I'm skeptical of unverified claims.",
                "I value accurate information.",
                "I correct misinformation when I see it.",
                "Truth and accuracy are important to me.",
            ],
            "positive": [
                "I try to verify information.",
                "I'm somewhat cautious about claims.",
                "I consider the source of information.",
                "I check facts when possible.",
                "I value reliable sources.",
            ],
            "neutral": [
                "I don't always verify information.",
                "Sometimes I take things at face value.",
                "I'm moderately concerned about accuracy.",
                "I try to be careful but don't always check.",
                "I have mixed practices around fact-checking.",
            ],
            "negative": [
                "I don't always fact-check.",
                "I might share before verifying.",
                "Checking sources is sometimes too much effort.",
                "I rely on my intuition about truth.",
                "I'm less rigorous than I could be.",
            ],
            "very_negative": [
                "I trust my gut about what's true.",
                "Mainstream fact-checkers are biased.",
                "I believe what makes sense to me.",
                "I don't trust official sources.",
                "I rely on alternative information.",
            ],
        },
    },

    # ========== BRAND LOYALTY ==========

    "brand_loyalty": {
        "explanation": {
            "very_positive": [
                "I'm very loyal to brands I trust.",
                "I stick with brands that have served me well.",
                "Brand loyalty is important to me.",
                "I prefer to stay with familiar brands.",
                "I have strong brand preferences.",
            ],
            "positive": [
                "I tend to be loyal to good brands.",
                "I prefer brands I know over new ones.",
                "I'm reasonably brand loyal.",
                "I give preference to brands I trust.",
                "Brand familiarity matters to me.",
            ],
            "neutral": [
                "I'm not particularly brand loyal.",
                "I consider different brands each time.",
                "Brand matters somewhat but not everything.",
                "I'm flexible about brands.",
                "I don't have strong brand loyalties.",
            ],
            "negative": [
                "I switch brands easily.",
                "I look for the best deal regardless of brand.",
                "Brand loyalty seems overrated.",
                "I'm always open to trying new brands.",
                "I don't let brand affect my choices much.",
            ],
            "very_negative": [
                "I actively avoid brand loyalty.",
                "I always compare regardless of brand.",
                "Brands are just marketing to me.",
                "I see no reason for brand loyalty.",
                "I'm deliberately brand-agnostic.",
            ],
        },
    },

    # ========== SERVICE QUALITY ==========

    "service_quality": {
        "explanation": {
            "very_positive": [
                "The service was exceptional.",
                "I was very impressed with the service quality.",
                "Outstanding service experience.",
                "The service exceeded my expectations.",
                "Best service I've experienced.",
            ],
            "positive": [
                "Good service overall.",
                "I was satisfied with the service.",
                "The service met my expectations.",
                "Decent service quality.",
                "I had a positive service experience.",
            ],
            "neutral": [
                "The service was average.",
                "Nothing special about the service.",
                "Standard service experience.",
                "The service was okay.",
                "Neither good nor bad service.",
            ],
            "negative": [
                "The service was disappointing.",
                "I wasn't satisfied with the service.",
                "Service quality was below expectations.",
                "The service could be improved.",
                "I had issues with the service.",
            ],
            "very_negative": [
                "The service was terrible.",
                "Very poor service quality.",
                "I'm very dissatisfied with the service.",
                "Worst service I've experienced.",
                "The service was unacceptable.",
            ],
        },
    },

    # ========== ALGORITHM AVERSION ==========

    "algorithm_aversion": {
        "explanation": {
            "very_positive": [
                "I trust algorithmic recommendations.",
                "Algorithms often know better than humans.",
                "I embrace AI-driven decisions.",
                "Algorithms are more objective than humans.",
                "I prefer data-driven recommendations.",
            ],
            "positive": [
                "Algorithms can be helpful.",
                "I'm open to algorithmic suggestions.",
                "Algorithms have their place.",
                "I consider algorithmic recommendations.",
                "AI can assist human decision-making.",
            ],
            "neutral": [
                "I'm neutral about algorithms.",
                "Both humans and algorithms have merits.",
                "It depends on the context.",
                "I don't have strong feelings either way.",
                "Algorithms are just one input.",
            ],
            "negative": [
                "I prefer human judgment.",
                "Algorithms lack context and nuance.",
                "I'm skeptical of algorithmic decisions.",
                "Human expertise is often better.",
                "I don't fully trust algorithms.",
            ],
            "very_negative": [
                "I distrust algorithmic recommendations.",
                "Algorithms can't replace human judgment.",
                "I strongly prefer human decision-making.",
                "Algorithms are often wrong.",
                "I avoid algorithm-based decisions.",
            ],
        },
    },

    # ========== SOCIAL MEDIA ==========

    "social_media": {
        "explanation": {
            "very_positive": [
                "Social media enriches my life.",
                "I love staying connected online.",
                "Social media provides great value.",
                "I'm very active on social media.",
                "Social platforms are important to me.",
            ],
            "positive": [
                "I find social media useful.",
                "Social media helps me stay connected.",
                "I enjoy social media moderately.",
                "It's a good way to stay informed.",
                "Social media has benefits.",
            ],
            "neutral": [
                "I have mixed feelings about social media.",
                "Social media has pros and cons.",
                "I use it but don't love it.",
                "I'm indifferent about social media.",
                "It's just part of life now.",
            ],
            "negative": [
                "Social media is often negative.",
                "I try to limit my social media use.",
                "Social media can be harmful.",
                "I have concerns about social media.",
                "I'm reducing my social media time.",
            ],
            "very_negative": [
                "Social media is toxic.",
                "I strongly dislike social media.",
                "Social media damages society.",
                "I've quit or limited social media.",
                "Social platforms are harmful.",
            ],
        },
    },

    # ========== VACCINATION ==========

    "vaccination": {
        "explanation": {
            "very_positive": [
                "Vaccines are safe and effective.",
                "I strongly support vaccination.",
                "Vaccines are crucial for public health.",
                "I always get recommended vaccines.",
                "Vaccination is a responsibility.",
            ],
            "positive": [
                "I generally support vaccination.",
                "Vaccines provide important protection.",
                "I get most recommended vaccines.",
                "I believe in vaccine science.",
                "Vaccines are beneficial overall.",
            ],
            "neutral": [
                "I have mixed views on vaccination.",
                "I consider each vaccine individually.",
                "I'm not strongly pro or anti vaccine.",
                "I think about it case by case.",
                "I have some questions about vaccines.",
            ],
            "negative": [
                "I have concerns about some vaccines.",
                "I'm selective about vaccination.",
                "I question some vaccine recommendations.",
                "I prefer natural immunity sometimes.",
                "I'm cautious about new vaccines.",
            ],
            "very_negative": [
                "I'm very skeptical of vaccines.",
                "I avoid most vaccinations.",
                "Vaccines carry significant risks.",
                "I don't trust vaccine mandates.",
                "I oppose forced vaccination.",
            ],
        },
    },

    # ========== ONLINE LEARNING ==========

    "online_learning": {
        "explanation": {
            "very_positive": [
                "Online learning works great for me.",
                "I prefer online to in-person learning.",
                "Online education is very effective.",
                "I thrive in online learning environments.",
                "Digital learning is the future.",
            ],
            "positive": [
                "I find online learning helpful.",
                "Online courses can be effective.",
                "I appreciate the flexibility of online learning.",
                "Online learning has worked well for me.",
                "I'm satisfied with online education.",
            ],
            "neutral": [
                "Online learning has trade-offs.",
                "Both online and in-person have merits.",
                "I'm okay with online learning.",
                "It depends on the subject.",
                "I don't prefer one over the other.",
            ],
            "negative": [
                "I prefer in-person learning.",
                "Online learning is harder for me.",
                "I miss the classroom experience.",
                "Online learning has limitations.",
                "I struggle with online courses.",
            ],
            "very_negative": [
                "Online learning doesn't work for me.",
                "I strongly prefer in-person education.",
                "Online courses are inferior.",
                "I can't learn effectively online.",
                "Online education is inadequate.",
            ],
        },
    },

    # ========== MORAL DILEMMA ==========

    "moral_dilemma": {
        "explanation": {
            "very_positive": [
                "I made the choice that helped the most people.",
                "I prioritized the greater good.",
                "My decision maximized positive outcomes.",
                "I took the utilitarian approach.",
                "I focused on consequences rather than rules.",
            ],
            "positive": [
                "I tried to do what was best overall.",
                "I weighed competing values carefully.",
                "I considered all perspectives.",
                "I made a reasoned ethical choice.",
                "I aimed for the best outcome.",
            ],
            "neutral": [
                "This was a difficult dilemma.",
                "I'm not sure I made the right choice.",
                "There were valid arguments both ways.",
                "I struggled with this decision.",
                "Neither option felt completely right.",
            ],
            "negative": [
                "I followed my principles despite consequences.",
                "Some things are wrong regardless of outcomes.",
                "I prioritized rules over results.",
                "I couldn't violate my values.",
                "My conscience guided my choice.",
            ],
            "very_negative": [
                "I absolutely refused to cross moral lines.",
                "Some actions are simply wrong.",
                "I stood by my principles firmly.",
                "I couldn't live with the alternative.",
                "Rules exist for good reasons.",
            ],
        },
    },

    # ========== STRESS ==========

    "stress": {
        "explanation": {
            "very_positive": [
                "I handle stress very well.",
                "Stress doesn't affect me much.",
                "I'm very resilient under pressure.",
                "I actually perform better with some stress.",
                "I've developed strong coping skills.",
            ],
            "positive": [
                "I manage stress reasonably well.",
                "I can handle most stressful situations.",
                "I have good stress management skills.",
                "Stress is manageable for me.",
                "I cope with stress adequately.",
            ],
            "neutral": [
                "I deal with stress like most people.",
                "Sometimes stress affects me, sometimes not.",
                "My stress management is average.",
                "I have good days and bad days.",
                "I'm neither particularly stressed nor calm.",
            ],
            "negative": [
                "I struggle with stress sometimes.",
                "Stress affects me more than I'd like.",
                "I could improve my stress management.",
                "I feel stressed fairly often.",
                "Coping with stress is challenging.",
            ],
            "very_negative": [
                "I'm very affected by stress.",
                "Stress is a major problem for me.",
                "I have difficulty coping with pressure.",
                "I feel stressed most of the time.",
                "Stress significantly impacts my life.",
            ],
        },
    },

    # ========== NEGOTIATION ==========

    "negotiation": {
        "explanation": {
            "very_positive": [
                "I negotiated for a win-win outcome.",
                "I aimed for mutual benefit.",
                "Collaborative negotiation works best.",
                "I focused on creating value.",
                "Both parties should gain from negotiation.",
            ],
            "positive": [
                "I tried to find a fair compromise.",
                "I considered the other side's interests.",
                "I negotiated reasonably.",
                "I aimed for a balanced outcome.",
                "I was fair in my negotiation.",
            ],
            "neutral": [
                "I stuck to my position.",
                "I negotiated for my interests.",
                "I tried to get a good deal.",
                "I was neither aggressive nor passive.",
                "I took a moderate approach.",
            ],
            "negative": [
                "I pushed hard for my position.",
                "I prioritized my own interests.",
                "I was competitive in negotiation.",
                "I tried to get the best deal for myself.",
                "I negotiated firmly.",
            ],
            "very_negative": [
                "I aimed to maximize my gains.",
                "I was very competitive.",
                "I prioritized winning the negotiation.",
                "I wasn't concerned about fairness.",
                "I took an aggressive approach.",
            ],
        },
    },

    # ========== FINANCIAL DECISION ==========

    "financial_decision": {
        "explanation": {
            "very_positive": [
                "I made a careful, well-reasoned financial choice.",
                "I analyzed the options thoroughly.",
                "I feel confident about this financial decision.",
                "This was a sound financial choice.",
                "I considered all relevant factors.",
            ],
            "positive": [
                "I think this was a good financial decision.",
                "I weighed the costs and benefits.",
                "I made a reasonable financial choice.",
                "I'm satisfied with my decision.",
                "I considered the financial implications.",
            ],
            "neutral": [
                "I'm not sure if this was the best choice.",
                "Financial decisions are always uncertain.",
                "I made what seemed reasonable at the time.",
                "I have mixed feelings about this.",
                "Time will tell if it was the right choice.",
            ],
            "negative": [
                "I have some regrets about this decision.",
                "I might have chosen differently.",
                "This wasn't my best financial decision.",
                "I didn't fully consider all options.",
                "I'm somewhat uncertain about this choice.",
            ],
            "very_negative": [
                "I regret this financial decision.",
                "This was probably a mistake.",
                "I should have been more careful.",
                "I'm disappointed with my choice.",
                "I didn't make a good financial decision.",
            ],
        },
    },

    # ========== NEUROSCIENCE/NEUROECONOMICS ==========

    "neuroeconomics": {
        "explanation": {
            "very_positive": ["The decision felt intuitive and rewarding.", "My brain seemed to quickly process the best choice.", "The reward of making this choice felt immediate.", "I experienced a sense of clarity when deciding.", "The choice aligned well with my natural instincts."],
            "positive": ["The decision process felt relatively smooth.", "I was able to weigh options fairly quickly.", "The choice didn't require too much mental effort.", "My intuition guided me reasonably well.", "The decision felt comfortable."],
            "neutral": ["I'm not sure what drove this particular decision.", "The decision process was neither easy nor hard.", "I can't pinpoint exactly what influenced my choice.", "It was an average decision experience.", "No strong feelings about the process."],
            "negative": ["The decision felt mentally taxing.", "I struggled to process all the information.", "My thinking felt clouded or conflicted.", "The choice required more effort than expected.", "I felt uncertain throughout the process."],
            "very_negative": ["The decision was mentally exhausting.", "I felt overwhelmed by the options.", "My brain couldn't process everything effectively.", "The choice felt forced and uncomfortable.", "I had difficulty focusing on what mattered."],
        },
    },

    "impulse_control": {
        "explanation": {
            "very_positive": ["I exercised strong self-control.", "I resisted the temptation successfully.", "I thought carefully before acting.", "I prioritized long-term over immediate gratification.", "I controlled my impulses effectively."],
            "positive": ["I showed reasonable self-control.", "I managed to resist most temptations.", "I thought before acting.", "I considered the consequences.", "I maintained decent impulse control."],
            "neutral": ["Sometimes I controlled impulses, sometimes not.", "My self-control was average.", "I had mixed success with impulse control.", "It varied depending on the situation.", "No strong pattern in my self-control."],
            "negative": ["I struggled with self-control.", "Temptations were hard to resist.", "I acted impulsively at times.", "Short-term rewards influenced me too much.", "My impulse control was weaker than I'd like."],
            "very_negative": ["I gave in to temptation.", "I couldn't control my impulses.", "I acted without thinking.", "Immediate gratification won out.", "My self-control failed me."],
        },
    },

    "cognitive_load": {
        "explanation": {
            "very_positive": ["The task was easy to process mentally.", "I had plenty of mental capacity to handle this.", "The cognitive demands were manageable.", "My mind handled the complexity well.", "I felt mentally sharp and capable."],
            "positive": ["The mental demands were reasonable.", "I managed the cognitive load adequately.", "The task wasn't too mentally taxing.", "I had sufficient mental resources.", "The processing was fairly smooth."],
            "neutral": ["The mental demands were average.", "The cognitive load was neither high nor low.", "I had typical mental engagement.", "The processing was unremarkable.", "Standard mental effort was required."],
            "negative": ["The task was mentally demanding.", "I felt cognitive strain.", "My mental resources were stretched.", "The complexity was challenging.", "I struggled with the mental load."],
            "very_negative": ["I was cognitively overwhelmed.", "The mental demands were excessive.", "I couldn't process everything.", "My mind was overloaded.", "The cognitive burden was too high."],
        },
    },

    # ========== SPORTS PSYCHOLOGY ==========

    "sports_psychology": {
        "explanation": {
            "very_positive": ["Sports have been incredibly positive for me.", "Athletics helped me develop discipline and confidence.", "Competition brings out the best in me.", "Physical activity is central to my wellbeing.", "Sports taught me valuable life lessons."],
            "positive": ["I enjoy participating in sports.", "Athletics have been generally positive.", "Competition can be motivating.", "Physical activity benefits me.", "Sports have contributed to my growth."],
            "neutral": ["Sports are neither particularly important nor unimportant to me.", "I have average interest in athletics.", "Competition doesn't strongly affect me.", "Physical activity is just routine.", "My relationship with sports is neutral."],
            "negative": ["Sports haven't been particularly positive for me.", "Competition can be stressful.", "Athletics have caused some negative experiences.", "Physical activity can be challenging.", "Sports have had mixed effects on me."],
            "very_negative": ["Sports have been a source of stress.", "Competition negatively impacts me.", "Athletics have caused significant challenges.", "Physical activity has been discouraging.", "My sports experiences have been difficult."],
        },
    },

    "performance_anxiety": {
        "explanation": {
            "very_positive": ["I perform well under pressure.", "Anxiety doesn't affect my performance.", "I thrive in high-pressure situations.", "Competition energizes me.", "I channel nervous energy productively."],
            "positive": ["I manage performance pressure reasonably well.", "Some anxiety is motivating for me.", "I usually perform adequately under pressure.", "I've learned to cope with performance stress.", "Anxiety affects me minimally."],
            "neutral": ["Pressure affects me sometimes, sometimes not.", "My response to performance anxiety varies.", "I have average ability to handle pressure.", "Performance situations are hit or miss.", "Anxiety has mixed effects on my performance."],
            "negative": ["Performance pressure often affects me negatively.", "I struggle with anxiety in competitive situations.", "Pressure tends to hurt my performance.", "I get nervous before important events.", "Performance anxiety is challenging for me."],
            "very_negative": ["Performance anxiety significantly impairs me.", "I struggle greatly under pressure.", "Competition triggers intense anxiety.", "I often choke in important moments.", "Performance situations are very stressful."],
        },
    },

    "team_dynamics": {
        "explanation": {
            "very_positive": ["Our team works together excellently.", "Team chemistry is outstanding.", "Collaboration within the team is strong.", "We support each other effectively.", "Team dynamics enhance our performance."],
            "positive": ["Our team generally works well together.", "Team dynamics are positive.", "We collaborate effectively most of the time.", "Team members support each other.", "The group functions reasonably well."],
            "neutral": ["Team dynamics are average.", "We work together adequately.", "Collaboration is neither strong nor weak.", "Team functioning is typical.", "Our group dynamics are unremarkable."],
            "negative": ["Team dynamics could be improved.", "There are some conflicts within the team.", "Collaboration is sometimes difficult.", "Team chemistry has issues.", "Group functioning has challenges."],
            "very_negative": ["Team dynamics are problematic.", "There's significant conflict in the group.", "Collaboration is very difficult.", "Team chemistry is poor.", "The team functions poorly."],
        },
    },

    # ========== LEGAL PSYCHOLOGY ==========

    "legal_psychology": {
        "explanation": {
            "very_positive": ["The legal system worked fairly in this case.", "Justice was served appropriately.", "I trust the legal process.", "The procedures were handled correctly.", "The outcome was just and fair."],
            "positive": ["The legal system generally worked well.", "Justice was mostly served.", "I have reasonable trust in the process.", "Procedures were adequate.", "The outcome was acceptable."],
            "neutral": ["I have mixed views on the legal process.", "The system worked as expected.", "My trust in the legal process is average.", "Procedures were standard.", "The outcome was neither good nor bad."],
            "negative": ["The legal process had some problems.", "Justice wasn't fully served.", "I have concerns about the process.", "Procedures could have been better.", "The outcome was somewhat unfair."],
            "very_negative": ["The legal system failed.", "Justice was not served.", "I distrust the legal process.", "Procedures were mishandled.", "The outcome was unjust."],
        },
    },

    "jury_decision": {
        "explanation": {
            "very_positive": ["The evidence clearly supported this verdict.", "I'm confident in my decision.", "The facts led me to this conclusion.", "I carefully considered all testimony.", "This verdict reflects justice."],
            "positive": ["The evidence pointed toward this verdict.", "I'm reasonably confident in my decision.", "I considered the facts carefully.", "The testimony supported this conclusion.", "I believe this is the right verdict."],
            "neutral": ["The evidence was mixed.", "I had difficulty reaching this decision.", "The facts weren't entirely clear.", "I'm uncertain about this verdict.", "It was a close call."],
            "negative": ["I had doubts about this verdict.", "The evidence was insufficient.", "I struggled with this decision.", "The facts didn't clearly point one way.", "I'm not fully confident in this choice."],
            "very_negative": ["I disagree with this verdict.", "The evidence didn't support this conclusion.", "This decision was very difficult.", "I believe this verdict may be wrong.", "I have serious doubts about this outcome."],
        },
    },

    "procedural_justice": {
        "explanation": {
            "very_positive": ["The process was completely fair.", "I was treated with respect.", "My voice was heard.", "The procedures were transparent.", "I trust the fairness of the process."],
            "positive": ["The process was generally fair.", "I was treated reasonably well.", "I had opportunity to be heard.", "Procedures were mostly transparent.", "The process seemed fair."],
            "neutral": ["The process was average in fairness.", "Treatment was neither good nor bad.", "I somewhat had my voice heard.", "Procedures were partially transparent.", "Fairness was mixed."],
            "negative": ["The process had fairness issues.", "I wasn't treated very well.", "My voice wasn't fully heard.", "Procedures lacked transparency.", "The process seemed somewhat unfair."],
            "very_negative": ["The process was unfair.", "I was treated poorly.", "My voice was ignored.", "Procedures were opaque.", "The process was unjust."],
        },
    },

    # ========== FOOD/NUTRITION ==========

    "food_psychology": {
        "explanation": {
            "very_positive": ["I have a healthy relationship with food.", "Eating brings me joy without guilt.", "I make nutritious choices naturally.", "Food is a positive part of my life.", "I enjoy eating mindfully."],
            "positive": ["My relationship with food is generally good.", "I usually make healthy choices.", "Eating is mostly enjoyable.", "I have reasonable eating habits.", "Food is generally positive for me."],
            "neutral": ["My relationship with food is average.", "Eating is neither particularly enjoyable nor stressful.", "My food choices are mixed.", "I don't think much about food.", "My eating habits are typical."],
            "negative": ["I sometimes struggle with food choices.", "Eating can be stressful.", "My relationship with food has challenges.", "I make unhealthy choices sometimes.", "Food can be a source of stress."],
            "very_negative": ["I have a difficult relationship with food.", "Eating causes significant stress.", "I struggle with food choices.", "My eating habits are problematic.", "Food is a major challenge for me."],
        },
    },

    "eating_behavior": {
        "explanation": {
            "very_positive": ["I eat mindfully and healthily.", "My eating patterns are well-regulated.", "I have excellent dietary habits.", "I make thoughtful food choices.", "I eat appropriate portions."],
            "positive": ["I generally eat well.", "My eating patterns are reasonable.", "I have decent dietary habits.", "I usually make good food choices.", "My portions are appropriate."],
            "neutral": ["My eating behavior is average.", "I eat without particular thought.", "My dietary patterns are typical.", "Food choices are mixed.", "My eating habits are unremarkable."],
            "negative": ["My eating behavior could improve.", "I sometimes overeat or undereat.", "My dietary patterns are irregular.", "I make poor food choices at times.", "Eating habits are challenging."],
            "very_negative": ["My eating behavior is problematic.", "I struggle with eating regulation.", "My dietary patterns are unhealthy.", "I consistently make poor choices.", "Eating is very challenging for me."],
        },
    },

    "body_image": {
        "explanation": {
            "very_positive": ["I'm very satisfied with my body.", "I have a positive body image.", "I feel comfortable in my appearance.", "I appreciate my body.", "I feel confident about how I look."],
            "positive": ["I'm generally satisfied with my body.", "My body image is mostly positive.", "I feel reasonably comfortable with my appearance.", "I accept my body.", "I feel decent about how I look."],
            "neutral": ["I have mixed feelings about my body.", "My body image is neither positive nor negative.", "I don't think much about my appearance.", "I have average body satisfaction.", "My feelings about my body are neutral."],
            "negative": ["I'm somewhat dissatisfied with my body.", "My body image has challenges.", "I feel uncomfortable with some aspects of my appearance.", "Body satisfaction is difficult.", "I'm critical of how I look."],
            "very_negative": ["I'm very dissatisfied with my body.", "I have a negative body image.", "I feel very uncomfortable with my appearance.", "I dislike my body.", "Body image causes significant distress."],
        },
    },

    # ========== HUMAN FACTORS ==========

    "user_experience": {
        "explanation": {
            "very_positive": ["The interface was intuitive and easy to use.", "I had an excellent user experience.", "Everything worked exactly as expected.", "The design made my task effortless.", "I would highly recommend this experience."],
            "positive": ["The interface was reasonably easy to use.", "The user experience was good.", "Most things worked as expected.", "The design was helpful.", "I had a positive experience overall."],
            "neutral": ["The interface was average.", "The user experience was neither good nor bad.", "Some things worked, some didn't.", "The design was adequate.", "My experience was unremarkable."],
            "negative": ["The interface was somewhat confusing.", "The user experience could be improved.", "Several things didn't work as expected.", "The design hindered my task.", "I had a frustrating experience."],
            "very_negative": ["The interface was very confusing.", "The user experience was poor.", "Nothing worked as expected.", "The design made my task impossible.", "I would not recommend this."],
        },
    },

    "human_error": {
        "explanation": {
            "very_positive": ["I performed without any errors.", "The system helped prevent mistakes.", "I was very careful and accurate.", "Error prevention was excellent.", "I made no significant mistakes."],
            "positive": ["I made few errors.", "The system helped catch some mistakes.", "I was reasonably careful.", "Error prevention was adequate.", "Minor mistakes only."],
            "neutral": ["I made an average number of errors.", "Some mistakes were caught, some weren't.", "I was somewhat careful.", "Error prevention was partial.", "Typical error rate."],
            "negative": ["I made several errors.", "The system didn't prevent mistakes well.", "I wasn't as careful as I should have been.", "Error prevention was inadequate.", "I made more mistakes than usual."],
            "very_negative": ["I made many errors.", "The system allowed too many mistakes.", "I wasn't careful enough.", "Error prevention failed.", "I made significant mistakes."],
        },
    },

    "safety_behavior": {
        "explanation": {
            "very_positive": ["I always follow safety protocols.", "Safety is my top priority.", "I'm very conscious of risks.", "I take all precautions.", "I've never had a safety incident."],
            "positive": ["I usually follow safety protocols.", "Safety is important to me.", "I'm aware of most risks.", "I take reasonable precautions.", "I rarely have safety issues."],
            "neutral": ["I follow safety protocols sometimes.", "Safety is moderately important to me.", "I'm somewhat aware of risks.", "I take some precautions.", "Average safety record."],
            "negative": ["I don't always follow safety protocols.", "Safety isn't always my priority.", "I sometimes overlook risks.", "I don't take all precautions.", "I've had some safety issues."],
            "very_negative": ["I rarely follow safety protocols.", "Safety isn't a priority.", "I often overlook risks.", "I take few precautions.", "I've had significant safety issues."],
        },
    },

    # ========== CROSS-CULTURAL ==========

    "cross_cultural": {
        "explanation": {
            "very_positive": ["I embrace cultural diversity.", "Cross-cultural experiences enrich my life.", "I adapt well to different cultures.", "I value learning from other cultures.", "Cultural differences fascinate me."],
            "positive": ["I appreciate cultural diversity.", "Cross-cultural experiences are valuable.", "I adapt reasonably well to different cultures.", "I'm interested in other cultures.", "Cultural differences are interesting."],
            "neutral": ["I have average interest in cultural diversity.", "Cross-cultural experiences are neither important nor unimportant.", "I adapt somewhat to different cultures.", "Culture is just one factor.", "I'm neutral about cultural differences."],
            "negative": ["Cultural differences can be challenging.", "Cross-cultural experiences are sometimes difficult.", "I struggle to adapt to different cultures.", "I prefer my own culture.", "Cultural differences are confusing."],
            "very_negative": ["Cultural differences are very challenging.", "Cross-cultural experiences are stressful.", "I can't adapt to different cultures.", "I strongly prefer my own culture.", "Cultural differences are problematic."],
        },
    },

    "cultural_identity": {
        "explanation": {
            "very_positive": ["I have a strong cultural identity.", "My heritage is very important to me.", "I'm proud of my cultural background.", "My culture shapes who I am.", "I embrace my cultural roots."],
            "positive": ["I have a good sense of cultural identity.", "My heritage is important to me.", "I'm mostly proud of my background.", "My culture influences me.", "I value my cultural roots."],
            "neutral": ["I have average awareness of cultural identity.", "My heritage is somewhat important.", "I have mixed feelings about my background.", "Culture has some influence on me.", "I'm neutral about cultural roots."],
            "negative": ["I struggle with cultural identity.", "My heritage feels less relevant.", "I have conflicts about my background.", "Culture doesn't define me much.", "Cultural roots aren't very important."],
            "very_negative": ["I have a weak cultural identity.", "My heritage isn't important to me.", "I reject my cultural background.", "Culture has no influence on me.", "I've disconnected from cultural roots."],
        },
    },

    # ========== POSITIVE PSYCHOLOGY ==========

    "gratitude": {
        "explanation": {
            "very_positive": ["I feel deeply grateful.", "I appreciate so many things in my life.", "Gratitude is central to my wellbeing.", "I regularly count my blessings.", "I'm thankful for what I have."],
            "positive": ["I feel grateful.", "I appreciate good things in my life.", "Gratitude is important to me.", "I often feel thankful.", "I recognize my blessings."],
            "neutral": ["I feel average gratitude.", "I sometimes appreciate things.", "Gratitude isn't particularly prominent.", "I occasionally feel thankful.", "I'm somewhat aware of blessings."],
            "negative": ["I struggle to feel grateful.", "I don't appreciate things as much as I should.", "Gratitude is difficult for me.", "I rarely feel thankful.", "I focus more on what's missing."],
            "very_negative": ["I feel very little gratitude.", "I don't appreciate what I have.", "Gratitude is foreign to me.", "I never feel thankful.", "I only see what's lacking."],
        },
    },

    "resilience": {
        "explanation": {
            "very_positive": ["I bounce back quickly from setbacks.", "Adversity makes me stronger.", "I'm highly resilient.", "I overcome challenges effectively.", "Setbacks don't keep me down."],
            "positive": ["I recover reasonably well from setbacks.", "I can handle adversity.", "I'm fairly resilient.", "I cope with challenges.", "I manage to overcome difficulties."],
            "neutral": ["My resilience is average.", "I sometimes recover, sometimes struggle.", "Adversity has mixed effects on me.", "I have typical coping ability.", "Setbacks affect me moderately."],
            "negative": ["I struggle to bounce back from setbacks.", "Adversity is difficult for me.", "My resilience could be better.", "I have trouble coping.", "Setbacks affect me significantly."],
            "very_negative": ["I have great difficulty recovering from setbacks.", "Adversity overwhelms me.", "I have low resilience.", "I can't cope with challenges.", "Setbacks devastate me."],
        },
    },

    "life_satisfaction": {
        "explanation": {
            "very_positive": ["I'm extremely satisfied with my life.", "Life is wonderful.", "I have everything I need.", "I couldn't ask for more.", "I'm very happy with how things are."],
            "positive": ["I'm satisfied with my life.", "Life is good.", "I have what I need.", "Things are going well.", "I'm happy with my situation."],
            "neutral": ["I'm neither satisfied nor dissatisfied.", "Life is okay.", "I have some of what I need.", "Things are average.", "I have mixed feelings about my life."],
            "negative": ["I'm somewhat dissatisfied with my life.", "Life could be better.", "I'm missing things I need.", "Things aren't going well.", "I'm unhappy with some aspects of my life."],
            "very_negative": ["I'm very dissatisfied with my life.", "Life is difficult.", "I don't have what I need.", "Things are going poorly.", "I'm unhappy with my life."],
        },
    },

    # ========== GENDER/SEXUALITY ==========

    "gender_roles": {
        "explanation": {
            "very_positive": ["I embrace flexible gender roles.", "Traditional roles don't define me.", "I believe in gender equality.", "People should choose their own roles.", "Gender shouldn't limit anyone."],
            "positive": ["I support flexible gender roles.", "Traditional roles aren't essential.", "I lean toward gender equality.", "Roles should be chosen individually.", "Gender shouldn't be limiting."],
            "neutral": ["I have mixed views on gender roles.", "Some traditional roles make sense.", "I'm moderate on gender equality.", "Context matters for roles.", "Gender role views vary."],
            "negative": ["I lean toward traditional gender roles.", "Some distinctions are natural.", "Complete equality isn't realistic.", "Biology influences roles.", "Traditional patterns have value."],
            "very_negative": ["I believe in traditional gender roles.", "Men and women have different roles.", "Traditional distinctions are important.", "Biology determines roles.", "Traditional patterns should continue."],
        },
    },

    # ========== RELATIONSHIPS ==========

    "attachment": {
        "explanation": {
            "very_positive": ["I feel securely attached in relationships.", "I'm comfortable with intimacy.", "I trust my partners completely.", "Close relationships are natural for me.", "I have healthy attachment patterns."],
            "positive": ["I have mostly secure attachments.", "I'm fairly comfortable with intimacy.", "I generally trust my partners.", "Close relationships are important.", "My attachment style is healthy."],
            "neutral": ["My attachment style varies.", "Intimacy is sometimes comfortable, sometimes not.", "I have mixed trust levels.", "Relationships have ups and downs.", "My attachment patterns are typical."],
            "negative": ["I sometimes struggle with attachment.", "Intimacy can be uncomfortable.", "I have trust issues at times.", "Close relationships are challenging.", "My attachment style has challenges."],
            "very_negative": ["I have significant attachment difficulties.", "Intimacy is very uncomfortable.", "I struggle greatly with trust.", "Close relationships are very hard.", "My attachment patterns are problematic."],
        },
    },

    "romantic_relationships": {
        "explanation": {
            "very_positive": ["My romantic relationship is wonderful.", "I'm deeply in love and satisfied.", "We communicate excellently.", "Our relationship is strong.", "I feel completely fulfilled romantically."],
            "positive": ["My romantic relationship is good.", "I'm happy with my partner.", "We communicate well.", "Our relationship is healthy.", "I'm satisfied romantically."],
            "neutral": ["My romantic relationship is average.", "We have our ups and downs.", "Communication is mixed.", "The relationship is okay.", "I have mixed romantic satisfaction."],
            "negative": ["My romantic relationship has challenges.", "There are issues with my partner.", "Communication could be better.", "The relationship is struggling.", "I'm not fully satisfied romantically."],
            "very_negative": ["My romantic relationship is very difficult.", "There are serious problems.", "Communication has broken down.", "The relationship is failing.", "I'm very unsatisfied romantically."],
        },
    },

    "social_support": {
        "explanation": {
            "very_positive": ["I have excellent social support.", "Many people care about me.", "I can always count on others.", "My support network is strong.", "I never feel alone."],
            "positive": ["I have good social support.", "Several people care about me.", "I usually have support when needed.", "My support network is adequate.", "I rarely feel alone."],
            "neutral": ["My social support is average.", "Some people care about me.", "I sometimes have support.", "My support network is limited.", "I occasionally feel alone."],
            "negative": ["My social support is limited.", "Few people seem to care.", "I often lack support.", "My support network is weak.", "I frequently feel alone."],
            "very_negative": ["I have no social support.", "Nobody cares about me.", "I have no one to turn to.", "My support network is nonexistent.", "I always feel alone."],
        },
    },

    # ========== NEW DOMAINS (v2.4.5) ==========

    # AI ALIGNMENT & ETHICS
    "ai_alignment": {
        "explanation": {
            "very_positive": ["AI alignment research is crucial for humanity's future.", "We must ensure AI systems share human values.", "I strongly support efforts to make AI beneficial.", "AI safety should be a top priority.", "Aligned AI can help solve major global challenges."],
            "positive": ["AI alignment is important for safe development.", "I support research into AI value alignment.", "Making AI systems beneficial seems worthwhile.", "AI safety research is valuable.", "We should work on aligning AI with human values."],
            "neutral": ["AI alignment is one of many concerns.", "I'm not sure how serious the alignment problem is.", "It's hard to know what AI values should be.", "I have mixed views on AI alignment priorities.", "The importance of alignment research is unclear."],
            "negative": ["AI alignment concerns may be overstated.", "Other AI issues seem more pressing.", "I'm skeptical about alignment as a problem.", "Current AI doesn't need complex alignment.", "Alignment research seems premature."],
            "very_negative": ["AI alignment fears are exaggerated.", "We shouldn't slow AI progress for alignment.", "The alignment problem is overhyped.", "I don't believe AI poses existential risks.", "Alignment concerns distract from real issues."],
        },
    },

    "ai_ethics": {
        "explanation": {
            "very_positive": ["AI ethics are essential for responsible development.", "We must consider ethical implications of AI.", "I strongly support ethical AI guidelines.", "AI developers have moral obligations.", "Ethics should guide all AI decisions."],
            "positive": ["AI ethics matter for good outcomes.", "Ethical considerations in AI are important.", "I support ethical AI development.", "AI should be developed responsibly.", "Ethics should inform AI decisions."],
            "neutral": ["AI ethics is one consideration among many.", "It's hard to apply ethics to AI systems.", "I have mixed views on AI ethics.", "The right ethical framework is unclear.", "AI ethics is complicated."],
            "negative": ["AI ethics can slow beneficial progress.", "Ethical concerns about AI are often overblown.", "Market forces will guide AI better than ethics.", "Too much focus on AI ethics.", "Ethics shouldn't restrict AI innovation."],
            "very_negative": ["AI ethics is mostly hand-wringing.", "Ethical restrictions on AI harm progress.", "I reject most AI ethics concerns.", "AI ethics is a distraction.", "Innovation matters more than AI ethics."],
        },
    },

    # CLIMATE SCIENCE & ACTION
    "climate_action": {
        "explanation": {
            "very_positive": ["Urgent climate action is absolutely essential.", "We must act immediately on climate change.", "Climate action is the defining issue of our time.", "I strongly support aggressive climate policies.", "We need transformative climate action now."],
            "positive": ["Climate action is important and needed.", "I support policies to address climate change.", "We should do more for the climate.", "Climate action makes sense.", "I care about climate action."],
            "neutral": ["Climate action has costs and benefits.", "I have mixed views on climate policies.", "Some climate action is needed, but not sure how much.", "Climate policy trade-offs are complex.", "I'm moderate on climate action."],
            "negative": ["Climate action costs are too high.", "Current policies go too far.", "Economic concerns should balance climate action.", "I'm skeptical of aggressive climate policies.", "Climate action is often misguided."],
            "very_negative": ["Climate action hurts the economy unnecessarily.", "Climate policies are government overreach.", "I oppose most climate action proposals.", "Climate activism is excessive.", "Climate action is a waste of resources."],
        },
    },

    "climate_justice": {
        "explanation": {
            "very_positive": ["Climate justice is essential for equity.", "Those least responsible suffer most from climate change.", "I strongly support climate justice principles.", "Wealthy nations owe climate debt to developing countries.", "Environmental racism must be addressed."],
            "positive": ["Climate justice concerns are valid.", "Equity should be part of climate policy.", "I support fair climate burden-sharing.", "Climate impacts are unevenly distributed.", "Justice should inform climate action."],
            "neutral": ["Climate justice is complicated.", "I have mixed views on climate equity.", "It's hard to know what's fair.", "Climate justice claims vary in validity.", "I'm uncertain about climate justice."],
            "negative": ["Climate justice can complicate needed action.", "Individual responsibility matters more.", "Climate justice claims are sometimes excessive.", "Other priorities compete with climate justice.", "Climate justice is too politicized."],
            "very_negative": ["Climate justice is a political agenda.", "I reject the climate justice framework.", "Personal responsibility, not collective guilt.", "Climate justice distracts from solutions.", "I don't accept climate justice claims."],
        },
    },

    # HEALTH DISPARITIES
    "health_disparities": {
        "explanation": {
            "very_positive": ["Health disparities are a critical injustice.", "We must address systemic health inequities.", "I strongly support eliminating health disparities.", "Everyone deserves equal health outcomes.", "Health equity should be a national priority."],
            "positive": ["Health disparities are concerning.", "We should work to reduce health inequities.", "I support policies addressing health disparities.", "Equal health access is important.", "Reducing health disparities matters."],
            "neutral": ["Health disparities exist but causes are complex.", "I have mixed views on health equity policies.", "Individual factors also affect health.", "Addressing disparities is complicated.", "I'm moderate on health disparity solutions."],
            "negative": ["Health disparities have multiple causes beyond policy.", "Individual choices affect health significantly.", "Not all disparities are inequities.", "Current approaches may not work.", "Health disparity emphasis is sometimes excessive."],
            "very_negative": ["Health disparities are mostly due to individual choices.", "I reject systemic explanations for health differences.", "Personal responsibility is what matters.", "Health equity policies are misguided.", "Health disparity focus is political."],
        },
    },

    "healthcare_access": {
        "explanation": {
            "very_positive": ["Universal healthcare access is a right.", "Everyone deserves affordable quality care.", "I strongly support expanded healthcare access.", "Healthcare should not depend on wealth.", "Access barriers must be eliminated."],
            "positive": ["Healthcare access is important.", "I support improving access to care.", "More people should have healthcare.", "Access to care should be expanded.", "Healthcare should be more affordable."],
            "neutral": ["Healthcare access trade-offs are real.", "I have mixed views on healthcare policy.", "Expanding access has costs.", "The best approach is unclear.", "I'm moderate on healthcare access."],
            "negative": ["Current healthcare access is adequate.", "Market solutions are better than mandates.", "Expanding access is too expensive.", "Quality matters more than access.", "Government healthcare expansion is problematic."],
            "very_negative": ["Healthcare is not a right.", "I oppose government healthcare expansion.", "Personal responsibility for healthcare.", "Current access is sufficient.", "Healthcare mandates are wrong."],
        },
    },

    # GENOMICS & PERSONALIZED MEDICINE
    "genomics": {
        "explanation": {
            "very_positive": ["Genomics will revolutionize medicine.", "Genetic knowledge empowers health decisions.", "I'm very excited about genomic advances.", "Personalized medicine is the future.", "Genomics offers tremendous benefits."],
            "positive": ["Genomics has promising applications.", "Genetic information can be useful.", "I'm interested in genomic medicine.", "Personalized treatments seem valuable.", "Genomics research is worthwhile."],
            "neutral": ["Genomics has potential and risks.", "I have mixed views on genetic medicine.", "Benefits and concerns both seem valid.", "It's hard to know the right approach.", "I'm uncertain about genomics."],
            "negative": ["Genomics raises concerning issues.", "Genetic information has privacy risks.", "I'm skeptical of genomic medicine hype.", "Genetic determinism is concerning.", "Genomics risks outweigh benefits."],
            "very_negative": ["Genomics is dangerous territory.", "Genetic information shouldn't be collected.", "I strongly oppose genomic medicine trends.", "Privacy and discrimination risks are too high.", "We shouldn't pursue genomic medicine."],
        },
    },

    "genetic_testing": {
        "explanation": {
            "very_positive": ["Genetic testing provides valuable health information.", "I support widespread genetic testing.", "Knowing my genetic risks empowers me.", "Genetic testing should be accessible to all.", "Testing enables better health decisions."],
            "positive": ["Genetic testing can be useful.", "I'm open to genetic testing for myself.", "Testing has legitimate uses.", "Some genetic tests are worthwhile.", "Genetic information can help health decisions."],
            "neutral": ["Genetic testing has pros and cons.", "I'm undecided about genetic testing.", "It depends on the specific test.", "Testing isn't clearly good or bad.", "I have mixed views on genetic testing."],
            "negative": ["Genetic testing raises privacy concerns.", "I'm cautious about genetic testing.", "Testing can cause unnecessary worry.", "Insurance and employment discrimination risks.", "Genetic testing is often not useful."],
            "very_negative": ["Genetic testing is an invasion of privacy.", "I oppose genetic testing.", "Testing creates more problems than it solves.", "Genetic information should stay private.", "I refuse genetic testing."],
        },
    },

    # DIGITAL SOCIETY
    "algorithmic_fairness": {
        "explanation": {
            "very_positive": ["Algorithmic fairness is essential for justice.", "AI systems must be unbiased.", "I strongly support algorithmic auditing.", "Biased algorithms cause real harm.", "Fairness must be designed into systems."],
            "positive": ["Algorithmic fairness matters.", "AI systems should be checked for bias.", "I support fairness in algorithms.", "Biased algorithms are concerning.", "We should work on fair AI."],
            "neutral": ["Algorithmic fairness is complicated.", "Defining fairness is difficult.", "I have mixed views on this.", "Trade-offs between fairness criteria exist.", "It's hard to know the right approach."],
            "negative": ["Algorithmic fairness can harm accuracy.", "Fairness mandates may be counterproductive.", "I'm skeptical of fairness requirements.", "Too much focus on algorithmic bias.", "Fairness concerns are sometimes overblown."],
            "very_negative": ["Algorithmic fairness is a misguided concept.", "I oppose fairness mandates for AI.", "Accuracy matters more than fairness.", "Fairness requirements harm innovation.", "I reject the algorithmic fairness agenda."],
        },
    },

    "automation_anxiety": {
        "explanation": {
            "very_positive": ["Automation will create new opportunities.", "I'm optimistic about the automated future.", "Technology always creates new jobs.", "Automation will improve quality of life.", "I embrace automation fully."],
            "positive": ["Automation has more benefits than risks.", "New jobs will replace old ones.", "I'm generally positive about automation.", "Technology usually helps workers.", "Automation can be managed well."],
            "neutral": ["Automation has both upsides and downsides.", "Job impacts are hard to predict.", "I have mixed feelings about automation.", "Some jobs will be lost, some created.", "The future is uncertain."],
            "negative": ["I'm concerned about job displacement.", "Automation threatens many livelihoods.", "Workers aren't prepared for automation.", "Transition will be difficult.", "Automation anxiety is justified."],
            "very_negative": ["Automation will devastate the job market.", "Mass unemployment is coming.", "Technology is replacing humans unfairly.", "Workers will suffer greatly.", "I'm very worried about automation."],
        },
    },

    # ========== ADDITIONAL SOCIAL SCIENCE DOMAINS (v1.1.0) ==========

    # MORAL PSYCHOLOGY
    "moral_judgment": {
        "explanation": {
            "very_positive": ["This action is clearly morally right.", "The ethical choice here is obvious.", "I strongly support this moral stance.", "This aligns with my core values.", "Morally, this is the correct path."],
            "positive": ["This seems like the ethical choice.", "I think this is morally acceptable.", "This action appears justified.", "The moral reasoning here is sound.", "I can support this ethically."],
            "neutral": ["Morally, this is a gray area.", "I can see ethical arguments both ways.", "This is a difficult moral question.", "The ethics here are complex.", "I'm torn on the morality of this."],
            "negative": ["I have moral concerns about this.", "This doesn't seem entirely ethical.", "I'm uncomfortable with this morally.", "There are ethical issues here.", "This raises moral red flags."],
            "very_negative": ["This is morally wrong.", "I strongly oppose this on ethical grounds.", "This violates fundamental moral principles.", "This is clearly unethical.", "I cannot support this morally."],
        },
    },

    "ethical_dilemma": {
        "explanation": {
            "very_positive": ["Despite the dilemma, I'm confident in my choice.", "The right answer became clear to me.", "I feel good about my ethical reasoning.", "This dilemma has a defensible solution.", "My moral compass guided me well."],
            "positive": ["I made my best ethical judgment.", "I think my choice was reasonable.", "I weighed the considerations carefully.", "My decision seems morally sound.", "I'm relatively comfortable with my choice."],
            "neutral": ["This was a genuine dilemma with no easy answer.", "Both options had merit and problems.", "I struggled with this ethical question.", "Any choice here involves moral trade-offs.", "I'm not sure there's a right answer."],
            "negative": ["This dilemma made me uncomfortable.", "I'm not satisfied with my choice.", "The options were both problematic.", "I wish there was a better alternative.", "This ethical conflict was troubling."],
            "very_negative": ["This dilemma had no good solution.", "I hated having to choose.", "Both options felt morally wrong.", "This situation was deeply troubling.", "I'm distressed by this ethical bind."],
        },
    },

    # COGNITIVE BIASES
    "confirmation_bias": {
        "explanation": {
            "very_positive": ["I actively seek out diverse viewpoints.", "I challenge my own assumptions regularly.", "I'm very open to evidence that contradicts my views.", "I work hard to avoid confirmation bias.", "I embrace information that challenges me."],
            "positive": ["I try to consider different perspectives.", "I'm open to changing my mind.", "I seek out various sources.", "I recognize my biases exist.", "I make an effort to be balanced."],
            "neutral": ["I probably have some confirmation bias.", "I'm not sure how biased I am.", "Like everyone, I have preferences.", "I try but likely fall short.", "Avoiding bias is difficult."],
            "negative": ["I tend to favor information I agree with.", "I probably don't seek opposing views enough.", "My information sources are limited.", "I recognize this is a weakness.", "I could do better at being open-minded."],
            "very_negative": ["I admit I mostly seek confirming information.", "I tend to dismiss opposing views.", "My media diet is probably biased.", "I'm not very open to contrary evidence.", "I prefer information that supports my views."],
        },
    },

    # SOCIAL COMPARISON
    "social_comparison": {
        "explanation": {
            "very_positive": ["Comparing myself to others motivates me.", "I learn from people who are doing well.", "Social comparison helps me improve.", "I feel inspired by others' success.", "Comparison gives me useful benchmarks."],
            "positive": ["Some social comparison can be helpful.", "I occasionally compare and find it useful.", "Seeing others' progress can motivate.", "I try to compare constructively.", "Comparison sometimes provides perspective."],
            "neutral": ["Social comparison has mixed effects on me.", "Sometimes it helps, sometimes it hurts.", "I have mixed feelings about comparing.", "It depends on the situation.", "I'm neutral about social comparison."],
            "negative": ["Social comparison often makes me feel worse.", "I tend to compare unfavorably.", "Comparing myself to others is discouraging.", "I find it more harmful than helpful.", "I try to avoid social comparison."],
            "very_negative": ["Social comparison is very harmful to me.", "I always feel worse after comparing.", "Comparison triggers negative feelings.", "I strongly dislike social comparison.", "It's toxic for my wellbeing."],
        },
    },

    # SELF-REGULATION
    "self_control": {
        "explanation": {
            "very_positive": ["I have excellent self-control.", "I'm very disciplined in my choices.", "Resisting temptation is easy for me.", "I consistently make good decisions.", "Self-regulation is a strength of mine."],
            "positive": ["I have reasonably good self-control.", "I usually resist temptation.", "I'm fairly disciplined.", "I make good choices most of the time.", "My self-control is adequate."],
            "neutral": ["My self-control is average.", "Sometimes I'm disciplined, sometimes not.", "I have mixed results with self-control.", "It depends on the situation.", "I'm neither strong nor weak in this area."],
            "negative": ["I struggle with self-control.", "Resisting temptation is difficult.", "I often give in to impulses.", "My discipline could be better.", "Self-regulation is challenging for me."],
            "very_negative": ["I have very poor self-control.", "I almost always give in to temptation.", "Discipline is extremely hard for me.", "I frequently make impulsive choices.", "Self-control is a major weakness."],
        },
    },

    # WELLBEING & LIFE SATISFACTION
    "life_satisfaction": {
        "explanation": {
            "very_positive": ["I'm very satisfied with my life overall.", "Life is going extremely well.", "I feel fulfilled and content.", "I have few regrets.", "My life is close to ideal."],
            "positive": ["I'm generally satisfied with my life.", "Things are going pretty well.", "I'm content for the most part.", "Life is good overall.", "I have much to be grateful for."],
            "neutral": ["My life satisfaction is average.", "Some things are good, some aren't.", "I have mixed feelings about my life.", "It could be better, could be worse.", "I'm neither satisfied nor dissatisfied."],
            "negative": ["I'm somewhat dissatisfied with my life.", "Things could be going better.", "I have significant concerns.", "My life isn't what I hoped.", "I'm not very satisfied overall."],
            "very_negative": ["I'm very dissatisfied with my life.", "Things are not going well at all.", "I have many regrets.", "My life is far from ideal.", "I'm deeply unsatisfied."],
        },
    },

    # EMOTION REGULATION
    "emotion_regulation": {
        "explanation": {
            "very_positive": ["I manage my emotions very effectively.", "I have excellent emotional control.", "I rarely get overwhelmed by feelings.", "I can regulate my emotions well.", "Emotional stability is a strength."],
            "positive": ["I generally manage emotions well.", "I have decent emotional control.", "I can usually regulate my feelings.", "I cope with emotions reasonably.", "My emotional regulation is good."],
            "neutral": ["My emotion regulation is average.", "Sometimes I manage well, sometimes not.", "I have typical emotional control.", "My coping varies by situation.", "I'm neither great nor poor at this."],
            "negative": ["I struggle to regulate emotions.", "My feelings often overwhelm me.", "Emotional control is difficult.", "I have trouble managing feelings.", "My emotion regulation needs work."],
            "very_negative": ["I have very poor emotion regulation.", "I'm frequently overwhelmed by feelings.", "Emotional control is extremely hard.", "My emotions control me.", "I struggle greatly with this."],
        },
    },

    # INTERPERSONAL RELATIONSHIPS
    "relationship_quality": {
        "explanation": {
            "very_positive": ["My relationships are extremely fulfilling.", "I have wonderful connections with others.", "My relationships are very satisfying.", "I feel deeply connected to people.", "My interpersonal life is excellent."],
            "positive": ["I have good relationships overall.", "My connections with others are positive.", "I'm satisfied with my relationships.", "I have meaningful connections.", "My relationships are generally healthy."],
            "neutral": ["My relationships are average.", "Some relationships are good, some not.", "I have mixed relationship quality.", "My interpersonal life is okay.", "Relationships are neither great nor poor."],
            "negative": ["My relationships need improvement.", "I'm not satisfied with my connections.", "My relationships have problems.", "I feel somewhat disconnected.", "My interpersonal life is lacking."],
            "very_negative": ["My relationships are very poor.", "I feel isolated and disconnected.", "My interpersonal life is troubled.", "I have few meaningful connections.", "Relationships are a major struggle."],
        },
    },

    # SCIENTIFIC LITERACY
    "scientific_reasoning": {
        "explanation": {
            "very_positive": ["I strongly value scientific evidence.", "I trust the scientific method.", "I base my views on research.", "Science provides reliable knowledge.", "I'm very science-oriented."],
            "positive": ["I generally trust scientific findings.", "I value evidence-based approaches.", "Science is usually reliable.", "I try to follow the research.", "I'm fairly science-minded."],
            "neutral": ["I have mixed views on science.", "Science has strengths and limits.", "I'm neither pro nor anti-science.", "It depends on the topic.", "I'm neutral about scientific claims."],
            "negative": ["I'm skeptical of some scientific claims.", "Scientists can be biased too.", "I question some research findings.", "Science doesn't have all the answers.", "I'm cautious about scientific claims."],
            "very_negative": ["I'm very skeptical of science.", "Scientists often get things wrong.", "I distrust many scientific claims.", "Science is overrated.", "I don't rely on scientific evidence."],
        },
    },

    # GROUP DYNAMICS
    "group_identity": {
        "explanation": {
            "very_positive": ["I strongly identify with my group.", "Group membership is very important to me.", "I feel a deep connection to my group.", "My group is central to who I am.", "I'm proud of my group membership."],
            "positive": ["I identify with my group.", "Being part of this group matters to me.", "I feel connected to my group.", "Group membership is meaningful.", "I value my group affiliation."],
            "neutral": ["I have moderate group identification.", "My group membership is somewhat important.", "I feel some connection to my group.", "Group identity matters to some extent.", "I'm neutral about group membership."],
            "negative": ["I don't strongly identify with my group.", "Group membership isn't very important.", "I feel somewhat disconnected from my group.", "Group identity doesn't define me much.", "I'm ambivalent about my group."],
            "very_negative": ["I don't identify with my group at all.", "Group membership is unimportant to me.", "I feel no connection to my group.", "Group identity doesn't matter.", "I reject group-based thinking."],
        },
    },

    # BEHAVIORAL INTENTIONS
    "behavior_intention": {
        "explanation": {
            "very_positive": ["I definitely intend to do this.", "I'm fully committed to this action.", "I will certainly follow through.", "My intention is very strong.", "I'm determined to do this."],
            "positive": ["I intend to do this.", "I'll likely follow through.", "I plan to take this action.", "My intention is fairly strong.", "I'm inclined to do this."],
            "neutral": ["I might do this.", "I'm undecided about taking action.", "My intention is uncertain.", "I could go either way.", "I haven't decided yet."],
            "negative": ["I probably won't do this.", "I'm unlikely to follow through.", "My intention is weak.", "I'm leaning against this action.", "I doubt I'll do this."],
            "very_negative": ["I definitely won't do this.", "I have no intention of doing this.", "I refuse to take this action.", "My intention is firmly against.", "I'm certain I won't do this."],
        },
    },

    # COGNITIVE LOAD
    "mental_effort": {
        "explanation": {
            "very_positive": ["This required minimal mental effort.", "I found this very easy to think about.", "The cognitive load was very light.", "This was effortless for me.", "I didn't have to think hard at all."],
            "positive": ["This was fairly easy to process.", "I didn't find it too demanding.", "The mental effort was manageable.", "This wasn't too taxing.", "I could think about this easily."],
            "neutral": ["The mental effort was moderate.", "This required some thinking.", "The cognitive demand was average.", "It wasn't too easy or hard.", "The effort required was typical."],
            "negative": ["This required significant mental effort.", "I found it somewhat demanding.", "The cognitive load was high.", "This was mentally taxing.", "I had to think quite hard."],
            "very_negative": ["This was very mentally demanding.", "I found it extremely difficult to process.", "The cognitive load was overwhelming.", "This was exhausting to think about.", "The mental effort was excessive."],
        },
    },
}


# ============================================================================
# QUESTION TYPE DETECTION
# ============================================================================

QUESTION_TYPE_PATTERNS: Dict[QuestionType, List[str]] = {
    # ========== EXPLANATORY QUESTIONS ==========
    QuestionType.EXPLANATION: [
        r'\bexplain\b', r'\bwhy\b', r'\breason\b', r'\bbecause\b',
        r'\bwhat made you\b', r'\bwhat led you\b', r'\bdescribe.*reasoning\b',
        r'\bwalk.*through\b', r'\bhelp.*understand\b',
    ],
    QuestionType.JUSTIFICATION: [
        r'\bjustify\b', r'\bdefend\b', r'\bsupport.*decision\b',
        r'\bwhy\s+did\s+you\s+(?:choose|select|pick)\b', r'\breason\s+for\b',
    ],
    QuestionType.REASONING: [
        r'\breasoning\b', r'\bthought\s+process\b', r'\bhow.*decide\b',
        r'\bwhat\s+(?:was\s+)?your\s+thinking\b', r'\bprocess\b',
    ],
    QuestionType.CAUSATION: [
        r'\bwhat\s+caused\b', r'\bwhy\s+do\s+you\s+think\b', r'\bcontributed\b',
        r'\bled\s+to\b', r'\bresulted\s+in\b', r'\bfactor\b',
    ],
    QuestionType.MOTIVATION: [
        r'\bmotivat\b', r'\bdriv(?:e|ing|en)\b', r'\bwhy\s+did\s+you\s+want\b',
        r'\bwhat\s+prompted\b', r'\binspir\b', r'\bcompel\b',
    ],

    # ========== DESCRIPTIVE QUESTIONS ==========
    QuestionType.DESCRIPTION: [
        r'\bdescribe\b', r'\btell\s+us\b', r'\bshare\b', r'\bnarrate\b',
        r'\bwalk\s+us\s+through\b', r'\bpaint\s+a\s+picture\b',
    ],
    QuestionType.NARRATION: [
        r'\btell\s+(?:us\s+)?about\b', r'\bstory\b', r'\bexperience\b',
        r'\btime\s+when\b', r'\binstance\b', r'\bsituation\b',
    ],
    QuestionType.ELABORATION: [
        r'\belaborate\b', r'\bexpand\b', r'\bmore\s+detail\b',
        r'\bfurther\b', r'\bgo\s+into\b', r'\bexplain\s+further\b',
    ],
    QuestionType.DETAIL: [
        r'\bdetail\b', r'\bspecific\b', r'\bexample\b', r'\binstance\b',
        r'\billustrate\b', r'\bparticular\b',
    ],

    # ========== EVALUATIVE QUESTIONS ==========
    QuestionType.EVALUATION: [
        r'\bevaluate\b', r'\brate\b', r'\bassess\b', r'\bjudge\b',
        r'\bhow\s+(?:well|good|bad)\b', r'\bquality\b',
    ],
    QuestionType.ASSESSMENT: [
        r'\bassess\b', r'\banalyz\b', r'\bexamine\b', r'\breview\b',
        r'\bcritique\b', r'\bappraisal\b',
    ],
    QuestionType.COMPARISON: [
        r'\bcompare\b', r'\bdifferent\b', r'\bsimilar\b', r'\bversus\b',
        r'\bbetter\s+(?:than|or)\b', r'\bworse\s+(?:than|or)\b', r'\bvs\b',
    ],
    QuestionType.CRITIQUE: [
        r'\bpros?\b.*\bcons?\b', r'\bstrength\b.*\bweakness\b', r'\badvantage\b',
        r'\bdisadvantage\b', r'\bbenefit\b', r'\bdrawback\b',
    ],
    QuestionType.RATING_EXPLANATION: [
        r'\bwhy\s+(?:did\s+you\s+)?(?:give|choose|select)\s+(?:that|this)\s+(?:rating|score|number)\b',
        r'\bexplain\s+(?:your\s+)?(?:rating|score)\b', r'\breason\s+for\s+(?:your\s+)?(?:rating|score)\b',
    ],

    # ========== REFLECTIVE QUESTIONS ==========
    QuestionType.REFLECTION: [
        r'\breflect\b', r'\blook\s+back\b', r'\brecall.*experience\b',
        r'\bthink\s+back\b', r'\bremember\s+how\b',
    ],
    QuestionType.INTROSPECTION: [
        r'\bwhat\s+were\s+you\s+thinking\b', r'\bgoing\s+through\s+your\s+mind\b',
        r'\binner\b', r'\bfelt\s+inside\b', r'\bthoughts\s+during\b',
    ],
    QuestionType.MEMORY: [
        r'\bremember\b', r'\brecall\b', r'\btime\s+when\b', r'\blast\s+time\b',
        r'\bpast\s+experience\b', r'\bcan\s+you\s+think\s+of\b',
    ],
    QuestionType.EXPERIENCE: [
        r'\bexperience\b', r'\bwent\s+through\b', r'\bhappened\b',
        r'\boccurred\b', r'\bencounter\b', r'\bsituation\b',
    ],

    # ========== OPINION/ATTITUDE QUESTIONS ==========
    QuestionType.OPINION: [
        r'\bopinion\b', r'\bthink\s+about\b', r'\bview\s+on\b', r'\bfeel\s+about\b',
        r'\bperspective\b', r'\bstance\b', r'\bposition\b',
    ],
    QuestionType.BELIEF: [
        r'\bbeliev\b', r'\bconvinced\b', r'\bcertain\b', r'\bsure\b',
        r'\bconfident\b', r'\btrust\b', r'\bfaith\b',
    ],
    QuestionType.PREFERENCE: [
        r'\bprefer\b', r'\bfavorite\b', r'\blike\s+(?:more|better)\b',
        r'\brather\b', r'\bchoose\b', r'\bopt\s+for\b',
    ],
    QuestionType.ATTITUDE: [
        r'\bfeel\s+about\b', r'\battitude\b', r'\bsentiment\b',
        r'\bdisposition\b', r'\btoward\b',
    ],
    QuestionType.VALUE: [
        r'\bvalue\b', r'\bprinciple\b', r'\bethic\b', r'\bmoral\b',
        r'\bimportant\s+to\s+you\b', r'\bmatter\s+to\s+you\b',
    ],

    # ========== FORWARD-LOOKING QUESTIONS ==========
    QuestionType.PREDICTION: [
        r'\bpredict\b', r'\bexpect\b', r'\bwhat.*happen\b', r'\bthink.*will\b',
        r'\bforecast\b', r'\banticipate\b', r'\bfuture\b',
    ],
    QuestionType.INTENTION: [
        r'\bintend\b', r'\bplan\b', r'\bgoing\s+to\b', r'\bwill\s+you\b',
        r'\bnext\s+step\b', r'\bdo\s+(?:next|differently)\b',
    ],
    QuestionType.SUGGESTION: [
        r'\bsuggest\b', r'\bimprove\b', r'\bchange\b', r'\bbetter\b',
        r'\brecommend\b', r'\badvice\b', r'\benhance\b',
    ],
    QuestionType.RECOMMENDATION: [
        r'\brecommend\b', r'\badvise\b', r'\bwould\s+you\s+suggest\b',
        r'\btip\b', r'\bguidance\b', r'\bpointer\b',
    ],

    # ========== ASSOCIATIVE QUESTIONS ==========
    QuestionType.ASSOCIATION: [
        r'\bcomes?\s+to\s+mind\b', r'\bassociate\b', r'\bthink\s+of\b',
        r'\bconnect\b', r'\blink\b', r'\bremind\b',
    ],
    QuestionType.IMPRESSION: [
        r'\bimpression\b', r'\bfirst\s+(?:thought|reaction)\b', r'\binitial\b',
        r'\bimmediately\b', r'\binstinct\b', r'\bgut\b',
    ],

    # ========== FEEDBACK QUESTIONS ==========
    QuestionType.FEEDBACK: [
        r'\bfeedback\b', r'\bcomments?\b', r'\bsuggestions?\b',
        r'\banything.*(?:confusing|unclear)\b', r'\bthoughts.*survey\b',
        r'\bhow\s+was\b', r'\bwhat\s+did\s+you\s+think\s+of\b',
    ],
    QuestionType.COMMENT: [
        r'\bcomment\b', r'\badditional\b', r'\bother\s+(?:thoughts|comments)\b',
        r'\banything\s+else\b', r'\bfinal\s+(?:thoughts|words)\b',
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
    # ========== BEHAVIORAL ECONOMICS ==========
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
        'ultimatum', 'proposer', 'responder', 'accept', 'reject', 'offer',
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
    StudyDomain.LOSS_AVERSION: [
        'loss', 'aversion', 'avoid loss', 'fear of losing', 'lose money',
        'potential loss', 'downside risk',
    ],
    StudyDomain.FRAMING_EFFECTS: [
        'fram', 'present', 'word', 'describ', 'format', 'way it was',
    ],
    StudyDomain.ANCHORING: [
        'anchor', 'initial value', 'starting point', 'reference', 'adjust',
    ],
    StudyDomain.SUNK_COST: [
        'sunk cost', 'already invested', 'past investment', 'continue', 'abandon',
    ],

    # ========== SOCIAL PSYCHOLOGY ==========
    StudyDomain.INTERGROUP: [
        'outgroup', 'ingroup', 'other group', 'group membership',
        'intergroup', 'different group', 'us vs them', 'we vs they',
        'group conflict', 'group competition', 'group boundaries',
        'social categorization', 'minimal group', 'group bias',
    ],
    StudyDomain.IDENTITY: [
        'identity', 'who you are', 'self-concept', 'belonging',
        'identification', 'sense of self', 'personal identity',
        'social identity', 'self-definition', 'identity formation',
        'self-perception', 'self-image', 'who am i', 'define yourself',
    ],
    StudyDomain.NORMS: [
        'norm', 'expected', 'appropriate', 'should do', 'convention',
        'social expectation', 'social norm', 'descriptive norm',
        'injunctive norm', 'prescriptive', 'what others do',
        'normative', 'standard behavior', 'unwritten rule', 'custom',
        'acceptable', 'typical behavior', 'common practice',
    ],
    StudyDomain.TRUST: [
        'trust', 'trustworthy', 'reliable', 'honest', 'dependable',
        'credible', 'credibility', 'faith', 'confidence in', 'belief in',
        'distrust', 'mistrust', 'suspicious', 'skeptical', 'doubtful',
        'trustworthiness', 'betrayal', 'broken trust', 'trusting',
        'institutional trust', 'interpersonal trust', 'general trust',
    ],
    StudyDomain.FAIRNESS: [
        'fair', 'unfair', 'equal', 'equitable', 'just', 'justice',
        'equity', 'equality', 'injustice', 'unjust', 'impartial',
        'biased', 'discriminatory', 'unbiased', 'balanced', 'even-handed',
        'distributive justice', 'procedural fairness', 'fair treatment',
        'deserve', 'entitlement', 'proportional', 'merit',
    ],
    StudyDomain.COOPERATION: [
        'cooperat', 'collaborate', 'work together', 'joint', 'mutual benefit',
        'teamwork', 'partnership', 'collective action', 'coordination',
        'shared goal', 'common goal', 'mutual', 'reciprocal', 'reciprocity',
        'cooperative', 'helping each other', 'joint effort', 'synergy',
        'collective', 'pooling resources', 'working as a team',
    ],
    StudyDomain.CONFORMITY: [
        'conform', 'social pressure', 'group influence', 'majority', 'peer',
        'peer pressure', 'follow the crowd', 'go along', 'fit in',
        'compliance', 'obedience', 'herd', 'bandwagon', 'groupthink',
        'social conformity', 'normative influence', 'informational influence',
    ],
    StudyDomain.PROSOCIAL: [
        'helping', 'altru', 'prosocial', 'charity', 'donation', 'volunteer',
        'kindness', 'generosity', 'benevolence', 'selfless', 'good deed',
        'give back', 'philanthropy', 'caring', 'supportive', 'compassionate',
        'humanitarian', 'empathetic action', 'civic duty', 'community service',
    ],
    StudyDomain.SOCIAL_INFLUENCE: [
        'influence', 'persuad', 'social proof', 'comply', 'obedien',
        'persuasion', 'convince', 'sway', 'impact', 'affect behavior',
        'change mind', 'attitude change', 'behavior change', 'nudge',
        'manipulation', 'social pressure', 'authority', 'power',
    ],
    StudyDomain.ATTRIBUTION: [
        'attribut', 'cause', 'explain', 'blame', 'credit', 'responsib',
        'reason', 'why', 'internal cause', 'external cause', 'dispositional',
        'situational', 'fundamental attribution error', 'locus of control',
        'accountability', 'fault', 'culpability', 'explanation',
    ],
    StudyDomain.STEREOTYPE: [
        'stereotype', 'generaliz', 'category', 'group characteristic',
        'assumption', 'preconception', 'typify', 'label', 'categorize',
        'overgeneralize', 'group trait', 'perceived characteristic',
        'implicit stereotype', 'explicit stereotype', 'stereotype threat',
    ],
    StudyDomain.PREJUDICE: [
        'prejud', 'discriminat', 'bias', 'racist', 'sexist',
        'bigotry', 'intolerance', 'xenophob', 'homophob', 'ageism',
        'ableism', 'discrimination', 'hostile', 'negative attitude',
        'out-group hostility', 'hate', 'derogatory', 'marginaliz',
    ],
    StudyDomain.SELF_ESTEEM: [
        'self-esteem', 'self esteem', 'confidence', 'self-worth', 'self image',
        'self-value', 'self-respect', 'self-regard', 'self-confidence',
        'insecurity', 'low self-esteem', 'high self-esteem', 'ego',
        'self-evaluation', 'self-perception', 'self-belief',
    ],
    StudyDomain.EMPATHY: [
        'empathy', 'empathic', 'compassion', 'feel for others', 'perspective taking',
        'understand feelings', 'emotional understanding', 'put yourself in',
        'walk in shoes', 'emotional attunement', 'sympathy', 'caring',
        'affective empathy', 'cognitive empathy', 'empathetic', 'empathize',
    ],

    # ========== POLITICAL SCIENCE ==========
    StudyDomain.POLITICAL: [
        'politic', 'democrat', 'republican', 'liberal', 'conservative',
        'government', 'policy', 'election', 'vote',
    ],
    StudyDomain.POLARIZATION: [
        'polariz', 'divided', 'partisan', 'other side', 'opposing view',
    ],
    StudyDomain.PARTISANSHIP: [
        'partisan', 'party', 'democrat', 'republican', 'liberal', 'conservative',
    ],
    StudyDomain.VOTING: [
        'vote', 'election', 'ballot', 'candidate', 'electoral',
    ],
    StudyDomain.MEDIA: [
        'media', 'news', 'journalism', 'coverage', 'report',
    ],
    StudyDomain.POLICY_ATTITUDES: [
        'policy', 'legislation', 'law', 'regulation', 'government action',
    ],
    StudyDomain.CIVIC_ENGAGEMENT: [
        'civic', 'citizen', 'participat', 'community', 'volunteer',
    ],
    StudyDomain.POLITICAL_TRUST: [
        'trust government', 'trust politician', 'institutional trust',
    ],
    StudyDomain.IDEOLOGY: [
        'ideolog', 'left', 'right', 'progressive', 'traditionalist',
    ],
    StudyDomain.MISINFORMATION: [
        'misinformation', 'fake news', 'fact check', 'false claim', 'debunk',
    ],

    # ========== CONSUMER/MARKETING ==========
    StudyDomain.CONSUMER: [
        'product', 'brand', 'purchase', 'buy', 'consumer', 'shopping',
        'customer', 'retail', 'e-commerce', 'ecommerce', 'checkout',
        'cart', 'online shopping', 'store', 'shop', 'buyer', 'shopper',
        'marketplace', 'merchant', 'vendor', 'seller', 'goods', 'item',
        'commodity', 'merchandise', 'consumerism', 'spending', 'wallet',
    ],
    StudyDomain.BRAND: [
        'brand', 'logo', 'company name', 'brand loyalty', 'brand image',
        'brand perception', 'brand awareness', 'brand recognition',
        'brand equity', 'brand value', 'corporate brand', 'branded',
        'branding', 'trademark', 'brand identity', 'brand association',
    ],
    StudyDomain.ADVERTISING: [
        'ad', 'advertis', 'commercial', 'marketing', 'promotion',
        'campaign', 'banner', 'sponsored', 'endorsement', 'celebrity',
        'influencer marketing', 'native ad', 'programmatic', 'targeting',
        'retargeting', 'impression', 'click', 'ctr', 'conversion',
        'ad recall', 'brand lift', 'promotional', 'promo',
    ],
    StudyDomain.PRODUCT_EVALUATION: [
        'product', 'purchase', 'buy', 'quality', 'review', 'rating',
        'star rating', 'product review', 'customer review', 'feedback',
        'feature', 'specification', 'performance', 'durability',
        'reliability', 'product quality', 'value', 'worth', 'assess',
    ],
    StudyDomain.PURCHASE_INTENT: [
        'intend to buy', 'purchase intent', 'likely to purchase', 'would buy',
        'willing to purchase', 'plan to buy', 'consider buying', 'interest',
        'shopping intention', 'buying intention', 'likelihood of purchase',
        'propensity to buy', 'purchase decision', 'buying decision',
    ],
    StudyDomain.BRAND_LOYALTY: [
        'brand loyal', 'stick with brand', 'prefer brand', 'brand switch',
        'repeat purchase', 'brand commitment', 'brand attachment',
        'favorite brand', 'go-to brand', 'always buy', 'never switch',
        'loyal customer', 'brand advocate', 'brand fan', 'devoted',
    ],
    StudyDomain.PRICE_PERCEPTION: [
        'price', 'cost', 'value for money', 'expensive', 'cheap', 'worth',
        'affordable', 'pricing', 'discount', 'sale', 'deal', 'bargain',
        'premium', 'budget', 'luxury', 'overpriced', 'underpriced',
        'price sensitive', 'willingness to pay', 'wtp', 'price point',
    ],
    StudyDomain.SERVICE_QUALITY: [
        'service', 'customer service', 'support', 'staff', 'helpful',
        'responsiveness', 'reliability', 'assurance', 'empathy',
        'tangibles', 'service experience', 'service encounter',
        'wait time', 'queue', 'friendly', 'courteous', 'professional',
    ],
    StudyDomain.CUSTOMER_SATISFACTION: [
        'satisf', 'pleased', 'happy with', 'content', 'dissatisf',
        'delight', 'exceeds expectations', 'meets expectations',
        'disappointed', 'frustrat', 'csat', 'nps', 'net promoter',
        'customer experience', 'cx', 'experience rating',
    ],
    StudyDomain.WORD_OF_MOUTH: [
        'recommend', 'tell others', 'word of mouth', 'refer', 'share',
        'referral', 'ewom', 'online review', 'testimonial', 'endorsement',
        'buzz', 'viral', 'spread', 'pass along', 'tell a friend',
        'social proof', 'user generated', 'ugc', 'rating',
    ],

    # ========== ORGANIZATIONAL BEHAVIOR ==========
    StudyDomain.WORKPLACE: [
        'work', 'job', 'employee', 'organization', 'colleague', 'office',
    ],
    StudyDomain.LEADERSHIP: [
        'leader', 'manager', 'boss', 'supervisor', 'management',
    ],
    StudyDomain.TEAMWORK: [
        'teamwork', 'team', 'group project', 'collaboration',
    ],
    StudyDomain.MOTIVATION: [
        'motivat', 'drive', 'engagement', 'interest', 'passion',
    ],
    StudyDomain.JOB_SATISFACTION: [
        'job satisf', 'happy at work', 'enjoy work', 'work satisf',
    ],
    StudyDomain.ORGANIZATIONAL_COMMITMENT: [
        'commit', 'loyal', 'stay with', 'turnover', 'leave job',
    ],
    StudyDomain.WORK_LIFE_BALANCE: [
        'work-life', 'work life balance', 'personal time', 'family time',
    ],
    StudyDomain.EMPLOYEE_ENGAGEMENT: [
        'engag', 'involved', 'invested', 'enthusiasm', 'passionate',
    ],
    StudyDomain.ORGANIZATIONAL_CULTURE: [
        'culture', 'values', 'company culture', 'organizational culture',
    ],

    # ========== TECHNOLOGY/AI ==========
    StudyDomain.AI_ATTITUDES: [
        'ai', 'artificial intelligence', 'algorithm', 'machine learning',
        'robot', 'automation', 'chatbot', 'gpt', 'llm', 'language model',
        'neural network', 'deep learning', 'generative ai', 'gen ai',
        'large language model', 'ai system', 'intelligent system',
        'autonomous', 'ai-powered', 'ai tool', 'ai assistant', 'copilot',
        'claude', 'openai', 'anthropic', 'gemini', 'transformer',
        'foundation model', 'artificial', 'intelligent agent',
    ],
    StudyDomain.PRIVACY: [
        'privacy', 'data', 'personal information', 'tracking', 'surveillance',
        'data collection', 'gdpr', 'consent', 'data protection', 'data sharing',
        'monitoring', 'personal data', 'confidential', 'anonymity', 'anonymous',
    ],
    StudyDomain.AUTOMATION: [
        'automat', 'robot', 'machine', 'ai replace', 'robotic', 'robotics',
        'automated system', 'self-driving', 'autonomous vehicle', 'driverless',
        'smart home', 'iot', 'automated process', 'workflow automation',
        'rpa', 'robotic process', 'bot', 'automated task',
    ],
    StudyDomain.ALGORITHM_AVERSION: [
        'algorithm', 'ai decision', 'computer decision', 'automated decision',
        'algorithmic', 'machine decision', 'ai recommendation', 'recommender',
        'algorithmic bias', 'black box', 'explainability', 'ai judgment',
        'computer-based', 'data-driven decision', 'predictive model',
    ],
    StudyDomain.TECHNOLOGY_ADOPTION: [
        'adopt', 'new technology', 'try new', 'early adopter', 'tech savvy',
        'technology acceptance', 'digital adoption', 'tech literacy',
        'digital transformation', 'innovation', 'cutting edge', 'emerging tech',
        'tech use', 'technology use', 'digital skill', 'technophobe',
    ],
    StudyDomain.SOCIAL_MEDIA: [
        'social media', 'facebook', 'twitter', 'instagram', 'tiktok', 'online',
        'linkedin', 'snapchat', 'youtube', 'reddit', 'platform', 'post',
        'share', 'like', 'follow', 'influencer', 'viral', 'content creator',
        'social network', 'feed', 'timeline', 'notification', 'scroll',
    ],
    StudyDomain.DIGITAL_WELLBEING: [
        'screen time', 'digital', 'phone use', 'internet use', 'tech addiction',
        'digital detox', 'smartphone', 'device use', 'digital health',
        'online time', 'doom scrolling', 'nomophobia', 'fomo', 'digital balance',
    ],
    StudyDomain.HUMAN_AI_INTERACTION: [
        'interact with ai', 'human-ai', 'chatbot', 'virtual assistant',
        'conversational ai', 'dialogue system', 'voice assistant', 'siri',
        'alexa', 'human-machine', 'ai interaction', 'talk to ai',
        'chat with ai', 'ai conversation', 'human-robot', 'hri', 'hai',
    ],
    StudyDomain.CYBERSECURITY: [
        'security', 'hack', 'cyber', 'password', 'protection', 'phishing',
        'malware', 'breach', 'data breach', 'encryption', 'secure',
        'authentication', 'two-factor', '2fa', 'identity theft', 'scam',
    ],

    # ========== HEALTH PSYCHOLOGY ==========
    StudyDomain.HEALTH: [
        'health', 'medical', 'doctor', 'illness', 'wellbeing', 'wellness',
        'disease', 'symptom', 'diagnosis', 'healthcare', 'hospital', 'nurse',
        'therapy', 'prescription', 'healthy', 'unhealthy', 'sick', 'sickness',
        'clinic', 'physician', 'health care', 'health status', 'physical health',
        'body', 'fitness', 'prevention', 'screening', 'check-up', 'checkup',
    ],
    StudyDomain.MEDICAL_DECISION: [
        'medical', 'health decision', 'treatment', 'doctor', 'patient',
        'treatment option', 'medical choice', 'clinical decision', 'prognosis',
        'surgery', 'procedure', 'intervention', 'therapeutic', 'cure',
        'diagnosis', 'second opinion', 'informed consent', 'risk benefit',
        'treatment preference', 'shared decision', 'medical advice',
    ],
    StudyDomain.WELLBEING: [
        'wellbeing', 'well-being', 'happiness', 'life satisfaction', 'quality of life',
        'flourishing', 'thriving', 'positive affect', 'subjective wellbeing',
        'psychological wellbeing', 'emotional wellbeing', 'life quality',
        'satisfied with life', 'contentment', 'fulfillment', 'meaning in life',
    ],
    StudyDomain.HEALTH_BEHAVIOR: [
        'exercise', 'diet', 'sleep', 'smoking', 'alcohol', 'healthy habit',
        'physical activity', 'nutrition', 'eating', 'drink', 'substance',
        'sedentary', 'active lifestyle', 'workout', 'fitness routine',
        'healthy eating', 'junk food', 'weight', 'obesity', 'bmi',
        'health behavior', 'lifestyle', 'habit', 'behavioral health',
    ],
    StudyDomain.MENTAL_HEALTH: [
        'mental health', 'anxiety', 'depression', 'stress', 'psychological',
        'emotional health', 'psychiatric', 'therapy', 'counseling', 'therapist',
        'psychologist', 'mental illness', 'mental disorder', 'mood', 'emotion',
        'wellbeing', 'psychological distress', 'ptsd', 'trauma', 'burnout',
        'mental wellbeing', 'emotional distress', 'panic', 'worry',
    ],
    StudyDomain.VACCINATION: [
        'vaccin', 'immuniz', 'shot', 'jab', 'vaccine hesitan',
        'anti-vax', 'antivax', 'immunization', 'booster', 'dose',
        'covid vaccine', 'flu shot', 'vaccination rate', 'herd immunity',
        'vaccine safety', 'vaccine efficacy', 'vaccine side effect',
        'vaccination intention', 'vaccine confidence', 'vaccine trust',
    ],
    StudyDomain.PAIN_MANAGEMENT: [
        'pain', 'chronic pain', 'medication', 'relief', 'suffer',
        'painkiller', 'analgesic', 'opioid', 'pain treatment', 'ache',
        'discomfort', 'pain intensity', 'pain threshold', 'pain tolerance',
        'acute pain', 'neuropathic', 'pain relief', 'pain control',
    ],
    StudyDomain.HEALTH_ANXIETY: [
        'health anxiety', 'worry about health', 'hypochondr', 'illness anxiety',
        'cyberchondria', 'health concern', 'fear of illness', 'health worry',
        'somatic symptom', 'illness worry', 'health fear', 'disease anxiety',
    ],
    StudyDomain.PATIENT_PROVIDER: [
        'doctor', 'patient', 'provider', 'communication', 'trust doctor',
        'physician', 'nurse', 'healthcare provider', 'patient-doctor',
        'medical communication', 'bedside manner', 'patient satisfaction',
        'adherence', 'compliance', 'follow instructions', 'patient care',
        'healthcare professional', 'clinician', 'patient experience',
    ],
    StudyDomain.CHRONIC_ILLNESS: [
        'chronic', 'long-term', 'managing', 'living with', 'condition',
        'chronic disease', 'chronic condition', 'ongoing', 'persistent',
        'diabetes', 'hypertension', 'heart disease', 'arthritis', 'asthma',
        'autoimmune', 'disability', 'impairment', 'management', 'self-care',
        'coping with illness', 'disease management', 'long-term condition',
    ],

    # ========== EDUCATION ==========
    StudyDomain.EDUCATION: [
        'education', 'school', 'university', 'college', 'learning',
    ],
    StudyDomain.LEARNING: [
        'learn', 'education', 'study', 'knowledge', 'skill',
    ],
    StudyDomain.ACADEMIC_MOTIVATION: [
        'study motivation', 'academic', 'grades', 'achievement', 'perform',
    ],
    StudyDomain.TEACHING_EFFECTIVENESS: [
        'teach', 'instructor', 'professor', 'effective teaching',
    ],
    StudyDomain.ONLINE_LEARNING: [
        'online', 'remote', 'virtual', 'distance', 'e-learning',
    ],
    StudyDomain.EDUCATIONAL_TECHNOLOGY: [
        'edtech', 'educational technology', 'learning platform', 'digital learning',
    ],
    StudyDomain.STUDENT_ENGAGEMENT: [
        'engag', 'particip', 'attentive', 'interest', 'involv',
    ],
    StudyDomain.ASSESSMENT_FEEDBACK: [
        'feedback', 'assessment', 'grade', 'evaluation', 'test',
    ],

    # ========== ETHICS/MORAL ==========
    StudyDomain.ETHICS: [
        'ethic', 'moral', 'right', 'wrong', 'principl',
    ],
    StudyDomain.MORAL_JUDGMENT: [
        'moral', 'ethical', 'right', 'wrong', 'should', 'ought',
    ],
    StudyDomain.MORAL_DILEMMA: [
        'dilemma', 'trolley', 'sacrifice', 'utilitarian', 'deontolog',
    ],
    StudyDomain.ETHICAL_LEADERSHIP: [
        'ethical leader', 'integrity', 'honest leader', 'moral leader',
    ],
    StudyDomain.CORPORATE_ETHICS: [
        'corporate', 'business ethics', 'csr', 'corporate social',
    ],
    StudyDomain.RESEARCH_ETHICS: [
        'research ethics', 'informed consent', 'deception', 'irb',
    ],
    StudyDomain.MORAL_EMOTIONS: [
        'guilt', 'shame', 'pride', 'moral emotion', 'regret',
    ],
    StudyDomain.VALUES: [
        'values', 'priorities', 'what matters', 'important to me',
    ],

    # ========== ENVIRONMENTAL ==========
    StudyDomain.ENVIRONMENTAL: [
        'environment', 'climate', 'green', 'sustainab', 'eco', 'carbon',
    ],
    StudyDomain.SUSTAINABILITY: [
        'sustainab', 'green', 'eco', 'renewable', 'recycle',
    ],
    StudyDomain.CLIMATE_ATTITUDES: [
        'climate', 'global warming', 'greenhouse', 'carbon',
    ],
    StudyDomain.PRO_ENVIRONMENTAL: [
        'environmentally friendly', 'green behavior', 'eco-friendly',
    ],
    StudyDomain.GREEN_CONSUMPTION: [
        'green product', 'eco product', 'sustainable product', 'ethical consumption',
    ],
    StudyDomain.CONSERVATION: [
        'conserv', 'protect', 'preserve', 'wildlife', 'nature',
    ],
    StudyDomain.ENERGY_BEHAVIOR: [
        'energy', 'electricity', 'power', 'solar', 'renewable',
    ],
    StudyDomain.ENVIRONMENTAL_JUSTICE: [
        'environmental justice', 'pollution', 'affected communities',
    ],

    # ========== COGNITIVE PSYCHOLOGY ==========
    StudyDomain.COGNITIVE: [
        'cognitive', 'thinking', 'mental process', 'mind',
    ],
    StudyDomain.DECISION_MAKING: [
        'decision', 'choice', 'select', 'option', 'alternative',
    ],
    StudyDomain.MEMORY: [
        'remember', 'recall', 'forget', 'memory', 'recognition',
    ],
    StudyDomain.ATTENTION: [
        'attention', 'focus', 'concentrate', 'distract',
    ],
    StudyDomain.REASONING: [
        'reason', 'logic', 'infer', 'deduc', 'induc',
    ],
    StudyDomain.PROBLEM_SOLVING: [
        'problem', 'solve', 'solution', 'figure out',
    ],
    StudyDomain.COGNITIVE_BIAS: [
        'bias', 'heuristic', 'shortcut', 'cognitive error',
    ],
    StudyDomain.METACOGNITION: [
        'metacognit', 'thinking about thinking', 'self-aware', 'monitoring',
    ],

    # ========== DEVELOPMENTAL ==========
    StudyDomain.DEVELOPMENTAL: [
        'develop', 'age', 'life stage', 'mature',
    ],
    StudyDomain.PARENTING: [
        'parent', 'child', 'mother', 'father', 'raising',
    ],
    StudyDomain.CHILDHOOD: [
        'child', 'kid', 'young', 'grow up', 'adolescent',
    ],
    StudyDomain.AGING: [
        'aging', 'older', 'elderly', 'senior', 'retirement',
    ],
    StudyDomain.LIFE_TRANSITIONS: [
        'transition', 'change', 'new phase', 'moving on',
    ],
    StudyDomain.INTERGENERATIONAL: [
        'generation', 'millennial', 'boomer', 'gen z', 'gen x',
    ],

    # ========== CLINICAL ==========
    StudyDomain.CLINICAL: [
        'clinical', 'disorder', 'symptom', 'diagnosis', 'treatment',
    ],
    StudyDomain.ANXIETY: [
        'anxiety', 'anxious', 'worry', 'nervous', 'panic',
    ],
    StudyDomain.DEPRESSION: [
        'depress', 'sad', 'hopeless', 'low mood', 'suicid',
    ],
    StudyDomain.COPING: [
        'cope', 'deal with', 'handle', 'manage', 'adapt',
    ],
    StudyDomain.THERAPY_ATTITUDES: [
        'therapy', 'counseling', 'psychologist', 'mental health treatment',
    ],
    StudyDomain.STRESS: [
        'stress', 'pressure', 'overwhelm', 'burnout', 'strain',
    ],

    # ========== COMMUNICATION ==========
    StudyDomain.COMMUNICATION: [
        'communicat', 'message', 'information', 'convey',
    ],
    StudyDomain.PERSUASION: [
        'persuad', 'convinc', 'influence', 'argument', 'appeal',
    ],
    StudyDomain.MEDIA_EFFECTS: [
        'media effect', 'media impact', 'influence of media',
    ],
    StudyDomain.INTERPERSONAL: [
        'interpersonal', 'relationship', 'interaction', 'social',
    ],
    StudyDomain.PUBLIC_OPINION: [
        'public opinion', 'popular', 'majority think', 'general public',
    ],
    StudyDomain.NARRATIVE: [
        'story', 'narrative', 'told', 'account', 'anecdote',
    ],

    # ========== ECONOMICS ==========
    StudyDomain.ECONOMICS: [
        'economic', 'money', 'financial', 'market', 'trade',
    ],
    StudyDomain.NEGOTIATION: [
        'negotiat', 'bargain', 'deal', 'agreement', 'compromise',
    ],
    StudyDomain.BARGAINING: [
        'bargain', 'haggle', 'price negotiation', 'deal-making',
    ],
    StudyDomain.FINANCIAL_DECISION: [
        'financial decision', 'money decision', 'investment', 'spend',
    ],
    StudyDomain.SAVING_BEHAVIOR: [
        'save', 'saving', 'retirement fund', 'emergency fund',
    ],
    StudyDomain.ECONOMIC_EXPECTATIONS: [
        'economic outlook', 'expect', 'forecast', 'prediction',
    ],

    # ========== NEUROSCIENCE/NEUROECONOMICS ==========
    StudyDomain.NEUROECONOMICS: [
        'neuroeconomics', 'brain', 'neural', 'neuroscience', 'fmri', 'neuroimaging',
    ],
    StudyDomain.REWARD_PROCESSING: [
        'reward', 'dopamine', 'pleasure', 'anticipation', 'gratification',
    ],
    StudyDomain.IMPULSE_CONTROL: [
        'impulse', 'inhibition', 'self-control', 'impulsive', 'restraint',
    ],
    StudyDomain.EMOTIONAL_REGULATION: [
        'emotion regulation', 'emotional control', 'manage emotion', 'regulate feeling',
    ],
    StudyDomain.NEURAL_DECISION: [
        'neural decision', 'brain decision', 'cognitive neuroscience', 'decision brain',
    ],
    StudyDomain.COGNITIVE_LOAD: [
        'cognitive load', 'mental effort', 'working memory', 'processing capacity',
    ],

    # ========== SPORTS PSYCHOLOGY ==========
    StudyDomain.SPORTS_PSYCHOLOGY: [
        'sport', 'athlete', 'competition', 'athletic', 'exercise',
    ],
    StudyDomain.ATHLETIC_MOTIVATION: [
        'athletic motivation', 'training motivation', 'sports drive', 'competitive drive',
    ],
    StudyDomain.TEAM_DYNAMICS: [
        'team dynamic', 'team cohesion', 'teamwork sport', 'team chemistry',
    ],
    StudyDomain.PERFORMANCE_ANXIETY: [
        'performance anxiety', 'choking', 'competition anxiety', 'stage fright',
    ],
    StudyDomain.COACH_ATHLETE: [
        'coach', 'coaching', 'trainer', 'athlete relationship', 'mentor sport',
    ],
    StudyDomain.FAN_BEHAVIOR: [
        'fan', 'spectator', 'supporter', 'fandom', 'cheering',
    ],

    # ========== LEGAL PSYCHOLOGY ==========
    StudyDomain.LEGAL_PSYCHOLOGY: [
        'legal', 'law', 'court', 'justice system', 'forensic',
    ],
    StudyDomain.JURY_DECISION: [
        'jury', 'verdict', 'deliberation', 'juror', 'trial decision',
    ],
    StudyDomain.WITNESS_MEMORY: [
        'witness', 'eyewitness', 'testimony', 'lineup', 'identification',
    ],
    StudyDomain.PROCEDURAL_JUSTICE: [
        'procedural justice', 'fair process', 'due process', 'treatment by authorities',
    ],
    StudyDomain.CRIMINAL_JUSTICE: [
        'criminal', 'crime', 'offender', 'sentencing', 'punishment',
    ],
    StudyDomain.LEGAL_COMPLIANCE: [
        'compliance law', 'follow rules', 'regulatory compliance', 'legal obedience',
    ],

    # ========== FOOD/NUTRITION PSYCHOLOGY ==========
    StudyDomain.FOOD_PSYCHOLOGY: [
        'food', 'eating', 'nutrition', 'diet', 'meal',
    ],
    StudyDomain.EATING_BEHAVIOR: [
        'eating behavior', 'food consumption', 'eating pattern', 'dietary habit',
    ],
    StudyDomain.FOOD_CHOICE: [
        'food choice', 'food selection', 'choose food', 'food preference',
    ],
    StudyDomain.NUTRITION_KNOWLEDGE: [
        'nutrition knowledge', 'healthy eating', 'nutritional', 'dietary knowledge',
    ],
    StudyDomain.BODY_IMAGE: [
        'body image', 'appearance', 'weight concern', 'body satisfaction',
    ],
    StudyDomain.DIET_ADHERENCE: [
        'diet adherence', 'dietary compliance', 'follow diet', 'stick to diet',
    ],

    # ========== HUMAN FACTORS/ERGONOMICS ==========
    StudyDomain.HUMAN_FACTORS: [
        'human factors', 'ergonomic', 'usability', 'interface', 'design',
    ],
    StudyDomain.USER_EXPERIENCE: [
        'user experience', 'ux', 'usability', 'user interface', 'interaction design',
    ],
    StudyDomain.INTERFACE_DESIGN: [
        'interface', 'ui design', 'screen design', 'visual design', 'layout',
    ],
    StudyDomain.SAFETY_BEHAVIOR: [
        'safety', 'hazard', 'risk prevention', 'protective behavior', 'accident',
    ],
    StudyDomain.WORKLOAD: [
        'workload', 'task demand', 'mental demand', 'work demand', 'capacity',
    ],
    StudyDomain.HUMAN_ERROR: [
        'human error', 'mistake', 'slip', 'lapse', 'error prevention',
    ],

    # ========== CROSS-CULTURAL ==========
    StudyDomain.CROSS_CULTURAL: [
        'cross-cultural', 'culture', 'cultural difference', 'multicultural',
    ],
    StudyDomain.CULTURAL_VALUES: [
        'cultural value', 'tradition', 'custom', 'heritage', 'cultural norm',
    ],
    StudyDomain.ACCULTURATION: [
        'acculturation', 'cultural adaptation', 'immigrant', 'cultural integration',
    ],
    StudyDomain.CULTURAL_IDENTITY: [
        'cultural identity', 'ethnic identity', 'heritage identity', 'bicultural',
    ],
    StudyDomain.GLOBAL_ATTITUDES: [
        'global', 'international', 'worldwide', 'globalization', 'foreign',
    ],

    # ========== POSITIVE PSYCHOLOGY ==========
    StudyDomain.POSITIVE_PSYCHOLOGY: [
        'positive psychology', 'happiness', 'wellbeing', 'thriving', 'optimal',
    ],
    StudyDomain.GRATITUDE: [
        'gratitude', 'thankful', 'appreciate', 'grateful', 'blessing',
    ],
    StudyDomain.RESILIENCE: [
        'resilience', 'bounce back', 'recover', 'adversity', 'overcome',
    ],
    StudyDomain.FLOURISHING: [
        'flourish', 'thrive', 'optimal functioning', 'human flourishing',
    ],
    StudyDomain.LIFE_SATISFACTION: [
        'life satisfaction', 'satisfied with life', 'quality of life', 'life evaluation',
    ],

    # ========== GENDER/SEXUALITY ==========
    StudyDomain.GENDER_PSYCHOLOGY: [
        'gender', 'sex difference', 'masculin', 'feminin', 'gender studies',
    ],
    StudyDomain.GENDER_ROLES: [
        'gender role', 'gender expectation', 'traditional gender', 'gender stereotype',
    ],
    StudyDomain.SEXUALITY_ATTITUDES: [
        'sexuality', 'sexual attitude', 'sexual orientation', 'sexual',
    ],
    StudyDomain.LGBTQ_EXPERIENCE: [
        'lgbtq', 'gay', 'lesbian', 'transgender', 'queer', 'bisexual',
    ],

    # ========== RELATIONSHIP/ATTACHMENT ==========
    StudyDomain.RELATIONSHIP: [
        'relationship', 'partner', 'couple', 'interpersonal', 'connection',
    ],
    StudyDomain.ATTACHMENT: [
        'attachment', 'attachment style', 'secure', 'anxious', 'avoidant',
    ],
    StudyDomain.ROMANTIC_RELATIONSHIPS: [
        'romantic', 'dating', 'love', 'romantic partner', 'intimate relationship',
    ],
    StudyDomain.SOCIAL_SUPPORT: [
        'social support', 'support network', 'emotional support', 'help from others',
    ],

    # ========== ADVANCED SOCIAL SCIENCE DOMAINS (v1.1.0) ==========

    # Cognitive & Self-Regulation
    StudyDomain.SELF_CONTROL: [
        'self-control', 'self control', 'willpower', 'impulse control', 'discipline',
        'temptation', 'resist', 'delay gratification', 'self-regulation',
    ],
    StudyDomain.EMOTION_REGULATION: [
        'emotion regulation', 'emotional control', 'manage emotions', 'coping',
        'emotional response', 'regulate feelings', 'emotional stability',
    ],
    StudyDomain.CONFIRMATION_BIAS: [
        'confirmation bias', 'seek information', 'biased search', 'motivated reasoning',
        'selective exposure', 'echo chamber', 'filter bubble',
    ],
    StudyDomain.MENTAL_EFFORT: [
        'mental effort', 'cognitive load', 'thinking effort', 'cognitive demand',
        'mental workload', 'processing', 'cognitive resources',
    ],

    # Interpersonal & Group
    StudyDomain.RELATIONSHIP_QUALITY: [
        'relationship quality', 'relationship satisfaction', 'interpersonal',
        'connection quality', 'closeness', 'relationship health',
    ],
    StudyDomain.GROUP_IDENTITY: [
        'group identity', 'social identity', 'group membership', 'collective identity',
        'in-group', 'group belonging', 'we-feeling',
    ],
    StudyDomain.BEHAVIOR_INTENTION: [
        'intention', 'intend to', 'plan to', 'likely to', 'willingness',
        'behavioral intention', 'future behavior',
    ],

    # Scientific Reasoning
    StudyDomain.SCIENTIFIC_REASONING: [
        'scientific reasoning', 'evidence-based', 'research', 'scientific method',
        'empirical', 'data-driven', 'scientific thinking',
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


def add_variation(response: str, persona_verbosity: float, persona_formality: float, local_rng: random.Random = None) -> str:
    """Add natural variation to a response based on persona traits.

    v1.0.0: Uses local RNG for deterministic, question-specific variation.
    v1.1.0: Added research-grounded phrasing for formal personas.
    """
    rng = local_rng or random.Random()
    result = response

    # Add hedging for less confident personas
    if rng.random() < (1 - persona_verbosity) * 0.4:
        hedge = rng.choice(HEDGING_PHRASES)
        if not result.startswith(('I ', 'My ')):
            result = hedge + result[0].lower() + result[1:]

    # Add connector for verbose personas
    if persona_verbosity > 0.6 and rng.random() < 0.3:
        connector = rng.choice(CONNECTORS)
        result += connector + rng.choice([
            "this is how I see it.",
            "that's my perspective.",
            "I feel fairly strongly about this.",
            "it just makes sense to me.",
        ])

    # v1.1.0: Add research-grounded phrasing for formal/high-engagement personas
    if persona_formality > 0.6 and rng.random() < 0.25:
        research_phrases = [
            " This aligns with what I've read about the topic.",
            " From what I understand, this is a common perspective.",
            " Based on my understanding, this makes sense.",
            " I think this reflects broader patterns in how people think.",
            " This is consistent with my general knowledge on the subject.",
            " Considering the evidence, this seems reasonable.",
        ]
        result = result.rstrip('.!?') + '.' + rng.choice(research_phrases)

    # Add casual modifiers for informal personas
    if persona_formality < 0.4 and rng.random() < 0.3:
        modifier = rng.choice(CASUAL_MODIFIERS)
        words = result.split()
        if len(words) > 3:
            insert_pos = rng.randint(1, min(3, len(words)-1))
            words.insert(insert_pos, modifier)
            result = ' '.join(words)

    # v1.1.0: Add thoughtful elaboration for high-verbosity personas
    if persona_verbosity > 0.7 and rng.random() < 0.2:
        elaborations = [
            " When I think about it more deeply, several factors come into play.",
            " There are multiple considerations that influenced my thinking.",
            " I've given this some thought and considered different angles.",
            " Looking at this from different perspectives helps clarify my view.",
        ]
        result = result.rstrip('.!?') + '.' + rng.choice(elaborations)

    # Add concluding phrase occasionally
    if rng.random() < 0.2 and not result.endswith(('.', '!', '?')):
        result += rng.choice(CONCLUDING_PHRASES)

    return result


# ============================================================================
# MAIN RESPONSE GENERATOR CLASS
# ============================================================================

class ComprehensiveResponseGenerator:
    """
    LLM-quality response generator for open-ended survey questions.

    Provides context-aware, persona-specific responses across 50+ research domains.

    v1.1.0 Enhancements:
    - Response fingerprinting to prevent duplicate sentences within a dataset
    - Topic-aware response generation that stays on survey subject
    - Sentence variation algorithms for natural text
    - Scientific grounding with research-based language patterns
    """

    # Class-level response fingerprint tracking for dataset uniqueness
    _used_responses: set = set()
    _used_sentences: set = set()
    _session_id: int = 0

    # Sentence variation patterns for natural diversity (v1.2.0: expanded for academic variety)
    SENTENCE_STARTERS = [
        # Personal perspective starters
        "I think ", "I feel ", "In my view, ", "From my perspective, ",
        "I believe ", "It seems to me that ", "My sense is that ",
        "I'd say ", "Personally, ", "For me, ", "Looking at this, ",
        "Considering this, ", "Reflecting on this, ", "In my experience, ",
        # Academic/thoughtful starters
        "Upon consideration, ", "After thinking about it, ", "My impression is that ",
        "Based on my understanding, ", "From what I can tell, ", "As I see it, ",
        "To my mind, ", "In my estimation, ", "My view is that ",
        "Having considered this, ", "On reflection, ", "My position is that ",
        # Engaged/confident starters
        "I'm fairly certain that ", "I would argue that ", "My take is that ",
        "The way I see it, ", "Speaking from experience, ", "I've come to think that ",
        "It's clear to me that ", "I've found that ", "What strikes me is that ",
        # Tentative/hedged starters
        "I suspect that ", "My initial thought is that ", "Tentatively, I'd say ",
        "I'm inclined to think that ", "If I had to say, ", "My gut feeling is that ",
    ]

    TRANSITION_PHRASES = [
        # Additive transitions
        "Also, ", "Additionally, ", "Moreover, ", "Furthermore, ",
        "On top of that, ", "What's more, ", "Beyond that, ",
        "Plus, ", "And ", "Not only that, but ", "Similarly, ",
        # Elaborative transitions
        "In addition to that, ", "Along the same lines, ", "Building on that, ",
        "To elaborate, ", "To add to this, ", "Following from that, ",
        "Equally important, ", "Another point is that ", "Related to this, ",
        # Contrastive transitions
        "That said, ", "However, ", "On the other hand, ", "At the same time, ",
        "Yet, ", "Still, ", "Conversely, ", "Nevertheless, ",
        # Causal/consequential transitions
        "As a result, ", "Consequently, ", "Because of this, ", "For this reason, ",
        "This means that ", "It follows that ", "Given that, ",
        # Emphatic transitions
        "Importantly, ", "Notably, ", "Significantly, ", "In particular, ",
        "Especially, ", "What's particularly relevant is ", "Key to this is that ",
    ]

    CONCLUDING_PHRASES = [
        # Casual conclusions
        "That's my take on it.", "That's how I see it.", "Those are my thoughts.",
        "That's what comes to mind.", "That's my perspective.",
        "That's where I stand.", "That sums up my view.",
        # Academic/formal conclusions
        "This reflects my overall assessment.", "That encapsulates my thinking on this.",
        "This is the conclusion I've reached.", "That represents my considered view.",
        "This summarizes my position.", "Those are my key observations.",
        # Reflective conclusions
        "That's what I've concluded after thinking it through.",
        "This is where my reflection has led me.", "That's my honest assessment.",
        "This captures my main impressions.", "Those are the points that stand out to me.",
        # Open-ended conclusions
        "There's more I could say, but those are the main points.",
        "I think that covers my main thoughts.", "That's the gist of my perspective.",
        "I believe that addresses the key aspects.", "That's what seems most relevant to me.",
    ]

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.study_context: Dict[str, Any] = {}
        # Initialize fresh session for uniqueness tracking
        ComprehensiveResponseGenerator._session_id += 1
        ComprehensiveResponseGenerator._used_responses = set()
        ComprehensiveResponseGenerator._used_sentences = set()

    def set_study_context(self, context: Dict[str, Any]):
        """Set the study context for context-aware generation."""
        self.study_context = context or {}

    def _get_response_fingerprint(self, response: str) -> str:
        """Generate a fingerprint for detecting duplicate responses."""
        # Normalize: lowercase, remove punctuation, collapse spaces
        normalized = re.sub(r'[^\w\s]', '', response.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        # Take first 100 chars as fingerprint (catches near-duplicates without
        # being too aggressive - 50 chars caused false positives on short responses)
        return normalized[:100]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences for deduplication."""
        # Split on sentence-ending punctuation followed by space or end of string
        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Normalize each sentence and filter out empty/trivial ones
        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if len(s) > 10:  # Skip very short fragments
                sentences.append(s)
        return sentences

    def _normalize_sentence(self, sentence: str) -> str:
        """Normalize a sentence for comparison (lowercase, no punctuation, collapsed spaces)."""
        normalized = re.sub(r'[^\w\s]', '', sentence.lower())
        return re.sub(r'\s+', ' ', normalized).strip()

    def _has_high_sentence_overlap(self, response: str, threshold: float = 0.5) -> bool:
        """Check if more than threshold fraction of sentences in response are duplicates.

        Compares each sentence in the response against all previously used responses
        stored in _used_responses fingerprints. Uses sentence-level normalized comparison
        to detect near-duplicate paragraphs that differ only in starters/transitions.

        Args:
            response: The candidate response text to check.
            threshold: Fraction of sentences that must be duplicates to flag (default 0.5).

        Returns:
            True if the response has too many duplicate sentences.
        """
        sentences = self._split_into_sentences(response)
        if not sentences:
            return False

        duplicate_count = 0
        for sentence in sentences:
            norm = self._normalize_sentence(sentence)
            if norm and norm in ComprehensiveResponseGenerator._used_sentences:
                duplicate_count += 1

        overlap_ratio = duplicate_count / len(sentences)
        return overlap_ratio > threshold

    def _vary_single_sentence(self, sentence: str, local_rng: random.Random) -> str:
        """Apply variation to a single sentence to make it unique.

        Tries multiple strategies: synonym substitution, starter change,
        qualifier addition, and structural tweaks.

        Args:
            sentence: The sentence to vary.
            local_rng: Seeded RNG for deterministic variation.

        Returns:
            A varied version of the sentence.
        """
        strategies = [0, 1, 2, 3]
        local_rng.shuffle(strategies)

        for strategy in strategies:
            if strategy == 0:
                # Add or change sentence starter
                starter = local_rng.choice(self.SENTENCE_STARTERS)
                # Remove existing starter if present
                stripped = sentence
                for existing_starter in self.SENTENCE_STARTERS:
                    if sentence.lower().startswith(existing_starter.lower()):
                        stripped = sentence[len(existing_starter):]
                        break
                if stripped:
                    varied = starter + stripped[0].lower() + stripped[1:]
                    norm = self._normalize_sentence(varied)
                    if norm not in ComprehensiveResponseGenerator._used_sentences:
                        return varied

            elif strategy == 1:
                # Add a qualifier prefix
                qualifiers = [
                    "Honestly, ", "In truth, ", "Frankly, ",
                    "To be fair, ", "Realistically, ", "Genuinely, ",
                ]
                qualifier = local_rng.choice(qualifiers)
                varied = qualifier + sentence[0].lower() + sentence[1:] if sentence else sentence
                norm = self._normalize_sentence(varied)
                if norm not in ComprehensiveResponseGenerator._used_sentences:
                    return varied

            elif strategy == 2:
                # Synonym substitution within the sentence
                synonym_map = {
                    'good': ['fine', 'decent', 'solid'],
                    'bad': ['poor', 'inadequate', 'subpar'],
                    'very': ['quite', 'really', 'particularly'],
                    'interesting': ['intriguing', 'noteworthy', 'compelling'],
                    'important': ['significant', 'crucial', 'essential'],
                }
                varied = sentence
                for word, synonyms in synonym_map.items():
                    pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                    if pattern.search(varied):
                        replacement = local_rng.choice(synonyms)
                        varied = pattern.sub(replacement, varied, count=1)
                        break
                if varied != sentence:
                    norm = self._normalize_sentence(varied)
                    if norm not in ComprehensiveResponseGenerator._used_sentences:
                        return varied

            elif strategy == 3:
                # Rephrase ending
                suffixes = [
                    ", I think", ", in my view", ", as I see it",
                    ", from my perspective", ", to my mind",
                ]
                base = sentence.rstrip('.!?,;')
                suffix = local_rng.choice(suffixes)
                varied = base + suffix + '.'
                norm = self._normalize_sentence(varied)
                if norm not in ComprehensiveResponseGenerator._used_sentences:
                    return varied

        # Fallback: just prepend a unique hedge word
        hedges = ["Essentially, ", "Basically, ", "Fundamentally, ", "Simply put, ", "Put simply, "]
        hedge = local_rng.choice(hedges)
        return hedge + sentence[0].lower() + sentence[1:] if sentence else sentence

    def _enforce_sentence_uniqueness(self, response: str, local_rng: random.Random) -> str:
        """Final pass: replace any sentence that has already been used in the dataset.

        Splits the response into sentences, checks each against the class-level
        _used_sentences set, and replaces duplicates with varied versions. This
        is the KEY mechanism that prevents identical sentences from appearing
        across different responses in the same dataset.

        Args:
            response: The full response text.
            local_rng: Seeded RNG for deterministic variation.

        Returns:
            Response with all duplicate sentences replaced by unique variants.
        """
        sentences = self._split_into_sentences(response)
        if not sentences:
            return response

        result_sentences: List[str] = []
        for sentence in sentences:
            norm = self._normalize_sentence(sentence)
            if norm and norm in ComprehensiveResponseGenerator._used_sentences:
                # This sentence was already used - replace with a varied version
                varied = self._vary_single_sentence(sentence, local_rng)
                varied_norm = self._normalize_sentence(varied)
                ComprehensiveResponseGenerator._used_sentences.add(varied_norm)
                result_sentences.append(varied)
            else:
                # Sentence is unique - register it and keep as-is
                if norm:
                    ComprehensiveResponseGenerator._used_sentences.add(norm)
                result_sentences.append(sentence)

        return ' '.join(result_sentences)

    def _ensure_unique_response(self, response: str, local_rng: random.Random, max_attempts: int = 8) -> str:
        """Ensure response is unique within the dataset by varying if needed.

        Uses two-tier deduplication:
        1. Fingerprint check (first 100 chars normalized) for exact/near-exact duplicates
        2. Sentence-level overlap check (>50% shared sentences) for paragraph-level duplicates

        Args:
            response: The candidate response text.
            local_rng: Seeded RNG for deterministic variation.
            max_attempts: Maximum variation attempts before fallback (default 8).

        Returns:
            A unique response string.
        """
        fingerprint = self._get_response_fingerprint(response)

        # Check both fingerprint uniqueness AND sentence-level overlap
        is_fingerprint_dup = fingerprint in ComprehensiveResponseGenerator._used_responses
        is_sentence_dup = self._has_high_sentence_overlap(response)

        if not is_fingerprint_dup and not is_sentence_dup:
            ComprehensiveResponseGenerator._used_responses.add(fingerprint)
            # Register individual sentences for future overlap checks
            for sentence in self._split_into_sentences(response):
                norm = self._normalize_sentence(sentence)
                if norm:
                    ComprehensiveResponseGenerator._used_sentences.add(norm)
            return response

        # Response is a duplicate (by fingerprint or sentence overlap) - vary it
        for attempt in range(max_attempts):
            varied = self._vary_response(response, local_rng, attempt)
            new_fingerprint = self._get_response_fingerprint(varied)
            new_is_fingerprint_dup = new_fingerprint in ComprehensiveResponseGenerator._used_responses
            new_is_sentence_dup = self._has_high_sentence_overlap(varied)
            if not new_is_fingerprint_dup and not new_is_sentence_dup:
                ComprehensiveResponseGenerator._used_responses.add(new_fingerprint)
                for sentence in self._split_into_sentences(varied):
                    norm = self._normalize_sentence(sentence)
                    if norm:
                        ComprehensiveResponseGenerator._used_sentences.add(norm)
                return varied

        # Fallback: add unique suffix
        unique_suffix = local_rng.choice([
            " That's my honest view.", " That reflects my thinking.",
            " That's what I genuinely feel.", " Those are my real thoughts.",
            " I wanted to share that perspective.", " That's my actual experience.",
        ])
        final = response.rstrip('.!?') + '.' + unique_suffix
        ComprehensiveResponseGenerator._used_responses.add(self._get_response_fingerprint(final))
        for sentence in self._split_into_sentences(final):
            norm = self._normalize_sentence(sentence)
            if norm:
                ComprehensiveResponseGenerator._used_sentences.add(norm)
        return final

    def _vary_response(self, response: str, local_rng: random.Random, variation_level: int = 0) -> str:
        """Apply variation to make a response unique while preserving meaning."""
        # Level 0: Change sentence starter
        if variation_level == 0:
            starter = local_rng.choice(self.SENTENCE_STARTERS)
            # Remove existing starter if present
            for existing_starter in self.SENTENCE_STARTERS:
                if response.lower().startswith(existing_starter.lower()):
                    response = response[len(existing_starter):]
                    break
            return starter + response[0].lower() + response[1:] if response else response

        # Level 1: Add transition phrase in middle
        elif variation_level == 1:
            sentences = response.split('. ')
            if len(sentences) >= 2:
                transition = local_rng.choice(self.TRANSITION_PHRASES)
                sentences[1] = transition.lower() + sentences[1][0].lower() + sentences[1][1:] if sentences[1] else sentences[1]
                return '. '.join(sentences)

        # Level 2: Change concluding phrase
        elif variation_level == 2:
            concluder = local_rng.choice(self.CONCLUDING_PHRASES)
            return response.rstrip('.!?') + '. ' + concluder

        # Level 3: Reorder clauses if possible
        elif variation_level == 3:
            if ' because ' in response.lower():
                parts = response.split(' because ', 1)
                if len(parts) == 2:
                    return parts[1].capitalize().rstrip('.') + ', which is why ' + parts[0].lower()
            elif ' but ' in response.lower():
                parts = response.split(' but ', 1)
                if len(parts) == 2:
                    return 'While ' + parts[1].rstrip('.') + ', ' + parts[0].lower()

        # Level 4: Add qualifier
        elif variation_level == 4:
            qualifiers = [
                "To be honest, ", "Thinking about it, ", "Upon reflection, ",
                "When I consider this, ", "Given the circumstances, ",
            ]
            return local_rng.choice(qualifiers) + response[0].lower() + response[1:]

        # Level 5: Synonym substitution for common adjectives/adverbs
        elif variation_level == 5:
            synonym_map: Dict[str, List[str]] = {
                'good': ['fine', 'decent', 'solid', 'positive'],
                'bad': ['poor', 'inadequate', 'subpar', 'lacking'],
                'very': ['quite', 'really', 'particularly', 'rather'],
                'interesting': ['intriguing', 'noteworthy', 'thought-provoking', 'compelling'],
                'important': ['significant', 'crucial', 'essential', 'key'],
                'great': ['excellent', 'outstanding', 'remarkable', 'impressive'],
                'difficult': ['challenging', 'tough', 'demanding', 'complex'],
                'easy': ['straightforward', 'simple', 'manageable', 'uncomplicated'],
                'big': ['large', 'substantial', 'considerable', 'major'],
                'small': ['minor', 'slight', 'modest', 'limited'],
            }
            varied = response
            for word, synonyms in synonym_map.items():
                # Match whole words only (case-insensitive)
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                if pattern.search(varied):
                    replacement = local_rng.choice(synonyms)
                    # Preserve original capitalization of first char
                    def _replace_preserving_case(match: re.Match) -> str:
                        original = match.group(0)
                        if original[0].isupper():
                            return replacement[0].upper() + replacement[1:]
                        return replacement
                    varied = pattern.sub(_replace_preserving_case, varied, count=1)
            return varied

        # Level 6: Add personal anecdote/experience reference phrase
        elif variation_level == 6:
            anecdote_phrases = [
                "In my own experience with similar situations, ",
                "This reminds me of when I encountered something like this. ",
                "I've dealt with something like this before and ",
                "From what I've personally gone through, ",
                "Relating this to my own life, ",
                "Having faced comparable circumstances, ",
                "Drawing from my own experience, ",
                "I can relate to this because ",
            ]
            phrase = local_rng.choice(anecdote_phrases)
            # Insert at the beginning, adjusting case of the original start
            if response:
                return phrase + response[0].lower() + response[1:]
            return response

        # Level 7: Restructure sentences (split one into two, or combine two into one)
        elif variation_level == 7:
            sentences = self._split_into_sentences(response)
            if len(sentences) == 1 and len(sentences[0]) > 40:
                # Split a long sentence into two at a natural break point
                text = sentences[0]
                # Try splitting at conjunctions/commas
                split_patterns = [', and ', ', but ', ', which ', ', so ', ', yet ']
                for pattern in split_patterns:
                    if pattern in text.lower():
                        idx = text.lower().index(pattern)
                        first_part = text[:idx].rstrip(',') + '.'
                        second_part = text[idx + len(pattern):].strip()
                        if second_part:
                            second_part = second_part[0].upper() + second_part[1:]
                            return first_part + ' ' + second_part
                        break
            elif len(sentences) >= 2:
                # Combine first two short sentences into one compound sentence
                s1 = sentences[0].rstrip('.!?')
                s2 = sentences[1]
                if s2:
                    s2_lower = s2[0].lower() + s2[1:] if s2 else s2
                    connectors = [', and ', ', and I think ', '. Moreover, ', ' - and furthermore, ']
                    connector = local_rng.choice(connectors)
                    combined = s1 + connector + s2_lower
                    remaining = sentences[2:]
                    if remaining:
                        return combined + ' ' + ' '.join(remaining)
                    return combined

        return response

    def generate(
        self,
        question_text: str,
        sentiment: str = "neutral",  # very_positive, positive, neutral, negative, very_negative
        persona_verbosity: float = 0.5,
        persona_formality: float = 0.5,
        persona_engagement: float = 0.5,
        condition: str = "",
        question_name: str = "",
        participant_seed: int = 0,
    ) -> str:
        """
        Generate a context-appropriate response to an open-ended question.

        v1.0.0: CRITICAL FIX - Each question now gets a unique response using
        a per-question seeded RNG. This ensures the same participant gives
        DIFFERENT responses to DIFFERENT questions.

        Args:
            question_text: The question being answered
            sentiment: Overall sentiment of the response
            persona_verbosity: How verbose the persona is (0-1)
            persona_formality: How formal the persona is (0-1)
            persona_engagement: How engaged the persona is (0-1)
            condition: Experimental condition (for context)
            question_name: Unique identifier for the question (CRITICAL for uniqueness)
            participant_seed: Base random seed for this participant

        Returns:
            Generated response text
        """
        # v1.0.0 CRITICAL FIX: Create a UNIQUE seed using STABLE hash
        # Python's hash() can vary between runs due to hash randomization
        # Use a deterministic hash based on character ordinals
        combined_id = f"{question_name}|{question_text}"
        if combined_id:
            # Stable hash that will be the same across Python runs
            question_hash_stable = sum(ord(c) * (i + 1) * 31 for i, c in enumerate(combined_id[:200]))
        else:
            question_hash_stable = 0
        unique_seed = (participant_seed + question_hash_stable) % (2**31)

        # Create a LOCAL random generator for this specific question
        # This is CRITICAL - we do NOT use the global random module
        local_rng = random.Random(unique_seed)

        # Detect question type and domain
        q_type = detect_question_type(question_text)
        domain = detect_study_domain(self.study_context, question_text)

        # Extract keywords from the question text for unique response generation
        question_keywords = self._extract_question_keywords(question_text)

        # Get appropriate template (using local RNG)
        response = self._get_template_response(domain, q_type, sentiment, local_rng)

        # Personalize response based on question content (using local RNG)
        response = self._personalize_for_question(response, question_text, question_keywords, condition, local_rng)

        # Handle disengaged/careless personas
        if persona_engagement < 0.3:
            response = self._make_careless(response, persona_engagement, local_rng)

        # Add variation (using local RNG)
        response = add_variation(response, persona_verbosity, persona_formality, local_rng)

        # Adjust length based on verbosity
        if persona_verbosity < 0.3:
            response = self._shorten(response, local_rng)
        elif persona_verbosity > 0.7:
            response = self._extend(response, domain, sentiment, local_rng)

        # v1.1.0: Add topic-specific context to keep response on-topic
        response = self._add_topic_context(response, question_text, question_keywords, domain, local_rng)

        # v1.0.0: Ensure we never return an empty response
        if not response or not response.strip():
            fallback_responses = [
                "I found this interesting to consider.",
                "This made me think about my experience.",
                "I had some thoughts about this.",
                "Overall, it was a reasonable experience.",
                "I appreciated the opportunity to reflect on this.",
            ]
            response = local_rng.choice(fallback_responses)

        # v1.1.0: Ensure response is unique within the dataset
        response = self._ensure_unique_response(response, local_rng)

        # v1.3.0: Final sentence-level uniqueness check
        # Split response into sentences. For each sentence already in
        # _used_sentences, replace it with a varied version. This is the KEY
        # improvement that prevents the exact same sentences from appearing
        # across different responses in a dataset.
        response = self._enforce_sentence_uniqueness(response, local_rng)

        return response

    def _add_topic_context(
        self,
        response: str,
        question_text: str,
        keywords: List[str],
        domain: StudyDomain,
        local_rng: random.Random
    ) -> str:
        """Add topic-specific context to ensure response stays on-topic.

        v1.1.0: Ensures responses are always relevant to the survey topic.
        v1.2.0: Enhanced with 80+ topic categories and academic phrasing.
        """
        if not keywords or local_rng.random() > 0.4:  # 40% chance to add context
            return response

        # Build topic phrases based on extracted keywords (v1.2.0: greatly expanded)
        topic_phrases = {
            # Research-related keywords
            'study': ['in this study', 'for this research', 'in this survey', 'as part of this study', 'within this research context'],
            'research': ['from a research perspective', 'considering the research', 'in this research setting', 'given the research focus'],
            'experiment': ['in this experiment', 'during the experiment', 'within this experimental context', 'given the experimental setup'],
            'survey': ['in this survey', 'for the survey', 'responding to this survey', 'in answering this survey'],
            'question': ['regarding this question', 'in response to this', 'when considering this question', 'to address this'],

            # Decision-related keywords
            'decision': ['regarding my decision', 'about choosing', 'when deciding', 'in the decision-making process', 'as I weighed the options'],
            'choice': ['with this choice', 'in making this choice', 'when faced with this choice', 'considering my options'],
            'option': ['considering the options', 'between the options', 'evaluating the alternatives', 'weighing the options presented'],
            'tradeoff': ['considering the tradeoffs', 'balancing these factors', 'weighing the costs and benefits'],
            'preference': ['based on my preferences', 'given what I prefer', 'considering my personal preferences'],

            # Evaluation keywords
            'product': ['about this product', 'for this item', 'regarding what was shown', 'evaluating this product', 'concerning this offering'],
            'service': ['about the service', 'regarding the service quality', 'in terms of service', 'evaluating the service provided'],
            'brand': ['about the brand', 'regarding this brand', 'in terms of brand perception', 'considering brand attributes'],
            'quality': ['in terms of quality', 'regarding quality aspects', 'considering the quality', 'evaluating quality'],
            'value': ['in terms of value', 'considering the value proposition', 'regarding perceived value'],
            'price': ['considering the price', 'given the cost', 'in terms of pricing', 'regarding the price point'],
            'purchase': ['regarding this purchase', 'in terms of buying', 'considering a purchase', 'when thinking about buying'],

            # Experience keywords
            'experience': ['from my experience', 'based on my experience', 'drawing on my experience', 'reflecting on past experiences'],
            'scenario': ['in this scenario', 'given the situation', 'under these circumstances', 'in this hypothetical situation'],
            'situation': ['in this situation', 'given these circumstances', 'under these conditions', 'facing this situation'],
            'context': ['in this context', 'given the context', 'within this particular context', 'considering the context'],

            # Social/behavioral keywords
            'people': ['when thinking about others', 'considering other people', 'in social situations', 'regarding how people behave'],
            'trust': ['regarding trust', 'about trusting', 'in terms of trustworthiness', 'when it comes to trust'],
            'fair': ['about fairness', 'regarding what seems fair', 'in terms of equity', 'considering fairness'],
            'risk': ['considering the risk', 'about the uncertainty', 'when evaluating risk', 'regarding potential risks'],
            'social': ['in social terms', 'from a social perspective', 'considering social dynamics', 'regarding social aspects'],
            'cooperation': ['regarding cooperation', 'in terms of working together', 'considering collaborative aspects'],
            'norm': ['considering social norms', 'regarding expectations', 'in terms of what is expected'],

            # Technology keywords
            'technology': ['about technology', 'regarding tech', 'concerning technology use', 'in terms of technology'],
            'ai': ['about AI', 'regarding artificial intelligence', 'concerning AI systems', 'when it comes to AI'],
            'algorithm': ['about algorithms', 'regarding automated systems', 'concerning algorithmic decisions', 'in terms of automation'],
            'privacy': ['regarding privacy', 'concerning data privacy', 'in terms of privacy protection', 'about personal information'],
            'data': ['concerning data', 'regarding data use', 'in terms of information handling', 'about data practices'],
            'online': ['in online contexts', 'regarding digital interactions', 'concerning online behavior'],
            'digital': ['in the digital realm', 'regarding digital experiences', 'concerning digital platforms'],

            # Health and wellbeing keywords
            'health': ['regarding health', 'concerning wellbeing', 'in terms of health outcomes', 'about health-related matters'],
            'wellbeing': ['concerning wellbeing', 'regarding quality of life', 'in terms of overall wellness'],
            'stress': ['regarding stress', 'concerning anxiety levels', 'in terms of mental load', 'about stress management'],
            'emotion': ['regarding emotions', 'concerning feelings', 'in emotional terms', 'about emotional responses'],
            'feeling': ['about my feelings', 'regarding how I feel', 'in terms of emotional reactions'],

            # Work and organizational keywords
            'work': ['in the workplace', 'regarding work situations', 'concerning professional contexts', 'at work'],
            'job': ['regarding my job', 'concerning work tasks', 'in terms of job responsibilities'],
            'team': ['regarding team dynamics', 'concerning group work', 'in terms of collaboration'],
            'leader': ['regarding leadership', 'concerning management', 'in terms of leadership qualities'],
            'organization': ['within organizations', 'regarding organizational matters', 'in institutional contexts'],

            # Environmental keywords
            'environment': ['regarding the environment', 'concerning environmental issues', 'in environmental terms'],
            'climate': ['concerning climate change', 'regarding climate issues', 'about environmental sustainability'],
            'sustain': ['regarding sustainability', 'concerning sustainable practices', 'in terms of long-term impact'],

            # Political/civic keywords
            'politics': ['regarding political matters', 'concerning civic issues', 'in political terms'],
            'policy': ['regarding policy', 'concerning this policy issue', 'in terms of policy implications'],
            'government': ['regarding government', 'concerning public institutions', 'about governmental action'],
            'vote': ['regarding voting', 'concerning electoral choices', 'in terms of political participation'],

            # Education keywords
            'learn': ['regarding learning', 'concerning education', 'in terms of knowledge acquisition'],
            'education': ['concerning education', 'regarding educational experiences', 'in academic contexts'],
            'student': ['as a learner', 'from a student perspective', 'regarding educational experiences'],

            # Ethics and morality keywords
            'moral': ['from a moral standpoint', 'regarding ethical considerations', 'in terms of what is right'],
            'ethic': ['regarding ethics', 'concerning ethical principles', 'from an ethical perspective'],
            'right': ['regarding what is right', 'concerning proper conduct', 'in terms of correctness'],
            'wrong': ['regarding what seems wrong', 'concerning inappropriate actions', 'in terms of problematic aspects'],
            'responsibility': ['regarding responsibility', 'concerning accountability', 'in terms of obligations'],

            # Financial keywords
            'money': ['regarding money', 'concerning financial matters', 'in financial terms'],
            'finance': ['regarding finances', 'concerning financial decisions', 'in terms of monetary considerations'],
            'invest': ['regarding investment', 'concerning financial planning', 'in terms of investing'],
            'save': ['regarding savings', 'concerning financial security', 'in terms of saving money'],
            'spend': ['regarding spending', 'concerning purchases', 'in terms of expenditure'],

            # Relationship keywords
            'relationship': ['regarding relationships', 'concerning interpersonal connections', 'in terms of relating to others'],
            'friend': ['regarding friendship', 'concerning social bonds', 'in terms of friendly relations'],
            'family': ['regarding family', 'concerning family matters', 'in terms of family relationships'],
            'partner': ['regarding my partner', 'concerning romantic relationships', 'in terms of partnerships'],

            # Identity and self keywords
            'identity': ['regarding identity', 'concerning self-concept', 'in terms of who I am'],
            'self': ['regarding myself', 'concerning personal aspects', 'in terms of self-perception'],
            'personal': ['on a personal level', 'regarding personal matters', 'in terms of individual experience'],
            'belief': ['regarding my beliefs', 'concerning what I believe', 'in terms of my convictions'],
            'value': ['regarding my values', 'concerning what I value', 'in terms of personal principles'],
        }

        # Find matching topic phrase
        for keyword in keywords[:3]:  # Check top 3 keywords
            keyword_lower = keyword.lower()
            for topic_key, phrases in topic_phrases.items():
                if topic_key in keyword_lower or keyword_lower in topic_key:
                    phrase = local_rng.choice(phrases)
                    # Insert phrase naturally into response
                    if local_rng.random() < 0.5:
                        # Add at beginning
                        return f"{phrase.capitalize()}, {response[0].lower()}{response[1:]}"
                    else:
                        # Add before last sentence
                        sentences = response.rsplit('. ', 1)
                        if len(sentences) == 2:
                            return f"{sentences[0]}. {phrase.capitalize()}, {sentences[1][0].lower()}{sentences[1][1:]}"
                    break

        return response

    def _extract_question_keywords(self, question_text: str) -> List[str]:
        """Extract meaningful keywords from question text for response customization."""
        if not question_text:
            return []

        # Common stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why',
            'how', 'what', 'which', 'who', 'whom', 'whose', 'that', 'this',
            'these', 'those', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'up', 'about', 'into', 'over', 'after', 'your', 'you', 'our',
            'please', 'describe', 'explain', 'tell', 'us', 'me', 'briefly', 'any',
        }

        # Extract words (3+ characters, alphabetic)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', question_text.lower())
        keywords = [w for w in words if w not in stop_words]

        # Return unique keywords, prioritizing longer words
        unique_keywords = list(dict.fromkeys(keywords))
        return sorted(unique_keywords, key=len, reverse=True)[:5]

    def _personalize_for_question(
        self,
        response: str,
        question_text: str,
        keywords: List[str],
        condition: str,
        local_rng: random.Random = None
    ) -> str:
        """Personalize the response based on question content and condition.

        v1.0.0: Uses local RNG for deterministic, question-specific personalization.
        """
        rng = local_rng or random.Random()

        if not keywords and not condition:
            return response

        # Build context-specific phrases based on question keywords
        question_lower = question_text.lower() if question_text else ""

        # Add topic-specific introductions based on keywords
        topic_intros = {
            'decision': ['When making this decision, ', 'Regarding my decision, '],
            'product': ['About the product, ', 'For this product, '],
            'experience': ['Based on my experience, ', 'From what I experienced, '],
            'feeling': ['I felt that ', 'My feeling was that '],
            'opinion': ['In my opinion, ', 'I think that '],
            'task': ['For this task, ', 'While doing this task, '],
            'choice': ['With my choice, ', 'Regarding my choice, '],
            'recommend': ['I would recommend ', 'My recommendation is '],
            'improve': ['To improve, ', 'For improvement, '],
            'scenario': ['In this scenario, ', 'Given the scenario, '],
            'purchase': ['For this purchase, ', 'Regarding buying, '],
            'service': ['About the service, ', 'For the service, '],
        }

        for keyword, intros in topic_intros.items():
            if keyword in question_lower or any(kw == keyword for kw in keywords):
                intro = rng.choice(intros)
                if not response.lower().startswith(intro.lower().split()[0]):
                    response = intro + response[0].lower() + response[1:]
                break

        # Add condition-specific context
        condition_lower = condition.lower() if condition else ""
        if 'ai' in condition_lower and 'no ai' not in condition_lower:
            ai_phrases = [
                " The AI aspect was interesting.",
                " Considering the AI involvement.",
                "",  # Sometimes no addition
            ]
            if rng.random() < 0.3:
                response += rng.choice(ai_phrases)
        elif 'hedonic' in condition_lower:
            if rng.random() < 0.3:
                response += " The enjoyment factor was notable."
        elif 'utilitarian' in condition_lower:
            if rng.random() < 0.3:
                response += " The practical aspects mattered."

        return response

    def _get_template_response(
        self,
        domain: StudyDomain,
        q_type: QuestionType,
        sentiment: str,
        local_rng: random.Random = None
    ) -> str:
        """Get a template response for the given domain and sentiment.

        v1.0.0: Uses local RNG for deterministic, question-specific selection.
        """
        rng = local_rng or random.Random()
        domain_key = domain.value

        # Try to find domain-specific templates
        if domain_key in DOMAIN_TEMPLATES:
            templates = DOMAIN_TEMPLATES[domain_key]
            if "explanation" in templates:  # Most common question type
                sentiment_templates = templates["explanation"].get(sentiment, templates["explanation"].get("neutral", []))
                if sentiment_templates:
                    return rng.choice(sentiment_templates)

        # Fall back to general templates
        if "general" in DOMAIN_TEMPLATES:
            templates = DOMAIN_TEMPLATES["general"]["explanation"]
            sentiment_templates = templates.get(sentiment, templates.get("neutral", []))
            if sentiment_templates:
                return rng.choice(sentiment_templates)

        return "No specific comment."

    def _make_careless(self, response: str, engagement: float, local_rng: random.Random = None) -> str:
        """Transform response to reflect careless/disengaged responding.

        v1.0.0: Uses local RNG for deterministic selection.
        """
        rng = local_rng or random.Random()
        careless_responses = [
            "ok", "fine", "idk", "whatever", "no reason",
            "just because", "not sure", "dont know", "didnt think about it",
            "i guess so", "no comment", "n/a", "nothing really",
        ]

        if engagement < 0.2:
            return rng.choice(careless_responses)
        elif engagement < 0.3:
            # Very short version
            words = response.split()[:3]
            return ' '.join(words).rstrip('.,!?') + '.'
        return response

    def _shorten(self, response: str, local_rng: random.Random = None) -> str:
        """Shorten a response for low-verbosity personas.

        v1.0.0: Added local_rng parameter for consistency (unused but keeps interface uniform).
        """
        sentences = response.split('.')
        if len(sentences) > 1:
            return sentences[0].strip() + '.'
        words = response.split()
        if len(words) > 10:
            return ' '.join(words[:8]).rstrip('.,!?') + '.'
        return response

    def _extend(self, response: str, domain: StudyDomain, sentiment: str, local_rng: random.Random = None) -> str:
        """Extend a response for high-verbosity personas with domain awareness.

        v1.0.0: Uses local RNG for deterministic, question-specific extensions.
        """
        rng = local_rng or random.Random()

        # Domain-specific extensions
        domain_extensions = {
            StudyDomain.DICTATOR_GAME: {
                "positive": [" Fairness is important to me.", " I tried to be reasonable."],
                "negative": [" The situation felt unfair.", " I wasn't sure what was right."],
            },
            StudyDomain.RISK_PREFERENCE: {
                "positive": [" I'm comfortable with some uncertainty.", " Risk doesn't bother me much."],
                "negative": [" I prefer to play it safe.", " Too much uncertainty makes me nervous."],
            },
            StudyDomain.TRUST_GAME: {
                "positive": [" I generally try to give people the benefit of the doubt."],
                "negative": [" Trust has to be earned.", " I'm cautious about trusting others."],
            },
            StudyDomain.CONSUMER_BEHAVIOR: {
                "positive": [" I enjoy shopping for things like this.", " Quality matters to me."],
                "negative": [" I'm picky about what I buy.", " Value for money is important."],
            },
            StudyDomain.AI_ATTITUDES: {
                "positive": [" Technology can be really helpful.", " I'm open to new approaches."],
                "negative": [" I prefer human judgment.", " AI isn't always the answer."],
            },
        }

        # General extensions
        general_extensions = {
            "very_positive": [
                " I feel good about this overall.",
                " This was a positive experience.",
                " I'm glad to share my perspective.",
                " This resonated with me.",
            ],
            "positive": [
                " That's my take on it.",
                " I think that covers the main points.",
                " Hopefully that makes sense.",
                " That's how I see it anyway.",
            ],
            "neutral": [
                " I could go either way on this.",
                " It's hard to say for certain.",
                " I see both sides.",
                " There are trade-offs.",
            ],
            "negative": [
                " There's room for improvement.",
                " I expected better.",
                " This could be handled differently.",
                " Something needs to change.",
            ],
            "very_negative": [
                " This really bothered me.",
                " I have serious concerns.",
                " This was frustrating.",
                " I'm disappointed.",
            ],
        }

        # Try domain-specific extension first
        if domain in domain_extensions:
            sentiment_key = "positive" if sentiment in ["very_positive", "positive"] else "negative"
            domain_ext = domain_extensions[domain].get(sentiment_key, [])
            if domain_ext and rng.random() < 0.5:
                return response + rng.choice(domain_ext)

        # Fall back to general extensions
        extension_list = general_extensions.get(sentiment, general_extensions["neutral"])
        return response + rng.choice(extension_list)


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
