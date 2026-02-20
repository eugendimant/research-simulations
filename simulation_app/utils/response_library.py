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
- Template-based text generation with domain-specific response banks
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

Version: 1.8.5 - Improved domain detection with weighted scoring and disambiguation
"""

__version__ = "1.2.0.5"

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
                "I shared my honest views and I feel pretty good about the topic overall.",
                "I've thought about this a lot and I'm optimistic about where things are heading.",
                "This is something I care about and my perspective is generally positive.",
                "I gave thoughtful answers based on my genuine beliefs. I feel strongly about this.",
                "I reflected carefully and shared my actual opinions. I have positive feelings here.",
            ],
            "positive": [
                "I gave my honest thoughts. Generally feel pretty good about this.",
                "My responses reflect how I actually feel. Mostly positive overall.",
                "I answered based on my real opinions and experiences.",
                "Shared my genuine perspective on this. It's something I've considered before.",
                "I reflected on my views and answered honestly.",
            ],
            "neutral": [
                "I shared my honest thoughts. I don't feel too strongly one way or another.",
                "Gave my genuine reactions. Pretty middle of the road for me.",
                "My answers reflect my actual views. Not strongly positive or negative.",
                "I answered honestly. This isn't something I have extreme opinions about.",
                "Shared my perspective. I can see both sides of this.",
            ],
            "negative": [
                "I shared my honest concerns. There are real issues here in my opinion.",
                "My perspective on this is fairly critical. I answered honestly about that.",
                "I have some real reservations and tried to express them accurately.",
                "Honestly I'm not very positive about this and my answers reflect that.",
                "I expressed my genuine frustrations. This is something that concerns me.",
            ],
            "very_negative": [
                "I feel strongly negative about this and my answers reflect that honestly.",
                "I have serious concerns and I didn't hold back in expressing them.",
                "This is something I feel frustrated about and I answered accordingly.",
                "My perspective is very critical. I think there are major problems here.",
                "I expressed genuine dissatisfaction. There's a lot that needs to change.",
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
                "Social media has genuinely enriched my life by connecting me with communities I never would have found otherwise.",
                "I love how social media lets me stay in touch with friends and family spread across the world.",
                "Online platforms have given me access to diverse perspectives and ideas that broadened my worldview.",
                "I find social media to be a powerful tool for staying informed, sharing experiences, and building real relationships.",
                "Social media has opened doors for me professionally and personally in ways I deeply appreciate.",
            ],
            "positive": [
                "I find social media useful for keeping up with friends and staying informed about things I care about.",
                "Social media helps me maintain connections that would otherwise fade, which I value.",
                "I enjoy scrolling through my feeds and seeing what people are up to, though I try to be mindful about it.",
                "Online platforms are a good way to discover new interests, follow news, and stay connected.",
                "Social media has more benefits than drawbacks in my experience, as long as I manage my time on it.",
            ],
            "neutral": [
                "I have genuinely mixed feelings about social media because I see both the benefits and the downsides.",
                "Social media is something I use out of habit more than enthusiasm at this point.",
                "I can see how social media connects people, but I also notice how it can make me feel worse sometimes.",
                "I'm pretty indifferent about social media and just use it when it's convenient.",
                "Social media is just part of modern life now, and I neither love nor hate it.",
            ],
            "negative": [
                "I find social media more draining than enriching, and I've been trying to cut back.",
                "Scrolling through social media often leaves me feeling worse than before I started.",
                "I have real concerns about how social media affects attention spans, mental health, and real-world relationships.",
                "Social media platforms are designed to be addictive, and I resent how much time they take from me.",
                "I've noticed social media making me more anxious and more prone to comparing myself to others.",
            ],
            "very_negative": [
                "Social media is genuinely toxic and I believe it's damaging our ability to communicate and relate to each other.",
                "I've significantly cut back on social media because of how negatively it was affecting my mental health.",
                "The spread of misinformation, outrage, and performative behavior on social media is deeply harmful to society.",
                "I strongly dislike how social media platforms exploit our attention and manipulate our emotions for profit.",
                "Social media has made people lonelier, angrier, and more divided, and I try to avoid it as much as possible.",
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
                "I feel very confident about this financial decision because I did my research and weighed all the options carefully.",
                "I made a thorough analysis of the costs, benefits, and risks before committing, and I feel good about it.",
                "This financial choice aligns perfectly with my long-term goals and risk tolerance.",
                "I consulted multiple sources and crunched the numbers, so I'm genuinely satisfied with my decision.",
                "I believe this was the smartest financial move I could have made given the circumstances.",
            ],
            "positive": [
                "I think this was a solid financial decision, even if I didn't analyze every possible angle.",
                "I weighed the main costs and benefits and feel reasonably good about where I ended up.",
                "My financial choice seems sound based on what I know, though nothing is ever certain with money.",
                "I'm satisfied with this decision because it balanced my financial security with my goals.",
                "I considered the financial implications carefully enough to feel comfortable moving forward.",
            ],
            "neutral": [
                "I'm honestly not sure if this was the best financial choice, but it seemed reasonable at the time.",
                "Financial decisions always involve uncertainty, and I could see this going either way.",
                "I made what felt like a middle-ground financial decision without overthinking or underthinking it.",
                "I have mixed feelings about this financial choice because the trade-offs weren't clear-cut.",
                "Time will tell whether this was a wise financial move or not, and I'm okay with that uncertainty.",
            ],
            "negative": [
                "Looking back, I have some regrets about this financial decision and wish I had explored more options.",
                "I feel like I let emotional impulses drive this financial choice rather than careful analysis.",
                "This wasn't my best financial moment, and I realize I didn't fully consider the risks.",
                "I'm somewhat worried about the financial consequences of this decision and wish I could undo it.",
                "I probably should have been more patient and done more research before making this financial commitment.",
            ],
            "very_negative": [
                "I deeply regret this financial decision and it has caused me significant stress and worry.",
                "This was a serious financial mistake that I should have avoided, and the consequences are real.",
                "I feel terrible about this choice because I ignored warning signs and acted impulsively with my money.",
                "This financial decision has set me back considerably and I'm angry at myself for making it.",
                "I made a very poor financial judgment and I'm now dealing with the fallout.",
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
            "very_positive": [
                "The task was straightforward and I had plenty of mental capacity to handle everything being asked of me.",
                "I found the cognitive demands perfectly manageable and was able to think clearly throughout the entire task.",
                "My mind handled the complexity well and I felt focused and sharp the whole time.",
                "I didn't feel mentally strained at all, and the information was presented in a way that was easy to process.",
                "I felt mentally capable and engaged, without any sense of being overwhelmed by the task requirements.",
            ],
            "positive": [
                "The mental demands were reasonable and I managed to process everything without too much effort.",
                "I handled the cognitive load adequately, even if there were moments that required closer attention.",
                "The task wasn't too mentally taxing overall, and I had sufficient mental resources to do it well.",
                "I was able to keep track of the important information and the processing was fairly smooth.",
                "The cognitive demands were within my comfort zone, though I needed to concentrate at some points.",
            ],
            "neutral": [
                "The mental demands felt average to me, neither too easy nor particularly overwhelming.",
                "I had a typical level of mental engagement with the task and didn't notice it being especially hard or easy.",
                "The cognitive load was somewhere in the middle, requiring moderate effort to process everything.",
                "I wasn't sure whether the task was mentally demanding or not because it felt fairly standard.",
                "My mental effort was about what I'd expect for a task like this, nothing unusual in either direction.",
            ],
            "negative": [
                "The task was more mentally demanding than I expected and I could feel my concentration fading.",
                "I felt cognitive strain trying to hold multiple pieces of information in mind simultaneously.",
                "My mental resources were stretched thin and I'm not confident I processed everything accurately.",
                "The complexity of the task was genuinely challenging and I had to re-read things multiple times.",
                "I struggled with the mental load because there was too much to keep track of at once.",
            ],
            "very_negative": [
                "I was cognitively overwhelmed by the demands of this task and couldn't keep up.",
                "The mental demands were excessive and I felt like my brain simply shut down at certain points.",
                "I couldn't process everything being asked of me because the cognitive burden was just too high.",
                "My mind was completely overloaded and I'm sure my responses suffered as a result.",
                "The sheer amount of information and complexity made this task mentally exhausting and frustrating.",
            ],
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
            "very_positive": [
                "I genuinely thrive under pressure and find that high-stakes situations bring out my best performance.",
                "Anxiety doesn't affect my performance because I've learned to channel nervous energy into focus and preparation.",
                "Competition energizes me rather than paralyzing me, and I actually feel more alive in pressure situations.",
                "I perform well when it counts because I've developed mental strategies for staying calm and focused.",
                "I've come to welcome performance pressure as something that sharpens my abilities rather than undermining them.",
            ],
            "positive": [
                "I manage performance pressure reasonably well, though I still feel butterflies before important moments.",
                "Some anxiety actually motivates me to prepare more thoroughly, so it works in my favor most of the time.",
                "I usually perform adequately under pressure even though I might not feel totally calm inside.",
                "I've learned to cope with performance stress through practice and experience, and it's gotten easier over time.",
                "Anxiety affects me minimally during actual performance, even if I worry beforehand.",
            ],
            "neutral": [
                "Pressure affects me differently depending on the situation, with some contexts being worse than others.",
                "My response to performance anxiety is inconsistent and I can't always predict how I'll handle it.",
                "I have average ability to handle pressure compared to most people, neither thriving nor crumbling.",
                "Performance situations are genuinely hit or miss for me, and preparation doesn't always determine the outcome.",
                "Anxiety has mixed effects on my performance, sometimes helping and sometimes getting in the way.",
            ],
            "negative": [
                "Performance pressure often affects me negatively, and I notice my skills degrading when the stakes are high.",
                "I struggle with anxiety in competitive or evaluative situations, which is frustrating because I know my abilities.",
                "Pressure tends to hurt my performance because I start overthinking and second-guessing myself.",
                "I get quite nervous before important events and that nervousness carries into my actual performance.",
                "Performance anxiety is a real challenge for me and it has cost me opportunities I was otherwise qualified for.",
            ],
            "very_negative": [
                "Performance anxiety significantly impairs my abilities and I've missed out on a lot because of it.",
                "I struggle greatly under pressure and the physical symptoms of anxiety make it almost impossible to focus.",
                "Competition triggers intense anxiety that I can't control, no matter how much I prepare.",
                "I often choke in important moments because the fear of failure becomes a self-fulfilling prophecy.",
                "Performance situations are so stressful for me that I sometimes avoid them entirely to protect myself.",
            ],
        },
    },

    "team_dynamics": {
        "explanation": {
            "very_positive": [
                "Our team works together excellently and there's a genuine sense of trust and mutual respect among members.",
                "The chemistry within our team is outstanding and we bring out the best in each other's work.",
                "Collaboration flows naturally because everyone contributes their strengths and communicates openly.",
                "I feel genuinely supported by my team members and we hold each other accountable in a positive way.",
                "Team dynamics are a real source of motivation for me and I look forward to working with this group.",
            ],
            "positive": [
                "Our team generally works well together, even though we sometimes disagree on approaches.",
                "The group dynamics are mostly positive and I feel comfortable sharing ideas with the team.",
                "We collaborate effectively most of the time and people pull their weight.",
                "Team members generally support each other and communication is decent.",
                "I'm reasonably satisfied with how our team functions and the results we produce together.",
            ],
            "neutral": [
                "Our team dynamics are fairly average, with some people contributing more than others.",
                "We work together adequately but I wouldn't describe our collaboration as exceptional or terrible.",
                "The group functions at a typical level, with occasional friction but nothing unusual.",
                "Team chemistry is neither strong nor weak, and we get the job done without much drama.",
                "Our dynamics are unremarkable, which I suppose is better than being dysfunctional.",
            ],
            "negative": [
                "Team dynamics could definitely be improved because there are some underlying tensions that affect our work.",
                "There are conflicts within the team that make collaboration harder than it should be.",
                "Communication within our group is lacking and people sometimes work at cross-purposes.",
                "I feel like some team members don't pull their weight, which breeds resentment.",
                "The group has chemistry issues that make meetings and shared projects more stressful than they need to be.",
            ],
            "very_negative": [
                "Team dynamics are genuinely problematic and the conflict within our group makes productive work very difficult.",
                "There's significant interpersonal tension that has eroded trust and morale on the team.",
                "Collaboration feels almost impossible because people are defensive, competitive, or disengaged.",
                "Team chemistry is poor and I dread having to work with this group on shared tasks.",
                "The team functions so poorly that I think we'd actually be more productive working separately.",
            ],
        },
    },

    # ========== LEGAL PSYCHOLOGY ==========

    "legal_psychology": {
        "explanation": {
            "very_positive": [
                "The legal system worked fairly in this case and I feel confident that due process was followed properly.",
                "Justice was served appropriately and the outcome reflects a careful consideration of all the evidence.",
                "I trust the legal process because it gave everyone involved a fair chance to be heard.",
                "The procedures were handled correctly and transparently, which makes me feel the system works.",
                "I came away from this believing that the legal system can deliver just outcomes when it functions properly.",
            ],
            "positive": [
                "The legal system generally worked well here, even if there were minor imperfections in the process.",
                "Justice was mostly served and the outcome seems reasonable given the circumstances.",
                "I have reasonable trust in the legal process based on how this case was handled.",
                "The procedures were adequate and the people involved seemed to take their roles seriously.",
                "Overall the outcome was acceptable and I feel the key issues were addressed.",
            ],
            "neutral": [
                "I have mixed views on how the legal process worked in this case because there were both strengths and weaknesses.",
                "The system worked as expected, which means it was neither inspiring nor particularly disappointing.",
                "My trust in the legal process is average after seeing how things played out here.",
                "The procedures followed standard protocol, but that doesn't necessarily mean the outcome was fair.",
                "I'm not sure whether the legal process produced the right result or not because it's genuinely complicated.",
            ],
            "negative": [
                "The legal process had some real problems that undermine my confidence in the fairness of the outcome.",
                "I don't think justice was fully served because important factors seemed to be overlooked or downplayed.",
                "I have concerns about how the process was conducted, particularly regarding who got to be heard and who didn't.",
                "The procedures could have been handled much better, and the gaps affected the quality of the outcome.",
                "The outcome struck me as somewhat unfair and I'm left wondering if the system really works for everyone.",
            ],
            "very_negative": [
                "The legal system fundamentally failed in this case and I'm deeply troubled by what happened.",
                "Justice was not served and the outcome feels like a miscarriage of the principles the system claims to uphold.",
                "I have lost trust in the legal process after seeing how poorly the procedures were handled.",
                "The proceedings were mishandled in ways that directly affected the fairness of the result.",
                "The outcome was unjust and it reinforces my belief that the legal system doesn't work equally for everyone.",
            ],
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
            "very_positive": [
                "I have a genuinely healthy and joyful relationship with food that I've cultivated over time.",
                "Eating brings me pleasure without guilt, and I feel I've found a good balance between enjoyment and nutrition.",
                "I make nutritious choices naturally because I've learned to listen to what my body actually needs.",
                "Food is a wonderful and positive part of my life that connects me with culture, creativity, and other people.",
                "I enjoy eating mindfully and appreciate the flavors, textures, and social aspects of meals.",
            ],
            "positive": [
                "My relationship with food is generally healthy, though I have occasional indulgences that I don't stress about.",
                "I usually make reasonably healthy food choices and feel good about my overall dietary patterns.",
                "Eating is mostly an enjoyable experience for me and I don't overthink my food decisions.",
                "I have reasonable eating habits and I try to balance nutrition with the pleasure of eating.",
                "Food is a generally positive part of my life, even though I'm not perfect about my choices.",
            ],
            "neutral": [
                "My relationship with food is pretty average and I don't spend a lot of energy thinking about eating patterns.",
                "Eating is neither particularly enjoyable nor stressful for me, it's just something I do.",
                "My food choices are mixed between healthy and unhealthy, depending on my mood and what's available.",
                "I don't have strong feelings about food in general and my habits are fairly typical.",
                "My relationship with food is unremarkable and I eat without much deliberation most of the time.",
            ],
            "negative": [
                "I sometimes struggle with food choices and feel guilty about what I eat more often than I'd like.",
                "Eating can be stressful for me because I'm caught between wanting to enjoy food and worrying about health.",
                "My relationship with food has its challenges, including emotional eating and inconsistent habits.",
                "I make unhealthy food choices more often than I should and I'm not happy about my eating patterns.",
                "Food has become a source of stress and internal conflict for me rather than something purely enjoyable.",
            ],
            "very_negative": [
                "I have a genuinely difficult relationship with food that affects my daily life and emotional wellbeing.",
                "Eating causes me significant stress and anxiety, and I often feel out of control around food.",
                "I struggle deeply with food choices and my eating patterns have become a source of real distress.",
                "My eating habits are problematic and I feel trapped in cycles that I can't seem to break.",
                "Food is one of the biggest challenges in my life and it affects my self-esteem, social life, and health.",
            ],
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
            "very_positive": [
                "I feel genuinely comfortable and confident in my own skin most days.",
                "I appreciate what my body can do for me rather than fixating on how it looks.",
                "I have worked hard to build a positive relationship with my body and it shows in how I carry myself.",
                "My body image is something I feel good about because I focus on health rather than appearance standards.",
                "I look in the mirror and feel proud of who I am, imperfections and all.",
            ],
            "positive": [
                "I'm generally satisfied with my body, though some days are better than others.",
                "I try to focus on what my body can do rather than how it compares to ideals.",
                "My body image is mostly positive and I don't spend too much energy worrying about appearance.",
                "I accept my body for what it is and that acceptance has grown over time.",
                "I feel reasonably comfortable with my appearance in most situations.",
            ],
            "neutral": [
                "I have mixed feelings about my body depending on the day and context.",
                "My body image is neither strongly positive nor negative, just kind of there.",
                "I don't spend a lot of time thinking about my appearance one way or another.",
                "Some aspects of my body I'm fine with, others I'm less sure about.",
                "I have average body satisfaction and it doesn't dominate my thinking.",
            ],
            "negative": [
                "I'm somewhat dissatisfied with my body and it affects my confidence in social settings.",
                "I find myself comparing my appearance to others more than I'd like to admit.",
                "There are specific aspects of my appearance that bother me and I think about them often.",
                "I feel uncomfortable with how I look in certain situations, like at the beach or in photos.",
                "My body image struggles have gotten in the way of enjoying activities I used to like.",
            ],
            "very_negative": [
                "I feel deeply unhappy with my body and it affects my daily life and mood.",
                "I avoid mirrors, photos, and social situations because of how I feel about my appearance.",
                "My negative body image has caused me real distress and impacts my self-worth.",
                "I constantly compare myself to others and always feel like I come up short.",
                "Body image concerns dominate my thoughts more than I want to admit and it's exhausting.",
            ],
        },
    },

    # ========== HUMAN FACTORS ==========

    "user_experience": {
        "explanation": {
            "very_positive": [
                "The interface was intuitive and I was able to accomplish everything I needed without any confusion or hesitation.",
                "I had an excellent user experience because the design anticipated my needs and made every step feel natural.",
                "Everything worked exactly as I expected and the flow from one step to the next was seamless.",
                "The design made my task genuinely effortless, and I appreciated the attention to detail in the layout and interactions.",
                "I would highly recommend this experience to others because it was polished, responsive, and easy to navigate.",
            ],
            "positive": [
                "The interface was reasonably easy to use and I was able to figure out most features without help.",
                "The user experience was good overall, with just a few minor things that could be slightly better.",
                "Most things worked as I expected them to, and the overall design was clean and helpful.",
                "I had a positive experience and the design choices generally made sense for what I was trying to do.",
                "The interface was well-organized and I didn't encounter any major obstacles completing my task.",
            ],
            "neutral": [
                "The interface was average and neither particularly impressive nor problematic to use.",
                "The user experience was mixed, with some aspects working smoothly and others requiring extra effort.",
                "Some features worked well and others felt clunky, so my overall impression is fairly neutral.",
                "The design was adequate for getting things done, but nothing about the experience stood out to me.",
                "My experience was unremarkable, which I suppose means it didn't get in my way too much.",
            ],
            "negative": [
                "The interface was somewhat confusing and I had to spend extra time figuring out how to do basic things.",
                "The user experience could be significantly improved because several interactions felt unintuitive.",
                "I found the design frustrating at times because it didn't match my expectations for how things should work.",
                "Several features didn't behave as I expected, which slowed me down and created unnecessary friction.",
                "I had a somewhat frustrating experience because the interface seemed to work against me rather than with me.",
            ],
            "very_negative": [
                "The interface was very confusing and I struggled to complete even simple tasks without getting lost.",
                "The user experience was genuinely poor, with confusing navigation, unresponsive elements, and unclear labels.",
                "Nothing worked the way I expected it to and the design made my task far harder than it should have been.",
                "The design actively hindered my ability to accomplish what I came to do and I felt frustrated throughout.",
                "I would not recommend this to anyone because the experience was confusing, slow, and poorly designed.",
            ],
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
            "very_positive": [
                "I genuinely embrace cultural diversity and find that cross-cultural interactions make my life richer and more interesting.",
                "Learning about different cultural perspectives has been one of the most rewarding aspects of my personal growth.",
                "I adapt well to different cultures because I approach differences with curiosity rather than judgment.",
                "Cross-cultural experiences have taught me that there are many valid ways to live and think about the world.",
                "Cultural differences fascinate me and I actively seek out opportunities to engage with people from different backgrounds.",
            ],
            "positive": [
                "I appreciate cultural diversity and try to be respectful and open when encountering different customs and values.",
                "Cross-cultural experiences are valuable to me, even when they push me outside my comfort zone.",
                "I adapt reasonably well to different cultures, though I recognize I still have biases to work through.",
                "I'm interested in learning about other cultures and believe it makes me a more well-rounded person.",
                "Cultural differences are interesting to me and I generally see them as enriching rather than threatening.",
            ],
            "neutral": [
                "I have mixed feelings about cultural diversity because while it's enriching, it can also be confusing or challenging.",
                "Cross-cultural experiences are neither particularly important nor unimportant to me personally.",
                "I adapt somewhat to different cultures but don't go out of my way to seek cross-cultural interactions.",
                "Culture is just one of many factors that shape people, and I don't think about it constantly.",
                "I'm fairly neutral about cultural differences and take them as they come without strong reactions.",
            ],
            "negative": [
                "Cultural differences can be genuinely challenging to navigate, and I sometimes feel lost or uncomfortable.",
                "I find cross-cultural interactions difficult because I'm never sure if I'm doing or saying the right thing.",
                "I struggle to adapt to unfamiliar cultural contexts and tend to prefer environments I'm familiar with.",
                "I lean toward preferring my own cultural context because it feels more natural and less stressful.",
                "Cultural differences sometimes create misunderstandings that I find frustrating and hard to resolve.",
            ],
            "very_negative": [
                "Cultural differences are very challenging for me and cross-cultural situations make me genuinely uncomfortable.",
                "I find cross-cultural experiences stressful because the expectations and norms feel unpredictable.",
                "I struggle significantly to adapt to different cultures and often feel like an outsider.",
                "I strongly prefer staying within my own cultural context where I understand how things work.",
                "Cultural differences have caused me real problems in communication and relationships, and I find them overwhelming.",
            ],
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
            "very_positive": [
                "I feel deeply grateful for the people, opportunities, and experiences that fill my life.",
                "Gratitude is genuinely central to my wellbeing, and I make a conscious effort to appreciate what I have.",
                "I regularly count my blessings and that practice has made me a happier and more content person.",
                "I appreciate so many things in my life, from my relationships to my health to simple daily pleasures.",
                "I'm truly thankful for what I have and I try to express that appreciation to the people around me.",
            ],
            "positive": [
                "I feel grateful for the good things in my life and try to keep perspective on what matters.",
                "Gratitude is important to me and I notice the positive aspects of my life more often than not.",
                "I often feel thankful for the people who support me and the opportunities I've had.",
                "I recognize my blessings and try not to take the good things in my life for granted.",
                "I appreciate the positive aspects of my life, even though I don't always stop to acknowledge them.",
            ],
            "neutral": [
                "My sense of gratitude is fairly average and depends on what's happening in my life at any given time.",
                "I sometimes appreciate the good things in my life but I don't actively practice gratitude.",
                "Gratitude isn't particularly prominent in my daily thinking, though I do feel thankful occasionally.",
                "I have a mixed relationship with gratitude because while I know I should be thankful, I also see what's lacking.",
                "I'm somewhat aware of my blessings but I wouldn't say gratitude is a defining feature of my personality.",
            ],
            "negative": [
                "I struggle to feel grateful because I tend to focus on what's wrong or missing in my life.",
                "I know I should appreciate what I have more, but gratitude doesn't come naturally to me.",
                "Gratitude is genuinely difficult for me, especially when I'm going through hard times.",
                "I rarely feel thankful because my attention is drawn to problems and challenges rather than blessings.",
                "I focus more on what's missing in my life than on what I have, which I realize isn't healthy.",
            ],
            "very_negative": [
                "I feel very little gratitude because my life circumstances make it hard to see the positive.",
                "I don't appreciate what I have because I'm too consumed by stress, disappointment, and unmet needs.",
                "Gratitude feels foreign to me and I genuinely struggle to find things to be thankful for.",
                "I almost never feel thankful because it seems like things are always going wrong for me.",
                "I can only see what's lacking in my life and gratitude seems like something other, luckier people feel.",
            ],
        },
    },

    "resilience": {
        "explanation": {
            "very_positive": [
                "I bounce back from setbacks quickly because I've learned that failure is just a stepping stone to growth.",
                "Adversity genuinely makes me stronger and more resourceful, and I've seen that pattern repeatedly in my life.",
                "I'm highly resilient because I've developed strong coping strategies and a support network I can rely on.",
                "When challenges come my way, I feel energized to overcome them rather than defeated by them.",
                "I've been through enough tough times to know that I can handle almost anything life throws at me.",
            ],
            "positive": [
                "I recover reasonably well from setbacks, though it sometimes takes me a little while to regain my footing.",
                "I can handle adversity when it comes, and I usually find a way through even when things feel hard.",
                "I'm fairly resilient in most situations, and I try to learn from difficult experiences.",
                "I cope with challenges by leaning on the strategies that have worked for me before.",
                "Setbacks affect me but don't derail me for long because I know things eventually improve.",
            ],
            "neutral": [
                "My resilience really depends on the type and severity of the setback I'm dealing with.",
                "Sometimes I handle adversity well, and other times I really struggle to get back on track.",
                "I have an average ability to cope with challenges and I wouldn't describe myself as especially tough or fragile.",
                "My recovery from setbacks is mixed because some hit harder than others depending on the circumstances.",
                "I'm not sure how resilient I really am because I haven't been tested in ways that push me to my limits.",
            ],
            "negative": [
                "I struggle to bounce back from setbacks and tend to dwell on what went wrong for a long time.",
                "Adversity takes a real toll on me emotionally and it's hard for me to find the motivation to keep going.",
                "My resilience could definitely be better because I tend to get stuck in negative thought patterns after failures.",
                "I have trouble coping with major challenges and often feel overwhelmed when things go wrong.",
                "Setbacks affect me significantly and it takes me longer than most people to recover.",
            ],
            "very_negative": [
                "I have great difficulty recovering from setbacks and sometimes feel like giving up entirely.",
                "Adversity overwhelms me completely and I often feel paralyzed when faced with serious challenges.",
                "I have very low resilience and even small setbacks can send me into a spiral.",
                "I feel unable to cope with challenges and often don't know how to pick myself back up.",
                "Setbacks devastate me and leave lasting emotional scars that make future challenges even harder.",
            ],
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
            "very_positive": [
                "I strongly embrace flexible gender roles and believe everyone should be free to define their own path regardless of gender.",
                "Traditional gender roles don't define me, and I think society benefits when people can contribute based on their strengths rather than their gender.",
                "I believe deeply in gender equality because I've seen how restrictive expectations hurt people of all genders.",
                "People should absolutely choose their own roles in life, and gender shouldn't be a limiting factor in any domain.",
                "Gender shouldn't limit anyone's aspirations, career choices, or how they express themselves in relationships.",
            ],
            "positive": [
                "I support flexible gender roles and think society is moving in the right direction on this issue.",
                "Traditional roles aren't essential to a good life, and I think people should have the freedom to choose differently.",
                "I lean toward gender equality in most contexts, even though I recognize change can be gradual.",
                "I believe roles in families and workplaces should be chosen based on individual preferences, not gender expectations.",
                "Gender shouldn't be a primary factor in determining what people can do or how they're treated.",
            ],
            "neutral": [
                "I have mixed views on gender roles because I can see arguments for both traditional and progressive approaches.",
                "Some traditional gender role distinctions seem to work for some people, while others find them constraining.",
                "I'm moderate on gender equality issues and think the right approach depends on the specific context.",
                "I think context matters a lot when discussing gender roles, and blanket statements in either direction feel too simplistic.",
                "My views on gender roles vary depending on the specific issue and I don't identify strongly with either camp.",
            ],
            "negative": [
                "I lean toward traditional gender roles because I think they reflect some real differences that shouldn't be ignored.",
                "I believe some gender distinctions are natural and trying to eliminate all of them creates its own problems.",
                "I'm skeptical that complete gender equality is achievable or even desirable in all domains.",
                "I think biological differences between genders have real implications for social roles that we shouldn't dismiss.",
                "Traditional patterns in gender roles have existed for good reasons and I think we're too quick to discard them.",
            ],
            "very_negative": [
                "I believe strongly in traditional gender roles because I think they provide structure and stability that benefits families.",
                "Men and women have fundamentally different strengths and roles, and I think society works best when those are respected.",
                "Traditional gender distinctions are important to me and I worry about what happens when they're eroded.",
                "I believe biology plays a major role in determining appropriate roles for men and women in society.",
                "I think traditional gender role patterns should be maintained because departing from them has caused many social problems.",
            ],
        },
    },

    # ========== RELATIONSHIPS ==========

    "attachment": {
        "explanation": {
            "very_positive": [
                "I feel securely attached in my close relationships and find it easy to depend on others and let them depend on me.",
                "I'm very comfortable with emotional intimacy and closeness, and I don't worry much about being abandoned.",
                "I trust my partners deeply because I've learned that vulnerability is a strength in close relationships.",
                "Close relationships come naturally to me, and I feel confident that the people I love will be there when I need them.",
                "I have a healthy attachment style where I can be independent while also valuing deep emotional bonds.",
            ],
            "positive": [
                "I have mostly secure attachment patterns, though I occasionally have moments of doubt in close relationships.",
                "I'm fairly comfortable with intimacy and closeness, even if it took me some time to get there.",
                "I generally trust the people I'm close to, though I'm aware that trust needs to be maintained.",
                "Close relationships are important to me and I feel I can navigate them reasonably well.",
                "My attachment patterns are pretty healthy overall, and I feel lucky about that.",
            ],
            "neutral": [
                "My attachment style really depends on the relationship and the person I'm with.",
                "Sometimes I crave closeness and other times I need space, and that balance shifts a lot.",
                "I have mixed feelings about depending on others because it feels risky but also necessary.",
                "Intimacy is comfortable in some relationships but anxiety-provoking in others for me.",
                "My attachment patterns are kind of average, with both secure and insecure tendencies.",
            ],
            "negative": [
                "I sometimes struggle with attachment because I worry about getting too close or being rejected.",
                "Emotional intimacy can feel uncomfortable for me, and I tend to pull back when things get too intense.",
                "I have trust issues in relationships that make it hard to fully open up to people.",
                "Close relationships are challenging because I often feel anxious about whether the other person really cares.",
                "I notice patterns in my attachment style that create problems in my relationships.",
            ],
            "very_negative": [
                "I have significant difficulty with attachment and find close relationships to be a major source of anxiety.",
                "Emotional intimacy feels threatening to me, and I often push people away before they can hurt me.",
                "I struggle greatly with trusting others, which makes real closeness nearly impossible.",
                "My attachment patterns have caused repeated problems in my relationships and I feel stuck in them.",
                "Close relationships are extremely difficult for me because I'm constantly worried about abandonment or loss of independence.",
            ],
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
        "evaluation": {
            "positive": [
                "Upward comparison can be genuinely motivating when I see it as aspirational rather than threatening.",
                "I think comparing ourselves to people we admire helps set realistic goals.",
                "Social comparison gives me a helpful frame of reference for where I stand.",
                "Looking at how others handle similar challenges helps me calibrate my own progress.",
                "I find that downward comparison occasionally reminds me to appreciate what I have.",
                "Comparison becomes constructive when I focus on learning rather than judging myself.",
            ],
            "neutral": [
                "Social comparison seems to be a double-edged sword depending on the direction.",
                "I think whether comparison helps or hurts depends a lot on the context and your mindset.",
                "There are times when comparing is useful and times when it just makes things worse.",
                "It's hard to evaluate comparison overall because the effects are so mixed.",
                "I can see the argument for both sides - comparison can motivate or demoralize.",
                "The impact of social comparison seems to vary a lot from person to person.",
            ],
            "negative": [
                "Upward comparison tends to make me feel inadequate more often than inspired.",
                "I think social media has made social comparison far more toxic than it used to be.",
                "Constantly measuring yourself against others erodes your sense of self-worth.",
                "The problem with comparison is that you usually compare your worst to others' best.",
                "I find that comparison creates a cycle of dissatisfaction that's hard to break.",
                "Downward comparison might feel good briefly but it doesn't actually improve anything.",
            ],
        },
    },

    # SELF-REGULATION
    "self_control": {
        "explanation": {
            "very_positive": [
                "I have excellent self-control and can resist temptation even when the short-term payoff is really appealing.",
                "I'm very disciplined in my choices and I attribute a lot of my success to being able to delay gratification.",
                "Resisting temptation comes relatively easy to me because I always keep my long-term goals in focus.",
                "I consistently make deliberate decisions rather than impulsive ones, and I'm proud of that discipline.",
                "Self-regulation is genuinely a strength of mine, whether it's diet, spending, or time management.",
            ],
            "positive": [
                "I have reasonably good self-control, though I'm not immune to giving in occasionally.",
                "I usually resist temptation by reminding myself of the bigger picture and what matters more.",
                "I'm fairly disciplined in most areas of my life, even if I slip up sometimes.",
                "I make good choices most of the time by pausing before acting on impulse.",
                "My self-control is adequate and I generally don't regret decisions I make in the moment.",
            ],
            "neutral": [
                "My self-control is honestly pretty average, and it really depends on the domain and my energy levels.",
                "Sometimes I'm very disciplined and other times I completely give in to impulse, with no clear pattern.",
                "I have mixed results with self-control depending on whether the temptation is food, spending, or something else.",
                "My ability to resist temptation varies so much that I can't say I'm either strong or weak in this area.",
                "I'm somewhere in the middle with self-control, doing okay in some areas and struggling in others.",
            ],
            "negative": [
                "I struggle with self-control more than I'd like to admit, especially with things like snacking or procrastinating.",
                "Resisting temptation is genuinely difficult for me, and I often give in even when I know better.",
                "I frequently act on impulse and regret the decision later, which is a frustrating pattern.",
                "My discipline could be a lot better, and I often wonder why self-control seems easier for other people.",
                "Self-regulation is challenging for me and it affects multiple areas of my life.",
            ],
            "very_negative": [
                "I have very poor self-control and it has caused real problems in my finances, health, and relationships.",
                "I almost always give in to temptation and feel powerless to stop myself in the moment.",
                "Discipline is extremely hard for me and I've tried many strategies that just don't seem to work.",
                "I frequently make impulsive choices that I deeply regret, but I can't seem to break the cycle.",
                "Self-control is one of my biggest weaknesses and it's something I feel genuinely ashamed about.",
            ],
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
            "very_positive": [
                "I manage my emotions very effectively by using strategies like reappraisal and mindfulness that I've practiced over time.",
                "I have excellent emotional control and can stay calm under pressure without suppressing how I actually feel.",
                "I rarely get overwhelmed by my feelings because I've learned to acknowledge emotions without letting them dictate my actions.",
                "I can regulate my emotions well, even in stressful situations, and I think that's one of my strongest personal qualities.",
                "Emotional stability comes naturally to me and I'm able to maintain a balanced perspective even when things get intense.",
            ],
            "positive": [
                "I generally manage my emotions well, though particularly stressful days can still throw me off balance.",
                "I have decent emotional control and can usually prevent my feelings from driving impulsive decisions.",
                "I can usually regulate my feelings by taking a step back and thinking through the situation rationally.",
                "I cope with emotional challenges reasonably well and have some strategies that work for me.",
                "My emotional regulation is good overall, even though I still have room for improvement in certain areas.",
            ],
            "neutral": [
                "My emotion regulation is honestly pretty average, with good days and bad days in roughly equal measure.",
                "Sometimes I manage my feelings well and other times they get the better of me, depending on the situation.",
                "I have typical emotional control for someone my age and don't stand out as especially regulated or dysregulated.",
                "My ability to cope with strong emotions varies a lot depending on what's going on in my life.",
                "I'm neither great nor poor at managing my emotions, and it really just depends on the context.",
            ],
            "negative": [
                "I struggle to regulate my emotions, especially when I'm stressed, tired, or feeling overwhelmed.",
                "My feelings often overwhelm me before I can think clearly about what's actually going on.",
                "Emotional control is genuinely difficult for me and I frequently react in ways I later regret.",
                "I have trouble managing strong feelings like anger, sadness, or anxiety once they take hold.",
                "My emotion regulation definitely needs work because I'm often at the mercy of my moods.",
            ],
            "very_negative": [
                "I have very poor emotion regulation and my feelings frequently spiral out of control.",
                "I'm almost always overwhelmed by strong emotions and don't know how to bring myself back to baseline.",
                "Emotional control is extremely hard for me and it affects my relationships, work, and daily functioning.",
                "My emotions essentially control me rather than the other way around, and it causes real problems.",
                "I struggle greatly with managing my feelings and it's one of the biggest challenges in my life.",
            ],
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

    # ========== NARRATIVE TRANSPORTATION ==========

    "narrative_transportation": {
        "explanation": {
            "positive": [
                "I got completely absorbed in the story and lost track of time while reading it.",
                "The narrative pulled me in so deeply that I felt like I was living in the story world.",
                "I found myself emotionally invested in the characters and their outcomes.",
                "The story was so vivid that I could picture everything happening as if I were there.",
                "I was fully transported into the narrative and forgot about my surroundings.",
                "The fictional world felt real to me while I was engaged with it.",
            ],
            "neutral": [
                "I followed the story but didn't feel particularly drawn into it.",
                "The narrative held my attention but I was still aware I was reading a story.",
                "I engaged with the plot at a surface level without strong emotional involvement.",
                "The story was interesting enough but I wouldn't say I was transported.",
                "I read through it and understood the events but didn't feel immersed.",
                "My engagement with the narrative was moderate and fairly detached.",
            ],
            "negative": [
                "I had trouble getting into the story and my mind kept wandering.",
                "The narrative felt flat and I couldn't connect with the characters at all.",
                "I was aware the whole time that this was just a story and it didn't move me.",
                "I found it hard to stay engaged and kept thinking about other things.",
                "The story didn't feel believable enough for me to get absorbed in it.",
                "I remained emotionally detached throughout the entire narrative.",
            ],
        },
        "evaluation": {
            "positive": [
                "Narrative transportation is a powerful mechanism for shifting attitudes because it bypasses critical resistance.",
                "Stories that deeply absorb readers are far more persuasive than straightforward arguments.",
                "I think the ability to get lost in a story is a fundamental part of how fiction shapes our beliefs.",
                "Emotional engagement with characters creates lasting attitude change in ways that facts alone cannot.",
                "Being transported into a narrative makes the message more compelling because you experience it rather than just hear it.",
            ],
            "neutral": [
                "The effects of narrative transportation seem to depend heavily on the quality of the story and the reader.",
                "Getting absorbed in fiction might shift attitudes temporarily but I'm not sure how lasting those effects are.",
                "Narrative persuasion is interesting but it probably works better for some people than others.",
                "I think transportation matters for entertainment but its persuasive impact may be overstated.",
                "Whether narrative absorption changes beliefs likely depends on how closely the story relates to real issues.",
            ],
            "negative": [
                "I think people overestimate how much a story can actually change someone's mind about real issues.",
                "Being absorbed in fiction is enjoyable but it shouldn't be confused with genuine attitude change.",
                "Narrative transportation seems like a form of manipulation when used deliberately for persuasion.",
                "The persuasive power of stories concerns me because it bypasses rational evaluation of evidence.",
                "I'm skeptical that getting lost in a fictional world translates to meaningful real-world belief changes.",
            ],
        },
    },

    # ========== GRATITUDE INTERVENTION ==========

    "gratitude_intervention": {
        "explanation": {
            "positive": [
                "Writing about things I'm grateful for genuinely shifted my mood and outlook.",
                "The gratitude exercise helped me notice positive aspects of my life I usually overlook.",
                "Focusing on thankfulness made me realize how much I take for granted day to day.",
                "I found the gratitude practice surprisingly effective at reducing my stress and worry.",
                "Listing things I appreciate gave me a sense of perspective on my problems.",
                "The intervention reminded me that I have a lot of good things going on even when times are tough.",
            ],
            "neutral": [
                "I did the gratitude exercise but I'm not sure it changed how I actually feel.",
                "Writing about gratitude was fine but it felt a bit forced rather than natural.",
                "I can see how this might help some people but it didn't have a strong effect on me.",
                "The practice was pleasant enough but I wouldn't say it transformed my thinking.",
                "I listed grateful things as instructed but my overall mood stayed about the same.",
                "It was an interesting exercise though I'm uncertain about its lasting impact.",
            ],
            "negative": [
                "Forced gratitude feels artificial to me and doesn't address real problems.",
                "I struggled to come up with things to be grateful for, which actually made me feel worse.",
                "The gratitude exercise felt dismissive of genuine difficulties I'm dealing with.",
                "Being told to focus on positives when things are hard feels like toxic positivity.",
                "I found the intervention irritating because it implies my negative feelings are just a perspective problem.",
                "Gratitude practices seem like a superficial band-aid that ignores structural issues people face.",
            ],
        },
        "evaluation": {
            "positive": [
                "Gratitude interventions have strong evidence behind them for boosting subjective wellbeing.",
                "I think gratitude practices work because they train attention toward positive experiences that are already there.",
                "Counting blessings is one of the most accessible and cost-effective psychological interventions available.",
                "Regular gratitude practice creates a genuine upward spiral of positive emotion and prosocial behavior.",
                "The research on gratitude journals improving sleep, mood, and relationships is very convincing.",
            ],
            "neutral": [
                "Gratitude interventions seem to work for some people and not others depending on personality.",
                "The evidence is mixed on how long the benefits of gratitude exercises actually last.",
                "I think gratitude practices are helpful as one tool but they're not a cure-all for wellbeing.",
                "The effectiveness probably depends on whether someone does it voluntarily or is assigned to do it.",
                "Gratitude interventions have promise but more research is needed on who benefits most.",
            ],
            "negative": [
                "Gratitude interventions can be harmful when they pressure people to minimize real suffering.",
                "I think the gratitude movement oversells its benefits and ignores important boundary conditions.",
                "Forced gratitude can backfire for people dealing with depression, trauma, or genuine hardship.",
                "The emphasis on gratitude places the burden on individuals rather than addressing systemic problems.",
                "I find the research on gratitude interventions less robust than popular psychology books suggest.",
            ],
        },
    },

    # ========== MORAL CLEANSING ==========

    "moral_cleansing": {
        "explanation": {
            "positive": [
                "After doing something I felt bad about, engaging in a good deed helped restore my sense of being a decent person.",
                "I feel like my ethical identity is important to me, and I actively work to maintain it through my actions.",
                "When I fall short of my moral standards, I feel driven to compensate by doing something positive.",
                "I believe redemption is possible and that past mistakes can be offset by genuine good behavior.",
                "My moral self-image matters deeply to me and I take steps to protect it when it feels threatened.",
                "I've noticed that guilt motivates me to act more ethically afterward, almost like a moral reset.",
            ],
            "neutral": [
                "I'm not sure whether doing good after doing bad actually balances things out morally.",
                "Sometimes I feel the urge to compensate for ethical failures but I question whether it really helps.",
                "My moral identity fluctuates and I don't always feel compelled to restore it after a lapse.",
                "The idea of moral cleansing makes sense psychologically but I'm unsure if it reflects true moral growth.",
                "I have mixed feelings about whether guilt-driven good deeds are genuinely ethical or just self-serving.",
                "I think about my moral identity sometimes but I wouldn't say it dominates my decision-making.",
            ],
            "negative": [
                "I don't think you can erase unethical behavior by doing something good afterward.",
                "The concept of moral cleansing feels like a convenient way to excuse bad behavior.",
                "Doing a good deed after a bad one doesn't actually undo the harm that was caused.",
                "I worry that moral licensing lets people behave worse because they feel they've earned ethical credit.",
                "Sacred values shouldn't be tradeable and the idea that you can compensate for violating them is troubling.",
                "Guilt-driven charity isn't the same as genuine moral commitment in my view.",
            ],
        },
        "evaluation": {
            "positive": [
                "Moral cleansing is a fascinating mechanism that shows how deeply people care about maintaining an ethical self-image.",
                "The research on moral self-regulation reveals an important psychological need for consistency between actions and values.",
                "I think moral cleansing is actually healthy because it motivates people to repair harm and recommit to their values.",
                "The connection between guilt and prosocial behavior demonstrates that moral emotions serve an important social function.",
                "Understanding moral cleansing dynamics helps explain why people donate after ethical lapses, which still benefits recipients.",
            ],
            "neutral": [
                "Moral cleansing is psychologically interesting but its implications for actual ethical behavior are ambiguous.",
                "The research shows people regulate their moral self-concept, but whether that produces better behavior overall is debatable.",
                "I think moral self-regulation is real but it's unclear whether it leads to genuine moral improvement or just psychological comfort.",
                "The moral cleansing effect seems to depend heavily on whether the person acknowledges the initial transgression.",
                "It's a well-documented phenomenon but I'm uncertain whether it's ultimately positive or negative for moral behavior.",
            ],
            "negative": [
                "Moral cleansing enables a troubling cycle where people feel licensed to behave unethically as long as they compensate later.",
                "I think the research on moral licensing shows that self-regulation can actually undermine consistent ethical behavior.",
                "The problem with moral cleansing is that it treats ethics like a bank account rather than a consistent commitment.",
                "Moral self-regulation might maintain self-image but it doesn't necessarily reduce the total harm people cause.",
                "I find the implications concerning because it suggests people use good deeds as permission slips for future transgressions.",
            ],
        },
    },

    # ========== DIGITAL WELLBEING ==========

    "digital_wellbeing": {
        "explanation": {
            "positive": [
                "Reducing my screen time made a noticeable difference in my sleep quality and overall mood.",
                "I've set boundaries with my phone and it feels liberating to not check notifications constantly.",
                "Taking a digital detox helped me reconnect with activities and people I had been neglecting.",
                "I realized how much of my day I was spending scrolling mindlessly and cutting back felt empowering.",
                "Managing my technology use has improved my focus and ability to be present in conversations.",
                "I feel much better since I started being intentional about when and how I use my devices.",
            ],
            "neutral": [
                "I've tried reducing screen time but I'm not sure it made a huge difference either way.",
                "My phone use is probably average and I don't feel strongly that it's a problem.",
                "I think about digital wellbeing sometimes but I haven't made major changes to my habits.",
                "Technology is a mixed bag for me - some screen time is valuable and some is wasteful.",
                "I use my devices a lot but I wouldn't say I'm addicted or that it's seriously hurting me.",
                "I can see the arguments for reducing phone use but I also rely on it for a lot of things.",
            ],
            "negative": [
                "I know I spend too much time on my phone but I find it really hard to stop.",
                "Notification overload is genuinely stressful and I feel tethered to my devices against my will.",
                "My screen time habits have gotten worse and it's affecting my sleep and relationships.",
                "I feel anxious when I'm away from my phone which tells me something is wrong.",
                "Digital devices have made it impossible for me to focus for extended periods anymore.",
                "I resent how much time technology takes from me but I can't seem to break the cycle.",
            ],
        },
        "evaluation": {
            "positive": [
                "Digital wellbeing initiatives are important because excessive screen time has documented effects on mental health.",
                "I think intentional technology use is one of the most important skills for modern life.",
                "Screen time awareness tools and digital detox practices can meaningfully improve quality of life.",
                "The research linking excessive phone use to anxiety and sleep disruption is compelling and actionable.",
                "Setting boundaries with technology is essential for maintaining focus, relationships, and psychological health.",
                "I believe digital literacy should include knowing when to disconnect, not just how to use devices.",
            ],
            "neutral": [
                "The digital wellbeing conversation is important but I think the panic about screen time is sometimes exaggerated.",
                "Not all screen time is equal and I think the focus should be on quality of use rather than total hours.",
                "Digital wellbeing matters but the evidence on specific harms of phone use is more nuanced than headlines suggest.",
                "I think the effects of technology on wellbeing depend heavily on what you're doing and who you are.",
                "There's probably a healthy balance with technology but I'm not convinced the science has found it yet.",
            ],
            "negative": [
                "I think the digital wellbeing movement puts too much responsibility on individuals instead of tech companies.",
                "Screen time limits feel paternalistic and ignore that many people need devices for work and social connection.",
                "Digital detox culture is mostly a privilege for people who can afford to disconnect from their obligations.",
                "The moral panic about phones resembles past panics about television and radio that turned out to be overblown.",
                "I'm skeptical that individual digital wellbeing practices can solve problems created by addictive design choices.",
            ],
        },
    },

    # ========== NEW DOMAINS (v1.0.9.3) ==========

    # NOSTALGIA
    "nostalgia": {
        "explanation": {
            "very_positive": [
                "I love revisiting memories from my past because they remind me of who I am and where I came from.",
                "Nostalgia fills me with warmth and gratitude for the experiences and people that shaped my life.",
                "Looking back on meaningful moments from my past brings me genuine joy and a sense of connection to my younger self.",
                "I cherish nostalgic memories because they give me a sense of continuity and meaning in my life.",
                "Thinking about the past makes me deeply appreciate the relationships and experiences I've been lucky to have.",
            ],
            "positive": [
                "I enjoy looking back on fond memories, even though I know things have changed since then.",
                "Nostalgia generally feels pleasant to me and reminds me of good times and important people.",
                "Revisiting memories from my past usually puts me in a positive mood and gives me perspective.",
                "I find comfort in nostalgic thoughts because they remind me of simpler or happier times.",
                "Looking back on my past is mostly a positive experience that helps me appreciate my journey.",
            ],
            "neutral": [
                "I have mixed feelings about nostalgia because while some memories are pleasant, others are bittersweet.",
                "Looking back on the past doesn't strongly affect me one way or another most of the time.",
                "Nostalgia is something I experience occasionally but it doesn't play a major role in my emotional life.",
                "I'm fairly neutral about revisiting old memories because the past is the past to me.",
                "Sometimes nostalgia feels good and sometimes it just makes me aware of how much has changed.",
            ],
            "negative": [
                "Nostalgia often makes me sad because it reminds me of things I've lost or can never get back.",
                "Looking back on the past tends to make me feel worse rather than better about where I am now.",
                "I find nostalgic memories bittersweet at best, and they usually leave me feeling a sense of loss.",
                "Revisiting the past is emotionally difficult because it highlights what's changed for the worse.",
                "I try to avoid nostalgia because dwelling on the past prevents me from focusing on the present.",
            ],
            "very_negative": [
                "Nostalgia is genuinely painful for me because it brings up memories of people and times I've lost forever.",
                "Looking back on the past fills me with deep sadness and regret about how things turned out.",
                "I find nostalgic thoughts overwhelming because they remind me of everything that went wrong or slipped away.",
                "The past was so much better than my present that thinking about it only makes my current situation feel worse.",
                "I strongly avoid nostalgic thinking because it sends me into a spiral of sadness and longing for what's gone.",
            ],
        },
    },

    # FORGIVENESS
    "forgiveness": {
        "explanation": {
            "very_positive": [
                "I believe deeply in forgiveness because holding grudges only hurts the person who was wronged.",
                "I find it relatively easy to forgive others because I understand that everyone makes mistakes.",
                "Forgiveness is central to my approach to relationships, and I've seen how it heals both parties.",
                "I've experienced the freedom that comes with letting go of resentment and I try to practice forgiveness actively.",
                "I believe that forgiving someone doesn't mean condoning what they did, but it frees me from carrying anger.",
            ],
            "positive": [
                "I generally try to forgive others because I know it's better for my own mental health and relationships.",
                "Forgiveness doesn't come instantly for me, but I usually get there eventually with time and reflection.",
                "I think forgiveness is important, even though some situations take longer to work through than others.",
                "I lean toward forgiveness in most situations because I don't want bitterness to define my life.",
                "I believe in giving people second chances when they seem genuinely sorry for what they did.",
            ],
            "neutral": [
                "My ability to forgive really depends on the severity of what happened and whether the person is truly sorry.",
                "I have mixed feelings about forgiveness because some things feel forgivable and others genuinely don't.",
                "I'm not sure where I stand on forgiveness because it seems like the right thing to do but it's incredibly hard sometimes.",
                "Forgiveness is complicated for me and I don't think there's a universal rule about when to forgive.",
                "I can forgive some things easily but other transgressions feel like they cross a line that's hard to come back from.",
            ],
            "negative": [
                "I find forgiveness very difficult, especially when someone has deeply hurt me or violated my trust.",
                "I tend to hold onto grievances longer than I probably should because letting go feels like excusing bad behavior.",
                "Forgiveness doesn't come naturally to me and I often struggle with resentment long after the event.",
                "I'm skeptical of the pressure to forgive because some actions don't deserve forgiveness in my view.",
                "I have trouble forgiving because I worry that it signals to the other person that what they did was acceptable.",
            ],
            "very_negative": [
                "I find it nearly impossible to forgive people who have seriously wronged me, and I don't think I should have to.",
                "Forgiveness feels like letting someone off the hook for real harm, and I refuse to do that.",
                "I hold onto resentment because the pain that was caused to me was real and forgiving would dishonor that.",
                "I strongly resist the idea that I owe forgiveness to people who deliberately hurt me.",
                "Some things are simply unforgivable and I reject the notion that forgiveness is always the right path.",
            ],
        },
    },

    # GRATITUDE EXPERIENCE
    "gratitude_experience": {
        "explanation": {
            "very_positive": [
                "I feel an overwhelming sense of thankfulness when I reflect on the meaningful experiences in my life.",
                "Practicing gratitude has genuinely transformed my outlook and I feel deeply appreciative every day.",
                "I experience gratitude intensely and it brings me a profound sense of contentment and connection.",
                "I'm deeply thankful for the kindness others have shown me and the opportunities I've been given.",
                "Gratitude comes naturally to me and it enhances almost every aspect of my emotional life.",
            ],
            "positive": [
                "I generally feel appreciative of the good things in my life and try to acknowledge them regularly.",
                "Gratitude is something I experience often, especially when I take a moment to reflect on what I have.",
                "I feel thankful for my relationships, health, and opportunities, even when other things are challenging.",
                "Practicing gratitude has been helpful for my wellbeing and I try to make it a regular habit.",
                "I experience genuine appreciation for the positive aspects of my life more often than not.",
            ],
            "neutral": [
                "My experience of gratitude is fairly average and depends a lot on my current mood and circumstances.",
                "I feel appreciative sometimes, but gratitude isn't something I actively cultivate or think about much.",
                "I have mixed experiences with gratitude, feeling thankful in some moments and indifferent in others.",
                "I understand the value of gratitude but I wouldn't say it's a prominent emotion in my daily experience.",
                "My gratitude experience is inconsistent and varies quite a bit depending on what's going on in my life.",
            ],
            "negative": [
                "I find it difficult to feel genuinely grateful because my focus tends to drift toward what's wrong or missing.",
                "Gratitude feels forced when I try to practice it, like I'm going through the motions without real feeling.",
                "I struggle with feeling appreciative because the challenges in my life overshadow the good things.",
                "The idea of gratitude sometimes feels dismissive of the real difficulties I'm experiencing.",
                "I rarely experience spontaneous feelings of gratitude and have to work hard to notice the positives.",
            ],
            "very_negative": [
                "I almost never feel grateful and the concept feels hollow given what I've been through.",
                "Gratitude feels impossible when your circumstances are genuinely difficult and no amount of reframing changes that.",
                "I resent being told to feel grateful because it minimizes the very real struggles I face every day.",
                "My experience with gratitude is essentially nonexistent because I can't seem to find much to be thankful for.",
                "I feel angry when people suggest gratitude as a solution because it doesn't address the root causes of my unhappiness.",
            ],
        },
    },

    # SLEEP QUALITY
    "sleep_quality": {
        "explanation": {
            "very_positive": [
                "I consistently sleep well and wake up feeling rested and ready to tackle the day.",
                "My sleep quality is excellent and I credit it with helping me function at my best mentally and physically.",
                "I fall asleep easily, stay asleep through the night, and feel genuinely refreshed every morning.",
                "I've established a sleep routine that works perfectly for me and I rarely have trouble sleeping.",
                "Good sleep is one of the foundations of my wellbeing and I'm fortunate that it comes naturally to me.",
            ],
            "positive": [
                "I generally sleep well and most nights I feel reasonably rested when I wake up.",
                "My sleep quality is pretty good, with occasional nights that aren't as restful as I'd like.",
                "I usually fall asleep within a reasonable time and get enough rest to function well during the day.",
                "Sleep is mostly a positive experience for me, even if I have the occasional restless night.",
                "I feel fairly well-rested most of the time, which helps me maintain my energy and focus.",
            ],
            "neutral": [
                "My sleep quality is average, with some nights being better than others and no strong pattern.",
                "I wouldn't describe my sleep as particularly good or bad, it's just kind of there.",
                "Some nights I sleep great and others I toss and turn, and I'm not sure what makes the difference.",
                "My sleep is inconsistent and I don't feel strongly about it one way or another.",
                "I get through most days fine but I wouldn't say I feel fully rested on a regular basis either.",
            ],
            "negative": [
                "My sleep quality is poor and it affects my mood, energy, and ability to concentrate during the day.",
                "I frequently have trouble falling asleep or staying asleep, which leaves me feeling drained.",
                "I wake up feeling tired more often than not, and it's starting to take a toll on my daily life.",
                "Sleep has become a source of frustration because I can never seem to get enough quality rest.",
                "I lie awake many nights with racing thoughts and the lack of sleep makes everything harder.",
            ],
            "very_negative": [
                "My sleep quality is terrible and chronic insomnia has become one of the biggest problems in my life.",
                "I dread going to bed because I know I'll spend hours lying awake, exhausted but unable to fall asleep.",
                "The lack of quality sleep has seriously damaged my health, relationships, and ability to function.",
                "I'm constantly exhausted because I can never get a full night of uninterrupted, restful sleep.",
                "Sleep deprivation has become a vicious cycle that affects everything in my life and I feel desperate for relief.",
            ],
        },
    },

    # CLIMATE CHANGE
    "climate_change": {
        "explanation": {
            "very_positive": [
                "I'm deeply committed to addressing climate change and believe we have both the technology and moral obligation to act now.",
                "Climate change is the defining challenge of our generation and I'm optimistic that collective action can make a real difference.",
                "I take personal responsibility for my environmental footprint and advocate strongly for systemic changes to reduce emissions.",
                "I believe the science on climate change is clear and compelling, and we need to listen to the experts and act urgently.",
                "I feel hopeful about climate solutions like renewable energy and sustainable practices that are already making progress.",
            ],
            "positive": [
                "I take climate change seriously and try to make environmentally conscious choices in my daily life.",
                "I believe we should be doing more to address climate change, even though the scale of the problem feels daunting.",
                "I'm concerned about the effects of climate change on future generations and support policies that address it.",
                "I think the evidence for human-caused climate change is strong and we have a responsibility to respond.",
                "I care about sustainability and make reasonable efforts to reduce my environmental impact.",
            ],
            "neutral": [
                "I have mixed feelings about climate change because while the science seems concerning, the solutions feel complex.",
                "I'm somewhere in the middle on climate change policy because I see both the environmental and economic concerns.",
                "I think climate change is real but I'm not sure how much individuals can actually do about it.",
                "I care about the environment to some extent but climate change isn't the issue I feel most passionate about.",
                "I accept that the climate is changing but I'm uncertain about the best way to respond to the problem.",
            ],
            "negative": [
                "I'm skeptical about the urgency of climate change claims and think the economic costs of action are being ignored.",
                "I think climate change concerns are often exaggerated and the proposed solutions would hurt ordinary people the most.",
                "I disagree with many climate policies because they seem to prioritize environmental goals over economic reality.",
                "I'm not convinced that drastic climate action is necessary or that the predictions are as certain as claimed.",
                "Climate change activism feels more like political ideology than genuine science to me.",
            ],
            "very_negative": [
                "I believe climate change fears are greatly exaggerated and the push for climate action is driven by political agendas.",
                "I strongly oppose most climate change policies because they destroy jobs and raise costs without meaningful environmental benefit.",
                "I think the climate change narrative is used to control people and restrict economic freedom.",
                "I reject the idea that humans are primarily responsible for climate change and I'm tired of the alarmism.",
                "Climate change activism has become a religion that doesn't tolerate dissent, and I refuse to buy into the hysteria.",
            ],
        },
    },

    # ARTIFICIAL INTELLIGENCE
    "artificial_intelligence": {
        "explanation": {
            "very_positive": [
                "I'm genuinely excited about artificial intelligence and believe it has the potential to solve some of humanity's biggest challenges.",
                "AI technology is one of the most promising developments in human history, and I'm optimistic about its applications in healthcare, science, and education.",
                "I embrace AI tools in my daily life and work because they make me more productive and open up new possibilities.",
                "I believe artificial intelligence will create more opportunities than it eliminates and I'm excited to see where it goes.",
                "AI represents a transformative leap in human capability and I think the benefits far outweigh the risks if we develop it responsibly.",
            ],
            "positive": [
                "I generally view artificial intelligence positively and think it has useful applications in many areas of life.",
                "AI technology has impressed me with what it can do, even though I think we need to be thoughtful about how we use it.",
                "I'm moderately optimistic about AI because I've seen it help people be more efficient and solve complex problems.",
                "I believe AI can be a force for good as long as we maintain human oversight and address bias concerns.",
                "I find artificial intelligence tools helpful in my work and I'm open to seeing how the technology continues to develop.",
            ],
            "neutral": [
                "I have mixed feelings about artificial intelligence because I can see both the tremendous potential and the real risks.",
                "AI is interesting technology but I'm not sure whether the long-term effects will be mostly positive or negative.",
                "I think the conversation about AI is important but I don't feel strongly optimistic or pessimistic about it.",
                "My views on artificial intelligence are fairly balanced because there are legitimate arguments on both sides.",
                "I use some AI tools but I also have questions about privacy, job displacement, and accountability that keep me from fully embracing it.",
            ],
            "negative": [
                "I'm concerned about artificial intelligence because I think the rush to deploy it is outpacing our ability to manage the risks.",
                "AI technology worries me because of its potential to eliminate jobs, invade privacy, and concentrate power in fewer hands.",
                "I'm skeptical about the hype around artificial intelligence and think the downsides are being seriously underestimated.",
                "I have real concerns about AI bias, lack of transparency, and the erosion of human decision-making in critical areas.",
                "Artificial intelligence makes me uncomfortable because I don't fully understand how it works and I don't trust the companies building it.",
            ],
            "very_negative": [
                "I believe artificial intelligence poses serious dangers to employment, privacy, democracy, and human autonomy.",
                "AI technology deeply concerns me because I think we are creating systems we cannot control or fully understand.",
                "I'm strongly opposed to the unchecked development of artificial intelligence because the risks are existential.",
                "I think AI is being used to exploit and manipulate people while the companies profiting from it avoid accountability.",
                "Artificial intelligence represents a threat to what makes us human, and I believe we should be far more cautious about its development.",
            ],
        },
    },

    # REMOTE WORK
    "remote_work": {
        "explanation": {
            "very_positive": [
                "Working remotely has been genuinely transformative for my quality of life, productivity, and work-life balance.",
                "I love working from home because it eliminates commuting, gives me flexibility, and lets me create my ideal work environment.",
                "Remote work has made me more productive because I can focus without the distractions and politics of an office.",
                "I feel strongly that remote work should be a permanent option because it benefits both employees and employers.",
                "Working remotely has improved my health, relationships, and job satisfaction in ways I didn't expect.",
            ],
            "positive": [
                "I generally enjoy remote work because of the flexibility it provides, even though it has some challenges.",
                "Working from home has been a mostly positive experience that I'd prefer to continue over going back to an office full-time.",
                "I appreciate the work-life balance that remote work offers, though I do miss some aspects of in-person collaboration.",
                "Remote work suits my personality and working style well, and I feel I perform better with the autonomy it provides.",
                "I'm satisfied with remote work arrangements and think they should be widely available for jobs that can support them.",
            ],
            "neutral": [
                "I have genuinely mixed feelings about remote work because I see real benefits and real drawbacks.",
                "Some days I love working from home and other days I miss the structure and social aspects of an office.",
                "Remote work is fine for certain tasks but I think some types of collaboration genuinely require being in person.",
                "I don't feel strongly about remote versus in-office work because both have their advantages depending on the situation.",
                "My experience with remote work has been average, with productivity gains offset by isolation and boundary issues.",
            ],
            "negative": [
                "I find remote work isolating and struggle with the lack of social interaction and clear boundaries between work and home.",
                "Working from home has been challenging for me because I need the structure and accountability of an office environment.",
                "I miss the spontaneous collaboration and casual social connections that come with working alongside colleagues.",
                "Remote work has blurred the lines between my professional and personal life in ways that aren't healthy for me.",
                "I've noticed my mental health and motivation declining since switching to remote work, and I'd prefer to go back to the office.",
            ],
            "very_negative": [
                "Remote work has been terrible for me and has significantly worsened my mental health, productivity, and sense of connection.",
                "I feel deeply isolated and disconnected working from home, and it has affected both my work performance and personal life.",
                "Working remotely has destroyed the work-life boundaries I used to have and I feel like I'm always on the clock.",
                "I strongly believe remote work is harmful for team cohesion, company culture, and most people's wellbeing.",
                "I desperately want to return to in-person work because remote work has made me lonely, unmotivated, and professionally stagnant.",
            ],
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
        'government', 'policy', 'election', 'vote', 'congress',
        'senate', 'parliament', 'legislature', 'political party',
        'left-wing', 'right-wing', 'centrist', 'moderate', 'bipartisan',
        'political science', 'poli sci', 'political behavior',
        'political attitude', 'political opinion', 'campaign',
    ],
    StudyDomain.POLARIZATION: [
        'polariz', 'divided', 'partisan', 'other side', 'opposing view',
        'affective polarization', 'ideological divide', 'partisan divide',
        'culture war', 'political division', 'tribalism', 'polarized',
        'increasingly divided', 'cross-partisan', 'depolarization',
    ],
    StudyDomain.PARTISANSHIP: [
        'partisan', 'party', 'democrat', 'republican', 'liberal', 'conservative',
        'party identification', 'party loyalty', 'party affiliation',
        'gop', 'dnc', 'rnc', 'red state', 'blue state', 'swing voter',
        'independent voter', 'cross-party', 'party line',
    ],
    StudyDomain.VOTING: [
        'vote', 'election', 'ballot', 'candidate', 'electoral',
        'voter', 'turnout', 'voter turnout', 'primary', 'caucus',
        'poll', 'polling', 'exit poll', 'swing state', 'referendum',
        'absentee', 'mail-in ballot', 'voter registration', 'suffrage',
    ],
    StudyDomain.MEDIA: [
        'media', 'news', 'journalism', 'coverage', 'report',
        'newspaper', 'broadcast', 'cable news', 'press', 'journalist',
        'media bias', 'news source', 'editorial', 'headline',
        'breaking news', 'news consumption', 'media diet', 'newsroom',
    ],
    StudyDomain.POLICY_ATTITUDES: [
        'policy', 'legislation', 'law', 'regulation', 'government action',
        'public policy', 'policy preference', 'policy support',
        'policy opposition', 'welfare', 'immigration policy',
        'gun control', 'healthcare policy', 'tax policy', 'reform',
        'policy proposal', 'policy debate', 'regulatory',
    ],
    StudyDomain.CIVIC_ENGAGEMENT: [
        'civic', 'citizen', 'participat', 'community', 'volunteer',
        'civic duty', 'civic participation', 'community involvement',
        'town hall', 'public meeting', 'grassroots', 'activism',
        'advocacy', 'petition', 'protest', 'civil society', 'civics',
    ],
    StudyDomain.POLITICAL_TRUST: [
        'trust government', 'trust politician', 'institutional trust',
        'trust in government', 'political confidence', 'government trust',
        'trust in congress', 'trust in parliament', 'democratic trust',
        'trust in democracy', 'political legitimacy', 'government approval',
    ],
    StudyDomain.IDEOLOGY: [
        'ideolog', 'left', 'right', 'progressive', 'traditionalist',
        'conservative ideology', 'liberal ideology', 'libertarian',
        'socialism', 'capitalism', 'populism', 'populist',
        'nationalist', 'globalist', 'neoconservative', 'neoliberal',
        'political spectrum', 'authoritarian', 'egalitarian',
    ],
    StudyDomain.MISINFORMATION: [
        'misinformation', 'fake news', 'fact check', 'false claim', 'debunk',
        'disinformation', 'conspiracy', 'conspiracy theory', 'hoax',
        'misleading', 'propaganda', 'infodemic', 'deepfake',
        'media manipulation', 'truth decay', 'post-truth', 'fact-check',
        'prebunking', 'inoculation', 'credibility assessment',
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
        'employer', 'staff', 'personnel', 'workforce', 'workplace',
        'company', 'firm', 'corporation', 'hr', 'human resources',
        'occupational', 'professional', 'coworker', 'co-worker',
    ],
    StudyDomain.LEADERSHIP: [
        'leader', 'manager', 'boss', 'supervisor', 'management',
        'leadership style', 'transformational', 'transactional',
        'servant leader', 'charismatic', 'visionary', 'directive',
        'participative', 'executive', 'ceo', 'cfo', 'cto',
        'leadership effectiveness', 'managerial', 'supervisory',
    ],
    StudyDomain.TEAMWORK: [
        'teamwork', 'team', 'group project', 'collaboration',
        'team performance', 'team effectiveness', 'team building',
        'group work', 'collaborative', 'team member', 'team composition',
        'team diversity', 'cross-functional', 'agile team', 'scrum',
    ],
    StudyDomain.MOTIVATION: [
        'motivat', 'drive', 'engagement', 'interest', 'passion',
        'intrinsic motivation', 'extrinsic motivation', 'incentive',
        'reward', 'recognition', 'goal setting', 'self-determination',
        'autonomy', 'competence', 'relatedness', 'amotivation',
        'mastery', 'purpose', 'aspiration',
    ],
    StudyDomain.JOB_SATISFACTION: [
        'job satisf', 'happy at work', 'enjoy work', 'work satisf',
        'job satisfaction', 'work satisfaction', 'satisfied with job',
        'job contentment', 'work enjoyment', 'job happiness',
        'job dissatisfaction', 'unhappy at work', 'job attitude',
    ],
    StudyDomain.ORGANIZATIONAL_COMMITMENT: [
        'commit', 'loyal', 'stay with', 'turnover', 'leave job',
        'organizational commitment', 'affective commitment',
        'normative commitment', 'continuance commitment', 'retention',
        'attrition', 'quit', 'resignation', 'employee loyalty',
        'employee retention', 'intent to leave', 'job hopping',
    ],
    StudyDomain.WORK_LIFE_BALANCE: [
        'work-life', 'work life balance', 'personal time', 'family time',
        'work-life conflict', 'work-family conflict', 'boundary',
        'flexible schedule', 'parental leave', 'childcare',
        'overwork', 'workaholic', 'time off', 'vacation', 'pto',
        'work-life integration', 'spillover',
    ],
    StudyDomain.EMPLOYEE_ENGAGEMENT: [
        'engag', 'involved', 'invested', 'enthusiasm', 'passionate',
        'employee engagement', 'work engagement', 'vigor', 'dedication',
        'absorption', 'disengaged', 'quiet quitting', 'engagement survey',
        'gallup', 'employee voice', 'psychological contract',
    ],
    StudyDomain.ORGANIZATIONAL_CULTURE: [
        'culture', 'values', 'company culture', 'organizational culture',
        'culture fit', 'cultural alignment', 'corporate culture',
        'workplace culture', 'culture change', 'subculture',
        'cultural values', 'mission', 'vision statement', 'core values',
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
        'classroom', 'student', 'teacher', 'curriculum', 'pedagogy',
        'higher education', 'k-12', 'k12', 'elementary', 'secondary',
        'undergraduate', 'graduate', 'academic', 'educational',
        'instruction', 'semester', 'course', 'lecture', 'tutorial',
    ],
    StudyDomain.LEARNING: [
        'learn', 'education', 'study', 'knowledge', 'skill',
        'learning outcome', 'mastery', 'comprehension', 'understanding',
        'transfer', 'retention', 'scaffolding', 'zone of proximal',
        'constructivism', 'metacognitive', 'deep learning',
        'surface learning', 'active learning', 'collaborative learning',
    ],
    StudyDomain.ACADEMIC_MOTIVATION: [
        'study motivation', 'academic', 'grades', 'achievement', 'perform',
        'academic motivation', 'academic performance', 'gpa',
        'study habits', 'academic self-efficacy', 'test anxiety',
        'achievement goal', 'mastery goal', 'performance goal',
        'academic procrastination', 'academic engagement', 'homework',
    ],
    StudyDomain.TEACHING_EFFECTIVENESS: [
        'teach', 'instructor', 'professor', 'effective teaching',
        'teaching quality', 'teaching method', 'pedagogy', 'didactic',
        'teaching evaluation', 'course evaluation', 'teacher training',
        'professional development', 'teaching practice', 'flipped classroom',
        'lecture', 'tutoring', 'scaffolding', 'differentiated instruction',
    ],
    StudyDomain.ONLINE_LEARNING: [
        'online learning', 'remote learning', 'virtual learning',
        'distance learning', 'e-learning', 'elearning', 'mooc',
        'learning management system', 'lms', 'canvas', 'blackboard',
        'moodle', 'zoom class', 'synchronous', 'asynchronous',
        'blended learning', 'hybrid learning', 'coursera', 'edx',
    ],
    StudyDomain.EDUCATIONAL_TECHNOLOGY: [
        'edtech', 'educational technology', 'learning platform', 'digital learning',
        'educational software', 'learning app', 'adaptive learning',
        'intelligent tutoring', 'gamification', 'educational game',
        'simulation', 'virtual lab', 'learning analytics', 'ai in education',
        'educational ai', 'personalized learning', 'smart classroom',
    ],
    StudyDomain.STUDENT_ENGAGEMENT: [
        'student engagement', 'class participation', 'attentive', 'involved',
        'student involvement', 'attendance', 'active participation',
        'classroom engagement', 'academic engagement', 'student voice',
        'student agency', 'student interest', 'disengaged student',
    ],
    StudyDomain.ASSESSMENT_FEEDBACK: [
        'feedback', 'assessment', 'grade', 'evaluation', 'test',
        'formative assessment', 'summative assessment', 'rubric',
        'peer assessment', 'self-assessment', 'grading', 'exam',
        'quiz', 'standardized test', 'portfolio', 'criterion-referenced',
        'norm-referenced', 'diagnostic assessment', 'learning assessment',
    ],

    # ========== ETHICS/MORAL ==========
    StudyDomain.ETHICS: [
        'ethic', 'moral', 'right', 'wrong', 'principl',
        'ethical reasoning', 'moral reasoning', 'moral philosophy',
        'normative', 'virtue', 'duty', 'obligation', 'conscience',
        'ethical behavior', 'moral behavior', 'ethical decision',
    ],
    StudyDomain.MORAL_JUDGMENT: [
        'moral', 'ethical', 'right', 'wrong', 'should', 'ought',
        'moral judgment', 'moral evaluation', 'ethical judgment',
        'moral foundation', 'moral intuition', 'moral cognition',
        'prescriptive', 'proscriptive', 'permissible', 'impermissible',
    ],
    StudyDomain.MORAL_DILEMMA: [
        'dilemma', 'trolley', 'sacrifice', 'utilitarian', 'deontolog',
        'trolley problem', 'footbridge', 'moral conflict', 'lesser evil',
        'moral trade-off', 'ethical dilemma', 'thought experiment',
        'lifeboat', 'consequential', 'kantian', 'virtue ethics',
    ],
    StudyDomain.ETHICAL_LEADERSHIP: [
        'ethical leader', 'integrity', 'honest leader', 'moral leader',
        'ethical management', 'leader integrity', 'principled leadership',
        'authentic leadership', 'ethical culture', 'ethical climate',
        'leadership ethics', 'moral courage', 'ethical role model',
    ],
    StudyDomain.CORPORATE_ETHICS: [
        'corporate', 'business ethics', 'csr', 'corporate social',
        'corporate responsibility', 'corporate governance', 'esg',
        'stakeholder', 'shareholder', 'corporate sustainability',
        'ethical business', 'corporate misconduct', 'whistleblower',
        'compliance', 'corporate fraud', 'greenwashing',
    ],
    StudyDomain.RESEARCH_ETHICS: [
        'research ethics', 'informed consent', 'deception', 'irb',
        'institutional review board', 'ethics committee', 'ethical approval',
        'participant rights', 'debriefing', 'confidentiality',
        'research misconduct', 'fabrication', 'falsification', 'plagiarism',
        'responsible conduct', 'ethical guidelines', 'helsinki declaration',
    ],
    StudyDomain.MORAL_EMOTIONS: [
        'guilt', 'shame', 'pride', 'moral emotion', 'regret',
        'moral outrage', 'indignation', 'contempt', 'disgust',
        'righteous anger', 'moral elevation', 'moral awe',
        'embarrassment', 'remorse', 'repentance', 'moral distress',
    ],
    StudyDomain.VALUES: [
        'values', 'priorities', 'what matters', 'important to me',
        'personal values', 'core values', 'value system', 'belief system',
        'schwartz values', 'moral values', 'cultural values',
        'value orientation', 'value hierarchy', 'terminal values',
        'instrumental values', 'value conflict', 'value alignment',
    ],

    # ========== ENVIRONMENTAL ==========
    StudyDomain.ENVIRONMENTAL: [
        'environment', 'climate', 'green', 'sustainab', 'eco', 'carbon',
        'ecological', 'ecosystem', 'biodiversity', 'pollution',
        'environmental concern', 'environmental awareness', 'planet',
        'natural resources', 'deforestation', 'ocean', 'plastic',
        'waste', 'environmental attitude', 'environmental psychology',
    ],
    StudyDomain.SUSTAINABILITY: [
        'sustainab', 'green', 'eco', 'renewable', 'recycle',
        'sustainable development', 'circular economy', 'zero waste',
        'upcycle', 'compost', 'reuse', 'reduce', 'carbon neutral',
        'sustainable living', 'sustainable consumption', 'sdg',
        'sustainable development goals', 'triple bottom line',
    ],
    StudyDomain.CLIMATE_ATTITUDES: [
        'climate', 'global warming', 'greenhouse', 'carbon',
        'climate change', 'climate skeptic', 'climate denier',
        'climate belief', 'climate concern', 'climate opinion',
        'climate anxiety', 'eco-anxiety', 'solastalgia',
        'climate science', 'ipcc', 'paris agreement', 'cop',
    ],
    StudyDomain.PRO_ENVIRONMENTAL: [
        'environmentally friendly', 'green behavior', 'eco-friendly',
        'pro-environmental', 'pro environmental', 'green action',
        'environmental behavior', 'ecological behavior',
        'environmental activism', 'environmental volunteering',
        'green living', 'carbon reduction', 'low carbon',
    ],
    StudyDomain.GREEN_CONSUMPTION: [
        'green product', 'eco product', 'sustainable product', 'ethical consumption',
        'green purchasing', 'eco-label', 'organic', 'fair trade',
        'ethically sourced', 'conscious consumer', 'sustainable fashion',
        'green marketing', 'eco-packaging', 'local food', 'farm to table',
    ],
    StudyDomain.CONSERVATION: [
        'conserv', 'protect', 'preserve', 'wildlife', 'nature',
        'conservation behavior', 'national park', 'endangered species',
        'habitat', 'restoration', 'marine conservation', 'forest',
        'rewilding', 'biodiversity loss', 'species protection',
    ],
    StudyDomain.ENERGY_BEHAVIOR: [
        'energy', 'electricity', 'power', 'solar', 'renewable',
        'energy saving', 'energy conservation', 'energy efficiency',
        'smart meter', 'thermostat', 'energy consumption', 'kilowatt',
        'energy audit', 'insulation', 'green building', 'leed',
        'carbon offset', 'energy transition',
    ],
    StudyDomain.ENVIRONMENTAL_JUSTICE: [
        'environmental justice', 'pollution', 'affected communities',
        'environmental racism', 'toxic waste', 'environmental inequality',
        'disproportionate exposure', 'environmental burden',
        'frontline communities', 'nimby', 'environmental health',
        'contamination', 'brownfield', 'superfund',
    ],

    # ========== COGNITIVE PSYCHOLOGY ==========
    StudyDomain.COGNITIVE: [
        'cognitive', 'thinking', 'mental process', 'mind',
        'cognition', 'cognitive psychology', 'perception', 'judgment',
        'cognitive science', 'mental representation', 'schema',
        'information processing', 'dual process', 'system 1', 'system 2',
    ],
    StudyDomain.DECISION_MAKING: [
        'decision', 'choice', 'select', 'option', 'alternative',
        'decision making', 'decide', 'choosing', 'preference',
        'trade-off', 'tradeoff', 'dilemma', 'evaluation', 'judgment',
        'optimal choice', 'rational choice', 'satisficing', 'maximizing',
    ],
    StudyDomain.MEMORY: [
        'remember', 'recall', 'forget', 'memory', 'recognition',
        'working memory', 'short-term memory', 'long-term memory',
        'episodic memory', 'semantic memory', 'procedural memory',
        'encoding', 'retrieval', 'amnesia', 'nostalgia', 'flashbulb',
        'false memory', 'memory distortion', 'recognition memory',
    ],
    StudyDomain.ATTENTION: [
        'attention', 'focus', 'concentrate', 'distract',
        'selective attention', 'divided attention', 'sustained attention',
        'attentional bias', 'inattentional blindness', 'change blindness',
        'vigilance', 'alertness', 'mind wandering', 'daydream',
        'attentional control', 'executive attention', 'adhd',
    ],
    StudyDomain.REASONING: [
        'reason', 'logic', 'infer', 'deduc', 'induc',
        'logical reasoning', 'deductive reasoning', 'inductive reasoning',
        'syllogism', 'analogy', 'causal reasoning', 'abductive',
        'probabilistic reasoning', 'bayesian', 'critical thinking',
    ],
    StudyDomain.PROBLEM_SOLVING: [
        'problem', 'solve', 'solution', 'figure out',
        'problem solving', 'insight', 'eureka', 'aha moment',
        'trial and error', 'strategy', 'heuristic', 'algorithm',
        'creative problem', 'complex problem', 'puzzle', 'riddle',
    ],
    StudyDomain.COGNITIVE_BIAS: [
        'bias', 'heuristic', 'shortcut', 'cognitive error',
        'cognitive bias', 'availability heuristic', 'representativeness',
        'anchoring bias', 'sunk cost fallacy', 'hindsight bias',
        'dunning-kruger', 'overconfidence', 'optimism bias',
        'negativity bias', 'status quo bias', 'endowment effect',
        'framing effect', 'primacy effect', 'recency effect',
    ],
    StudyDomain.METACOGNITION: [
        'metacognit', 'thinking about thinking', 'self-aware', 'monitoring',
        'metacognitive', 'metacognition', 'calibration', 'confidence',
        'judgment of learning', 'feeling of knowing', 'tip of tongue',
        'self-monitoring', 'self-regulation', 'cognitive monitoring',
        'metamemory', 'metacomprehension',
    ],

    # ========== DEVELOPMENTAL ==========
    StudyDomain.DEVELOPMENTAL: [
        'develop', 'age', 'life stage', 'mature',
        'developmental psychology', 'lifespan', 'life span',
        'growth', 'developmental stage', 'milestone', 'maturation',
        'ontogeny', 'developmental trajectory',
    ],
    StudyDomain.PARENTING: [
        'parent', 'child', 'mother', 'father', 'raising',
        'parenting style', 'authoritative', 'authoritarian', 'permissive',
        'helicopter parent', 'attachment parenting', 'co-parenting',
        'discipline', 'child rearing', 'maternal', 'paternal',
        'family', 'caregiver', 'childcare', 'parental involvement',
    ],
    StudyDomain.CHILDHOOD: [
        'child', 'kid', 'young', 'grow up', 'adolescent',
        'childhood', 'infant', 'toddler', 'preschool', 'teenager',
        'youth', 'adolescence', 'puberty', 'juvenile', 'minor',
        'developmental stage', 'pediatric', 'early childhood',
    ],
    StudyDomain.AGING: [
        'aging', 'older', 'elderly', 'senior', 'retirement',
        'gerontology', 'geriatric', 'old age', 'later life',
        'age-related', 'cognitive decline', 'ageism', 'active aging',
        'successful aging', 'longevity', 'centenarian', 'dementia',
        'alzheimer', 'memory decline', 'elder care',
    ],
    StudyDomain.LIFE_TRANSITIONS: [
        'transition', 'change', 'new phase', 'moving on',
        'life transition', 'major change', 'turning point',
        'milestone', 'graduation', 'marriage', 'divorce', 'relocation',
        'empty nest', 'career change', 'loss', 'bereavement', 'grief',
    ],
    StudyDomain.INTERGENERATIONAL: [
        'generation', 'millennial', 'boomer', 'gen z', 'gen x',
        'generational', 'baby boomer', 'generation x', 'generation z',
        'gen alpha', 'generation gap', 'intergenerational',
        'generational difference', 'cohort', 'age group', 'age cohort',
    ],

    # ========== CLINICAL ==========
    StudyDomain.CLINICAL: [
        'clinical', 'disorder', 'symptom', 'diagnosis', 'treatment',
        'clinical psychology', 'psychopathology', 'dsm', 'icd',
        'psychiatric', 'mental illness', 'psychological disorder',
        'clinical intervention', 'evidence-based treatment', 'cbt',
        'cognitive behavioral', 'psychotherapy', 'clinical trial',
    ],
    StudyDomain.ANXIETY: [
        'anxiety', 'anxious', 'worry', 'nervous', 'panic',
        'anxiety disorder', 'generalized anxiety', 'gad',
        'social anxiety', 'phobia', 'agoraphobia', 'ocd',
        'obsessive compulsive', 'fear', 'apprehension', 'dread',
        'anxiety attack', 'hypervigilance', 'anxiousness',
    ],
    StudyDomain.DEPRESSION: [
        'depress', 'sad', 'hopeless', 'low mood', 'suicid',
        'depression', 'depressive', 'major depressive', 'mdd',
        'anhedonia', 'melancholy', 'dysthymia', 'persistent depressive',
        'bipolar', 'mood disorder', 'antidepressant', 'ssri',
        'hopelessness', 'helplessness', 'worthlessness',
    ],
    StudyDomain.COPING: [
        'cope', 'deal with', 'handle', 'manage', 'adapt',
        'coping strategy', 'coping mechanism', 'coping style',
        'problem-focused', 'emotion-focused', 'avoidant coping',
        'active coping', 'adaptive coping', 'maladaptive coping',
        'coping resource', 'resilience', 'stress management',
    ],
    StudyDomain.THERAPY_ATTITUDES: [
        'therapy', 'counseling', 'psychologist', 'mental health treatment',
        'therapy seeking', 'help-seeking', 'treatment seeking',
        'stigma', 'mental health stigma', 'therapy attitude',
        'counselor', 'psychiatrist', 'treatment barrier',
        'therapeutic alliance', 'therapy outcome', 'psychotherapy',
        'mindfulness', 'mindfulness-based', 'mbsr', 'mbct',
    ],
    StudyDomain.STRESS: [
        'stress', 'pressure', 'overwhelm', 'burnout', 'strain',
        'stressor', 'acute stress', 'chronic stress', 'daily hassle',
        'life event', 'stress response', 'cortisol', 'fight or flight',
        'perceived stress', 'pss', 'stress appraisal', 'taxing',
        'demanding', 'overloaded', 'stress coping',
    ],

    # ========== COMMUNICATION ==========
    StudyDomain.COMMUNICATION: [
        'communicat', 'message', 'information', 'convey',
        'communication', 'discourse', 'dialogue', 'rhetoric',
        'speech', 'language', 'framing', 'audience', 'sender',
        'receiver', 'channel', 'feedback', 'nonverbal', 'verbal',
    ],
    StudyDomain.PERSUASION: [
        'persuad', 'convinc', 'influence', 'argument', 'appeal',
        'persuasion', 'persuasive', 'elaboration likelihood', 'elm',
        'central route', 'peripheral route', 'attitude change',
        'counter-argument', 'source credibility', 'message framing',
        'inoculation', 'resistance to persuasion', 'reactance',
    ],
    StudyDomain.MEDIA_EFFECTS: [
        'media effect', 'media impact', 'influence of media',
        'media exposure', 'cultivation theory', 'agenda setting',
        'framing theory', 'priming', 'third-person effect',
        'media violence', 'media representation', 'media influence',
        'screen time', 'media literacy', 'media psychology',
    ],
    StudyDomain.INTERPERSONAL: [
        'interpersonal', 'relationship', 'interaction', 'social',
        'interpersonal communication', 'face-to-face', 'conversation',
        'self-disclosure', 'empathy', 'listening', 'rapport',
        'interpersonal conflict', 'negotiation', 'compromise',
        'social skills', 'emotional intelligence', 'eq',
    ],
    StudyDomain.PUBLIC_OPINION: [
        'public opinion', 'popular', 'majority think', 'general public',
        'public sentiment', 'public attitudes', 'opinion poll',
        'survey research', 'public perception', 'public view',
        'societal attitudes', 'collective opinion', 'zeitgeist',
    ],
    StudyDomain.NARRATIVE: [
        'story', 'narrative', 'told', 'account', 'anecdote',
        'storytelling', 'narrative persuasion', 'transportation',
        'narrative engagement', 'story structure', 'narrative identity',
        'personal narrative', 'counter-narrative', 'testimonial',
        'lived experience', 'vignette', 'scenario',
    ],

    # ========== ECONOMICS ==========
    StudyDomain.ECONOMICS: [
        'economic', 'money', 'financial', 'market', 'trade',
        'economy', 'gdp', 'inflation', 'recession', 'supply',
        'demand', 'microeconomic', 'macroeconomic', 'fiscal',
        'monetary', 'economics', 'economist', 'economic theory',
    ],
    StudyDomain.NEGOTIATION: [
        'negotiat', 'bargain', 'deal', 'agreement', 'compromise',
        'negotiation', 'concession', 'mediation', 'arbitration',
        'batna', 'best alternative', 'distributive', 'integrative',
        'win-win', 'impasse', 'negotiation strategy', 'dealmaking',
    ],
    StudyDomain.BARGAINING: [
        'bargain', 'haggle', 'price negotiation', 'deal-making',
        'bargaining power', 'counter-offer', 'offer', 'bid',
        'take it or leave it', 'opening offer', 'final offer',
    ],
    StudyDomain.FINANCIAL_DECISION: [
        'financial decision', 'money decision', 'investment', 'spend',
        'financial choice', 'spending decision', 'allocation',
        'budget', 'expenditure', 'financial planning', 'portfolio',
        'asset allocation', 'diversification', 'roi',
    ],
    StudyDomain.SAVING_BEHAVIOR: [
        'save', 'saving', 'retirement fund', 'emergency fund',
        'savings', 'piggy bank', 'rainy day fund', 'nest egg',
        'saving rate', 'saving habit', 'undersaving', 'oversaving',
        'automatic saving', 'commitment device', 'mental accounting',
    ],
    StudyDomain.ECONOMIC_EXPECTATIONS: [
        'economic outlook', 'expect', 'forecast', 'prediction',
        'economic expectation', 'consumer confidence', 'economic sentiment',
        'economic optimism', 'economic pessimism', 'future economy',
        'economic forecast', 'market expectation', 'inflation expectation',
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

    # ========== PARENT / BROAD DOMAINS (missing keywords) ==========

    StudyDomain.BEHAVIORAL_ECONOMICS: [
        'behavioral economics', 'behavioural economics', 'economic experiment',
        'experimental economics', 'economic game', 'economic decision',
        'economic behavior', 'bounded rationality', 'prospect theory',
        'nudge', 'choice architecture', 'behavioral finance',
    ],
    StudyDomain.SOCIAL_PSYCHOLOGY: [
        'social psychology', 'social behavior', 'social cognition',
        'social perception', 'social judgment', 'interpersonal behavior',
        'group behavior', 'group dynamics', 'social interaction',
        'social processes', 'social phenomena', 'bystander', 'obedience',
    ],
    StudyDomain.ORGANIZATIONAL: [
        'organizational', 'organisation', 'workplace behavior',
        'organizational psychology', 'industrial organizational',
        'i-o psychology', 'io psychology', 'work psychology',
        'organizational development', 'org behavior', 'ob research',
    ],
    StudyDomain.TECHNOLOGY: [
        'technology', 'tech', 'digital', 'software', 'hardware', 'app',
        'application', 'device', 'gadget', 'computer', 'internet',
        'web', 'platform', 'system', 'tool', 'innovation',
    ],

    # ========== FINANCIAL PSYCHOLOGY (6 domains) ==========

    StudyDomain.FINANCIAL_PSYCHOLOGY: [
        'financial psychology', 'money psychology', 'financial behavior',
        'financial attitudes', 'money attitudes', 'financial wellbeing',
        'economic psychology', 'financial decision making',
    ],
    StudyDomain.FINANCIAL_LITERACY: [
        'financial literacy', 'financial knowledge', 'financial education',
        'money management', 'financial skills', 'financial competence',
        'economic literacy', 'numeracy', 'compound interest',
    ],
    StudyDomain.INVESTMENT_BEHAVIOR: [
        'investment', 'investing', 'stock', 'portfolio', 'asset',
        'stock market', 'trading', 'investor', 'returns', 'bonds',
        'mutual fund', 'etf', 'cryptocurrency', 'crypto', 'bitcoin',
        'market volatility', 'investment decision', 'risk tolerance',
    ],
    StudyDomain.DEBT_ATTITUDES: [
        'debt', 'credit', 'loan', 'borrow', 'mortgage', 'credit card',
        'credit score', 'indebtedness', 'debt management', 'debt stress',
        'student loan', 'payday loan', 'debt aversion', 'debt attitudes',
    ],
    StudyDomain.RETIREMENT_PLANNING: [
        'retirement', 'pension', '401k', '401(k)', 'ira', 'superannuation',
        'retirement planning', 'retirement savings', 'retirement age',
        'social security', 'retirement income', 'retire early',
    ],
    StudyDomain.FINANCIAL_STRESS: [
        'financial stress', 'money worry', 'financial anxiety', 'money stress',
        'financial strain', 'financial hardship', 'economic hardship',
        'financial insecurity', 'financial distress', 'poverty',
        'financial burden', 'money problems', 'can\'t afford',
    ],

    # ========== PERSONALITY PSYCHOLOGY (6 domains) ==========

    StudyDomain.PERSONALITY: [
        'personality', 'trait', 'temperament', 'disposition', 'character',
        'individual differences', 'personality type', 'personality assessment',
        'personality measure', 'personality scale', 'personality inventory',
    ],
    StudyDomain.BIG_FIVE: [
        'big five', 'big 5', 'ocean', 'openness', 'conscientiousness',
        'extraversion', 'agreeableness', 'neuroticism', 'extroversion',
        'introversion', 'five factor', 'ffm', 'neo-pi', 'neo pi',
        'personality traits', 'tipi', 'bfi',
    ],
    StudyDomain.NARCISSISM: [
        'narcissism', 'narcissist', 'narcissistic', 'grandiosity',
        'self-admiration', 'entitlement', 'superiority', 'vanity',
        'npi', 'narcissistic personality', 'covert narcissism',
    ],
    StudyDomain.DARK_TRIAD: [
        'dark triad', 'machiavellianism', 'psychopathy', 'dark tetrad',
        'sadism', 'subclinical psychopathy', 'dark personality',
        'malevolent personality', 'aversive personality',
    ],
    StudyDomain.TRAIT_ASSESSMENT: [
        'trait assessment', 'trait measurement', 'personality test',
        'trait evaluation', 'self-report measure', 'personality questionnaire',
        'trait rating', 'character assessment', 'dispositional measure',
    ],
    StudyDomain.SELF_CONCEPT: [
        'self-concept', 'self concept', 'self-identity', 'self-schema',
        'self-knowledge', 'self-understanding', 'self-construal',
        'self-perception', 'self-awareness', 'self-reflection',
        'possible selves', 'ideal self', 'actual self',
    ],

    # ========== GAMING/ENTERTAINMENT (6 domains) ==========

    StudyDomain.GAMING_PSYCHOLOGY: [
        'gaming', 'video game', 'videogame', 'gamer', 'game play',
        'gameplay', 'game design', 'game addiction', 'gaming disorder',
        'gaming motivation', 'game engagement', 'game experience',
        'console', 'pc gaming', 'mobile gaming', 'game behavior',
    ],
    StudyDomain.ESPORTS: [
        'esports', 'e-sports', 'competitive gaming', 'professional gaming',
        'esport', 'tournament', 'league of legends', 'valorant', 'csgo',
        'dota', 'overwatch', 'pro gamer', 'esports team',
    ],
    StudyDomain.GAMBLING: [
        'gambling', 'gamble', 'casino', 'betting', 'poker', 'slot',
        'lottery', 'wager', 'sports betting', 'online gambling',
        'problem gambling', 'gambling addiction', 'gambling behavior',
        'gambling disorder', 'responsible gambling',
    ],
    StudyDomain.ENTERTAINMENT_MEDIA: [
        'entertainment', 'movie', 'film', 'television', 'tv show',
        'series', 'streaming', 'netflix', 'disney', 'music',
        'podcast', 'media consumption', 'binge watching', 'cinema',
        'media entertainment', 'leisure media',
    ],
    StudyDomain.STREAMING_BEHAVIOR: [
        'streaming', 'stream', 'twitch', 'youtube live', 'live stream',
        'content creator', 'streamer', 'subscriber', 'donation',
        'streaming platform', 'streaming service', 'binge watch',
        'on demand', 'cord cutting', 'cord cutter',
    ],
    StudyDomain.VIRTUAL_REALITY: [
        'virtual reality', 'vr', 'augmented reality', 'ar', 'xr',
        'mixed reality', 'metaverse', 'immersive', 'headset',
        'oculus', 'quest', 'hololens', 'spatial computing',
        'vr experience', 'virtual environment', '3d environment',
    ],

    # ========== SOCIAL MEDIA RESEARCH (6 domains) ==========

    StudyDomain.SOCIAL_MEDIA_USE: [
        'social media use', 'social media behavior', 'social media habits',
        'social media consumption', 'social media engagement', 'posting',
        'sharing online', 'social media effects', 'screen time social',
        'social networking', 'sns', 'social media frequency',
    ],
    StudyDomain.ONLINE_IDENTITY: [
        'online identity', 'digital identity', 'online persona',
        'online self', 'online self-presentation', 'profile',
        'avatar', 'username', 'online reputation', 'digital self',
        'virtual identity', 'online impression management',
    ],
    StudyDomain.DIGITAL_COMMUNICATION: [
        'digital communication', 'online communication', 'texting',
        'messaging', 'email', 'chat', 'instant message', 'emoji',
        'computer-mediated communication', 'cmc', 'video call',
        'zoom', 'teams', 'slack', 'asynchronous communication',
    ],
    StudyDomain.INFLUENCER_MARKETING: [
        'influencer', 'influencer marketing', 'sponsored content',
        'brand ambassador', 'sponsored post', 'paid partnership',
        'micro-influencer', 'macro-influencer', 'creator economy',
        'social media marketing', 'influencer endorsement',
    ],
    StudyDomain.ONLINE_COMMUNITIES: [
        'online community', 'online group', 'forum', 'subreddit',
        'discord', 'community', 'member', 'moderator',
        'virtual community', 'online forum', 'message board',
        'community engagement', 'community building',
    ],
    StudyDomain.SOCIAL_COMPARISON: [
        'social comparison', 'compare self', 'upward comparison',
        'downward comparison', 'comparison', 'comparing myself',
        'compare to others', 'keeping up', 'envy', 'jealousy',
        'relative deprivation', 'better than', 'worse than',
    ],

    # ========== WORKPLACE BEHAVIOR (6 domains) ==========

    StudyDomain.REMOTE_WORK: [
        'remote work', 'work from home', 'wfh', 'telecommute',
        'telecommuting', 'telework', 'hybrid work', 'remote employee',
        'distributed team', 'virtual team', 'home office',
        'flexible work', 'remote collaboration', 'work remotely',
    ],
    StudyDomain.WORKPLACE_DIVERSITY: [
        'workplace diversity', 'diversity', 'inclusion', 'dei',
        'diversity equity inclusion', 'representation', 'inclusive',
        'minority', 'underrepresented', 'equal opportunity',
        'affirmative action', 'workplace equity', 'belonging',
    ],
    StudyDomain.BURNOUT: [
        'burnout', 'burn out', 'burned out', 'exhaustion',
        'emotional exhaustion', 'depersonalization', 'cynicism',
        'work fatigue', 'job burnout', 'occupational burnout',
        'compassion fatigue', 'caregiver burnout', 'maslach',
    ],
    StudyDomain.CAREER_DEVELOPMENT: [
        'career', 'career development', 'career growth', 'promotion',
        'career path', 'professional development', 'career planning',
        'career advancement', 'career change', 'career transition',
        'mentoring', 'career satisfaction', 'career goal',
    ],
    StudyDomain.WORKPLACE_CONFLICT: [
        'workplace conflict', 'conflict at work', 'office conflict',
        'workplace dispute', 'coworker conflict', 'team conflict',
        'interpersonal conflict', 'organizational conflict',
        'work disagreement', 'workplace incivility', 'workplace bullying',
    ],
    StudyDomain.ORGANIZATIONAL_JUSTICE: [
        'organizational justice', 'workplace fairness', 'fair treatment',
        'distributive justice', 'interactional justice', 'voice',
        'procedural justice at work', 'employee fairness', 'equity theory',
        'unfair treatment', 'workplace justice', 'organizational fairness',
    ],

    # ========== DECISION SCIENCE (6 domains) ==========

    StudyDomain.DECISION_SCIENCE: [
        'decision science', 'decision theory', 'judgment and decision',
        'jdm', 'decision analysis', 'decision research',
        'behavioral decision', 'choice behavior', 'decision process',
    ],
    StudyDomain.CHOICE_ARCHITECTURE: [
        'choice architecture', 'option presentation', 'choice design',
        'decision environment', 'choice set', 'menu design',
        'option framing', 'choice structure', 'decision structure',
    ],
    StudyDomain.NUDGE: [
        'nudge', 'nudging', 'behavioral nudge', 'gentle push',
        'default option', 'libertarian paternalism', 'behavioral intervention',
        'choice default', 'opt-in', 'opt-out', 'soft paternalism',
    ],
    StudyDomain.DEFAULT_EFFECTS: [
        'default effect', 'default option', 'status quo', 'opt-out',
        'pre-selected', 'default choice', 'inertia', 'default bias',
        'status quo bias', 'default setting', 'automatic enrollment',
    ],
    StudyDomain.INFORMATION_OVERLOAD: [
        'information overload', 'choice overload', 'too many options',
        'decision fatigue', 'cognitive overload', 'analysis paralysis',
        'option overload', 'information saturation', 'overwhelmed by choices',
        'paradox of choice', 'overchoice', 'decision difficulty',
    ],
    StudyDomain.REGRET: [
        'regret', 'anticipated regret', 'counterfactual', 'what if',
        'should have', 'would have', 'missed opportunity',
        'regret aversion', 'decision regret', 'action regret',
        'inaction regret', 'buyer remorse', 'post-decision regret',
    ],

    # ========== TRUST & CREDIBILITY (5 domains) ==========

    StudyDomain.INSTITUTIONAL_TRUST: [
        'institutional trust', 'trust in institutions', 'trust government',
        'trust in government', 'public trust', 'trust in democracy',
        'trust in parliament', 'trust in congress', 'trust in judiciary',
        'systemic trust', 'confidence in institutions',
    ],
    StudyDomain.EXPERT_CREDIBILITY: [
        'expert credibility', 'expert trust', 'expert authority',
        'expert opinion', 'trust expert', 'scientific expert',
        'expertise', 'qualified opinion', 'professional credibility',
        'expert knowledge', 'trust in experts',
    ],
    StudyDomain.SOURCE_CREDIBILITY: [
        'source credibility', 'credible source', 'information source',
        'source trustworthiness', 'reliable source', 'source quality',
        'source evaluation', 'news source', 'media credibility',
    ],
    StudyDomain.SCIENCE_TRUST: [
        'science trust', 'trust in science', 'scientific trust',
        'science credibility', 'trust scientists', 'science skepticism',
        'science denial', 'scientific consensus', 'peer review',
        'trust in research', 'science literacy',
    ],
    StudyDomain.MEDIA_TRUST: [
        'media trust', 'trust in media', 'news trust', 'media credibility',
        'journalism trust', 'press trust', 'mainstream media',
        'alternative media', 'media skepticism', 'news credibility',
        'media confidence', 'press credibility',
    ],

    # ========== INNOVATION & CREATIVITY (5 domains) ==========

    StudyDomain.INNOVATION: [
        'innovation', 'innovative', 'disruptive', 'breakthrough',
        'novel', 'novelty', 'new product development', 'invention',
        'innovative thinking', 'innovation adoption', 'innovation diffusion',
    ],
    StudyDomain.CREATIVITY: [
        'creativity', 'creative', 'creative thinking', 'divergent thinking',
        'brainstorm', 'ideation', 'imagination', 'creative problem solving',
        'artistic', 'original', 'inventive', 'creative performance',
    ],
    StudyDomain.ENTREPRENEURSHIP: [
        'entrepreneur', 'entrepreneurship', 'startup', 'start-up',
        'venture', 'founder', 'business creation', 'new venture',
        'entrepreneurial', 'small business', 'self-employed',
        'business plan', 'pitch', 'seed funding', 'angel investor',
    ],
    StudyDomain.IDEA_GENERATION: [
        'idea generation', 'ideation', 'brainstorming', 'concept development',
        'creative ideation', 'idea evaluation', 'idea selection',
        'problem finding', 'opportunity recognition',
    ],
    StudyDomain.CREATIVE_PROCESS: [
        'creative process', 'incubation', 'illumination', 'elaboration',
        'creative flow', 'flow state', 'inspiration', 'creative insight',
        'creative block', 'writer block', 'design thinking',
    ],

    # ========== RISK & SAFETY (5 domains) ==========

    StudyDomain.RISK_PERCEPTION: [
        'risk perception', 'perceived risk', 'risk assessment',
        'subjective risk', 'risk judgment', 'risk evaluation',
        'hazard assessment', 'threat perception', 'danger perception',
        'risk estimate', 'risk awareness',
    ],
    StudyDomain.SAFETY_ATTITUDES: [
        'safety attitudes', 'safety culture', 'safety climate',
        'safety behavior', 'safety compliance', 'safety motivation',
        'safety performance', 'safety management', 'occupational safety',
    ],
    StudyDomain.HAZARD_PERCEPTION: [
        'hazard perception', 'hazard awareness', 'hazard identification',
        'threat detection', 'danger recognition', 'hazard recognition',
        'driving hazard', 'workplace hazard', 'environmental hazard',
    ],
    StudyDomain.DISASTER_PREPAREDNESS: [
        'disaster preparedness', 'emergency preparedness', 'disaster planning',
        'evacuation', 'emergency response', 'natural disaster',
        'earthquake', 'hurricane', 'flood', 'wildfire', 'tornado',
        'crisis preparedness', 'disaster resilience', 'emergency kit',
    ],
    StudyDomain.RISK_COMMUNICATION: [
        'risk communication', 'communicating risk', 'risk messaging',
        'warning', 'risk information', 'risk disclosure', 'hazard warning',
        'public warning', 'risk advisory', 'safety communication',
    ],

    # ========== AI ALIGNMENT & ETHICS (6 domains - v2.4.5) ==========

    StudyDomain.AI_ALIGNMENT: [
        'ai alignment', 'alignment', 'value alignment', 'aligned ai',
        'ai values', 'ai goals', 'beneficial ai', 'ai purpose',
        'superintelligence', 'existential risk', 'ai risk',
    ],
    StudyDomain.AI_ETHICS: [
        'ai ethics', 'ethical ai', 'responsible ai', 'ai morality',
        'ai fairness', 'ai accountability', 'ai responsibility',
        'ai principles', 'ethical technology', 'tech ethics',
    ],
    StudyDomain.AI_SAFETY: [
        'ai safety', 'safe ai', 'ai containment', 'ai control',
        'ai reliability', 'ai robustness', 'ai failure',
        'ai malfunction', 'ai accident', 'ai harm',
    ],
    StudyDomain.MACHINE_VALUES: [
        'machine values', 'machine ethics', 'moral machine',
        'robot ethics', 'machine morality', 'computational ethics',
        'artificial morality', 'machine conscience',
    ],
    StudyDomain.AI_GOVERNANCE: [
        'ai governance', 'ai regulation', 'ai policy', 'ai law',
        'ai legislation', 'ai oversight', 'ai standards',
        'ai regulatory', 'governing ai', 'ai framework',
    ],
    StudyDomain.AI_TRANSPARENCY: [
        'ai transparency', 'explainable ai', 'xai', 'interpretable',
        'ai explainability', 'black box', 'model transparency',
        'algorithmic transparency', 'ai interpretability', 'glass box',
    ],

    # ========== CLIMATE SCIENCE & ACTION (6 domains - v2.4.5) ==========

    StudyDomain.CLIMATE_ACTION: [
        'climate action', 'climate activism', 'climate protest',
        'climate movement', 'climate strike', 'climate advocacy',
        'extinction rebellion', 'fridays for future', 'climate engagement',
    ],
    StudyDomain.CLIMATE_COMMUNICATION: [
        'climate communication', 'climate messaging', 'climate narrative',
        'climate framing', 'climate discourse', 'climate education',
        'climate information', 'communicating climate',
    ],
    StudyDomain.CARBON_FOOTPRINT: [
        'carbon footprint', 'carbon emission', 'carbon offset',
        'carbon neutral', 'carbon reduction', 'carbon tax',
        'carbon trading', 'co2', 'greenhouse gas', 'emission',
        'net zero', 'carbon capture', 'carbon budget',
    ],
    StudyDomain.CLIMATE_ADAPTATION: [
        'climate adaptation', 'climate resilience', 'adaptation strategy',
        'climate vulnerability', 'climate impact', 'sea level rise',
        'heat wave', 'drought', 'extreme weather', 'climate migration',
    ],
    StudyDomain.CLIMATE_JUSTICE: [
        'climate justice', 'climate equity', 'environmental justice',
        'climate inequality', 'disproportionate impact', 'just transition',
        'climate refugees', 'climate vulnerable', 'developing nations',
    ],
    StudyDomain.RENEWABLE_ENERGY: [
        'renewable energy', 'solar energy', 'wind energy', 'wind power',
        'solar power', 'clean energy', 'green energy', 'renewable',
        'photovoltaic', 'wind turbine', 'solar panel', 'geothermal',
        'hydroelectric', 'biofuel', 'energy transition',
    ],

    # ========== HEALTH DISPARITIES (6 domains - v2.4.5) ==========

    StudyDomain.HEALTH_DISPARITIES: [
        'health disparities', 'health disparity', 'health inequality',
        'health inequity', 'racial health gap', 'ethnic health gap',
        'socioeconomic health', 'health gap', 'unequal health',
    ],
    StudyDomain.HEALTHCARE_ACCESS: [
        'healthcare access', 'access to care', 'health insurance',
        'uninsured', 'underinsured', 'healthcare barrier', 'medicaid',
        'medicare', 'universal healthcare', 'affordable care',
        'healthcare cost', 'medical debt', 'healthcare coverage',
    ],
    StudyDomain.HEALTH_EQUITY: [
        'health equity', 'equitable health', 'fair health',
        'health justice', 'equal health', 'health for all',
        'universal health', 'equitable care', 'equitable access',
    ],
    StudyDomain.SOCIAL_DETERMINANTS: [
        'social determinants', 'social determinants of health', 'sdoh',
        'neighborhood', 'zip code', 'socioeconomic status', 'ses',
        'income inequality', 'food desert', 'housing instability',
        'education and health', 'poverty and health',
    ],
    StudyDomain.HEALTH_LITERACY: [
        'health literacy', 'health information', 'health comprehension',
        'medical literacy', 'patient education', 'health communication',
        'informed patient', 'understanding diagnosis', 'reading labels',
    ],
    StudyDomain.MEDICAL_MISTRUST: [
        'medical mistrust', 'healthcare mistrust', 'distrust doctors',
        'distrust medicine', 'tuskegee', 'medical experimentation',
        'vaccine hesitancy', 'medical skepticism', 'provider distrust',
        'historical mistrust', 'health system distrust',
    ],

    # ========== GENOMICS & PERSONALIZED MEDICINE (5 domains - v2.4.5) ==========

    StudyDomain.GENOMICS: [
        'genomics', 'genome', 'dna', 'gene', 'genetic', 'genetics',
        'chromosome', 'crispr', 'gene editing', 'sequencing',
        'whole genome', 'genomic data', 'precision medicine',
    ],
    StudyDomain.GENETIC_TESTING: [
        'genetic testing', 'genetic test', 'genetic screening',
        'prenatal testing', 'carrier testing', 'pharmacogenomics',
        '23andme', 'ancestry dna', 'direct-to-consumer genetic',
        'dtc genetic', 'genetic risk', 'genetic counseling',
    ],
    StudyDomain.PERSONALIZED_MEDICINE: [
        'personalized medicine', 'precision medicine', 'targeted therapy',
        'individualized treatment', 'tailored treatment', 'biomarker',
        'companion diagnostic', 'pharmacogenetics', 'personalized care',
    ],
    StudyDomain.GENE_THERAPY: [
        'gene therapy', 'gene treatment', 'genetic therapy',
        'gene editing therapy', 'crispr treatment', 'viral vector',
        'gene replacement', 'somatic gene', 'germline',
    ],
    StudyDomain.BIOETHICS: [
        'bioethics', 'medical ethics', 'clinical ethics', 'bioethical',
        'genetic ethics', 'reproductive ethics', 'end of life',
        'euthanasia', 'assisted suicide', 'organ donation', 'consent',
        'human subjects', 'research participant', 'informed consent',
    ],

    # ========== DIGITAL SOCIETY (5 domains - v2.4.5) ==========

    StudyDomain.DIGITAL_DIVIDE: [
        'digital divide', 'digital gap', 'digital inequality',
        'internet access', 'broadband', 'connectivity',
        'digital exclusion', 'tech access', 'digital poverty',
        'digital haves', 'digital have-nots',
    ],
    StudyDomain.ONLINE_POLARIZATION: [
        'online polarization', 'echo chamber', 'filter bubble',
        'political polarization online', 'polarized online', 'tribalism',
        'online extremism', 'radicalization', 'online divide',
    ],
    StudyDomain.ALGORITHMIC_FAIRNESS: [
        'algorithmic fairness', 'algorithmic bias', 'ai bias',
        'machine bias', 'fairness in ml', 'fair algorithm',
        'biased algorithm', 'discriminatory algorithm', 'ai discrimination',
        'algorithmic discrimination', 'fair machine learning',
    ],
    StudyDomain.DATA_PRIVACY: [
        'data privacy', 'data protection', 'personal data',
        'privacy concern', 'privacy policy', 'gdpr', 'ccpa',
        'data ownership', 'data rights', 'information privacy',
        'privacy regulation', 'data consent', 'surveillance capitalism',
    ],
    StudyDomain.DIGITAL_LITERACY: [
        'digital literacy', 'digital skills', 'digital competence',
        'media literacy', 'information literacy', 'internet literacy',
        'tech literacy', 'computational thinking', 'digital fluency',
    ],

    # ========== FUTURE OF WORK (5 domains - v2.4.5) ==========

    StudyDomain.AUTOMATION_ANXIETY: [
        'automation anxiety', 'job automation', 'robot replace',
        'ai replace job', 'technological unemployment', 'job displacement',
        'automation fear', 'automation threat', 'jobs at risk',
        'automated away', 'machine replace', 'fear of automation',
    ],
    StudyDomain.GIG_ECONOMY: [
        'gig economy', 'gig work', 'freelance', 'independent contractor',
        'uber', 'lyft', 'doordash', 'fiverr', 'upwork', 'task rabbit',
        'platform work', 'contingent work', 'gig worker', 'side hustle',
    ],
    StudyDomain.SKILLS_OBSOLESCENCE: [
        'skills obsolescence', 'reskilling', 'upskilling', 'skill gap',
        'skills mismatch', 'lifelong learning', 'continuous learning',
        'skill development', 'workforce training', 'skill shortage',
    ],
    StudyDomain.UNIVERSAL_BASIC_INCOME: [
        'universal basic income', 'ubi', 'basic income', 'guaranteed income',
        'unconditional income', 'cash transfer', 'income guarantee',
        'negative income tax', 'citizen dividend', 'basic income experiment',
    ],
    StudyDomain.HUMAN_MACHINE_COLLABORATION: [
        'human-machine collaboration', 'human machine collaboration',
        'human-ai collaboration', 'human ai teamwork', 'augmentation',
        'collaborative intelligence', 'human-machine teaming',
        'hybrid intelligence', 'centaur', 'cobots', 'cobot',
    ],
}


# ============================================================================
# COMPOUND DOMAIN PHRASES - High-confidence multi-word indicators
# ============================================================================
# These phrases are strong domain signals. When found, they receive very high
# scores (10 points) to ensure the correct domain wins over incidental keyword
# matches from other domains.

COMPOUND_DOMAIN_PHRASES: Dict[StudyDomain, List[str]] = {
    # Political Science - compound phrases that strongly indicate political content
    StudyDomain.POLITICAL: [
        'political polarization', 'political attitudes', 'political ideology',
        'political beliefs', 'political identity', 'political behavior',
        'political engagement', 'political participation', 'political opinion',
        'political discourse', 'political communication', 'political psychology',
        'political preferences', 'political affiliation', 'political orientation',
        'political views', 'political climate', 'political division',
    ],
    StudyDomain.POLARIZATION: [
        'political polarization', 'partisan polarization', 'ideological polarization',
        'affective polarization', 'attitude polarization', 'group polarization',
        'opinion polarization', 'polarized attitudes', 'increasingly divided',
        'partisan divide', 'ideological divide', 'political divide',
    ],
    StudyDomain.PARTISANSHIP: [
        'partisan identity', 'party identification', 'partisan bias',
        'party loyalty', 'partisan media', 'cross-party', 'bipartisan',
    ],
    StudyDomain.MISINFORMATION: [
        'fake news', 'misinformation spread', 'fact checking', 'false information',
        'news credibility', 'media literacy', 'information accuracy',
        'misinformation belief', 'conspiracy theory', 'debunking myths',
    ],

    # Consumer/Marketing - compound phrases for consumer studies
    StudyDomain.CONSUMER: [
        'consumer trust', 'consumer behavior', 'consumer attitudes',
        'consumer preferences', 'consumer decision', 'consumer perception',
        'consumer confidence', 'buying behavior', 'purchase behavior',
        'consumer choice', 'consumer experience', 'consumer research',
        'consumer psychology', 'consumer response', 'consumer willingness',
    ],
    StudyDomain.BRAND: [
        'brand trust', 'brand perception', 'brand awareness', 'brand loyalty',
        'brand attitude', 'brand preference', 'brand image',
    ],
    StudyDomain.ADVERTISING: [
        'advertising effectiveness', 'ad exposure', 'marketing message',
        'promotional content', 'advertising appeal', 'ad credibility',
    ],
    StudyDomain.PRODUCT_EVALUATION: [
        'product quality', 'product review', 'product rating', 'product trust',
        'product label', 'product information', 'product evaluation',
    ],

    # Technology/AI - compound phrases specific to tech-as-subject studies
    StudyDomain.AI_ATTITUDES: [
        'artificial intelligence attitudes', 'ai ethics', 'ai trust',
        'ai adoption', 'trust in ai', 'ai decision making', 'ai bias',
        'ai fairness', 'ai transparency', 'ai regulation', 'ai governance',
        'ai anxiety', 'ai literacy', 'autonomous vehicle', 'self-driving car',
        'robot interaction', 'ai-generated', 'machine learning model',
    ],
    StudyDomain.HUMAN_AI_INTERACTION: [
        'human-ai interaction', 'human ai interaction', 'human-robot interaction',
        'chatbot interaction', 'virtual assistant use', 'ai assistant',
        'conversational agent', 'talk to robot',
    ],

    # Behavioral Economics
    StudyDomain.BEHAVIORAL_ECONOMICS: [
        'behavioral economics', 'economic decision', 'economic game',
        'experimental economics', 'economic behavior',
    ],
    StudyDomain.DICTATOR_GAME: [
        'dictator game', 'allocation decision', 'give or keep',
    ],
    StudyDomain.TRUST_GAME: [
        'trust game', 'investment game', 'send and return',
    ],
    StudyDomain.PROSOCIAL: [
        'prosocial behavior', 'prosocial motivation', 'helping behavior',
        'charitable giving', 'altruistic behavior', 'dictator game',
    ],

    # Clinical Psychology
    StudyDomain.CLINICAL: [
        'clinical psychology', 'psychological treatment', 'clinical intervention',
        'clinical trial', 'clinical assessment', 'mental health treatment',
    ],
    StudyDomain.ANXIETY: [
        'anxiety disorder', 'anxiety symptoms', 'anxiety treatment',
        'generalized anxiety', 'social anxiety', 'anxiety intervention',
        'mindfulness for anxiety', 'anxiety reduction', 'anxiety management',
    ],
    StudyDomain.DEPRESSION: [
        'major depression', 'depressive symptoms', 'depression treatment',
        'clinical depression', 'depression intervention',
    ],
    StudyDomain.THERAPY_ATTITUDES: [
        'therapy attitudes', 'therapy seeking', 'mental health treatment',
        'counseling attitudes', 'psychotherapy attitudes',
        'mindfulness intervention', 'mindfulness-based',
    ],

    # Social Psychology
    StudyDomain.INTERGROUP: [
        'intergroup relations', 'intergroup conflict', 'intergroup bias',
        'intergroup contact', 'us versus them', 'outgroup attitudes',
    ],
    StudyDomain.SOCIAL_INFLUENCE: [
        'social influence', 'social pressure', 'social proof', 'peer influence',
    ],

    # Health Psychology
    StudyDomain.HEALTH: [
        'health behavior', 'health outcomes', 'health intervention',
        'public health', 'health promotion', 'health education',
    ],
    StudyDomain.MENTAL_HEALTH: [
        'mental health', 'psychological wellbeing', 'emotional wellbeing',
        'mental wellness', 'mental health awareness', 'mental health stigma',
    ],
    StudyDomain.WELLBEING: [
        'subjective wellbeing', 'psychological wellbeing', 'life satisfaction',
        'quality of life', 'emotional wellbeing',
    ],

    # Education
    StudyDomain.EDUCATION: [
        'educational intervention', 'learning outcomes', 'educational psychology',
        'student learning', 'academic performance',
    ],

    # Environmental
    StudyDomain.ENVIRONMENTAL: [
        'environmental attitudes', 'environmental behavior', 'climate change',
        'environmental concern', 'environmental policy',
    ],
    StudyDomain.CLIMATE_ATTITUDES: [
        'climate change', 'global warming', 'climate attitudes', 'climate policy',
        'climate action', 'carbon footprint',
    ],

    # Organizational Behavior
    StudyDomain.WORKPLACE: [
        'workplace behavior', 'workplace attitudes', 'workplace outcomes',
        'workplace environment', 'organizational behavior', 'employee behavior',
    ],
    StudyDomain.LEADERSHIP: [
        'leadership style', 'leadership effectiveness', 'transformational leadership',
        'servant leadership', 'ethical leadership', 'leadership behavior',
        'leadership development', 'leader-member exchange', 'lmx',
    ],
    StudyDomain.JOB_SATISFACTION: [
        'job satisfaction', 'work satisfaction', 'employee satisfaction',
        'job attitudes', 'workplace satisfaction',
    ],
    StudyDomain.BURNOUT: [
        'job burnout', 'workplace burnout', 'emotional exhaustion',
        'occupational burnout', 'burnout syndrome', 'burnout prevention',
    ],
    StudyDomain.REMOTE_WORK: [
        'remote work', 'work from home', 'hybrid work', 'telecommuting',
        'distributed team', 'virtual teamwork', 'remote collaboration',
    ],

    # Financial Psychology
    StudyDomain.FINANCIAL_PSYCHOLOGY: [
        'financial psychology', 'financial behavior', 'money psychology',
        'financial decision making', 'financial attitudes',
    ],
    StudyDomain.INVESTMENT_BEHAVIOR: [
        'investment behavior', 'investment decision', 'stock market behavior',
        'investor psychology', 'portfolio management',
    ],

    # Personality Psychology
    StudyDomain.PERSONALITY: [
        'personality psychology', 'personality traits', 'individual differences',
        'personality assessment', 'personality measurement',
    ],
    StudyDomain.BIG_FIVE: [
        'big five personality', 'five factor model', 'ocean model',
        'personality dimensions', 'neo personality',
    ],
    StudyDomain.DARK_TRIAD: [
        'dark triad personality', 'dark personality traits',
        'subclinical psychopathy', 'subclinical narcissism',
    ],

    # Decision Science
    StudyDomain.DECISION_SCIENCE: [
        'decision science', 'judgment and decision making',
        'behavioral decision research', 'decision theory',
    ],
    StudyDomain.NUDGE: [
        'behavioral nudge', 'nudge intervention', 'choice architecture',
        'libertarian paternalism', 'default option',
    ],

    # Technology/AI - additional
    StudyDomain.TECHNOLOGY_ADOPTION: [
        'technology adoption', 'technology acceptance', 'tam model',
        'technology acceptance model', 'unified theory of acceptance',
        'utaut', 'technology readiness', 'tech adoption',
    ],
    StudyDomain.AUTOMATION: [
        'automation attitudes', 'automated systems', 'automation trust',
        'autonomous systems', 'self-driving car', 'autonomous vehicle',
    ],
    StudyDomain.ALGORITHM_AVERSION: [
        'algorithm aversion', 'algorithmic decision', 'ai decision making',
        'automated decision making', 'algorithmic judgment',
    ],
    StudyDomain.DIGITAL_WELLBEING: [
        'digital wellbeing', 'digital well-being', 'screen time effects',
        'smartphone addiction', 'internet addiction', 'tech addiction',
    ],
    StudyDomain.CYBERSECURITY: [
        'cybersecurity behavior', 'online security', 'password behavior',
        'phishing awareness', 'information security',
    ],

    # Health - additional
    StudyDomain.VACCINATION: [
        'vaccine hesitancy', 'vaccination intention', 'vaccine acceptance',
        'vaccine attitudes', 'vaccine confidence', 'immunization behavior',
    ],
    StudyDomain.HEALTH_BEHAVIOR: [
        'health behavior change', 'health behavior model', 'health belief model',
        'theory of planned behavior', 'health promotion', 'preventive behavior',
    ],

    # Gaming/Entertainment
    StudyDomain.GAMING_PSYCHOLOGY: [
        'gaming psychology', 'video game effects', 'gaming behavior',
        'gaming addiction', 'gaming disorder', 'game engagement',
    ],
    StudyDomain.GAMBLING: [
        'gambling behavior', 'problem gambling', 'gambling addiction',
        'gambling psychology', 'responsible gambling',
    ],
    StudyDomain.VIRTUAL_REALITY: [
        'virtual reality experience', 'vr research', 'immersive technology',
        'augmented reality experience', 'mixed reality',
    ],

    # AI Ethics & Governance
    StudyDomain.AI_ETHICS: [
        'ai ethics', 'ethical ai', 'responsible ai', 'ai fairness',
        'ai accountability', 'trustworthy ai',
    ],
    StudyDomain.AI_GOVERNANCE: [
        'ai governance', 'ai regulation', 'ai policy', 'governing artificial intelligence',
    ],
    StudyDomain.AI_TRANSPARENCY: [
        'explainable ai', 'ai transparency', 'algorithmic transparency',
        'interpretable machine learning', 'xai research',
    ],

    # Social Media Research
    StudyDomain.SOCIAL_MEDIA_USE: [
        'social media use', 'social media effects', 'social media behavior',
        'social media consumption', 'social networking behavior',
    ],
    StudyDomain.INFLUENCER_MARKETING: [
        'influencer marketing', 'social media influencer', 'influencer endorsement',
        'sponsored content', 'influencer credibility',
    ],
    StudyDomain.SOCIAL_COMPARISON: [
        'social comparison theory', 'upward social comparison',
        'downward social comparison', 'social media comparison',
    ],
}


# ============================================================================
# DOMAIN NEGATIVE KEYWORDS - Prevent misidentification
# ============================================================================
# When these keywords are present alongside a domain's positive keywords,
# they indicate the text is NOT primarily about that domain. Each entry maps
# a domain to keywords that, when co-occurring with domain keywords, should
# reduce confidence in that domain assignment.

DOMAIN_NEGATIVE_KEYWORDS: Dict[StudyDomain, List[str]] = {
    # Technology/AI should NOT match when the study is primarily about:
    StudyDomain.AI_ATTITUDES: [
        'political polarization', 'partisan', 'voting behavior', 'election',
        'political attitudes', 'political ideology', 'democrat', 'republican',
        'dictator game', 'ultimatum game', 'public goods game',
        'prosocial behavior', 'charitable giving',
        'mindfulness intervention', 'anxiety treatment', 'therapy',
        'clinical trial', 'depression treatment',
    ],
    StudyDomain.AUTOMATION: [
        'political polarization', 'partisan', 'voting behavior',
        'dictator game', 'prosocial behavior',
    ],
    StudyDomain.ALGORITHM_AVERSION: [
        'political polarization', 'partisan', 'voting behavior',
    ],

    # Political domains should NOT match for pure tech/AI studies
    StudyDomain.POLITICAL: [
        'autonomous vehicle', 'self-driving', 'robot interaction',
        'chatbot', 'neural network', 'machine learning model',
        'deep learning', 'language model',
        'product evaluation', 'brand loyalty', 'consumer behavior',
    ],

    # Social Psychology TRUST should not override political trust or consumer trust
    StudyDomain.TRUST: [
        'autonomous vehicle', 'self-driving', 'ai system',
        'dictator game', 'ultimatum game', 'public goods game',
    ],

    # Consumer/Marketing should not match for political studies
    StudyDomain.CONSUMER: [
        'political polarization', 'partisan divide', 'voting behavior',
        'political ideology', 'election outcome',
        'anxiety disorder', 'depression treatment', 'clinical trial',
    ],

    # Clinical should not match for general health behavior studies
    StudyDomain.CLINICAL: [
        'consumer behavior', 'product evaluation', 'brand loyalty',
        'purchase intent', 'advertising',
        'political polarization', 'voting behavior', 'election',
    ],

    # Education should not match when context is clearly non-educational
    StudyDomain.EDUCATION: [
        'dictator game', 'ultimatum game', 'product evaluation',
        'brand loyalty', 'consumer behavior', 'advertising',
        'clinical trial', 'depression treatment', 'anxiety disorder',
    ],

    # Decision-making is very broad - prevent it from stealing from specific domains
    StudyDomain.DECISION_MAKING: [
        'political polarization', 'partisan', 'voting behavior',
        'product evaluation', 'brand loyalty', 'consumer behavior',
        'clinical trial', 'depression treatment', 'anxiety disorder',
        'dictator game', 'ultimatum game', 'public goods game',
    ],

    # Motivation is extremely broad - prevent false matches
    StudyDomain.MOTIVATION: [
        'political polarization', 'partisan', 'voting behavior',
        'product evaluation', 'brand loyalty', 'consumer behavior',
        'clinical trial', 'anxiety disorder', 'depression treatment',
        'dictator game', 'ultimatum game', 'public goods game',
        'autonomous vehicle', 'self-driving', 'ai system',
    ],

    # Stress is broad - should not steal from clinical or workplace domains
    StudyDomain.STRESS: [
        'product evaluation', 'brand loyalty', 'consumer behavior',
        'political polarization', 'voting behavior', 'election',
        'dictator game', 'ultimatum game',
    ],

    # Interpersonal is very broad - prevent it from stealing from specific domains
    StudyDomain.INTERPERSONAL: [
        'political polarization', 'partisan', 'voting behavior',
        'product evaluation', 'brand loyalty', 'consumer behavior',
        'dictator game', 'ultimatum game', 'public goods game',
        'autonomous vehicle', 'self-driving', 'ai system',
    ],

    # Coping is broad - prevent it from overriding more specific clinical domains
    StudyDomain.COPING: [
        'product evaluation', 'brand loyalty', 'consumer behavior',
        'political polarization', 'voting behavior',
        'dictator game', 'ultimatum game',
    ],

    # Gambling should not match lottery/gamble in behavioral economics contexts
    StudyDomain.GAMBLING: [
        'dictator game', 'ultimatum game', 'trust game',
        'risk preference', 'time preference', 'loss aversion',
    ],

    # Identity is very broad
    StudyDomain.IDENTITY: [
        'brand identity', 'brand image', 'corporate identity',
        'product evaluation', 'consumer behavior',
    ],
}


# ============================================================================
# DOMAIN CATEGORY GROUPS - For priority ordering
# ============================================================================
# Maps high-level categories to their constituent domains. When a category
# accumulates enough signal, its domains get a priority boost.

DOMAIN_CATEGORIES: Dict[str, List[StudyDomain]] = {
    'political_science': [
        StudyDomain.POLITICAL, StudyDomain.POLARIZATION, StudyDomain.PARTISANSHIP,
        StudyDomain.VOTING, StudyDomain.MEDIA, StudyDomain.POLICY_ATTITUDES,
        StudyDomain.CIVIC_ENGAGEMENT, StudyDomain.POLITICAL_TRUST,
        StudyDomain.IDEOLOGY, StudyDomain.MISINFORMATION,
    ],
    'consumer_marketing': [
        StudyDomain.CONSUMER, StudyDomain.BRAND, StudyDomain.ADVERTISING,
        StudyDomain.PRODUCT_EVALUATION, StudyDomain.PURCHASE_INTENT,
        StudyDomain.BRAND_LOYALTY, StudyDomain.PRICE_PERCEPTION,
        StudyDomain.SERVICE_QUALITY, StudyDomain.CUSTOMER_SATISFACTION,
        StudyDomain.WORD_OF_MOUTH,
    ],
    'technology_ai': [
        StudyDomain.TECHNOLOGY, StudyDomain.AI_ATTITUDES, StudyDomain.PRIVACY,
        StudyDomain.AUTOMATION, StudyDomain.ALGORITHM_AVERSION,
        StudyDomain.TECHNOLOGY_ADOPTION, StudyDomain.SOCIAL_MEDIA,
        StudyDomain.DIGITAL_WELLBEING, StudyDomain.HUMAN_AI_INTERACTION,
        StudyDomain.CYBERSECURITY,
    ],
    'behavioral_economics': [
        StudyDomain.BEHAVIORAL_ECONOMICS, StudyDomain.DICTATOR_GAME,
        StudyDomain.PUBLIC_GOODS, StudyDomain.TRUST_GAME,
        StudyDomain.ULTIMATUM_GAME, StudyDomain.PRISONERS_DILEMMA,
        StudyDomain.RISK_PREFERENCE, StudyDomain.TIME_PREFERENCE,
        StudyDomain.LOSS_AVERSION, StudyDomain.FRAMING_EFFECTS,
        StudyDomain.ANCHORING, StudyDomain.SUNK_COST,
    ],
    'clinical_psychology': [
        StudyDomain.CLINICAL, StudyDomain.ANXIETY, StudyDomain.DEPRESSION,
        StudyDomain.COPING, StudyDomain.THERAPY_ATTITUDES, StudyDomain.STRESS,
    ],
    'social_psychology': [
        StudyDomain.SOCIAL_PSYCHOLOGY, StudyDomain.INTERGROUP,
        StudyDomain.IDENTITY, StudyDomain.NORMS, StudyDomain.CONFORMITY,
        StudyDomain.PROSOCIAL, StudyDomain.TRUST, StudyDomain.FAIRNESS,
        StudyDomain.COOPERATION, StudyDomain.SOCIAL_INFLUENCE,
        StudyDomain.ATTRIBUTION, StudyDomain.STEREOTYPE,
        StudyDomain.PREJUDICE, StudyDomain.SELF_ESTEEM, StudyDomain.EMPATHY,
    ],
    'health_psychology': [
        StudyDomain.HEALTH, StudyDomain.MEDICAL_DECISION, StudyDomain.WELLBEING,
        StudyDomain.HEALTH_BEHAVIOR, StudyDomain.MENTAL_HEALTH,
        StudyDomain.VACCINATION, StudyDomain.PAIN_MANAGEMENT,
        StudyDomain.HEALTH_ANXIETY, StudyDomain.PATIENT_PROVIDER,
        StudyDomain.CHRONIC_ILLNESS,
    ],
    'education': [
        StudyDomain.EDUCATION, StudyDomain.LEARNING,
        StudyDomain.ACADEMIC_MOTIVATION, StudyDomain.TEACHING_EFFECTIVENESS,
        StudyDomain.ONLINE_LEARNING, StudyDomain.EDUCATIONAL_TECHNOLOGY,
        StudyDomain.STUDENT_ENGAGEMENT, StudyDomain.ASSESSMENT_FEEDBACK,
    ],
    'environmental': [
        StudyDomain.ENVIRONMENTAL, StudyDomain.SUSTAINABILITY,
        StudyDomain.CLIMATE_ATTITUDES, StudyDomain.PRO_ENVIRONMENTAL,
        StudyDomain.GREEN_CONSUMPTION, StudyDomain.CONSERVATION,
        StudyDomain.ENERGY_BEHAVIOR, StudyDomain.ENVIRONMENTAL_JUSTICE,
        StudyDomain.CLIMATE_ACTION, StudyDomain.CLIMATE_COMMUNICATION,
        StudyDomain.CARBON_FOOTPRINT, StudyDomain.CLIMATE_ADAPTATION,
        StudyDomain.CLIMATE_JUSTICE, StudyDomain.RENEWABLE_ENERGY,
    ],
    'organizational_behavior': [
        StudyDomain.ORGANIZATIONAL, StudyDomain.WORKPLACE,
        StudyDomain.LEADERSHIP, StudyDomain.TEAMWORK,
        StudyDomain.MOTIVATION, StudyDomain.JOB_SATISFACTION,
        StudyDomain.ORGANIZATIONAL_COMMITMENT, StudyDomain.WORK_LIFE_BALANCE,
        StudyDomain.EMPLOYEE_ENGAGEMENT, StudyDomain.ORGANIZATIONAL_CULTURE,
        StudyDomain.REMOTE_WORK, StudyDomain.WORKPLACE_DIVERSITY,
        StudyDomain.BURNOUT, StudyDomain.CAREER_DEVELOPMENT,
        StudyDomain.WORKPLACE_CONFLICT, StudyDomain.ORGANIZATIONAL_JUSTICE,
    ],
    'financial_psychology': [
        StudyDomain.FINANCIAL_PSYCHOLOGY, StudyDomain.FINANCIAL_LITERACY,
        StudyDomain.INVESTMENT_BEHAVIOR, StudyDomain.DEBT_ATTITUDES,
        StudyDomain.RETIREMENT_PLANNING, StudyDomain.FINANCIAL_STRESS,
    ],
    'personality_psychology': [
        StudyDomain.PERSONALITY, StudyDomain.BIG_FIVE,
        StudyDomain.NARCISSISM, StudyDomain.DARK_TRIAD,
        StudyDomain.TRAIT_ASSESSMENT, StudyDomain.SELF_CONCEPT,
    ],
    'gaming_entertainment': [
        StudyDomain.GAMING_PSYCHOLOGY, StudyDomain.ESPORTS,
        StudyDomain.GAMBLING, StudyDomain.ENTERTAINMENT_MEDIA,
        StudyDomain.STREAMING_BEHAVIOR, StudyDomain.VIRTUAL_REALITY,
    ],
    'social_media_research': [
        StudyDomain.SOCIAL_MEDIA, StudyDomain.SOCIAL_MEDIA_USE,
        StudyDomain.ONLINE_IDENTITY, StudyDomain.DIGITAL_COMMUNICATION,
        StudyDomain.INFLUENCER_MARKETING, StudyDomain.ONLINE_COMMUNITIES,
        StudyDomain.SOCIAL_COMPARISON,
    ],
    'decision_science': [
        StudyDomain.DECISION_SCIENCE, StudyDomain.CHOICE_ARCHITECTURE,
        StudyDomain.NUDGE, StudyDomain.DEFAULT_EFFECTS,
        StudyDomain.INFORMATION_OVERLOAD, StudyDomain.REGRET,
        StudyDomain.DECISION_MAKING, StudyDomain.COGNITIVE_BIAS,
    ],
    'ethics_moral': [
        StudyDomain.ETHICS, StudyDomain.MORAL_JUDGMENT,
        StudyDomain.MORAL_DILEMMA, StudyDomain.ETHICAL_LEADERSHIP,
        StudyDomain.CORPORATE_ETHICS, StudyDomain.RESEARCH_ETHICS,
        StudyDomain.MORAL_EMOTIONS, StudyDomain.VALUES,
        StudyDomain.BIOETHICS,
    ],
    'ai_alignment_ethics': [
        StudyDomain.AI_ALIGNMENT, StudyDomain.AI_ETHICS,
        StudyDomain.AI_SAFETY, StudyDomain.MACHINE_VALUES,
        StudyDomain.AI_GOVERNANCE, StudyDomain.AI_TRANSPARENCY,
        StudyDomain.ALGORITHMIC_FAIRNESS,
    ],
    'trust_credibility': [
        StudyDomain.INSTITUTIONAL_TRUST, StudyDomain.EXPERT_CREDIBILITY,
        StudyDomain.SOURCE_CREDIBILITY, StudyDomain.SCIENCE_TRUST,
        StudyDomain.MEDIA_TRUST,
    ],
    'health_disparities': [
        StudyDomain.HEALTH_DISPARITIES, StudyDomain.HEALTHCARE_ACCESS,
        StudyDomain.HEALTH_EQUITY, StudyDomain.SOCIAL_DETERMINANTS,
        StudyDomain.HEALTH_LITERACY, StudyDomain.MEDICAL_MISTRUST,
    ],
    'innovation_creativity': [
        StudyDomain.INNOVATION, StudyDomain.CREATIVITY,
        StudyDomain.ENTREPRENEURSHIP, StudyDomain.IDEA_GENERATION,
        StudyDomain.CREATIVE_PROCESS,
    ],
    'risk_safety': [
        StudyDomain.RISK_PERCEPTION, StudyDomain.SAFETY_ATTITUDES,
        StudyDomain.HAZARD_PERCEPTION, StudyDomain.DISASTER_PREPAREDNESS,
        StudyDomain.RISK_COMMUNICATION,
    ],
    'future_of_work': [
        StudyDomain.AUTOMATION_ANXIETY, StudyDomain.GIG_ECONOMY,
        StudyDomain.SKILLS_OBSOLESCENCE, StudyDomain.UNIVERSAL_BASIC_INCOME,
        StudyDomain.HUMAN_MACHINE_COLLABORATION,
    ],
    'digital_society': [
        StudyDomain.DIGITAL_DIVIDE, StudyDomain.ONLINE_POLARIZATION,
        StudyDomain.ALGORITHMIC_FAIRNESS, StudyDomain.DATA_PRIVACY,
        StudyDomain.DIGITAL_LITERACY,
    ],
}


def _keyword_matches(keyword: str, text: str) -> bool:
    """Check if a keyword matches in text, using word boundaries for short keywords.

    Short keywords (< 4 characters) like 'ai', 'ad', 'ux' are matched with
    word boundary checks to prevent false positives (e.g., 'ai' matching
    inside 'certain', 'explain', 'contain').

    Longer keywords use simple substring matching since they are specific
    enough to avoid false positives.
    """
    if len(keyword) < 4:
        # Word boundary matching for short keywords to prevent false positives
        pattern = r'\b' + re.escape(keyword) + r'\b'
        return bool(re.search(pattern, text))
    else:
        return keyword in text


def _keyword_score(keyword: str) -> float:
    """Score a keyword match based on its specificity.

    Multi-word phrases are more specific signals than single words.
    Longer phrases indicate stronger domain affinity.

    Weighted scoring system:
    - Compound phrases (2+ words): 5.0 points (domain-specific, e.g., 'political polarization')
    - 3+ word phrases: 7.0 points (very specific, e.g., 'fundamental attribution error')
    - Domain-unique single keyword (6+ chars): 2.0 points (e.g., 'polarization')
    - Generic/common single keyword (< 6 chars): 1.0 point (e.g., 'trust', 'bias')
    """
    word_count = len(keyword.split())
    if word_count >= 3:
        return 7.0
    elif word_count == 2:
        return 5.0
    elif len(keyword) >= 6:
        return 2.0
    else:
        return 1.0


def _score_text_for_domain(
    text: str,
    domain: StudyDomain,
    keywords: List[str],
    multiplier: float = 1.0,
) -> float:
    """Score a text segment for a given domain with an optional multiplier.

    Args:
        text: The lowercased text to search in.
        domain: The domain being scored.
        keywords: The keyword list from DOMAIN_KEYWORDS.
        multiplier: Score multiplier (e.g., 3.0 for title text).

    Returns:
        Weighted score for this domain in this text segment.
    """
    score = 0.0
    for kw in keywords:
        if _keyword_matches(kw, text):
            score += _keyword_score(kw) * multiplier
    return score


def detect_study_domain(study_context: Dict[str, Any], question_text: str) -> StudyDomain:
    """Detect the most appropriate domain for response generation.

    Uses a 7-layer scoring system for accurate domain identification:

    1. Title-weighted compound phrase matching (10 pts * 3x in title)
       - Multi-word phrases like 'political polarization' are the strongest signals
    2. Title-weighted keyword matching (1-7 pts * 3x in title)
       - Keywords found in the study title score 3x those in the description
    3. Condition and DV name analysis (2x weight)
       - Condition names and DV/scale names provide strong domain signals
    4. Negative keyword penalties (-3 pts each)
       - Prevent misidentification when contextual clues exclude a domain
    5. Category-level boosting
       - Multiple sibling domains scoring strengthens the category signal
    6. Context-clue fallback for weak signals
       - When top score is low, use broader context clues to make a best guess
    7. Tie-breaking by domain specificity
       - Prefer more specific domains (e.g., POLARIZATION over POLITICAL)

    Short keywords (< 4 chars) use word-boundary matching to avoid false
    positives (e.g., 'ai' matching inside 'certain', 'explain', etc.).
    """
    # --- Extract title and description separately for differential weighting ---
    title_text = ''
    description_parts: List[str] = []
    condition_text = ''
    dv_text = ''

    if study_context:
        # Survey name / study title gets 3x weight
        title_text = str(study_context.get('survey_name', '')).lower()
        # Topics also get elevated weight (treated as title-level)
        topics = study_context.get('topics', [])
        if topics:
            title_text += ' ' + ' '.join(str(t) for t in topics).lower()
        # Study domain hint from user selection (if provided)
        domain_hint = str(study_context.get('study_domain', '')).lower()
        if domain_hint and domain_hint != 'general':
            title_text += ' ' + domain_hint.replace('_', ' ')
        # Instructions and questions are description-level (1x weight)
        instructions = str(study_context.get('instructions_text', '')).lower()
        if instructions:
            description_parts.append(instructions)
        main_questions = study_context.get('main_questions', [])
        if main_questions:
            description_parts.append(' '.join(str(q) for q in main_questions).lower())
        # Extract condition names for analysis (2x weight)
        conditions = study_context.get('conditions', [])
        if conditions:
            condition_text = ' '.join(str(c) for c in conditions).lower()
        # Extract DV / scale names for analysis (2x weight)
        dvs = study_context.get('dvs', study_context.get('scales', []))
        if dvs:
            if isinstance(dvs, list):
                dv_text = ' '.join(str(d) if isinstance(d, str) else str(d.get('name', '')) for d in dvs).lower()
            elif isinstance(dvs, dict):
                dv_text = ' '.join(str(v) for v in dvs.values()).lower()

    question_lower = str(question_text).lower()
    description_parts.append(question_lower)
    description_text = ' '.join(description_parts)

    # Full combined text for compound phrase and negative keyword checks
    combined = (title_text + ' ' + description_text + ' ' + condition_text + ' ' + dv_text).strip()

    # --- Layer 1: Compound phrase matching (highest priority) ---
    # Compound phrases score 10 pts each, 3x if found in title
    domain_scores: Dict[StudyDomain, float] = {}
    for domain, phrases in COMPOUND_DOMAIN_PHRASES.items():
        for phrase in phrases:
            if phrase in title_text:
                domain_scores[domain] = domain_scores.get(domain, 0.0) + 30.0  # 10 * 3x
            elif phrase in combined:
                domain_scores[domain] = domain_scores.get(domain, 0.0) + 10.0

    # --- Layer 2: Title-weighted keyword matching ---
    # Keywords in title get 3x multiplier; keywords in description get 1x
    for domain, keywords in DOMAIN_KEYWORDS.items():
        title_score = _score_text_for_domain(title_text, domain, keywords, multiplier=3.0)
        desc_score = _score_text_for_domain(description_text, domain, keywords, multiplier=1.0)
        total = title_score + desc_score
        if total > 0:
            domain_scores[domain] = domain_scores.get(domain, 0.0) + total

    # --- Layer 3: Condition and DV name analysis ---
    # Condition names (e.g., "AI_recommendation", "high_trust") and DV names
    # (e.g., "purchase_intent", "anxiety_scale") provide strong domain signals
    if condition_text or dv_text:
        context_extra = (condition_text + ' ' + dv_text).strip()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            ctx_score = _score_text_for_domain(context_extra, domain, keywords, multiplier=2.0)
            if ctx_score > 0:
                domain_scores[domain] = domain_scores.get(domain, 0.0) + ctx_score
        # Also check compound phrases in condition/DV text
        for domain, phrases in COMPOUND_DOMAIN_PHRASES.items():
            for phrase in phrases:
                if phrase in context_extra:
                    domain_scores[domain] = domain_scores.get(domain, 0.0) + 8.0

    if not domain_scores:
        # --- Layer 6 (early): Context-clue fallback ---
        return _fallback_domain_detection(combined)

    # --- Layer 4: Negative keyword penalties ---
    # Each negative keyword match reduces domain score by 3 pts
    for domain, neg_keywords in DOMAIN_NEGATIVE_KEYWORDS.items():
        if domain in domain_scores:
            penalty = 0.0
            for neg_kw in neg_keywords:
                if neg_kw in combined:
                    penalty += 3.0
            if penalty > 0:
                domain_scores[domain] = max(0.0, domain_scores[domain] - penalty)
                if domain_scores[domain] <= 0:
                    del domain_scores[domain]

    if not domain_scores:
        return _fallback_domain_detection(combined)

    # --- Layer 5: Category-level boosting ---
    # When 2+ domains in the same category score, boost all domains in that
    # category by 20% per additional scoring sibling. This helps political
    # studies where POLITICAL + POLARIZATION both score to outrank a single
    # AI_ATTITUDES domain that accumulated incidental keyword matches.
    for _cat_name, cat_domains in DOMAIN_CATEGORIES.items():
        scoring_siblings = [d for d in cat_domains if d in domain_scores]
        if len(scoring_siblings) >= 2:
            boost_factor = 1.0 + (len(scoring_siblings) - 1) * 0.2
            for d in scoring_siblings:
                domain_scores[d] *= boost_factor

    # --- Layer 6: Context-clue fallback for weak signals ---
    # When the top score is very low (< 3.0), the match is likely incidental.
    # Use broader context clues to determine domain.
    top_score = max(domain_scores.values())
    if top_score < 3.0:
        fallback = _fallback_domain_detection(combined)
        if fallback != StudyDomain.GENERAL:
            return fallback

    # --- Layer 7: Tie-breaking by domain specificity ---
    # When scores are very close (within 5%), prefer the more specific domain.
    # Specific domains (sub-domains) are preferred over broad parent domains.
    # This prevents POLITICAL from beating POLARIZATION when both score similarly.
    threshold = top_score * 0.95  # Within 5%
    top_candidates = {d: s for d, s in domain_scores.items() if s >= threshold}

    if len(top_candidates) > 1:
        # Among near-tied candidates, prefer sub-domains (those with more
        # specific keywords). Use the compound phrase count as a proxy for
        # specificity -- domains with compound phrase matches are more specific.
        # Also prefer domains with fewer total keywords in DOMAIN_KEYWORDS
        # (fewer keywords = more specific domain).
        def _specificity_key(item: Tuple[StudyDomain, float]) -> Tuple[float, int, int]:
            domain, score = item
            compound_count = 0
            if domain in COMPOUND_DOMAIN_PHRASES:
                for phrase in COMPOUND_DOMAIN_PHRASES[domain]:
                    if phrase in combined:
                        compound_count += 1
            # Specificity bonus: domains with fewer keywords are more specific
            keyword_count = len(DOMAIN_KEYWORDS.get(domain, []))
            specificity_bonus = max(0, 30 - keyword_count)  # Fewer keywords = higher bonus
            # Sort by (score, compound_count, specificity_bonus) descending
            return (score, compound_count, specificity_bonus)

        return max(top_candidates.items(), key=_specificity_key)[0]

    # Return domain with highest score
    return max(domain_scores.items(), key=lambda x: x[1])[0]


# Context-clue fallback patterns for broad domain detection when keyword
# matching fails. Maps regex patterns to domains. Patterns are checked in
# order; the first match wins.
_FALLBACK_CONTEXT_PATTERNS: List[Tuple[str, StudyDomain]] = [
    # Marketing / consumer patterns
    (r'\b(buy|bought|purchas|shop|brand|product|customer|retail|store)\b', StudyDomain.CONSUMER),
    (r'\b(advertis|ad campaign|commerc|marketing|promo)\b', StudyDomain.ADVERTISING),
    # Political patterns
    (r'\b(politic|democrat|republican|liberal|conservative|election|vote|congress|senate)\b', StudyDomain.POLITICAL),
    # Health patterns
    (r'\b(health|medical|doctor|patient|hospital|clinic|disease|illness|symptom)\b', StudyDomain.HEALTH),
    (r'\b(anxiety|anxious|depress|panic|phobia|trauma|ptsd)\b', StudyDomain.MENTAL_HEALTH),
    (r'\b(vaccin|immuniz|booster|shot|jab)\b', StudyDomain.VACCINATION),
    # Technology / AI patterns
    (r'\b(ai|artificial intelligence|algorithm|machine learning|robot|chatbot|gpt|llm)\b', StudyDomain.AI_ATTITUDES),
    (r'\b(social media|facebook|instagram|tiktok|twitter|youtube)\b', StudyDomain.SOCIAL_MEDIA),
    # Education patterns
    (r'\b(school|university|college|student|teacher|classroom|curriculum|learn)\b', StudyDomain.EDUCATION),
    # Workplace patterns
    (r'\b(employee|workplace|job|manager|boss|supervisor|office|company|firm)\b', StudyDomain.WORKPLACE),
    (r'\b(leader|leadership|manager|ceo|executive)\b', StudyDomain.LEADERSHIP),
    # Environmental patterns
    (r'\b(environment|climate|sustainab|eco-|green|carbon|recycl|pollution)\b', StudyDomain.ENVIRONMENTAL),
    # Behavioral economics patterns
    (r'\b(dictator|allocat|split|endowment|recipien)\b', StudyDomain.DICTATOR_GAME),
    (r'\b(risk|gamble|lottery|probability|certain)\b', StudyDomain.RISK_PREFERENCE),
    # Ethics patterns
    (r'\b(ethic|moral|right|wrong|dilemma|virtue|duty)\b', StudyDomain.ETHICS),
    # Relationship patterns
    (r'\b(relationship|partner|couple|romantic|dating|marriage|love)\b', StudyDomain.RELATIONSHIP),
    # Food patterns
    (r'\b(food|eating|diet|nutrition|meal|appetite|hunger)\b', StudyDomain.FOOD_PSYCHOLOGY),
]


def _fallback_domain_detection(text: str) -> StudyDomain:
    """Fallback domain detection using broad context-clue regex patterns.

    Called when primary keyword matching yields no results or very low scores.
    Uses broader regex patterns to make a best guess at the domain from
    general context clues in the study description.

    Args:
        text: The lowercased combined text to analyze.

    Returns:
        The best-guess domain, or GENERAL if no patterns match.
    """
    if not text or not text.strip():
        return StudyDomain.GENERAL

    for pattern, domain in _FALLBACK_CONTEXT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return domain

    return StudyDomain.GENERAL


# ============================================================================
# RESPONSE VARIATIONS AND MODIFIERS
# ============================================================================

# v1.1.0.3: Overhauled phrase banks to remove telltale academic patterns.
# Real survey respondents do NOT write "From my perspective," or "Furthermore,".
# All phrases must sound like they were typed quickly in a text box.
HEDGING_PHRASES = [
    "I think ", "I guess ", "I feel like ", "maybe ",
    "I'd say ", "probably ", "not sure but ", "I suppose ",
]

CONNECTORS = [
    " and ", " plus ", " also ", " but also ",
    " and also ", " — and ", ", and ", " on top of that ",
]

CONCLUDING_PHRASES = [
    "", "", "", "", "", "",  # 75% no conclusion — most people just stop
    ".", ".",
]

CASUAL_MODIFIERS = [
    "honestly", "basically", "like", "just", "kinda", "really", "literally",
]

FORMAL_MODIFIERS = [
    "certainly", "particularly", "especially", "notably",
]


def add_variation(response: str, persona_verbosity: float, persona_formality: float, local_rng: random.Random = None) -> str:
    """Add lightweight natural variation without telltale patterns.

    v1.1.0.3: OVERHAULED — removed all academic-sounding append phrases
    ("This aligns with what I've read", "From what I understand, this is a
    common perspective", etc.) that no real survey respondent ever writes.
    Now only applies: hedging for uncertain personas, casual filler for
    informal personas, and connector phrases for verbose personas.  All
    additions sound like real human speech, not essay conclusions.
    """
    rng = local_rng or random.Random()
    result = response

    # Hedging for less confident / lower-verbosity personas
    if rng.random() < (1 - persona_verbosity) * 0.35:
        hedge = rng.choice(HEDGING_PHRASES)
        if not result.startswith(('I ', 'My ', 'i ', 'my ')) and len(result) >= 2:
            result = hedge + result[0].lower() + result[1:]

    # Casual filler insertion for informal personas (natural position)
    if persona_formality < 0.4 and rng.random() < 0.25:
        modifier = rng.choice(CASUAL_MODIFIERS)
        words = result.split()
        if len(words) > 4:
            insert_pos = rng.randint(1, min(3, len(words) - 1))
            words.insert(insert_pos, modifier)
            result = ' '.join(words)

    # Verbose personas sometimes add a brief continuation (NOT academic)
    if persona_verbosity > 0.65 and rng.random() < 0.20:
        _continuations = [
            " and I stand by that",
            " which is really what it comes down to for me",
            " and I don't think that's going to change anytime soon",
            " and honestly most people I know feel the same way",
        ]
        result = result.rstrip('.!?') + rng.choice(_continuations)

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

    # v1.1.0.3: Sentence starters cleaned — removed academic patterns
    # ("Upon consideration,", "In my estimation,", "My position is that").
    # Real survey respondents use simple, direct openings.
    SENTENCE_STARTERS = [
        # Common openers (how real people actually start sentences)
        "I think ", "I feel like ", "honestly ", "I'd say ",
        "for me ", "the way I see it ", "I guess ",
        "from my experience ", "I mean ", "look ",
        # Slightly more considered
        "after thinking about it ", "my take is ",
        "the thing is ", "what I've noticed is ",
        "I've always thought ", "my gut says ",
    ]

    # v1.1.0.3: Transition phrases cleaned of academic-essay patterns.
    # Real survey respondents write "and", "but", "also" — NOT "Furthermore,"
    # "Consequently,", "What's particularly relevant is".
    TRANSITION_PHRASES = [
        # How real people connect thoughts
        "Also ", "And ", "Plus ", "But ", "But also ",
        "On top of that, ", "Another thing is ", "And honestly ",
        # Contrast
        "That said, ", "But at the same time, ", "Then again, ",
        "At the same time though, ", "Still, ",
    ]

    CONCLUDING_PHRASES = [
        # Real people mostly just STOP writing — no formal conclusion.
        "", "", "", "", "", "", "", "",
        # Rare casual wrap-ups (real people say these, not essay conclusions)
        "That's basically it.", "So yeah.", "Anyway.",
    ]

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.study_context: Dict[str, Any] = {}
        # Initialize fresh session for uniqueness tracking
        ComprehensiveResponseGenerator._session_id += 1
        ComprehensiveResponseGenerator._used_responses = set()
        ComprehensiveResponseGenerator._used_sentences = set()
        # v1.1.0.4: Track used corpus responses to prevent repetition
        self._used_corpus_indices: Dict[str, set] = {}  # key: domain|sentiment → set of indices

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
                if len(stripped) >= 2:
                    varied = starter + stripped[0].lower() + stripped[1:]
                    norm = self._normalize_sentence(varied)
                    if norm not in ComprehensiveResponseGenerator._used_sentences:
                        return varied

            elif strategy == 1:
                # v1.4.11: Expanded qualifier prefixes
                qualifiers = [
                    "Honestly, ", "In truth, ", "Frankly, ",
                    "To be fair, ", "Realistically, ", "Genuinely, ",
                    "Looking back, ", "On reflection, ", "In retrospect, ",
                    "All things considered, ", "At the end of the day, ",
                    "When I think about it, ", "From what I can tell, ",
                    "In my experience, ", "As far as I can tell, ",
                ]
                qualifier = local_rng.choice(qualifiers)
                varied = qualifier + sentence[0].lower() + sentence[1:] if len(sentence) >= 2 else sentence
                norm = self._normalize_sentence(varied)
                if norm not in ComprehensiveResponseGenerator._used_sentences:
                    return varied

            elif strategy == 2:
                # v1.4.11: Expanded synonym map from 5 → 30+ words
                synonym_map = {
                    'good': ['fine', 'decent', 'solid', 'positive', 'favorable'],
                    'bad': ['poor', 'inadequate', 'subpar', 'unfavorable', 'weak'],
                    'very': ['quite', 'really', 'particularly', 'especially', 'notably'],
                    'interesting': ['intriguing', 'noteworthy', 'compelling', 'engaging', 'thought-provoking'],
                    'important': ['significant', 'crucial', 'essential', 'vital', 'key'],
                    'think': ['believe', 'feel', 'consider', 'sense', 'suspect'],
                    'like': ['appreciate', 'enjoy', 'prefer', 'favor', 'value'],
                    'understand': ['grasp', 'comprehend', 'see', 'follow', 'recognize'],
                    'agree': ['concur', 'align', 'share that view', 'support that'],
                    'different': ['distinct', 'varied', 'divergent', 'separate', 'contrasting'],
                    'clear': ['obvious', 'evident', 'apparent', 'straightforward', 'plain'],
                    'difficult': ['challenging', 'tough', 'hard', 'complex', 'demanding'],
                    'easy': ['simple', 'straightforward', 'effortless', 'manageable'],
                    'helpful': ['useful', 'beneficial', 'valuable', 'constructive'],
                    'confusing': ['unclear', 'ambiguous', 'puzzling', 'muddled'],
                    'reasonable': ['fair', 'sensible', 'rational', 'sound', 'logical'],
                    'experience': ['encounter', 'involvement', 'interaction', 'exposure'],
                    'concerned': ['worried', 'troubled', 'uneasy', 'anxious'],
                    'surprised': ['taken aback', 'startled', 'caught off guard'],
                    'comfortable': ['at ease', 'content', 'relaxed', 'settled'],
                    'certain': ['sure', 'confident', 'convinced', 'positive'],
                    'probably': ['likely', 'presumably', 'perhaps', 'possibly'],
                    'overall': ['on the whole', 'all things considered', 'generally'],
                    'somewhat': ['moderately', 'fairly', 'to some extent', 'partially'],
                    'really': ['truly', 'genuinely', 'honestly', 'absolutely'],
                    'suggest': ['indicate', 'imply', 'point to', 'recommend'],
                    'impact': ['effect', 'influence', 'consequence', 'outcome'],
                    'approach': ['method', 'strategy', 'way', 'technique'],
                    'improve': ['enhance', 'strengthen', 'better', 'refine'],
                    'situation': ['circumstance', 'scenario', 'context', 'case'],
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
                # v1.4.11: Expanded ending rephrases
                suffixes = [
                    ", I think", ", in my view", ", as I see it",
                    ", from my perspective", ", to my mind",
                    ", at least in my opinion", ", or so it seems",
                    ", if I'm being honest", ", which is how I feel about it",
                    ", and that's my take", ", as far as I'm concerned",
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
        return hedge + sentence[0].lower() + sentence[1:] if len(sentence) >= 2 else sentence

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

        # Fallback: add a natural-sounding unique suffix (not academic)
        unique_suffix = local_rng.choice([
            " anyway that's how I feel", " so yeah",
            " but that's just me", " honestly",
            " that's really all I have to say", " for what it's worth",
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
            return starter + response[0].lower() + response[1:] if len(response) >= 2 else response

        # Level 1: Add transition phrase in middle
        elif variation_level == 1:
            sentences = response.split('. ')
            if len(sentences) >= 2:
                transition = local_rng.choice(self.TRANSITION_PHRASES)
                sentences[1] = transition.lower() + sentences[1][0].lower() + sentences[1][1:] if len(sentences[1]) >= 2 else sentences[1]
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
            if len(response) >= 2:
                return local_rng.choice(qualifiers) + response[0].lower() + response[1:]
            return response

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
                        if original[0].isupper() and len(replacement) >= 2:
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
            if len(response) >= 2:
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
                        if len(second_part) >= 2:
                            second_part = second_part[0].upper() + second_part[1:]
                            return first_part + ' ' + second_part
                        break
            elif len(sentences) >= 2:
                # Combine first two short sentences into one compound sentence
                s1 = sentences[0].rstrip('.!?')
                s2 = sentences[1]
                if s2:
                    s2_lower = s2[0].lower() + s2[1:] if len(s2) >= 2 else s2
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
        behavioral_profile: Optional[Dict[str, Any]] = None,
        question_intent: str = "",  # v1.0.8.4: Pre-computed intent from engine
        question_context: str = "",  # v1.0.8.4: Raw user-provided context
    ) -> str:
        """
        Generate a context-appropriate response to an open-ended question.

        v1.0.0: CRITICAL FIX - Each question now gets a unique response using
        a per-question seeded RNG. This ensures the same participant gives
        DIFFERENT responses to DIFFERENT questions.

        v1.0.4.8: Enhanced with behavioral_profile dict that carries the
        participant's numeric response patterns, intensity, consistency, and
        persona information. Used to ensure text-numeric coherence.

        Args:
            question_text: The question being answered
            sentiment: Overall sentiment of the response
            persona_verbosity: How verbose the persona is (0-1)
            persona_formality: How formal the persona is (0-1)
            persona_engagement: How engaged the persona is (0-1)
            condition: Experimental condition (for context)
            question_name: Unique identifier for the question (CRITICAL for uniqueness)
            participant_seed: Base random seed for this participant
            behavioral_profile: Dict with response_pattern, intensity, consistency_score,
                              behavioral_summary, trait_profile, persona_name

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

        # v1.0.3.8: Extract question context if embedded in question_text
        # The engine embeds "Question: ...\nContext: ..." format
        _embedded_context = ""
        if "\nContext: " in question_text:
            _ctx_parts = question_text.split("\nContext: ")
            if len(_ctx_parts) > 1:
                _embedded_context = _ctx_parts[1].split("\n")[0].strip()

        # v1.0.8.4: Use raw question_context as fallback if embedded context is empty
        _effective_context = _embedded_context or question_context

        # v1.0.9.2: PRIMARY PATH — Adaptive compositional builder.
        # Try to build a response compositionally from atomic parts.
        # This produces dramatically more natural text than template selection.
        # Falls back to template path if adaptive builder returns empty.
        _subject = self._extract_response_subject(question_text, _effective_context)
        _domain_key = domain.value if hasattr(domain, 'value') else str(domain)

        response = ""
        if _subject or question_text:
            _adaptive_topic = _subject or question_text[:80]
            try:
                response = self._build_adaptive_response(
                    topic=_adaptive_topic,
                    sentiment=sentiment,
                    question_intent=question_intent or "opinion",
                    question_text=question_text,
                    question_context=_effective_context,
                    condition=condition,
                    domain_key=_domain_key,
                    behavioral_profile=behavioral_profile or {},
                    rng=local_rng,
                )
            except Exception:
                response = ""

        # FALLBACK: Template path if adaptive builder returned empty
        if not response or len(response.strip()) < 10:
            response = self._get_template_response(
                domain, q_type, sentiment, local_rng,
                question_text=question_text,
                question_context=_effective_context,
                question_intent=question_intent,
                question_name=question_name,
            )
            # Personalize template response
            response = self._personalize_for_question(response, question_text, question_keywords, condition, local_rng)

        # v1.0.4.8: Override engagement if behavioral profile indicates straight-lining
        _effective_engagement = persona_engagement
        if behavioral_profile and isinstance(behavioral_profile, dict):
            if behavioral_profile.get('straight_lined'):
                _effective_engagement = min(_effective_engagement, 0.2)
            # Strongly opinionated participants (very high/low means) tend to write more
            if behavioral_profile.get('response_pattern') in ('strongly_positive', 'strongly_negative'):
                _effective_engagement = max(_effective_engagement, 0.5)

        # Handle disengaged/careless personas
        if _effective_engagement < 0.3:
            response = self._make_careless(response, _effective_engagement, local_rng,
                                          question_text=question_text,
                                          question_context=_effective_context,
                                          question_intent=question_intent,
                                          sentiment=sentiment)

        # Add variation (using local RNG)
        response = add_variation(response, persona_verbosity, persona_formality, local_rng)

        # Adjust length based on verbosity
        if persona_verbosity < 0.3:
            response = self._shorten(response, local_rng)
        elif persona_verbosity > 0.7:
            response = self._extend(response, domain, sentiment, local_rng)

        # v1.0.4.8: Behavioral coherence adjustment — ensure text tone matches
        # numeric response pattern from the behavioral profile
        if behavioral_profile and isinstance(behavioral_profile, dict):
            response = self._enforce_behavioral_coherence(
                response, behavioral_profile, sentiment, local_rng,
                question_intent=question_intent,
            )

        # v1.0.5.5: Disabled _add_topic_context() — it added meta-commentary
        # phrases ("in this study", "regarding this question") that made responses
        # sound academic rather than natural.  Compositional templates already
        # handle topic grounding without meta-commentary.
        # response = self._add_topic_context(response, question_text, question_keywords, domain, local_rng)

        # v1.1.0.4: LAST RESORT — use corpus response instead of old
        # "I thought about {variable_name}" fallback that produced gibberish.
        if not response or not response.strip():
            _domain_key_fb = domain.value if hasattr(domain, 'value') else str(domain)
            _corpus_fb = self._get_corpus_response(_domain_key_fb, sentiment, local_rng)
            if _corpus_fb:
                response = _corpus_fb
            else:
                # Ultra-last-resort: generic but still natural-sounding
                _generic_fallbacks = [
                    "I gave my honest answer based on how I feel about this.",
                    "I thought about it and answered based on my personal experience.",
                    "My response reflects my genuine views on the topic.",
                    "I tried to be honest about how I see things.",
                    "I answered based on what I've personally observed and experienced.",
                ]
                response = local_rng.choice(_generic_fallbacks)

        # v1.1.0.6 ITERATION 1: Response length calibration from log-normal
        # distribution (Denscombe 2008: mean ~40 words for engaged respondents).
        # This replaces the uniform-ish lengths template generation produces.
        if behavioral_profile and isinstance(behavioral_profile, dict):
            _bp_traits = behavioral_profile.get('trait_profile', {})
            _bp_engagement = _bp_traits.get('attention', persona_engagement)
            _bp_verbosity = _bp_traits.get('verbosity', persona_verbosity)
            response = self._calibrate_response_length(
                response, _bp_engagement, _bp_verbosity, local_rng,
            )

        # v1.1.0.6 ITERATION 3: Cross-response voice consistency —
        # Same participant writes with consistent stylistic patterns
        # (opinion verbs, punctuation habits, hedge preferences).
        if behavioral_profile and isinstance(behavioral_profile, dict):
            _bp_traits = behavioral_profile.get('trait_profile', {})
            response = self._apply_cross_response_voice_consistency(
                response, participant_seed, _bp_traits, local_rng,
            )

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
            'product': ['about this', 'regarding what was presented', 'evaluating this', 'concerning this'],
            'service': ['about the service', 'regarding the service quality', 'in terms of service', 'evaluating the service provided'],
            'brand': ['about the brand', 'regarding this brand', 'in terms of perception', 'considering the attributes'],
            'quality': ['in terms of quality', 'regarding quality aspects', 'considering the quality', 'evaluating quality'],
            'value': ['in terms of value', 'considering what matters', 'regarding perceived value'],
            'price': ['considering the cost', 'given the cost', 'in terms of pricing', 'regarding the price point'],
            'purchase': ['regarding this choice', 'in terms of choosing', 'considering this', 'when thinking about this'],

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
                        if len(response) >= 2:
                            return f"{phrase.capitalize()}, {response[0].lower()}{response[1:]}"
                        return f"{phrase.capitalize()}, {response}"
                    else:
                        # Add before last sentence
                        sentences = response.rsplit('. ', 1)
                        if len(sentences) == 2:
                            if len(sentences[1]) >= 2:
                                return f"{sentences[0]}. {phrase.capitalize()}, {sentences[1][0].lower()}{sentences[1][1:]}"
                            return f"{sentences[0]}. {phrase.capitalize()}, {sentences[1]}"
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

        v1.0.3.9: Fixed to avoid injecting consumer/product language into
        non-consumer studies. Topic intros are now domain-neutral. Condition
        extensions only apply to relevant domains.
        """
        rng = local_rng or random.Random()

        if not keywords and not condition:
            return response

        question_lower = question_text.lower() if question_text else ""

        # v1.0.5.5: Removed topic_intros system — it prepended phrases like
        # "The impact was that" or "In terms of impact," which, combined with
        # word-salad topic extraction, produced gibberish ("The impact was that
        # love hate Trump...").  Compositional templates now handle topic
        # grounding directly without needing keyword-triggered prefixes.

        # v1.0.3.9: Condition-specific context — ONLY for relevant domains
        # Detect study domain from question text to avoid cross-domain pollution
        _is_consumer_domain = any(w in question_lower for w in (
            'product', 'purchase', 'buy', 'brand', 'shopping', 'consumer',
            'advertis', 'marketing', 'recommend',
        ))
        _is_ai_domain = any(w in question_lower for w in (
            'artificial intelligence', ' ai ', 'algorithm', 'machine learning',
            'chatbot', 'automated', 'robot',
        ))
        # v1.0.4.4: Expanded domain detection for condition-specific personalization
        _is_political_domain = any(w in question_lower for w in (
            'political', 'partisan', 'democrat', 'republican', 'liberal',
            'conservative', 'voter', 'election', 'policy',
        ))
        _is_health_domain = any(w in question_lower for w in (
            'health', 'medical', 'wellness', 'disease', 'patient',
            'treatment', 'vaccination', 'exercise', 'diet',
        ))
        _is_moral_domain = any(w in question_lower for w in (
            'moral', 'ethical', 'right', 'wrong', 'justice', 'fair',
            'dilemma', 'trolley',
        ))
        _is_intergroup_domain = any(w in question_lower for w in (
            'group', 'ingroup', 'outgroup', 'team', 'partner', 'opponent',
            'other group', 'different group',
        ))
        _is_financial_domain = any(w in question_lower for w in (
            'invest', 'financial', 'money', 'saving', 'risk',
            'portfolio', 'market', 'stock',
        ))
        _is_environment_domain = any(w in question_lower for w in (
            'environment', 'climate', 'sustainab', 'green', 'carbon',
            'pollution', 'renewable',
        ))
        _is_education_domain = any(w in question_lower for w in (
            'learn', 'teach', 'student', 'classroom', 'education',
            'study', 'exam', 'course',
        ))

        condition_lower = condition.lower() if condition else ""
        if condition_lower:
            # v1.0.7.9: EXPANDED condition modifiers — more natural, more varied,
            # with 4-6 alternatives per condition type instead of 2.

            # AI conditions — only in AI/consumer/tech contexts
            if 'ai' in condition_lower and 'no ai' not in condition_lower:
                if (_is_consumer_domain or _is_ai_domain) and rng.random() < 0.35:
                    response += rng.choice([
                        " The AI recommendation influenced how I thought about this.",
                        " Knowing AI was involved made me think differently.",
                        " I considered the AI element when forming my opinion.",
                        " The AI aspect is hard to ignore in situations like this.",
                        " Whether or not the AI played a role affected my thinking.",
                    ])
            elif 'hedonic' in condition_lower:
                if _is_consumer_domain and rng.random() < 0.3:
                    response += rng.choice([
                        " The enjoyment factor was a big part of it for me.",
                        " How much pleasure I'd get from it mattered in my decision.",
                        " The fun aspect is what really caught my attention.",
                    ])
            elif 'utilitarian' in condition_lower:
                if _is_consumer_domain and rng.random() < 0.3:
                    response += rng.choice([
                        " Practicality was the main thing I considered.",
                        " I focused on how useful and functional it would be.",
                        " The practical benefits mattered most to me.",
                    ])

            # v1.0.7.9: Political condition modifiers — expanded
            if _is_political_domain and rng.random() < 0.35:
                if any(kw in condition_lower for kw in ['liberal', 'democrat', 'progressive', 'left']):
                    response += rng.choice([
                        " As someone who leans progressive, this really matters to me.",
                        " My left-leaning values definitely shaped this response.",
                        " I think about these issues from a progressive perspective.",
                        " Coming from where I do politically, I feel strongly about this.",
                        " My Democratic values inform how I see this.",
                    ])
                elif any(kw in condition_lower for kw in ['conservative', 'republican', 'right']):
                    response += rng.choice([
                        " As someone who leans conservative, this really matters to me.",
                        " My traditional values definitely shaped this response.",
                        " I think about these issues from a conservative perspective.",
                        " Coming from where I do politically, I feel strongly about this.",
                        " My Republican values inform how I see this.",
                    ])
                elif any(kw in condition_lower for kw in ['trump']):
                    if any(kw in condition_lower for kw in ['lover', 'supporter', 'pro']):
                        response += rng.choice([
                            " As a Trump supporter I feel pretty strongly about this.",
                            " Supporting Trump is part of who I am and it shapes my views.",
                            " My political beliefs align with Trump and that influences my answer.",
                        ])
                    elif any(kw in condition_lower for kw in ['hater', 'opponent', 'anti', 'against']):
                        response += rng.choice([
                            " I strongly oppose Trump and that affects how I see this.",
                            " My opposition to Trump is deeply held and relevant here.",
                            " Being anti-Trump is part of my political identity.",
                        ])

            # v1.0.7.9: Health condition modifiers — expanded
            if _is_health_domain and rng.random() < 0.35:
                if any(kw in condition_lower for kw in ['high risk', 'severe', 'threat', 'disease', 'danger']):
                    response += rng.choice([
                        " The health risks here really concern me.",
                        " Given how serious the health implications are, I take this very seriously.",
                        " Health threats like this make me think more carefully about my choices.",
                        " The severity of the health issue definitely affected my response.",
                    ])
                elif any(kw in condition_lower for kw in ['prevention', 'wellness', 'healthy', 'benefit']):
                    response += rng.choice([
                        " Taking a preventive approach just makes sense to me.",
                        " I'm always in favor of things that promote health and wellness.",
                        " Health benefits are a strong motivator for me.",
                        " Staying healthy is something I prioritize.",
                    ])
                elif any(kw in condition_lower for kw in ['vaccine', 'vaccin']):
                    response += rng.choice([
                        " My views on vaccination are part of how I think about this.",
                        " The vaccine angle is important to how I responded.",
                        " I have strong feelings about vaccines that shaped my answer.",
                    ])

            # v1.0.7.9: Moral/ethical condition modifiers — expanded
            if _is_moral_domain and rng.random() < 0.35:
                if any(kw in condition_lower for kw in ['moral', 'ethical', 'right', 'virtue']):
                    response += rng.choice([
                        " My sense of right and wrong was front and center here.",
                        " Ethical considerations weighed heavily in my thinking.",
                        " I couldn't separate my moral values from my response.",
                        " My conscience played a big role in how I answered.",
                    ])
                elif any(kw in condition_lower for kw in ['harm', 'kill', 'sacrifice', 'dilemma']):
                    response += rng.choice([
                        " The idea of someone getting hurt made this really difficult.",
                        " Situations involving harm force you to think about what you really value.",
                        " I struggled with the tradeoffs involved in potential harm.",
                    ])

            # v1.0.7.9: Intergroup condition modifiers — expanded with identity markers
            if (_is_intergroup_domain or _is_political_domain) and rng.random() < 0.40:
                _is_ingroup = any(kw in condition_lower for kw in ['ingroup', 'same group', 'same team',
                    'supporter supporter', 'lover lover', 'hater hater', 'democrat democrat',
                    'republican republican'])
                _is_outgroup = any(kw in condition_lower for kw in ['outgroup', 'different group',
                    'opposing', 'other group', 'lover hater', 'hater lover',
                    'supporter opponent', 'democrat republican', 'republican democrat'])

                if _is_ingroup:
                    response += rng.choice([
                        " Being paired with someone who shares my views felt comfortable.",
                        " It's easier to interact with someone from your own group.",
                        " Having shared values with the other person influenced my choice.",
                        " I felt a natural connection because we're on the same side.",
                        " When you share a background with someone it changes the dynamic.",
                        " I was more generous because we see things the same way.",
                    ])
                elif _is_outgroup:
                    response += rng.choice([
                        " Knowing the other person has different views than me affected my decision.",
                        " It's harder to be open when you're dealing with someone from the other side.",
                        " Group differences definitely played a role in how I responded.",
                        " I was more cautious because we don't see eye to eye.",
                        " Being matched with someone from a different group made me less trusting.",
                        " The divide between our groups is hard to ignore in a situation like this.",
                    ])

            # v1.0.7.9: Financial condition modifiers — expanded
            if _is_financial_domain and rng.random() < 0.35:
                if any(kw in condition_lower for kw in ['gain', 'profit', 'earn', 'high return', 'bonus']):
                    response += rng.choice([
                        " The potential financial upside influenced my thinking.",
                        " When there's money to be gained I tend to pay more attention.",
                        " The possibility of profit shaped my decision-making.",
                        " Financial incentives definitely played into my choice.",
                    ])
                elif any(kw in condition_lower for kw in ['loss', 'risk', 'lose', 'penalty', 'fine']):
                    response += rng.choice([
                        " The financial risk weighed heavily on my mind.",
                        " The possibility of losing money made me more cautious.",
                        " I'm pretty sensitive to financial losses and that showed here.",
                        " When there's a chance of losing money I think more carefully.",
                    ])

            # v1.0.7.9: Environmental condition modifiers — expanded
            if _is_environment_domain and rng.random() < 0.35:
                if any(kw in condition_lower for kw in ['sustainable', 'green', 'eco', 'renew']):
                    response += rng.choice([
                        " Sustainability is important to me in these kinds of decisions.",
                        " The environmental angle made me think more carefully about this.",
                        " Green options appeal to me on principle.",
                    ])
                elif any(kw in condition_lower for kw in ['pollut', 'unsustainable', 'harm', 'damage']):
                    response += rng.choice([
                        " The environmental damage aspect really bothers me.",
                        " Knowing about the environmental harm affected my response.",
                        " I can't ignore the environmental consequences.",
                    ])

            # v1.0.7.9: Education condition modifiers — NEW
            if _is_education_domain and rng.random() < 0.3:
                if any(kw in condition_lower for kw in ['online', 'remote', 'virtual', 'digital']):
                    response += rng.choice([
                        " The online format changes how I think about learning.",
                        " Remote learning has its own set of challenges and benefits.",
                        " The digital aspect influenced my perspective.",
                    ])
                elif any(kw in condition_lower for kw in ['in person', 'face to face', 'classroom', 'traditional']):
                    response += rng.choice([
                        " There's something about in-person learning that matters.",
                        " The classroom setting affects the learning experience.",
                        " Face-to-face interaction makes a difference.",
                    ])

            # v1.0.8.0: Cognitive/psychological condition modifiers — NEW
            _is_cognitive_domain = any(w in question_lower for w in (
                'cognitive', 'memory', 'attention', 'decision', 'thinking',
                'reasoning', 'judgment', 'bias', 'load', 'effort',
            ))
            if _is_cognitive_domain and rng.random() < 0.30:
                if any(kw in condition_lower for kw in ['high load', 'difficult', 'complex', 'time pressure', 'fast']):
                    response += rng.choice([
                        " The mental demands of this made it harder to think clearly.",
                        " I felt rushed and that probably affected my response.",
                        " When things are complex I rely more on my gut feeling.",
                        " The difficulty level made me less careful than I'd normally be.",
                    ])
                elif any(kw in condition_lower for kw in ['low load', 'easy', 'simple', 'no pressure', 'slow']):
                    response += rng.choice([
                        " I had enough time to really think this through.",
                        " The simpler format let me focus on what mattered.",
                        " I felt comfortable taking my time with this.",
                    ])

            # v1.0.8.0: Sports/competition condition modifiers — NEW
            _is_sports_domain = any(w in question_lower for w in (
                'sport', 'athlet', 'competition', 'team', 'training', 'coach',
                'perform', 'fitness', 'game', 'match',
            ))
            if _is_sports_domain and rng.random() < 0.30:
                if any(kw in condition_lower for kw in ['competitive', 'competition', 'rival', 'opponent', 'win']):
                    response += rng.choice([
                        " The competitive aspect really drives my thinking here.",
                        " Competition brings out a different side of me.",
                        " Wanting to win is a strong motivator in my response.",
                    ])
                elif any(kw in condition_lower for kw in ['cooperative', 'team', 'collaborative', 'together']):
                    response += rng.choice([
                        " Working together changes how I approach this.",
                        " The team aspect is important to how I think about this.",
                        " Collaboration matters a lot to me in these situations.",
                    ])

            # v1.0.8.0: Clinical/therapy condition modifiers — NEW
            _is_clinical_domain = any(w in question_lower for w in (
                'anxiety', 'depression', 'therapy', 'mental health', 'stress',
                'coping', 'clinical', 'counseling', 'wellbeing',
            ))
            if _is_clinical_domain and rng.random() < 0.30:
                if any(kw in condition_lower for kw in ['treatment', 'therapy', 'intervention', 'cbt', 'mindfulness']):
                    response += rng.choice([
                        " The approach described resonates with my experience.",
                        " I can see how this kind of support would make a difference.",
                        " Having structured help really matters for these issues.",
                    ])
                elif any(kw in condition_lower for kw in ['control', 'no treatment', 'waitlist', 'placebo']):
                    response += rng.choice([
                        " Without active support things tend to stay the same.",
                        " I responded based on how I normally handle things.",
                    ])

            # v1.0.8.0: Social/relationship condition modifiers — NEW
            _is_relationship_domain = any(w in question_lower for w in (
                'relationship', 'partner', 'friend', 'social', 'attachment',
                'intimacy', 'dating', 'marriage', 'loneliness', 'support',
            ))
            if _is_relationship_domain and rng.random() < 0.30:
                if any(kw in condition_lower for kw in ['close', 'intimate', 'secure', 'strong', 'supportive']):
                    response += rng.choice([
                        " Having close connections shapes how I see these things.",
                        " Feeling supported by others influences my perspective.",
                        " Secure relationships make a real difference.",
                    ])
                elif any(kw in condition_lower for kw in ['distant', 'conflict', 'insecure', 'rejection', 'lonely']):
                    response += rng.choice([
                        " Feeling disconnected from others affects how I think about this.",
                        " Relationship difficulties color my perspective here.",
                        " When connections feel strained everything is harder.",
                    ])

            # v1.0.8.0: Communication/media condition modifiers — NEW
            _is_communication_domain = any(w in question_lower for w in (
                'message', 'communicat', 'persuasi', 'media', 'news',
                'source', 'information', 'narrative', 'credib',
            ))
            if _is_communication_domain and rng.random() < 0.30:
                if any(kw in condition_lower for kw in ['expert', 'credible', 'scientific', 'authoritative', 'trustworthy']):
                    response += rng.choice([
                        " The credibility of the source matters a lot to me.",
                        " I pay more attention when the information comes from an expert.",
                        " A trustworthy source makes me take things more seriously.",
                    ])
                elif any(kw in condition_lower for kw in ['peer', 'personal', 'anecdot', 'testimonial', 'story']):
                    response += rng.choice([
                        " Personal stories hit differently than statistics.",
                        " Hearing someone's actual experience is more convincing to me.",
                        " Real-life accounts stick with me more than abstract facts.",
                    ])
                elif any(kw in condition_lower for kw in ['fake', 'misleading', 'biased', 'unreliable']):
                    response += rng.choice([
                        " I'm suspicious of information that seems biased.",
                        " Unreliable sources make me question everything.",
                    ])

            # v1.0.8.0: Food/body condition modifiers — NEW
            _is_food_domain = any(w in question_lower for w in (
                'food', 'diet', 'eating', 'nutrition', 'body image',
                'weight', 'meal', 'calorie', 'healthy eating',
            ))
            if _is_food_domain and rng.random() < 0.30:
                if any(kw in condition_lower for kw in ['healthy', 'organic', 'natural', 'nutritious', 'clean']):
                    response += rng.choice([
                        " Health-conscious options appeal to me.",
                        " I tend to favor choices that seem healthier.",
                        " The nutritional aspect influenced my thinking.",
                    ])
                elif any(kw in condition_lower for kw in ['unhealthy', 'processed', 'junk', 'indulgent', 'tempting']):
                    response += rng.choice([
                        " I'm aware of the temptation even when I know it's not ideal.",
                        " The appeal of indulgent options is hard to resist.",
                        " I try to be honest about what actually tempts me.",
                    ])

            # v1.0.8.0: Developmental/age condition modifiers — NEW
            _is_developmental_domain = any(w in question_lower for w in (
                'child', 'parent', 'aging', 'adolescent', 'elderly',
                'development', 'youth', 'generation', 'retirement',
            ))
            if _is_developmental_domain and rng.random() < 0.30:
                if any(kw in condition_lower for kw in ['young', 'child', 'adolescent', 'early', 'youth']):
                    response += rng.choice([
                        " Early experiences shape so much of who we become.",
                        " Youth perspectives are unique and important.",
                        " Development in the early years is critical.",
                    ])
                elif any(kw in condition_lower for kw in ['old', 'elder', 'aging', 'senior', 'late life']):
                    response += rng.choice([
                        " The perspective that comes with age is valuable.",
                        " Later life brings different challenges and priorities.",
                        " Aging changes how you think about things.",
                    ])

            # v1.0.8.0: Technology/social media condition modifiers — NEW
            _is_tech_domain = any(w in question_lower for w in (
                'technology', 'digital', 'app', 'online', 'internet',
                'social media', 'screen', 'device', 'platform',
            ))
            if _is_tech_domain and not _is_ai_domain and rng.random() < 0.30:
                if any(kw in condition_lower for kw in ['high use', 'frequent', 'heavy', 'active', 'engaged']):
                    response += rng.choice([
                        " As someone who uses technology a lot, I have opinions about this.",
                        " My extensive experience with this technology informs my response.",
                        " Being a heavy user gives me a particular perspective.",
                    ])
                elif any(kw in condition_lower for kw in ['low use', 'infrequent', 'minimal', 'limited', 'no access']):
                    response += rng.choice([
                        " I don't have as much experience with this technology.",
                        " Limited exposure gives me a different perspective.",
                        " Not being a heavy user shapes how I see this.",
                    ])

            # v1.0.8.0: Generic treatment/control — applies when no specific domain matched
            _any_domain_matched = any([_is_political_domain, _is_health_domain, _is_moral_domain,
                        _is_intergroup_domain, _is_financial_domain, _is_environment_domain,
                        _is_consumer_domain, _is_ai_domain, _is_education_domain,
                        _is_cognitive_domain, _is_sports_domain, _is_clinical_domain,
                        _is_relationship_domain, _is_communication_domain, _is_food_domain,
                        _is_developmental_domain, _is_tech_domain])
            if not _any_domain_matched:
                if any(kw in condition_lower for kw in ['treatment', 'experimental', 'intervention']):
                    if rng.random() < 0.25:
                        response += rng.choice([
                            " The specific situation presented to me shaped how I responded.",
                            " What I was shown definitely influenced my thinking.",
                            " The particular scenario affected my response.",
                            " The way things were set up made me think about this differently.",
                        ])
                elif any(kw in condition_lower for kw in ['control', 'baseline', 'neutral']):
                    if rng.random() < 0.15:
                        response += rng.choice([
                            " I just responded based on my natural inclinations.",
                            " I went with my default perspective on this.",
                            " Nothing particular pushed me in any direction.",
                        ])
                elif any(kw in condition_lower for kw in ['high', 'strong', 'extreme', 'intense']):
                    if rng.random() < 0.25:
                        response += rng.choice([
                            " The intensity of the situation influenced my response.",
                            " Strong conditions like this make you react more firmly.",
                        ])
                elif any(kw in condition_lower for kw in ['low', 'mild', 'weak', 'subtle']):
                    if rng.random() < 0.20:
                        response += rng.choice([
                            " The subtle framing made me respond more naturally.",
                            " Without strong cues I went with my honest instinct.",
                        ])

        return response

    def _get_template_response(
        self,
        domain: StudyDomain,
        q_type: QuestionType,
        sentiment: str,
        local_rng: random.Random = None,
        question_text: str = "",
        question_context: str = "",
        question_intent: str = "",  # v1.0.8.4: Pre-computed intent
        question_name: str = "",  # v1.0.8.5: For last-resort topic extraction
    ) -> str:
        """Get a template response for the given domain and sentiment.

        v1.0.3.8: Enhanced to generate context-grounded responses. When
        question_context is available, generates a response that directly
        addresses the question subject rather than using generic templates.
        Falls back to domain templates when no context is provided.
        """
        rng = local_rng or random.Random()
        domain_key = domain.value

        # v1.0.3.8: If we have question context, generate a context-grounded
        # response instead of picking a generic domain template. This is the
        # KEY improvement — responses now address the actual question topic.
        _subject = self._extract_response_subject(question_text, question_context)
        if _subject:
            return self._generate_context_grounded_response(
                _subject, sentiment, q_type, domain, rng,
                override_intent=question_intent,  # v1.0.8.4: Use pre-computed intent
            )

        # v1.0.5.0: Try question-type-matched templates first, then fall back to explanation
        _q_type_key = q_type.value if hasattr(q_type, 'value') else str(q_type)
        _type_fallback_chain = [_q_type_key, "evaluation", "explanation"]
        # Deduplicate while preserving order
        _seen_types = set()
        _type_chain = []
        for _tk in _type_fallback_chain:
            if _tk not in _seen_types:
                _seen_types.add(_tk)
                _type_chain.append(_tk)

        # Try to find domain-specific templates with question type routing
        if domain_key in DOMAIN_TEMPLATES:
            templates = DOMAIN_TEMPLATES[domain_key]
            for _tkey in _type_chain:
                if _tkey in templates:
                    sentiment_templates = templates[_tkey].get(sentiment, templates[_tkey].get("neutral", []))
                    if sentiment_templates:
                        return rng.choice(sentiment_templates)

        # Fall back to general templates with same type routing
        if "general" in DOMAIN_TEMPLATES:
            for _tkey in _type_chain:
                if _tkey in DOMAIN_TEMPLATES["general"]:
                    templates = DOMAIN_TEMPLATES["general"][_tkey]
                    sentiment_templates = templates.get(sentiment, templates.get("neutral", []))
                    if sentiment_templates:
                        return rng.choice(sentiment_templates)

        # v1.0.8.5: Last-resort fallback — multi-source topic extraction.
        # Try question_text → question_context → question_name (variable name)
        _fallback_stop = {'this', 'that', 'about', 'what', 'your', 'please',
                          'describe', 'explain', 'question', 'context', 'study',
                          'topic', 'condition', 'here', 'there', 'well', 'like',
                          'with', 'from', 'have', 'some', 'very', 'really',
                          'think', 'feel', 'answer', 'response', 'text', 'open'}
        _fb_phrase = ""
        for _src in [question_text, question_context, question_name]:
            if not _src:
                continue
            # For question_name, split on underscores/camelCase first
            _clean = re.sub(r'[_\-]+', ' ', _src) if _src == question_name else _src
            _clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', _clean)  # camelCase
            _words = re.findall(r'\b[a-zA-Z]{3,}\b', _clean.lower())
            _topic = [w for w in _words if w not in _fallback_stop][:3]
            if _topic:
                _fb_phrase = ' '.join(_topic)
                break
        if _fb_phrase:
            _fb_templates = [
                f"My honest take on {_fb_phrase} is based on my personal experience.",
                f"I feel a certain way about {_fb_phrase} and tried to express that.",
                f"When it comes to {_fb_phrase} I went with my genuine reaction.",
                f"{_fb_phrase} is something I have real feelings about.",
            ]
            return rng.choice(_fb_templates)
        return "I answered based on my honest feelings about what was asked."

    # v1.1.0.4: Topic intelligibility — detect whether an extracted topic
    # can be naturally used in a sentence.  Variable names like "respond_affect"
    # produce gibberish when interpolated into templates.
    _UNINTELLIGIBLE_PATTERNS = re.compile(
        r'^(respond|affect|score|scale|item|var|dv|iv|q\d|v\d|block|loop|'
        r'measure|variable|factor|construct|subscale|reverse|recode|'
        r'raw|total|sum|mean|avg|std|index|composite|latent|observed|'
        r'pre|post|time|wave|trial|stimulus|condition|control|treatment|'
        r'group|cell|level|manipulation|check|filler|buffer|distractor|'
        r'demographics?|debrief|consent|attention)\b',
        re.IGNORECASE,
    )
    _MIN_INTELLIGIBLE_WORDS = 3  # Need at least 3 real words for a usable topic

    def _is_topic_intelligible(self, topic: str) -> bool:
        """Check whether a topic string can be naturally interpolated into a sentence.

        v1.1.0.4: CRITICAL FIX.  The #1 source of gibberish responses was
        unintelligible topics like "respond affect score" being plugged into
        templates as "I strongly believe respond affect score".  This validator
        catches variable names, technical jargon, and other non-natural-language
        strings BEFORE they reach the template system.

        Returns True if the topic reads as natural language, False if it's a
        variable name, technical term, or otherwise unfit for interpolation.
        """
        if not topic or len(topic.strip()) < 4:
            return False
        _t = topic.strip().lower()
        # Pure numbers or codes
        if re.match(r'^[\d_\-\.]+$', _t):
            return False
        # Single word that's a technical term
        _words = _t.split()
        if len(_words) == 1 and self._UNINTELLIGIBLE_PATTERNS.match(_t):
            return False
        # Multi-word but every word is a technical/variable term
        _real_words = [w for w in _words if not self._UNINTELLIGIBLE_PATTERNS.match(w)
                       and len(w) > 2]
        if len(_real_words) < 1:
            return False
        # Looks like a variable name (camelCase, snake_case with no spaces originally)
        if re.match(r'^[a-z]+[A-Z]', topic.strip()):  # camelCase
            return False
        # All caps (constant name)
        if topic.strip().isupper() and len(topic.strip()) > 3:
            return False
        # Contains repeated underscores or hyphens (was a var name)
        if '__' in topic or '--' in topic:
            return False
        return True

    def _extract_response_subject(self, question_text: str, question_context: str) -> str:
        """Extract the core subject/topic from question text and context.

        v1.1.0.4: MAJOR OVERHAUL.  Previous versions returned raw variable
        names like "respond_affect_score" which produced gibberish when
        interpolated into templates.  Now includes:
        1. Multi-source extraction (context → question → variable name)
        2. Intelligibility validation (rejects variable names/jargon)
        3. Natural language conversion for common patterns
        4. Domain-aware fallback descriptions

        Returns a short phrase describing what the question is actually about,
        or empty string if we can't determine an intelligible subject.
        """
        # Source priority: explicit context > question text > variable name
        _sources: list = []

        # 1. User-provided question context (most specific)
        if question_context and question_context.strip():
            _sources.append(question_context.strip())

        # 2. Question text (may contain embedded context)
        if question_text and question_text.strip():
            _qt = question_text.strip()
            # Extract embedded context if present
            if '\nContext: ' in _qt:
                _parts = _qt.split('\nContext: ')
                if len(_parts) > 1:
                    _ctx = _parts[1].split('\n')[0].strip()
                    if _ctx:
                        _sources.append(_ctx)
                # Also use the question part
                _q_part = _parts[0].replace('Question: ', '').strip()
                if _q_part:
                    _sources.append(_q_part)
            else:
                _sources.append(_qt)

        for _src in _sources:
            # Clean up variable-name-style text
            _clean = _src
            if ' ' not in _clean:
                _clean = re.sub(r'[_\-]+', ' ', _clean).strip()

            # Try to extract the core noun phrase from question text
            # "What do you think about climate change?" → "climate change"
            # "How do you feel about the new policy?" → "the new policy"
            _about_match = re.search(
                r'(?:about|regarding|toward[s]?|on|of|with)\s+(.+?)(?:\?|$|\.|,)',
                _clean, re.IGNORECASE,
            )
            if _about_match:
                _candidate = _about_match.group(1).strip()
                if self._is_topic_intelligible(_candidate) and len(_candidate) > 4:
                    return _candidate

            # Try the full source if it's intelligible
            if self._is_topic_intelligible(_clean) and len(_clean) > 4:
                # Truncate very long sources to a usable topic
                if len(_clean) > 80:
                    # Take first meaningful clause
                    _clause = re.split(r'[.!?,;]', _clean)[0].strip()
                    if self._is_topic_intelligible(_clause) and len(_clause) > 4:
                        return _clause
                else:
                    return _clean

        # Nothing intelligible found — return empty (caller uses topic-free mode)
        return ''

    def _infer_topic_from_context(self, domain_key: str, condition: str,
                                   question_text: str, question_context: str) -> str:
        """Infer a natural-language topic from domain, condition, and question context.

        v1.1.0.4: When direct topic extraction fails (e.g., variable names),
        this method tries to construct a meaningful topic from surrounding
        context.  Returns a human-readable phrase like "political attitudes"
        or "trust between people" rather than a variable name.
        """
        # Try condition first — often contains meaningful content
        if condition:
            _cond_clean = re.sub(r'[_\-]+', ' ', condition).strip()
            _cond_stop = {
                'control', 'treatment', 'condition', 'group', 'neutral',
                'baseline', 'default', 'standard', 'cell', 'level', 'high',
                'low', 'experimental', 'comparison', 'manipulation',
            }
            _cw = [w for w in _cond_clean.lower().split() if w not in _cond_stop and len(w) > 2]
            if _cw and len(' '.join(_cw)) > 3:
                _candidate = ' '.join(_cw[:4])
                if self._is_topic_intelligible(_candidate):
                    return _candidate

        # Domain-based natural language topics
        _domain_topics: Dict[str, str] = {
            'political': 'political issues',
            'polarization': 'political polarization',
            'partisanship': 'partisan politics',
            'voting': 'voting and elections',
            'economic_games': 'fairness and sharing',
            'dictator_game': 'sharing decisions',
            'trust_game': 'trust between people',
            'ultimatum_game': 'fairness in negotiations',
            'public_goods': 'contributing to the common good',
            'consumer': 'product quality and value',
            'brand_loyalty': 'brand preferences',
            'health': 'health and healthcare',
            'technology': 'technology and society',
            'ai_attitudes': 'artificial intelligence',
            'moral': 'ethical choices',
            'environment': 'environmental issues',
            'education': 'education and learning',
            'social': 'social relationships',
            'workplace': 'workplace dynamics',
            'identity': 'personal identity',
            'risk': 'risk and uncertainty',
            'persuasion': 'persuasion and influence',
            'relationship': 'interpersonal relationships',
            'financial': 'financial decisions',
            'cooperation': 'cooperation between people',
            'fairness': 'fairness and justice',
        }
        if domain_key in _domain_topics:
            return _domain_topics[domain_key]

        # Last resort: try extracting from question_text with broader patterns
        _src = question_context or question_text or ""
        if _src:
            # Look for noun phrases after common question patterns
            _patterns = [
                r'(?:thoughts?|feelings?|views?|opinions?)\s+(?:on|about|regarding)\s+(.+?)(?:\?|$|\.)',
                r'(?:how do you (?:feel|think))\s+about\s+(.+?)(?:\?|$|\.)',
                r'(?:what (?:is|are) your)\s+(.+?)(?:\?|$|\.)',
                r'(?:describe|explain|discuss)\s+(?:your\s+)?(.+?)(?:\?|$|\.)',
            ]
            for _pat in _patterns:
                _m = re.search(_pat, _src, re.IGNORECASE)
                if _m:
                    _candidate = _m.group(1).strip()
                    if self._is_topic_intelligible(_candidate) and len(_candidate) > 3:
                        return _candidate[:60]

        return ''

    def _generate_context_grounded_response(
        self,
        subject: str,
        sentiment: str,
        q_type: QuestionType,
        domain: StudyDomain,
        rng: random.Random,
        override_intent: str = "",  # v1.0.8.4: Pre-computed intent from engine
    ) -> str:
        """Generate a response grounded in the specific question subject.

        v1.0.7.9: MAJOR OVERHAUL — Question-intent-driven compositional system.

        Previous versions (v1.0.3.8-v1.0.5.5) used a single set of opinion-statement
        cores ("I feel good about {topic}") regardless of what the question actually
        asked. This produced generic sentiment-about-topic responses that failed to
        answer the question.

        NEW ARCHITECTURE:
        1. Extract rich context: topic, action verbs, objects, key phrases from question
        2. Detect question intent with 8 categories (not just 5)
        3. Select intent-specific core templates that ANSWER the question
        4. Add domain-specific vocabulary to elaborations
        5. Compose: opener + intent-matched core + domain-enriched elaboration + coda

        This produces responses that read like actual survey answers, not generic
        opinion statements.
        """
        # ── Step 1: Extract rich context from the subject/question text ──
        _stop = {
            'the', 'a', 'an', 'this', 'that', 'these', 'those', 'its', 'it',
            'they', 'them', 'their', 'we', 'our', 'you', 'your', 'he', 'she',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about',
            'and', 'or', 'but', 'not', 'no', 'so', 'nor',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'need',
            'how', 'what', 'who', 'why', 'when', 'where', 'which',
            'want', 'wants', 'understand', 'think', 'feel', 'tell', 'share', 'describe',
            'explain', 'ask', 'asked', 'give', 'get', 'make', 'say', 'know', 'see',
            'participants', 'respondents', 'subjects', 'people', 'person',
            'primed', 'priming', 'prime', 'exposed', 'exposure', 'exposing',
            'presented', 'presenting', 'shown', 'showing', 'show',
            'told', 'telling', 'instructed', 'instructions',
            'assigned', 'randomly', 'random', 'randomized',
            'thinking', 'reading', 'viewing', 'watching', 'completing', 'answering',
            'reporting', 'sharing', 'responding',
            'before', 'after', 'during', 'following', 'prior',
            'then', 'next', 'first', 'second', 'third',
            'stories', 'story', 'experience', 'experiences',
            'whether', 'toward', 'towards', 'regarding',
            'question', 'questions', 'context', 'study', 'survey', 'experiment',
            'condition', 'conditions', 'topic', 'measure', 'measured',
            'response', 'responses', 'answer', 'answers', 'item', 'items',
            'much', 'more', 'most', 'very', 'really', 'just', 'also', 'please',
            'better', 'deeply', 'held', 'quite',
            'here', 'there', 'well', 'like', 'even', 'still',
            'only', 'each', 'every', 'both', 'many', 'overall',
            'certain', 'particular', 'specific', 'general',
        }
        _all_words = re.findall(r'\b[a-zA-Z]{3,}\b', subject.lower())

        # Entity detection (proper nouns, acronyms, known lowercase entities)
        # v1.0.8.5: Added lowercase entity detection for 80+ high-salience topics
        _known_lowercase_entities = {
            # Political figures & parties
            'trump', 'biden', 'obama', 'clinton', 'sanders', 'desantis', 'pelosi',
            'democrat', 'republican', 'conservative', 'liberal', 'progressive',
            'brexit', 'nato', 'putin', 'zelensky',
            # Tech & platforms
            'facebook', 'instagram', 'twitter', 'tiktok', 'snapchat', 'reddit',
            'google', 'amazon', 'apple', 'microsoft', 'tesla', 'uber', 'airbnb',
            'chatgpt', 'openai', 'bitcoin', 'crypto', 'blockchain', 'metaverse',
            # Health & science
            'covid', 'coronavirus', 'vaccine', 'pfizer', 'moderna', 'fauci',
            'cancer', 'diabetes', 'alzheimer', 'autism', 'adhd',
            # Social & cultural
            'blm', 'metoo', 'lgbtq', 'roe', 'wade', 'scotus', 'maga',
            'woke', 'cancel', 'defund', 'antifa', 'qanon',
            # Brands & products
            'coca-cola', 'pepsi', 'nike', 'adidas', 'starbucks', 'walmart',
            'netflix', 'spotify', 'disney', 'mcdonalds',
        }
        _entities: list = []
        _cap_words = re.findall(r'(?<=[a-z]\s)([A-Z][a-zA-Z]{2,})', subject)
        _entities.extend(w for w in _cap_words if w.lower() not in _stop)
        _acronyms = re.findall(r'\b([A-Z]{2,6})\b', subject)
        _entities.extend(a for a in _acronyms if a.lower() not in _stop)
        _first_words = re.findall(r'^([A-Z][a-zA-Z]{3,})', subject.strip())
        for _fw in _first_words:
            if _fw.lower() not in _stop and _fw not in _entities:
                _entities.append(_fw)
        # v1.0.8.5: Detect lowercase entities from known high-salience list
        for _lw in _all_words:
            if _lw in _known_lowercase_entities and _lw not in {e.lower() for e in _entities}:
                _entities.append(_lw.capitalize())  # Capitalize for display
        _seen_e: set = set()
        _uniq_entities: list = []
        for _e in _entities:
            if _e.lower() not in _seen_e:
                _seen_e.add(_e.lower())
                _uniq_entities.append(_e)
        _entities = _uniq_entities[:3]

        _content = [w for w in _all_words if w not in _stop and w not in {e.lower() for e in _entities}][:6]

        # v1.0.8.5: Negation-preserving bigram extraction — "not trusting" stays together
        _negation_bigrams: list = []
        _neg_words = {'not', 'no', 'never', 'lack', 'dis', 'un', 'without', 'anti'}
        for _i in range(len(_all_words) - 1):
            if _all_words[_i] in _neg_words and _all_words[_i + 1] not in _stop:
                _negation_bigrams.append(f"{_all_words[_i]} {_all_words[_i + 1]}")

        # Build primary topic (2 words max for template insertion)
        # v1.0.8.5: Negation bigrams get priority — semantic reversal prevention
        if _negation_bigrams:
            _topic = _negation_bigrams[0]
        elif _entities:
            _topic = _entities[0]
        elif _content:
            _topic = ' '.join(_content[:2])
        else:
            _topic = subject[:30].strip()
        _entity_lower = {e.lower() for e in _entities}
        _topic_parts = _topic.split()
        _topic_parts = [w.capitalize() if w.lower() in _entity_lower else w for w in _topic_parts]
        _topic = ' '.join(_topic_parts)

        # v1.0.8.0: Extract ACTION VERBS — expanded to cover ALL research domains
        _subj_lower = subject.lower()
        _action = ""
        _action_patterns = [
            # Economic games
            (r'\b(gave|give|giving|donated|donating|split|splitting|shared|sharing)\b', 'gave'),
            (r'\b(trusted|trusting|sent|sending)\b', 'trusted'),
            (r'\b(cooperated|cooperating|contributed|contributing)\b', 'cooperated'),
            (r'\b(rejected|rejecting|refused|refusing|declined)\b', 'rejected'),
            (r'\b(accepted|accepting|agreed|agreeing)\b', 'accepted'),
            (r'\b(punished|punishing|sanctioned|sanctioning)\b', 'punished'),
            (r'\b(forgave|forgiving|pardoned)\b', 'forgave'),
            (r'\b(risked|risking|gambled|gambling|bet|betting)\b', 'risked'),
            (r'\b(waited|waiting|delayed|delaying)\b', 'waited'),
            # General decisions
            (r'\b(chose|choose|choosing|picked|selected|decided)\b', 'chose'),
            (r'\b(bought|buy|buying|purchased|purchasing)\b', 'bought'),
            (r'\b(rated|rating|evaluated|evaluating|judged|judging)\b', 'rated'),
            (r'\b(voted|voting|supported|supporting|endorsed)\b', 'supported'),
            # Social/moral
            (r'\b(lied|lying|cheated|cheating|deceived|deceiving)\b', 'lied'),
            (r'\b(helped|helping|assisted|assisting|volunteered)\b', 'helped'),
            (r'\b(blamed|blaming|accused|accusing)\b', 'blamed'),
            (r'\b(confronted|confronting|challenged|challenging)\b', 'confronted'),
            # Communication/persuasion
            (r'\b(persuaded|convincing|influenced|influencing)\b', 'persuaded'),
            (r'\b(communicated|communicating|expressed|expressing)\b', 'communicated'),
            (r'\b(disclosed|disclosing|revealed|revealing)\b', 'disclosed'),
            (r'\b(recommended|recommending|suggested|advising)\b', 'recommended'),
            # Health/clinical
            (r'\b(coped|coping|managed|managing)\b', 'coped'),
            (r'\b(exercised|exercising|trained|training)\b', 'exercised'),
            (r'\b(ate|eating|consumed|consuming|dieted)\b', 'consumed'),
            (r'\b(meditated|meditating|practiced|practicing)\b', 'practiced'),
            # Learning/cognitive
            (r'\b(learned|learning|studied|studying|memorized)\b', 'learned'),
            (r'\b(recalled|remembering|recognized|recognizing)\b', 'recalled'),
            (r'\b(solved|solving|figured|figuring)\b', 'solved'),
            (r'\b(negotiated|negotiating|bargained|bargaining)\b', 'negotiated'),
            # Creative/work
            (r'\b(created|creating|designed|designing|built|building)\b', 'created'),
            (r'\b(led|leading|managed|supervising)\b', 'led'),
            (r'\b(competed|competing|performed|performing)\b', 'competed'),
        ]
        for _pat, _act_label in _action_patterns:
            if re.search(_pat, _subj_lower):
                _action = _act_label
                break

        # v1.0.8.0: Extract OBJECT/TARGET — expanded across ALL research domains
        _target = ""
        _target_patterns = [
            # Social/interpersonal
            (r'(?:the |your |their )?(other person|partner|opponent|stranger|recipient|receiver|friend|colleague)', None),
            (r'(?:the |your |their )?(other group|outgroup|ingroup|other team|rival group|minority group)', None),
            (r'(?:the |your |their )?(relationship|marriage|friendship|bond|connection)', None),
            # Economic/consumer
            (r'(?:the |your |their )?(product|brand|service|item|option|package|deal)', None),
            (r'(?:the |your |their )?(money|amount|endowment|offer|donation|salary|price)', None),
            (r'(?:the |your |their )?(investment|portfolio|savings|retirement|budget)', None),
            # Political/civic
            (r'(?:the |your |their )?(candidate|politician|leader|president|party|government)', None),
            (r'(?:the |your |their )?(policy|legislation|regulation|law|proposal)', None),
            # Decisions/outcomes
            (r'(?:the |your |their )?(decision|choice|outcome|result|consequence|tradeoff)', None),
            (r'(?:the |your |their )?(risk|gamble|bet|chance|probability|odds)', None),
            # Health/clinical
            (r'(?:the |your |their )?(treatment|medication|therapy|vaccine|diagnosis|symptom)', None),
            (r'(?:the |your |their )?(health|wellbeing|mental health|anxiety|diet|exercise)', None),
            (r'(?:the |your |their )?(doctor|therapist|counselor|provider|patient)', None),
            # Technology/AI
            (r'(?:the |your |their )?(algorithm|AI system|chatbot|technology|platform|app)', None),
            (r'(?:the |your |their )?(data|privacy|information|profile|account)', None),
            # Social/community
            (r'(?:the |your |their )?(community|society|group|team|organization|institution)', None),
            (r'(?:the |your |their )?(norm|rule|tradition|expectation|standard)', None),
            # Education/learning
            (r'(?:the |your |their )?(course|class|teacher|instructor|assignment|feedback)', None),
            (r'(?:the |your |their )?(student|learner|child|parent|mentor)', None),
            # Legal/justice
            (r'(?:the |your |their )?(defendant|suspect|victim|witness|jury|verdict)', None),
            # Environment
            (r'(?:the |your |their )?(environment|climate|carbon footprint|sustainability|pollution)', None),
            # Food/body
            (r'(?:the |your |their )?(food|meal|snack|diet|body|weight|nutrition)', None),
            # Media/communication
            (r'(?:the |your |their )?(message|article|news|source|media|story|argument)', None),
        ]
        for _tpat, _ in _target_patterns:
            _tm = re.search(_tpat, _subj_lower)
            if _tm:
                _target = _tm.group(0).strip()
                break

        # v1.0.8.0: Extract KEY PHRASES — expanded to 15+ patterns across all domains
        _key_phrases: list = []
        _phrase_patterns = [
            r'(?:feelings?|thoughts?|opinions?|views?)\s+(?:about|toward|on|regarding)\s+(.{3,80}?)(?:\.|$|\?|,)',
            r'(?:how|what)\s+(?:do you|did you)\s+(?:think|feel)\s+about\s+(.{3,80}?)(?:\.|$|\?|,)',
            r'(?:why did you|why do you)\s+(.{3,80}?)(?:\.|$|\?|,)',
            r'(?:describe|explain)\s+(?:your|how|why)\s+(.{3,80}?)(?:\.|$|\?|,)',
            r'(?:what made you|what led you to)\s+(.{3,80}?)(?:\.|$|\?|,)',
            # v1.0.8.0: NEW broad patterns
            r'(?:your experience with|your reaction to|your response to)\s+(.{3,80}?)(?:\.|$|\?|,)',
            r'(?:how has|how does|how did)\s+(.{3,80}?)\s+(?:affect|impact|influence|change)',
            r'(?:what do you think about|what are your thoughts on)\s+(.{3,80}?)(?:\.|$|\?|,)',
            r'(?:how would you describe|tell us about|share your)\s+(.{3,80}?)(?:\.|$|\?|,)',
            r'(?:what is your|what was your)\s+(?:experience|opinion|view|impression)\s+(?:of|about|with)\s+(.{3,80}?)(?:\.|$|\?|,)',
            r'(?:how do you cope with|how do you handle|how do you deal with)\s+(.{3,80}?)(?:\.|$|\?|,)',
            r'(?:what motivates you to|what encourages you to|what prevents you from)\s+(.{3,80}?)(?:\.|$|\?|,)',
            r'(?:imagine|suppose|consider)\s+(?:that|a situation where)\s+(.{3,80}?)(?:\.|$|\?|,)',
            r'(?:what would you do if|how would you react if)\s+(.{3,80}?)(?:\.|$|\?|,)',
            r'(?:what concerns you about|what excites you about|what worries you about)\s+(.{3,80}?)(?:\.|$|\?|,)',
            # v1.0.8.5: Imperative question patterns — "Please share...", "Reflect on..."
            r'(?:please\s+)?(?:share|reflect on|discuss|consider|think about)\s+(.{3,150}?)(?:\.|$|\?)',
            r'(?:please\s+)?(?:write about|elaborate on|comment on)\s+(.{3,150}?)(?:\.|$|\?)',
            # v1.0.8.5: Comparative patterns — "Compare X and Y", "Pros and cons of X"
            r'(?:compare|contrast)\s+(.{3,150}?)(?:\.|$|\?)',
            r'(?:pros and cons|advantages and disadvantages)\s+(?:of\s+)?(.{3,150}?)(?:\.|$|\?)',
        ]
        for _pp in _phrase_patterns:
            _pm = re.search(_pp, _subj_lower)
            if _pm:
                _kp = _pm.group(1).strip().rstrip('.')
                if len(_kp) > 3:
                    _key_phrases.append(_kp)
        # Use first key phrase as richer topic if available
        _rich_topic = _key_phrases[0] if _key_phrases else _topic

        # ── Step 2: Detect question INTENT — v1.0.8.4 with override ──
        # v1.0.8.4: Use pre-computed intent from engine if available (avoids
        # re-detection inconsistency between engine and ComprehensiveResponseGenerator)
        _intent = "opinion"  # default
        if override_intent and override_intent != "opinion":
            _intent = override_intent
        # v1.0.8.3: Check for creative/narrative/disclosure intents FIRST — most specific
        elif any(w in _subj_lower for w in ('conspiracy', 'theory', 'believe in', 'crazy belie',
                                           'paranormal', 'supernatural', 'superstition')):
            _intent = "creative_belief"
        elif any(w in _subj_lower for w in ('secret', 'only your family', 'nobody knows',
                                             'never told', 'private', 'confession', 'confess',
                                             'reveal', 'admit', 'embarrassing')):
            _intent = "personal_disclosure"
        elif any(w in _subj_lower for w in ('craziest', 'wildest', 'most memorable',
                                             'favorite', 'scariest', 'strangest', 'funniest',
                                             'most extreme', 'most interesting')):
            _intent = "creative_narrative"
        elif any(w in _subj_lower for w in ('tell us about a time', 'describe a time',
                                             'share a story', 'personal experience',
                                             'remember when', 'a situation where')):
            _intent = "personal_story"
        # Standard intents below — most specific first
        elif any(w in _subj_lower for w in ('why did', 'why do', 'explain why', 'reason for',
                                            'reasoning behind', 'what made you', 'what led you',
                                            'what caused you', 'what motivated', 'what drove')):
            _intent = "causal_explanation"
        elif any(w in _subj_lower for w in ('why', 'explain', 'reason', 'because', 'justify',
                                              'understand your', 'share your reasoning',
                                              'what factors', 'what considerations')):
            _intent = "explanation"
        elif any(w in _subj_lower for w in ('how do you feel', 'how did you feel', 'feelings about',
                                              'your feelings', 'emotional', 'how does', 'how did it make',
                                              'what emotions', 'your reaction', 'react to',
                                              'makes you feel', 'made you feel', 'your mood',
                                              'how did this affect you emotionally')):
            _intent = "emotional_reaction"
        elif any(w in _subj_lower for w in ('describe', 'tell us about', 'what happened',
                                              'walk us through', 'tell me about',
                                              'your experience', 'what was it like',
                                              'share your experience', 'share about',
                                              'paint a picture', 'give an account')):
            _intent = "description"
        elif any(w in _subj_lower for w in ('evaluate', 'rate', 'assess', 'compare', 'how would you rate',
                                              'your assessment', 'your evaluation',
                                              'pros and cons', 'strengths and weakness',
                                              'how well', 'how effective', 'quality of',
                                              'what did you think of', 'how satisfied')):
            _intent = "evaluation"
        elif any(w in _subj_lower for w in ('what do you think', 'your opinion', 'your view',
                                              'your thoughts', 'do you agree', 'do you believe',
                                              'your stance', 'your position', 'where do you stand',
                                              'what is your take', 'how do you see')):
            _intent = "opinion"
        elif any(w in _subj_lower for w in ('imagine if', 'suppose', 'hypothetically',
                                              'what would happen', 'what if', 'if you were',
                                              'what would you do if', 'in a scenario where')):
            _intent = "hypothetical"
        elif any(w in _subj_lower for w in ('would you', 'will you', 'do you plan', 'how likely',
                                              'in the future', 'what would you do',
                                              'predict', 'expect', 'forecast')):
            _intent = "prediction"
        elif any(w in _subj_lower for w in ('how do you cope', 'how do you handle', 'how do you manage',
                                              'what do you do when', 'how do you deal')):
            _intent = "description"
        elif any(w in _subj_lower for w in ('recommend', 'advice', 'suggest', 'what should',
                                              'what would you advise', 'how should',
                                              'tips for', 'best way to')):
            _intent = "recommendation"
        # v1.0.8.5: New intents — comparison, recall
        elif any(w in _subj_lower for w in ('compare', 'comparison', 'compared to', 'versus',
                                              'differ', 'difference', 'similarities',
                                              'better or worse', 'which is better',
                                              'pros and cons', 'advantages and disadvantages')):
            _intent = "comparison"
        elif any(w in _subj_lower for w in ('remember', 'recall', 'what do you remember',
                                              'what stuck with', 'looking back', 'in hindsight',
                                              'what stands out', 'think back')):
            _intent = "recall"
        elif _action:
            # If an action was detected, the question is likely asking about a decision
            _intent = "decision_explanation"

        # ── Step 3: Detect domain for vocabulary enrichment ──
        _dom_key = domain.value if hasattr(domain, 'value') else str(domain)
        _domain_vocab = self._get_domain_vocabulary(_dom_key, _subj_lower)

        # ── Step 4: Build INTENT-SPECIFIC core templates ──
        # Each intent type produces structurally different responses that actually
        # ANSWER the type of question being asked.

        _openers_casual = [
            "Honestly", "I gotta say", "For me personally", "I mean",
            "To be real", "Look", "I'll be honest", "The way I see it",
            "Thinking about it", "Being honest", "In my view", "So basically",
            "Ok so", "Yeah so", "Well", "Tbh", "Not gonna lie",
            "Real talk", "Here's the thing", "I'd say",
        ]
        _openers_formal = [
            "In my opinion", "I believe", "From my perspective",
            "After consideration", "Reflecting on this",
            "In my experience", "I would say", "To be frank",
        ]
        _openers = _openers_casual + _openers_formal

        # ── INTENT: causal_explanation / decision_explanation ──
        # "Why did you give $5?" → "I gave that amount because..."
        if _intent in ("causal_explanation", "decision_explanation"):
            if sentiment in ('very_positive', 'positive'):
                _cores = [
                    f"I chose to do that because {_rich_topic} felt like the right thing",
                    f"my decision was based on wanting to do the right thing when it comes to {_topic}",
                    f"I went with what I did because I believe in being fair about {_topic}",
                    f"the reason was pretty straightforward, I genuinely care about {_topic}",
                    f"I made that choice because {_topic} matters to me and I wanted to act on it",
                    f"it came down to my values, {_topic} is something I take seriously",
                    f"I decided that way because I felt good about {_topic} and wanted to show it",
                    f"my reasoning was that {_topic} deserves a positive response",
                    f"I acted the way I did because I have a favorable view of {_topic}",
                    f"when I thought about {_topic} I felt like being generous with my response",
                ]
                if _action:
                    _cores.extend([
                        f"I {_action} that way because it felt right given how I see {_topic}",
                        f"I {_action} based on my positive feelings about {_topic}",
                        f"the reason I {_action} is that I genuinely believe in {_topic}",
                    ])
                if _target:
                    _cores.extend([
                        f"I considered {_target} and felt positively about {_topic} so I went with that",
                        f"thinking about {_target} made me feel good about my decision on {_topic}",
                    ])
            elif sentiment in ('very_negative', 'negative'):
                _cores = [
                    f"I made that choice because I have real concerns about {_topic}",
                    f"my decision was based on not being comfortable with {_topic}",
                    f"I chose what I did because {_topic} doesn't sit right with me",
                    f"the reason was that I don't trust {_topic} or how it's going",
                    f"I went that way because {_topic} has been a negative experience for me",
                    f"my reasoning came from frustration with {_topic}",
                    f"I decided based on my skepticism about {_topic}",
                    f"I acted cautiously because {_topic} makes me uneasy",
                    f"my choice reflected my dissatisfaction with {_topic}",
                    f"I was hesitant because of how I feel about {_topic}",
                ]
                if _action:
                    _cores.extend([
                        f"I {_action} that way because I'm not on board with {_topic}",
                        f"the reason I {_action} is my negative experience with {_topic}",
                        f"I {_action} based on my concerns about {_topic}",
                    ])
                if _target:
                    _cores.extend([
                        f"considering {_target} and my doubts about {_topic} I went with the cautious option",
                        f"I wasn't confident about {_target} given my views on {_topic}",
                    ])
            else:  # neutral
                _cores = [
                    f"I made that choice without strong feelings about {_topic} either way",
                    f"my decision was kind of in the middle because I see both sides of {_topic}",
                    f"I went with what seemed reasonable given my mixed feelings on {_topic}",
                    f"I wasn't sure what the right call was with {_topic} so I split the difference",
                    f"there wasn't a strong pull either direction on {_topic} so I just went with my gut",
                    f"my reasoning was pretty neutral, {_topic} doesn't make me feel strongly",
                    f"I chose a moderate approach because {_topic} is complicated",
                    f"I didn't overthink it, {_topic} is one of those things I'm on the fence about",
                    f"I went with what felt natural since I don't have extreme views on {_topic}",
                    f"my decision was a middle ground because {_topic} has good and bad parts",
                ]
                if _action:
                    _cores.extend([
                        f"I {_action} without a strong reason, {_topic} just doesn't move me much",
                        f"I {_action} based on a gut feeling since I'm neutral on {_topic}",
                    ])

        # ── INTENT: emotional_reaction ──
        # "How do you feel about X?" → "X makes me feel..."
        elif _intent == "emotional_reaction":
            if sentiment in ('very_positive', 'positive'):
                _cores = [
                    f"when I think about {_topic} I feel pretty good about it",
                    f"{_topic} makes me feel hopeful and positive",
                    f"my emotional reaction to {_topic} is mostly positive",
                    f"I feel a sense of optimism when it comes to {_topic}",
                    f"{_topic} genuinely makes me happy to think about",
                    f"I get a warm feeling when I consider {_topic}",
                    f"emotionally {_topic} resonates with me in a good way",
                    f"my gut reaction to {_topic} is positive and I trust that feeling",
                    f"I feel encouraged by {_topic} and what it represents",
                    f"{_topic} gives me a sense of comfort and positivity",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _cores = [
                    f"{_topic} honestly makes me feel anxious",
                    f"my emotional reaction to {_topic} is frustration and concern",
                    f"when I think about {_topic} I feel uneasy",
                    f"{_topic} makes me feel worried about where things are headed",
                    f"I feel a sense of disappointment when it comes to {_topic}",
                    f"emotionally {_topic} hits me in a negative way",
                    f"my gut reaction to {_topic} is not great",
                    f"I feel frustrated when I think about {_topic}",
                    f"{_topic} stirs up negative emotions for me",
                    f"there's a sense of dread when I consider {_topic}",
                ]
            else:
                _cores = [
                    f"my feelings about {_topic} are honestly pretty mixed",
                    f"I don't feel strongly one way or another about {_topic}",
                    f"{_topic} doesn't stir up intense emotions for me",
                    f"emotionally I'm kind of neutral on {_topic}",
                    f"I feel ambivalent about {_topic} if I'm being honest",
                    f"{_topic} doesn't really move me emotionally",
                    f"my emotional response to {_topic} is muted",
                    f"I feel a mix of things about {_topic} that sort of cancel out",
                    f"some days {_topic} bothers me, other days I don't care",
                    f"I'm emotionally detached from {_topic} for the most part",
                ]

        # ── INTENT: description ──
        # "Describe your experience with X" → "My experience was..."
        elif _intent == "description":
            if sentiment in ('very_positive', 'positive'):
                _cores = [
                    f"my experience with {_topic} was generally positive",
                    f"what stood out to me about {_topic} was how well it worked",
                    f"when I engaged with {_topic} I found it to be a good experience overall",
                    f"I'd describe {_topic} as something that went well for me",
                    f"what I noticed about {_topic} was mostly positive things",
                    f"the main thing about {_topic} is that it left a good impression",
                    f"if I had to describe {_topic} I'd say it was a positive experience",
                    f"my interaction with {_topic} was better than I expected",
                    f"I'd characterize {_topic} as something worthwhile and positive",
                    f"the whole thing with {_topic} was a good experience in my view",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _cores = [
                    f"my experience with {_topic} was not great honestly",
                    f"what stood out about {_topic} was the problems with it",
                    f"when I dealt with {_topic} it was a frustrating experience",
                    f"I'd describe {_topic} as something that didn't go well",
                    f"the main thing about {_topic} is that it was disappointing",
                    f"if I had to describe {_topic} I'd focus on the negatives",
                    f"my interaction with {_topic} left me feeling let down",
                    f"I'd characterize {_topic} as a negative experience overall",
                    f"what I noticed about {_topic} was mostly issues and problems",
                    f"the whole thing with {_topic} was frustrating to deal with",
                ]
            else:
                _cores = [
                    f"my experience with {_topic} was just okay, nothing special",
                    f"what I noticed about {_topic} was a mix of good and bad",
                    f"when I engaged with {_topic} it was pretty unremarkable",
                    f"I'd describe {_topic} as average, had its moments",
                    f"the main thing about {_topic} is that it was fine but not memorable",
                    f"if I had to describe {_topic} I'd say it was middle of the road",
                    f"my interaction with {_topic} was neither here nor there",
                    f"I'd characterize {_topic} as a typical experience",
                    f"what happened with {_topic} was pretty standard stuff",
                    f"the whole thing with {_topic} was unremarkable but not bad",
                ]

        # ── INTENT: evaluation ──
        # "How would you rate X?" → "I'd rate X as..."
        elif _intent == "evaluation":
            if sentiment in ('very_positive', 'positive'):
                _cores = [
                    f"I'd give {_topic} a favorable assessment overall",
                    f"I rate {_topic} pretty highly based on what I've seen",
                    f"my evaluation of {_topic} is positive",
                    f"I think {_topic} measures up well against expectations",
                    f"from what I can tell {_topic} is doing things right",
                    f"I'd assess {_topic} as above average at least",
                    f"comparing to what I expected {_topic} exceeds it",
                    f"my honest evaluation is that {_topic} is good",
                    f"I'd say {_topic} scores well on the things that matter",
                    f"looking at it objectively {_topic} has more strengths than weaknesses",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _cores = [
                    f"I'd give {_topic} a poor assessment overall",
                    f"I rate {_topic} below average honestly",
                    f"my evaluation of {_topic} is not favorable",
                    f"I think {_topic} falls short of what it should be",
                    f"from what I've seen {_topic} has serious issues",
                    f"I'd assess {_topic} as needing major improvement",
                    f"compared to what I expected {_topic} disappoints",
                    f"my honest evaluation is that {_topic} isn't good enough",
                    f"I'd say {_topic} fails on the things that matter most",
                    f"looking at it objectively {_topic} has more problems than positives",
                ]
            else:
                _cores = [
                    f"I'd give {_topic} a mixed assessment",
                    f"I rate {_topic} somewhere in the middle",
                    f"my evaluation of {_topic} is neither great nor terrible",
                    f"I think {_topic} has some strengths and some weaknesses",
                    f"from what I can tell {_topic} is about average",
                    f"I'd assess {_topic} as adequate but not outstanding",
                    f"compared to what I expected {_topic} is about right",
                    f"my honest evaluation is that {_topic} is just ok",
                    f"I'd say {_topic} has room for improvement but isn't bad",
                    f"looking at it objectively {_topic} is a mixed bag",
                ]

        # ── INTENT: prediction ──
        # "Would you do this again?" / "How likely..." → "I would/wouldn't..."
        elif _intent == "prediction":
            if sentiment in ('very_positive', 'positive'):
                _cores = [
                    f"I would definitely engage with {_topic} again",
                    f"I'm likely to continue being positive about {_topic} going forward",
                    f"I can see myself being involved with {_topic} in the future",
                    f"in the future I'd expect to support {_topic}",
                    f"I'm optimistic about where {_topic} is going",
                    f"I would make the same choice about {_topic} again",
                    f"going forward I plan to stay engaged with {_topic}",
                    f"I'd do the same thing again when it comes to {_topic}",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _cores = [
                    f"I would probably avoid {_topic} in the future",
                    f"I'm unlikely to engage with {_topic} again",
                    f"going forward I'd be more cautious about {_topic}",
                    f"I don't see myself supporting {_topic} in the future",
                    f"I would make a different choice about {_topic} next time",
                    f"in the future I'd probably steer clear of {_topic}",
                    f"I'm pessimistic about where {_topic} is headed",
                    f"I would not do the same thing again regarding {_topic}",
                ]
            else:
                _cores = [
                    f"I'm not sure what I'd do about {_topic} in the future",
                    f"it depends on how {_topic} develops going forward",
                    f"I might or might not engage with {_topic} again",
                    f"my future involvement with {_topic} is uncertain",
                    f"I'd have to think about {_topic} more before deciding",
                    f"going forward it really depends on the circumstances with {_topic}",
                    f"I could go either way on {_topic} in the future",
                    f"I'd keep my options open regarding {_topic}",
                ]

        # ── INTENT: hypothetical (v1.0.8.4) ──
        # "Imagine if..." / "What would you do if..." → scenario-based response
        elif _intent == "hypothetical":
            if sentiment in ('very_positive', 'positive'):
                _cores = [
                    f"if that happened with {_topic} I think I'd embrace it wholeheartedly",
                    f"in that scenario I'd be pretty excited about {_topic}",
                    f"honestly if that were the case with {_topic} I'd be all for it",
                    f"I'd jump at the chance to engage positively with {_topic}",
                    f"in that hypothetical I would lean into {_topic} with enthusiasm",
                    f"if I imagine that scenario with {_topic} I see it going well",
                    f"that situation would make me feel optimistic about {_topic}",
                    f"I think in that case I'd support {_topic} even more than I already do",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _cores = [
                    f"if that happened with {_topic} I'd be pretty worried honestly",
                    f"in that scenario I'd probably try to distance myself from {_topic}",
                    f"honestly if that were the case with {_topic} I'd be concerned",
                    f"I'd be reluctant to go along with {_topic} in that situation",
                    f"in that hypothetical I would have serious reservations about {_topic}",
                    f"if I imagine that scenario with {_topic} I see real problems",
                    f"that situation would make me more cautious about {_topic}",
                    f"I think in that case I'd push back against {_topic}",
                ]
            else:
                _cores = [
                    f"if that happened with {_topic} I honestly don't know what I'd do",
                    f"in that scenario I'd probably need time to figure out {_topic}",
                    f"that's a tough one when it comes to {_topic}",
                    f"I'd have to weigh the options carefully with {_topic}",
                    f"in that hypothetical my response to {_topic} would depend on the details",
                    f"if I imagine that scenario I'm genuinely torn about {_topic}",
                    f"that situation would require a lot of thought about {_topic}",
                    f"I think in that case I'd want more info before deciding on {_topic}",
                ]

        # ── INTENT: recommendation (v1.0.8.4) ──
        # "What would you recommend?" / "What advice would you give?" → actionable suggestion
        elif _intent == "recommendation":
            if sentiment in ('very_positive', 'positive'):
                _cores = [
                    f"I'd definitely recommend engaging with {_topic}, it's been a positive experience",
                    f"my advice would be to give {_topic} a real chance",
                    f"I would suggest that people approach {_topic} with an open mind, it's worth it",
                    f"I'd recommend {_topic} based on my own experience with it",
                    f"I think the best approach to {_topic} is to dive in and embrace it",
                    f"my recommendation is to take {_topic} seriously and invest in it",
                    f"the best advice I can give about {_topic} is to stay positive and committed",
                    f"I would tell anyone asking about {_topic} to go for it",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _cores = [
                    f"I'd recommend being really careful with {_topic}",
                    f"my advice would be to think twice before getting involved with {_topic}",
                    f"I would suggest approaching {_topic} with serious caution",
                    f"I'd recommend that people look at the downsides of {_topic} first",
                    f"the best approach to {_topic} in my view is to be skeptical",
                    f"my recommendation is to avoid {_topic} unless absolutely necessary",
                    f"I'd tell anyone asking about {_topic} to do their research first",
                    f"the advice I'd give about {_topic} is to lower your expectations",
                ]
            else:
                _cores = [
                    f"I'd recommend doing some research before committing to {_topic}",
                    f"my advice is to consider both sides of {_topic} carefully",
                    f"I would suggest keeping an open mind about {_topic} but being realistic",
                    f"I'd recommend trying {_topic} on a small scale first to see how it goes",
                    f"the best approach to {_topic} is probably to start slow and evaluate",
                    f"my recommendation is to weigh your own priorities when it comes to {_topic}",
                    f"I'd tell someone asking about {_topic} that it depends on their situation",
                    f"the honest advice on {_topic} is that there's no one-size-fits-all answer",
                ]

        # ── INTENT: explanation (general "why" without specific action) ──
        elif _intent == "explanation":
            if sentiment in ('very_positive', 'positive'):
                _cores = [
                    f"the reason I feel this way is that {_topic} aligns with my values",
                    f"I think it comes down to the fact that {_topic} is something I genuinely believe in",
                    f"my reasoning is that {_topic} has consistently shown positive results",
                    f"I'd explain my position by saying {_topic} just makes sense to me",
                    f"the main reason is that my experiences with {_topic} have been positive",
                    f"I feel this way because {_topic} represents something I care about",
                    f"it's because {_topic} has earned my trust and support over time",
                    f"the explanation is simple, {_topic} works and I've seen the evidence",
                    f"my positive view of {_topic} comes from actual experience not just theory",
                    f"I support {_topic} because the benefits outweigh any downsides I can see",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _cores = [
                    f"the reason I feel this way is that {_topic} has let me down",
                    f"my reasoning is based on real problems I've seen with {_topic}",
                    f"I'd explain my position by pointing to the issues with {_topic}",
                    f"the main reason is that {_topic} hasn't lived up to expectations",
                    f"I feel this way because {_topic} causes more harm than good",
                    f"it comes down to the fact that {_topic} doesn't deliver on its promises",
                    f"my negative view of {_topic} is based on what I've actually observed",
                    f"the explanation is that {_topic} keeps failing in ways that matter",
                    f"I'm critical because {_topic} affects real people negatively",
                    f"I oppose {_topic} because the evidence points to serious problems",
                ]
            else:
                _cores = [
                    f"the reason I'm undecided is that {_topic} has both good and bad aspects",
                    f"I'd explain my position by saying {_topic} is genuinely complicated",
                    f"my reasoning is that the evidence on {_topic} is mixed",
                    f"the main reason I'm moderate is that {_topic} has real tradeoffs",
                    f"I feel this way because {_topic} isn't black and white",
                    f"it comes down to the fact that there are valid arguments about {_topic} on both sides",
                    f"my views on {_topic} are nuanced because the reality is complex",
                    f"I can't give a simple explanation because {_topic} genuinely has pros and cons",
                    f"my moderate stance on {_topic} comes from seeing multiple perspectives",
                    f"the reason I'm neutral is that I haven't seen enough to be convinced either way about {_topic}",
                ]

        # ── INTENT: creative_belief (v1.0.8.3) ──
        # "Tell us your conspiracy theory" → actual conspiracy theory content
        elif _intent == "creative_belief":
            _cores = [
                "I genuinely think there's way more government surveillance than people realize. There are patents for technology that supposedly doesn't exist yet and that's suspicious",
                "my theory is that big pharma deliberately suppresses cheap generic remedies because there's no money in curing people with $5 drugs",
                "I believe most major media outlets coordinate their coverage. Not like a secret society but they all push the same narratives and bury the same stories",
                "I think social media algorithms are specifically designed to make people angry and addicted. Not as a side effect but as the actual product",
                "I'm convinced the food industry knowingly puts addictive compounds in processed food. The sugar content in everything is not accidental",
                "I think most political scandals are coordinated distractions from actual policy changes happening behind the scenes",
                "my belief is that certain energy technologies have been suppressed because they'd disrupt too many powerful industries",
                "I genuinely believe the housing crisis is manufactured by investment firms who buy up supply to keep prices artificially high",
                "I think the education system is designed to produce compliant workers not critical thinkers. The whole structure is basically factory conditioning",
                "I believe there are way more backroom deals in politics than anyone admits. Lobbying is just legalized corruption",
                "I'm pretty sure pharmaceutical companies have way too much influence over what doctors prescribe",
                "my theory is that planned obsolescence is deliberately engineered. Companies make things break so you buy new ones",
                "I think climate change data is being downplayed by corporations who profit from the status quo",
                "I believe social media companies know their products destroy mental health and just don't care because engagement equals money",
            ]

        # ── INTENT: personal_disclosure (v1.0.8.3) ──
        # "Tell us something only your family knows" → actual personal disclosure
        elif _intent == "personal_disclosure":
            _cores = [
                "something my family knows is that I struggled badly with anxiety in my early twenties. I barely left the house for almost a year",
                "my family knows I almost dropped out of college. I was one semester away from quitting because I felt completely lost",
                "only my family knows about a medical scare I had a few years ago. I kept it completely private because I didn't want the attention",
                "something personal is that my family went through serious financial trouble when I was a teenager. It changed my relationship with money forever",
                "my family knows I'm much more sensitive than I let on. At work I seem easy-going but at home I worry about everything",
                "only my family knows I considered a completely different career path. I was accepted somewhere else but switched last minute",
                "my family knows I was bullied pretty severely growing up. By the time I met my current friends I'd completely reinvented myself",
                "something only my family would know is that I have a learning difference I've never told anyone at work about",
                "my family knows I went through a really rough patch after a major relationship ended. I basically stopped functioning for weeks",
                "something personal is that I secretly help a family member who's struggling financially. I don't talk about it to anyone else",
                "only my family knows about a promise I made years ago that I still keep. It's shaped a lot of my decisions without anyone realizing",
                "my family knows I have a completely different side to my personality that I never show at work or with friends",
            ]

        # ── INTENT: creative_narrative (v1.0.8.3) ──
        # "Tell us your craziest/wildest/most memorable X" → actual narrative
        elif _intent == "creative_narrative":
            _cores = [
                f"the craziest thing about {_topic} that I've experienced was totally unexpected. Everything I assumed turned out wrong and it changed how I think",
                f"I have a wild {_topic} story. I witnessed something most people wouldn't believe. It happened years ago and I still think about it",
                f"my most memorable experience with {_topic} happened when I was least expecting it. The situation was so bizarre I had to tell someone immediately",
                f"the wildest thing about {_topic} in my life escalated beyond anything I could've predicted. The chain of events seemed almost too perfect to be coincidental",
                f"I have a {_topic} story I rarely tell because people don't believe me. But it genuinely happened and it makes me question certain assumptions",
                f"my experience with {_topic} took such an unexpected turn that I still bring it up years later",
                f"here's my {_topic} story: I found myself in a situation where normal rules didn't apply. Everyone involved was confused",
                f"when it comes to {_topic} I once had an experience that completely defied my expectations",
            ]

        # ── INTENT: personal_story (v1.0.8.3) ──
        # "Tell us about a time when..." → personal narrative
        elif _intent == "personal_story":
            _cores = [
                f"there was a time when {_topic} came up in my life unexpectedly. I was dealing with a work situation and it forced me to confront how I actually felt",
                f"I remember a specific experience with {_topic} that stays with me. It happened during a difficult period and taught me something important about myself",
                f"my most significant experience with {_topic} was when I had to make a real decision about it, not just think abstractly",
                f"I have a personal story about {_topic} from when I was younger. At the time I didn't understand but looking back it makes sense",
                f"my experience with {_topic} really came into focus during a conversation with someone close to me. They said something that shifted my perspective",
                f"there's a specific moment involving {_topic} that changed how I approach things. It wasn't dramatic but it was a turning point for me",
            ]

        # ── INTENT: comparison (v1.0.8.5) ──
        # "Compare X and Y" / "Pros and cons" → balanced comparative response
        elif _intent == "comparison":
            if sentiment in ('very_positive', 'positive'):
                _cores = [
                    f"when I compare {_topic} to the alternatives I think {_topic} comes out ahead",
                    f"the main advantage of {_topic} is that it actually delivers on what it promises",
                    f"compared to other options {_topic} is stronger in the areas that matter to me",
                    f"I've weighed the pros and cons and {_topic} has more going for it",
                    f"what sets {_topic} apart is the real impact I've seen compared to alternatives",
                    f"there are trade-offs with anything but {_topic} wins in my comparison",
                    f"I'd say {_topic} has clear advantages over the alternatives I've considered",
                    f"the difference between {_topic} and the rest is noticeable in my experience",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _cores = [
                    f"when I compare {_topic} to alternatives it falls short in important ways",
                    f"the main disadvantage of {_topic} is that better options exist and people know it",
                    f"compared to other options {_topic} is weaker in the areas that actually matter",
                    f"I've weighed the pros and cons and {_topic} has too many downsides",
                    f"what makes {_topic} worse is that the alternatives don't have the same problems",
                    f"the comparison doesn't favor {_topic} in my experience",
                    f"I'd say {_topic} has clear disadvantages compared to what else is out there",
                    f"the difference between {_topic} and better options is pretty stark",
                ]
            else:
                _cores = [
                    f"when I compare {_topic} to alternatives there are genuine trade-offs either way",
                    f"the pros and cons of {_topic} are pretty balanced honestly",
                    f"compared to other options {_topic} has some strengths and some weaknesses",
                    f"I've considered the alternatives and {_topic} isn't clearly better or worse",
                    f"the comparison between {_topic} and other options depends on what you value",
                    f"it's hard to say whether {_topic} is better or worse — there are valid points both ways",
                    f"I see advantages and disadvantages of {_topic} compared to the alternatives",
                    f"the trade-offs with {_topic} make it hard to give a definitive comparison",
                ]

        # ── INTENT: recall (v1.0.8.5) ──
        # "What do you remember about X?" → memory-focused response
        elif _intent == "recall":
            if sentiment in ('very_positive', 'positive'):
                _cores = [
                    f"what I remember most about {_topic} is how good it felt at the time",
                    f"looking back on {_topic} the thing that stands out is the positive impact",
                    f"I recall {_topic} being a genuinely positive experience that stuck with me",
                    f"when I think back to {_topic} what I remember most is feeling good about it",
                    f"the main thing I remember about {_topic} is that it exceeded my expectations",
                    f"in hindsight {_topic} was even better than I realized at the time",
                    f"what sticks with me about {_topic} is how right it felt",
                    f"looking back {_topic} is something I remember fondly",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _cores = [
                    f"what I remember most about {_topic} is the frustration I felt",
                    f"looking back on {_topic} what stands out is how problematic it was",
                    f"I recall {_topic} being a negative experience that I haven't forgotten",
                    f"when I think back to {_topic} the first thing that comes to mind is the problems",
                    f"the main thing I remember about {_topic} is how disappointing it turned out",
                    f"in hindsight {_topic} was even worse than I thought at the time",
                    f"what sticks with me about {_topic} is the sense of frustration",
                    f"looking back {_topic} is something I remember with some bitterness",
                ]
            else:
                _cores = [
                    f"what I remember about {_topic} is kind of mixed honestly",
                    f"looking back on {_topic} I remember both good and bad parts",
                    f"my memories of {_topic} are neither strongly positive nor negative",
                    f"when I think back to {_topic} no single thing dominates my memory",
                    f"I recall {_topic} being a fairly unremarkable experience overall",
                    f"in hindsight {_topic} was about what I expected",
                    f"what sticks with me about {_topic} is the ambiguity of it all",
                    f"looking back {_topic} doesn't stir up strong feelings one way or another",
                ]

        # ── INTENT: opinion (default) ──
        else:
            if sentiment in ('very_positive', 'positive'):
                _cores = [
                    f"I feel good about {_topic} and where things are heading",
                    f"{_topic} is something I support and believe in",
                    f"I'm positive about {_topic} based on what I know",
                    f"I think {_topic} is going in a good direction",
                    f"my views on {_topic} are favorable for real reasons",
                    f"{_topic} lines up with how I see things",
                    f"there's a lot to appreciate about {_topic}",
                    f"my experience with {_topic} has been positive",
                    f"I believe {_topic} is on the right track",
                    f"I genuinely believe in {_topic}",
                    f"I keep coming back to how much I appreciate {_topic}",
                    f"{_topic} just resonates with me",
                    f"I've thought about {_topic} a lot and I'm still positive",
                    f"there's something about {_topic} that just clicks for me",
                    f"my feelings about {_topic} have only gotten more positive",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _cores = [
                    f"I'm not happy about {_topic} and I think that's justified",
                    f"{_topic} really concerns me on multiple levels",
                    f"I'm critical of {_topic} and the direction it's going",
                    f"{_topic} frustrates me because it could be so much better",
                    f"there are real problems with {_topic} that need addressing",
                    f"I have serious issues with {_topic}",
                    f"{_topic} is headed the wrong way in my view",
                    f"my experience with {_topic} has been largely negative",
                    f"I wish {_topic} was handled differently",
                    f"every time I think about {_topic} it bothers me more",
                    f"I've lost faith in {_topic}",
                    f"the more I learn about {_topic} the less I like it",
                    f"{_topic} has let me down repeatedly",
                    f"something is seriously wrong with {_topic}",
                    f"I'm pretty fed up with {_topic} at this point",
                ]
            else:  # neutral
                _cores = [
                    f"I have mixed feelings about {_topic}",
                    f"I can see both sides when it comes to {_topic}",
                    f"{_topic} is complicated for me to have a strong view on",
                    f"I'm somewhere in the middle on {_topic}",
                    f"I've gone back and forth on {_topic} honestly",
                    f"{_topic} has good and bad parts that kind of balance out",
                    f"I'm still sorting out how I feel about {_topic}",
                    f"there are valid points on both sides of {_topic}",
                    f"my take on {_topic} is pretty balanced",
                    f"{_topic} doesn't make me feel strongly one way or the other",
                    f"I'm kinda neutral on {_topic}",
                    f"some parts of {_topic} I like, some I don't",
                    f"I think {_topic} is more complicated than people make it",
                    f"I could be convinced either way about {_topic}",
                    f"there's no easy answer when it comes to {_topic}",
                ]

        # ── Step 5: Build DOMAIN-ENRICHED elaborations ──
        # Generic elaborations are used when domain-specific ones aren't available
        if sentiment in ('very_positive', 'positive'):
            _generic_elabs = [
                "It lines up with my values.",
                "My own experiences back this up.",
                "I've seen good things come from this.",
                "I feel confident about where I stand.",
                "It just makes sense to me.",
                "This really matters to me personally.",
                "My opinion has gotten stronger over time.",
                "I know not everyone agrees but I'm firm on this.",
                "This is based on what I've actually experienced.",
                "I've put real thought into this.",
                "I don't say this lightly.",
                "Life experience has taught me to value this.",
                "I stand behind what I said.",
            ]
        elif sentiment in ('very_negative', 'negative'):
            _generic_elabs = [
                "There are problems people aren't addressing.",
                "I've seen this go wrong firsthand.",
                "Things really need to change.",
                "This has been frustrating.",
                "My feelings have only gotten worse.",
                "The situation is worse than people realize.",
                "We can and should do better.",
                "I've tried to stay open-minded but it's difficult.",
                "I know others who feel the exact same way.",
                "I don't think I'm being unreasonable here.",
                "I feel strongly about this and I don't apologize for it.",
                "It's affecting real people and that's not ok.",
                "Something has got to give.",
            ]
        else:
            _generic_elabs = [
                "I try to keep an open mind.",
                "There are good arguments on both sides.",
                "I just gave my honest take.",
                "I'd need more info to feel strongly.",
                "It depends on the specifics for me.",
                "I try not to be extreme about anything.",
                "My views might shift as I learn more.",
                "Nuance matters here.",
                "I just call it like I see it.",
                "I don't feel the need to pick a side.",
                "I think the truth is somewhere in the middle.",
                "I can't pretend to have all the answers.",
                "I try to be honest even when it's complicated.",
            ]

        # Combine domain-specific and generic elaborations
        _elaborations = _domain_vocab.get('elaborations_' + ('pos' if sentiment in ('very_positive', 'positive') else 'neg' if sentiment in ('very_negative', 'negative') else 'neu'), [])
        _elaborations = list(_elaborations) + _generic_elabs  # domain-specific first for higher selection probability

        # ── Step 6: Compose response from parts ──
        core = rng.choice(_cores)
        # 50% chance to add opener
        if rng.random() < 0.50:
            opener = rng.choice(_openers)
            response = f"{opener}, {core}."
        else:
            response = f"{core[0].upper()}{core[1:]}."
        # 55% chance to add elaboration
        if rng.random() < 0.55:
            elab = rng.choice(_elaborations)
            response += f" {elab}"
        # 20% chance to add intent-specific coda
        if rng.random() < 0.20:
            _codas = {
                "causal_explanation": [
                    " That's basically why.", " That's my reasoning.", " That's what it came down to.",
                    " So yeah, that's why.", " Make of that what you will.",
                    " Those are the real reasons.", " I don't think there's a simpler way to explain it.",
                ],
                "decision_explanation": [
                    " That's why I made that choice.", " Looking back I'd probably do the same.",
                    " That's the honest reason.", " So that's what drove my decision.",
                    " I don't regret that choice.", " It felt like the right call at the time.",
                ],
                "emotional_reaction": [
                    " That's just how it makes me feel.", " Can't really help how I feel about it.",
                    " Hard to put into words exactly.", " I feel strongly about it.",
                    " Those are my honest emotions on it.",
                    " Feelings are what they are.", " I tried to express this genuinely.",
                ],
                "description": [
                    " That's basically what happened.", " That's how I'd sum it up.",
                    " That captures my experience.", " That's the gist of it.",
                    " I tried to be accurate about it.", " Those are the main things I noticed.",
                ],
                "evaluation": [
                    " That's my honest assessment.", " Take it for what it's worth.",
                    " That's how I'd rate it.", " I stand by that evaluation.",
                    " I tried to be fair.", " Those are my genuine impressions.",
                ],
                "explanation": [
                    " That's where I'm coming from.", " I think that explains it.",
                    " That's my reasoning in a nutshell.", " So that's why I think this way.",
                    " I hope that makes sense.", " That's the logic behind my answer.",
                ],
                "prediction": [
                    " That's my best guess.", " Time will tell if I'm right.",
                    " We'll see how it goes.", " That's what I expect anyway.",
                    " I could be wrong but that's where I'd put my money.",
                ],
                "hypothetical": [
                    " But who knows, it's all hypothetical.", " Hard to say for certain.",
                    " That's my honest reaction to the scenario.", " In reality it might go differently.",
                    " I'd have to actually be in that situation to know for sure.",
                ],
                "recommendation": [
                    " That's my honest recommendation.", " Take it or leave it, that's my advice.",
                    " Your situation might be different though.", " I hope that helps.",
                    " That's what I'd tell a friend.", " Do with that what you will.",
                ],
                "comparison": [
                    " That's how I see the comparison.", " Your priorities might differ though.",
                    " It really depends on what matters to you.", " That's my honest assessment of both sides.",
                    " Not everyone would agree with my ranking.", " The tradeoffs are real.",
                ],
                "recall": [
                    " That's what stuck with me.", " Memory is funny that way.",
                    " I think that captures it.", " Time changes perspective, but that's what I recall.",
                    " Those are the memories that stand out.", " Can't forget that even if I tried.",
                ],
                "opinion": [
                    " That's my honest take.", " I said what I said.",
                    " Take it or leave it.", " That's where I stand.",
                    " I'm not going to pretend otherwise.", " That's genuinely how I see it.",
                    " I don't think I'll change my mind on this.",
                ],
            }
            _intent_codas = _codas.get(_intent, _codas["opinion"])
            response += rng.choice(_intent_codas)
        return response

    def _get_domain_vocabulary(self, domain_key: str, subject_lower: str) -> Dict[str, list]:
        """Return domain-specific vocabulary for enriching elaborations.

        v1.0.7.9: NEW METHOD — Provides domain-specific sentence fragments that
        make elaborations sound like they come from someone who actually participated
        in that type of study, not just expressed generic opinions.

        Returns dict with keys: elaborations_pos, elaborations_neg, elaborations_neu
        """
        _vocab: Dict[str, list] = {}

        # ── Economic games vocabulary ──
        _econ_game_domains = {'dictator_game', 'trust_game', 'ultimatum_game', 'public_goods',
                              'prisoners_dilemma', 'cooperation', 'fairness'}
        _is_econ = domain_key in _econ_game_domains or any(
            w in subject_lower for w in ('dictator', 'trust game', 'ultimatum', 'public good',
                                          'prisoner', 'cooperat', 'split', 'endow'))
        if _is_econ:
            _is_dictator = 'dictator' in subject_lower or domain_key == 'dictator_game'
            _is_trust = 'trust' in subject_lower or domain_key == 'trust_game'
            _is_ultimatum = 'ultimatum' in subject_lower or domain_key == 'ultimatum_game'
            _is_public = 'public' in subject_lower or domain_key == 'public_goods'

            if _is_dictator:
                _vocab['elaborations_pos'] = [
                    "I believe sharing is the right thing to do even with strangers.",
                    "Fairness matters to me so I wanted to split things more evenly.",
                    "I'd want someone to be generous with me so I tried to do the same.",
                    "Keeping everything felt wrong even if the rules allowed it.",
                    "My upbringing taught me to share when I can.",
                    "I think about how the other person would feel getting nothing.",
                ]
                _vocab['elaborations_neg'] = [
                    "I don't know this person so I don't feel obligated to give much.",
                    "The rules said I could keep it so I did what made sense for me.",
                    "I look out for my own interests first, that's just practical.",
                    "Why give money to a complete stranger when you don't have to?",
                    "I earned the right to decide how it gets split.",
                    "Generosity has limits especially with people I'll never meet again.",
                ]
                _vocab['elaborations_neu'] = [
                    "I just picked something in the middle without thinking too hard.",
                    "I didn't want to seem greedy or too generous so I went moderate.",
                    "I split things the way I thought most people probably would.",
                    "No strong feelings about the amount, just went with what felt normal.",
                    "I gave a bit but kept a bit too, seemed reasonable.",
                ]
            elif _is_trust:
                _vocab['elaborations_pos'] = [
                    "I think showing trust brings out the best in people.",
                    "You have to give people a chance to prove they're trustworthy.",
                    "My experience is that trust usually gets reciprocated.",
                    "Starting with trust is better than starting with suspicion.",
                    "I'd rather risk being taken advantage of than be cynical.",
                ]
                _vocab['elaborations_neg'] = [
                    "I've been burned before so I'm careful about trusting strangers.",
                    "Trust needs to be earned and this person hasn't earned it.",
                    "The rational thing is to protect yourself when you don't know someone.",
                    "People are unpredictable so I played it safe.",
                    "I kept more because I wasn't confident they'd send anything back.",
                ]
                _vocab['elaborations_neu'] = [
                    "I sent some but not too much, a moderate level of trust.",
                    "I hedged my bets since I wasn't sure what they'd do.",
                    "Seemed smart to trust a little but not go all in.",
                    "I went with an amount I could afford to lose if they didn't return it.",
                ]
            elif _is_ultimatum:
                _vocab['elaborations_pos'] = [
                    "Fair offers deserve to be accepted.",
                    "I believe in rewarding people who are reasonable.",
                    "The split was fair enough and that matters to me.",
                    "Getting something is always better than getting nothing out of spite.",
                    "I accepted because the other person was being decent about it.",
                ]
                _vocab['elaborations_neg'] = [
                    "The offer was insultingly low and I'd rather reject on principle.",
                    "I refused because accepting would reward greedy behavior.",
                    "Sometimes you have to sacrifice money to send a message about fairness.",
                    "I'm not going to accept crumbs while someone else takes most of it.",
                    "Punishing unfair offers is how you keep people honest.",
                ]
                _vocab['elaborations_neu'] = [
                    "The offer was borderline so I could have gone either way.",
                    "I thought about rejecting but decided it wasn't worth it.",
                    "Not the best offer but not offensive either.",
                    "I accepted without strong feelings about the fairness of it.",
                ]
            elif _is_public:
                _vocab['elaborations_pos'] = [
                    "If everyone contributes then everyone benefits.",
                    "I believe in doing my part for the group.",
                    "Contributing to the common good is the right thing.",
                    "Free riding goes against my values even if it would pay off.",
                    "I trust that cooperation produces better outcomes for everyone.",
                ]
                _vocab['elaborations_neg'] = [
                    "Why contribute when others might free ride off my effort?",
                    "I can't control what others do so I protected my own share.",
                    "The returns on contributing aren't guaranteed so I was cautious.",
                    "I've seen group efforts fail before so I held back.",
                    "Looking out for myself seemed like the smart play here.",
                ]
                _vocab['elaborations_neu'] = [
                    "I contributed a bit but kept some for myself, seemed fair.",
                    "I wasn't sure what others would do so I split the difference.",
                    "A moderate contribution felt right given the uncertainty.",
                    "I went with what I thought an average person would do.",
                ]
            else:
                # Generic economic game elaborations
                _vocab['elaborations_pos'] = [
                    "I tried to be fair in how I approached this.",
                    "My choice reflected my belief in treating people decently.",
                    "I wanted to do the right thing in this situation.",
                ]
                _vocab['elaborations_neg'] = [
                    "I made the choice that was best for me personally.",
                    "Self-interest is rational when you don't know the other person.",
                    "I was strategic about my decision.",
                ]
                _vocab['elaborations_neu'] = [
                    "I went with what seemed like a moderate choice.",
                    "I didn't overthink it, just picked something reasonable.",
                ]

        # ── Political vocabulary ──
        elif domain_key in ('political', 'polarization', 'partisanship', 'voting', 'policy_attitudes') or \
                any(w in subject_lower for w in ('politic', 'democrat', 'republican', 'liberal', 'conservative',
                                                   'trump', 'biden', 'partisan', 'election', 'vote', 'policy')):
            _vocab['elaborations_pos'] = [
                "My political values are deeply held and this aligns with them.",
                "I think the country would be better off if more people felt this way.",
                "This reflects what I believe about how government should work.",
                "My views come from caring about the direction of the country.",
                "This isn't just politics to me, it's about real people's lives.",
                "I've always felt strongly about this and nothing has changed that.",
            ]
            _vocab['elaborations_neg'] = [
                "The political situation in this country genuinely worries me.",
                "I think we're heading in a dangerous direction politically.",
                "People on the other side just don't understand what's at stake.",
                "I'm frustrated because nothing seems to change no matter who's in charge.",
                "The political divide feels personal and it's hard to ignore.",
                "I wish politicians actually cared about regular people.",
            ]
            _vocab['elaborations_neu'] = [
                "I try not to get too caught up in political tribalism.",
                "I can see merit in arguments from different political perspectives.",
                "I don't think either side has all the answers.",
                "Politics is so polarized right now that being moderate feels lonely.",
                "I'd rather evaluate issues individually than follow a party line.",
                "I think most people are reasonable but the extremes get all the attention.",
            ]

        # ── Health/medical vocabulary ──
        elif domain_key in ('health', 'medical_decision', 'wellbeing', 'vaccination', 'stress') or \
                any(w in subject_lower for w in ('health', 'medical', 'vaccine', 'treatment', 'illness',
                                                   'wellbeing', 'mental health', 'doctor', 'patient')):
            _vocab['elaborations_pos'] = [
                "Taking care of my health is a priority for me.",
                "I trust medical professionals to guide these decisions.",
                "Good health is everything and I'm grateful for mine.",
                "I believe in being proactive about health rather than reactive.",
                "My positive health experiences reinforce my approach.",
            ]
            _vocab['elaborations_neg'] = [
                "Health concerns can be really stressful and overwhelming.",
                "I've had negative experiences that make me cautious about health decisions.",
                "The healthcare system doesn't always work for regular people.",
                "I worry about health outcomes and that affects how I think about this.",
                "Health anxiety is real and it impacts my daily life.",
            ]
            _vocab['elaborations_neu'] = [
                "Health is important but I try not to obsess over it.",
                "I do what I can but I'm not extreme about health choices.",
                "I follow basic health advice without going overboard.",
                "Health decisions are personal and there's no one right answer.",
            ]

        # ── Consumer/product vocabulary ──
        elif domain_key in ('consumer', 'brand', 'purchase_intent', 'product_evaluation',
                            'advertising', 'brand_loyalty', 'service_quality') or \
                any(w in subject_lower for w in ('product', 'brand', 'purchase', 'buy', 'shop',
                                                   'consumer', 'price', 'quality', 'advertis')):
            _vocab['elaborations_pos'] = [
                "Quality matters to me and this delivers on that.",
                "I can see myself coming back to this brand.",
                "The value for what you get is genuinely good.",
                "It met my expectations and then some.",
                "I'd recommend this to someone looking for something like it.",
            ]
            _vocab['elaborations_neg'] = [
                "There are better options out there for the same price.",
                "The quality doesn't match what they're charging.",
                "I expected more based on the marketing.",
                "I wouldn't waste my money on this again.",
                "It just doesn't deliver on what it promises.",
            ]
            _vocab['elaborations_neu'] = [
                "It's fine for what it is, nothing more nothing less.",
                "I'd consider alternatives before committing again.",
                "Average product, average experience.",
                "It does the job but nothing stands out about it.",
            ]

        # ── Moral/ethical vocabulary ──
        elif domain_key in ('moral_judgment', 'moral_dilemma', 'ethics') or \
                any(w in subject_lower for w in ('moral', 'ethic', 'right wrong', 'dilemma', 'justice')):
            _vocab['elaborations_pos'] = [
                "My moral compass tells me this is the right thing.",
                "I try to act in a way I'd be proud of.",
                "Ethics matter and I think this reflects good values.",
                "Doing the right thing isn't always easy but it's important.",
            ]
            _vocab['elaborations_neg'] = [
                "This crosses a line for me ethically.",
                "I can't support something I believe is morally wrong.",
                "My conscience tells me this isn't right.",
                "Some things are just wrong regardless of the consequences.",
            ]
            _vocab['elaborations_neu'] = [
                "Moral questions like this are genuinely difficult.",
                "I can see the ethical arguments on both sides.",
                "This is one of those situations where there's no clear right answer.",
                "My moral intuition is conflicted on this one.",
            ]

        # ── Intergroup/identity vocabulary ──
        elif domain_key in ('intergroup', 'identity', 'stereotype', 'prejudice') or \
                any(w in subject_lower for w in ('outgroup', 'ingroup', 'intergroup', 'discriminat',
                                                   'prejudice', 'stereotype', 'racial', 'ethnic')):
            _vocab['elaborations_pos'] = [
                "I try to treat everyone as individuals regardless of their group.",
                "People deserve to be judged on their own merits.",
                "I believe we have more in common than what divides us.",
                "Group membership shouldn't determine how you treat someone.",
            ]
            _vocab['elaborations_neg'] = [
                "I'd be lying if I said group differences didn't affect my thinking.",
                "It's hard to trust people when there's a clear group divide.",
                "I'm wary when dealing with people from different backgrounds.",
                "Past experiences with certain groups have shaped my views.",
            ]
            _vocab['elaborations_neu'] = [
                "I try not to let group membership affect my judgment but it's not easy.",
                "I'm aware of my biases and try to account for them.",
                "Group dynamics are complicated and I don't pretend to have it figured out.",
                "Sometimes group differences matter, sometimes they don't.",
            ]

        # ── Technology/AI vocabulary ──
        elif domain_key in ('ai_attitudes', 'privacy', 'automation', 'algorithm_aversion', 'social_media') or \
                any(w in subject_lower for w in ('artificial intelligence', 'algorithm', 'automat',
                                                   'privacy', 'data', 'technology', 'robot', 'machine')):
            _vocab['elaborations_pos'] = [
                "Technology generally makes things better when used responsibly.",
                "I'm optimistic about how tech can improve our lives.",
                "Innovation is important even if it's sometimes uncomfortable.",
                "I trust that the benefits of technology outweigh the risks.",
            ]
            _vocab['elaborations_neg'] = [
                "Technology is moving faster than our ability to control it.",
                "I worry about the unintended consequences of all this tech.",
                "Privacy erosion is a real concern that doesn't get enough attention.",
                "Just because we can build something doesn't mean we should.",
            ]
            _vocab['elaborations_neu'] = [
                "Technology is a tool and it depends on how people use it.",
                "I see both the promise and the risk in current tech trends.",
                "I'm neither a technophobe nor a tech evangelist.",
                "It's too early to know if this technology is net positive or negative.",
            ]

        # ── Environment/sustainability vocabulary ──
        elif domain_key in ('environmental', 'sustainability') or \
                any(w in subject_lower for w in ('climate', 'environment', 'sustain', 'pollution',
                                                   'green', 'carbon', 'ecology')):
            _vocab['elaborations_pos'] = [
                "We owe it to future generations to protect the environment.",
                "Climate change is real and we need to act now.",
                "Small individual choices add up when everyone participates.",
                "Sustainability isn't just idealism, it's practical.",
            ]
            _vocab['elaborations_neg'] = [
                "Environmental regulations often hurt working people.",
                "I think climate concerns are sometimes used to push agendas.",
                "Individual actions don't matter when corporations are the real polluters.",
                "Economic growth shouldn't be sacrificed for environmental goals.",
            ]
            _vocab['elaborations_neu'] = [
                "I try to be reasonable about environmental issues without going extreme.",
                "I care about the environment but I also care about practical realities.",
                "It's a balance between economic needs and environmental protection.",
                "I do what I can but I'm realistic about my impact.",
            ]

        # ── Workplace/organizational vocabulary ──
        elif domain_key in ('workplace', 'leadership', 'teamwork', 'motivation') or \
                any(w in subject_lower for w in ('work', 'job', 'boss', 'manager', 'team',
                                                   'employee', 'office', 'career', 'colleague')):
            _vocab['elaborations_pos'] = [
                "A good work environment makes a huge difference in quality of life.",
                "I feel valued and that motivates me to do my best.",
                "The people I work with make the job worthwhile.",
                "When management is competent, everything runs better.",
            ]
            _vocab['elaborations_neg'] = [
                "Work stress takes a real toll on your wellbeing.",
                "Management decisions often make no sense to the people doing the actual work.",
                "It's hard to stay motivated when you feel undervalued.",
                "Workplace issues affect your whole life, not just 9 to 5.",
            ]
            _vocab['elaborations_neu'] = [
                "Work is work, some days are good and some aren't.",
                "Most workplaces have the same basic problems.",
                "I try to keep a healthy perspective about my job.",
                "It's neither the best nor the worst work situation I've been in.",
            ]

        # ── Risk/decision vocabulary ──
        elif domain_key in ('risk_preference', 'loss_aversion', 'framing_effects', 'anchoring',
                            'sunk_cost', 'financial_decision', 'time_preference') or \
                any(w in subject_lower for w in ('risk', 'gamble', 'loss', 'gain', 'invest', 'chance',
                                                   'probability', 'certain', 'uncertain')):
            _vocab['elaborations_pos'] = [
                "Sometimes you have to take a chance to get ahead.",
                "Calculated risks have paid off for me in the past.",
                "I'm comfortable with uncertainty when the upside is worth it.",
                "Playing it too safe means missing out on opportunities.",
            ]
            _vocab['elaborations_neg'] = [
                "I've learned that protecting what you have is smarter than chasing more.",
                "Losses hurt more than gains feel good, that's just human nature.",
                "I prefer certainty because I've been burned by taking risks before.",
                "The downside risk outweighs any potential upside in my mind.",
            ]
            _vocab['elaborations_neu'] = [
                "It depends on the specific risk and what's at stake.",
                "I try to be rational about risks rather than going with emotion.",
                "Some risks are worth taking and some aren't, you have to evaluate each one.",
                "I'm neither a risk-taker nor completely risk-averse.",
            ]

        # ── Education/learning vocabulary ──
        elif domain_key in ('learning', 'online_learning', 'education', 'academic_motivation',
                            'teaching_effectiveness', 'student_engagement', 'assessment_feedback') or \
                any(w in subject_lower for w in ('learn', 'educat', 'school', 'teach', 'student',
                                                   'class', 'course', 'academic', 'grade', 'exam')):
            _vocab['elaborations_pos'] = [
                "Education is one of the most important things you can invest in.",
                "Good learning experiences stick with you for life.",
                "I value opportunities to grow and learn new things.",
                "When teaching is done well it can be transformative.",
            ]
            _vocab['elaborations_neg'] = [
                "The education system has real problems that need fixing.",
                "Not all learning experiences are created equal.",
                "I've had frustrating experiences that made me question the system.",
                "Education often fails to prepare people for the real world.",
            ]
            _vocab['elaborations_neu'] = [
                "Education is important but the system is imperfect.",
                "Some aspects of learning work well and others don't.",
                "I have mixed feelings about my educational experiences.",
                "There's always room for improvement in how we teach and learn.",
            ]

        # ── Clinical/mental health vocabulary ──
        elif domain_key in ('clinical', 'anxiety', 'depression', 'coping', 'therapy_attitudes',
                            'stress', 'mental_health', 'health_anxiety') or \
                any(w in subject_lower for w in ('anxiety', 'depress', 'therapy', 'counseling',
                                                   'mental health', 'coping', 'panic', 'phobia',
                                                   'trauma', 'ptsd', 'stress', 'burnout')):
            _vocab['elaborations_pos'] = [
                "I've found healthy ways to manage my mental health.",
                "Therapy has given me tools that actually help.",
                "Talking about mental health openly is important and I support that.",
                "I believe recovery is possible with the right support.",
                "My coping strategies have improved over time.",
            ]
            _vocab['elaborations_neg'] = [
                "Mental health struggles are exhausting and hard to explain to others.",
                "I wish there was less stigma around getting help.",
                "Some days are genuinely difficult and there's no easy fix.",
                "The mental health system isn't accessible enough for most people.",
                "Anxiety can be paralyzing even when you know it's irrational.",
            ]
            _vocab['elaborations_neu'] = [
                "Mental health is complicated and everyone's experience is different.",
                "Some approaches work for some people and not others.",
                "I try to be aware of my mental state without obsessing over it.",
                "There's no one-size-fits-all approach to mental wellness.",
            ]

        # ── Developmental/parenting vocabulary ──
        elif domain_key in ('developmental', 'parenting', 'childhood', 'aging',
                            'life_transitions', 'intergenerational') or \
                any(w in subject_lower for w in ('child', 'parent', 'aging', 'adolescen',
                                                   'develop', 'toddler', 'elderly', 'retirement',
                                                   'puberty', 'infant', 'teenager', 'generation')):
            _vocab['elaborations_pos'] = [
                "Watching growth and development is genuinely rewarding.",
                "Every stage of life has its own unique value.",
                "I think the early years really shape who we become.",
                "Supporting development at any age is meaningful work.",
                "Family relationships are central to how I think about this.",
            ]
            _vocab['elaborations_neg'] = [
                "These transitions can be really challenging to navigate.",
                "Not everyone has the support they need during critical stages.",
                "The pressure on parents today is unrealistic.",
                "Aging comes with real losses that people don't talk about enough.",
                "Growing up is harder now than it used to be.",
            ]
            _vocab['elaborations_neu'] = [
                "Development is a lifelong process with ups and downs.",
                "Every age brings its own challenges and rewards.",
                "I think there's no single right way to approach these stages.",
                "Some of these changes are natural and you just have to adapt.",
            ]

        # ── Personality psychology vocabulary ──
        elif domain_key in ('personality', 'big_five', 'narcissism', 'dark_triad',
                            'trait_assessment', 'self_concept') or \
                any(w in subject_lower for w in ('personality', 'extravert', 'introvert',
                                                   'narcissis', 'conscientiou', 'agreeable',
                                                   'neurotic', 'openness', 'trait', 'temperament')):
            _vocab['elaborations_pos'] = [
                "I'm pretty self-aware and comfortable with who I am.",
                "My personality is something I've come to appreciate over time.",
                "I think understanding yourself is the foundation for everything else.",
                "The way I naturally approach things tends to work out well.",
            ]
            _vocab['elaborations_neg'] = [
                "Some aspects of my personality make things harder.",
                "I struggle with certain tendencies even when I know they're not helpful.",
                "Personality isn't destiny but it can feel that way sometimes.",
                "I wish I could change some things about how I naturally react.",
            ]
            _vocab['elaborations_neu'] = [
                "Everyone has a mix of strengths and weaknesses in their personality.",
                "I've learned to work with my personality rather than against it.",
                "Some traits serve me well in certain situations and less so in others.",
                "Self-knowledge is useful but it doesn't solve everything.",
            ]

        # ── Sports psychology vocabulary ──
        elif domain_key in ('sports_psychology', 'athletic_motivation', 'team_dynamics',
                            'performance_anxiety', 'coach_athlete', 'fan_behavior') or \
                any(w in subject_lower for w in ('sport', 'athlet', 'team', 'coach', 'competition',
                                                   'perform', 'training', 'exercise', 'fitness',
                                                   'player', 'game', 'champion', 'winning')):
            _vocab['elaborations_pos'] = [
                "Competition brings out something in me that I really value.",
                "The discipline of training has shaped who I am as a person.",
                "Being part of a team is one of the most rewarding experiences.",
                "Physical activity is good for both body and mind.",
                "I thrive under the pressure of competitive situations.",
            ]
            _vocab['elaborations_neg'] = [
                "The pressure to perform can be overwhelming sometimes.",
                "Sports culture has some toxic elements that need addressing.",
                "Injuries and setbacks are mentally as tough as they are physical.",
                "Not everyone thrives in competitive environments.",
                "The expectations placed on athletes can be unrealistic.",
            ]
            _vocab['elaborations_neu'] = [
                "Sports and competition are important but they're not everything.",
                "I have a balanced view of what athletics can and can't do.",
                "Some competition is healthy but it can go too far.",
                "I appreciate sports without letting them define me.",
            ]

        # ── Legal psychology vocabulary ──
        elif domain_key in ('legal_psychology', 'jury_decision', 'witness_memory',
                            'procedural_justice', 'criminal_justice', 'legal_compliance') or \
                any(w in subject_lower for w in ('jury', 'court', 'legal', 'law', 'justice',
                                                   'witness', 'trial', 'crime', 'guilty',
                                                   'innocent', 'sentence', 'punish', 'judge')):
            _vocab['elaborations_pos'] = [
                "I believe in the justice system even though it's imperfect.",
                "Everyone deserves a fair process regardless of what they're accused of.",
                "The rule of law is what holds society together.",
                "I think most legal decisions try to be fair.",
            ]
            _vocab['elaborations_neg'] = [
                "The justice system is biased in ways that are hard to ignore.",
                "Too many innocent people get caught up in an unfair process.",
                "Legal outcomes depend too much on money and connections.",
                "The system punishes some people more than others for the same things.",
                "I've lost faith in the fairness of legal proceedings.",
            ]
            _vocab['elaborations_neu'] = [
                "The legal system is complicated and I don't have all the answers.",
                "Justice means different things to different people.",
                "Some cases are straightforward and some are genuinely difficult.",
                "I try to think about what's fair without being naive about reality.",
            ]

        # ── Food/nutrition vocabulary ──
        elif domain_key in ('food_psychology', 'eating_behavior', 'food_choice',
                            'nutrition_knowledge', 'body_image', 'diet_adherence') or \
                any(w in subject_lower for w in ('food', 'eating', 'diet', 'nutrition', 'meal',
                                                   'body image', 'weight', 'calorie', 'organic',
                                                   'healthy eating', 'snack', 'appetite')):
            _vocab['elaborations_pos'] = [
                "Food is one of life's great pleasures and I appreciate that.",
                "I've found an eating pattern that works well for me.",
                "Cooking and sharing food is important to my wellbeing.",
                "I feel good about the food choices I make most of the time.",
            ]
            _vocab['elaborations_neg'] = [
                "My relationship with food is complicated.",
                "Diet culture puts way too much pressure on people.",
                "I struggle with eating decisions more than I'd like to admit.",
                "Food marketing makes it hard to know what's actually healthy.",
                "Body image concerns affect how I think about eating.",
            ]
            _vocab['elaborations_neu'] = [
                "I try to eat well without being obsessive about it.",
                "Nutrition is important but I don't overthink every meal.",
                "Food choices are personal and I try not to judge others.",
                "Some days I eat better than others and that's fine.",
            ]

        # ── Communication/persuasion vocabulary ──
        elif domain_key in ('communication', 'persuasion', 'media_effects', 'interpersonal',
                            'public_opinion', 'narrative', 'misinformation') or \
                any(w in subject_lower for w in ('persuasi', 'media', 'communicat', 'message',
                                                   'narrative', 'misinformation', 'fake news',
                                                   'propaganda', 'rhetoric', 'debate')):
            _vocab['elaborations_pos'] = [
                "Good communication can genuinely change minds and build bridges.",
                "I appreciate when information is presented clearly and honestly.",
                "The power of a well-told story shouldn't be underestimated.",
                "Media can be a real force for good when done responsibly.",
            ]
            _vocab['elaborations_neg'] = [
                "There's too much manipulation in how information is presented.",
                "I'm skeptical of messages that seem designed to push me in a direction.",
                "Misinformation is a serious problem and it's getting worse.",
                "Media often distorts reality more than it reflects it.",
                "People are being manipulated without even realizing it.",
            ]
            _vocab['elaborations_neu'] = [
                "I try to evaluate messages critically regardless of the source.",
                "Communication is complicated and context always matters.",
                "Some media is helpful and some is harmful, you have to be discerning.",
                "I don't take any single source at face value.",
            ]

        # ── Relationships/attachment vocabulary ──
        elif domain_key in ('relationship', 'attachment', 'romantic_relationships',
                            'social_support', 'relationship_quality') or \
                any(w in subject_lower for w in ('relationship', 'attachment', 'romantic',
                                                   'partner', 'dating', 'marriage', 'divorce',
                                                   'intimacy', 'love', 'couple', 'breakup')):
            _vocab['elaborations_pos'] = [
                "Healthy relationships are one of the most important things in life.",
                "I've been lucky to have supportive people around me.",
                "Connection with others gives my life a lot of meaning.",
                "Trust and communication are the foundation of any good relationship.",
            ]
            _vocab['elaborations_neg'] = [
                "Relationships can cause as much pain as they do joy.",
                "I've had experiences that made me more guarded about letting people in.",
                "Not all relationships are healthy and it took me time to learn that.",
                "Past hurt makes it harder to be open with people.",
                "The expectations around relationships can be unrealistic.",
            ]
            _vocab['elaborations_neu'] = [
                "Relationships require effort and they're not always easy.",
                "Everyone has their own style of relating to others.",
                "I value relationships but also my independence.",
                "Some connections work out and some don't, that's just life.",
            ]

        # ── Cross-cultural vocabulary ──
        elif domain_key in ('cross_cultural', 'cultural_values', 'acculturation',
                            'cultural_identity', 'global_attitudes') or \
                any(w in subject_lower for w in ('cultur', 'cross-cultural', 'diversity',
                                                   'multicultural', 'immigrant', 'tradition',
                                                   'heritage', 'ethnicity', 'global')):
            _vocab['elaborations_pos'] = [
                "Diversity of perspectives makes communities stronger.",
                "My cultural background is something I'm proud of.",
                "Exposure to different cultures has broadened my worldview.",
                "I think cultural exchange benefits everyone involved.",
            ]
            _vocab['elaborations_neg'] = [
                "Cultural clashes can be difficult and draining.",
                "Not everyone's cultural experience is valued equally.",
                "Fitting into a new culture while keeping your own is hard.",
                "Some cultural practices are harmful even if they're traditional.",
            ]
            _vocab['elaborations_neu'] = [
                "Cultural differences are real but they don't have to divide us.",
                "I respect different traditions even when I don't fully understand them.",
                "Every culture has things worth keeping and things worth changing.",
                "Navigating cultural differences requires patience from everyone.",
            ]

        # ── Positive psychology vocabulary ──
        elif domain_key in ('positive_psychology', 'gratitude', 'resilience',
                            'flourishing', 'life_satisfaction') or \
                any(w in subject_lower for w in ('gratitude', 'resilience', 'flourish',
                                                   'happiness', 'wellbeing', 'life satisfaction',
                                                   'strengths', 'mindful', 'grateful', 'optimis')):
            _vocab['elaborations_pos'] = [
                "Focusing on what's going well has genuinely helped me.",
                "I believe everyone has strengths even if they don't see them.",
                "Gratitude is a practice that makes a real difference in my life.",
                "Finding meaning even in difficult times is something I value.",
            ]
            _vocab['elaborations_neg'] = [
                "Toxic positivity can be as harmful as negativity.",
                "Not everything can be fixed with a positive attitude.",
                "Sometimes life is genuinely hard and pretending otherwise doesn't help.",
                "The pressure to always be happy is exhausting.",
            ]
            _vocab['elaborations_neu'] = [
                "I try to be realistic while also appreciating the good things.",
                "Wellbeing is more complicated than just thinking positive thoughts.",
                "Everyone has good days and bad days and both are valid.",
                "I'm working on my outlook but it's a gradual process.",
            ]

        # ── Gender/sexuality vocabulary ──
        elif domain_key in ('gender_psychology', 'gender_roles', 'sexuality_attitudes',
                            'lgbtq_experience') or \
                any(w in subject_lower for w in ('gender', 'sexuality', 'lgbtq', 'sexual orientation',
                                                   'masculin', 'feminin', 'nonbinary', 'transgender',
                                                   'sex role', 'queer')):
            _vocab['elaborations_pos'] = [
                "I believe everyone deserves to live authentically.",
                "Progress on gender equality and LGBTQ rights is encouraging.",
                "People should be free to express who they are without fear.",
                "Diversity in gender and sexuality enriches our communities.",
            ]
            _vocab['elaborations_neg'] = [
                "Gender-based discrimination is still a major issue.",
                "Society still puts too many rigid expectations on people.",
                "LGBTQ people face real challenges that many don't understand.",
                "Stereotypes about gender and sexuality cause real harm.",
            ]
            _vocab['elaborations_neu'] = [
                "These are complex issues and I try to be thoughtful about them.",
                "I respect different perspectives on gender and sexuality.",
                "Society is changing in this area and I'm still processing my views.",
                "I think most people mean well even if they don't always get it right.",
            ]

        # ── Cognitive psychology vocabulary ──
        elif domain_key in ('cognitive', 'decision_making', 'memory', 'attention',
                            'reasoning', 'problem_solving', 'cognitive_bias', 'metacognition') or \
                any(w in subject_lower for w in ('cognitive', 'memory', 'attention', 'reasoning',
                                                   'problem solving', 'bias', 'thinking', 'judgment',
                                                   'perception', 'recall', 'recognition')):
            _vocab['elaborations_pos'] = [
                "I like to think things through carefully before deciding.",
                "I'm pretty confident in my ability to reason through problems.",
                "I try to be aware of my own biases when making judgments.",
                "Thoughtful decision-making usually leads to better outcomes.",
            ]
            _vocab['elaborations_neg'] = [
                "Sometimes I second-guess my own thinking too much.",
                "It's frustrating when I know my judgment was influenced by something irrelevant.",
                "My memory isn't always as reliable as I'd like it to be.",
                "Making decisions under pressure leads to mistakes.",
            ]
            _vocab['elaborations_neu'] = [
                "Everyone has cognitive strengths and weaknesses.",
                "I try to be systematic about decisions but I'm not always successful.",
                "Some choices require careful thought and some are better made quickly.",
                "I'm aware that my thinking can be biased even when I try to be objective.",
            ]

        # ── Neuroscience/cognitive load vocabulary ──
        elif domain_key in ('neuroeconomics', 'reward_processing', 'impulse_control',
                            'emotional_regulation', 'neural_decision', 'cognitive_load') or \
                any(w in subject_lower for w in ('brain', 'neural', 'impulse', 'self-control',
                                                   'cognitive load', 'mental effort', 'reward',
                                                   'willpower', 'concentration', 'focus')):
            _vocab['elaborations_pos'] = [
                "I'm pretty good at staying focused when it matters.",
                "I've learned to manage my impulses over time.",
                "Mental discipline is something I've worked on developing.",
                "I find it rewarding to push through difficult mental tasks.",
            ]
            _vocab['elaborations_neg'] = [
                "When I'm mentally exhausted my self-control goes out the window.",
                "It's hard to make good decisions when your brain is overloaded.",
                "I sometimes act on impulse and regret it later.",
                "Mental fatigue affects my judgment more than I'd like.",
            ]
            _vocab['elaborations_neu'] = [
                "Everyone has limits on their mental resources.",
                "Some tasks demand more mental effort than others.",
                "I do my best but I can't always be at peak focus.",
                "Self-control is easier in some situations than others.",
            ]

        # ── Human factors/UX vocabulary ──
        elif domain_key in ('human_factors', 'user_experience', 'interface_design',
                            'safety_behavior', 'workload', 'human_error') or \
                any(w in subject_lower for w in ('usability', 'interface', 'user experience',
                                                   'design', 'ergonomic', 'navigation', 'layout',
                                                   'prototype', 'accessibility', 'workflow')):
            _vocab['elaborations_pos'] = [
                "Good design makes a real difference in how people interact with things.",
                "I appreciate when things are intuitive and easy to use.",
                "A well-designed experience can actually be enjoyable.",
                "Attention to usability shows respect for the end user.",
            ]
            _vocab['elaborations_neg'] = [
                "Poor design is frustrating and makes simple tasks difficult.",
                "Too many things are designed without thinking about the actual user.",
                "Confusing interfaces lead to mistakes and that's not the user's fault.",
                "The people who design these things clearly don't use them.",
            ]
            _vocab['elaborations_neu'] = [
                "Design is a tradeoff between simplicity and functionality.",
                "Some designs work better for certain people than others.",
                "I notice both good and bad design choices in my daily life.",
                "Usability matters but it's not the only factor.",
            ]

        # ── Financial psychology vocabulary ──
        elif domain_key in ('financial_psychology', 'financial_literacy', 'investment_behavior',
                            'debt_attitudes', 'retirement_planning', 'financial_stress',
                            'saving_behavior') or \
                any(w in subject_lower for w in ('financial', 'debt', 'retirement', 'savings',
                                                   'budget', 'spending', 'income', 'wealth',
                                                   'investment', 'loan', 'credit', 'pension')):
            _vocab['elaborations_pos'] = [
                "I feel fairly confident about my financial decisions.",
                "Planning ahead financially has given me peace of mind.",
                "I've made some smart money choices that I'm glad about.",
                "Financial literacy is something everyone should have access to.",
            ]
            _vocab['elaborations_neg'] = [
                "Financial stress affects every other part of your life.",
                "The system is set up to make financial security hard for most people.",
                "I worry about money more than I'd like to admit.",
                "Financial mistakes can haunt you for years.",
                "Not everyone gets the same financial education or opportunities.",
            ]
            _vocab['elaborations_neu'] = [
                "Money matters are complicated and there's no one right answer.",
                "I try to be responsible with money without being obsessive.",
                "Some financial decisions have worked out and some haven't.",
                "I'm still figuring out the best approach to managing my finances.",
            ]

        # ── Gaming/entertainment vocabulary ──
        elif domain_key in ('gaming_psychology', 'esports', 'gambling', 'entertainment_media',
                            'streaming_behavior', 'virtual_reality') or \
                any(w in subject_lower for w in ('gaming', 'video game', 'gambl', 'esport',
                                                   'streaming', 'virtual reality', 'entertainment',
                                                   'play', 'casino', 'betting', 'VR')):
            _vocab['elaborations_pos'] = [
                "Gaming and entertainment are important parts of my life.",
                "Interactive media can be genuinely enriching experiences.",
                "I enjoy the social aspects of gaming and entertainment.",
                "Good entertainment provides both relaxation and stimulation.",
            ]
            _vocab['elaborations_neg'] = [
                "Too much screen time can become a real problem.",
                "The addictive design of some games and media concerns me.",
                "Gambling can destroy lives and needs better regulation.",
                "Entertainment consumption can become a way to avoid real life.",
            ]
            _vocab['elaborations_neu'] = [
                "Entertainment is fine in moderation but balance matters.",
                "I have mixed feelings about how much time people spend on screens.",
                "Gaming and media can be positive or negative depending on usage.",
                "I enjoy entertainment but try to be intentional about it.",
            ]

        # ── Social media vocabulary ──
        elif domain_key in ('social_media', 'social_media_use', 'online_identity',
                            'digital_communication', 'influencer_marketing', 'online_communities',
                            'social_comparison') or \
                any(w in subject_lower for w in ('social media', 'instagram', 'facebook', 'twitter',
                                                   'tiktok', 'influencer', 'online communit',
                                                   'follower', 'posting', 'scrolling', 'likes')):
            _vocab['elaborations_pos'] = [
                "Social media has helped me stay connected with people I care about.",
                "Online communities can provide real support and belonging.",
                "I've found valuable information and perspectives through social media.",
                "Digital connection has genuine benefits when used well.",
            ]
            _vocab['elaborations_neg'] = [
                "Social media comparison is harmful to mental health.",
                "The addictive design of these platforms is intentional and concerning.",
                "Online interactions can be toxic and draining.",
                "I worry about how social media is affecting our ability to connect in person.",
                "The curated reality of social media is damaging to self-image.",
            ]
            _vocab['elaborations_neu'] = [
                "Social media is a tool and how you use it matters.",
                "I have a complicated relationship with social platforms.",
                "Some online experiences are great and some are awful.",
                "I try to be intentional about my social media use.",
            ]

        # ── Decision science/nudge vocabulary ──
        elif domain_key in ('decision_science', 'choice_architecture', 'nudge',
                            'default_effects', 'information_overload', 'regret') or \
                any(w in subject_lower for w in ('nudge', 'default', 'choice architect',
                                                   'information overload', 'regret', 'opt-in',
                                                   'opt-out', 'decision fatigue')):
            _vocab['elaborations_pos'] = [
                "Making the right default option available is genuinely helpful.",
                "I appreciate when choices are structured to help good decisions.",
                "Reducing decision complexity is a real benefit.",
                "Smart choice design can improve outcomes for everyone.",
            ]
            _vocab['elaborations_neg'] = [
                "I don't like being nudged without my knowledge.",
                "Too many choices is paralyzing and leads to worse decisions.",
                "Regret about past decisions is hard to shake.",
                "Choice manipulation feels disrespectful even when well-intentioned.",
            ]
            _vocab['elaborations_neu'] = [
                "How choices are presented definitely influences decisions.",
                "I try to be aware of how options are framed before I choose.",
                "Some nudges are helpful and some cross a line.",
                "Decision-making is harder than most people give it credit for.",
            ]

        # ── Innovation/creativity vocabulary ──
        elif domain_key in ('innovation', 'creativity', 'entrepreneurship',
                            'idea_generation', 'creative_process') or \
                any(w in subject_lower for w in ('innovat', 'creativ', 'entrepreneur',
                                                   'invention', 'brainstorm', 'startup',
                                                   'original', 'imagination', 'design thinking')):
            _vocab['elaborations_pos'] = [
                "I value creative thinking and finding new solutions.",
                "Innovation is what drives real progress in any field.",
                "I enjoy the process of coming up with new ideas.",
                "Original thinking should be encouraged more than it is.",
            ]
            _vocab['elaborations_neg'] = [
                "Not every new idea is a good one and that's okay.",
                "Innovation for its own sake can be wasteful.",
                "The pressure to be creative all the time is exhausting.",
                "Too many innovative ideas fail because execution is hard.",
            ]
            _vocab['elaborations_neu'] = [
                "Creativity is important but so is refining existing approaches.",
                "Some innovations genuinely help while others are just hype.",
                "I appreciate creativity but also value practicality.",
                "Good ideas need both imagination and discipline to succeed.",
            ]

        # ── Risk/safety perception vocabulary ──
        elif domain_key in ('risk_perception', 'safety_attitudes', 'hazard_perception',
                            'disaster_preparedness', 'risk_communication') or \
                any(w in subject_lower for w in ('hazard', 'safety', 'disaster', 'emergency',
                                                   'preparedness', 'danger', 'risk perception',
                                                   'warning', 'accident', 'catastrophe')):
            _vocab['elaborations_pos'] = [
                "Being prepared gives me confidence to handle what comes.",
                "I think most risks can be managed with the right precautions.",
                "Safety awareness doesn't have to mean living in fear.",
                "I feel better knowing I've thought about potential risks.",
            ]
            _vocab['elaborations_neg'] = [
                "Some risks are genuinely terrifying and hard to prepare for.",
                "People underestimate dangers until it's too late.",
                "Safety systems fail more often than people realize.",
                "The consequences of ignoring risks can be devastating.",
            ]
            _vocab['elaborations_neu'] = [
                "Risk assessment requires balancing fear with rationality.",
                "Not all risks are equal and you have to prioritize.",
                "Some level of risk is unavoidable in daily life.",
                "I try to be cautious without being paralyzed by worry.",
            ]

        # ── Negotiation/bargaining vocabulary ──
        elif domain_key in ('negotiation', 'bargaining', 'economic_expectations') or \
                any(w in subject_lower for w in ('negotiat', 'bargain', 'compromise', 'deal',
                                                   'agreement', 'concession', 'offer', 'counteroffer')):
            _vocab['elaborations_pos'] = [
                "Finding a deal that works for everyone is satisfying.",
                "I'm a pretty good negotiator when I need to be.",
                "Compromise doesn't have to mean losing, it can mean winning differently.",
                "Good agreements require understanding what the other side needs.",
            ]
            _vocab['elaborations_neg'] = [
                "Negotiations can feel adversarial and stressful.",
                "I don't like being pressured into unfavorable terms.",
                "Power imbalances make fair negotiation almost impossible.",
                "Some people negotiate in bad faith and that ruins it for everyone.",
            ]
            _vocab['elaborations_neu'] = [
                "Negotiation outcomes depend on so many factors beyond your control.",
                "I approach negotiations pragmatically rather than emotionally.",
                "Some deals work out and some don't, that's the nature of it.",
                "Both sides usually have to give something to reach agreement.",
            ]

        # ── Trust & credibility vocabulary ──
        elif domain_key in ('institutional_trust', 'expert_credibility', 'source_credibility',
                            'science_trust', 'media_trust') or \
                any(w in subject_lower for w in ('credib', 'trustworth', 'reliable', 'expert',
                                                   'authority', 'institution', 'science trust',
                                                   'media trust', 'fake', 'authentic')):
            _vocab['elaborations_pos'] = [
                "I think experts generally have earned our trust.",
                "Credible sources are essential for making good decisions.",
                "I value institutions that have proven themselves reliable.",
                "Trust in science and expertise is worth maintaining.",
            ]
            _vocab['elaborations_neg'] = [
                "Institutions have broken the public's trust too many times.",
                "I'm skeptical of so-called experts who have their own agendas.",
                "Credibility is hard to rebuild once it's lost.",
                "Too much of what passes for expertise is actually opinion.",
            ]
            _vocab['elaborations_neu'] = [
                "Trust should be earned rather than automatically given.",
                "I evaluate credibility on a case-by-case basis.",
                "Some sources are more reliable than others and that's important to recognize.",
                "I try to verify information before accepting it.",
            ]

        # ── Workplace behavior vocabulary (remote work, burnout, etc.) ──
        elif domain_key in ('remote_work', 'workplace_diversity', 'burnout',
                            'career_development', 'workplace_conflict', 'organizational_justice',
                            'job_satisfaction', 'work_life_balance', 'employee_engagement') or \
                any(w in subject_lower for w in ('remote work', 'work from home', 'burnout',
                                                   'career', 'promotion', 'diversity',
                                                   'inclusion', 'work-life', 'overtime', 'salary')):
            _vocab['elaborations_pos'] = [
                "Having good working conditions makes all the difference.",
                "I appreciate workplaces that respect their employees.",
                "Career growth opportunities keep me motivated.",
                "Work can be genuinely fulfilling when the environment is right.",
            ]
            _vocab['elaborations_neg'] = [
                "Burnout is real and workplaces don't take it seriously enough.",
                "Workplace inequality affects morale and productivity.",
                "The expectation to always be available is destroying work-life balance.",
                "Career advancement shouldn't require sacrificing your wellbeing.",
            ]
            _vocab['elaborations_neu'] = [
                "Work is a big part of life and getting it right matters.",
                "Every workplace has its strengths and weaknesses.",
                "Finding the right balance between career and personal life is ongoing.",
                "Some aspects of work are rewarding and some are just necessary.",
            ]

        # ── AI ethics/alignment vocabulary ──
        elif domain_key in ('ai_alignment', 'ai_ethics', 'ai_safety', 'machine_values',
                            'ai_governance', 'ai_transparency', 'algorithmic_fairness') or \
                any(w in subject_lower for w in ('ai alignment', 'ai ethics', 'ai safety',
                                                   'algorithmic bias', 'ai fairness', 'autonomous',
                                                   'ai regulation', 'artificial general')):
            _vocab['elaborations_pos'] = [
                "I think responsible AI development is possible with the right safeguards.",
                "AI can be a tremendous force for good if we get the ethics right.",
                "I'm optimistic that we can develop AI that reflects human values.",
                "Transparency in AI systems builds trust and accountability.",
            ]
            _vocab['elaborations_neg'] = [
                "AI development is moving too fast for proper safety measures.",
                "Algorithmic bias is a serious problem that's being ignored.",
                "I don't trust that AI developers have everyone's interests in mind.",
                "The risks of uncontrolled AI development genuinely worry me.",
            ]
            _vocab['elaborations_neu'] = [
                "AI is a powerful tool that requires careful governance.",
                "The benefits and risks of AI depend on how it's developed and deployed.",
                "I think both AI optimists and pessimists have valid points.",
                "Regulation should keep pace with technology but often doesn't.",
            ]

        # ── Health disparities vocabulary ──
        elif domain_key in ('health_disparities', 'healthcare_access', 'health_equity',
                            'social_determinants', 'health_literacy', 'medical_mistrust') or \
                any(w in subject_lower for w in ('health disparit', 'healthcare access',
                                                   'health equity', 'social determinant',
                                                   'medical mistrust', 'uninsured')):
            _vocab['elaborations_pos'] = [
                "Access to healthcare should be a basic right for everyone.",
                "I'm encouraged by efforts to reduce health disparities.",
                "Health equity is worth fighting for in every community.",
                "Better access to information helps people make informed health choices.",
            ]
            _vocab['elaborations_neg'] = [
                "The gap in healthcare access between groups is unacceptable.",
                "People's health shouldn't depend on their income or zip code.",
                "Medical mistrust has real historical roots that can't be dismissed.",
                "The healthcare system fails the people who need it most.",
            ]
            _vocab['elaborations_neu'] = [
                "Health equity is a complex problem without easy solutions.",
                "Progress on healthcare access has been uneven at best.",
                "I recognize that different communities have very different health experiences.",
                "Systemic change is needed but it's hard to know where to start.",
            ]

        return _vocab

    # ══════════════════════════════════════════════════════════════════════
    # v1.1.0.4: CORPUS-BASED RESPONSE SYSTEM
    # ══════════════════════════════════════════════════════════════════════
    # Complete, self-contained survey responses that sound like real humans.
    # These do NOT use topic interpolation — they are ready-to-use as-is.
    # Organized by: domain × sentiment.
    #
    # Scientific grounding:
    # - Response patterns from Krosnick (1991) satisficing theory
    # - Length distributions from Schaefer & Dillman (1998)
    # - Content patterns from Hobbs & Green (2024) open-ended response theory
    # - Quality tiers from Curran (2016) careless responding research
    # ══════════════════════════════════════════════════════════════════════

    _CORPUS_RESPONSES: Dict[str, Dict[str, List[str]]] = {
        # ── POLITICAL / SOCIAL ISSUES ──────────────────────────────
        'political': {
            'positive': [
                "I think this is actually a step in the right direction. Not perfect, but it addresses some real problems that people have been ignoring for too long.",
                "My views on this have gotten stronger over the years. I used to be on the fence but watching how things played out convinced me that the people pushing for change are right.",
                "I'm cautiously optimistic. The last few years have shown that progress is possible even when things feel stuck. Not everyone agrees with me but I feel good about where this is heading.",
                "Honestly, I was skeptical at first but the more I learned about it the more it made sense. I talked to a few friends who had different perspectives and it actually reinforced my view.",
                "I support this and I think most people would too if they looked at the actual data instead of just the headlines. There's a lot of misinformation out there that muddies the waters.",
                "This resonates with my values. I grew up in a household where we talked about these issues a lot and I've always felt strongly that we need to do better as a society.",
                "The evidence is pretty clear to me. I've read enough about this topic to feel confident in my position, even though I know not everyone sees it the same way.",
                "I believe in this because I've seen the impact firsthand in my community. It's not abstract for me — real people I know have benefited.",
            ],
            'negative': [
                "I honestly don't think this is the right approach. The intentions might be good but the execution has been terrible and regular people end up worse off.",
                "This is one of those issues where I feel like the people in charge are completely out of touch with what average Americans actually deal with on a daily basis.",
                "I've watched this play out for years and it never delivers what's promised. At some point you have to look at the track record and admit it's not working.",
                "My problem isn't with the goal, it's with how they're going about it. There are better ways to address this that don't create so many new problems.",
                "I used to be more open-minded about this but after seeing the results I've become much more critical. The gap between the rhetoric and the reality is enormous.",
                "I think the people pushing this hardest are the ones least affected by the consequences. Easy to support something when you don't have to live with the downsides.",
                "This goes against some basic principles I hold. I've thought about it carefully and I keep coming back to the same concerns no matter how I look at it.",
                "The data doesn't support what advocates claim. I've looked into it and the actual outcomes are far less impressive than the talking points suggest.",
            ],
            'neutral': [
                "I can see valid points on both sides honestly. Some days I lean one way and some days the other. It's genuinely complicated and I don't trust anyone who pretends otherwise.",
                "I try to stay informed but the more I learn about this the less sure I am of my position. There are trade-offs no matter what approach you take.",
                "My views are somewhere in the middle. I don't buy the extreme arguments from either side. Reality is usually more nuanced than the debate makes it seem.",
                "I've gone back and forth on this enough times that I've kind of made peace with being undecided. Not everything needs a firm opinion and this is one of those things.",
                "Both sides make points I agree with and points I think are wrong. I wish there was more room for nuance in how we talk about these things.",
                "I know people I respect who feel very differently about this and they all have good reasons. That tells me it's not as simple as picking a side.",
            ],
        },
        # ── ECONOMIC GAMES / BEHAVIORAL ECONOMICS ─────────────────
        'economic_games': {
            'positive': [
                "I tried to be fair because that's just the kind of person I am. I know some people would keep more for themselves but I'd rather both of us walk away feeling okay about it.",
                "I gave a decent amount because I genuinely believe in treating people the way I'd want to be treated. Even if the other person is a stranger, they're still a person.",
                "I was generous because I think that's what makes sense in the long run. Being stingy might pay off once but it's not how I want to go through life.",
                "My decision was pretty easy actually. I thought about what I'd want someone to do if the roles were reversed and went with that. It felt right.",
                "I split it fairly because I don't think the money is worth feeling guilty about. I'd rather have less and feel good about my choice than keep everything and feel bad.",
                "Honestly, I tend to be trusting of other people. I know that can backfire sometimes but I'd rather err on the side of generosity than be the person who hoards everything.",
            ],
            'negative': [
                "I kept more for myself and I'm fine with that. It's a one-time interaction with someone I'll never see again, so the strategic choice seemed obvious.",
                "I played it safe because in my experience, people don't return favors as often as you'd hope. I've been burned before by being too generous with strangers.",
                "I was practical about it. The rational move is to keep more, and I don't think there's anything wrong with looking out for your own interests in this kind of situation.",
                "I didn't give much because I had no reason to trust that the other person would be fair to me. Without trust there's no basis for generosity.",
                "My approach was strategic. I thought about what would maximize my outcome and acted accordingly. I don't think that makes me a bad person, just realistic.",
                "I kept the larger share because the situation basically rewards self-interest. If the rules incentivize keeping more, that's what most rational people would do.",
            ],
            'neutral': [
                "I went with something in the middle because I couldn't decide whether to be generous or strategic. Splitting it roughly evenly seemed like the safe choice.",
                "Honestly I just went with my gut and didn't overthink it. I gave a moderate amount and moved on. I don't think there's a right answer here.",
                "I wasn't sure what the other person would do so I hedged my bets. Not too generous, not too stingy. Somewhere in the comfortable middle.",
                "I didn't have a strong strategy going in. I just picked a number that felt reasonable and didn't stress about whether it was optimal.",
                "My approach was pretty simple — I didn't want to feel like a jerk but I also didn't want to be a pushover. The middle ground felt right.",
            ],
        },
        # ── CONSUMER / PRODUCT / BRAND ────────────────────────────
        'consumer': {
            'positive': [
                "I've been using this for a while and it's been really solid. Not flashy or anything, just consistently good quality that I can rely on day to day.",
                "I'd recommend this to friends and family without hesitation. The value for what you pay is excellent and I haven't had any issues with it.",
                "My experience has been great. I did some research before choosing and I'm glad I went with this option. It does what it's supposed to do and does it well.",
                "I'm satisfied with my choice. It might not be the most exciting product out there but it works reliably and that's what matters most to me.",
                "The quality is noticeably better than alternatives I've tried. I switched to this about a year ago and haven't looked back since.",
                "I trust this brand because they've been consistent over time. Quality hasn't dropped and they stand behind what they sell which I respect.",
            ],
            'negative': [
                "I was pretty disappointed honestly. The marketing made it seem much better than it actually is, and the reality didn't match my expectations at all.",
                "I wouldn't buy this again. The quality was below what I'd expect for the price, and customer service wasn't helpful when I tried to address the issues.",
                "My experience was frustrating. It broke down faster than it should have and getting a replacement was a hassle. I expected better from a company this size.",
                "Not worth the money in my opinion. There are cheaper alternatives that work just as well or better. I feel like I was paying for the brand name more than anything.",
                "I regret this purchase. I should have read more reviews before buying because the problems I experienced seem to be pretty common based on what others say.",
                "The gap between what's advertised and what you actually get is huge. I've moved on to a competitor and the difference in quality is noticeable.",
            ],
            'neutral': [
                "It's fine. Not great, not terrible. It does the basic job but nothing about it really stands out or makes me feel strongly one way or another.",
                "I have mixed feelings about it. Some aspects are quite good but others are lacking. Overall it's an average experience, nothing to write home about.",
                "It meets my basic needs but I wouldn't say I'm passionate about it. If something better came along at a similar price I'd probably switch.",
                "I don't have strong feelings about this. It works, it's reasonably priced, and it does what I need. That's about all I can say.",
            ],
        },
        # ── HEALTH / MEDICAL / WELLBEING ──────────────────────────
        'health': {
            'positive': [
                "I think preventive care is really important and I wish more people took it seriously. When I started paying attention to my health the difference was significant.",
                "My experience with the healthcare system has been mostly positive. I've been lucky to have good providers who actually listen and take time to explain things.",
                "I'm a believer in evidence-based approaches. When the research clearly supports something, I think we should follow the science rather than relying on anecdotes.",
                "Taking care of your health is one of those things that pays off exponentially. Small changes in habits can lead to huge improvements in quality of life.",
                "I trust medical professionals because they've spent years training and studying. While no system is perfect, the expertise they bring matters a lot.",
            ],
            'negative': [
                "The healthcare system has a lot of problems that nobody seems willing to address. Costs keep going up while actual care quality doesn't improve proportionally.",
                "I've had some negative experiences with healthcare providers that made me skeptical. When doctors don't listen to patients, it erodes trust in the whole system.",
                "Access is the biggest issue in my opinion. It doesn't matter how good the treatments are if people can't afford them or can't get appointments.",
                "I'm frustrated with how the system works. Too much paperwork, too long waits, and too little actual face-to-face time with providers who know your history.",
                "The pharmaceutical industry has way too much influence over medical practice. It's hard to trust recommendations when there's so much money at stake.",
            ],
            'neutral': [
                "Healthcare is complicated and I don't think there are easy answers. Some things work well and others need major improvement.",
                "I try to take care of myself but I also recognize the system has limitations. You do what you can and hope for the best with the rest.",
                "My views on healthcare are mixed. I've had both good and bad experiences and I think most people's situations are similarly nuanced.",
            ],
        },
        # ── TECHNOLOGY / AI ───────────────────────────────────────
        'technology': {
            'positive': [
                "Technology has made my life easier in ways I couldn't have imagined ten years ago. I'm generally optimistic about where things are heading.",
                "I embrace new tools and innovations. Sure, there are risks, but the benefits far outweigh the downsides when technology is developed responsibly.",
                "The progress we've made is impressive and I think we're just scratching the surface of what's possible. I find it exciting rather than scary.",
                "I think technology is a net positive for society even though it creates new challenges. The key is thoughtful regulation, not resistance to change.",
            ],
            'negative': [
                "I'm concerned about how fast things are moving without enough thought about the consequences. Just because we can doesn't mean we should.",
                "Technology has created as many problems as it's solved in my view. Privacy erosion, screen addiction, misinformation — these are serious issues.",
                "I worry about the impact on jobs and communities. Not everyone benefits equally from technological change and the people left behind often get ignored.",
                "The companies driving these changes are motivated by profit, not public good. That makes me skeptical about whether the outcomes will actually benefit regular people.",
            ],
            'neutral': [
                "I use technology every day but I also see the downsides. It's a tool that can be used well or poorly, and right now I think we're doing both.",
                "I'm neither excited nor worried — just trying to adapt. Technology changes whether you want it to or not, so I focus on using it wisely.",
                "There are real benefits and real risks. I try not to be either a technophobe or a blind optimist, just someone who pays attention to both sides.",
            ],
        },
        # ── MORAL / ETHICAL JUDGMENT ──────────────────────────────
        'moral': {
            'positive': [
                "I think doing the right thing matters even when it's hard or inconvenient. My moral compass has been shaped by people I admire who stood up for their values.",
                "I believe people are fundamentally good and that most of us want to do the right thing. We just don't always agree on what the right thing is.",
                "Ethics aren't abstract to me — they're about how you treat actual people in everyday situations. And I try to treat people the way I'd want to be treated.",
            ],
            'negative': [
                "I think there's a lot of moral hypocrisy in the world. People talk about values but their actions tell a different story. That bothers me deeply.",
                "Some things are just wrong and I don't think you need to hear both sides to know that. Moral relativism has its limits and we've gone too far with it.",
                "The ethical failures I've witnessed have made me more cynical about people's intentions. Too often, doing the right thing takes a back seat to self-interest.",
            ],
            'neutral': [
                "Most ethical questions don't have clear answers in my experience. What seems obviously right in one context can be questionable in another.",
                "I try to be a good person but I also know that moral judgments are complicated. I'm suspicious of anyone who thinks they have all the answers.",
            ],
        },
        # ── ENVIRONMENT / SUSTAINABILITY ──────────────────────────
        'environment': {
            'positive': [
                "I care about the environment and try to make choices that reflect that. Not perfectly, but I believe individual actions add up when enough people participate.",
                "Climate change is real and we need to take it seriously. I've changed some of my habits because I think it matters, even if my contribution is small.",
                "I think we have a responsibility to future generations. Sustainability isn't just about us — it's about what kind of world we leave behind.",
            ],
            'negative': [
                "The environmental regulations are often more about politics than science. They hurt working people and businesses without delivering meaningful results.",
                "I'm skeptical of the doom and gloom narrative. The predictions keep changing and the proposed solutions often cause more economic harm than environmental benefit.",
                "Individual sacrifice doesn't matter when corporations and other countries aren't held to the same standard. It feels pointless and performative.",
            ],
            'neutral': [
                "I think the environment is important but so are people's livelihoods. Finding the right balance is harder than either side admits.",
                "I recycle and try to be mindful but I'm not sure how much difference it makes in the grand scheme. The problem is systemic, not individual.",
            ],
        },
        # ── EDUCATION / LEARNING ──────────────────────────────────
        'education': {
            'positive': [
                "Education changed my life and I believe it's one of the most important investments a society can make. Everyone deserves access to quality learning opportunities.",
                "I had some teachers who made a huge impact on my thinking and I'm grateful for that. Good education doesn't just teach facts — it teaches you how to think.",
                "Learning is a lifelong process and I try to keep growing. Whether it's formal education or self-directed study, the pursuit of knowledge is valuable.",
            ],
            'negative': [
                "The education system is outdated and doesn't prepare people for the real world. We're teaching to tests instead of developing critical thinking skills.",
                "College has become too expensive for what you get. The debt burden is crushing a whole generation and not everyone needs a four-year degree to succeed.",
                "I was failed by the school system in several ways. Too many students fall through the cracks because the system isn't designed to accommodate different learning styles.",
            ],
            'neutral': [
                "Education has its strengths and weaknesses. Some people thrive in traditional settings while others learn better through different paths. One size doesn't fit all.",
                "I have mixed feelings about the current system. It works for some people but there's clearly room for improvement in many areas.",
            ],
        },
        # ── SOCIAL / INTERPERSONAL / IDENTITY ─────────────────────
        'social': {
            'positive': [
                "I believe people are generally good-natured and most social problems come from misunderstanding rather than malice. We have more in common than we think.",
                "Community matters to me. I've seen what happens when people come together around shared goals and it gives me hope for how we handle disagreements.",
                "I think empathy is the most important quality a person can have. If more people tried to genuinely understand each other, a lot of problems would diminish.",
            ],
            'negative': [
                "People are more divided than ever and I don't see it getting better anytime soon. Social media has made it worse by creating echo chambers.",
                "Trust in institutions and in each other has eroded badly. I'm not sure how we rebuild that when people can't even agree on basic facts anymore.",
                "There's too much tribalism in how we interact with each other. People define themselves by their group and treat outsiders with suspicion. It's toxic.",
            ],
            'neutral': [
                "Human nature is complicated. We're capable of incredible generosity and terrible cruelty, sometimes in the same person. I try not to idealize or demonize people.",
                "Social dynamics are always changing and I think it's hard to make sweeping generalizations. Some things are better, some are worse, and most are just different.",
            ],
        },
        # ── WORKPLACE / ORGANIZATIONAL ────────────────────────────
        'workplace': {
            'positive': [
                "I enjoy my work and feel like what I do matters. Not every day is perfect but overall I'm in a good place professionally and that makes a big difference.",
                "A good workplace culture makes all the difference. When you feel valued and supported, you naturally want to do your best work.",
                "I've been fortunate to work with good people who push me to grow. The collaborative aspects of my work are what keep me motivated.",
            ],
            'negative': [
                "The modern workplace expects too much for too little. Burnout is everywhere and companies talk about wellness while making things worse with unrealistic demands.",
                "Management doesn't understand what employees actually deal with. There's a disconnect between leadership's vision and the reality on the ground.",
                "I'm frustrated with workplace politics and the way merit often takes a back seat to other factors. It's demoralizing when hard work isn't recognized.",
            ],
            'neutral': [
                "Work is work. Some days are good, some aren't. I try to focus on what I can control and not get too caught up in the bigger picture.",
                "My feelings about my workplace are mixed. There are things I appreciate and things that could be much better. It's a trade-off like everything else.",
            ],
        },
        # ── GENERAL / FALLBACK ────────────────────────────────────
        'general': {
            'positive': [
                "I feel positive about this and have for a while. It aligns with my personal experience and values, and the more I learn about it the more confident I become.",
                "My overall take is favorable. There are always things that could be improved but on balance I think the positives significantly outweigh the negatives.",
                "I support this based on what I've seen and experienced. Not just abstract reasoning but actual real-world outcomes that I've witnessed firsthand.",
                "I've thought about this carefully and I keep arriving at the same positive conclusion. It just makes sense to me on multiple levels.",
                "I came into this with an open mind and ended up feeling good about it. The evidence and my personal experience both point in the same direction.",
                "Honestly this is something I care about and I think it deserves more support than it gets. People who dismiss it haven't really engaged with it seriously.",
                "My experience has been overwhelmingly positive. I know that's not universal but I can only speak to what I've personally seen and encountered.",
                "I believe in this and I'm not shy about saying so. It's grounded in solid reasoning and backed up by outcomes I've observed directly.",
            ],
            'negative': [
                "I have serious concerns about this that haven't been adequately addressed. The more I look into it, the more problems I see with how it's being handled.",
                "My experience has been negative and talking to others confirms it's not just me. There are systemic issues that need attention before this can work properly.",
                "I was willing to give this a fair chance but the results speak for themselves. When something consistently underperforms, at some point you have to call it what it is.",
                "The problems here are deeper than people want to admit. Surface-level fixes won't work because the fundamental approach has significant flaws.",
                "I'm opposed to this and I have concrete reasons based on what I've personally witnessed. It's not just abstract disagreement — I've seen the real-world impact.",
                "This has been handled poorly from the start and the consequences are becoming harder to ignore. I wish I felt differently but the evidence keeps mounting.",
                "My frustration with this has grown over time. Initially I was neutral but repeated disappointments have pushed me firmly into the critical camp.",
                "I don't think the people advocating for this fully understand the downsides. They're focused on the theory while ignoring what actually happens in practice.",
            ],
            'neutral': [
                "I genuinely don't have a strong opinion on this. I've considered different perspectives and I can see merit in arguments on both sides.",
                "My views are mixed and I'm okay with that. Not everything requires a definitive position and I think this is one of those genuinely ambiguous situations.",
                "I go back and forth depending on which aspect I'm thinking about. The overall picture is complicated enough that I resist simplifying it.",
                "I try to be thoughtful about this rather than reactive. There are legitimate concerns and legitimate benefits, and they don't cancel each other out neatly.",
                "I know people who feel very strongly on both sides and I respect their views. My own position is somewhere in the messy middle, which I've made peace with.",
                "This is one of those topics where I find myself agreeing with different people at different times. I don't think that's wishy-washy — I think it's realistic.",
            ],
        },
    }

    def _get_corpus_response(self, domain_key: str, sentiment: str,
                             rng: random.Random) -> str:
        """Select a complete, natural-sounding response from the corpus.

        v1.1.0.4: These responses are COMPLETE sentences written to sound
        like real survey respondents.  They don't need topic interpolation
        because they're self-contained natural language.  Used when topic
        extraction fails or returns unintelligible text.

        Tracks used responses to avoid giving the same corpus response to
        multiple participants.  When the pool is exhausted, resets and
        starts reusing with personalization to differentiate.
        """
        _sent_key = 'positive' if sentiment in ('very_positive', 'positive') else \
                    ('negative' if sentiment in ('very_negative', 'negative') else 'neutral')

        # Map domain_key to corpus key
        _domain_map: Dict[str, str] = {
            'dictator_game': 'economic_games', 'trust_game': 'economic_games',
            'ultimatum_game': 'economic_games', 'public_goods': 'economic_games',
            'prisoners_dilemma': 'economic_games', 'cooperation': 'economic_games',
            'fairness': 'economic_games',
            'polarization': 'political', 'partisanship': 'political',
            'voting': 'political', 'policy_attitudes': 'political',
            'intergroup': 'political',
            'product_evaluation': 'consumer', 'brand_loyalty': 'consumer',
            'purchase_intent': 'consumer', 'advertising': 'consumer',
            'medical_decision': 'health', 'wellbeing': 'health',
            'vaccination': 'health', 'stress': 'health',
            'moral_judgment': 'moral', 'ethics': 'moral',
            'ai_attitudes': 'technology', 'automation': 'technology',
            'climate': 'environment', 'sustainability': 'environment',
            'learning': 'education',
            'identity': 'social', 'norms': 'social', 'conformity': 'social',
            'job_satisfaction': 'workplace', 'leadership': 'workplace',
            'motivation': 'workplace',
            'risk_preference': 'general', 'risk': 'general',
            'persuasion': 'general', 'credibility': 'general',
            'relationship': 'social', 'attachment': 'social',
            'financial_decision': 'general', 'investment': 'general',
        }
        _corpus_key = _domain_map.get(domain_key, domain_key)
        if _corpus_key not in self._CORPUS_RESPONSES:
            _corpus_key = 'general'

        _pool = self._CORPUS_RESPONSES[_corpus_key].get(_sent_key, [])
        if not _pool:
            _pool = self._CORPUS_RESPONSES['general'].get(_sent_key, [])
        if not _pool:
            return ""

        # Track used indices to avoid giving same response to different participants
        _track_key = f"{_corpus_key}|{_sent_key}"
        if _track_key not in self._used_corpus_indices:
            self._used_corpus_indices[_track_key] = set()
        _used = self._used_corpus_indices[_track_key]

        # Find unused responses
        _available = [i for i in range(len(_pool)) if i not in _used]
        if not _available:
            # All used — reset tracker (personalization will differentiate)
            _used.clear()
            _available = list(range(len(_pool)))

        _idx = rng.choice(_available)
        _used.add(_idx)
        return _pool[_idx]

    def _personalize_corpus_response(self, response: str, traits: Dict[str, Any],
                                      behavioral_profile: Dict[str, Any],
                                      rng: random.Random) -> str:
        """Transform a corpus response to sound like a specific persona.

        v1.1.0.4: PERSONA-DRIVEN PERSONALIZATION ENGINE.  Takes a complete
        corpus response and applies transformations based on persona traits
        to make each simulated participant sound unique:

        1. Verbosity modulation — truncate for terse participants, extend for verbose
        2. Formality adjustment — add contractions/slang for casual, remove for formal
        3. Sentence restructuring — vary the sentence order and structure
        4. Personal detail injection — add "my [relation]" or "I remember when" fragments
        5. Imperfection injection — typos, run-ons, missing punctuation for low-engagement

        Based on:
        - Krosnick (1991): Satisficing produces shorter, less detailed responses
        - LIWC profiles: Different demographics use different pronoun/verb patterns
        - Schmidt (2019): Response quality varies with engagement and device type
        """
        if not response:
            return response

        _verbosity = traits.get('verbosity', 0.5)
        _formality = traits.get('formality', 0.5)
        _engagement = traits.get('attention', 0.5)
        _extremity = traits.get('extremity', 0.4)

        _result = response

        # ── 1. Verbosity modulation ──────────────────────────────
        _sentences = re.split(r'(?<=[.!?])\s+', _result.strip())
        if _verbosity < 0.3 and len(_sentences) > 2:
            # Terse persona: keep only 1-2 sentences
            _keep = rng.choice([1, 2])
            _result = ' '.join(_sentences[:_keep])
        elif _verbosity < 0.4 and len(_sentences) > 3:
            _keep = rng.choice([2, 3])
            _result = ' '.join(_sentences[:_keep])

        # ── 2. Formality adjustment ──────────────────────────────
        if _formality < 0.3:
            # Casual: add contractions, lowercase openers, remove periods
            _casual_subs = [
                (r"\bI am\b", "I'm"), (r"\bdo not\b", "don't"),
                (r"\bdoes not\b", "doesn't"), (r"\bcannot\b", "can't"),
                (r"\bwould not\b", "wouldn't"), (r"\bit is\b", "it's"),
                (r"\bthat is\b", "that's"), (r"\bI have\b", "I've"),
                (r"\bI would\b", "I'd"), (r"\bthey are\b", "they're"),
                (r"\bwill not\b", "won't"), (r"\bshould not\b", "shouldn't"),
            ]
            for _pat, _rep in _casual_subs:
                if rng.random() < 0.7:
                    _result = re.sub(_pat, _rep, _result, flags=re.IGNORECASE)
            # Sometimes drop final period
            if rng.random() < 0.3 and _result.endswith('.'):
                _result = _result[:-1]
            # Casual opener
            if rng.random() < 0.25:
                _openers = ["idk ", "tbh ", "lol ", "ok so ", "i mean "]
                _result = rng.choice(_openers) + _result[0].lower() + _result[1:]
        elif _formality > 0.7:
            # Formal: expand contractions
            _formal_subs = [
                (r"\bdon't\b", "do not"), (r"\bcan't\b", "cannot"),
                (r"\bwon't\b", "will not"), (r"\bI'm\b", "I am"),
                (r"\bI've\b", "I have"), (r"\bit's\b", "it is"),
                (r"\bthat's\b", "that is"), (r"\bI'd\b", "I would"),
                (r"\bthey're\b", "they are"), (r"\bwouldn't\b", "would not"),
            ]
            for _pat, _rep in _formal_subs:
                if rng.random() < 0.5:
                    _result = re.sub(_pat, _rep, _result, flags=re.IGNORECASE)

        # ── 3. Sentence restructuring — swap sentence order ──────
        _re_sentences = re.split(r'(?<=[.!?])\s+', _result.strip())
        if len(_re_sentences) >= 3 and rng.random() < 0.3:
            # Swap two random sentences
            _i, _j = rng.sample(range(len(_re_sentences)), 2)
            _re_sentences[_i], _re_sentences[_j] = _re_sentences[_j], _re_sentences[_i]
            _result = ' '.join(_re_sentences)

        # ── 4. Personal detail injection ─────────────────────────
        if _verbosity > 0.5 and rng.random() < 0.25:
            _details = [
                " My partner thinks the same way about this.",
                " I was just talking about this with a friend last week.",
                " I noticed this more after moving to a new city.",
                " Growing up, my parents had strong views on this.",
                " I've been thinking about this more lately.",
                " Someone at work brought this up recently and it got me thinking.",
            ]
            _result = _result.rstrip('.') + '.' + rng.choice(_details)

        # ── 5. Imperfection injection for low engagement ─────────
        if _engagement < 0.4 and rng.random() < 0.35:
            _imperfections = [
                # Run-on sentence
                lambda t: t.replace('. ', ', ', 1) if '. ' in t else t,
                # Missing apostrophe
                lambda t: t.replace("don't", "dont", 1) if "don't" in t else t,
                lambda t: t.replace("I'm", "Im", 1) if "I'm" in t else t,
                lambda t: t.replace("can't", "cant", 1) if "can't" in t else t,
                # Lowercase first letter
                lambda t: t[0].lower() + t[1:] if t else t,
                # Double space
                lambda t: t.replace(' ', '  ', 1) if rng.random() < 0.3 else t,
            ]
            _imperfection = rng.choice(_imperfections)
            _result = _imperfection(_result)

        # ── 6. Extremity amplification ───────────────────────────
        if _extremity > 0.7 and rng.random() < 0.3:
            _intensifiers = [
                ("I think", rng.choice(["I strongly believe", "I firmly think", "I'm convinced"])),
                ("I feel", rng.choice(["I deeply feel", "I really feel", "I'm certain"])),
                ("good", rng.choice(["excellent", "fantastic", "outstanding"])),
                ("bad", rng.choice(["terrible", "awful", "disastrous"])),
                ("important", rng.choice(["crucial", "absolutely essential", "vitally important"])),
            ]
            for _old, _new in _intensifiers:
                if _old in _result.lower():
                    _result = re.sub(re.escape(_old), _new, _result, count=1, flags=re.IGNORECASE)
                    break

        return _result

    # ===================================================================
    # v1.0.9.2: ADAPTIVE COMPOSITIONAL RESPONSE BUILDER
    # ===================================================================
    # Replaces fixed-template selection with a multi-component builder
    # that constructs natural-sounding responses from atomic parts.
    # Components: [opener?] + [position] + [reasoning?] + [elaboration?] + [qualifier?]
    # Each component is independently selected based on sentiment, intent,
    # topic, and persona traits.  This produces dramatically more natural
    # and varied text than selecting from a fixed template bank.
    # ===================================================================

    # --- Natural language fillers and hedges by persona style --------
    _NL_FILLERS_CASUAL = [
        "honestly ", "I mean ", "like ", "tbh ", "look ", "idk but ",
        "so basically ", "for real ", "ngl ", "okay so ",
    ]
    _NL_FILLERS_MODERATE = [
        "honestly ", "I think ", "in my view ", "from what I can tell ",
        "speaking personally ", "I have to say ", "the way I see it ",
    ]
    _NL_FILLERS_FORMAL = [
        "in my considered opinion ", "from my perspective ",
        "upon reflection ", "after careful thought ",
        "I would argue that ", "it seems to me that ",
    ]
    _HEDGES = [
        "I guess", "probably", "maybe", "kind of", "sort of",
        "I suppose", "in a way", "more or less",
    ]
    _INTENSIFIERS = [
        "really", "absolutely", "definitely", "strongly",
        "completely", "deeply", "truly", "without a doubt",
    ]
    _CONNECTORS_CASUAL = [
        ". And ", ". Plus ", ". Also ", ", and ", " — ",
        ". Like ", ". I think ", ". But yeah ",
    ]
    _CONNECTORS_MODERATE = [
        ". Additionally, ", ". Moreover, ", ". That said, ",
        ". On top of that, ", ". I also think ", ". At the same time, ",
    ]
    _SELF_CORRECTIONS = [
        " — well, actually ", " — or rather, ", " — I mean ",
        ", well, ", ", or at least that's how I see it",
    ]

    # ── Response archetype weights by persona trait profile ──────────
    # Each archetype represents a structurally DIFFERENT way real humans
    # answer open-ended survey questions.  Selection is weighted by
    # formality, engagement, extremity, and verbosity so the same study
    # produces a realistic mix of response styles.
    #
    # Archetypes (v1.0.9.8):
    #   direct_answer    – Position only, blunt.  "I like it, plain and simple"
    #   story_first      – Anecdote → conclusion
    #   reasoning_first  – Evidence → position
    #   rhetorical       – Rhetorical question → answer
    #   concession       – "Sure X has merits, but honestly…"
    #   stream           – Connected fragments, conversational flow
    #   list_style       – Multiple short points
    #   emotional_burst  – Feeling erupts first, then explanation
    _ARCHETYPE_WEIGHTS = {
        # (formality_tier, engagement_tier) → {archetype: weight}
        # formality_tier: 'casual' (<0.35), 'moderate' (0.35-0.7), 'formal' (>0.7)
        # engagement_tier: 'low' (<0.4), 'mid' (0.4-0.7), 'high' (>0.7)
        ('casual', 'low'):  {'direct_answer': 40, 'stream': 30, 'emotional_burst': 15, 'rhetorical': 10, 'list_style': 5},
        ('casual', 'mid'):  {'stream': 25, 'direct_answer': 20, 'story_first': 15, 'emotional_burst': 15, 'rhetorical': 10, 'list_style': 10, 'concession': 5},
        ('casual', 'high'): {'story_first': 25, 'stream': 20, 'emotional_burst': 15, 'reasoning_first': 10, 'list_style': 10, 'rhetorical': 10, 'concession': 10},
        ('moderate', 'low'):  {'direct_answer': 50, 'stream': 20, 'list_style': 15, 'rhetorical': 10, 'concession': 5},
        ('moderate', 'mid'):  {'reasoning_first': 20, 'direct_answer': 20, 'concession': 15, 'story_first': 15, 'list_style': 10, 'rhetorical': 10, 'stream': 10},
        ('moderate', 'high'): {'reasoning_first': 25, 'story_first': 20, 'concession': 15, 'list_style': 10, 'emotional_burst': 10, 'rhetorical': 10, 'direct_answer': 10},
        ('formal', 'low'):  {'direct_answer': 45, 'reasoning_first': 25, 'concession': 15, 'list_style': 15},
        ('formal', 'mid'):  {'reasoning_first': 30, 'concession': 25, 'direct_answer': 15, 'list_style': 15, 'story_first': 15},
        ('formal', 'high'): {'reasoning_first': 30, 'concession': 20, 'story_first': 20, 'list_style': 15, 'rhetorical': 10, 'direct_answer': 5},
    }

    def _select_archetype(self, formality: float, engagement: float, rng: random.Random) -> str:
        """Select a response archetype weighted by persona traits."""
        f_tier = 'casual' if formality < 0.35 else ('formal' if formality > 0.7 else 'moderate')
        e_tier = 'low' if engagement < 0.4 else ('high' if engagement > 0.7 else 'mid')
        weights = self._ARCHETYPE_WEIGHTS.get((f_tier, e_tier), self._ARCHETYPE_WEIGHTS[('moderate', 'mid')])
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

    def _build_adaptive_response(
        self,
        topic: str,
        sentiment: str,
        question_intent: str,
        question_text: str,
        question_context: str,
        condition: str,
        domain_key: str,
        behavioral_profile: Dict[str, Any],
        rng: random.Random,
    ) -> str:
        """Build a human-like response using structurally diverse archetypes.

        v1.0.9.8: STRUCTURAL DIVERSITY ENGINE.  Instead of the rigid
        opener→position→reasoning→elaboration→qualifier formula, this now
        selects from 8 structurally distinct response archetypes weighted by
        persona traits.  Real survey respondents answer in wildly different
        structural patterns — some blurt out a position, some tell a story,
        some list reasons, some start with a rhetorical question.  This
        engine reproduces that structural diversity.

        v1.1.0.4: TOPIC INTELLIGIBILITY GATE.  Before interpolating any
        topic into templates, validates that the topic is natural language
        (not a variable name like "respond_affect_score").  When the topic
        is unintelligible, uses corpus-based complete responses instead of
        template interpolation.  This is the #1 fix for gibberish output.
        """
        _traits = behavioral_profile.get('trait_profile', {}) if behavioral_profile else {}
        _verbosity = _traits.get('verbosity', 0.5)
        _formality = _traits.get('formality', 0.5)
        _engagement = _traits.get('attention', 0.5)
        _extremity = _traits.get('extremity', 0.4)
        _sd = _traits.get('social_desirability', 0.3)
        _intensity = behavioral_profile.get('intensity', 0.5) if behavioral_profile else 0.5

        # --- v1.1.0.4: TOPIC INTELLIGIBILITY CHECK -------------------------
        # If the topic is a variable name or technical jargon, DON'T try to
        # interpolate it into templates.  Instead, use a complete corpus
        # response that stands on its own without needing a topic variable.
        _topic_usable = self._is_topic_intelligible(topic) if topic else False
        if not _topic_usable:
            # Try to infer a natural-language topic from domain/condition
            _inferred = self._infer_topic_from_context(
                domain_key, condition, question_text, question_context,
            )
            if _inferred and self._is_topic_intelligible(_inferred):
                topic = _inferred
                _topic_usable = True
            else:
                topic = "this"

        # --- v1.1.0.3: Ultra-short handler for very low engagement ------
        # ~15-20% of real survey responses are 1-5 words. Disengaged personas
        # should produce these directly without running the full archetype
        # pipeline (which always produces multi-sentence text).
        _is_straight_lined = behavioral_profile.get('straight_lined', False) if behavioral_profile else False
        if (_engagement < 0.2 or _is_straight_lined) and rng.random() < 0.6:
            return self._build_ultra_short(topic, sentiment, _formality, rng)
        if _engagement < 0.35 and _verbosity < 0.3 and rng.random() < 0.35:
            return self._build_ultra_short(topic, sentiment, _formality, rng)

        # --- v1.1.0.4: CORPUS-BASED PATH for unintelligible topics ------
        # When topic is just "this", use complete pre-written responses
        # instead of templates that would produce "I believe in this" gibberish.
        if not _topic_usable:
            _corpus_resp = self._get_corpus_response(domain_key, sentiment, rng)
            if _corpus_resp:
                # Apply FULL personalization pipeline to make each persona unique
                _corpus_resp = self._personalize_corpus_response(
                    _corpus_resp, _traits, behavioral_profile or {}, rng,
                )
                return _corpus_resp

        # --- Extract question words to mirror in response ---------------
        _q_mirror_words: List[str] = []
        _q_source = question_context or question_text or ""
        if _q_source:
            _qw = re.findall(r'\b[a-zA-Z]{4,}\b', _q_source.lower())
            _mirror_stop = {
                'what', 'your', 'about', 'please', 'describe', 'explain',
                'think', 'feel', 'would', 'could', 'should', 'question',
                'answer', 'response', 'study', 'survey', 'participant',
                'have', 'does', 'this', 'that', 'with', 'from', 'more',
                'most', 'very', 'really', 'some', 'much', 'many', 'just',
            }
            _q_mirror_words = [w for w in _qw if w not in _mirror_stop][:4]

        # --- Determine formality tier -----------------------------------
        if _formality > 0.7:
            _connector_bank = self._CONNECTORS_MODERATE
        elif _formality < 0.3:
            _connector_bank = self._CONNECTORS_CASUAL
        else:
            _connector_bank = self._CONNECTORS_MODERATE

        # --- Shared component builders ----------------------------------
        _pos_args = (topic, sentiment, question_intent, _q_mirror_words,
                     _extremity, _intensity, condition, rng)
        _reas_args = (topic, sentiment, question_intent, domain_key,
                      _q_mirror_words, condition, rng)
        _elab_args = (topic, sentiment, domain_key, _q_mirror_words, rng)

        # --- Select archetype -------------------------------------------
        _arch = self._select_archetype(_formality, _engagement, rng)

        # === ARCHETYPE ASSEMBLY =========================================
        if _arch == 'direct_answer':
            # Blunt, no preamble.  "I like it, simple as that."
            _position = self._build_position(*_pos_args)
            _response = _position
            # 20% chance of a tiny coda
            if rng.random() < 0.2:
                _codas = [", simple as that", ", end of story", ", nothing more to say",
                          ", that's it", ", pretty straightforward", ""]
                _response += rng.choice(_codas)

        elif _arch == 'story_first':
            # Anecdote → conclusion.  "Last time I ... so now I think ..."
            _anecdote = self._build_anecdote(topic, sentiment, domain_key, condition, rng)
            _position = self._build_position(*_pos_args)
            _bridges = [" So yeah, ", " That's why ", " Which is why ", " And that's basically why ",
                        " Long story short, ", " Point is, "]
            _response = _anecdote + rng.choice(_bridges) + _position

        elif _arch == 'reasoning_first':
            # Evidence → position.  "The way I see it, X does Y. That's why I ..."
            _reasoning = self._build_reasoning(*_reas_args)
            _position = self._build_position(*_pos_args)
            _bridges = [". That's why ", ". So overall ", ". Bottom line, ", ". Which means ",
                        ". And so "]
            _response = _reasoning[0].upper() + _reasoning[1:] + rng.choice(_bridges) + _position

        elif _arch == 'rhetorical':
            # Rhetorical question → answer.  "Do I think X works? Not really."
            _rq = self._build_rhetorical_question(topic, question_intent, rng)
            _position = self._build_position(*_pos_args)
            _response = _rq + " " + _position[0].upper() + _position[1:]

        elif _arch == 'concession':
            # "Sure X has its merits, but ..."
            _conc = self._build_concession(topic, sentiment, rng)
            _position = self._build_position(*_pos_args)
            _response = _conc + " " + _position

        elif _arch == 'stream':
            # Connected fragments, conversational.
            # "X is just... idk, it feels like they don't care, you know?"
            _position = self._build_position(*_pos_args)
            _frag = self._build_stream_fragment(topic, sentiment, domain_key, rng)
            _trail = rng.choice(["", " you know?", " idk", " but yeah", " or whatever",
                                 " I guess", " so yeah"])
            _response = _position + ", " + _frag + _trail

        elif _arch == 'list_style':
            # Multiple short points.  "a few things — first X, also Y, and Z"
            _points = self._build_list_points(topic, sentiment, domain_key, condition, rng)
            _intros = ["a few things: ", "basically: ", "my thoughts — ",
                       "couple things. ", "ok so ", ""]
            _response = rng.choice(_intros) + _points

        elif _arch == 'emotional_burst':
            # Feeling erupts first.  "This genuinely makes me angry. X is ..."
            _burst = self._build_emotional_burst(topic, sentiment, _extremity, rng)
            _reasoning = self._build_reasoning(*_reas_args)
            _response = _burst + ". " + _reasoning[0].upper() + _reasoning[1:]

        else:
            # Fallback: classic linear (keeps backward compat)
            _position = self._build_position(*_pos_args)
            _reasoning = self._build_reasoning(*_reas_args)
            _conn = rng.choice(_connector_bank) if rng.random() < 0.6 else ". "
            _response = _position + _conn + _reasoning

        # --- Optional elaboration for verbose personas ------------------
        if _verbosity > 0.55 and rng.random() < (_verbosity * 0.5) and _arch not in ('direct_answer', 'list_style'):
            _elab = self._build_elaboration(*_elab_args)
            if _elab:
                _response += ". " + _elab

        # --- Optional qualifier for high-SD personas --------------------
        if _sd > 0.5 and rng.random() < (_sd * 0.35) and _arch not in ('direct_answer', 'stream'):
            _qual = self._build_qualifier(topic, sentiment, rng)
            if _qual:
                _response += ". " + _qual

        # --- v1.1.0.3: Trait-driven text modulation -----------------------
        # Apply persona trait effects that go beyond archetype selection.
        # These make each simulated persona acoustically distinct.
        _response = self._apply_trait_modulation(
            _response, _traits, behavioral_profile or {}, rng,
        )

        # --- v1.1.0.6 ITERATION 1: LIWC-informed linguistic profile ------
        # Discourse markers, cognitive process markers, experiential grounding.
        # Scientific basis: Pennebaker & King (1999), Tausczik & Pennebaker (2010).
        _response = self._apply_liwc_linguistic_profile(
            _response, _traits, behavioral_profile or {}, rng,
        )

        # --- v1.1.0.6 ITERATION 2: Pragmatic naturalness -----------------
        # Clause combination, sentence starter variety, incomplete thoughts.
        # Scientific basis: Krosnick (1999), Hobbs (1979) discourse coherence.
        _response = self._apply_pragmatic_naturalness(
            _response, _traits, rng,
        )

        # --- v1.1.0.6 ITERATION 3: Vocabulary calibration ----------------
        # Register-appropriate vocabulary, casual contractions, filler words.
        # Scientific basis: Zipf's law, Biber et al. (1999) register variation.
        _response = self._apply_vocabulary_calibration(
            _response, _traits, rng,
        )

        # --- v1.1.0.3: Naturalize topic references -----------------------
        # Replace repeated "{topic}" mentions with pronouns and implicit refs.
        # Real humans don't say "trust" 4 times in 2 sentences.
        _response = self._naturalize_topic_references(_response, topic, rng)

        # --- Natural language polish ------------------------------------
        _response = self._apply_natural_polish(
            _response, _formality, _engagement, _extremity, _verbosity, rng,
        )

        # --- v1.1.0.3: Sentence-length variety enforcement ----------------
        # Real responses have high within-response variance in sentence length.
        _response = self._enforce_sentence_length_variety(_response, rng)

        # Ensure ends with period
        _response = _response.strip()
        if _response and _response[-1] not in '.!?':
            _response += '.'

        # Capitalize first letter
        if _response:
            _response = _response[0].upper() + _response[1:]

        return _response

    # ── Ultra-short response templates (v1.1.0.3) ───────────────────
    # Real survey data shows 15-20% of OE responses are 1-5 words.
    # These bypass the full archetype pipeline for disengaged participants.

    def _build_ultra_short(self, topic: str, sentiment: str,
                           formality: float, rng: random.Random) -> str:
        """Generate a 1-5 word response for disengaged/careless participants.

        v1.1.0.3: Real OE data has a heavy left tail — many responses are
        extremely short.  Straight-liners and low-engagement participants
        produce terse, often incomplete responses.  This method generates
        those directly rather than shortening full responses.
        """
        _t = topic
        # Extract a short topic word (max 2 words)
        _tw = ' '.join(_t.split()[:2]).lower().strip()
        if not _tw or _tw in ('this', 'that', 'it'):
            _tw = 'it'

        if sentiment in ('very_positive', 'positive'):
            _pool = [
                f"{_tw} is fine", f"good", f"yeah {_tw} is good",
                f"I like {_tw}", f"{_tw} works", f"fine",
                f"its good", f"ok yeah", f"makes sense",
                f"I agree", f"yeah", f"sure",
                f"{_tw} is ok", f"im for it", f"positive",
                f"sounds good", f"yep", f"thumbs up",
            ]
        elif sentiment in ('very_negative', 'negative'):
            _pool = [
                f"{_tw} is bad", f"not great", f"no thanks",
                f"dont like {_tw}", f"nah", f"not really",
                f"disagree", f"nope", f"meh",
                f"{_tw} sucks", f"not a fan", f"hard pass",
                f"terrible", f"I dont like it", f"no",
                f"not good", f"bad idea", f"against it",
            ]
        else:
            _pool = [
                f"{_tw} is ok I guess", f"idk", f"neutral",
                f"no opinion", f"dont care", f"whatever",
                f"maybe", f"could go either way", f"meh",
                f"not sure", f"50/50", f"depends",
                f"no strong feelings", f"eh", f"unsure",
                f"I guess", f"kinda neutral", f"fine either way",
            ]

        _resp = rng.choice(_pool)
        # Very casual: sometimes no capitalization, no period
        if formality < 0.4 and rng.random() < 0.7:
            _resp = _resp.lower()
        return _resp

    # ── Sentence-length enforcement (v1.1.0.3) ────────────────────
    def _enforce_sentence_length_variety(self, text: str,
                                          rng: random.Random) -> str:
        """Ensure sentences within a response vary in length.

        v1.1.0.3: Real responses have HIGH variance in sentence length
        within a single response.  Someone might write "Yeah." followed
        by a 20-word sentence followed by "Exactly."  Template-generated
        text tends to produce sentences of similar length.  This method
        breaks that uniformity.
        """
        _sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(_sentences) < 2:
            return text

        _lengths = [len(s.split()) for s in _sentences]
        _mean_len = sum(_lengths) / len(_lengths) if _lengths else 0

        # If sentences are too uniform (all within 3 words of each other),
        # either truncate a long one or fragment a short one
        _range = max(_lengths) - min(_lengths) if _lengths else 0
        if _range < 4 and len(_sentences) >= 2 and rng.random() < 0.4:
            # Pick a random sentence and either shorten or fragment it
            _idx = rng.randint(0, len(_sentences) - 1)
            _words = _sentences[_idx].split()
            if len(_words) > 6 and rng.random() < 0.5:
                # Truncate to a fragment
                _cut = rng.randint(2, 4)
                _sentences[_idx] = ' '.join(_words[:_cut]) + '.'
            elif len(_words) > 8:
                # Split into two sentences
                _mid = len(_words) // 2
                _sentences[_idx] = ' '.join(_words[:_mid]) + '. ' + \
                    ' '.join(_words[_mid:])[0].upper() + ' '.join(_words[_mid:])[1:]

        return ' '.join(_sentences)

    # ── Trait modulation data (v1.1.0.3) ─────────────────────────────
    _SD_HEDGES = [
        "or at least that's how I see it",
        "I could be wrong about this though",
        "but that's just my take",
        "I realize others might feel differently",
        "not that my opinion is the only valid one",
        "I try to be fair about it",
        "if that makes sense",
    ]
    _ACQUIESCENCE_PREFIXES = [
        "yeah I think ", "I'd agree that ", "that makes sense, ",
        "sure, ", "I mean yeah ", "right so ",
    ]
    _ACQUIESCENCE_SUFFIXES = [
        " which makes sense", " and I think that's fair",
        " and I agree with that", " which I get",
    ]

    def _apply_trait_modulation(
        self,
        text: str,
        traits: Dict[str, float],
        behavioral_profile: Dict[str, Any],
        rng: random.Random,
    ) -> str:
        """Apply persona trait effects beyond archetype selection.

        v1.1.0.3: Makes each simulated persona acoustically distinct by
        modulating text based on previously unused trait dimensions:
        - Social desirability → qualifying hedges and diplomatic language
        - Acquiescence → agreement-forward phrasing and deferential tone
        - Reading speed → sentence length and fragment frequency
        - Consistency → anaphoric repetition / thematic anchoring

        Scientific basis:
        - Paulhus (2002): High-SD manifests as qualifying language
        - Krosnick (1991): Acquiescence bias shows as agreement-forward responses
        - Tourangeau et al. (2000): Cognitive processing speed affects response elaboration
        """
        if not text or not traits:
            return text

        _sd = traits.get('social_desirability', 0.3)
        _acq = traits.get('acquiescence', 0.5)
        _speed = traits.get('reading_speed', 0.5)  # proxy for cognitive speed
        _consistency = traits.get('consistency', 0.5)

        # ── 1. SOCIAL DESIRABILITY HEDGING ────────────────────────────
        # High-SD personas add qualifying/diplomatic hedges (Paulhus 2002)
        if _sd > 0.65 and rng.random() < (_sd - 0.35):
            _hedge = rng.choice(self._SD_HEDGES)
            # Add at end of first sentence or at end of response
            _sentences = text.split('. ')
            if len(_sentences) >= 2 and rng.random() < 0.5:
                _sentences[0] = _sentences[0].rstrip('.') + ', ' + _hedge
                text = '. '.join(_sentences)
            else:
                text = text.rstrip('.!?') + ', ' + _hedge

        # ── 2. ACQUIESCENCE BIAS ──────────────────────────────────────
        # High-acquiescence personas use agreement-forward language
        if _acq > 0.65 and rng.random() < (_acq - 0.4):
            if rng.random() < 0.6:
                # Prefix: start with agreement
                _prefix = rng.choice(self._ACQUIESCENCE_PREFIXES)
                if text and text[0].isupper():
                    text = _prefix + text[0].lower() + text[1:]
                else:
                    text = _prefix + text
            else:
                # Suffix: end with agreement
                _suffix = rng.choice(self._ACQUIESCENCE_SUFFIXES)
                text = text.rstrip('.!?') + _suffix

        # ── 3. READING SPEED → SENTENCE COMPLEXITY ────────────────────
        # Fast readers (high reading_speed) produce shorter, choppier sentences.
        # Slow/careful readers produce longer, more elaborated sentences.
        if _speed > 0.75 and rng.random() < 0.35:
            # Fast: break a long sentence into two short ones
            _words = text.split()
            if len(_words) > 12:
                _mid = len(_words) // 2
                # Find a natural break point near the middle
                for _offset in range(3):
                    for _try_pos in [_mid + _offset, _mid - _offset]:
                        if 0 < _try_pos < len(_words) and _words[_try_pos].lower() in (
                            'and', 'but', 'so', 'because', 'since', 'which', 'that', 'when',
                        ):
                            _words[_try_pos] = '. ' + _words[_try_pos].capitalize()
                            text = ' '.join(_words)
                            break
                    else:
                        continue
                    break
        elif _speed < 0.3 and rng.random() < 0.25:
            # Slow: merge two short sentences into one with a connector
            _sentences = text.split('. ')
            if len(_sentences) >= 2 and len(_sentences[0].split()) < 10:
                _connectors = [' and ', ' because ', ' since ', ', and also ']
                _sentences[0] = _sentences[0].rstrip('.') + rng.choice(_connectors) + \
                    _sentences[1][0].lower() + _sentences[1][1:] if len(_sentences[1]) > 1 else _sentences[0]
                _sentences.pop(1)
                text = '. '.join(_sentences)

        # ── 4. CONSISTENCY → THEMATIC ANCHORING ───────────────────────
        # High-consistency personas reinforce their main point
        if _consistency > 0.7 and rng.random() < 0.20:
            _anchors = [
                " and I've felt this way for a long time",
                " and I'm pretty consistent on this",
                " and that hasn't changed",
                " which is what I've always thought",
            ]
            text = text.rstrip('.!?') + rng.choice(_anchors)

        return text

    def _naturalize_topic_references(self, text: str, topic: str,
                                      rng: random.Random) -> str:
        """Replace repeated topic noun phrases with natural alternatives.

        v1.1.0.3: TOPIC INTEGRATION OVERHAUL.  Real humans mention a topic
        once by name, then switch to "it", "this", "that", "the whole thing",
        or just drop it entirely.  Having the same noun phrase appear 3+ times
        in a short response is a dead giveaway of template generation.

        Strategy:
        - Keep the FIRST occurrence of the topic as-is
        - Replace subsequent occurrences with natural alternatives
        - Alternatives vary by position in the response
        """
        if not topic or len(topic) < 3:
            return text
        # Don't process very short topics that might be common words
        if topic.lower() in ('this', 'that', 'it', 'the'):
            return text

        # Find all occurrences of the topic (case-insensitive)
        _tl = topic.lower()
        _text_lower = text.lower()
        _positions = []
        _start = 0
        while True:
            _idx = _text_lower.find(_tl, _start)
            if _idx == -1:
                break
            _positions.append(_idx)
            _start = _idx + len(_tl)

        if len(_positions) <= 1:
            return text  # 0 or 1 occurrence — nothing to naturalize

        # Replacements for 2nd, 3rd, etc. occurrences
        _replacements_mid = ["it", "this", "that", "the whole thing",
                              "all of it", "the situation"]
        _replacements_start = ["It", "This", "That"]

        # Process from end to start to preserve indices
        for i, pos in enumerate(reversed(_positions)):
            if i == len(_positions) - 1:
                continue  # This is the first occurrence (keep it)
            # Decide replacement
            # Is this at the start of a sentence?
            _at_start = pos == 0 or text[pos - 2:pos] == '. '
            if _at_start:
                _repl = rng.choice(_replacements_start)
            else:
                _repl = rng.choice(_replacements_mid)
            # Preserve original case pattern
            _orig = text[pos:pos + len(topic)]
            if _orig[0].isupper() and not _at_start:
                _repl = _repl[0].upper() + _repl[1:]
            text = text[:pos] + _repl + text[pos + len(topic):]

        return text

    # ── New archetype-specific builders (v1.0.9.8) ──────────────────

    def _build_anecdote(self, topic: str, sentiment: str, domain_key: str,
                        condition: str, rng: random.Random) -> str:
        """Build a short personal anecdote with concrete, domain-aware details.

        v1.1.0.3: Anecdotes now include specific plausible scenarios rather
        than generic "I had a good/bad experience with {topic}" patterns.
        """
        _t = topic
        _tl = (topic + ' ' + (condition or '') + ' ' + (domain_key or '')).lower()

        # Domain-specific anecdotes with concrete detail
        if any(w in _tl for w in ('politic', 'trump', 'biden', 'democrat', 'republican', 'vote', 'partisan', 'polariz')):
            if sentiment in ('very_positive', 'positive'):
                _anecs = [
                    f"last Thanksgiving my family actually had a productive conversation about {_t} for once",
                    f"a coworker who normally disagrees with me on politics agreed with my take on {_t}",
                    f"I read a really good piece in the news about {_t} that crystallized my thinking",
                    f"I watched a debate about {_t} and the side I agree with actually made better arguments",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _anecs = [
                    f"my family can barely have dinner together anymore because of arguments about {_t}",
                    f"I lost a friend over a disagreement about {_t} which really opened my eyes",
                    f"I watched the news coverage of {_t} and it just confirmed all my worries",
                    f"a relative shared something on Facebook about {_t} and the comments were toxic",
                ]
            else:
                _anecs = [
                    f"my friend group is completely split on {_t} and honestly both sides make points",
                    f"I've heard compelling takes on {_t} from people I respect on different sides",
                    f"the last few years have made me less sure about where I stand on {_t}",
                ]
        elif any(w in _tl for w in ('dictator', 'trust game', 'ultimatum', 'split', 'allocat', 'economic_game', 'endow', 'cooperation')):
            if sentiment in ('very_positive', 'positive'):
                _anecs = [
                    "I remember thinking about what kind of person I want to be, and being generous felt right",
                    "I put myself in the other person's shoes and realized I'd want them to be fair to me",
                    "I've always been the type to share when I can, even with people I don't know",
                    "the decision was easy for me because being stingy just isn't who I am",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _anecs = [
                    "I thought about it strategically and keeping more just made the most sense",
                    "I've been too generous before and it backfired so I was more careful this time",
                    "I didn't feel any connection to the other person so I looked out for myself",
                    "I figured they'd do the same thing to me so why not play it smart",
                ]
            else:
                _anecs = [
                    "I honestly just went with my gut and didn't overthink the numbers",
                    "I split the difference because I didn't want to feel bad either way",
                    "I wasn't sure what the right call was so I went somewhere in the middle",
                ]
        elif any(w in _tl for w in ('health', 'medical', 'vaccine', 'treatment', 'wellbeing')):
            if sentiment in ('very_positive', 'positive'):
                _anecs = [
                    f"after I started taking my health more seriously I noticed a huge difference",
                    f"my doctor explained the benefits and once I tried it the results were clear",
                    f"someone in my family went through something similar and the outcome was great",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _anecs = [
                    f"I had a bad reaction once and that made me really cautious about these things",
                    f"my mom had a terrible experience with something similar and it scared me",
                    f"the whole process felt like they cared more about money than my wellbeing",
                ]
            else:
                _anecs = [
                    f"I've heard success stories and horror stories so I honestly don't know",
                    f"my own health journey has taught me that nothing works the same for everyone",
                ]
        else:
            # General anecdotes with concrete details
            if sentiment in ('very_positive', 'positive'):
                _anecs = [
                    f"I actually had a conversation about {_t} with a friend recently and it confirmed what I already thought",
                    f"there was a moment when {_t} just clicked for me and I've felt positive about it since",
                    f"someone I respect changed my mind about {_t} by showing me a different angle",
                    f"I was on the fence about {_t} until I saw the results firsthand",
                    f"I read something about {_t} that really resonated with my own experience",
                ]
            elif sentiment in ('very_negative', 'negative'):
                _anecs = [
                    f"I gave {_t} a fair chance and it let me down, that experience stuck with me",
                    f"someone I know got really hurt by {_t} and watching that changed my view",
                    f"I used to be neutral on {_t} until I saw what actually happens in practice",
                    f"the turning point for me was when {_t} directly affected someone I care about",
                    f"I tried to keep an open mind about {_t} but the evidence was overwhelming",
                ]
            else:
                _anecs = [
                    f"I've gone back and forth on {_t} enough times that I've stopped trying to pick a side",
                    f"different people I trust have completely opposite takes on {_t}",
                    f"my experience with {_t} really depends on the context and the day honestly",
                    f"I've seen {_t} work out great and also seen it go badly, so it's complicated",
                ]
        return rng.choice(_anecs)

    def _build_rhetorical_question(self, topic: str, intent: str,
                                    rng: random.Random) -> str:
        """Build a rhetorical question about the topic."""
        _t = topic
        _questions = [
            f"Do I think {_t} is a good thing?",
            f"How do I really feel about {_t}?",
            f"Is {_t} something I care about?",
            f"Would I change anything about {_t}?",
            f"Does {_t} matter in the grand scheme of things?",
            f"Am I surprised by how {_t} turned out?",
            f"What would I tell someone asking about {_t}?",
            f"Can I honestly say {_t} is working?",
        ]
        return rng.choice(_questions)

    def _build_concession(self, topic: str, sentiment: str,
                          rng: random.Random) -> str:
        """Build a concession that acknowledges the other side."""
        _t = topic
        if sentiment in ('very_positive', 'positive'):
            _conc = [
                f"I know {_t} isn't perfect, but",
                f"sure there are criticisms of {_t}, but honestly",
                f"people have their problems with {_t} and some are valid, but",
                f"look, {_t} has flaws like anything, but",
                f"I get why some people are skeptical of {_t}, however",
            ]
        elif sentiment in ('very_negative', 'negative'):
            _conc = [
                f"I'll admit {_t} has some good points, but",
                f"maybe {_t} works for some people, but personally",
                f"I can see the appeal of {_t} on paper, but in practice",
                f"there might be a case for {_t}, but from where I stand",
                f"even giving {_t} the benefit of the doubt,",
            ]
        else:
            _conc = [
                f"there are good arguments for and against {_t}, and",
                f"I see what people mean about {_t} on both sides, so",
                f"depending on how you look at it {_t} could go either way, and",
                f"it's not black and white with {_t},",
                f"{_t} has pros and cons and honestly",
            ]
        return rng.choice(_conc)

    def _build_stream_fragment(self, topic: str, sentiment: str,
                                domain_key: str, rng: random.Random) -> str:
        """Build a stream-of-consciousness continuation fragment."""
        _t = topic
        if sentiment in ('very_positive', 'positive'):
            _frags = [
                f"it just feels right to me",
                f"like there's real value there if you pay attention",
                f"and honestly the more I think about it the more I'm on board",
                f"it's one of those things that actually makes sense",
                f"I keep coming back to the same conclusion which is that it works",
            ]
        elif sentiment in ('very_negative', 'negative'):
            _frags = [
                f"it's just not working the way it should",
                f"and I don't think anyone is really being honest about that",
                f"the whole thing feels off to me honestly",
                f"like nobody wants to say it but it's a problem",
                f"and every time I look at it I see more issues",
            ]
        else:
            _frags = [
                f"it could go either way at this point",
                f"like I genuinely can't decide how I feel",
                f"some days I'm for it and some days I'm not",
                f"it's one of those things I keep going back and forth on",
                f"I don't have a strong take one way or the other honestly",
            ]
        return rng.choice(_frags)

    def _build_list_points(self, topic: str, sentiment: str, domain_key: str,
                           condition: str, rng: random.Random) -> str:
        """Build a list-style response with multiple short points."""
        _t = topic
        # Build 2-3 short points
        _n = rng.choice([2, 2, 3])
        if sentiment in ('very_positive', 'positive'):
            _pool = [
                f"it actually works", f"people benefit from it",
                f"the outcomes are mostly good", f"I've seen real results",
                f"it aligns with my values", f"the evidence supports it",
                f"it addresses a real need", f"the intentions are good",
                f"alternatives are worse", f"the approach makes sense",
            ]
        elif sentiment in ('very_negative', 'negative'):
            _pool = [
                f"it doesn't deliver on promises", f"people get hurt by it",
                f"the outcomes are disappointing", f"I've seen it fail",
                f"it goes against what I believe", f"the evidence isn't there",
                f"it creates new problems", f"the execution is poor",
                f"there are better alternatives", f"the reasoning is flawed",
            ]
        else:
            _pool = [
                f"there are pros and cons", f"it depends on the situation",
                f"some parts work better than others", f"the jury is still out",
                f"reasonable people disagree", f"context matters a lot",
                f"I can see both sides", f"more info would help",
                f"it's not straightforward", f"results are mixed",
            ]
        _points = rng.sample(_pool, min(_n, len(_pool)))
        _separators = [", ", ". Also ", ". And ", ", and also ", ", plus "]
        _result = _points[0]
        for p in _points[1:]:
            _result += rng.choice(_separators) + p
        return _result

    def _build_emotional_burst(self, topic: str, sentiment: str,
                                extremity: float, rng: random.Random) -> str:
        """Build an emotional opening burst."""
        _t = topic
        if sentiment in ('very_positive', 'positive'):
            if extremity > 0.6:
                _bursts = [
                    f"I genuinely love what {_t} represents",
                    f"this is something I feel really passionate about",
                    f"honestly {_t} gives me hope",
                    f"I can't say enough good things about {_t}",
                    f"my gut reaction to {_t} is overwhelmingly positive",
                ]
            else:
                _bursts = [
                    f"I have a good feeling about {_t}",
                    f"{_t} makes me feel optimistic",
                    f"there's something about {_t} that resonates with me",
                    f"I get a positive vibe from {_t}",
                ]
        elif sentiment in ('very_negative', 'negative'):
            if extremity > 0.6:
                _bursts = [
                    f"this genuinely makes me angry",
                    f"I am so frustrated with {_t}",
                    f"I can't believe how bad {_t} has gotten",
                    f"{_t} is honestly infuriating to me",
                    f"everything about {_t} bothers me",
                ]
            else:
                _bursts = [
                    f"I don't have good feelings about {_t}",
                    f"{_t} worries me honestly",
                    f"something about {_t} just rubs me the wrong way",
                    f"I'm not comfortable with {_t}",
                ]
        else:
            _bursts = [
                f"I have really conflicted feelings about {_t}",
                f"I honestly go back and forth on {_t}",
                f"{_t} makes me feel... complicated things",
                f"I wish I had a clearer take on {_t}",
            ]
        return rng.choice(_bursts)

    def _build_position(
        self,
        topic: str,
        sentiment: str,
        intent: str,
        mirror_words: List[str],
        extremity: float,
        intensity: float,
        condition: str,
        rng: random.Random,
    ) -> str:
        """Build the core position statement that directly answers the question."""
        _t = topic
        # Use a mirror word if available to echo the question
        _mirror = rng.choice(mirror_words) if mirror_words and rng.random() < 0.4 else ""

        # Strength modulator
        if extremity > 0.7 or intensity > 0.7:
            _str_pos = ["strongly believe", "am completely convinced", "feel very strongly", "have no doubt"]
            _str_neg = ["really can't stand", "strongly disagree with", "am very opposed to", "feel strongly against"]
        elif extremity < 0.3:
            _str_pos = ["kind of like", "somewhat agree with", "lean toward supporting", "mildly feel good about"]
            _str_neg = ["am a bit skeptical of", "have some concerns about", "am not fully on board with", "have mixed feelings about"]
        else:
            _str_pos = ["support", "am in favor of", "feel positively about", "believe in"]
            _str_neg = ["have issues with", "disagree with", "am concerned about", "am critical of"]

        _pos_opinion = rng.choice(_str_pos)
        _neg_opinion = rng.choice(_str_neg)

        # Build intent-specific positions
        if intent == "opinion":
            if sentiment in ('very_positive', 'positive'):
                candidates = [
                    f"I {_pos_opinion} {_t}",
                    f"my take on {_t} is positive",
                    f"when it comes to {_t} I'm on board",
                    f"{_t} is something I {_pos_opinion.replace('am ', '').replace('feel ', 'feel ')}",
                ]
            elif sentiment in ('very_negative', 'negative'):
                candidates = [
                    f"I {_neg_opinion} {_t}",
                    f"my take on {_t} is pretty negative",
                    f"when it comes to {_t} I'm not a fan",
                    f"{_t} is something I {_neg_opinion.replace('am ', '').replace('have ', 'have ')}",
                ]
            else:
                candidates = [
                    f"I have mixed feelings about {_t}",
                    f"my views on {_t} are somewhere in the middle",
                    f"I can see both sides when it comes to {_t}",
                    f"{_t} is complicated and I'm not fully decided",
                ]
        elif intent in ("explanation", "reasoning"):
            if sentiment in ('very_positive', 'positive'):
                candidates = [
                    f"the reason I feel good about {_t} is pretty straightforward",
                    f"I {_pos_opinion} {_t} because of what I've experienced",
                    f"my positive view on {_t} comes from personal experience",
                ]
            elif sentiment in ('very_negative', 'negative'):
                candidates = [
                    f"I {_neg_opinion} {_t} and I can explain why",
                    f"my problems with {_t} are based on what I've actually seen",
                    f"the reason I'm negative on {_t} is concrete",
                ]
            else:
                candidates = [
                    f"my reasoning about {_t} is that it's a trade-off",
                    f"I see arguments on both sides of {_t}",
                    f"the nuance with {_t} is what makes it hard to have a strong take",
                ]
        elif intent == "emotional_reaction":
            if sentiment in ('very_positive', 'positive'):
                candidates = [
                    f"{_t} makes me feel pretty good honestly",
                    f"I get a positive feeling when I think about {_t}",
                    f"emotionally I'm drawn to {_t}",
                ]
            elif sentiment in ('very_negative', 'negative'):
                candidates = [
                    f"{_t} genuinely frustrates me",
                    f"thinking about {_t} makes me upset",
                    f"emotionally {_t} brings up negative feelings for me",
                ]
            else:
                candidates = [
                    f"I don't have a strong emotional reaction to {_t}",
                    f"{_t} doesn't really stir up strong feelings for me",
                    f"emotionally I'm pretty neutral about {_t}",
                ]
        elif intent == "description":
            candidates = [
                f"my experience with {_t} has been {rng.choice(['interesting', 'notable', 'fairly typical', 'worth describing'])}",
                f"when I think about {_t} what comes to mind is my own experience",
                f"I can describe my relationship with {_t} pretty directly",
            ]
        elif intent == "evaluation":
            if sentiment in ('very_positive', 'positive'):
                candidates = [
                    f"I'd rate {_t} positively overall",
                    f"my evaluation of {_t} is favorable",
                    f"looking at {_t} objectively I think it's good",
                ]
            elif sentiment in ('very_negative', 'negative'):
                candidates = [
                    f"my assessment of {_t} is not great",
                    f"I'd evaluate {_t} negatively based on what I know",
                    f"looking at {_t} critically there are real problems",
                ]
            else:
                candidates = [
                    f"my evaluation of {_t} is mixed",
                    f"I'd give {_t} a moderate rating",
                    f"looking at {_t} I see strengths and weaknesses",
                ]
        elif intent == "prediction":
            if sentiment in ('very_positive', 'positive'):
                candidates = [
                    f"I think {_t} will turn out well",
                    f"my prediction for {_t} is optimistic",
                    f"I'm hopeful about where {_t} is headed",
                ]
            elif sentiment in ('very_negative', 'negative'):
                candidates = [
                    f"I'm not optimistic about {_t}",
                    f"my prediction is that {_t} won't end well",
                    f"I expect {_t} to go poorly",
                ]
            else:
                candidates = [
                    f"it's hard to predict what will happen with {_t}",
                    f"I think {_t} could go either way",
                    f"my prediction for {_t} is uncertain",
                ]
        elif intent == "recall":
            candidates = [
                f"what I remember about {_t} is specific",
                f"I can recall my experience with {_t} pretty clearly",
                f"thinking back on {_t} a few things stand out",
            ]
        else:
            # Generic fallback covers remaining intents
            if sentiment in ('very_positive', 'positive'):
                candidates = [
                    f"I {_pos_opinion} {_t}",
                    f"my view on {_t} is positive",
                    f"I'm generally in favor of {_t}",
                ]
            elif sentiment in ('very_negative', 'negative'):
                candidates = [
                    f"I {_neg_opinion} {_t}",
                    f"my view on {_t} is negative",
                    f"I'm generally against {_t}",
                ]
            else:
                candidates = [
                    f"I'm neutral on {_t}",
                    f"my thoughts on {_t} are middle-of-the-road",
                    f"I don't have strong feelings about {_t} either way",
                ]

        _pos = rng.choice(candidates)
        # Inject a mirror word reference occasionally
        if _mirror and _mirror.lower() not in _pos.lower() and rng.random() < 0.3:
            _pos += f", especially regarding {_mirror}"
        return _pos

    # ── Concrete detail banks for domain-specific reasoning (v1.1.0.3) ──
    # Real survey respondents cite specific experiences, situations, and
    # examples — not generic "I've had good experiences with {topic}".
    # These banks provide plausible concrete details per domain.
    _CONCRETE_DETAILS = {
        'political': {
            'positive': [
                "like at Thanksgiving my uncle and I actually agreed on something for once",
                "my friend who I disagree with politically even admitted the same thing",
                "I read an article about it last week that really made the case well",
                "the local elections showed it can actually work in practice",
                "even my coworkers who are on the other side politically see the logic",
                "I saw a town hall meeting where people from both parties worked together",
                "my neighbor who votes differently actually changed my mind on one thing",
                "there was a segment on the news that actually presented it fairly for once",
                "I talked to my dad about it and even he came around a little",
                "a professor I had in college predicted exactly this and was right",
            ],
            'negative': [
                "just look at how divided my own family has gotten over it",
                "I saw a news segment that laid out all the failures pretty clearly",
                "a friend of mine lost a job opportunity because of how polarized things are",
                "social media makes it so much worse, every thread is a fight",
                "I can't even bring it up at work without someone getting upset",
                "I unfollowed like half my friends on Facebook because of the arguments",
                "my sister and I didn't talk for two months after the last election",
                "the hypocrisy I see from both sides honestly makes me sick",
                "I used to be more engaged but the toxicity drove me away",
                "every holiday dinner turns into a debate now and it's exhausting",
            ],
            'neutral': [
                "my friend group is split right down the middle on it",
                "the last few elections have made me question both sides honestly",
                "I've read compelling arguments from different perspectives",
                "everyone I talk to has a completely different take on it",
                "I keep switching who I agree with depending on the specific issue",
                "my roommate and I cancel each other's votes out every election basically",
            ],
        },
        'economic_games': {
            'positive': [
                "splitting things evenly just feels like the fair thing to do",
                "I thought about what I'd want if I were on the other end",
                "keeping everything seemed like it would bother me afterward",
                "I figured a reasonable offer would get accepted and we'd both be better off",
                "I've always been taught that sharing is the right thing",
                "I'd feel guilty keeping it all even if nobody would know",
                "my parents raised me to be fair and that just kicked in",
                "I figured being generous might come back around eventually",
            ],
            'negative': [
                "I didn't know this person so why would I give away what I earned",
                "the rules said I could keep it all, so taking more just made sense",
                "giving money to a stranger with no guarantee felt like a bad deal",
                "I've been burned being too generous before so I played it safe",
                "nobody ever gives me free money so I wasn't about to either",
                "I'm a college student, I need every dollar I can get honestly",
                "the rational thing to do is maximize your own outcome, full stop",
                "I figured the other person would do the same to me",
            ],
            'neutral': [
                "I just went with what seemed like a normal amount, nothing fancy",
                "I didn't want to overthink it so I picked somewhere in the middle",
                "I figured most people would do something similar",
                "I went back and forth for a minute and then just picked a number",
                "it felt weird to keep everything but also weird to give half away",
            ],
        },
        # Game-subtype details — chosen when game type is detected
        'dictator_game': {
            'positive': [
                "I decided to give a decent chunk because hoarding it felt selfish",
                "I thought about the other person sitting there getting nothing and I couldn't do that",
                "splitting it close to even seemed like what a decent person would do",
                "I gave more than half actually because I wanted to be generous",
                "the allocation was my chance to show what kind of person I am",
            ],
            'negative': [
                "I kept most of it because there were no consequences for doing so",
                "the other person has no say in it so I looked out for myself",
                "I gave a token amount just to not feel completely terrible about it",
                "why would I split money with someone I'll never interact with again",
                "my allocation reflected what I think is rational, not sentimental",
            ],
            'neutral': [
                "I gave about a third which seemed like a reasonable middle ground",
                "I didn't want to keep everything but I also didn't want to give half",
                "the amount I chose was what felt normal without overthinking",
            ],
        },
        'trust_game': {
            'positive': [
                "I sent a good amount because I believe people return the favor",
                "investing in trust seemed like the smart play in the long run",
                "I trusted them to reciprocate and I think most people would",
                "I wanted to signal that I'm someone worth cooperating with",
                "sending money was a leap of faith but the potential payoff was worth it",
            ],
            'negative': [
                "I didn't send much because there's no guarantee of getting anything back",
                "trusting a stranger with my money just isn't something I do",
                "the risk of them keeping everything was too high for my comfort",
                "I held back because I've been taken advantage of before in similar situations",
                "the strategic move was to minimize my exposure to loss",
            ],
            'neutral': [
                "I sent a moderate amount to test the waters without risking too much",
                "I split the difference between trusting fully and not trusting at all",
                "I wanted to show some trust without being reckless about it",
            ],
        },
        'ultimatum_game': {
            'positive': [
                "the offer was fair enough that accepting was the obvious choice",
                "I proposed a fair split because lowball offers get rejected and then nobody wins",
                "I accepted because getting something beats getting nothing out of pride",
                "I made a generous offer because I want both of us to walk away happy",
                "strategic fairness just makes more sense than trying to exploit someone",
            ],
            'negative': [
                "the offer was so low it felt disrespectful so I rejected it",
                "I'd rather get nothing than accept an insulting offer on principle",
                "I tried to keep as much as possible because why not push my luck",
                "unfair proposals deserve to be punished even if it costs me",
                "I rejected because accepting teaches them that lowballing works",
            ],
            'neutral': [
                "the offer was right around what I expected so it was an easy decision",
                "I proposed something in the middle range, not too greedy not too generous",
                "I went with what seemed standard for this kind of thing",
            ],
        },
        'public_goods': {
            'positive': [
                "I contributed a lot because the group does better when everyone pitches in",
                "free-riding felt wrong even though nobody would know",
                "I put in my share because that's how cooperation is supposed to work",
                "if everyone contributes the multiplier makes it worth it for all of us",
                "I believe in doing my part for the common good even with strangers",
            ],
            'negative': [
                "why contribute when others will carry the group anyway",
                "I kept my money because I figured others would contribute enough",
                "the incentive to free-ride is just too strong when contributions are anonymous",
                "I've seen group projects fail because of freeriders and I'm not gonna be the sucker",
                "rational self-interest says contribute less and benefit from others' contributions",
            ],
            'neutral': [
                "I contributed about half, not nothing but not everything either",
                "I matched what I guessed others would put in, seemed fair",
                "group decisions are tricky so I went with a middle-of-the-road contribution",
            ],
        },
        'health': {
            'positive': [
                "after my doctor recommended it I started seeing the benefits pretty quickly",
                "I know someone who made a similar health change and it transformed their life",
                "the research I've seen on it is pretty convincing to me",
                "I noticed a real difference in how I felt within a few weeks",
                "my fitness improved noticeably after I started taking it seriously",
                "a friend who's a nurse told me the science behind it and I was sold",
                "I've been tracking my numbers and the improvement is measurable",
                "my grandmother swore by something similar and she lived to 94",
            ],
            'negative': [
                "a family member had a really bad experience with something similar",
                "I've dealt with side effects before and it made me more cautious",
                "the costs are just too high for what you actually get",
                "my own experience with the healthcare system hasn't been great",
                "my insurance wouldn't even cover it which tells you something",
                "I read about the side effects and decided it wasn't worth the risk",
                "I know three people who tried it and none of them stuck with it",
                "the pharmaceutical industry profits too much for me to trust them fully",
            ],
            'neutral': [
                "I've seen it work for some people and not for others",
                "my doctor said it depends on the individual which is frustrating",
                "I want to believe the research but it's hard to know what to trust",
                "I'm waiting for more long-term studies before I make up my mind",
                "my partner thinks one thing and my doctor thinks another so who knows",
            ],
        },
        'consumer': {
            'positive': [
                "I've been using it for a while now and it hasn't let me down",
                "my friend recommended it and they were right, it's solid",
                "compared to what I was using before this is a huge improvement",
                "for the price point you really can't beat the quality",
                "I checked the reviews before buying and they were accurate",
                "I've already recommended it to like three people",
                "the build quality is noticeably better than the competitors",
                "I was skeptical at first but after a month I'm converted",
            ],
            'negative': [
                "it broke within a month which was incredibly frustrating",
                "I found something better for half the price online",
                "the reviews were way too positive for what you actually get",
                "customer service was useless when I had an issue with it",
                "I returned it the same week, it was that disappointing",
                "my old one lasted three years and this one barely lasted three months",
                "they clearly cut corners on the materials to hit that price point",
                "I wish I'd read more reviews before I bought it",
            ],
            'neutral': [
                "it does what it's supposed to do, nothing more nothing less",
                "I'd probably try a different brand next time just to compare",
                "the quality is fine but nothing really stands out about it",
                "for the price I paid I guess it's acceptable",
                "it's not something I think about much honestly, it just works",
            ],
        },
        'trust': {
            'positive': [
                "I've found that when you give people a chance they usually come through",
                "my experience is that trust builds on itself, once you start it grows",
                "I know it's a risk but I'd rather trust than be suspicious all the time",
                "my best friendships are all built on mutual trust and vulnerability",
                "I lent money to a friend once and they paid me back the next day",
                "I've had good experiences trusting strangers, maybe I'm just lucky",
                "my coworker trusted me with something personal and it deepened our bond",
            ],
            'negative': [
                "I've been lied to enough times to know better than to trust blindly",
                "in my experience people will take advantage if given the opportunity",
                "I learned the hard way that you need to verify before you trust",
                "I had a roommate who stole from me so yeah my trust is low",
                "an ex-friend spread my personal business around and that changed me",
                "I trusted a business partner once and got completely screwed over",
                "online you literally can't trust anyone, I've been scammed twice",
            ],
            'neutral': [
                "trust is something I give cautiously and it has to be earned",
                "some people are trustworthy and some aren't, that's just reality",
                "I trust in some contexts but not others, it depends on the stakes",
                "I'm not paranoid but I'm not naive either, somewhere in between",
                "it really depends on whether I've interacted with them before",
            ],
        },
        'social': {
            'positive': [
                "I've been part of groups where cooperation really paid off for everyone",
                "when my community came together on something similar the results were amazing",
                "I've seen how the right group dynamic can change people for the better",
                "my volleyball team is a great example of how teamwork actually works",
                "I organized a neighborhood cleanup and the turnout blew me away",
                "my study group in college was the reason I passed organic chemistry",
                "I've seen strangers help each other during emergencies and it restores my faith",
            ],
            'negative': [
                "I've been in groups where one person ruined it for everyone else",
                "people say they'll cooperate but then do their own thing when it matters",
                "social pressure makes people agree to things they don't actually support",
                "every group project in school was me doing 80% of the work",
                "I tried to start a community garden and nobody else showed up after week one",
                "people are great at promising to help and terrible at following through",
                "I left a club because the drama was worse than middle school",
            ],
            'neutral': [
                "group dynamics are unpredictable, I've seen it go both ways",
                "some people are team players and some aren't, you can't always tell",
                "my experience in groups is hit or miss honestly",
                "it really depends on who's in the group and what the goal is",
                "some of my best and worst experiences have been in group settings",
            ],
        },
        'moral': {
            'positive': [
                "I was raised to believe in doing the right thing even when it's hard",
                "I think about how my choices affect other people, not just myself",
                "there are some lines I just won't cross regardless of the consequences",
                "my parents always said your character is what you do when nobody's watching",
                "I once returned a wallet with $200 in it because it was the right thing",
                "I'd rather lose money than compromise what I believe is right",
                "I volunteer at a shelter on weekends because I feel a responsibility to help",
            ],
            'negative': [
                "people love to talk about ethics but rarely follow through in practice",
                "I've seen supposedly moral people do terrible things when nobody was watching",
                "the system doesn't reward doing the right thing so why bother sometimes",
                "I watched a coworker lie to get a promotion and nothing happened to them",
                "corporations talk about values but their actions tell a different story",
                "it's easy to be moral when it costs you nothing, the real test is sacrifice",
                "I used to believe people were mostly good but experience has proven otherwise",
            ],
            'neutral': [
                "ethics gets complicated when you think about real-world tradeoffs",
                "what's right in one situation might not be in another, it depends",
                "I try to do the right thing but I don't pretend it's always clear",
                "moral questions rarely have clean answers, that's what makes them hard",
                "I think most people are trying their best with imperfect information",
            ],
        },
        'education': {
            'positive': [
                "I had a teacher in high school who genuinely changed how I think",
                "the best class I ever took made me excited to learn for the first time",
                "I learn better when I can actually discuss things instead of just listening",
                "studying abroad opened my eyes to how differently people approach problems",
                "tutoring other students honestly helped me understand the material better too",
            ],
            'negative': [
                "the education system failed me in a lot of ways growing up",
                "I had a professor who clearly didn't care and it killed my motivation",
                "standardized testing never measured what I was actually good at",
                "I spent four years in college and half the classes were a waste of time",
                "the cost of education is insane for what you actually get out of it",
            ],
            'neutral': [
                "school was fine for me, some good teachers some bad ones",
                "I learned more from life experience than from any classroom honestly",
                "education matters but it's not the only path to success",
                "some subjects I loved and some I just survived, pretty normal I think",
            ],
        },
        'technology': {
            'positive': [
                "my phone basically runs my life and I'm honestly okay with that",
                "technology made it possible for me to work from home which saved my sanity",
                "I can video call my grandparents across the country and that's amazing",
                "apps have made managing my finances so much easier",
                "I found my current job through LinkedIn so I can't complain about tech",
            ],
            'negative': [
                "I spend way too much time on my phone and I know it",
                "social media has made my anxiety noticeably worse",
                "every app wants my data and I'm tired of being the product",
                "my kid is on screens constantly and I worry about what that's doing",
                "technology was supposed to save time but I'm busier than ever",
            ],
            'neutral': [
                "technology is great until it breaks and then you realize how dependent you are",
                "I use it because I have to but I'm not excited about where it's going",
                "some tech improvements are genuinely helpful and some are just gimmicks",
                "I'm somewhere between early adopter and technophobe honestly",
            ],
        },
        'environment': {
            'positive': [
                "I started composting last year and it's surprisingly satisfying",
                "switching to public transit saved me money and reduced my guilt",
                "I've seen my neighborhood get greener since they planted those trees",
                "my local farmer's market is proof that sustainable food can work",
                "my company started a recycling program and people actually use it",
            ],
            'negative': [
                "I try to recycle but honestly I think corporations are the real problem",
                "the cost of going green is just not realistic for most people",
                "I watched a documentary about ocean pollution and it was depressing",
                "every time I read about climate change I feel more helpless",
                "the people making the rules fly private jets so why should I give up straws",
            ],
            'neutral': [
                "I do what I can but I know it's a drop in the bucket",
                "the environment is important but I have more pressing concerns day to day",
                "I'm not a denier but I'm also not sure what individual action really does",
                "it's a huge problem that requires systemic change, not just personal choices",
            ],
        },
        'workplace': {
            'positive': [
                "my current boss actually listens to feedback which is rare",
                "I got promoted last year because someone finally noticed my work",
                "the team I'm on now is the best group I've ever worked with",
                "flexible hours changed my quality of life more than a raise would",
                "my mentor at work gave me advice that completely shifted my career trajectory",
            ],
            'negative': [
                "I've had managers who take credit for their employees' work",
                "the office politics at my last job drove me to quit",
                "I watched qualified people get passed over for promotions because of who they knew",
                "burnout is treated like a personal failure when it's usually a management failure",
                "I worked 60-hour weeks for a year and got the same raise as everyone else",
            ],
            'neutral': [
                "work is work, some days are good and some days I count the hours",
                "my job pays the bills but it's not my passion, and that's okay",
                "I've had good jobs and bad jobs and the difference is usually the people",
                "the workplace has changed a lot and I'm still figuring out how I feel about it",
            ],
        },
        'identity': {
            'positive': [
                "I'm proud of where I come from and how it shaped who I am",
                "finding a community of people like me was life-changing honestly",
                "I feel more confident now that I've accepted all parts of myself",
                "representation matters and seeing people like me succeed gives me hope",
                "my cultural background gives me a perspective that others find valuable",
            ],
            'negative': [
                "I've been judged for my identity more times than I can count",
                "people make assumptions about me before I even open my mouth",
                "I had to hide parts of myself growing up and that leaves a mark",
                "stereotypes about my group are exhausting to deal with constantly",
                "I've been passed over for opportunities because of who I am not what I can do",
            ],
            'neutral': [
                "identity is complicated and I'm still figuring out what it means to me",
                "I don't think about it every day but it definitely shapes my experience",
                "some aspects of my identity matter more in certain contexts than others",
                "I'm more than just one label but I understand why people categorize",
            ],
        },
        'risk': {
            'positive': [
                "I took a risk on a new job and it was the best decision I ever made",
                "sometimes you have to bet on yourself and it pays off",
                "I invested early in something everyone said was a bad idea and it worked out",
                "the biggest gains in my life came from taking calculated risks",
                "I moved to a new city knowing nobody and it turned out great",
            ],
            'negative': [
                "I took a gamble once and lost badly, that taught me to be cautious",
                "I've seen people lose everything by being too risky with money",
                "the downside of risk is something people don't talk about enough",
                "I'd rather have a guaranteed smaller gain than risk losing it all",
                "a family member's risky investment wiped out their retirement savings",
            ],
            'neutral': [
                "I'm not a big risk taker but I'm not completely risk-averse either",
                "it depends on what's at stake, small risks sure, big ones probably not",
                "I calculate the odds before making any risky decision",
                "some risks are worth it and some aren't, you have to know the difference",
            ],
        },
        'fairness': {
            'positive': [
                "I was raised to treat everyone equally regardless of circumstances",
                "I've seen fair treatment change someone's entire outlook on life",
                "when I coach my kid's team I make sure everyone gets equal playing time",
                "the fairest boss I ever had earned more loyalty than any other",
                "I split everything equally when I'm in charge because it's the principle",
            ],
            'negative': [
                "life isn't fair and pretending it is just makes it worse",
                "I've watched people game the system while honest people get nothing",
                "the concept of fairness is nice but implementation always favors someone",
                "my own experience has shown me that merit doesn't always win",
                "equal treatment sounds good but some people need more support than others",
            ],
            'neutral': [
                "fairness means different things to different people and that's the problem",
                "I try to be fair but I know my own biases sometimes get in the way",
                "the fair thing to do isn't always the same as the equal thing to do",
                "I think about fairness a lot but rarely feel like I get it perfectly right",
            ],
        },
        'persuasion': {
            'positive': [
                "I read the argument and honestly it changed how I think about the issue",
                "the evidence presented was pretty convincing to me",
                "I appreciate when someone takes the time to make a well-reasoned case",
                "I was on the fence before but the logic made sense",
                "a good argument can absolutely change my mind if the facts are there",
            ],
            'negative': [
                "I could see right through the manipulation and it was insulting",
                "the argument had so many holes I'm surprised anyone falls for it",
                "being told what to think usually makes me want to do the opposite",
                "I'm skeptical of anyone trying too hard to convince me of something",
                "the whole thing felt like propaganda wrapped up in nice language",
            ],
            'neutral': [
                "the argument had some good points but also some I disagreed with",
                "I need more time to decide if I actually buy it or not",
                "some parts were convincing and other parts felt like a stretch",
                "I'm not sure whether I was persuaded or just worn down honestly",
            ],
        },
        'ai': {
            'positive': [
                "I use AI tools at work every day and my productivity has doubled",
                "my kid's school is using AI tutoring and their grades have improved",
                "AI caught a medical condition my doctor missed so I'm a believer",
                "the AI recommendations on my streaming service are genuinely better than my own choices",
            ],
            'negative': [
                "I got replaced by an AI tool at my last job so I'm not exactly a fan",
                "AI makes so many mistakes that people just blindly trust",
                "the thought of AI making decisions about my life terrifies me",
                "I've seen AI-generated misinformation fool people I know personally",
                "my concern is that nobody is really thinking about what happens when it goes wrong",
            ],
            'neutral': [
                "AI is a tool and like any tool it depends on who's using it",
                "I'm cautiously optimistic but also cautiously worried about AI",
                "some AI applications are amazing and some are terrifying, hard to generalize",
                "I don't fully understand how it works so I'm reserving judgment",
            ],
        },
        'relationship': {
            'positive': [
                "my partner and I have been together for years and communication is everything",
                "I learned that being vulnerable actually makes relationships stronger",
                "the best relationship advice I got was to listen more than you talk",
                "my closest friendships are the ones where we can be completely honest",
            ],
            'negative': [
                "I went through a bad breakup and it took me years to trust again",
                "I've had friendships end over really petty things and it still bothers me",
                "people I thought would be there for me disappeared when things got hard",
                "relationships are exhausting when you're the only one putting in effort",
            ],
            'neutral': [
                "relationships take work and sometimes I have the energy and sometimes I don't",
                "I've had amazing relationships and terrible ones, luck plays a big role",
                "every relationship teaches you something even if it ends badly",
                "I'm still figuring out what a healthy relationship even looks like",
            ],
        },
        'financial': {
            'positive': [
                "I started budgeting two years ago and my stress levels dropped immediately",
                "investing small amounts early on has really paid off for me",
                "I paid off my student loans last year and the relief was incredible",
                "my parents taught me about money young and I'm grateful for that",
            ],
            'negative': [
                "I'm living paycheck to paycheck and there's no safety net",
                "I made a bad investment based on a friend's advice and lost a lot",
                "the cost of everything has gone up but my salary hasn't kept pace",
                "financial stress affects literally every other area of my life",
                "I have more debt than savings and I'm in my thirties, it's scary",
            ],
            'neutral': [
                "I'm not great with money but I'm not terrible either",
                "finances are something I think about but try not to stress over",
                "I'm doing okay financially but I know one emergency could change that",
                "money management is a skill nobody taught me so I'm learning as I go",
            ],
        },
    }

    def _build_reasoning(
        self,
        topic: str,
        sentiment: str,
        intent: str,
        domain_key: str,
        mirror_words: List[str],
        condition: str,
        rng: random.Random,
    ) -> str:
        """Build a reasoning clause with concrete, domain-specific details.

        v1.1.0.3: CONCRETE DETAIL INJECTION.  Instead of generic reasoning
        like "my experience has been positive", this now injects specific,
        plausible personal experiences that make responses sound like they
        come from a real person with real memories.
        """
        _t = topic
        _sent_key = 'positive' if sentiment in ('very_positive', 'positive') else \
                    ('negative' if sentiment in ('very_negative', 'negative') else 'neutral')

        # --- Try domain-specific concrete detail first ---
        _detail = ""
        _domain_map = {
            'dictator_game': 'economic_games', 'trust_game': 'trust',
            'ultimatum_game': 'economic_games', 'public_goods': 'economic_games',
            'prisoners_dilemma': 'economic_games', 'cooperation': 'social',
            'fairness': 'fairness', 'risk_preference': 'risk',
            'polarization': 'political', 'partisanship': 'political',
            'voting': 'political', 'policy_attitudes': 'political',
            'product_evaluation': 'consumer', 'brand_loyalty': 'consumer',
            'purchase_intent': 'consumer', 'advertising': 'consumer',
            'medical_decision': 'health', 'wellbeing': 'health',
            'vaccination': 'health', 'stress': 'health',
            'intergroup': 'social', 'identity': 'identity',
            'norms': 'social', 'conformity': 'social',
            'moral_judgment': 'moral', 'ethics': 'moral',
            'education': 'education', 'learning': 'education',
            'technology': 'technology', 'ai_attitudes': 'ai',
            'automation': 'ai', 'environment': 'environment',
            'climate': 'environment', 'sustainability': 'environment',
            'workplace': 'workplace', 'leadership': 'workplace',
            'job_satisfaction': 'workplace', 'motivation': 'workplace',
            'persuasion': 'persuasion', 'credibility': 'persuasion',
            'risk': 'risk', 'gambling': 'risk',
            'attachment': 'relationship', 'intimacy': 'relationship',
            'financial_decision': 'financial', 'investment': 'financial',
        }
        # Detect domain from domain_key or topic words
        # v1.1.0.3: Game-subtype detection — dictator/trust/ultimatum/PGG get
        # their own detail banks for domain-specific vocabulary
        _det_domain = _domain_map.get(domain_key, '')
        if _det_domain == 'economic_games' or not _det_domain:
            _tl_game = (topic + ' ' + (condition or '') + ' ' + domain_key).lower()
            if 'dictator' in _tl_game:
                _det_domain = 'dictator_game'
            elif 'ultimatum' in _tl_game:
                _det_domain = 'ultimatum_game'
            elif 'public good' in _tl_game or 'public_good' in _tl_game:
                _det_domain = 'public_goods'
            elif 'trust game' in _tl_game or 'trust_game' in _tl_game:
                _det_domain = 'trust_game'
        if not _det_domain:
            _tl = (topic + ' ' + (condition or '')).lower()
            if any(w in _tl for w in ('politic', 'trump', 'biden', 'democrat', 'republican', 'vote', 'election', 'partisan')):
                _det_domain = 'political'
            elif any(w in _tl for w in ('dictator', 'trust game', 'ultimatum', 'split', 'allocat', 'endow', 'send money')):
                _det_domain = 'economic_games'
            elif any(w in _tl for w in ('health', 'medical', 'vaccine', 'doctor', 'treatment', 'illness')):
                _det_domain = 'health'
            elif any(w in _tl for w in ('product', 'brand', 'buy', 'purchase', 'price', 'shop', 'quality')):
                _det_domain = 'consumer'
            elif any(w in _tl for w in ('trust', 'reciproc', 'cooperat', 'betray')):
                _det_domain = 'trust'
            elif any(w in _tl for w in ('group', 'team', 'social', 'communit', 'belong')):
                _det_domain = 'social'
            elif any(w in _tl for w in ('moral', 'ethic', 'right', 'wrong', 'justice')):
                _det_domain = 'moral'
            elif any(w in _tl for w in ('fair', 'equal', 'equit', 'deserv', 'merit')):
                _det_domain = 'fairness'
            elif any(w in _tl for w in ('educat', 'learn', 'school', 'teach', 'student', 'college', 'universit')):
                _det_domain = 'education'
            elif any(w in _tl for w in ('technolog', 'digital', 'internet', 'app', 'software', 'computer')):
                _det_domain = 'technology'
            elif any(w in _tl for w in ('ai', 'artificial', 'automat', 'robot', 'algorithm', 'machine learn')):
                _det_domain = 'ai'
            elif any(w in _tl for w in ('environ', 'climate', 'green', 'sustain', 'pollut', 'recycle')):
                _det_domain = 'environment'
            elif any(w in _tl for w in ('work', 'job', 'employ', 'boss', 'manag', 'office', 'career')):
                _det_domain = 'workplace'
            elif any(w in _tl for w in ('identit', 'race', 'gender', 'cultur', 'ethnic', 'divers')):
                _det_domain = 'identity'
            elif any(w in _tl for w in ('risk', 'gambl', 'chance', 'uncertain', 'bet', 'odds')):
                _det_domain = 'risk'
            elif any(w in _tl for w in ('persuad', 'convinc', 'argument', 'messag', 'credib', 'propag')):
                _det_domain = 'persuasion'
            elif any(w in _tl for w in ('relat', 'partner', 'dating', 'marriage', 'friend', 'love')):
                _det_domain = 'relationship'
            elif any(w in _tl for w in ('financ', 'money', 'invest', 'debt', 'saving', 'budget', 'income')):
                _det_domain = 'financial'

        _bank = self._CONCRETE_DETAILS.get(_det_domain, {})
        _details = _bank.get(_sent_key, [])
        if _details and rng.random() < 0.65:
            # 65% chance to use a concrete detail instead of generic reasoning
            _detail = rng.choice(_details)

        if _detail:
            # Build reasoning around the concrete detail
            _bridges = ["", "I mean ", "like ", "for example "]
            _reason = rng.choice(_bridges) + _detail
        else:
            # Fallback to generic but improved reasoning
            if _sent_key == 'positive':
                _reasons = [
                    f"I've actually seen {_t} work in practice, not just in theory",
                    f"the people around me who engage with {_t} seem to be doing well",
                    f"every time I've given {_t} a chance it has surprised me positively",
                    f"what changed my mind was watching {_t} play out in real situations",
                    f"I didn't always feel this way but the evidence piled up",
                    f"I've tried the alternatives and {_t} comes out on top for me",
                ]
            elif _sent_key == 'negative':
                _reasons = [
                    f"I've watched {_t} fail too many times to give it the benefit of the doubt",
                    f"the gap between what {_t} promises and what it delivers is huge",
                    f"I know people personally who were let down by {_t}",
                    f"the track record just doesn't support optimism about {_t}",
                    f"I used to be more open-minded about {_t} but that changed",
                    f"the more I learn about {_t} the worse it looks honestly",
                ]
            else:
                _reasons = [
                    f"I know smart people on opposite sides of {_t} and they both make sense",
                    f"my experience with {_t} has been inconsistent which makes it hard to judge",
                    f"for every success story about {_t} there's a failure story too",
                    f"the answer probably depends on specifics that vary a lot",
                    f"I keep changing my mind about {_t} depending on what angle I consider",
                ]
            _reason = rng.choice(_reasons)

        # Condition-aware extension (20% chance)
        if condition and rng.random() < 0.20:
            _cond_words = re.findall(r'\b[a-zA-Z]{4,}\b', condition.lower())
            _cond_stop = {'control', 'treatment', 'condition', 'group', 'neutral',
                          'baseline', 'default', 'standard', 'cell', 'level', 'high', 'low'}
            _cond_content = [w for w in _cond_words if w not in _cond_stop][:2]
            if _cond_content:
                _cond_phrase = ' '.join(_cond_content)
                _extensions = [
                    f" and {_cond_phrase} is a big part of why",
                    f" especially when you factor in {_cond_phrase}",
                    f" and the {_cond_phrase} aspect makes it even clearer",
                ]
                _reason += rng.choice(_extensions)

        return _reason

    def _build_elaboration(
        self,
        topic: str,
        sentiment: str,
        domain_key: str,
        mirror_words: List[str],
        rng: random.Random,
    ) -> str:
        """Build an elaboration that adds depth."""
        _t = topic
        # First try domain-specific vocab
        _vocab = self._get_domain_vocabulary(domain_key, _t)
        if sentiment in ('very_positive', 'positive') and _vocab.get('elaborations_pos'):
            return rng.choice(_vocab['elaborations_pos'])
        elif sentiment in ('very_negative', 'negative') and _vocab.get('elaborations_neg'):
            return rng.choice(_vocab['elaborations_neg'])
        elif _vocab.get('elaborations_neu'):
            return rng.choice(_vocab['elaborations_neu'])

        # Generic elaborations
        if sentiment in ('very_positive', 'positive'):
            _elabs = [
                "It's one of those things that has genuinely earned my support over time.",
                "I've talked to other people who feel the same way and it confirms what I already thought.",
                "The more I learn about it the more convinced I become.",
                "It might not be perfect but the positives far outweigh the negatives.",
            ]
        elif sentiment in ('very_negative', 'negative'):
            _elabs = [
                "I don't think enough people realize how problematic this really is.",
                "Every time I hear more about it my concerns grow.",
                "I wish I could be more positive but the issues are too significant to ignore.",
                "People who support this haven't seen what I've seen.",
            ]
        else:
            _elabs = [
                "I try to stay informed and make up my own mind rather than just following what others say.",
                "The nuance gets lost in most conversations about this unfortunately.",
                "I think more people are in the middle on this than the discourse suggests.",
                "It's one of those things where your personal experience really shapes your view.",
            ]
        return rng.choice(_elabs)

    def _build_qualifier(
        self, topic: str, sentiment: str, rng: random.Random,
    ) -> str:
        """Build a qualifying statement (typically from high-SD personas)."""
        _qualifiers = [
            "I realize not everyone will agree with me on this and that's okay",
            "I tried to be fair and consider other viewpoints before forming my opinion",
            "I'm open to changing my mind if I see compelling evidence",
            "I know my perspective is just one of many",
            "I don't claim to have all the answers here",
            "Obviously my experience might not be representative of everyone's",
            "I want to be careful not to overgeneralize from my own experience",
        ]
        return rng.choice(_qualifiers)

    # ── Verbal tic system (v1.1.0.3) ──────────────────────────────────
    # Real humans have consistent verbal tics — one person peppers in "honestly",
    # another says "like" constantly, another starts with "I mean".  Each
    # participant gets one assigned tic that appears 1-3 times in their response.
    _VERBAL_TIC_POOLS = [
        # Each sub-list is a tic "family" — one is assigned per participant
        ["honestly", "honestly ", "honestly,"],
        ["like", "like ", "like,"],
        ["basically", "basically ", "basically,"],
        ["I mean", "I mean ", "I mean,"],
        ["you know", "you know ", "you know,"],
        ["just", "just ", "I just"],
        ["literally", "literally ", "I literally"],
        ["actually", "actually ", "actually,"],
        ["right", "right ", "right,"],
        ["anyway", "anyway ", "anyway,"],
    ]

    # Synonym rotation to prevent vocabulary monotony across participants
    _SYNONYM_ROTATIONS = {
        'good': ['decent', 'solid', 'fine', 'nice', 'great', 'positive', 'alright'],
        'bad': ['terrible', 'awful', 'poor', 'negative', 'rough', 'not great', 'lousy'],
        'think': ['believe', 'feel', 'reckon', 'figure', 'guess', 'say'],
        'important': ['big deal', 'significant', 'key', 'a priority', 'crucial', 'major'],
        'agree': ['go along with', 'support', 'back', 'see the point of', 'get behind'],
        'disagree': ['push back on', 'have issues with', 'take issue with', 'question'],
        'really': ['genuinely', 'truly', 'seriously', 'honestly', 'definitely'],
        'problem': ['issue', 'concern', 'trouble', 'challenge', 'difficulty'],
        'experience': ['time with', 'history with', 'dealings with', 'run-ins with'],
    }

    def _apply_verbal_tic(self, text: str, participant_seed: int,
                          formality: float, rng: random.Random) -> str:
        """Assign and apply a consistent verbal tic for this participant.

        v1.1.0.3: Real people have distinctive filler words that repeat
        in their speech.  The "honestly" person says it 2-3 times.  The
        "like" person sprinkles it throughout.  This makes each participant
        sound like a unique individual rather than a template.
        """
        if formality > 0.6:
            return text  # Formal writers don't use verbal tics

        # Deterministically assign a tic family based on participant seed
        _tic_idx = participant_seed % len(self._VERBAL_TIC_POOLS)
        _tic_family = self._VERBAL_TIC_POOLS[_tic_idx]
        _tic = _tic_family[0]  # Primary tic word

        # Count existing occurrences
        _existing = text.lower().count(_tic.lower().strip())
        if _existing >= 2:
            return text  # Already has this tic naturally

        # Insert 1-2 times at natural positions
        _n_insert = rng.randint(1, 2) - _existing
        if _n_insert <= 0:
            return text

        _words = text.split()
        if len(_words) < 5:
            return text

        for _ in range(_n_insert):
            if len(_words) < 5:
                break
            # Insert at a natural position (after 2nd-5th word, or after a comma)
            _candidates = []
            for i in range(2, min(len(_words) - 1, max(6, len(_words) // 2))):
                if _words[i - 1].endswith(',') or _words[i - 1].endswith('.') or i in (2, 3, 4):
                    _candidates.append(i)
            if _candidates:
                _pos = rng.choice(_candidates)
                _words.insert(_pos, _tic_family[rng.randint(0, len(_tic_family) - 1)].strip() + ',')

        return ' '.join(_words)

    def _apply_synonym_rotation(self, text: str, participant_seed: int,
                                 rng: random.Random) -> str:
        """Rotate common words to synonyms based on participant identity.

        v1.1.0.3: Prevents vocabulary monotony across participants.
        Each participant deterministically maps common words to specific
        synonyms, so Participant A always says "decent" where Participant B
        says "solid" and Participant C says "fine".
        """
        _local_rng = random.Random(participant_seed + 9999)
        for _word, _synonyms in self._SYNONYM_ROTATIONS.items():
            if _word in text.lower() and _local_rng.random() < 0.3:
                # Pick a deterministic synonym for this participant
                _syn = _synonyms[participant_seed % len(_synonyms)]
                # Replace first occurrence only (preserve case)
                _idx = text.lower().find(_word)
                if _idx >= 0:
                    _orig = text[_idx:_idx + len(_word)]
                    if _orig[0].isupper():
                        _syn = _syn[0].upper() + _syn[1:]
                    text = text[:_idx] + _syn + text[_idx + len(_word):]
        return text

    # ══════════════════════════════════════════════════════════════════
    # FREE-FORM TEXT QUALITY — ITERATIONS 1-3 (v1.1.0.6)
    # Scientific basis: Pennebaker & King (1999) LIWC linguistic styles,
    # Tausczik & Pennebaker (2010) psychological meaning of words,
    # Krosnick (1991) satisficing theory, Denscombe (2008) response length.
    # ══════════════════════════════════════════════════════════════════

    # ── ITERATION 1: LIWC-informed linguistic profile system ──────────
    # Maps persona trait dimensions to empirically-grounded linguistic
    # features per Pennebaker & King (1999) — Big Five → LIWC correlations.
    #
    # Key findings implemented:
    # - Extraversion → more social words, positive emotion, higher word count
    # - Neuroticism → more negative emotion, anxiety words, first-person singular
    # - Openness → longer words, more articles, insight words
    # - Agreeableness → more positive emotion, fewer negative, fewer swear words
    # - Conscientiousness → fewer negations, organized structure

    # Discourse markers that real humans use organically (not template-like)
    # Source: ACL taxonomy of discourse markers in dialog
    _DISCOURSE_MARKERS_CASUAL = [
        "well ", "so ", "anyway ", "like ", "you know ", "I mean ",
        "ok so ", "right so ", "basically ", "honestly ", "look ",
    ]
    _DISCOURSE_MARKERS_MODERATE = [
        "that said, ", "in any case, ", "on that note, ", "to be fair, ",
        "interestingly, ", "admittedly, ", "at the end of the day, ",
    ]
    _DISCOURSE_MARKERS_FORMAL = [
        "broadly speaking, ", "to that end, ", "in particular, ",
        "on balance, ", "with that in mind, ", "as I see it, ",
    ]

    # Cognitive process markers — signals of genuine thought (Tausczik 2010)
    _COGNITIVE_PROCESS_MARKERS = {
        'insight': ["I realize ", "I noticed ", "it occurred to me ", "what struck me was ",
                    "I came to see ", "I figured out ", "it dawned on me "],
        'causation': ["because ", "the reason is ", "that's why ", "which led to ",
                      "so then ", "that caused ", "it made me "],
        'certainty': ["I'm sure ", "without question ", "there's no doubt ", "clearly ",
                      "I'm certain ", "I know for a fact "],
        'tentative': ["I think maybe ", "it seems like ", "possibly ", "I'm not sure but ",
                      "it might be ", "there's a chance ", "I wonder if "],
        'discrepancy': ["but then ", "even though ", "despite that ", "on the other hand ",
                        "at the same time ", "however ", "and yet "],
    }

    # Personal experience grounding phrases (makes responses feel lived-in)
    _EXPERIENTIAL_GROUNDING = {
        'recent': [
            "just the other day ", "recently ", "last week ",
            "a few days ago ", "this past month ", "not too long ago ",
        ],
        'habitual': [
            "I've always ", "every time I ", "growing up I ",
            "for as long as I can remember ", "I tend to ", "usually I ",
        ],
        'specific_person': [
            "my friend ", "someone I know ", "a coworker of mine ",
            "my neighbor ", "a family member ", "someone in my class ",
        ],
        'sensory': [
            "I remember feeling ", "I could see ", "what stood out was ",
            "the thing that hit me was ", "I vividly recall ", "I noticed ",
        ],
    }

    # Response length calibration — log-normal distribution parameters
    # Source: Denscombe (2008): mean ~40 words for engaged respondents
    _LENGTH_PARAMS = {
        'high_engagement': {'mu': 3.5, 'sigma': 0.55},    # median ~33 words
        'moderate_engagement': {'mu': 2.8, 'sigma': 0.5},  # median ~16 words
        'low_engagement': {'mu': 1.6, 'sigma': 0.6},       # median ~5 words
    }

    def _apply_liwc_linguistic_profile(
        self,
        text: str,
        traits: Dict[str, Any],
        behavioral_profile: Dict[str, Any],
        rng: random.Random,
    ) -> str:
        """Apply LIWC-informed linguistic transformations based on persona traits.

        v1.1.0.6 ITERATION 1.
        Maps persona dimensions to empirically validated linguistic features:
        - High social desirability → more hedging, diplomatic qualifiers
        - High extremity → more emotional/intense language, certainty markers
        - Low extremity → tentative language, discrepancy markers
        - High engagement → cognitive process markers (insight, causation)
        - Low formality → discourse markers, filler words, contractions

        Scientific basis: Pennebaker & King (1999), Tausczik & Pennebaker (2010).
        """
        if not text or not traits:
            return text

        _extremity = traits.get('extremity', 0.4)
        _engagement = traits.get('attention', 0.5)
        _formality = traits.get('formality', 0.5)
        _verbosity = traits.get('verbosity', 0.5)
        _sd = traits.get('social_desirability', 0.3)

        _sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if not _sentences:
            return text

        # ── 1. DISCOURSE MARKER INSERTION ───────────────────────────────
        # Real humans begin sentences with discourse markers ~20-40% of time
        if _formality < 0.4:
            _dm_bank = self._DISCOURSE_MARKERS_CASUAL
            _dm_prob = 0.25
        elif _formality > 0.65:
            _dm_bank = self._DISCOURSE_MARKERS_FORMAL
            _dm_prob = 0.15
        else:
            _dm_bank = self._DISCOURSE_MARKERS_MODERATE
            _dm_prob = 0.18

        # Don't add marker to first sentence (would sound unnatural)
        for i in range(1, len(_sentences)):
            if rng.random() < _dm_prob and len(_sentences[i]) > 15:
                _marker = rng.choice(_dm_bank)
                _s = _sentences[i]
                # Only add if sentence doesn't already start with a marker
                if not any(_s.lower().startswith(m.strip()) for m in _dm_bank):
                    _sentences[i] = _marker + _s[0].lower() + _s[1:]

        # ── 2. COGNITIVE PROCESS MARKERS ────────────────────────────────
        # Engaged respondents show more cognitive processing language
        if _engagement > 0.55 and _verbosity > 0.4 and rng.random() < 0.30:
            if _extremity > 0.65:
                _cog_type = 'certainty'
            elif _extremity < 0.35:
                _cog_type = 'tentative'
            elif rng.random() < 0.5:
                _cog_type = 'insight'
            else:
                _cog_type = 'causation'

            _cog_marker = rng.choice(self._COGNITIVE_PROCESS_MARKERS[_cog_type])
            # Insert at start of a middle sentence
            if len(_sentences) >= 2:
                _insert_idx = rng.randint(1, len(_sentences) - 1)
                _s = _sentences[_insert_idx]
                if not any(_s.lower().startswith(cm.strip().lower())
                           for cm_list in self._COGNITIVE_PROCESS_MARKERS.values()
                           for cm in cm_list):
                    _sentences[_insert_idx] = _cog_marker + _s[0].lower() + _s[1:]

        # ── 3. EXPERIENTIAL GROUNDING ───────────────────────────────────
        # Engaged respondents ground opinions in personal experience
        if _engagement > 0.5 and _verbosity > 0.45 and rng.random() < 0.25:
            _exp_type = rng.choice(['recent', 'habitual', 'specific_person', 'sensory'])
            _exp_phrase = rng.choice(self._EXPERIENTIAL_GROUNDING[_exp_type])
            # Add as a brief clause at the end of a sentence
            if len(_sentences) >= 2:
                _target = rng.randint(0, len(_sentences) - 1)
                _s = _sentences[_target].rstrip('.!?')
                # Only add if sentence is long enough to handle it
                if len(_s.split()) > 4:
                    _bridges = [" — ", ", like ", ", and "]
                    _sentences[_target] = _s + rng.choice(_bridges) + _exp_phrase.rstrip() + "."

        text = ' '.join(_sentences)
        return text

    def _calibrate_response_length(
        self,
        text: str,
        engagement: float,
        verbosity: float,
        rng: random.Random,
    ) -> str:
        """Calibrate response length to match log-normal distribution from real data.

        v1.1.0.6 ITERATION 1.
        Real survey response lengths follow a log-normal distribution
        (Denscombe 2008): most are short, with a long right tail.
        This method adjusts generated text to match empirical distributions
        rather than the uniform-ish lengths that template generation produces.
        """
        import math
        # Determine engagement tier
        if engagement > 0.6:
            _params = self._LENGTH_PARAMS['high_engagement']
        elif engagement > 0.35:
            _params = self._LENGTH_PARAMS['moderate_engagement']
        else:
            _params = self._LENGTH_PARAMS['low_engagement']

        # Adjust by verbosity
        _mu = _params['mu'] + (verbosity - 0.5) * 0.4
        _sigma = _params['sigma']

        # Sample target word count from log-normal
        _target = int(math.exp(rng.gauss(_mu, _sigma)))
        _target = max(2, min(_target, 120))  # Clamp to reasonable range

        _words = text.split()
        _current = len(_words)

        if _current <= _target + 3:
            return text  # Close enough, don't truncate good content

        # Truncate at a sentence boundary nearest to target
        _sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        _running = 0
        _kept: List[str] = []
        for _s in _sentences:
            _s_words = len(_s.split())
            if _running + _s_words > _target and _kept:
                break
            _kept.append(_s)
            _running += _s_words

        if _kept:
            _result = ' '.join(_kept)
            if _result and _result[-1] not in '.!?':
                _result += '.'
            return _result
        return text

    # ── ITERATION 2: Pragmatic naturalness & cognitive process markers ─

    # Clause combination patterns — how humans naturally chain ideas
    # Source: Discourse coherence models (Hobbs 1979, Grosz & Sidner 1986)
    _CLAUSE_COMBINERS = {
        'additive': [" and ", ", and also ", " plus ", ", not to mention "],
        'contrastive': [" but ", " although ", " even though ", ", yet "],
        'causal': [" so ", " which is why ", " because of that ", " and that's why "],
        'temporal': [" and then ", " after that ", " eventually ", ", and over time "],
        'elaborative': [" — meaning ", " which basically means ", ", in other words ", " like "],
    }

    # Sentence-initial variation patterns — prevent "I [verb]" monotony
    _SENTENCE_STARTERS_ENGAGED = [
        "The thing is, ", "What gets me is ", "Part of it is ",
        "One thing I noticed is ", "Here's the thing — ",
        "From what I can tell, ", "The way I look at it, ",
        "Looking at it now, ", "Having gone through that, ",
    ]
    _SENTENCE_STARTERS_CASUAL = [
        "Like ", "Ok so ", "Tbh ", "Ngl ", "Idk but ",
        "Fr though ", "Lowkey ", "Not gonna lie ",
    ]

    # Incomplete thought patterns — casual personas trail off
    _TRAILING_OFF_PATTERNS = [
        "...", ".. idk", "... but yeah", ".. anyway",
        "... or something like that", "... you know what I mean",
        ".. but whatever", "... hard to explain",
    ]

    def _apply_pragmatic_naturalness(
        self,
        text: str,
        traits: Dict[str, Any],
        rng: random.Random,
    ) -> str:
        """Apply pragmatic naturalness transformations.

        v1.1.0.6 ITERATION 2.
        Makes text sound like natural human writing through:
        1. Clause combination instead of short choppy sentences
        2. Varied sentence starters (break "I [verb]" monotony)
        3. Incomplete thoughts for casual personas
        4. Natural repetition avoidance
        5. Question-asking within responses (engaged personas)

        Scientific basis: Krosnick (1999) satisficing spectrum,
        Hobbs (1979) discourse coherence.
        """
        if not text or len(text) < 20:
            return text

        _formality = traits.get('formality', 0.5)
        _engagement = traits.get('attention', 0.5)
        _verbosity = traits.get('verbosity', 0.5)

        _sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(_sentences) < 2:
            return text

        # ── 1. CLAUSE COMBINATION ───────────────────────────────────────
        # Real humans combine short clauses with conjunctions rather than
        # writing isolated short sentences.  "I like X. It works well."
        # becomes "I like X and it works well."
        if len(_sentences) >= 3 and rng.random() < 0.30:
            _combine_idx = rng.randint(0, len(_sentences) - 2)
            _s1 = _sentences[_combine_idx]
            _s2 = _sentences[_combine_idx + 1]
            # Only combine if both are short-to-medium length
            if len(_s1.split()) < 12 and len(_s2.split()) < 12:
                # Pick combiner type based on relationship
                _s2_lower = _s2.lower()
                if any(w in _s2_lower for w in ('but', 'however', 'although', 'yet', 'though')):
                    _ctype = 'contrastive'
                elif any(w in _s2_lower for w in ('so', 'because', 'therefore', 'thus')):
                    _ctype = 'causal'
                elif any(w in _s2_lower for w in ('then', 'after', 'later', 'eventually')):
                    _ctype = 'temporal'
                else:
                    _ctype = rng.choice(['additive', 'elaborative'])

                _combiner = rng.choice(self._CLAUSE_COMBINERS[_ctype])
                _combined = _s1.rstrip('.!?') + _combiner + _s2[0].lower() + _s2[1:]
                _sentences[_combine_idx] = _combined
                _sentences.pop(_combine_idx + 1)

        # ── 2. SENTENCE STARTER VARIATION ───────────────────────────────
        # Detect "I [verb]" repetition — the #1 tell of generated text
        _i_verb_count = sum(1 for s in _sentences if re.match(r"^I\s+[a-z]", s))
        if _i_verb_count >= 2 and len(_sentences) >= 2:
            # Replace one "I [verb]" start with a varied starter
            _starter_bank = (self._SENTENCE_STARTERS_CASUAL if _formality < 0.35
                             else self._SENTENCE_STARTERS_ENGAGED)
            _replaced = False
            for i in range(1, len(_sentences)):  # Skip first sentence
                if re.match(r"^I\s+[a-z]", _sentences[i]) and not _replaced and rng.random() < 0.5:
                    _starter = rng.choice(_starter_bank)
                    _sentences[i] = _starter + _sentences[i][0].lower() + _sentences[i][1:]
                    _replaced = True

        # ── 3. INCOMPLETE THOUGHTS (casual, lower engagement) ───────────
        # ~8-12% of casual survey responses have trailing-off endings
        if _formality < 0.35 and _engagement < 0.55 and rng.random() < 0.12:
            _last = _sentences[-1].rstrip('.!?')
            # Cut the sentence partway and add trailing off
            _words = _last.split()
            if len(_words) > 5:
                _cut = rng.randint(max(3, len(_words) // 2), len(_words) - 1)
                _sentences[-1] = ' '.join(_words[:_cut]) + rng.choice(self._TRAILING_OFF_PATTERNS)

        # ── 4. EMBEDDED QUESTION (engaged personas think aloud) ─────────
        # Engaged respondents sometimes pose questions to themselves
        if _engagement > 0.6 and _verbosity > 0.5 and rng.random() < 0.12:
            _self_questions = [
                "And honestly, is that so wrong?",
                "I don't know, what does that say about me?",
                "But should I feel differently? Maybe.",
                "Does that make sense? I think it does.",
                "Am I biased here? Probably, but still.",
            ]
            _sentences.append(rng.choice(_self_questions))

        text = ' '.join(_sentences)
        return text

    # ── ITERATION 3: Vocabulary frequency + cross-response polish ─────

    # Vocabulary frequency tiers based on COCA/BNC word lists
    # Source: Zipf's law — high-frequency words for casual, mixed for educated
    _VOCAB_UPGRADES_EDUCATED = {
        # Common word → more sophisticated alternative (for high-openness personas)
        'good': ['beneficial', 'constructive', 'favorable', 'worthwhile'],
        'bad': ['detrimental', 'problematic', 'concerning', 'counterproductive'],
        'big': ['significant', 'substantial', 'considerable', 'profound'],
        'small': ['modest', 'marginal', 'negligible', 'incremental'],
        'a lot': ['considerably', 'substantially', 'to a great extent'],
        'get': ['obtain', 'acquire', 'gain', 'attain'],
        'show': ['demonstrate', 'illustrate', 'indicate', 'reveal'],
        'help': ['facilitate', 'enable', 'contribute to', 'support'],
        'thing': ['aspect', 'element', 'factor', 'dimension'],
        'stuff': ['matters', 'issues', 'considerations', 'factors'],
    }

    _VOCAB_DOWNGRADES_CASUAL = {
        # Formal word → more casual alternative (for low-openness personas)
        'however': ['but', 'though', 'still'],
        'therefore': ['so', 'thats why', 'which is why'],
        'additionally': ['also', 'plus', 'and'],
        'nevertheless': ['still', 'but still', 'even so'],
        'regarding': ['about', 'on', 'with'],
        'consider': ['think about', 'look at', 'weigh'],
        'subsequently': ['then', 'after that', 'later'],
        'demonstrate': ['show', 'prove', 'make clear'],
        'significant': ['big', 'major', 'huge', 'real'],
        'perspective': ['take', 'view', 'angle', 'side'],
    }

    # Contraction patterns by formality tier
    # Source: Biber et al. (1999) Longman Grammar — spoken vs written register
    _CASUAL_CONTRACTIONS_EXTENDED = {
        'going to ': "gonna ", 'want to ': "wanna ", 'got to ': "gotta ",
        'kind of ': "kinda ", 'sort of ': "sorta ", 'a lot of ': "a lotta ",
        'I do not know': "idk", 'to be honest': "tbh",
        'in my opinion': "imo", 'let me ': "lemme ",
    }

    # Filler word frequency calibration by engagement (Krosnick satisficing spectrum)
    _FILLER_WORDS_BY_TIER = {
        'optimizing': [],  # No fillers — careful, considered responses
        'weak_satisficing': ['like', 'kinda', 'I guess', 'sort of', 'I think'],
        'strong_satisficing': ['idk', 'whatever', 'I guess', 'meh'],
    }

    def _apply_vocabulary_calibration(
        self,
        text: str,
        traits: Dict[str, Any],
        rng: random.Random,
    ) -> str:
        """Apply vocabulary calibration based on persona's education/openness level.

        v1.1.0.6 ITERATION 3.
        Real humans have characteristic vocabulary ranges:
        - High-openness/educated → occasional sophisticated vocabulary
        - Low-openness/casual → simpler, concrete vocabulary
        - Young/casual → informal contractions (gonna, wanna, kinda)

        This avoids the "uncanny valley" where all generated responses
        use the same middle-register vocabulary regardless of persona.

        Scientific basis: Pennebaker & King (1999) — Openness predicts
        longer words, more articles, more insight words.
        Zipf's law for natural vocabulary distribution.
        """
        if not text or len(text) < 15:
            return text

        _formality = traits.get('formality', 0.5)
        _openness = traits.get('openness', 0.5) if 'openness' in traits else traits.get('verbosity', 0.5)
        _engagement = traits.get('attention', 0.5)

        # ── 1. VOCABULARY REGISTER ADJUSTMENT ───────────────────────────
        if _openness > 0.7 and _formality > 0.5 and rng.random() < 0.25:
            # Educated persona: occasionally upgrade common words
            for _common, _upgrades in self._VOCAB_UPGRADES_EDUCATED.items():
                if ' ' + _common + ' ' in ' ' + text.lower() + ' ' and rng.random() < 0.20:
                    _upgrade = rng.choice(_upgrades)
                    # Case-preserving single replacement
                    _idx = text.lower().find(_common)
                    if _idx >= 0:
                        _orig = text[_idx:_idx + len(_common)]
                        if _orig[0].isupper():
                            _upgrade = _upgrade[0].upper() + _upgrade[1:]
                        text = text[:_idx] + _upgrade + text[_idx + len(_common):]
                        break  # Max 1 upgrade per response

        elif _formality < 0.35 and rng.random() < 0.25:
            # Casual persona: downgrade formal words
            for _formal, _downgrades in self._VOCAB_DOWNGRADES_CASUAL.items():
                if _formal.lower() in text.lower() and rng.random() < 0.25:
                    _downgrade = rng.choice(_downgrades)
                    _idx = text.lower().find(_formal.lower())
                    if _idx >= 0:
                        _orig = text[_idx:_idx + len(_formal)]
                        if _orig[0].isupper():
                            _downgrade = _downgrade[0].upper() + _downgrade[1:]
                        text = text[:_idx] + _downgrade + text[_idx + len(_formal):]
                        break

        # ── 2. CASUAL CONTRACTIONS (very informal personas) ─────────────
        # Source: Biber et al. (1999) — spoken register features
        if _formality < 0.3 and rng.random() < 0.20:
            for _full, _casual in self._CASUAL_CONTRACTIONS_EXTENDED.items():
                if _full.lower() in text.lower() and rng.random() < 0.30:
                    _idx = text.lower().find(_full.lower())
                    if _idx >= 0:
                        text = text[:_idx] + _casual + text[_idx + len(_full):]
                        break  # Max 1 per response

        # ── 3. FILLER WORD CALIBRATION ──────────────────────────────────
        # Satisficing tier determines filler word density
        if _engagement > 0.65:
            _tier = 'optimizing'
        elif _engagement > 0.35:
            _tier = 'weak_satisficing'
        else:
            _tier = 'strong_satisficing'

        _fillers = self._FILLER_WORDS_BY_TIER[_tier]
        if _fillers and _formality < 0.5 and rng.random() < 0.20:
            _filler = rng.choice(_fillers)
            _words = text.split()
            if len(_words) > 6:
                # Insert filler at a natural position
                _pos = rng.randint(2, min(5, len(_words) - 2))
                _words.insert(_pos, _filler + ',')
                text = ' '.join(_words)

        return text

    def _apply_cross_response_voice_consistency(
        self,
        text: str,
        participant_seed: int,
        traits: Dict[str, Any],
        rng: random.Random,
    ) -> str:
        """Ensure consistent voice patterns within a participant across questions.

        v1.1.0.6 ITERATION 3.
        A real person writes with consistent stylistic patterns:
        - Same person always uses "I think" vs "I feel" vs "I believe"
        - Same person consistently uses or avoids contractions
        - Same person has consistent sentence length preferences
        - Same person has consistent punctuation habits

        This is independent of content — it's the "accent" of their writing.
        """
        if not text or len(text) < 15:
            return text

        # Create participant-deterministic RNG for voice features
        _voice_rng = random.Random(participant_seed + 77777)

        # ── 1. OPINION VERB CONSISTENCY ─────────────────────────────────
        # Each person has a preferred opinion verb
        _opinion_verbs = ["think", "feel", "believe", "reckon", "figure", "say"]
        _preferred_verb = _opinion_verbs[_voice_rng.randint(0, len(_opinion_verbs) - 1)]
        _other_verbs = [v for v in ["think", "feel", "believe"] if v != _preferred_verb]
        for _ov in _other_verbs:
            if f"I {_ov} " in text and _voice_rng.random() < 0.50:
                text = text.replace(f"I {_ov} ", f"I {_preferred_verb} ", 1)

        # ── 2. PUNCTUATION PERSONALITY ──────────────────────────────────
        # Some people overuse exclamation marks, others never use them
        _uses_exclamation = _voice_rng.random() < 0.25
        _uses_ellipsis = _voice_rng.random() < 0.20
        _formality = traits.get('formality', 0.5)

        if not _uses_exclamation and '!' in text:
            text = text.replace('!', '.')
        elif _uses_exclamation and _formality < 0.5 and rng.random() < 0.20:
            # Add occasional exclamation
            _sents = text.split('. ')
            if len(_sents) >= 2:
                _exc_idx = rng.randint(0, len(_sents) - 1)
                if _sents[_exc_idx].endswith('.'):
                    _sents[_exc_idx] = _sents[_exc_idx][:-1] + '!'
                text = '. '.join(_sents)

        if _uses_ellipsis and _formality < 0.45 and rng.random() < 0.15:
            # Replace one period with ellipsis
            _first_period = text.find('. ')
            if _first_period > 10:
                text = text[:_first_period] + '... ' + text[_first_period + 2:]

        # ── 3. HEDGE WORD PREFERENCE ────────────────────────────────────
        # Each person has a go-to hedge word
        _hedge_words = ["kind of", "sort of", "somewhat", "fairly", "pretty much",
                        "more or less", "to some extent"]
        _preferred_hedge = _hedge_words[_voice_rng.randint(0, len(_hedge_words) - 1)]
        _generic_hedges = ["kind of", "sort of", "somewhat"]
        for _gh in _generic_hedges:
            if _gh in text.lower() and _gh != _preferred_hedge and _voice_rng.random() < 0.40:
                _idx = text.lower().find(_gh)
                if _idx >= 0:
                    text = text[:_idx] + _preferred_hedge + text[_idx + len(_gh):]
                    break

        return text

    # ── Typo simulation data (v1.1.0.3) ───────────────────────────────
    # Based on QWERTY keyboard adjacency.  Only applied to casual personas.
    _ADJACENT_KEYS = {
        'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr', 'f': 'dg',
        'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk', 'k': 'jl', 'l': 'k',
        'm': 'n', 'n': 'bm', 'o': 'ip', 'p': 'o', 'r': 'et', 's': 'ad',
        't': 'ry', 'u': 'yi', 'v': 'cb', 'w': 'qe', 'y': 'tu',
    }
    # Common naturally-occurring typos in survey data
    _COMMON_TYPOS = {
        'the ': ['teh ', 'the '], 'and ': ['adn ', 'and '],
        'that ': ['taht ', 'that '], 'with ': ['wiht ', 'with '],
        'this ': ['tihs ', 'thsi ', 'this '], 'about ': ['abotu ', 'about '],
        'their ': ['thier ', 'their '], 'because ': ['becuase ', 'beacuse ', 'because '],
        'really ': ['realy ', 'relly ', 'really '], 'people ': ['poeple ', 'people '],
    }

    def _apply_natural_polish(
        self,
        text: str,
        formality: float,
        engagement: float,
        extremity: float,
        verbosity: float,
        rng: random.Random,
    ) -> str:
        """Apply natural imperfections to make text sound typed by a real person.

        v1.1.0.3: NATURAL IMPERFECTION ENGINE.  Real survey responses contain:
        - Typos and misspellings (especially in casual, fast typing)
        - Missing apostrophes ("dont", "cant")
        - Comma splices and run-on sentences
        - Missing final periods
        - Double spaces
        - Inconsistent capitalization
        - Sentence fragments
        - Parenthetical asides
        - Self-interruptions

        The rate of imperfections scales inversely with formality and
        directly with casualness.  Formal personas produce cleaner text.
        """
        if not text:
            return text

        # ── 1. CONTRACTIONS (nearly universal in informal writing) ──────
        if formality < 0.65 and rng.random() < (0.8 - formality):
            _contractions = {
                'I am ': "I'm ", 'I have ': "I've ", 'I will ': "I'll ",
                'I would ': "I'd ", 'do not ': "don't ", 'does not ': "doesn't ",
                'did not ': "didn't ", 'is not ': "isn't ", 'are not ': "aren't ",
                'was not ': "wasn't ", 'were not ': "weren't ",
                'cannot ': "can't ", 'could not ': "couldn't ",
                'would not ': "wouldn't ", 'should not ': "shouldn't ",
                'it is ': "it's ", 'that is ': "that's ",
                'there is ': "there's ", 'they are ': "they're ",
                'we are ': "we're ", ' will not ': " won't ",
            }
            for _full, _short in _contractions.items():
                if _full.lower() in text.lower():
                    _idx = text.lower().find(_full.lower())
                    text = text[:_idx] + _short + text[_idx + len(_full):]

        # ── 2. MISSING APOSTROPHES (casual personas) ────────────────────
        # ~15-30% of casual survey respondents skip apostrophes
        if formality < 0.35 and rng.random() < 0.30:
            _apost_map = {
                "don't": "dont", "can't": "cant", "won't": "wont",
                "didn't": "didnt", "wouldn't": "wouldnt", "shouldn't": "shouldnt",
                "isn't": "isnt", "aren't": "arent", "wasn't": "wasnt",
                "I'm": "Im", "I've": "Ive", "I'd": "Id", "I'll": "Ill",
                "it's": "its", "that's": "thats", "there's": "theres",
            }
            # Only strip 1-2 apostrophes to look natural, not all of them
            _n_strip = rng.randint(1, 2)
            _found = [(k, v) for k, v in _apost_map.items() if k in text]
            if _found:
                for k, v in rng.sample(_found, min(_n_strip, len(_found))):
                    text = text.replace(k, v, 1)

        # ── 3. TYPOS (casual, low-engagement personas) ──────────────────
        # ~2-5% typo rate for casual respondents
        if formality < 0.4 and engagement < 0.6 and rng.random() < 0.20:
            # Apply 1 common word typo
            for _correct, _typos in self._COMMON_TYPOS.items():
                if _correct in text.lower() and rng.random() < 0.15:
                    _typo = rng.choice(_typos[:-1])  # Last entry is correct (control)
                    _idx = text.lower().find(_correct)
                    text = text[:_idx] + _typo + text[_idx + len(_correct):]
                    break  # Max 1 typo per response

        # ── 4. COMMA SPLICES (very common in real survey data) ──────────
        # "I think it's fine, it doesn't bother me much"
        if formality < 0.5 and rng.random() < 0.20:
            _sentences = text.split('. ')
            if len(_sentences) >= 2:
                _splice_at = rng.randint(0, len(_sentences) - 2)
                _sentences[_splice_at] = _sentences[_splice_at].rstrip('.') + ', ' + \
                    _sentences[_splice_at + 1][0].lower() + _sentences[_splice_at + 1][1:] \
                    if len(_sentences[_splice_at + 1]) > 1 else _sentences[_splice_at]
                _sentences.pop(_splice_at + 1)
                text = '. '.join(_sentences)

        # ── 5. MISSING FINAL PERIOD (~25% of casual respondents) ────────
        if formality < 0.5 and rng.random() < 0.25:
            text = text.rstrip('.')

        # ── 6. LOWERCASE START (casual personas sometimes don't capitalize)
        if formality < 0.3 and rng.random() < 0.20:
            if text and text[0].isupper():
                text = text[0].lower() + text[1:]

        # ── 7. SELF-CORRECTION (moderate engagement) ────────────────────
        if 0.3 < engagement < 0.8 and rng.random() < 0.10:
            _sentences = text.split('. ')
            if len(_sentences) >= 2:
                _insert_at = rng.randint(0, len(_sentences) - 2)
                _correction = rng.choice(self._SELF_CORRECTIONS)
                _sentences[_insert_at] = _sentences[_insert_at] + _correction
                text = '. '.join(_sentences)

        # ── 8. PARENTHETICAL ASIDE (moderate-high engagement) ───────────
        if engagement > 0.5 and verbosity > 0.4 and rng.random() < 0.10:
            _asides = [
                " (and this is just me)", " (at least from what I've seen)",
                " (I could be wrong though)", " (not that it matters much)",
                " (which is kind of the point)", " (or maybe I'm overthinking it)",
            ]
            _words = text.split()
            if len(_words) > 6:
                _pos = rng.randint(3, min(8, len(_words) - 2))
                _words.insert(_pos, rng.choice(_asides))
                text = ' '.join(_words)

        # ── 9. TRAILING THOUGHTS (verbose personas) ─────────────────────
        if verbosity > 0.7 and rng.random() < 0.20:
            _trails = [
                " but I could go on", " and there's more to it",
                " but that's the gist", " if that makes sense",
                " but anyway", " so yeah",
            ]
            text = text.rstrip('.!?')
            text += rng.choice(_trails)

        # ── 10. DOUBLE SPACE (occasional, realistic artifact) ───────────
        if rng.random() < 0.05:
            _words = text.split(' ')
            if len(_words) > 3:
                _pos = rng.randint(1, len(_words) - 2)
                _words[_pos] = _words[_pos] + ' '  # extra space
                text = ' '.join(_words)

        # ── 11. VERBAL TICS (per-participant filler words) ────────────
        # v1.1.0.3: Assign a consistent verbal tic per participant
        _seed_val = hash(text[:20]) if text else 0  # Deterministic from content
        if formality < 0.55 and rng.random() < 0.35:
            text = self._apply_verbal_tic(text, _seed_val, formality, rng)

        # ── 12. SYNONYM ROTATION (cross-participant diversity) ────────
        # v1.1.0.3: Rotate common words to prevent vocabulary monotony
        if rng.random() < 0.40:
            text = self._apply_synonym_rotation(text, _seed_val, rng)

        return text

    def _enforce_behavioral_coherence(
        self,
        response: str,
        behavioral_profile: Dict[str, Any],
        sentiment: str,
        local_rng: random.Random,
        question_intent: str = "",
    ) -> str:
        """Ensure open-text response is coherent with numeric behavioral data.

        v1.0.5.0: Comprehensive behavioral coherence pipeline with:
        1. Straight-liner truncation (engagement-matched text length)
        2. Sentiment polarity correction (tone must match ratings)
        3. Intensity-driven vocabulary injection (extreme raters sound extreme)
        4. Social desirability modulation (high-SD adds qualifying hedges)
        5. Consistency-driven thematic coherence (consistent raters = consistent themes)
        6. Extremity-driven absolute language (extreme responders use strong words)

        v1.0.8.4: Added question_intent parameter. For narrative/creative intents
        (creative_belief, personal_disclosure, creative_narrative, personal_story),
        sentiment polarity correction is SKIPPED because it would destroy the
        generated content. A conspiracy theory doesn't need a "positive" lead-in
        just because the participant rated other items positively.

        Scientific basis:
        - Krosnick (1999): Satisficing respondents show cross-method consistency
          in their low effort (both numeric and open-ended)
        - Podsakoff et al. (2003): Common method variance creates within-person
          consistency patterns that should be reflected in simulated data
        - Greenleaf (1992): Extreme response style is trait-like and consistent
        - Paulhus (2002): Social desirability manifests as qualifying language
        """
        if not response or not behavioral_profile:
            return response

        _pattern = behavioral_profile.get('response_pattern', 'unknown')
        _mean = behavioral_profile.get('response_mean')
        _intensity = behavioral_profile.get('intensity', 0.5)
        _straight = behavioral_profile.get('straight_lined', False)
        _traits = behavioral_profile.get('trait_profile', {})
        _sd = _traits.get('social_desirability', 0.3)
        _extremity = _traits.get('extremity', 0.4)
        _consistency = behavioral_profile.get('consistency_score', 0.5)

        # 1. Straight-liners get minimal text regardless of content
        if _straight:
            sentences = re.split(r'(?<=[.!?])\s+', response)
            if len(sentences) > 1:
                response = sentences[0]
            words = response.split()
            if len(words) > 12:
                response = ' '.join(words[:8])
            return response

        # v1.0.8.4: Skip polarity correction for narrative/creative intents.
        # A conspiracy theory or personal disclosure shouldn't be overridden with
        # sentiment-correction phrases like "Overall I felt pretty positively about this."
        _narrative_intents = {'creative_belief', 'personal_disclosure', 'creative_narrative',
                              'personal_story', 'hypothetical'}
        _skip_polarity = question_intent in _narrative_intents

        # 2. Expanded sentiment indicator lists for polarity correction
        _positive_indicators = [
            'good', 'great', 'like', 'liked', 'enjoy', 'enjoyed', 'positive',
            'happy', 'pleased', 'love', 'loved', 'agree', 'support', 'trust',
            'glad', 'nice', 'awesome', 'amazing', 'wonderful', 'appreciate',
            'favorable', 'hopeful', 'optimistic', 'encouraged', 'satisfied',
            'impressed', 'confident', 'exciting', 'interesting', 'fun',
        ]
        _negative_indicators = [
            'bad', 'terrible', 'hate', 'dislike', 'negative', 'angry',
            'frustrated', 'upset', 'disagree', 'distrust', 'awful',
            'horrible', 'worst', 'annoying', 'disappointing', 'concerned',
            'worried', 'skeptical', 'uneasy', 'critical', 'problems',
            'wrong', 'fail', 'poor', 'weak', 'unfair', 'disturbing',
        ]
        _resp_lower = response.lower()
        _has_positive = any(w in _resp_lower for w in _positive_indicators)
        _has_negative = any(w in _resp_lower for w in _negative_indicators)

        # Positive rater with negative-sounding text
        # v1.0.8.4: Skip for narrative intents — don't prepend sentiment to stories/theories
        if not _skip_polarity and _pattern in ('strongly_positive', 'moderately_positive') and _mean is not None and _mean >= 5.0:
            if _has_negative and not _has_positive:
                _positive_leads = [
                    "Overall I felt pretty positively about this. ",
                    "I generally had a good impression. ",
                    "Looking back I think it was mostly positive. ",
                    "Despite some concerns, I'm largely positive. ",
                ]
                response = local_rng.choice(_positive_leads) + response

        # Negative rater with positive-sounding text
        elif not _skip_polarity and _pattern in ('strongly_negative', 'moderately_negative') and _mean is not None and _mean <= 3.0:
            if _has_positive and not _has_negative:
                _negative_leads = [
                    "Honestly I wasn't too happy about this. ",
                    "I had some real concerns about this. ",
                    "I didn't feel great about it. ",
                    "I have significant reservations. ",
                ]
                response = local_rng.choice(_negative_leads) + response

        # 3. v1.0.5.0: Enhanced intensity matching — probability scales with intensity
        # Extreme raters (high intensity) should sound emphatic more often
        # v1.0.8.5: Use recall-specific intensifiers for recall intent (memories
        # shouldn't get "I feel really strongly about this" — they should get
        # "I remember this vividly")
        _is_recall = question_intent == "recall"
        if _intensity > 0.5 and _mean is not None:
            # Scale probability with intensity: 0.5→25%, 0.7→45%, 0.9→65%
            _intensity_prob = 0.25 + (_intensity - 0.5) * 1.0
            if _mean >= 5.5 and local_rng.random() < _intensity_prob:
                _pos_intensifiers = [
                    " I remember this vividly and it was great.",
                    " That memory really stayed with me in a good way.",
                    " Looking back, this stands out as a genuinely positive experience.",
                    " I can picture it clearly and it still makes me feel good.",
                ] if _is_recall else [
                    " I feel really strongly about this.",
                    " This is something I care a lot about.",
                    " I'm genuinely enthusiastic about this.",
                    " This really resonated with me on a personal level.",
                    " I can't overstate how much this matters to me.",
                    " My feelings on this are very clear and very positive.",
                    " I'd go so far as to say this was genuinely great.",
                    " This is exactly the kind of thing I can get behind.",
                ]
                if not response.rstrip().endswith('.'):
                    response = response.rstrip() + '.'
                response = response.rstrip() + local_rng.choice(_pos_intensifiers)
            elif _mean <= 2.5 and local_rng.random() < _intensity_prob:
                _neg_intensifiers = [
                    " That memory still bothers me when I think about it.",
                    " Looking back, the frustration is still fresh.",
                    " I remember it clearly and it still doesn't sit well.",
                    " That experience left a lasting negative impression.",
                ] if _is_recall else [
                    " I really didn't like this at all.",
                    " This was genuinely frustrating.",
                    " I have serious issues with this.",
                    " This needs significant improvement.",
                    " I can't say enough about how much this bothered me.",
                    " My negative feelings on this are very strong.",
                    " I would not want to experience this again.",
                    " There are real and serious problems here.",
                ]
                if not response.rstrip().endswith('.'):
                    response = response.rstrip() + '.'
                response = response.rstrip() + local_rng.choice(_neg_intensifiers)

        # 4. v1.0.5.0: Social desirability modulation
        # High-SD participants add qualifying/hedging language to avoid seeming extreme
        if _sd > 0.65 and local_rng.random() < 0.4:
            _sd_qualifiers = [
                "I tried to be thoughtful about this. ",
                "I want to be fair in my assessment. ",
                "I considered different perspectives. ",
                "I tried to give a balanced response. ",
                "I thought carefully before answering. ",
                "I wanted to be honest but also respectful. ",
            ]
            # Only prepend if response doesn't already start with a qualifier
            if not any(response.lower().startswith(q[:8].lower()) for q in _sd_qualifiers):
                response = local_rng.choice(_sd_qualifiers) + response

        # 5. v1.0.5.0: Extremity-driven absolute language
        # Extreme responders (high extremity trait) use strong/absolute words
        if _extremity > 0.7 and local_rng.random() < 0.5:
            # Replace hedging words with absolutes
            _hedge_to_absolute = {
                'somewhat ': 'very ', 'kind of ': 'really ', 'a bit ': 'quite ',
                'fairly ': 'extremely ', 'sort of ': 'definitely ',
                'might ': 'clearly ', 'maybe ': 'definitely ',
                'i think ': 'I know ', 'i guess ': 'I\'m certain ',
            }
            _modified = response
            for _hedge, _absolute in _hedge_to_absolute.items():
                if _hedge in _modified.lower():
                    _idx = _modified.lower().find(_hedge)
                    _modified = _modified[:_idx] + _absolute + _modified[_idx + len(_hedge):]
                    break  # Only replace one to avoid over-correction
            response = _modified

        # 6. v1.0.9.3: Low-engagement shortening — moderately disengaged responders
        # produce shorter, less elaborated text (but not as extreme as straight-liners)
        _attention = _traits.get('attention_level', 0.5)
        if _attention < 0.35 and not _straight:
            sentences = re.split(r'(?<=[.!?])\s+', response)
            if len(sentences) > 2:
                response = ' '.join(sentences[:2])
            words = response.split()
            if len(words) > 20:
                response = ' '.join(words[:16]) + '.'

        # 7. v1.0.9.3: Consistency-driven response anchoring
        # When consistency is low, inject uncertainty markers at ~30% rate
        if _consistency < 0.35 and local_rng.random() < 0.30:
            _uncertainty_markers = [
                "I'm not entirely sure but ",
                "It's hard to say exactly, but ",
                "I go back and forth on this, but ",
                "My feelings are mixed, but ",
            ]
            if not any(response.lower().startswith(m[:12].lower()) for m in _uncertainty_markers):
                response = local_rng.choice(_uncertainty_markers) + response[0].lower() + response[1:]

        # 8. v1.0.9.3: Formality adjustment for high-formality personas
        _formality = _traits.get('formality', 0.5)
        if _formality > 0.75 and local_rng.random() < 0.35:
            _casual_to_formal = {
                "don't": "do not", "can't": "cannot", "won't": "will not",
                "shouldn't": "should not", "couldn't": "could not",
                "it's": "it is", "that's": "that is",
            }
            _modified = response
            for _cas, _form in _casual_to_formal.items():
                if _cas in _modified.lower():
                    _idx = _modified.lower().find(_cas)
                    _modified = _modified[:_idx] + _form + _modified[_idx + len(_cas):]
                    break
            response = _modified

        return response

    def _make_careless(self, response: str, engagement: float, local_rng: random.Random = None,
                        question_text: str = "", question_context: str = "",
                        question_intent: str = "", sentiment: str = "neutral") -> str:
        """Transform response to reflect careless/disengaged responding.

        v1.0.3.8: Careless responses are now STILL ON-TOPIC. A careless
        participant in a Trump study writes 'trump is ok i guess' not
        'the study was fine'. We extract topic words from the response
        and build a short topic-relevant careless response.

        v1.0.3.10: Added question_text parameter as fallback source for
        topic extraction. If topic words can't be extracted from the
        response itself, we extract from the original question text.
        This eliminates the 'it' pronoun fallback entirely.

        v1.0.8.4: Added question_context and question_intent parameters.
        Context serves as secondary fallback after question_text for topic
        extraction. Intent enables intent-specific careless templates —
        careless respondents answering narrative/creative questions still
        produce a SHORT version of that content type, not generic "idk".
        """
        rng = local_rng or random.Random()

        if engagement < 0.2:
            # Extract a topic word from the response to stay on-topic
            _words = re.findall(r'\b[a-zA-Z]{4,}\b', response.lower())
            # v1.0.4.7: Unified stop word list — includes researcher-instruction vocabulary
            _stop = {
                'this', 'that', 'about', 'think', 'feel', 'have', 'some',
                'with', 'from', 'very', 'really', 'quite', 'when', 'what',
                'their', 'honestly', 'strong', 'pretty', 'based', 'things',
                'going', 'important', 'something', 'direction', 'personal',
                'feelings', 'topic', 'comes', 'tried', 'answer', 'genuine',
                'views', 'experiences', 'shaped', 'resonates', 'mixed',
                'good', 'positive', 'negative', 'frustrated', 'pleased',
                'mostly', 'strongly', 'heading', 'right',
                # Researcher instruction vocabulary (v1.0.4.7)
                'participants', 'respondents', 'subjects', 'people',
                'primed', 'priming', 'exposed', 'exposure',
                'presented', 'presenting', 'shown', 'showing',
                'told', 'telling', 'instructed', 'instructions',
                'assigned', 'randomly', 'random', 'randomized',
                'thinking', 'reading', 'viewing', 'watching', 'completing',
                'answering', 'reporting', 'sharing', 'responding',
                'before', 'after', 'during', 'following', 'prior',
                'stories', 'story', 'experience',
                'whether', 'toward', 'towards',
                'question', 'questions', 'context', 'study', 'survey',
                'condition', 'conditions', 'measure', 'measured',
                'response', 'responses', 'describe', 'explain', 'please',
                'open', 'ended', 'text', 'variable',
                # v1.0.6.3: Additional stop words
                'here', 'there', 'well', 'like', 'even', 'still',
                'only', 'each', 'every', 'both', 'many', 'overall',
                'certain', 'particular', 'specific', 'general',
            }
            _topic_words = [w for w in _words if w not in _stop][:3]

            # v1.0.3.10: If response didn't yield topic words, try question_text
            if not _topic_words and question_text:
                _qt_words = re.findall(r'\b[a-zA-Z]{4,}\b', question_text.lower())
                _topic_words = [w for w in _qt_words if w not in _stop][:3]

            # v1.0.8.4: If question_text didn't yield topic words, try question_context
            if not _topic_words and question_context:
                _ctx_words = re.findall(r'\b[a-zA-Z]{4,}\b', question_context.lower())
                _topic_words = [w for w in _ctx_words if w not in _stop][:3]

            _topic = ' '.join(_topic_words[:2]) if _topic_words else 'the question'

            # v1.0.8.4: Intent-specific careless templates for narrative/creative questions.
            # Even careless respondents produce SHORT content matching the question type,
            # not just generic "idk". A careless person answering "what's your conspiracy
            # theory" writes "idk maybe government stuff" not "idk".
            if question_intent == "creative_belief":
                careless_templates = [
                    "idk maybe the government is hiding stuff",
                    "i think theres some conspiracy about big companies but idk details",
                    f"something about {_topic} being rigged probably",
                    "everything is a conspiracy if you think about it lol",
                    f"{_topic} idk its all sketchy",
                    "the usual stuff about politicians lying idk",
                    f"i dont trust {_topic} but cant explain why really",
                    "something something cover up idk",
                ]
                return rng.choice(careless_templates)
            elif question_intent == "personal_disclosure":
                careless_templates = [
                    "i dont really want to share that stuff",
                    "my family knows some things but whatever",
                    f"idk something about {_topic} i guess",
                    "theres stuff but id rather not say",
                    "nothing that interesting honestly",
                    "my family knows me better than most people i guess",
                    "personal stuff that i dont talk about",
                    "id rather keep it to myself tbh",
                ]
                return rng.choice(careless_templates)
            elif question_intent in ("creative_narrative", "personal_story"):
                careless_templates = [
                    f"i have a {_topic} story but its not that interesting",
                    f"something happened with {_topic} once idk",
                    f"i dont remember much about {_topic}",
                    f"{_topic} thing happened to me once whatever",
                    "cant really think of anything specific",
                    f"i guess there was this one time with {_topic}",
                    f"nothing crazy about {_topic} in my life",
                    "idk i dont have a good story for this",
                ]
                return rng.choice(careless_templates)
            elif question_intent == "hypothetical":
                careless_templates = [
                    f"idk probably whatever about {_topic}",
                    f"id just go with the flow on {_topic}",
                    "i have no idea honestly",
                    f"depends on {_topic} i guess",
                    "hard to say",
                    f"idk id figure out {_topic} when it happens",
                    "whatever seemed right i guess",
                ]
                return rng.choice(careless_templates)
            elif question_intent == "recommendation":
                careless_templates = [
                    f"idk just do whatever with {_topic}",
                    f"id probably just go with {_topic}",
                    "no real advice honestly",
                    f"idk figure out {_topic} yourself i guess",
                    f"whatever works for {_topic}",
                    "i dont have good advice for this",
                ]
                return rng.choice(careless_templates)
            elif question_intent == "comparison":
                careless_templates = [
                    f"idk {_topic} is about the same as the rest",
                    f"cant really tell the difference with {_topic}",
                    f"{_topic} is whatever compared to other stuff",
                    "theyre all pretty similar honestly",
                    f"idk if {_topic} is better or worse",
                    f"doesnt matter {_topic} or something else",
                ]
                return rng.choice(careless_templates)
            elif question_intent == "recall":
                careless_templates = [
                    f"dont remember much about {_topic}",
                    f"i forget about {_topic} honestly",
                    f"idk something happened with {_topic}",
                    "cant really remember",
                    f"nothing stands out about {_topic}",
                    f"{_topic} was forgettable honestly",
                ]
                return rng.choice(careless_templates)

            # v1.0.8.5: Sentiment-aware careless templates — careless participants
            # who rated positively still write short positive text, not neutral.
            if sentiment in ('very_positive', 'positive'):
                careless_templates = [
                    f"{_topic} is good",
                    f"yeah {_topic} is fine",
                    f"i like {_topic}",
                    f"{_topic} is ok yeah",
                    f"good about {_topic}",
                    f"{_topic} sure its good",
                    f"ya {_topic} is alright",
                    f"i support {_topic} i guess",
                    f"{_topic} seems good to me",
                    f"no complaints about {_topic}",
                ]
            elif sentiment in ('very_negative', 'negative'):
                careless_templates = [
                    f"dont like {_topic}",
                    f"{_topic} is bad",
                    f"not a fan of {_topic}",
                    f"{_topic} sucks honestly",
                    f"nah {_topic}",
                    f"{_topic} is terrible",
                    f"problems with {_topic}",
                    f"not great about {_topic}",
                    f"{_topic} needs to change",
                    f"disappointed with {_topic}",
                ]
            else:
                careless_templates = [
                    f"{_topic} is ok i guess",
                    f"idk {_topic}",
                    f"{_topic} whatever",
                    f"meh {_topic}",
                    f"{_topic} is fine",
                    f"sure {_topic}",
                    f"{_topic}",
                    f"dont care about {_topic}",
                    f"{_topic} its fine i guess",
                    f"i dont really have strong feelings about {_topic}",
                    f"{_topic} is what it is",
                    f"not much to say about {_topic}",
                    f"{_topic} doesnt really matter to me",
                    f"ya {_topic} sure",
                    f"no comment on {_topic}",
                    f"{_topic} is alright",
                    f"i guess {_topic} is ok",
                    f"nothing to add about {_topic}",
                ]
            return rng.choice(careless_templates)
        elif engagement < 0.3:
            # Very short version — keep first meaningful fragment
            words = response.split()[:7]
            _short = ' '.join(words).rstrip('.,!?') + '.'
            # v1.0.8.0: Lowercase to add inattentive feel
            if rng.random() < 0.5:
                _short = _short.lower()
            return _short
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

        v1.0.3.9: Expanded domain-specific extensions to cover 20+ domains.
        General extensions are now more substantive and less generic.
        """
        rng = local_rng or random.Random()

        # v1.0.4.3: Comprehensive domain-specific extensions for 30+ domains
        # Each domain has sentiment-aligned elaborations grounded in how real
        # participants in that domain actually write about their experiences.
        domain_extensions = {
            # --- Behavioral Economics domains ---
            StudyDomain.DICTATOR_GAME: {
                "positive": [" Fairness is important to me.", " I tried to be reasonable about sharing.",
                             " I believe in treating people generously."],
                "negative": [" The situation felt unfair.", " I wasn't sure what the right split was.",
                             " I kept more because the stakes felt real to me."],
            },
            StudyDomain.RISK_PREFERENCE: {
                "positive": [" I'm comfortable with some uncertainty.", " Risk doesn't bother me much.",
                             " Sometimes you have to take chances."],
                "negative": [" I prefer to play it safe.", " Too much uncertainty makes me nervous.",
                             " I'd rather keep what I have than gamble."],
            },
            StudyDomain.TRUST_GAME: {
                "positive": [" I generally try to give people the benefit of the doubt.",
                             " I think most people will reciprocate if you trust them."],
                "negative": [" Trust has to be earned.", " I'm cautious about trusting others.",
                             " People don't always follow through on their word."],
            },
            StudyDomain.ULTIMATUM_GAME: {
                "positive": [" I think fair splits are the right approach.", " Both sides deserve respect."],
                "negative": [" Low offers feel insulting.", " It's hard to accept something that seems unfair."],
            },
            StudyDomain.PUBLIC_GOODS: {
                "positive": [" Contributing to the group makes sense when everyone does their part.",
                             " Cooperation pays off in the long run."],
                "negative": [" Free riders make it hard to trust the group.",
                             " I'm reluctant to contribute when others don't."],
            },
            StudyDomain.LOSS_AVERSION: {
                "positive": [" Focusing on what I could gain helps me decide.",
                             " I try not to let fear of loss control my choices."],
                "negative": [" The thought of losing what I have really bothers me.",
                             " I'd rather avoid a loss than pursue a gain."],
            },
            StudyDomain.FRAMING_EFFECTS: {
                "positive": [" The way things were presented made the choice clearer.",
                             " I felt the information helped me decide."],
                "negative": [" I noticed the framing and tried to think past it.",
                             " The presentation made me more cautious."],
            },
            StudyDomain.BEHAVIORAL_ECONOMICS: {
                "positive": [" I considered the tradeoffs carefully.", " My decision felt rational."],
                "negative": [" The decision was harder than I expected.", " I'm not sure I chose optimally."],
            },

            # --- Consumer/Marketing domains ---
            StudyDomain.CONSUMER: {
                "positive": [" Quality matters to me when choosing.", " I'm drawn to well-made things.",
                             " The option aligned with what I value."],
                "negative": [" I'm particular about what I choose.", " Value matters a lot to me.",
                             " I wasn't impressed with the options."],
            },

            # --- AI/Technology domains ---
            StudyDomain.AI_ATTITUDES: {
                "positive": [" Technology can be really helpful when used well.", " I'm open to new approaches.",
                             " AI tools have a lot of potential if done right."],
                "negative": [" I prefer human judgment in most cases.", " Automation isn't always the answer.",
                             " I worry about relying too much on technology."],
            },

            # --- Political/Social domains ---
            StudyDomain.POLITICAL: {
                "positive": [" These are issues I care about deeply.", " My political views guide my thinking here.",
                             " I believe civic engagement matters."],
                "negative": [" The political situation frustrates me.", " I think our politics are broken.",
                             " Politicians don't seem to listen."],
            },
            StudyDomain.POLARIZATION: {
                "positive": [" I try to see things from multiple angles.", " Common ground is possible.",
                             " We need to listen more and judge less."],
                "negative": [" The divide keeps getting worse.", " People don't listen to each other.",
                             " Both sides seem to just talk past each other."],
            },
            StudyDomain.INTERGROUP: {
                "positive": [" People are individuals first.", " Group labels don't define everyone.",
                             " I try to judge people on their own merits."],
                "negative": [" Group differences are real and sometimes hard to navigate.",
                             " It's difficult to overcome ingrained biases."],
            },
            StudyDomain.IDENTITY: {
                "positive": [" My identity shapes how I see things.", " Being true to who I am matters.",
                             " I'm proud of where I come from."],
                "negative": [" Identity shouldn't define everything.", " Labels can be limiting.",
                             " People shouldn't be put in boxes."],
            },
            StudyDomain.NORMS: {
                "positive": [" Social expectations can guide good behavior.", " Norms exist for a reason.",
                             " Following shared rules makes communities work."],
                "negative": [" Not all social norms make sense.", " Conformity isn't always right.",
                             " Some expectations are outdated."],
            },
            StudyDomain.TRUST: {
                "positive": [" I believe most people try to do the right thing.",
                             " Trust is the foundation of good relationships."],
                "negative": [" It's hard to know who to trust these days.",
                             " Trust has been eroded by too many broken promises."],
            },

            # --- Health domains ---
            StudyDomain.HEALTH: {
                "positive": [" Taking care of my health is a priority for me.",
                             " I believe prevention is better than cure."],
                "negative": [" Health decisions can be overwhelming.",
                             " I wish health information were easier to understand."],
            },

            # --- Environmental domains ---
            StudyDomain.ENVIRONMENTAL: {
                "positive": [" I think we all have a responsibility to the planet.",
                             " Small changes add up when everyone participates."],
                "negative": [" Environmental problems feel overwhelming sometimes.",
                             " It's hard to know if individual actions really make a difference."],
            },

            # --- Education domains ---
            StudyDomain.EDUCATION: {
                "positive": [" Learning new things is rewarding.", " Good teaching makes a real difference.",
                             " I value educational opportunities."],
                "negative": [" The education system has real problems.", " Not everyone has equal access to learning.",
                             " Some teaching methods just don't work."],
            },

            # --- Moral/Ethics domains ---
            StudyDomain.ETHICS: {
                "positive": [" I try to make decisions that align with my values.",
                             " Doing the right thing matters even when it's hard."],
                "negative": [" Ethical dilemmas don't have easy answers.",
                             " I struggle with situations where there's no clear right choice."],
            },

            # --- Fairness domains ---
            StudyDomain.FAIRNESS: {
                "positive": [" Everyone deserves to be treated fairly.",
                             " Fairness is a core value for me."],
                "negative": [" The system isn't fair to everyone.",
                             " Inequality really bothers me."],
            },

            # --- Cooperation domains ---
            StudyDomain.COOPERATION: {
                "positive": [" Working together usually leads to better outcomes.",
                             " Cooperation builds stronger communities."],
                "negative": [" It's hard to cooperate when others don't pull their weight.",
                             " Trust is essential for real cooperation."],
            },

            # --- Conformity domains ---
            StudyDomain.CONFORMITY: {
                "positive": [" There's value in going along with the group sometimes.",
                             " Social harmony matters."],
                "negative": [" I don't like feeling pressured to conform.",
                             " Independent thinking is important."],
            },

            # --- Self-esteem domains ---
            StudyDomain.SELF_ESTEEM: {
                "positive": [" I feel good about who I am.", " My self-confidence has grown over time."],
                "negative": [" Self-doubt is something I deal with.", " It's hard not to compare yourself to others."],
            },

            # --- Empathy domains ---
            StudyDomain.EMPATHY: {
                "positive": [" Understanding others' feelings comes naturally to me.",
                             " I try to put myself in other people's shoes."],
                "negative": [" It's emotionally draining to feel everything so deeply.",
                             " Empathy fatigue is real."],
            },

            # --- Prosocial domains ---
            StudyDomain.PROSOCIAL: {
                "positive": [" Helping others gives me genuine satisfaction.",
                             " I believe in giving back to the community."],
                "negative": [" It's frustrating when generosity isn't appreciated.",
                             " You can't help everyone, even if you want to."],
            },

            # --- Prejudice/Stereotype domains ---
            StudyDomain.PREJUDICE: {
                "positive": [" I think we've made progress on reducing prejudice.",
                             " Most people are better than their stereotypes."],
                "negative": [" Prejudice is still a major problem.", " Stereotypes do real harm to people."],
            },
            StudyDomain.STEREOTYPE: {
                "positive": [" People are more complex than any stereotype.",
                             " I try to see past initial impressions."],
                "negative": [" Stereotypes are hard to shake once formed.",
                             " Media reinforces harmful stereotypes."],
            },

            # --- Social influence domains ---
            StudyDomain.SOCIAL_INFLUENCE: {
                "positive": [" I think social influence can be a force for good.",
                             " We naturally look to others for guidance."],
                "negative": [" It's important to think for yourself.",
                             " Peer pressure can lead to bad decisions."],
            },

            # --- Attribution domains ---
            StudyDomain.ATTRIBUTION: {
                "positive": [" I try to understand why people act the way they do.",
                             " Context matters more than people think."],
                "negative": [" People are too quick to blame individuals.",
                             " We underestimate how much situations shape behavior."],
            },

            # --- Clinical/Mental Health domains ---
            StudyDomain.CLINICAL: {
                "positive": [" I believe in getting support when you need it.",
                             " Mental health awareness has come a long way."],
                "negative": [" Mental health is still misunderstood by too many people.",
                             " Getting help shouldn't be this difficult."],
            },
            StudyDomain.ANXIETY: {
                "positive": [" I've found ways to manage my anxiety that actually work.",
                             " Understanding anxiety has helped me deal with it better."],
                "negative": [" Anxiety can make everything feel overwhelming.",
                             " Living with anxiety is exhausting in ways others don't see."],
            },
            StudyDomain.DEPRESSION: {
                "positive": [" Recovery is possible and that gives me hope.",
                             " Talking openly about depression matters."],
                "negative": [" Depression makes even simple things feel impossible.",
                             " The stigma around depression is still a barrier to getting help."],
            },
            StudyDomain.COPING: {
                "positive": [" Developing good coping strategies has been worth the effort.",
                             " Everyone can build resilience with the right tools."],
                "negative": [" Sometimes coping mechanisms aren't enough.",
                             " It's hard to cope when the problems keep piling up."],
            },
            StudyDomain.STRESS: {
                "positive": [" Some stress pushes me to perform better.",
                             " I've gotten better at managing my stress over time."],
                "negative": [" Chronic stress takes a serious physical and mental toll.",
                             " The sources of stress in modern life never seem to let up."],
            },

            # --- Developmental domains ---
            StudyDomain.DEVELOPMENTAL: {
                "positive": [" Every stage of life brings new opportunities.",
                             " Growth and change are natural and mostly positive."],
                "negative": [" Life transitions are harder than people prepare you for.",
                             " Not everyone gets the support they need during key stages."],
            },
            StudyDomain.PARENTING: {
                "positive": [" Being a parent is challenging but deeply rewarding.",
                             " Children teach you as much as you teach them."],
                "negative": [" Parenting pressure is unrealistic and exhausting.",
                             " There's too much judgment of how people raise their kids."],
            },
            StudyDomain.AGING: {
                "positive": [" Age brings wisdom and perspective.", " Getting older has its own advantages."],
                "negative": [" Aging comes with real losses that society ignores.",
                             " The way older people are treated is often disrespectful."],
            },

            # --- Personality domains ---
            StudyDomain.PERSONALITY: {
                "positive": [" Understanding my personality helps me navigate the world.",
                             " I'm comfortable with who I am."],
                "negative": [" Some personality traits make life harder.",
                             " It's frustrating when your natural tendencies work against you."],
            },
            StudyDomain.NARCISSISM: {
                "positive": [" A healthy level of self-regard is important.",
                             " Confidence isn't the same as arrogance."],
                "negative": [" Dealing with narcissistic people is incredibly draining.",
                             " Our culture rewards narcissistic behavior too much."],
            },

            # --- Sports domains ---
            StudyDomain.SPORTS_PSYCHOLOGY: {
                "positive": [" Sports teach discipline and resilience.",
                             " Competition brings out my best effort."],
                "negative": [" The pressure to perform can be mentally damaging.",
                             " Sports culture has toxic elements that need addressing."],
            },
            StudyDomain.PERFORMANCE_ANXIETY: {
                "positive": [" A little nervousness actually helps me perform better.",
                             " I've learned to channel anxiety into focus."],
                "negative": [" Performance anxiety can be paralyzing.",
                             " The fear of failure sometimes holds me back."],
            },
            StudyDomain.TEAM_DYNAMICS: {
                "positive": [" Good teamwork is incredibly satisfying.",
                             " The whole is greater than the sum of its parts when a team clicks."],
                "negative": [" Team conflict can undermine everything.",
                             " Free riders on teams make it unfair for everyone else."],
            },

            # --- Legal domains ---
            StudyDomain.LEGAL_PSYCHOLOGY: {
                "positive": [" I believe in the principle of justice for all.",
                             " The legal system, while imperfect, serves an important purpose."],
                "negative": [" The justice system has serious equity problems.",
                             " Legal outcomes depend too much on who can afford a good lawyer."],
            },
            StudyDomain.JURY_DECISION: {
                "positive": [" Jury service is an important civic duty.",
                             " Ordinary people can make fair legal judgments."],
                "negative": [" Jurors are influenced by biases they don't even realize.",
                             " The jury system has significant flaws."],
            },
            StudyDomain.PROCEDURAL_JUSTICE: {
                "positive": [" Fair processes matter as much as fair outcomes.",
                             " People deserve to have their voice heard in decisions that affect them."],
                "negative": [" Procedures that look fair on paper aren't always fair in practice.",
                             " Process fairness is meaningless if the outcome is predetermined."],
            },

            # --- Food/Nutrition domains ---
            StudyDomain.FOOD_PSYCHOLOGY: {
                "positive": [" Food is one of life's great pleasures.",
                             " Making mindful food choices feels good."],
                "negative": [" My relationship with food is complicated.",
                             " Diet culture puts unhealthy pressure on people."],
            },
            StudyDomain.EATING_BEHAVIOR: {
                "positive": [" I've found an eating pattern that works for me.",
                             " Being thoughtful about what I eat has been beneficial."],
                "negative": [" Eating habits are hard to change even when you want to.",
                             " Emotional eating is a real struggle."],
            },
            StudyDomain.BODY_IMAGE: {
                "positive": [" I've learned to appreciate my body for what it can do.",
                             " Body positivity has helped my self-image."],
                "negative": [" Body image issues affect so many people.",
                             " Social media makes body image problems much worse."],
            },

            # --- Communication domains ---
            StudyDomain.COMMUNICATION: {
                "positive": [" Clear communication can solve most problems.",
                             " I value honest and direct conversation."],
                "negative": [" Miscommunication causes so many unnecessary conflicts.",
                             " People don't listen enough."],
            },
            StudyDomain.PERSUASION: {
                "positive": [" Good persuasion is about finding common ground.",
                             " I respect well-crafted arguments even when I disagree."],
                "negative": [" Manipulation disguised as persuasion is everywhere.",
                             " I'm wary of techniques designed to change my mind without my awareness."],
            },
            StudyDomain.MEDIA_EFFECTS: {
                "positive": [" Media can inform and educate when done well.",
                             " I appreciate access to diverse perspectives through media."],
                "negative": [" Media distortion of reality is a serious problem.",
                             " People are too easily influenced by what they see on screens."],
            },

            # --- Relationship domains ---
            StudyDomain.RELATIONSHIP: {
                "positive": [" Good relationships are worth investing in.",
                             " Connection with others is essential to wellbeing."],
                "negative": [" Relationships can be a source of real pain.",
                             " Past hurt makes it hard to open up to people."],
            },
            StudyDomain.ATTACHMENT: {
                "positive": [" Secure connections have been healing for me.",
                             " Feeling safe with someone makes all the difference."],
                "negative": [" Attachment patterns from childhood are hard to change.",
                             " Insecure attachment affects every relationship I have."],
            },
            StudyDomain.SOCIAL_SUPPORT: {
                "positive": [" Having people who support you makes everything easier.",
                             " I'm grateful for the support system I have."],
                "negative": [" Not everyone has the support they need.",
                             " Loneliness is a bigger problem than people acknowledge."],
            },

            # --- Cognitive domains ---
            StudyDomain.COGNITIVE: {
                "positive": [" I trust my ability to think things through.",
                             " Careful reasoning usually leads to better outcomes."],
                "negative": [" My own biases frustrate me sometimes.",
                             " Clear thinking is harder than it should be."],
            },
            StudyDomain.DECISION_MAKING: {
                "positive": [" I feel good about most decisions I make.",
                             " Taking time to decide usually pays off."],
                "negative": [" Decision paralysis is real and I struggle with it.",
                             " I second-guess myself more than I should."],
            },
            StudyDomain.MEMORY: {
                "positive": [" My memory serves me well in most situations.",
                             " I value the experiences that have shaped my memories."],
                "negative": [" Memory is less reliable than we think.",
                             " I wish I could trust my recollections more."],
            },

            # --- Financial Psychology domains ---
            StudyDomain.FINANCIAL_DECISION: {
                "positive": [" I've made some good financial choices.",
                             " Financial planning gives me a sense of control."],
                "negative": [" Financial decisions keep me up at night.",
                             " Money problems create stress in every area of life."],
            },
            StudyDomain.FINANCIAL_PSYCHOLOGY: {
                "positive": [" Financial literacy has genuinely helped me.",
                             " I feel more confident about money decisions now."],
                "negative": [" The financial system isn't designed to help regular people.",
                             " Money stress is constant and wearing."],
            },

            # --- Cross-Cultural domains ---
            StudyDomain.CROSS_CULTURAL: {
                "positive": [" Cultural diversity enriches everyone's experience.",
                             " I appreciate learning about different ways of life."],
                "negative": [" Cultural misunderstandings can be painful.",
                             " Not all cultural differences are easy to bridge."],
            },

            # --- Positive Psychology domains ---
            StudyDomain.POSITIVE_PSYCHOLOGY: {
                "positive": [" Focusing on strengths really does make a difference.",
                             " Gratitude practice has improved my outlook."],
                "negative": [" Forced positivity can be harmful.",
                             " You can't just think your way out of real problems."],
            },
            StudyDomain.GRATITUDE: {
                "positive": [" Practicing gratitude has genuinely changed how I see things.",
                             " I try to appreciate what I have."],
                "negative": [" It's hard to feel grateful when things are genuinely bad.",
                             " Gratitude can feel forced when circumstances are difficult."],
            },
            StudyDomain.RESILIENCE: {
                "positive": [" Bouncing back from adversity has made me stronger.",
                             " Resilience is a skill that can be developed."],
                "negative": [" Not everyone has the resources to be resilient.",
                             " The expectation to always bounce back is exhausting."],
            },

            # --- Gender/Sexuality domains ---
            StudyDomain.GENDER_PSYCHOLOGY: {
                "positive": [" Progress on gender equality is encouraging.",
                             " People should be free to be themselves."],
                "negative": [" Gender discrimination is still pervasive.",
                             " Rigid gender expectations harm everyone."],
            },

            # --- Neuroscience domains ---
            StudyDomain.COGNITIVE_LOAD: {
                "positive": [" I manage mental demands pretty well.",
                             " Working through complex tasks is satisfying."],
                "negative": [" Mental overload leads to poor decisions.",
                             " I can feel when my brain is at capacity."],
            },
            StudyDomain.IMPULSE_CONTROL: {
                "positive": [" I've gotten better at managing my impulses.",
                             " Self-control is worth developing."],
                "negative": [" Impulse control is harder when you're stressed or tired.",
                             " I sometimes act before thinking and regret it."],
            },

            # --- Human Factors domains ---
            StudyDomain.USER_EXPERIENCE: {
                "positive": [" Good design makes life easier.",
                             " I appreciate things that are intuitive to use."],
                "negative": [" Bad design is frustrating and wastes time.",
                             " Too many things are designed without the user in mind."],
            },
            StudyDomain.SAFETY_BEHAVIOR: {
                "positive": [" Being safety-conscious is just smart.",
                             " Prevention is always better than dealing with consequences."],
                "negative": [" Safety measures can't prevent everything.",
                             " People don't take safety seriously until something goes wrong."],
            },

            # --- Innovation domains ---
            StudyDomain.INNOVATION: {
                "positive": [" New ideas drive progress.",
                             " I'm excited by innovative approaches to problems."],
                "negative": [" Not every innovation actually improves things.",
                             " Change for its own sake isn't always good."],
            },
            StudyDomain.CREATIVITY: {
                "positive": [" Creative thinking leads to the best solutions.",
                             " I value originality and fresh perspectives."],
                "negative": [" Creativity is undervalued in most settings.",
                             " The pressure to be creative can stifle actual creativity."],
            },

            # --- Risk/Safety domains ---
            StudyDomain.RISK_PERCEPTION: {
                "positive": [" Understanding risks helps me make better choices.",
                             " I'm reasonably comfortable with calculated risks."],
                "negative": [" Some risks are genuinely frightening.",
                             " People systematically misjudge how risky things are."],
            },

            # --- Social Media domains ---
            StudyDomain.SOCIAL_MEDIA: {
                "positive": [" Social media keeps me connected to people.",
                             " Online platforms have genuine community value."],
                "negative": [" Social media is designed to be addictive.",
                             " The comparison culture online is toxic."],
            },

            # --- Negotiation domains ---
            StudyDomain.NEGOTIATION: {
                "positive": [" Good negotiation benefits everyone involved.",
                             " I'm a fair negotiator and that gets results."],
                "negative": [" Negotiations can feel adversarial and stressful.",
                             " Power imbalances make fair negotiation nearly impossible."],
            },

            # --- Gambling domain ---
            StudyDomain.GAMBLING: {
                "positive": [" A little risk-taking can be fun in the right context.",
                             " I know my limits when it comes to betting."],
                "negative": [" Gambling can easily get out of control.",
                             " The industry preys on people's vulnerabilities."],
            },

            # --- Remote Work domain ---
            StudyDomain.REMOTE_WORK: {
                "positive": [" Remote work has improved my quality of life.",
                             " The flexibility of working from home is valuable."],
                "negative": [" Working remotely can be isolating.",
                             " The boundary between work and personal life disappears."],
            },

            # --- Burnout domain ---
            StudyDomain.BURNOUT: {
                "positive": [" Recognizing burnout early helps you address it.",
                             " Setting boundaries has helped me avoid burning out."],
                "negative": [" Burnout is real and workplaces don't take it seriously.",
                             " The pace of modern work is unsustainable for most people."],
            },
        }

        # v1.0.8.0: More substantive general extensions — EXPANDED with more variety
        general_extensions = {
            "very_positive": [
                " I feel strongly about this and wanted to be clear.",
                " This really matters to me personally.",
                " I've thought about this a lot.",
                " I'm genuinely enthusiastic about this.",
                " I can't say enough good things about where this is headed.",
                " My positive feelings here are very strong and very real.",
            ],
            "positive": [
                " That's where I stand on this.",
                " Those are my honest thoughts.",
                " I tried to be clear about my views.",
                " I feel good about where I land on this.",
                " My experience has been positive and that shaped my answer.",
                " I think my perspective is fair and grounded.",
            ],
            "neutral": [
                " I could see arguments either way.",
                " It's more nuanced than a simple answer.",
                " I don't have a strong lean here.",
                " I'm still sorting out how I feel about it.",
                " There's merit in multiple perspectives on this.",
                " I try to stay balanced when thinking about things like this.",
            ],
            "negative": [
                " I think things need to change.",
                " I wish the situation were different.",
                " There are real problems here.",
                " My concerns are genuine and based on what I've seen.",
                " I can't pretend this is working when it clearly isn't.",
                " Something needs to be done about this.",
            ],
            "very_negative": [
                " This is genuinely concerning to me.",
                " I feel strongly that something is wrong.",
                " I couldn't hold back my frustration.",
                " This is one of those things that really gets under my skin.",
                " I have serious problems with this and I'm not going to sugarcoat it.",
                " My negative feelings about this are deeply felt.",
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
    'COMPOUND_DOMAIN_PHRASES',
    'DOMAIN_NEGATIVE_KEYWORDS',
    'DOMAIN_CATEGORIES',
]
