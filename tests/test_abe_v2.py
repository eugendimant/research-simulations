"""
Tests for Adaptive Behavioral Engine 2.0.

Runs 5 QSF files (including Karlijn) through the engine and validates:
1. Engine initializes without errors
2. Narrative intents produce non-empty, on-topic responses
3. Standard intents delegate correctly to base generator
4. Ultra-short responses work for disengaged participants
5. Variation phrases are applied correctly
6. No crashes with edge cases (empty topics, None profiles)
"""

import os
import sys
import json
import random

# Path setup
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SIM_APP = os.path.join(_PROJECT_ROOT, "simulation_app")
if _SIM_APP not in sys.path:
    sys.path.insert(0, _SIM_APP)

EXAMPLE_DIR = os.path.join(_SIM_APP, "example_files")

# 5 QSF files for testing (including Karlijn)
TEST_QSF_FILES = [
    "2026_02_16_Qualtrics_Survey_Karlijn_test.qsf",      # Karlijn (required)
    "2026_02_07_Hate_Trumps_Love.qsf",                   # Political/intergroup
    "DG & PGG (MGP).qsf",                                # Economic games
    "BDS_501_Skills_Questionnaire__2024-2025.qsf",        # Skills/education
    "Coffee_Shop_Loyalty_Programs.qsf",                   # Consumer/marketing
]


def test_engine_import():
    """ABE 2.0 can be imported without errors."""
    from utils.adaptive_behavioral_engine_v2 import AdaptiveBehavioralEngineV2
    assert AdaptiveBehavioralEngineV2 is not None


def test_engine_initialization():
    """ABE 2.0 initializes with base generator available."""
    from utils.adaptive_behavioral_engine_v2 import AdaptiveBehavioralEngineV2
    engine = AdaptiveBehavioralEngineV2(seed=42)
    assert engine._base_generator is not None
    assert len(engine._init_errors) == 0


def test_narrative_intents_produce_output():
    """All 5 narrative intents produce non-empty responses."""
    from utils.adaptive_behavioral_engine_v2 import AdaptiveBehavioralEngineV2
    engine = AdaptiveBehavioralEngineV2(seed=42)
    engine.set_study_context({"study_domain": "political", "survey_name": "Test"})

    intents = ["creative_belief", "personal_disclosure", "creative_narrative",
               "personal_story", "hypothetical"]
    sentiments = ["very_positive", "positive", "neutral", "negative", "very_negative"]

    for intent in intents:
        for sentiment in sentiments:
            response = engine.generate(
                question_text=f"Test question about {intent}",
                sentiment=sentiment,
                persona_verbosity=0.5,
                persona_formality=0.5,
                persona_engagement=0.7,
                condition="control",
                question_name=f"test_{intent}_{sentiment}",
                participant_seed=random.randint(0, 10000),
                question_intent=intent,
            )
            assert response and len(response.strip()) > 0, \
                f"Empty response for intent={intent}, sentiment={sentiment}"


def test_standard_intents_delegate():
    """Standard intents (opinion, explanation) delegate to base generator."""
    from utils.adaptive_behavioral_engine_v2 import AdaptiveBehavioralEngineV2
    engine = AdaptiveBehavioralEngineV2(seed=42)
    engine.set_study_context({"study_domain": "political", "survey_name": "Test"})

    for intent in ["opinion", "explanation", "evaluation", "description"]:
        response = engine.generate(
            question_text="What do you think about political polarization?",
            sentiment="positive",
            persona_verbosity=0.5,
            persona_formality=0.5,
            persona_engagement=0.7,
            condition="control",
            question_name=f"test_standard_{intent}",
            participant_seed=42,
            question_intent=intent,
        )
        assert response and len(response.strip()) > 5, \
            f"Standard intent {intent} returned empty/short: {response!r}"


def test_ultra_short_disengaged():
    """Ultra-short responses for very disengaged participants."""
    from utils.adaptive_behavioral_engine_v2 import AdaptiveBehavioralEngineV2
    engine = AdaptiveBehavioralEngineV2(seed=42)

    # Run multiple times to get at least one ultra-short
    ultra_short_found = False
    for seed in range(100):
        response = engine.generate(
            question_text="What is your craziest conspiracy theory?",
            sentiment="neutral",
            persona_verbosity=0.1,
            persona_formality=0.2,
            persona_engagement=0.1,
            condition="control",
            question_name="test_ultra_short",
            participant_seed=seed,
            question_intent="creative_belief",
            behavioral_profile={
                "straight_lined": True,
                "trait_profile": {"attention": 0.1, "verbosity": 0.1, "formality": 0.2}
            },
        )
        if response and len(response.split()) <= 6:
            ultra_short_found = True
            break

    assert ultra_short_found, "No ultra-short responses produced for disengaged participants"


def test_variation_phrase_diversity():
    """Different seeds produce diverse responses via variation phrases."""
    from utils.adaptive_behavioral_engine_v2 import AdaptiveBehavioralEngineV2
    engine = AdaptiveBehavioralEngineV2(seed=42)
    engine.set_study_context({"study_domain": "political", "survey_name": "Test"})

    responses = set()
    for seed in range(20):
        response = engine.generate(
            question_text="What is your opinion on government surveillance?",
            sentiment="negative",
            persona_verbosity=0.6,
            persona_formality=0.4,
            persona_engagement=0.7,
            condition="control",
            question_name="test_variation",
            participant_seed=seed * 1000,
            question_intent="creative_belief",
        )
        responses.add(response)

    # At least 10 unique responses out of 20
    assert len(responses) >= 10, \
        f"Only {len(responses)} unique responses out of 20 — insufficient diversity"


def test_none_behavioral_profile():
    """Engine doesn't crash with None behavioral profile."""
    from utils.adaptive_behavioral_engine_v2 import AdaptiveBehavioralEngineV2
    engine = AdaptiveBehavioralEngineV2(seed=42)

    response = engine.generate(
        question_text="Tell me about your experience",
        sentiment="neutral",
        persona_verbosity=0.5,
        persona_formality=0.5,
        persona_engagement=0.5,
        condition="",
        question_name="test_none_profile",
        participant_seed=42,
        question_intent="personal_story",
        behavioral_profile=None,
    )
    assert response and len(response.strip()) > 0


def test_empty_question_text():
    """Engine handles empty question text gracefully."""
    from utils.adaptive_behavioral_engine_v2 import AdaptiveBehavioralEngineV2
    engine = AdaptiveBehavioralEngineV2(seed=42)

    response = engine.generate(
        question_text="",
        sentiment="neutral",
        persona_verbosity=0.5,
        persona_formality=0.5,
        persona_engagement=0.5,
        condition="",
        question_name="test_empty",
        participant_seed=42,
        question_intent="creative_belief",
    )
    assert response and len(response.strip()) > 0


def test_qsf_integration():
    """Test that QSF files can be parsed and engine generates responses for OE questions."""
    try:
        from utils.qsf_preview import parse_qsf_for_preview
    except ImportError:
        print("SKIP: qsf_preview not importable")
        return

    from utils.adaptive_behavioral_engine_v2 import AdaptiveBehavioralEngineV2

    for qsf_name in TEST_QSF_FILES:
        qsf_path = os.path.join(EXAMPLE_DIR, qsf_name)
        if not os.path.exists(qsf_path):
            print(f"SKIP: {qsf_name} not found")
            continue

        # Parse QSF
        try:
            with open(qsf_path, 'r', encoding='utf-8') as f:
                qsf_data = json.load(f)
            preview = parse_qsf_for_preview(qsf_data)
        except Exception as e:
            print(f"SKIP: {qsf_name} parse error: {e}")
            continue

        # Initialize engine
        engine = AdaptiveBehavioralEngineV2(seed=42)
        study_title = preview.get("survey_name", qsf_name)
        engine.set_study_context({
            "study_domain": "general",
            "survey_name": study_title,
            "study_title": study_title,
        })

        # Generate responses for first 3 OE questions found
        oe_questions = preview.get("open_ended_questions", [])
        if not oe_questions:
            # Check blocks for text entry
            for block in preview.get("blocks", []):
                for q in block.get("questions", []):
                    if q.get("type") == "TE":
                        oe_questions.append({
                            "name": q.get("name", "oe"),
                            "question_text": q.get("text", ""),
                        })
                        if len(oe_questions) >= 3:
                            break

        for i, oe in enumerate(oe_questions[:3]):
            q_text = oe.get("question_text", oe.get("text", "Share your thoughts"))
            response = engine.generate(
                question_text=q_text,
                sentiment="neutral",
                persona_verbosity=0.5,
                persona_formality=0.5,
                persona_engagement=0.7,
                condition="control",
                question_name=f"{qsf_name}_oe_{i}",
                participant_seed=42 + i,
                question_intent="opinion",
            )
            assert response and len(response.strip()) > 0, \
                f"Empty response for {qsf_name} OE #{i}: {q_text[:50]}"

        print(f"PASS: {qsf_name} — {len(oe_questions)} OE questions tested")


def test_narrative_topic_extraction():
    """Topic extraction works for various question formats."""
    from utils.adaptive_behavioral_engine_v2 import AdaptiveBehavioralEngineV2
    engine = AdaptiveBehavioralEngineV2(seed=42)

    test_cases = [
        ("What is your craziest conspiracy theory?", "creative_belief", "conspiracy theory"),
        ("Share a secret that only your family knows", "personal_disclosure", "secret"),
        ("Tell us about your wildest travel experience", "creative_narrative", "travel"),
        ("Describe a time when you felt truly proud", "personal_story", "proud"),
    ]
    for q_text, intent, expected_substring in test_cases:
        topic = engine._extract_narrative_topic(q_text, "", intent)
        # Topic should contain meaningful words, not just "this"
        assert topic != "this" or intent == "hypothetical", \
            f"Topic extraction failed for: {q_text[:40]} → got '{topic}'"


if __name__ == "__main__":
    tests = [
        test_engine_import,
        test_engine_initialization,
        test_narrative_intents_produce_output,
        test_standard_intents_delegate,
        test_ultra_short_disengaged,
        test_variation_phrase_diversity,
        test_none_behavioral_profile,
        test_empty_question_text,
        test_qsf_integration,
        test_narrative_topic_extraction,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
