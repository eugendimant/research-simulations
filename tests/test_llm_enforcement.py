import pytest

from simulation_app.utils.enhanced_simulation_engine import EnhancedSimulationEngine
from simulation_app.utils.llm_response_generator import (
    LLMResponseGenerator,
    _is_low_quality_response,
)


def _basic_engine(**overrides):
    kwargs = dict(
        study_title="Test Study",
        study_description="Test description",
        sample_size=24,
        conditions=["No AI x Utilitarian", "AI x Hedonic", "AI x Utilitarian", "No AI x Hedonic"],
        factors=[],
        scales=[{"name": "Main_DV", "variable_name": "Main_DV", "num_items": 1, "scale_points": 7}],
        additional_vars=[],
        demographics={"age_mean": 35, "age_sd": 10, "gender_quota": 50},
        open_ended_questions=[{"name": "Q_OE", "question_text": "Why?", "type": "text"}],
        seed=123,
    )
    kwargs.update(overrides)
    return EnhancedSimulationEngine(**kwargs)


def test_llm_generator_blocks_template_fallback_when_disabled():
    gen = LLMResponseGenerator(seed=1, allow_template_fallback=False)
    gen._providers = []
    gen._api_available = False

    with pytest.raises(RuntimeError, match="template fallback is disabled"):
        gen.generate(question_text="Why?", condition="A", participant_seed=42)


def test_open_ended_dedup_does_not_mutate_condition_column(monkeypatch):
    engine = _basic_engine(allow_template_fallback=True)
    engine.llm_generator = None

    def _same_response(*args, **kwargs):
        return "This is intentionally duplicated text for dedup testing."

    monkeypatch.setattr(engine, "_generate_open_response", _same_response)

    df, _ = engine.generate()

    observed = set(df["CONDITION"].unique().tolist())
    expected = set(engine.conditions)
    assert observed.issubset(expected)
    assert len(observed) == len(expected)


def test_google_ai_is_prioritized_first_in_builtin_provider_chain():
    gen = LLMResponseGenerator(seed=2, allow_template_fallback=False)
    assert gen._providers, "Expected at least one provider in chain"
    assert gen._providers[0].name.startswith("google_ai")


def test_quality_filter_flags_generic_gibberish():
    bad = "In my estimation, the enjoyment factor was notable and practical aspects mattered."
    assert _is_low_quality_response(bad, topic_tokens=["trump", "politics"])


def test_engine_init_does_not_fail_with_missing_os_regression():
    engine = _basic_engine(allow_template_fallback=False)
    joined_log = "\n".join(engine.validation_log)
    assert "name 'os' is not defined" not in joined_log
