import pytest

from simulation_app.utils.enhanced_simulation_engine import EnhancedSimulationEngine
from simulation_app.utils.llm_response_generator import LLMResponseGenerator


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

    def _same_response(*args, **kwargs):
        return "This is intentionally duplicated text for dedup testing."

    monkeypatch.setattr(engine, "_generate_open_response", _same_response)

    df, _ = engine.generate()

    observed = set(df["CONDITION"].unique().tolist())
    expected = set(engine.conditions)
    assert observed.issubset(expected)
    assert len(observed) == len(expected)
