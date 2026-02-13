import pytest
import simulation_app.utils.llm_response_generator as llm_mod

from simulation_app.utils.enhanced_simulation_engine import EnhancedSimulationEngine
from simulation_app.utils.llm_response_generator import (
    LLMResponseGenerator,
    _is_low_quality_response,
    _extract_topic_tokens,
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


def test_metadata_includes_llm_init_error_field():
    engine = _basic_engine(allow_template_fallback=False)
    engine.llm_generator = None
    engine.llm_init_error = "forced test error"
    df, metadata = engine.generate()
    assert len(df) == 24
    assert metadata.get("llm_init_error") == "forced test error"


def test_open_ended_visible_questions_are_never_blank(monkeypatch):
    engine = _basic_engine(allow_template_fallback=True)
    engine.llm_generator = None

    monkeypatch.setattr(engine.survey_flow_handler, "is_question_visible", lambda *_args, **_kwargs: True)

    df, _ = engine.generate()
    oe_col = [c for c in df.columns if c.lower().startswith("q_oe") or c.lower().endswith("q_oe") or c == "Q_OE"]
    assert oe_col, "Expected OE column to exist"
    col = oe_col[0]
    assert (df[col].astype(str).str.strip() != "").all()


def test_llm_generator_returns_non_empty_when_fallback_enabled_and_providers_down():
    gen = LLMResponseGenerator(seed=7, allow_template_fallback=True)
    gen._providers = []
    gen._api_available = False
    out = gen.generate(question_text="Explain your view", condition="A", participant_seed=99)
    assert isinstance(out, str)
    assert out.strip() != ""


def test_llm_emergency_fallback_is_contextual_and_non_empty():
    gen = LLMResponseGenerator(seed=11, allow_template_fallback=True)
    gen._providers = []
    gen._api_available = False
    gen._fallback = None
    out = gen.generate(
        question_text="How did the ad change your trust in the product?",
        condition="High transparency",
        sentiment="negative",
        participant_seed=101,
    )
    assert out.strip() != ""
    assert "trust" in out.lower() or "product" in out.lower() or "transparency" in out.lower()


def test_prefill_pool_respects_runtime_budget(monkeypatch):
    gen = LLMResponseGenerator(seed=3, allow_template_fallback=True)
    gen._api_available = True

    # Force time jump so budget check trips immediately inside prefill loop.
    _t = {"v": 0.0}
    def _fake_time():
        _t["v"] += 30.0
        return _t["v"]
    monkeypatch.setattr(llm_mod.time, "time", _fake_time)
    monkeypatch.setattr(gen, "_generate_batch", lambda *args, **kwargs: ["synthetic response"])

    out = gen.prefill_pool(
        question_text="Why did you choose this option?",
        condition="A",
        sentiments=["positive"],
        count_per_sentiment=30,
    )
    assert out < 30
    assert gen.stats.get("prefill_timeouts", 0) >= 1


def test_topic_tokens_include_study_context_terms():
    tokens = _extract_topic_tokens(
        question_text="How did the ad change your trust?",
        condition="High transparency",
        study_title="AI Pricing Experiment",
        study_description="Consumer behavior around product trust and fairness",
    )
    joined = " ".join(tokens)
    assert "transparency" in joined or "pricing" in joined or "fairness" in joined


def test_pool_defaults_are_runtime_bounded():
    gen = LLMResponseGenerator(seed=5)
    assert gen.MIN_POOL_PER_BUCKET <= 20
    assert gen.MAX_POOL_PER_BUCKET <= 60
