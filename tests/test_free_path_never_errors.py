"""The free (Built-in AI) path must always produce data, never an error.

Eugen's requirement: "I want people to be able to use the free version too,
without errors." With no provider key configured the app used to disable the
generate button and show an error banner. It must instead fall through to the
built-in non-LLM text cascade and deliver a complete dataset.
"""

import pytest

from utils import llm_response_generator as lrg
from utils.enhanced_simulation_engine import EnhancedSimulationEngine
from utils.llm_response_generator import BUILTIN_PROVIDER_SECRETS

_ALL_KEY_VARS = sorted(
    {name for names in BUILTIN_PROVIDER_SECRETS.values() for name in names}
    | {"LLM_API_KEY"}
)


@pytest.fixture
def no_keys(monkeypatch):
    for var in _ALL_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    for const in ("_DEFAULT_GROQ_KEY", "_DEFAULT_CEREBRAS_KEY",
                  "_DEFAULT_GOOGLE_AI_KEY", "_DEFAULT_OPENROUTER_KEY",
                  "_DEFAULT_MISTRAL_KEY", "_DEFAULT_SAMBANOVA_KEY",
                  "_DEFAULT_API_KEY"):
        monkeypatch.setattr(lrg, const, "", raising=False)
    return None


def _engine(**overrides):
    kwargs = dict(
        study_title="Trust and cooperation",
        study_description="A trust game study on cooperation between strangers",
        sample_size=12,
        conditions=["Control", "Treatment"],
        factors=[],
        scales=[{"name": "Trust", "type": "likert", "items": 3,
                 "scale_min": 1, "scale_max": 7}],
        additional_vars=[],
        demographics={},
        open_ended_questions=[{
            "name": "why_trust",
            "question_text": "Why do you trust or distrust the other player?",
        }],
        # This is what app.py now passes for the "free_llm" method.
        allow_template_fallback=True,
        use_socsim_experimental=True,
        seed=42,
    )
    kwargs.update(overrides)
    return EnhancedSimulationEngine(**kwargs)


def test_free_path_produces_complete_data_without_any_key(no_keys):
    df, _meta = _engine().generate()

    assert len(df) == 12
    oe_cols = [c for c in df.columns if "why_trust" in c.lower()]
    assert oe_cols, "the open-ended column is missing entirely"

    answers = df[oe_cols[0]].dropna().astype(str).str.strip()
    assert len(answers) == 12, "some participants have no open-ended response"
    assert (answers != "").all(), "blank open-ended responses were produced"

    # Responses must be on topic, not placeholder filler (CLAUDE.md: no response
    # should ever be off-topic).
    joined = " ".join(answers).lower()
    assert any(word in joined for word in ("trust", "distrust", "player")), (
        "fallback responses are off-topic: %s" % answers.head(3).tolist()
    )


def test_free_path_does_not_raise_llm_exhausted(no_keys):
    """allow_template_fallback=True must suppress LLMExhaustedMidGeneration."""
    from utils.enhanced_simulation_engine import LLMExhaustedMidGeneration

    engine = _engine()
    assert engine.allow_template_fallback is True
    try:
        engine.generate()
    except LLMExhaustedMidGeneration as exc:  # pragma: no cover
        pytest.fail("free path raised LLMExhaustedMidGeneration: %s" % exc)
