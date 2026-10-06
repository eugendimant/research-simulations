"""Built-in AI must actually attempt LLM generation.

Regression guard for the v1.2.9.1 -> v1.2.9.4 defect. v1.2.9.1 set
`allow_template_fallback = True` for the Built-in AI method so the free path
could never dead-end. But the engine's two LLM gates read
"allow_template_fallback and free_llm_oe_cap == 0" as *do not try the LLM at
all* — the meaning the Template/ABE methods give that flag. The OE cap was only
set when N > MAX_FREE_LLM_N, so:

    N <= 100  ->  cap 0  ->  zero API calls, all text from the built-in engine
    N >  100  ->  cap 100 ->  LLM attempted

Built-in AI therefore made no LLM call at all in the common case, silently,
while working for large runs. These tests pin the invariant from both ends: the
engine predicate, and the flag combination the app sends for each method.
"""

import pytest

from utils.enhanced_simulation_engine import EnhancedSimulationEngine

MAX_FREE_LLM_N = 100  # mirrors app.py


def _engine(**kwargs):
    """Minimal engine; only the two LLM flags matter for these tests."""
    return EnhancedSimulationEngine(
        study_title="T", study_description="D", sample_size=10,
        conditions=["a", "b"], factors={}, scales=[], additional_vars=[],
        demographics=[], **kwargs
    )


# The flag combinations app.py sends, per generation method. "free_llm" is
# Built-in AI; it wants the LLM tried AND graceful fallback.
_METHOD_FLAGS = {
    "template":   {"allow_template_fallback": True,  "cap_is_set": False},
    "abe_v2":     {"allow_template_fallback": True,  "cap_is_set": False},
    "experimental": {"allow_template_fallback": True, "cap_is_set": False},
    "own_api":    {"allow_template_fallback": False, "cap_is_set": False},
    "free_llm":   {"allow_template_fallback": True,  "cap_is_set": True},
}

_SHOULD_TRY_LLM = {"template": False, "abe_v2": False, "experimental": False,
                   "own_api": True, "free_llm": True}


@pytest.mark.parametrize("method", sorted(_METHOD_FLAGS))
def test_each_method_gets_the_llm_behaviour_it_asks_for(method):
    flags = _METHOD_FLAGS[method]
    engine = _engine(
        allow_template_fallback=flags["allow_template_fallback"],
        free_llm_oe_cap=MAX_FREE_LLM_N if flags["cap_is_set"] else 0,
    )
    assert engine.llm_attempts_allowed() is _SHOULD_TRY_LLM[method], (
        "method %r: llm_attempts_allowed() is %s, expected %s"
        % (method, engine.llm_attempts_allowed(), _SHOULD_TRY_LLM[method])
    )


@pytest.mark.parametrize("n", [1, 10, 20, 99, 100, 101, 500, 10000])
def test_builtin_ai_attempts_the_llm_at_every_sample_size(n):
    """The exact defect: this failed for every n <= 100 before v1.2.9.4."""
    # What app.py now computes for free_llm, for any N.
    cap = MAX_FREE_LLM_N
    engine = _engine(allow_template_fallback=True, free_llm_oe_cap=cap)
    assert engine.llm_attempts_allowed() is True, (
        "Built-in AI would make zero LLM calls at N=%d" % n
    )


def test_graceful_fallback_is_still_enabled_for_builtin_ai():
    """Trying the LLM must not come at the cost of the no-dead-end guarantee."""
    engine = _engine(allow_template_fallback=True,
                     free_llm_oe_cap=MAX_FREE_LLM_N)
    assert engine.allow_template_fallback is True
    assert engine.llm_attempts_allowed() is True


def test_template_engine_still_skips_the_llm_entirely():
    """The other meaning of the flag must keep working: no wasted API calls."""
    engine = _engine(allow_template_fallback=True, free_llm_oe_cap=0)
    assert engine.llm_attempts_allowed() is False


def test_cap_is_an_upper_bound_not_a_floor():
    """Setting the cap for small N must not cap anything that matters."""
    engine = _engine(allow_template_fallback=True,
                     free_llm_oe_cap=MAX_FREE_LLM_N)
    # A 20-participant run is far below the cap, so no participant is ever
    # switched to template text by the cap itself.
    assert engine.free_llm_oe_cap >= 20
