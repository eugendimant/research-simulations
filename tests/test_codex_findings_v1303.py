"""Two findings from the Codex review of PR #365.

Both are regressions a reader would not catch by eye: one sends a credential to
the wrong vendor, the other puts a published norm below the scale it is being
mapped onto. Each is pinned here so neither can come back quietly.
"""
from __future__ import annotations

import pytest

from utils import llm_response_generator as lrg
from utils.scientific_knowledge_base import CONSTRUCT_NORMS, get_construct_norm


# ---------------------------------------------------------------------------
# A removed provider's key must never be forwarded to a different vendor
# ---------------------------------------------------------------------------

_RETIRED_KEYS = [
    ("Cerebras", "csk-" + "c" * 44),
    ("Mistral AI", "m" * 32),
]


@pytest.mark.parametrize("provider,key", _RETIRED_KEYS)
def test_a_removed_providers_key_is_named_not_guessed(provider, key):
    assert lrg.retired_provider_for_key(key) == provider


@pytest.mark.parametrize("provider,key", _RETIRED_KEYS)
def test_a_removed_providers_key_is_never_routed_to_another_vendor(provider, key):
    """Deleting a detection branch must not mean the key falls to the default.

    Cerebras and Mistral left the chain in v1.3.0.0. Their key shapes did not
    stop existing: a user who still has one pastes it into "AI (your API key)".
    The generic ">30 characters means Groq" default would then send that
    credential to Groq as a Bearer token — the request fails, and the key is
    disclosed to a vendor it was never issued for.
    """
    assert lrg.detect_provider_from_key(key) is None, (
        "%s key was routed somewhere" % provider
    )


@pytest.mark.parametrize("provider,key", _RETIRED_KEYS)
def test_a_removed_providers_key_builds_no_provider(monkeypatch, provider, key):
    """Not even as the un-detected-key fallback, which also defaults to Groq."""
    for names in lrg.BUILTIN_PROVIDER_SECRETS.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    for const in ("_DEFAULT_GROQ_KEY", "_DEFAULT_GOOGLE_AI_KEY",
                  "_DEFAULT_OPENROUTER_KEY", "_DEFAULT_SAMBANOVA_KEY",
                  "_DEFAULT_API_KEY"):
        monkeypatch.setattr(lrg, const, "", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER_HINT", raising=False)

    gen = lrg.LLMResponseGenerator(api_key=key)

    assert all(p.api_key != key for p in gen._providers), (
        "%s key reached the chain as %r" % (
            provider, [p.name for p in gen._providers if p.api_key == key])
    )
    # And the runtime path refuses it too.
    assert gen.add_runtime_provider(key) is False


def test_a_supported_key_still_routes_normally():
    """The guard must not swallow the keys that do work."""
    for key, expected in [
        ("gsk_" + "g" * 40, "groq"),
        ("sk-or-v1-" + "0" * 48, "openrouter"),
        ("AIzaSy" + "A" * 33, "google_ai"),
        ("AQ." + "A" * 48, "google_ai"),
        ("snova-" + "s" * 36, "sambanova"),
    ]:
        detected = lrg.detect_provider_from_key(key)
        assert detected is not None and detected["name"] == expected, key


# ---------------------------------------------------------------------------
# A published norm must land inside the scale it is rescaled onto
# ---------------------------------------------------------------------------

def test_every_published_mean_sits_inside_its_own_declared_range():
    """A mean outside its own scale means the entry's origin is mis-declared."""
    bad = []
    for key, norm in CONSTRUCT_NORMS.items():
        low = norm.scale_min
        high = norm.scale_min + norm.scale_points - 1
        if not low <= norm.mean <= high:
            bad.append((key, norm.mean, low, high))
    assert not bad, "norm mean outside its own scale: %s" % bad[:5]


@pytest.mark.parametrize("target_points", [5, 7, 9])
def test_no_norm_rescales_outside_the_target_scale(target_points):
    """The failure Codex caught, stated generally.

    Rescaling by a ratio of maxima has no notion of where a scale starts, so
    every zero-based instrument (a forced-choice proportion like the NPI, or a
    0-3 per-item mean like the PHQ-9) landed below the target scale's own
    minimum — an impossible value the calibration then acted on.
    """
    bad = []
    for key in CONSTRUCT_NORMS:
        result = get_construct_norm(key, target_scale_points=target_points)
        if result is None:
            continue
        if not 1.0 <= result["mean"] <= float(target_points):
            bad.append((key, round(result["mean"], 3)))
    assert not bad, "rescaled outside 1..%d: %s" % (target_points, bad[:5])


def test_the_npi_rescales_to_its_arithmetically_correct_place():
    """0.388 of NPI-40 items endorsed is a 0-1 proportion, not a point on 1-2.

    On a 1-7 scale that is 1 + 0.388 * 6 = 3.328, slightly below the midpoint.
    The old ratio form gave 0.388 * 7/2 = 1.358, which is near the floor and
    made automatic narcissism calibration push simulated scores far too low.
    """
    result = get_construct_norm("narcissism_npi", target_scale_points=7)
    assert result is not None
    assert result["mean"] == pytest.approx(3.328, abs=1e-3)


def test_an_ordinary_likert_norm_maps_endpoint_to_endpoint():
    """A 1-k scale is unaffected in kind: position within the range is kept."""
    from utils.scientific_knowledge_base import ConstructNorm

    norm = ConstructNorm(source="synthetic", construct="x", scale_name="x",
                         scale_points=5, mean=3.0, sd=1.0)
    CONSTRUCT_NORMS["__test_midpoint__"] = norm
    try:
        result = get_construct_norm("__test_midpoint__", target_scale_points=7)
        # Midpoint of 1..5 is the midpoint of 1..7.
        assert result["mean"] == pytest.approx(4.0)
        # An SD is a width: it scales by the span ratio (6/4), not by 7/5.
        assert result["sd"] == pytest.approx(1.5)
    finally:
        CONSTRUCT_NORMS.pop("__test_midpoint__", None)
