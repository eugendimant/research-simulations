"""Built-in LLM provider configuration from deployment secrets.

Regression guard for the v1.2.9.0 → v1.2.9.1 defect: the committed free-tier
keys were (correctly) removed in v1.2.9.0, but with no keys in the environment
the app still told users "Free AI providers are currently not responding" and
to try again in a few hours. The real state was "no key is configured", which
no amount of waiting fixes.

These tests pin down three things:
  1. With no keys set, the provider chain is empty and both status calls report
     reason == "not_configured" together with the documented secret names.
  2. With a key set in the environment, the matching provider slot is built —
     i.e. a configured deployment actually picks its keys up, including when
     the key becomes readable after this module was first imported.
  3. The documented secret names stay in sync with the provider slots.
"""

import os
import re
import types

import pytest

from utils import llm_response_generator as lrg
from utils.llm_response_generator import (
    BUILTIN_PROVIDER_SECRETS,
    LLMResponseGenerator,
    builtin_provider_key_status,
    missing_builtin_provider_secrets,
)

# Every env var that can feed a built-in provider slot, plus the user-key vars
# the constructor also consults.
_ALL_KEY_VARS = sorted(
    {name for names in BUILTIN_PROVIDER_SECRETS.values() for name in names}
    | {"LLM_API_KEY", "LLM_PROVIDER_HINT"}
)


@pytest.fixture
def no_keys(monkeypatch):
    """Simulate a deployment with no provider secrets configured at all."""
    for var in _ALL_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    # The module-level constants are resolved at import time; a developer
    # machine with keys exported would otherwise leak them into the test.
    for const in ("_DEFAULT_GROQ_KEY", "_DEFAULT_CEREBRAS_KEY",
                  "_DEFAULT_GOOGLE_AI_KEY", "_DEFAULT_OPENROUTER_KEY",
                  "_DEFAULT_MISTRAL_KEY", "_DEFAULT_SAMBANOVA_KEY",
                  "_DEFAULT_API_KEY"):
        monkeypatch.setattr(lrg, const, "", raising=False)
    # The deployment owner may add utils/builtin_free_keys.py. Neutralise it so
    # "no keys configured" keeps that meaning whether or not the file exists.
    monkeypatch.setattr(lrg, "_import_bundled_keys", lambda: None)
    return None


def test_no_keys_yields_empty_provider_chain(no_keys):
    gen = LLMResponseGenerator()
    assert gen._providers == []
    assert gen.is_llm_available is False


def test_no_keys_reports_not_configured_not_unreachable(no_keys):
    """The UI branches on `reason`; it must not say 'not responding' here."""
    gen = LLMResponseGenerator()

    status = gen.check_connectivity(timeout=1)
    assert status["available"] is False
    assert status["reason"] == "not_configured"
    assert "GROQ_API_KEY" in status["missing_secrets"]

    health = gen.health_check(timeout=1)
    assert health["ok"] is False
    assert health["reason"] == "not_configured"
    assert "GROQ_API_KEY" in health["missing_secrets"]


def test_missing_secrets_lists_every_slot_when_nothing_is_set(no_keys):
    assert builtin_provider_key_status() == {
        slot: False for slot in BUILTIN_PROVIDER_SECRETS
    }
    assert sorted(missing_builtin_provider_secrets()) == sorted(
        names[0] for names in BUILTIN_PROVIDER_SECRETS.values()
    )


@pytest.mark.parametrize(
    ("slot", "env_var"),
    [(slot, names[0]) for slot, names in BUILTIN_PROVIDER_SECRETS.items()],
)
def test_each_documented_secret_builds_its_provider(no_keys, monkeypatch, slot, env_var):
    """A key set in the environment must reach the provider chain.

    Keys are resolved per instance, so one set *after* import still counts —
    that is what makes `st.secrets` on Streamlit Cloud work.
    """
    monkeypatch.setenv(env_var, "test-key-%s" % slot)

    assert builtin_provider_key_status()[slot] is True
    assert env_var not in missing_builtin_provider_secrets()

    gen = LLMResponseGenerator()
    assert gen._providers, "a configured key must produce at least one provider"
    assert any(p.name.startswith(slot) for p in gen._providers), (
        "%s did not build a provider for slot %s; chain was %s"
        % (env_var, slot, [p.name for p in gen._providers])
    )

    # With a key present the status is "unreachable" at worst, never
    # "not_configured" — we must not tell an admin to set a key they already
    # set. Stub the transport so the assertion needs no network.
    monkeypatch.setattr(lrg, "_call_llm_api", lambda *a, **k: None)
    assert gen.health_check(timeout=1)["reason"] == "unreachable"
    assert gen.check_connectivity(timeout=1)["reason"] == "unreachable"


def test_secret_names_are_documented(no_keys):
    """Every secret name must appear in the deployment doc."""
    doc = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "DEPLOYMENT_SECRETS.md",
    )
    with open(doc, encoding="utf-8") as fh:
        text = fh.read()
    for names in BUILTIN_PROVIDER_SECRETS.values():
        for name in names:
            assert name in text, "%s is undocumented in DEPLOYMENT_SECRETS.md" % name


def test_no_key_material_in_source():
    """The generator must never carry keys or an obfuscation scheme again."""
    path = os.path.abspath(lrg.__file__).replace(".pyc", ".py")
    with open(path, encoding="utf-8") as fh:
        source = fh.read()
    assert "_XK" not in source, "XOR key-obfuscation reintroduced"
    # The provider-detection helper legitimately mentions bare key prefixes, so
    # match only a prefix followed by enough key characters to be a real key.
    for prefix in ("gsk_", "csk-", "sk-or-", "AIzaSy", "snova-"):
        hit = re.search(re.escape(prefix) + r"[A-Za-z0-9_\-]{20,}", source)
        assert hit is None, "literal %s key found in source: %s" % (prefix, hit)


# The failover order users rely on. v1.2.9.0 removed the committed keys but did
# NOT change this order; pinning it here means a future key/provider edit cannot
# silently reorder or drop a provider.
_EXPECTED_ORDER = [
    "google_ai_3_lite",
    "google_ai_flash",
    "google_ai_lite",
    "groq_builtin",
    "groq_qwen_builtin",
    "cerebras_builtin",
    "sambanova_builtin",
    "mistral_builtin",
    "openrouter_builtin",
]


def test_provider_failover_order_is_stable(no_keys, monkeypatch):
    """With every key set, the chain must be tried in the documented order."""
    for names in BUILTIN_PROVIDER_SECRETS.values():
        monkeypatch.setenv(names[0], "test-key-%s" % names[0].lower())

    gen = LLMResponseGenerator()
    builtin = [p.name for p in gen._providers if p.name.endswith("_builtin")
               or p.name.startswith("google_ai_")]
    assert builtin == _EXPECTED_ORDER, (
        "built-in provider order drifted: %s" % builtin
    )


def test_every_configured_provider_is_tried_in_order(no_keys, monkeypatch):
    """check_connectivity must actually probe the chain, not bail out early."""
    for names in BUILTIN_PROVIDER_SECRETS.values():
        monkeypatch.setenv(names[0], "test-key-%s" % names[0].lower())

    tried = []

    def _record(api_url, api_key, model, *a, **k):
        tried.append(model)
        return None  # every provider "fails" so the loop runs to the end

    monkeypatch.setattr(lrg, "_call_llm_api", _record)

    gen = LLMResponseGenerator()
    status = gen.check_connectivity(timeout=1)

    assert status["available"] is False
    assert status["reason"] == "unreachable"
    expected = [p.model for p in gen._providers]
    assert tried == expected, (
        "providers were not tried in chain order: tried %s, chain %s"
        % (tried, expected)
    )


def test_first_working_provider_wins(no_keys, monkeypatch):
    """The chain stops at the first provider that answers."""
    for names in BUILTIN_PROVIDER_SECRETS.values():
        monkeypatch.setenv(names[0], "test-key-%s" % names[0].lower())

    calls = []

    def _first_ok(api_url, api_key, model, *a, **k):
        calls.append(model)
        return "OK"

    monkeypatch.setattr(lrg, "_call_llm_api", _first_ok)

    gen = LLMResponseGenerator()
    status = gen.check_connectivity(timeout=1)

    assert status["available"] is True
    assert status["provider"] == _EXPECTED_ORDER[0]
    assert len(calls) == 1, "chain kept probing after a provider answered"


# ---------------------------------------------------------------------------
# Optional bundled-key module (utils/builtin_free_keys.py)
#
# The repository ships WITHOUT that file. These tests must therefore pass both
# ways: unchanged when it is absent, and correctly when the deployment owner
# adds it. A fake module stands in so no real key is ever needed here.
# ---------------------------------------------------------------------------

def _install_fake_bundle(monkeypatch, **attrs):
    """Make _import_bundled_keys() return a stub module with these attributes."""
    module = types.ModuleType("utils.builtin_free_keys")
    for name, value in attrs.items():
        setattr(module, name, value)
    monkeypatch.setattr(lrg, "_import_bundled_keys", lambda: module)
    return module


def test_bundled_module_lookup_never_raises():
    """The real import path must be safe whether or not the file exists.

    No stub here on purpose: this is the only test that touches the actual
    module lookup, so it passes both before and after the owner adds the file.
    """
    module = lrg._import_bundled_keys()  # must not raise either way
    keys = lrg.bundled_provider_keys()
    status = lrg.bundled_provider_key_status()

    assert set(status) == set(BUILTIN_PROVIDER_SECRETS)
    if module is None:
        assert keys == {}
        assert not any(status.values())
    else:
        # File present: whatever it covers must be non-empty strings, and the
        # status must agree with the keys actually found.
        assert all(isinstance(v, str) and v.strip() for v in keys.values())
        assert status == {slot: slot in keys for slot in BUILTIN_PROVIDER_SECRETS}


def test_malformed_bundled_module_never_breaks_the_app(no_keys, monkeypatch):
    """A broken bundled module must degrade, not take the app down."""
    def _raise(name):
        raise ValueError("deliberately malformed")

    monkeypatch.setattr(lrg.importlib, "import_module", _raise)
    assert lrg._import_bundled_keys() is None
    assert lrg.bundled_provider_keys() == {}
    # The generator still builds, just with no providers.
    assert LLMResponseGenerator()._providers == []


def test_old_key_block_variable_names_are_what_the_chain_reads(no_keys, monkeypatch):
    """The pre-v1.2.9.0 block pastes in unchanged.

    That block defined exactly these six names, so the chain must read them
    under those names and nothing else.
    """
    assert set(lrg._BUNDLED_KEY_ATTRS.values()) == {
        "_DEFAULT_GOOGLE_AI_KEY",
        "_DEFAULT_GROQ_KEY",
        "_DEFAULT_CEREBRAS_KEY",
        "_DEFAULT_SAMBANOVA_KEY",
        "_DEFAULT_MISTRAL_KEY",
        "_DEFAULT_OPENROUTER_KEY",
    }

    _install_fake_bundle(monkeypatch, **{
        attr: "bundled-%s" % slot
        for slot, attr in lrg._BUNDLED_KEY_ATTRS.items()
    })

    assert lrg.bundled_provider_keys() == {
        slot: "bundled-%s" % slot for slot in lrg._BUNDLED_KEY_ATTRS
    }
    assert all(lrg.bundled_provider_key_status().values())
    assert lrg.missing_builtin_provider_secrets() == []


def test_bundled_keys_build_the_chain_in_the_old_order(no_keys, monkeypatch):
    _install_fake_bundle(monkeypatch, **{
        attr: "bundled-%s" % slot
        for slot, attr in lrg._BUNDLED_KEY_ATTRS.items()
    })

    gen = LLMResponseGenerator()
    assert [p.name for p in gen._providers] == _EXPECTED_ORDER

    # With bundled keys present this is never reported as unconfigured.
    monkeypatch.setattr(lrg, "_call_llm_api", lambda *a, **k: None)
    assert gen.health_check(timeout=1)["reason"] == "unreachable"


def test_bundled_key_is_tried_before_the_deployment_secret(no_keys, monkeypatch):
    """Bundled first, then env/st.secrets behind it — neither is dropped."""
    _install_fake_bundle(monkeypatch,
                         _DEFAULT_GOOGLE_AI_KEY="bundled-google")
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-google")

    gen = LLMResponseGenerator()
    keys = [p.api_key for p in gen._providers]

    assert keys, "no providers were built"
    assert keys[0] == "bundled-google", "the bundled key is not tried first"
    assert "secret-google" in keys, "the deployment secret was dropped"
    assert keys.index("bundled-google") < keys.index("secret-google")


def test_bundled_partial_coverage_still_uses_secrets_for_other_slots(
        no_keys, monkeypatch):
    _install_fake_bundle(monkeypatch, _DEFAULT_GROQ_KEY="bundled-groq")
    monkeypatch.setenv("MISTRAL_API_KEY", "secret-mistral")

    gen = LLMResponseGenerator()
    keys = {p.api_key for p in gen._providers}
    assert "bundled-groq" in keys
    assert "secret-mistral" in keys
    assert lrg.missing_builtin_provider_secrets() == [
        BUILTIN_PROVIDER_SECRETS[slot][0]
        for slot in BUILTIN_PROVIDER_SECRETS
        if slot not in ("groq", "mistral")
    ]


def test_provider_chain_log_never_contains_key_bytes(no_keys, monkeypatch, caplog):
    """The diagnostic log line must not carry even a key prefix."""
    _install_fake_bundle(monkeypatch,
                         _DEFAULT_GROQ_KEY="gsk_supersecretvalue123456")

    with caplog.at_level("INFO", logger=lrg.logger.name):
        LLMResponseGenerator()

    text = caplog.text
    assert "gsk_super" not in text
    assert "supersecret" not in text
    assert "(key set)" in text
