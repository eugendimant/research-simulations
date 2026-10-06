"""Regression tests for latent bugs fixed in v1.2.8.8 (see per-test docstrings)."""
import ast
import math
import os
import random
import re
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_APP = os.path.join(_ROOT, "simulation_app")
if _APP not in sys.path:
    sys.path.insert(0, _APP)

_UTILS = os.path.join(_APP, "utils")


# ---------------------------------------------------------------------------
# 1. instructor_report forest-plot fallback must not raise NameError
# ---------------------------------------------------------------------------
def test_forest_plot_fallback_has_no_nameerror(monkeypatch):
    import utils.instructor_report as ir

    if not ir.MATPLOTLIB_AVAILABLE:
        pytest.skip("matplotlib unavailable")
    gen = ir.ComprehensiveInstructorReport.__new__(ir.ComprehensiveInstructorReport)
    comps = [
        {"comparison": "A vs B", "cohens_d": 0.5, "significant": True},
        {"comparison": "A vs C", "cohens_d": -0.2, "significant": False},
    ]
    real_subplots = ir.plt.subplots
    calls = {"n": 0}

    def flaky_subplots(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("forced primary failure")
        return real_subplots(*a, **k)

    monkeypatch.setattr(ir.plt, "subplots", flaky_subplots)
    out = gen._create_effect_size_forest_plot(comps)
    assert calls["n"] == 2, "fallback path was not exercised"
    assert isinstance(out, str) and len(out) > 100, "fallback should produce a base64 PNG"


# ---------------------------------------------------------------------------
# 2. No duplicate constant dict keys in the three modules
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fname", [
    "enhanced_simulation_engine.py", "response_library.py", "llm_response_generator.py",
])
def test_no_duplicate_dict_literal_keys(fname):
    path = os.path.join(_UTILS, fname)
    tree = ast.parse(open(path).read())
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        seen = {}
        for k in node.keys:
            if isinstance(k, ast.Constant):
                if k.value in seen:
                    problems.append(f"{fname}:{k.lineno}: duplicate key {k.value!r} (first at line {seen[k.value]})")
                else:
                    seen[k.value] = k.lineno
    assert not problems, "\n".join(problems)


def test_merged_templates_keep_both_variants():
    from utils.response_library import DOMAIN_TEMPLATES
    ml = DOMAIN_TEMPLATES["moral_judgment"]["explanation"]["very_positive"]
    assert "This action is clearly morally right." in ml
    assert "Morally, this is the correct path." in ml  # from the formerly shadowed entry
    ls = DOMAIN_TEMPLATES["life_satisfaction"]["explanation"]["very_positive"]
    assert "Life is wonderful." in ls and "My life is close to ideal." in ls
    assert len(ml) == len(set(ml))


# ---------------------------------------------------------------------------
# 3. Late-binding closure in synonym substitution
# ---------------------------------------------------------------------------
def test_synonym_substitution_each_word_gets_own_replacement():
    from utils.response_library import ComprehensiveResponseGenerator
    gen = ComprehensiveResponseGenerator()
    synonyms = {
        "good": ['fine', 'decent', 'solid', 'positive'],
        "bad": ['poor', 'inadequate', 'subpar', 'lacking'],
        "easy": ['straightforward', 'simple', 'manageable', 'uncomplicated'],
        "great": ['excellent', 'outstanding', 'remarkable', 'impressive'],
    }
    for seed in range(20):
        out = gen._vary_response("The good plan was bad but easy and Great.", random.Random(seed), 5)
        words = re.findall(r"[A-Za-z]+", out)
        assert words[1] in synonyms["good"]
        assert words[4] in synonyms["bad"]
        assert words[6] in synonyms["easy"]
        assert words[8].lower() in synonyms["great"]
        assert words[8][0].isupper()  # case preserved


# ---------------------------------------------------------------------------
# 4. instructor_report NaN check / closures
# ---------------------------------------------------------------------------
def test_p_significance_handles_nan_none_and_values():
    from utils.instructor_report import ComprehensiveInstructorReport as G
    for bad in (float("nan"), np.nan, None):
        assert G._p_significance(bad)["sig_label"] == "ns"
    assert G._p_significance(0.0004)["sig_label"] == "***"
    assert G._p_significance(0.03)["sig_label"] == "*"
    assert G._p_significance(0.5)["sig_label"] == "ns"
    src = open(os.path.join(_UTILS, "instructor_report.py")).read()


def test_no_late_binding_closures_in_loop_sites():
    """ruff B023 (function definition does not bind loop variable) must stay clean."""
    import shutil
    import subprocess
    import sys
    ruff = shutil.which("ruff")
    cmd = [ruff] if ruff else [sys.executable, "-m", "ruff"]
    try:
        out = subprocess.run(
            cmd + ["check", "--select", "B023", "--no-cache",
                   os.path.join(_UTILS, "instructor_report.py"),
                   os.path.join(_UTILS, "response_library.py")],
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("ruff not available")
    if "No module named ruff" in out.stderr:
        pytest.skip("ruff not installed")
    assert out.returncode == 0, out.stdout + out.stderr


# ---------------------------------------------------------------------------
def _gen(seed=42, n=400, items=5):
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine
    eng = EnhancedSimulationEngine(
        study_title="T", study_description="d", sample_size=n,
        conditions=["Control", "Treatment"], factors=[],
        scales=[{"name": "Trust", "num_items": items, "scale_points": 7,
                 "reverse_items": [], "_validated": True}],
        additional_vars=[], demographics={"age_mean": 35, "age_sd": 10},
        attention_rate=0.95, random_responder_rate=0.02, effect_sizes=[],
        open_ended_questions=[], seed=seed)
    df, _ = eng.generate()
    return df


def _alpha(X):
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))


def test_construct_scale_alpha_and_sd_realistic_and_deterministic():
    df1 = _gen(seed=7)
    df2 = _gen(seed=7)
    cols = [f"Trust_{i}" for i in range(1, 6)]
    assert df1[cols].equals(df2[cols]), "same seed must give identical numeric output"
    alphas, sds = [], []
    for _, g in df1.groupby("CONDITION"):
        X = g[cols].to_numpy(float)
        alphas.append(_alpha(X))
        sds.append(X.std(axis=0, ddof=1).mean())
    assert 0.70 <= np.mean(alphas) <= 0.93
    assert np.mean(sds) >= 1.0


# ---------------------------------------------------------------------------
# 6. constants module
# ---------------------------------------------------------------------------
def test_topic_stop_words_is_frozenset_superset_of_claude_md():
    from utils.constants import TOPIC_STOP_WORDS
    assert isinstance(TOPIC_STOP_WORDS, frozenset)
    md = open(os.path.join(_ROOT, "CLAUDE.md"), encoding="utf-8").read()
    m = re.search(r"_stop\s*=\s*(\{.*?\})", md, re.S)
    assert m, "stop-word pattern not found in CLAUDE.md"
    expected = ast.literal_eval(m.group(1))
    assert expected <= TOPIC_STOP_WORDS


# ---------------------------------------------------------------------------
# 7. Composites and dropout metadata agree with the delivered data
# ---------------------------------------------------------------------------
def _engine_with_missing(seed):
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine
    eng = EnhancedSimulationEngine(
        study_title="T", study_description="d", sample_size=80,
        conditions=["Control", "Treatment"], factors=[{"name": "G", "levels": ["Control", "Treatment"]}],
        scales=[{"name": "Trust", "num_items": 4, "scale_points": 7, "reverse_items": [2]},
                {"name": "Sat", "num_items": 1, "scale_points": 7}],
        additional_vars=[], demographics={"age_mean": 35, "age_sd": 10},
        open_ended_questions=[], seed=seed, missing_data_rate=0.05, dropout_rate=0.08)
    eng.llm_generator.disable_permanently("test: no network")
    df, md = eng.generate()
    return df, md


@pytest.mark.parametrize("seed", [3, 5, 7, 11])
def test_composites_equal_mean_of_delivered_items_and_dropouts_match(seed):
    df, md = _engine_with_missing(seed)
    items = df[[f"Trust_{i}" for i in range(1, 5)]].astype(float).copy()
    items["Trust_2"] = 8 - items["Trust_2"]
    expected = items.mean(axis=1)
    both_nan = expected.isna() & df["Trust_mean"].isna()
    assert (both_nan | ((expected - df["Trust_mean"]).abs() <= 0.006)).all()
    # every counted dropout must have blanked at least one scale cell
    cols = [c for c in df.columns if c.startswith(("Trust_", "Sat_")) and not c.endswith("_mean")]
    blanked = df[cols].isna().any(axis=1).sum()
    assert md["missing_data"]["dropout_count"] <= blanked


def test_every_domain_template_bank_is_reachable():
    """A DOMAIN_TEMPLATES key that is neither a StudyDomain value nor aliased can never be selected."""
    from utils.response_library import DOMAIN_TEMPLATES, StudyDomain, _DOMAIN_TEMPLATE_ALIASES
    values = {d.value for d in StudyDomain}
    aliased = {a for targets in _DOMAIN_TEMPLATE_ALIASES.values() for a in targets}
    unreachable = sorted(k for k in DOMAIN_TEMPLATES if k not in values and k not in aliased)
    assert not unreachable, unreachable
    for alias_source in _DOMAIN_TEMPLATE_ALIASES:
        assert alias_source in values, alias_source
