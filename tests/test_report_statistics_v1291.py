"""Statistical accuracy of the instructor-analysis report (v1.2.9.1).

The deployed app has no scipy (simulation_app/requirements.txt lists numpy and pandas only), so the
numbers the report prints come from numpy-only implementations there.  These tests pin:

* the distribution functions (t, F, chi-square, Shapiro-Wilk, Mann-Whitney, Kruskal-Wallis, Fisher) to
  scipy reference values, with and without scipy,
* one ordering / one contrast ("A - B") / one effect-size label / one p format in Markdown and HTML,
* the regression covariates, reference level and adjusted R2,
* the factorial ANOVA (exact level matching, residual row, three factors, Type III sums of squares),
* small-sample behaviour: t-based confidence intervals, Yates correction, chi-square validity,
  analytic N, Holm-adjusted pairwise p-values, zero-variance cells, Group_1 / Group_2 labels.

Everything builds small DataFrames directly (no engine run).  Tests that need scipy for a comparison skip
when it is missing; the reference values hard-coded below were produced with scipy 1.17.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import utils.instructor_report as ir  # noqa: E402
from utils.instructor_report import ComprehensiveInstructorReport  # noqa: E402

HAS_SCIPY = ir.SCIPY_AVAILABLE
if HAS_SCIPY:
    from scipy import stats as sp_stats


@pytest.fixture(params=["scipy", "numpy-fallback"])
def stats_mode(request, monkeypatch):
    """Run the test with scipy and with the numpy implementations the deployed app uses."""
    if request.param == "numpy-fallback":
        monkeypatch.setattr(ir, "SCIPY_AVAILABLE", False)
        monkeypatch.setattr(ir, "scipy_stats", None)
    elif not ir.SCIPY_AVAILABLE:
        pytest.skip("scipy is not installed")
    return request.param


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _scale(name: str, k: int = 1) -> dict:
    return {"name": name, "num_items": k, "scale_points": 7, "scale_min": 1, "scale_max": 7, "type": "likert"}


def _study(cond_scores: dict, name: str = "Trust", seed: int = 7, factors=None, meta_order=None):
    """DataFrame with one single-item DV (<name>_1), Age and Gender, plus its metadata."""
    rng = np.random.RandomState(seed)
    rows = []
    for cond, values in cond_scores.items():
        for v in values:
            rows.append({
                "PARTICIPANT_ID": len(rows) + 1, "CONDITION": cond, "Age": int(rng.randint(18, 70)),
                "Gender": str(rng.choice(["Male", "Female"])), "Attention_Pass_Rate": 1.0,
                "Completion_Time_Seconds": 600, "Exclude_Recommended": 0, f"{name}_1": v,
            })
    df = pd.DataFrame(rows)
    meta = {
        "study_title": "Statistics study", "study_description": "Synthetic.", "conditions": list(meta_order or cond_scores),
        "sample_size": len(df), "factors": factors or [], "open_ended_questions": [], "scales": [_scale(name)],
        "run_id": "R1", "generation_timestamp": "2026-10-06T10:00:00", "effect_sizes_observed": [],
        "scale_generation_log": [{"name": name, "columns_generated": [f"{name}_1"]}],
    }
    return df, meta


def _reports(df, meta, prereg: str = ""):
    md = ComprehensiveInstructorReport().generate_comprehensive_report(
        df=df, metadata=meta, schema_validation={}, prereg_text=prereg, team_info={})
    html = ComprehensiveInstructorReport().generate_html_report(
        df=df, metadata=meta, schema_validation={}, prereg_text=prereg, team_info={})
    return md, html


def _text(markup: str) -> str:
    markup = re.sub(r"<style.*?</style>", " ", markup, flags=re.S)
    markup = re.sub(r"<svg.*?</svg>", " ", markup, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", markup))


# ---------------------------------------------------------------------------
# 1. distribution functions (reference values from scipy; no scipy needed to run)
# ---------------------------------------------------------------------------
T2_REF = [(0.1, 10, 0.9223207185644082), (2.0, 10, 0.07338803477074037), (1.96, 58, 0.054806164352606664),
          (2.5, 28, 0.01855092306954576), (3.0, 98, 0.0034232644637296143), (1.0, 150, 0.31892095448042684),
          (6.5, 40, 9.376009811443226e-08), (12.0, 7, 6.358310378185098e-06), (0.0, 20, 1.0),
          (2.2, 15, 0.04389557512749648), (1.5, 9998, 0.13364597813861417)]
F_REF = [(1.0, 1, 10, 0.34089313230206), (4.0, 1, 58, 0.05019046804144834), (3.0, 3, 56, 0.038100647804244124),
         (2.5, 4, 295, 0.04272098092821325), (9.0, 2, 27, 0.001011677008277577), (0.3, 7, 40, 0.949719743767358),
         (25.0, 3, 120, 1.2334744094818624e-12), (1.8, 12, 5000, 0.0425682137850568)]
CHI_REF = [(0.5, 1, 0.47950012218695337), (3.84, 1, 0.05004352124870519), (5.99, 2, 0.05003662708658629),
           (11.1, 5, 0.0494329339438485), (30.0, 9, 0.00043872177097947936), (110.0, 105, 0.34991532656432855),
           (2.0, 20, 0.9999998885745217), (60.0, 30, 0.0009206823961486636)]
TCRIT_REF = [(1, 12.706204736174705), (2, 4.302652729749464), (4, 2.7764451051977943), (9, 2.2621571627982053),
             (14, 2.144786687917804), (29, 2.045229642132704), (58, 2.001717484145236),
             (198, 1.9720174778363146), (998, 1.9623438462163345)]


@pytest.mark.parametrize("t,df,p", T2_REF)
def test_t_distribution_is_exact_without_scipy(t, df, p):
    assert ir._t_two_sided_p(t, df) == pytest.approx(p, rel=1e-9, abs=1e-15)
    tail = ir._t_sf(t, df)
    assert tail == pytest.approx(0.5 * p, rel=1e-9, abs=1e-15)
    assert ir._t_sf(-t, df) == pytest.approx(1.0 - 0.5 * p, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("f,d1,d2,p", F_REF)
def test_f_distribution_is_exact_without_scipy(f, d1, d2, p):
    assert ir._f_sf(f, d1, d2) == pytest.approx(p, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("x,df,p", CHI_REF)
def test_chi_square_distribution_is_exact_without_scipy(x, df, p):
    assert ir._chi2_sf(x, df) == pytest.approx(p, rel=1e-9, abs=1e-15)
    assert ir._chi2_cdf(x, df) == pytest.approx(1.0 - p, abs=1e-12)


@pytest.mark.parametrize("df,crit", TCRIT_REF)
def test_t_critical_value_in_both_modes(df, crit, stats_mode):
    assert ir._t_crit(df) == pytest.approx(crit, rel=1e-9)
    assert ir._t_isf(0.025, df) == pytest.approx(crit, rel=1e-9)


def test_extreme_and_degenerate_arguments_do_not_raise():
    assert ir._t_two_sided_p(1e9, 5) == 0.0 or ir._t_two_sided_p(1e9, 5) < 1e-30
    assert math.isnan(ir._t_two_sided_p(float("nan"), 5))
    assert ir._t_two_sided_p(float("inf"), 5) == 0.0
    assert math.isnan(ir._f_sf(1.0, 0, 5))
    assert ir._f_sf(0.0, 2, 5) == 1.0
    assert ir._chi2_sf(0.0, 3) == 1.0
    assert ir._normal_cdf(-40.0) >= 0.0 and ir._normal_sf(40.0) >= 0.0


def test_welch_pooled_anova_and_levene_match_reference_in_both_modes(stats_mode):
    a = np.array([4.1, 5.2, 3.9, 6.0, 5.5, 4.7, 5.1])
    b = np.array([3.0, 3.8, 4.4, 2.9, 3.5, 4.0, 3.3, 3.7, 4.2])
    df = pd.DataFrame({"CONDITION": ["A"] * len(a) + ["B"] * len(b), "y": np.concatenate([a, b])})
    res = ComprehensiveInstructorReport()._run_statistical_tests(df, "y", "CONDITION")
    assert res["t_test"]["statistic"] == pytest.approx(4.059290662198196, rel=1e-9)
    assert res["t_test"]["p_value"] == pytest.approx(0.001171757285262715, rel=1e-8)
    assert res["welch_t_test"]["statistic"] == pytest.approx(3.8701489520458505, rel=1e-9)
    assert res["welch_t_test"]["p_value"] == pytest.approx(0.002987387778555365, rel=1e-8)
    assert res["anova"]["f_statistic"] == pytest.approx(16.47784068020947, rel=1e-9)
    assert res["levene_test"]["p_value"] == pytest.approx(0.4075424612879576, rel=1e-8)


def test_two_group_t_test_and_anova_p_are_identical_in_both_modes(stats_mode):
    rng = np.random.RandomState(11)
    for n, shift in ((8, 0.0), (30, 0.4), (150, 0.2), (400, 0.1)):
        df = pd.DataFrame({"CONDITION": ["A"] * n + ["B"] * n,
                           "y": np.concatenate([rng.normal(0, 1, n), rng.normal(shift, 1, n)])})
        res = ComprehensiveInstructorReport()._run_statistical_tests(df, "y", "CONDITION")
        assert abs(res["t_test"]["p_value"] - res["anova"]["p_value"]) < 1e-9
        assert res["anova"]["f_statistic"] == pytest.approx(res["t_test"]["statistic"] ** 2, rel=1e-9)


def test_shapiro_wilk_fallback_is_the_real_test_not_a_skew_kurtosis_rule():
    x20 = np.array([2.1, 3.4, 1.9, 5.6, 4.4, 3.8, 2.7, 6.1, 3.3, 4.9, 2.2, 3.9, 5.1, 4.1, 3.0, 2.8, 4.6, 3.5, 7.9, 3.6])
    w, p = ir._numpy_shapiro(x20)
    assert w == pytest.approx(0.9428735544551161, abs=1e-6) and p == pytest.approx(0.27156002989501854, abs=1e-5)
    x30 = np.array([4, 5, 5, 6, 3, 7, 4, 4, 5, 6, 2, 5, 5, 6, 7, 4, 3, 5, 6, 5, 4, 4, 5, 6, 5, 3, 5, 4, 6, 5], float)
    w, p = ir._numpy_shapiro(x30)
    assert w == pytest.approx(0.9376559129996451, abs=1e-6) and p == pytest.approx(0.07866361057775868, abs=1e-5)
    w, p = ir._numpy_shapiro(np.array([1.2, 2.5, 0.8]))
    assert w == pytest.approx(0.9145569620253162, abs=1e-8) and p == pytest.approx(0.43346373142742645, abs=1e-8)
    w, p = ir._numpy_shapiro(np.array([1.2, 2.5, 0.8, 3.9, 2.2, 1.0, 0.5]))
    assert w == pytest.approx(0.901632823233378, abs=1e-6) and p == pytest.approx(0.34098362752473127, abs=1e-5)
    assert ir._numpy_shapiro(np.array([3.0, 3.0, 3.0, 3.0])) == (1.0, 1.0)


def test_normality_verdict_is_labelled_shapiro_wilk_in_both_modes(stats_mode):
    rng = np.random.RandomState(2)
    df = pd.DataFrame({"CONDITION": ["A"] * 40 + ["B"] * 40, "y": np.clip(np.round(rng.normal(4.2, 1.4, 80)), 1, 7)})
    res = ComprehensiveInstructorReport()._run_statistical_tests(df, "y", "CONDITION")
    assert res["normality_test"]["test_name"] == "Shapiro-Wilk"
    ref = ir._numpy_shapiro(df["y"].to_numpy())[1]
    assert res["normality_test"]["p_value"] == pytest.approx(ref, abs=2e-6)


def test_mann_whitney_and_kruskal_fallbacks_match_reference():
    u, p = ir._numpy_mannwhitneyu(np.array([3, 5, 4, 6, 7, 5, 4, 6]), np.array([2, 3, 4, 3, 5, 2, 4, 3]))
    assert u == pytest.approx(54.5) and p == pytest.approx(0.01847584433291784, rel=1e-9)          # ties: asymptotic
    u, p = ir._numpy_mannwhitneyu(np.array([1.5, 2.7, 3.1, 0.4, 2.2]), np.array([3.9, 4.4, 5.0, 2.9, 6.1, 4.8]))
    assert u == pytest.approx(1.0) and p == pytest.approx(0.008658008658008658, rel=1e-9)          # no ties, small: exact
    h, p = ir._numpy_kruskal([1, 2, 2, 3, 4, 5], [3, 4, 4, 5, 6, 7], [5, 6, 6, 7, 8, 9, 9])
    assert h == pytest.approx(11.45265632943569, rel=1e-9) and p == pytest.approx(0.0032590218823319627, rel=1e-9)


def test_chi_square_2x2_has_the_yates_correction_and_fisher_is_exact():
    tab = np.array([[12, 5], [6, 11]])
    chi2, p, dof, _ = ir._numpy_chi2_contingency(tab)
    assert (dof, chi2) == (1, pytest.approx(2.951388888888889))
    assert p == pytest.approx(0.08580378797002607, rel=1e-9)
    chi2_plain, p_plain, _, _ = ir._numpy_chi2_contingency(tab, correction=False)
    assert chi2_plain == pytest.approx(4.25) and p_plain == pytest.approx(0.03925033046769265, rel=1e-9)
    assert ir._fisher_exact_2x2(tab) == pytest.approx(0.08441466872675717, rel=1e-9)
    assert ir._fisher_exact_2x2(np.array([[2, 7], [6, 1]])) == pytest.approx(0.040559440559440565, rel=1e-9)
    chi2, p, dof, _ = ir._numpy_chi2_contingency(np.array([[10, 12, 8], [14, 9, 11]]))
    assert dof == 2 and chi2 == pytest.approx(1.3240945501007422) and p == pytest.approx(0.515794280059632, rel=1e-9)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy is needed for the dense comparison")
def test_numpy_distribution_functions_agree_with_scipy_on_a_dense_grid():
    worst = 0.0
    for df in list(range(1, 41)) + [58, 98, 150, 298, 500, 2000]:
        for t in list(np.linspace(0.0, 8.0, 41)) + [12.0, 30.0]:
            worst = max(worst, abs(ir._t_two_sided_p(t, df) - 2 * sp_stats.t.sf(t, df)))
    for d1 in (1, 2, 3, 5, 12):
        for d2 in (3, 10, 28, 58, 200, 1000):
            for f in list(np.linspace(0.05, 10.0, 40)) + [30.0, 100.0]:
                worst = max(worst, abs(ir._f_sf(f, d1, d2) - sp_stats.f.sf(f, d1, d2)))
    for df in list(range(1, 21)) + [40, 100, 400]:
        for x in list(np.linspace(0.1, 2.5 * df + 15, 40)):
            worst = max(worst, abs(ir._chi2_sf(x, df) - sp_stats.chi2.sf(x, df)))
    assert worst < 1e-9
    for df in (1, 2, 3, 5, 10, 30, 100, 1000):
        for tail in (0.1, 0.05, 0.025, 0.005, 0.0005):
            assert ir._t_isf(tail, df) == pytest.approx(sp_stats.t.isf(tail, df), rel=1e-9)


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy is needed for the comparison")
def test_numpy_nonparametric_tests_agree_with_scipy_on_likert_data():
    rng = np.random.RandomState(5)
    for n1, n2 in ((5, 7), (8, 8), (12, 30), (60, 61)):
        a = np.clip(np.round(rng.normal(4.0, 1.5, n1)), 1, 7)
        b = np.clip(np.round(rng.normal(4.6, 1.5, n2)), 1, 7)
        u0, p0 = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        u1, p1 = ir._numpy_mannwhitneyu(a, b)
        assert u1 == pytest.approx(u0) and p1 == pytest.approx(p0, abs=1e-9)
        h0, kp0 = sp_stats.kruskal(a, b)
        h1, kp1 = ir._numpy_kruskal(a, b)
        assert h1 == pytest.approx(h0, rel=1e-9) and kp1 == pytest.approx(kp0, abs=1e-9)
        w0, sp0 = sp_stats.shapiro(np.concatenate([a, b]))
        w1, sp1 = ir._numpy_shapiro(np.concatenate([a, b]))
        assert w1 == pytest.approx(w0, abs=1e-6) and sp1 == pytest.approx(sp0, abs=1e-5)


# ---------------------------------------------------------------------------
# 2. one ordering, "A - B" contrasts, one effect label, one p format
# ---------------------------------------------------------------------------
def test_conditions_are_ordered_once_metadata_first():
    assert ir._order_conditions(["B", "A", "C"]) == ["B", "A", "C"]
    assert ir._order_conditions(["B", "A", "C"], ["A", "B"]) == ["A", "B", "C"]
    assert ir._order_conditions(["B", float("nan"), "A"], ["Z", "A"]) == ["A", "B"]


def test_two_group_statistics_follow_the_metadata_order(stats_mode):
    rng = np.random.RandomState(3)
    df = pd.DataFrame({"CONDITION": ["Treatment"] * 30 + ["Control"] * 30,
                       "y": np.concatenate([rng.normal(1.0, 1, 30), rng.normal(0.0, 1, 30)])})
    rep = ComprehensiveInstructorReport()
    by_data = rep._run_statistical_tests(df, "y", "CONDITION")
    by_meta = rep._run_statistical_tests(df, "y", "CONDITION", condition_order=["Control", "Treatment"])
    assert by_data["t_test"]["contrast"] == "Treatment - Control" and by_meta["t_test"]["contrast"] == "Control - Treatment"
    assert by_meta["t_test"]["statistic"] == pytest.approx(-by_data["t_test"]["statistic"])
    assert by_meta["cohens_d"]["value"] == pytest.approx(-by_data["cohens_d"]["value"])
    assert by_meta["t_test"]["p_value"] == pytest.approx(by_data["t_test"]["p_value"])


def test_markdown_and_html_report_the_same_signed_contrast(stats_mode):
    rng = np.random.RandomState(4)
    scores = {"Treatment": rng.normal(5.2, 1.0, 40), "Control": rng.normal(4.2, 1.0, 40)}   # data order: Treatment first
    df, meta = _study(scores, meta_order=["Control", "Treatment"])
    md, html = _reports(df, meta)
    d_md = float(re.search(r"Cohen's d, Control - Treatment\):\*\* (-?\d+\.\d+)", md).group(1))
    d_html = float(re.search(r"Effect Size \(Cohen's d\):</strong> (-?\d+\.\d+)", html).group(1))
    t_html = float(re.search(r"\bt = (-?\d+\.\d+)", html).group(1))
    assert d_md < 0 and d_html < 0 and t_html < 0
    assert d_md == pytest.approx(d_html, abs=2e-3)
    assert "difference = Control - Treatment" in html and "Control - Treatment" in html


def test_cohens_d_label_is_one_function_with_one_set_of_cut_points():
    cuts = [(0.0, "negligible"), (0.099, "negligible"), (0.1, "very small"), (0.199, "very small"), (0.2, "small"),
            (0.499, "small"), (0.5, "medium"), (0.799, "medium"), (0.8, "large"), (1.199, "large"), (1.2, "very large")]
    for d, label in cuts:
        assert ir._cohens_d_label(d) == label and ir._cohens_d_label(-d) == label
        assert ComprehensiveInstructorReport()._interpret_cohens_d(d) == label
    assert ir._cohens_d_label(float("nan")) == "undefined"


def test_the_same_d_gets_the_same_label_in_markdown_html_and_sentences(stats_mode):
    base = np.linspace(-1.7, 1.7, 40)
    base = (base - base.mean()) / base.std(ddof=1)
    for shift, label in ((0.15, "very small"), (0.35, "small"), (0.65, "medium")):
        df, meta = _study({"Control": base + shift, "Treatment": base})
        md, html = _reports(df, meta)
        d_md = float(re.search(r"Cohen's d, Control - Treatment\):\*\* (-?\d+\.\d+)", md).group(1))
        assert d_md == pytest.approx(shift, abs=2e-3)
        assert f"→ {label.capitalize()} effect" in md
        assert f"The effect size is <strong>{label}</strong>" in html
        for other in {"negligible", "very small", "small", "medium", "large", "very large"} - {label}:
            assert f"The effect size is <strong>{other}</strong>" not in html
            assert f"→ {other.capitalize()} effect" not in md


def test_p_values_below_one_in_a_thousand_are_never_printed_as_zero(stats_mode):
    rng = np.random.RandomState(8)
    df, meta = _study({"Control": rng.normal(3.0, 0.6, 60), "Treatment": rng.normal(5.0, 0.6, 60)})
    md, html = _reports(df, meta)
    assert "&lt; .001" in html
    assert not re.search(r"p\s*=\s*0\.0{3,4}(?![0-9])", _text(html) + md)
    assert not re.search(r">\s*0\.0000\s*<", html)
    svg = ir.svg_charts.create_bar_chart_svg({"A": (3.0, 0.2), "B": (5.0, 0.2)}, p_value=1e-12, effect_size=3.0)
    assert "p &lt; .001" in svg and "0.0000" not in svg


def test_report_p_formatters_agree():
    assert ir._fmt_p(0.0) == "< .001" and ir._fmt_p_html(0.0004) == "&lt; .001" and ir._fmt_p(0.0234) == "0.0234"
    assert ir._p_eq(0.00001) == "p < .001" and ir._p_eq(0.5) == "p = 0.5000" and ir._p_eq(float("nan")) == "p n/a"
    assert ir._fmt_stat(float("inf")) == "undefined" and ir._fmt_stat(1.23456, 2) == "1.23"


# ---------------------------------------------------------------------------
# 3. regression: covariates by whole word, reference level, adjusted R2
# ---------------------------------------------------------------------------
def _regression_frame(seed: int = 5, n: int = 90):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "CONDITION": np.repeat(["Zed", "Alpha", "Mid"], n // 3),
        "Age": rng.randint(18, 70, n), "Gender": rng.choice(["Male", "Female"], n),
    })
    base = rng.normal(0, 1, n)
    for stem in ("Message", "Engagement"):
        for i in (1, 2, 3):
            df[f"{stem}_{i}"] = np.round(4 + base + rng.normal(0, 0.7, n) + (df["CONDITION"] == "Mid") * 0.5, 2)
    df["_composite"] = df[["Engagement_1", "Engagement_2", "Engagement_3"]].mean(axis=1)
    return df


def test_control_columns_are_matched_by_whole_word():
    df = _regression_frame()
    df["Image_1"] = 1.0
    df["Percentage"] = 2.0
    df["Participant_Age"] = df["Age"]
    assert ir._match_control_columns(df, "age") == ["Age"]                       # exact name wins
    assert "Message_1" not in ir._match_control_columns(df, "age")
    only = df.drop(columns=["Age"])
    assert ir._match_control_columns(only, "age") == ["Participant_Age"]
    assert ir._match_control_columns(df, "controlling for age") == ["Age"]
    assert ir._match_control_columns(df, "engagement") == []                      # a DV's own items are never covariates
    assert ir._match_control_columns(df, "engagement", exclude=[]) == []
    assert ir._find_demographic_column(df, ("age",)) == "Age"
    assert ir._find_demographic_column(df, ("gender", "sex"), numeric=False) == "Gender"


def test_prereg_age_mention_does_not_pull_in_message_or_engagement_columns(stats_mode):
    rep = ComprehensiveInstructorReport()
    df = _regression_frame()
    exclude = [c for c in df.columns if c.startswith(("Message_", "Engagement_"))]
    res = rep._run_regression_analysis(df, "_composite", "CONDITION", prereg_controls=["age", "gender"],
                                       condition_order=["Zed", "Alpha", "Mid"], exclude_columns=exclude)
    assert res["controls_included"] == ["Age", "Gender"]
    assert not any(k.startswith(("Message", "Engagement")) for k in res["coefficients"])
    assert res["model_fit"]["r_squared"] < 0.9
    assert not (set(res["coefficients"]) & set(exclude))
    # without the exclusion list the safety net (item-like names) still protects the model
    res2 = rep._run_regression_analysis(df, "_composite", "CONDITION", prereg_controls=["age", "engagement"])
    assert res2["controls_included"] == ["Age", "Gender"] and res2["model_fit"]["r_squared"] < 0.9


def test_prereg_parser_matches_age_and_gender_as_whole_words():
    rep = ComprehensiveInstructorReport()
    assert "age" not in rep._parse_prereg_hypotheses("The message and engagement scales are the outcomes.")["control_variables"]
    assert "gender" not in rep._parse_prereg_hypotheses("Participants come from Essex.")["control_variables"]
    parsed = rep._parse_prereg_hypotheses("We control for age and gender.")["control_variables"]
    assert "age" in parsed and "gender" in parsed


def test_engagement_dv_with_a_prereg_age_control_gets_a_sane_regression_in_html(stats_mode):
    rng = np.random.RandomState(2)
    df, meta = _study({"Control": rng.normal(4, 1, 60), "Treatment": rng.normal(4.6, 1, 60)}, name="Engagement")
    for i in (1, 2, 3):                       # a second DV whose items contain the letters "age"
        df[f"Message_{i}"] = np.clip(np.round(rng.normal(4, 1, len(df))), 1, 7)
    meta["scales"].append(_scale("Message", 3))
    meta["scale_generation_log"].append({"name": "Message", "columns_generated": ["Message_1", "Message_2", "Message_3"]})
    _, html = _reports(df, meta, prereg="Controlling for age and gender.")
    text = _text(html)
    assert "Control variables included: Age, Gender" in text
    assert not re.search(r"R² = 1\.0000", text)
    assert "Message_1" not in text.split("Regression Analysis")[1][:1500]


def test_adjusted_r2_uses_n_minus_k_with_k_counting_the_intercept(stats_mode):
    df = _regression_frame(seed=9)
    res = ComprehensiveInstructorReport()._run_regression_analysis(
        df, "_composite", "CONDITION", condition_order=["Zed", "Alpha", "Mid"],
        exclude_columns=[c for c in df.columns if c.startswith(("Message_", "Engagement_"))])
    names = list(res["coefficients"])
    X = np.column_stack([np.ones(len(df))] + [
        {"Alpha": (df["CONDITION"] == "Alpha"), "Mid": (df["CONDITION"] == "Mid")}[n].astype(float).to_numpy() if n in ("Alpha", "Mid")
        else ((df["Age"] - df["Age"].mean()) / (df["Age"].std() + 1e-10)).to_numpy() if n == "Age"
        else (df["Gender"] == "Male").astype(float).to_numpy()
        for n in names if n != "intercept"])
    y = df["_composite"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
    n, k = X.shape
    assert res["model_fit"]["r_squared"] == pytest.approx(r2, abs=1e-9)
    assert res["model_fit"]["adj_r_squared"] == pytest.approx(1 - (1 - r2) * (n - 1) / (n - k), abs=1e-9)
    assert res["model_fit"]["adj_r_squared"] != pytest.approx(1 - (1 - r2) * (n - 1) / (n - k - 1), abs=1e-6)
    assert res["model_fit"]["df_residual"] == n - k
    assert res["coefficients"]["Alpha"]["estimate"] == pytest.approx(beta[names.index("Alpha")], abs=1e-9)


def test_regression_reference_level_is_the_first_condition_not_the_alphabetical_one(stats_mode):
    df = _regression_frame()
    rep = ComprehensiveInstructorReport()
    res = rep._run_regression_analysis(df, "_composite", "CONDITION", condition_order=["Zed", "Alpha", "Mid"])
    assert res["reference_category"] == "Zed" and res["reference_levels"]["Condition"] == "Zed"
    assert "Zed" not in res["coefficients"] and {"Alpha", "Mid"} <= set(res["coefficients"])
    assert res["reference_levels"]["Gender"] == "Female"
    res = rep._run_regression_analysis(df, "_composite", "CONDITION")           # no order: first appearance
    assert res["reference_category"] == "Zed"
    rng = np.random.RandomState(1)
    study, meta = _study({"Zed": rng.normal(4, 1, 30), "Alpha": rng.normal(5, 1, 30)})
    _, html = _reports(study, meta)
    assert "Reference levels:" in html and "Condition = Zed" in html and "Gender = Female" in html


# ---------------------------------------------------------------------------
# 4. factorial ANOVA
# ---------------------------------------------------------------------------
AGENT = {"name": "Agent", "levels": ["AI", "No AI"]}           # deliberately the order that broke substring matching
PRODUCT = {"name": "Product", "levels": ["Hedonic", "Utilitarian"]}


def _factorial_frame(sizes=None, seed=3, factors=(AGENT, PRODUCT), sep=" x ", effects=(0.5, 0.3)):
    import itertools
    rng = np.random.RandomState(seed)
    rows = []
    for combo in itertools.product(*[f["levels"] for f in factors]):
        n = (sizes or {}).get(combo, 25)
        mu = effects[0] * (combo[0] == factors[0]["levels"][0]) + effects[1] * (combo[1] == factors[1]["levels"][1])
        for v in rng.normal(mu, 1, n):
            rows.append({"CONDITION": sep.join(combo), "y": v})
    return pd.DataFrame(rows)


def test_factor_levels_are_matched_exactly_not_by_substring():
    parse = ir._parse_condition_levels
    fac = [AGENT, PRODUCT]
    assert parse("No AI x Hedonic", fac) == ("No AI", "Hedonic")
    assert parse("AI x Hedonic", fac) == ("AI", "Hedonic")
    assert parse("No AI × Utilitarian (new)", fac) == ("No AI", "Utilitarian")
    assert parse("Hedonic x AI", [PRODUCT, AGENT]) == ("Hedonic", "AI")
    sex = [{"name": "Sex", "levels": ["Male", "Female"]}, {"name": "Tone", "levels": ["Receptive", "Unreceptive"]}]
    assert parse("Female x Unreceptive", sex) == ("Female", "Unreceptive")
    assert parse("Male_Receptive", sex) == ("Male", "Receptive")
    assert parse("Control", fac) is None and parse("AI x Hedonic x Extra", fac) is None
    numeric = [{"name": "Dose", "levels": ["1", "2", "3", "4"]}, {"name": "Frame", "levels": ["gain", "loss"]}]
    assert parse("3_loss", numeric) == ("3", "loss") and parse("11_loss", numeric) is None


def test_factorial_result_exposes_the_residual_under_its_own_key_and_matches_the_classical_ss(stats_mode):
    df = _factorial_frame()
    res = ComprehensiveInstructorReport()._run_factorial_anova(df, "y", [AGENT, PRODUCT], "CONDITION")
    assert "error" not in res and res["residual"]["df"] == 96 and len(res["terms"]) == 3
    assert res["main_effect_1"]["factor"] == "Agent" and res["interaction"]["factors"] == "Agent × Product"
    grand = df["y"].mean()
    parts = df["CONDITION"].str.replace("No AI", "NoAI").str.split(" x ", expand=True)
    ss_a = sum(len(g) * (g["y"].mean() - grand) ** 2 for _, g in df.groupby(parts[0]))
    assert res["main_effect_1"]["ss"] == pytest.approx(ss_a, rel=1e-9)
    cell_means = df.groupby("CONDITION")["y"].transform("mean")
    assert res["residual"]["ss"] == pytest.approx(((df["y"] - cell_means) ** 2).sum(), rel=1e-9)
    f = res["main_effect_1"]["f_statistic"]
    assert res["main_effect_1"]["p_value"] == pytest.approx(ir._f_sf(f, 1, 96), abs=1e-9)
    assert "No AI × Hedonic" in res["cell_statistics"] and res["cell_statistics"]["AI × Hedonic"]["n"] == 25
    assert res["total"]["df"] == 99 and "Factors analysed: Agent (2 levels) × Product (2 levels)" in res["design_note"]


def test_type_iii_sums_of_squares_are_right_for_unequal_cells(stats_mode):
    sizes = {("AI", "Hedonic"): 31, ("AI", "Utilitarian"): 9, ("No AI", "Hedonic"): 12, ("No AI", "Utilitarian"): 25}
    df = _factorial_frame(sizes=sizes, seed=6, effects=(0.5, 0.3))
    res = ComprehensiveInstructorReport()._run_factorial_anova(df, "y", [AGENT, PRODUCT], "CONDITION")
    means = {c: g["y"].mean() for c, g in df.groupby("CONDITION")}
    counts = {c: len(g) for c, g in df.groupby("CONDITION")}

    def ss(weights):          # SS of a contrast of cell means: L^2 / sum(w^2 / n)
        return sum(w * means[c] for c, w in weights.items()) ** 2 / sum(w * w / counts[c] for c, w in weights.items())

    a, b, c, d = "AI x Hedonic", "AI x Utilitarian", "No AI x Hedonic", "No AI x Utilitarian"
    assert res["main_effect_1"]["ss"] == pytest.approx(ss({a: .5, b: .5, c: -.5, d: -.5}), rel=1e-9)
    assert res["main_effect_2"]["ss"] == pytest.approx(ss({a: -.5, b: .5, c: -.5, d: .5}), rel=1e-9)
    assert res["interaction"]["ss"] == pytest.approx(ss({a: 1, b: -1, c: -1, d: 1}), rel=1e-9)
    assert "Type III" in res["design_note"]


def test_three_factors_are_analysed_with_all_interactions(stats_mode):
    valence = {"name": "Valence", "levels": ["Pos", "Neg"]}
    import itertools
    rng = np.random.RandomState(12)
    rows = []
    for combo in itertools.product(AGENT["levels"], PRODUCT["levels"], valence["levels"]):
        for v in rng.normal(0.4 * (combo[0] == "AI") - 0.5 * (combo[2] == "Neg"), 1, 20):
            rows.append({"CONDITION": " x ".join(combo), "y": v})
    df = pd.DataFrame(rows)
    res = ComprehensiveInstructorReport()._run_factorial_anova(df, "y", [AGENT, PRODUCT, valence], "CONDITION")
    assert "error" not in res and len(res["terms"]) == 7 and res["residual"]["df"] == 160 - 8
    assert [t["term"] for t in res["terms"]][-1] == "Agent × Product × Valence"
    assert "Agent (2 levels) × Product (2 levels) × Valence (2 levels)" in res["design_note"]
    grand = df["y"].mean()
    val = df["CONDITION"].str.split(" x ").str[2]
    ss_val = sum(len(g) * (g["y"].mean() - grand) ** 2 for _, g in df.groupby(val))
    assert next(t for t in res["terms"] if t["term"] == "Valence")["ss"] == pytest.approx(ss_val, rel=1e-9)


def test_a_fourth_factor_is_disclosed_not_silently_dropped(stats_mode):
    import itertools
    facs = [AGENT, PRODUCT, {"name": "Valence", "levels": ["Pos", "Neg"]}, {"name": "Source", "levels": ["Human", "Bot"]}]
    rng = np.random.RandomState(4)
    df = pd.DataFrame([{"CONDITION": " x ".join(c), "y": v} for c in itertools.product(*[f["levels"] for f in facs])
                       for v in rng.normal(0, 1, 6)])
    res = ComprehensiveInstructorReport()._run_factorial_anova(df, "y", facs, "CONDITION")
    assert "error" not in res and res["factors_not_analysed"] == ["Source"]
    assert "not analysed (their levels are pooled): Source" in res["design_note"]


def test_a_design_that_is_not_fully_crossed_gets_a_stated_reason(stats_mode):
    tone = {"name": "Tone", "levels": ["Warm", "Neutral", "Cold"]}
    df = _factorial_frame(factors=(AGENT, tone))
    df = df[df["CONDITION"] != "No AI x Cold"]                           # one of the six cells has no data
    res = ComprehensiveInstructorReport()._run_factorial_anova(df, "y", [AGENT, tone], "CONDITION")
    assert "complete crossing" in res["error"] and "5 of 6 cells" in res["error"] and res.get("not_crossed")
    cond_scores = {c: g["y"].to_numpy() for c, g in df.groupby("CONDITION", sort=False)}
    study, meta = _study(cond_scores, factors=[AGENT, tone])
    _, html = _reports(study, meta)
    assert ("Factorial ANOVA (Agent x Tone) could not be computed: the conditions do not form a complete crossing"
            in _text(html))
    assert "Factorial ANOVA (Main Effects" not in html


def test_factorial_table_appears_in_the_html_report_and_a_control_group_is_disclosed(stats_mode):
    df = _factorial_frame(seed=5)
    cond_scores = {c: g["y"].to_numpy() for c, g in df.groupby("CONDITION", sort=False)}
    cond_scores["Control"] = np.random.RandomState(1).normal(0, 1, 25)
    study, meta = _study(cond_scores, factors=[AGENT, PRODUCT])
    _, html = _reports(study, meta)
    text = _text(html)
    assert "Factorial ANOVA (Main Effects & Interaction)" in text
    assert re.search(r"Residual \d+\.\d+ 96 \d+\.\d+ - - -", text)
    assert "Agent × Product" in text and "25 participants in conditions outside the factorial crossing (Control)" in text


# ---------------------------------------------------------------------------
# 5. small samples, validity, analytic N, Holm, zero variance, labels
# ---------------------------------------------------------------------------
def test_confidence_intervals_use_the_t_quantile_not_1_96(stats_mode):
    vals = {"Control": np.array([3.0, 4.0, 5.0, 6.0, 4.5]), "Treatment": np.array([5.0, 6.0, 7.0, 5.5, 6.5])}
    df, meta = _study(vals)
    md, html = _reports(df, meta)
    for cond, v in vals.items():
        half = 2.7764451051977943 * v.std(ddof=1) / math.sqrt(len(v))
        lo, hi = v.mean() - half, v.mean() + half
        assert f"[{lo:.3f}, {hi:.3f}]" in md and f"[{lo:.3f}, {hi:.3f}]" in html
        too_narrow = 1.96 * v.std(ddof=1) / math.sqrt(len(v))
        assert f"[{v.mean() - too_narrow:.3f}, {v.mean() + too_narrow:.3f}]" not in html
    assert ir._t_ci_halfwidth(1.0, 1) == 0.0 and ir._t_ci_halfwidth(2.0, 5) == pytest.approx(2.7764451051977943 * 2 / math.sqrt(5))


def test_displayed_n_is_the_analytic_n_when_scores_are_missing(stats_mode):
    rng = np.random.RandomState(6)
    a = rng.normal(4, 1, 30)
    b = rng.normal(5, 1, 30)
    a[:7] = np.nan
    b[:3] = np.nan
    df, meta = _study({"Control": a, "Treatment": b})
    md, html = _reports(df, meta)
    assert "| Control | 23 |" in md and "| Treatment | 27 |" in md and "Control 30, Treatment 30" in md
    text = _text(html)
    assert re.search(r"Control 23 \d", text) and re.search(r"Treatment 27 \d", text)
    assert "Assigned to the condition, including those without a score: Control 30, Treatment 30" in text
    res = ComprehensiveInstructorReport()._run_statistical_tests(df.assign(y=df["Trust_1"]), "y", "CONDITION")
    assert res["descriptives"]["Control"]["n"] == 23 and res["descriptives"]["Control"]["n_assigned"] == 30
    assert res["t_test"]["df"] == 23 + 27 - 2


def test_groups_with_fewer_than_two_scores_are_disclosed(stats_mode):
    rng = np.random.RandomState(3)
    df, meta = _study({"A": rng.normal(4, 1, 20), "B": rng.normal(5, 1, 20), "C": [4.5], "D": [np.nan, np.nan, 5.0]})
    _, html = _reports(df, meta)
    text = _text(html)
    assert "Not included in the tests below (fewer than 2 scores): C (n = 1), D (n = 1)" in text
    assert "The tests compare the remaining 2 conditions" in text
    res = ComprehensiveInstructorReport()._run_statistical_tests(df.assign(y=df["Trust_1"]), "y", "CONDITION")
    assert [g["condition"] for g in res["groups_excluded"]] == ["C", "D"] and res["conditions"] == ["A", "B"]
    assert "t_test" in res and res["anova"]["num_groups"] == 2


def test_holm_adjustment_is_the_step_down_procedure():
    assert ir._holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert ir._holm_adjust([0.5]) == [0.5]
    adjusted = ir._holm_adjust([0.2, float("nan"), 0.01])
    assert adjusted[0] == pytest.approx(0.2) and math.isnan(adjusted[1]) and adjusted[2] == pytest.approx(0.02)
    assert ir._holm_adjust([0.9, 0.8]) == [1.0, 1.0]


def test_pairwise_verdicts_use_holm_adjusted_p_and_the_table_says_so(stats_mode):
    rng = np.random.RandomState(21)
    scores = {c: rng.normal(m, 1, 30) for c, m in (("A", 0.0), ("B", 0.45), ("C", 0.9), ("D", 0.1))}
    df, meta = _study(scores)
    res = ComprehensiveInstructorReport()._run_statistical_tests(df.assign(y=df["Trust_1"]), "y", "CONDITION", condition_order=list(scores))
    pairs = res["pairwise_comparisons"]
    assert len(pairs) == 6 and res["pairwise_adjustment"] == "Holm"
    assert [p["p_adjusted"] for p in pairs] == pytest.approx(ir._holm_adjust([p["p_value"] for p in pairs]))
    for p in pairs:
        assert p["p_adjusted"] >= p["p_value"] - 1e-15
        assert p["significant"] == (p["p_adjusted"] < 0.05)
        assert p["contrast"].replace(" - ", " vs ") == p["comparison"]
    raw_only = [p for p in pairs if p["p_value"] < 0.05 and p["p_adjusted"] >= 0.05]
    _, html = _reports(df, meta)
    text = _text(html)
    assert "p (raw)" in text and "p (Holm)" in text and "adjusted for the 6 comparisons" in text
    assert "after Holm adjustment" in text
    if raw_only:                                  # a pair that raw p would have called significant is not
        assert any(not p["significant"] for p in raw_only)


def test_zero_variance_groups_give_an_undefined_test_not_infinity_or_significance(stats_mode):
    rep = ComprehensiveInstructorReport()
    flat = pd.DataFrame({"CONDITION": ["A"] * 10 + ["B"] * 10, "y": [3.0] * 10 + [5.0] * 10})
    res = rep._run_statistical_tests(flat, "y", "CONDITION")
    assert math.isnan(res["t_test"]["statistic"]) and math.isnan(res["t_test"]["p_value"])
    assert res["t_test"]["significant"] is False and res["t_test"]["defined"] is False
    assert math.isnan(res["anova"]["f_statistic"]) and res["anova"]["significant"] is False
    assert math.isnan(res["cohens_d"]["value"]) and res["cohens_d"]["interpretation"] == "undefined"
    assert res["levene_test"]["homogeneous"] is True
    mixed = pd.DataFrame({"CONDITION": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
                          "y": [3.0] * 10 + [5.0] * 10 + list(np.random.RandomState(1).normal(4, 1, 10))})
    res = rep._run_statistical_tests(mixed, "y", "CONDITION")
    constant_pair = next(p for p in res["pairwise_comparisons"] if p["comparison"] == "A vs B")
    assert math.isnan(constant_pair["t_stat"]) and constant_pair["significant"] is False
    assert all(math.isfinite(p["p_adjusted"]) for p in res["pairwise_comparisons"] if p["comparison"] != "A vs B")


def test_chi_square_validity_flag_and_wording(stats_mode):
    ok = ir._association_test(pd.DataFrame([[30, 28], [27, 31]]))
    assert ok["valid"] and ok["yates"] and ok["dof"] == 1 and "fisher_p" not in ok
    small = ir._association_test(pd.DataFrame([[2, 7], [6, 1]]))
    assert not small["valid"] and small["share_expected_below_5"] == 1.0
    assert small["fisher_p"] == pytest.approx(0.040559440559440565, rel=1e-9)
    big = ir._association_test(pd.DataFrame([[1, 2, 0, 1], [0, 3, 1, 0], [2, 1, 1, 0], [1, 1, 0, 2]]))
    assert not big["valid"] and "fisher_p" not in big
    assert ir._association_test(pd.DataFrame([[5, 6, 7]]))["status"] == "single_condition"
    assert ir._association_test(pd.DataFrame([[5], [6]]))["status"] == "single_category"
    rng = np.random.RandomState(5)
    scores = {f"C{i}": rng.normal(4, 1, 6) for i in range(5)}           # 30 participants, 10 cells: expected counts < 5
    df, meta = _study(scores)
    _, html = _reports(df, meta)
    text = _text(html)
    assert "Too few expected counts for the chi-square test" in text and "randomization appears successful" not in text


def test_chi_square_in_the_report_has_yates_and_a_single_condition_sentence(stats_mode):
    rng = np.random.RandomState(7)
    df, meta = _study({"Control": rng.normal(4, 1, 60), "Treatment": rng.normal(4, 1, 60)}, seed=2)
    _, html = _reports(df, meta)
    ct = pd.crosstab(df["CONDITION"], df["Gender"]).to_numpy()
    chi2, p, dof, _ = ir._numpy_chi2_contingency(ct)
    assert f"χ² = {chi2:.3f}, df = 1" in _text(html) and "Yates continuity correction" in html
    one, meta1 = _study({"Only": rng.normal(4, 1, 40)})
    _, html1 = _reports(one, meta1)
    text1 = _text(html1)
    assert "single condition: no between-condition comparison" in text1 and "df = 0" not in text1
    assert "randomization appears successful" not in text1


def test_group_1_group_2_group_3_stay_distinct(stats_mode):
    rng = np.random.RandomState(9)
    scores = {"Group_1": rng.normal(4, 1, 20), "Group_2": rng.normal(5, 1, 20), "Group_3": rng.normal(6, 1, 20)}
    labels = ir._condition_display_labels(list(scores))
    assert labels == {"Group_1": "Group_1", "Group_2": "Group_2", "Group_3": "Group_3"}
    assert ir._condition_display_labels(["Control (new)", "Treatment_2"]) == {"Control (new)": "Control", "Treatment_2": "Treatment"}
    df, meta = _study(scores)
    res = ComprehensiveInstructorReport()._run_statistical_tests(df.assign(y=df["Trust_1"]), "y", "CONDITION")
    assert list(res["descriptives"]) == ["Group_1", "Group_2", "Group_3"] and res["anova"]["num_groups"] == 3
    reg = ComprehensiveInstructorReport()._run_regression_analysis(df.assign(y=df["Trust_1"]), "y", "CONDITION")
    assert {"Group_2", "Group_3"} <= set(reg["coefficients"]) and reg["reference_category"] == "Group_1"
    _, html = _reports(df, meta)
    for lab in scores:
        assert len(re.findall(rf"<td>{lab}</td><td>20</td>", html)) >= 2      # descriptives and condition distribution
    assert not re.search(r"<td>Group</td>", html)
    assert re.search(r"Group_1 vs Group_2", html)
    # a contaminated value must not relabel rows of the real conditions
    contaminated = df.copy()
    contaminated.loc[0, "CONDITION"] = "Yeah Group_2"
    aligned = ir._align_condition_values(contaminated, list(scores))
    assert (aligned["CONDITION"] == "Group_3").sum() == 20 and (aligned["CONDITION"] == "Group_1").sum() == 19


def test_fallback_and_scipy_reports_agree_on_every_statistic(monkeypatch):
    """The numbers of the Markdown/HTML report do not depend on scipy being installed."""
    if not ir.SCIPY_AVAILABLE:
        pytest.skip("scipy is not installed")
    rng = np.random.RandomState(31)
    scores = {c: rng.normal(m, 1.1, 35) for c, m in (("A", 4.0), ("B", 4.5), ("C", 5.1))}
    df, meta = _study(scores)

    def stat_lines(markup: str):
        text = _text(markup)
        return sorted(set(re.findall(r"(?:t|F|χ²|R²) = -?\d+\.\d+|p = \d\.\d+|p &lt; \.001|p < \.001|\[-?\d+\.\d+, -?\d+\.\d+\]", text)))

    with_scipy = stat_lines(_reports(df, meta)[1])
    monkeypatch.setattr(ir, "SCIPY_AVAILABLE", False)
    monkeypatch.setattr(ir, "scipy_stats", None)
    without = stat_lines(_reports(df, meta)[1])
    assert with_scipy == without and len(with_scipy) > 10
