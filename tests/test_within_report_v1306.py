"""Analysis of within-subjects and mixed data: statistics and report (v1.3.0.6).

* ``utils/within_stats.py`` against hand computation (textbook sums of squares, an independent route to
  Mauchly's W and the Greenhouse-Geisser epsilon) and, when scipy is installed, against scipy; the numpy-only path
  the deployed app uses is pinned to reference values produced with scipy 1.17,
* the instructor report (Markdown and HTML) and the student summary of a repeated-measures run contain paired and
  repeated-measures statistics, never between-subjects tests, with one sign convention.
"""
import math
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))
warnings.filterwarnings("ignore")

import utils.instructor_report as ir  # noqa: E402
import utils.within_stats as WS  # noqa: E402
from utils.enhanced_simulation_engine import EffectSizeSpec, EnhancedSimulationEngine  # noqa: E402
from utils.instructor_report import ComprehensiveInstructorReport, InstructorReportGenerator  # noqa: E402

HAS_SCIPY = ir.SCIPY_AVAILABLE
if HAS_SCIPY:
    from scipy import stats as sp_stats


@pytest.fixture(params=["scipy", "numpy-fallback"])
def stats_mode(request, monkeypatch):
    """Run with scipy and with the numpy distribution functions the deployed app uses."""
    if request.param == "numpy-fallback":
        monkeypatch.setattr(ir, "SCIPY_AVAILABLE", False)
        monkeypatch.setattr(ir, "scipy_stats", None)
    elif not ir.SCIPY_AVAILABLE:
        pytest.skip("scipy is not installed")
    return request.param


def _data(n=36, seed=2024):
    rng = np.random.RandomState(seed)
    base = rng.randn(n, 1)
    return np.round(base + rng.randn(n, 3) * 0.9 + np.array([0, 0.35, 0.8]), 3)


# reference values produced with scipy 1.17 for _data()
REF = {
    "t_01": -3.8839785133355837, "p_01": 0.000436118607303444, "dz_01": -0.6473297522225974, "dav_01": -0.5836475976773536,
    "F": 8.416384904478779, "p": 0.0005302778595056178, "p_gg": 0.0006214939406837895, "eps_gg": 0.9678312657141757,
    "mauchly_w": 0.9667620426975079, "mauchly_p": 0.5629023235979875, "peta2": 0.19385273377771617,
    "geta2": 0.06248186283468555, "fr_chi2": 9.5, "fr_p": 0.008651695203120634, "fr_w": 0.13194444444444445,
    "wil_p": 0.002783855190500617,
}


# ---------------------------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------------------------
def test_paired_t_matches_reference_and_hand_formulas(stats_mode):
    Y = _data()
    r = WS.paired_t_test(Y[:, 0], Y[:, 1])
    diff = Y[:, 0] - Y[:, 1]
    assert r["n"] == 36
    assert r["mean_diff"] == pytest.approx(diff.mean())
    assert r["t"] == pytest.approx(diff.mean() / (diff.std(ddof=1) / math.sqrt(36)))
    assert r["t"] == pytest.approx(REF["t_01"], rel=1e-9)
    assert r["p"] == pytest.approx(REF["p_01"], rel=1e-6)
    assert r["d_z"] == pytest.approx(REF["dz_01"], rel=1e-9)
    assert r["d_av"] == pytest.approx(diff.mean() / ((Y[:, 0].std(ddof=1) + Y[:, 1].std(ddof=1)) / 2), rel=1e-9)
    assert r["d_av"] == pytest.approx(REF["dav_01"], rel=1e-9)
    lo, hi = r["ci"]
    assert lo < r["mean_diff"] < hi
    assert (hi - lo) / 2 == pytest.approx(ir._t_crit(35) * diff.std(ddof=1) / 6, rel=1e-9)


def test_d_z_and_d_av_are_linked_by_the_correlation():
    Y = _data()
    r = WS.paired_t_test(Y[:, 0], Y[:, 1])
    s1, s2, rho = Y[:, 0].std(ddof=1), Y[:, 1].std(ddof=1), r["r"]
    sd_diff = math.sqrt(s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
    assert r["sd_diff"] == pytest.approx(sd_diff, rel=1e-9)
    equal_sd = math.sqrt(2 * (1 - rho))
    # exact when the two SDs are equal; close here
    assert abs(r["d_z"] - r["d_av"] / equal_sd) < 0.1 * abs(r["d_z"])


def test_paired_t_uses_complete_pairs_only_and_survives_degenerate_input():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([2.0, 2.5, 3.0, np.nan, 7.0])
    assert WS.paired_t_test(x, y)["n"] == 3
    assert WS.paired_t_test([1.0], [2.0])["degenerate"]
    same = WS.paired_t_test([1, 2, 3, 4], [1, 2, 3, 4])
    assert same["p"] == 1.0 and math.isfinite(same["mean_diff"])


def test_wilcoxon_signed_rank_is_exact_and_matches_scipy(stats_mode):
    Y = _data()
    w = WS.wilcoxon_signed_rank(Y[:, 0], Y[:, 2])
    assert w["exact"] and w["w_plus"] == 147.0
    assert w["p"] == pytest.approx(REF["wil_p"], rel=1e-6)
    if HAS_SCIPY:
        assert w["p"] == pytest.approx(sp_stats.wilcoxon(Y[:, 0], Y[:, 2]).pvalue, rel=1e-6)


def test_wilcoxon_with_ties_uses_the_corrected_normal_approximation():
    x = np.array([3, 4, 5, 3, 4, 5, 6, 2, 3, 4, 5, 4], dtype=float)
    y = np.array([2, 4, 3, 3, 2, 5, 4, 2, 2, 3, 5, 2], dtype=float)
    w = WS.wilcoxon_signed_rank(x, y)
    assert not w["exact"] and 0 < w["p"] < 0.05
    if HAS_SCIPY:
        ref = sp_stats.wilcoxon(x, y, method="approx", correction=True).pvalue
        assert w["p"] == pytest.approx(ref, rel=0.02)


# ---------------------------------------------------------------------------------------------
# repeated-measures ANOVA
# ---------------------------------------------------------------------------------------------
def test_one_way_rm_anova_matches_textbook_sums_of_squares_and_reference(stats_mode):
    Y = _data()
    n, k = Y.shape
    gm = Y.mean()
    ss_c = n * ((Y.mean(0) - gm) ** 2).sum()
    ss_s = k * ((Y.mean(1) - gm) ** 2).sum()
    ss_e = ((Y - gm) ** 2).sum() - ss_c - ss_s
    f_hand = (ss_c / (k - 1)) / (ss_e / ((k - 1) * (n - 1)))
    t = WS.rm_anova(Y)["terms"][0]
    assert t["f"] == pytest.approx(f_hand, rel=1e-9) and t["f"] == pytest.approx(REF["F"], rel=1e-9)
    assert (t["df1"], t["df2"]) == (2, 70)
    assert t["p"] == pytest.approx(REF["p"], rel=1e-5)
    assert t["partial_eta2"] == pytest.approx(ss_c / (ss_c + ss_e), rel=1e-9) == pytest.approx(REF["peta2"], rel=1e-9)
    assert t["generalized_eta2"] == pytest.approx(ss_c / (ss_c + ss_s + ss_e), rel=1e-9) == pytest.approx(REF["geta2"], rel=1e-9)


def test_sphericity_matches_an_independent_route(stats_mode):
    Y = _data()
    n, k = Y.shape
    S = np.cov(Y.T)
    P = np.eye(k) - np.ones((k, k)) / k
    Sc = P @ S @ P                                           # double-centred covariance (no contrast matrix)
    eps = np.trace(Sc) ** 2 / ((k - 1) * np.trace(Sc @ Sc))
    ev = np.sort(np.linalg.eigvalsh(Sc))[1:]                 # the k-1 non-zero eigenvalues
    w = float(np.prod(ev) / (ev.mean() ** (k - 1)))
    t = WS.rm_anova(Y)["terms"][0]
    assert t["eps_gg"] == pytest.approx(eps, rel=1e-9) == pytest.approx(REF["eps_gg"], rel=1e-9)
    assert t["mauchly_w"] == pytest.approx(w, rel=1e-8) == pytest.approx(REF["mauchly_w"], rel=1e-8)
    assert t["mauchly_p"] == pytest.approx(REF["mauchly_p"], rel=1e-5)
    assert t["p_gg"] == pytest.approx(REF["p_gg"], rel=1e-5)
    assert t["p"] <= t["p_gg"] + 1e-12 and t["eps_hf"] >= t["eps_gg"] - 1e-12


def test_two_level_rm_anova_equals_the_paired_t_squared(stats_mode):
    Y = _data()[:, :2]
    t = WS.paired_t_test(Y[:, 0], Y[:, 1])["t"]
    a = WS.rm_anova(Y)["terms"][0]
    assert a["f"] == pytest.approx(t ** 2, rel=1e-9) and a["df1"] == 1


def test_factorial_within_main_effects_equal_paired_tests_on_marginals(stats_mode):
    rng = np.random.RandomState(5)
    Y = rng.randn(30, 1) + rng.randn(30, 4) + np.array([0, 0.5, 0.2, 1.0])
    a = WS.rm_anova(Y, [2, 2], ["A", "B"])
    rows = {t["term"]: t for t in a["terms"]}
    marg_a = WS.paired_t_test(Y[:, [0, 1]].mean(1), Y[:, [2, 3]].mean(1))["t"]
    marg_b = WS.paired_t_test(Y[:, [0, 2]].mean(1), Y[:, [1, 3]].mean(1))["t"]
    inter = WS.paired_t_test(Y[:, 0] - Y[:, 1], Y[:, 2] - Y[:, 3])["t"]
    assert rows["A"]["f"] == pytest.approx(marg_a ** 2, rel=1e-9)
    assert rows["B"]["f"] == pytest.approx(marg_b ** 2, rel=1e-9)
    assert rows["A x B"]["f"] == pytest.approx(inter ** 2, rel=1e-9)


def test_friedman_matches_reference_and_scipy(stats_mode):
    f = WS.friedman_test(_data())
    assert f["chi2"] == pytest.approx(REF["fr_chi2"], rel=1e-9) and f["p"] == pytest.approx(REF["fr_p"], rel=1e-6)
    assert f["kendall_w"] == pytest.approx(REF["fr_w"], rel=1e-9) == pytest.approx(f["chi2"] / (36 * 2), rel=1e-9)
    if HAS_SCIPY:
        Y = _data()
        assert f["chi2"] == pytest.approx(sp_stats.friedmanchisquare(*Y.T).statistic, rel=1e-9)


def test_pairwise_paired_applies_holm():
    rows = WS.pairwise_paired(_data(), ["A", "B", "C"])
    assert [(r["label_1"], r["label_2"]) for r in rows] == [("A", "B"), ("A", "C"), ("B", "C")]
    raw = sorted(r["p"] for r in rows)
    holm = sorted(r["p_holm"] for r in rows)
    assert holm[0] == pytest.approx(min(1.0, 3 * raw[0])) and all(h >= p for h, p in zip(holm, raw))


# ---------------------------------------------------------------------------------------------
# mixed ANOVA
# ---------------------------------------------------------------------------------------------
def test_mixed_anova_matches_textbook_decomposition(stats_mode):
    rng = np.random.RandomState(8)
    G, n_g, K = 2, 20, 3
    groups = np.repeat(["T", "C"], n_g)
    Y = rng.randn(G * n_g, 1) + rng.randn(G * n_g, K) + np.array([0, 0.3, 0.6]) + np.where(groups == "T", 1, 0)[:, None] * np.array([0, 0.2, 0.7])
    N = G * n_g
    gm = Y.mean()
    ss_total = ((Y - gm) ** 2).sum()
    gmeans = np.array([Y[groups == g].mean() for g in ("T", "C")])
    ss_g = K * n_g * ((gmeans - gm) ** 2).sum()
    subj = Y.mean(1)
    ss_subj_within = K * sum(((subj[groups == g] - Y[groups == g].mean()) ** 2).sum() for g in ("T", "C"))
    ss_t = N * ((Y.mean(0) - gm) ** 2).sum()
    cell = np.array([[Y[groups == g][:, j].mean() for j in range(K)] for g in ("T", "C")])
    ss_gt = n_g * ((cell - gmeans[:, None] - Y.mean(0)[None, :] + gm) ** 2).sum()
    ss_err = ss_total - ss_g - ss_subj_within - ss_t - ss_gt
    f_g = (ss_g / 1) / (ss_subj_within / (N - 2))
    f_t = (ss_t / 2) / (ss_err / (2 * (N - 2)))
    f_gt = (ss_gt / 2) / (ss_err / (2 * (N - 2)))
    m = WS.mixed_anova(Y, groups, [K], ["Time"], between_name="Group")
    rows = {t["term"]: t for t in m["terms"]}
    assert rows["Group"]["f"] == pytest.approx(f_g, rel=1e-9)
    assert rows["Time"]["f"] == pytest.approx(f_t, rel=1e-9)
    assert rows["Group x Time"]["f"] == pytest.approx(f_gt, rel=1e-9)
    assert (rows["Time"]["df1"], rows["Time"]["df2"]) == (2, 76) and rows["Group x Time"]["df1"] == 2
    if HAS_SCIPY:
        assert rows["Group"]["p"] == pytest.approx(sp_stats.f.sf(f_g, 1, N - 2), rel=1e-9)


def test_mixed_interaction_for_two_levels_equals_the_difference_score_t_test(stats_mode):
    rng = np.random.RandomState(9)
    groups = np.repeat(["T", "C"], 25)
    Y = rng.randn(50, 1) + rng.randn(50, 2) + np.where(groups == "T", 1, 0)[:, None] * np.array([0, 0.8])
    m = WS.mixed_anova(Y, groups, [2], ["Time"])
    f_int = next(t for t in m["terms"] if t["type"] == "interaction")["f"]
    d = WS.interaction_by_difference_scores(Y[:, 0], Y[:, 1], groups)
    assert d["t"] ** 2 == pytest.approx(f_int, rel=1e-9)
    assert {g["group"] for g in d["per_group"]} == {"T", "C"}
    if HAS_SCIPY:
        a = (Y[:, 0] - Y[:, 1])[groups == "T"]
        b = (Y[:, 0] - Y[:, 1])[groups == "C"]
        assert d["p"] == pytest.approx(sp_stats.ttest_ind(a, b).pvalue, rel=1e-9)


# ---------------------------------------------------------------------------------------------
# the reports
# ---------------------------------------------------------------------------------------------
def _scale(name, items):
    return {"name": name, "variable_name": name, "num_items": items, "scale_points": 7, "scale_min": 1, "scale_max": 7,
            "reverse_items": [], "type": "likert" if items > 1 else "single_item"}


@pytest.fixture(scope="module")
def within_run():
    lv = ["Baseline", "Week 4", "Week 8"]
    e = EnhancedSimulationEngine(
        study_title="Wellbeing programme", study_description="A mindfulness programme and wellbeing over eight weeks.",
        sample_size=90, conditions=lv, factors=[{"name": "Time", "levels": lv}],
        scales=[_scale("Wellbeing", 4), _scale("Stress", 1)], additional_vars=[], demographics={"gender_quota": 50},
        seed=11, dropout_rate=0.05,
        effect_sizes=[EffectSizeSpec("Wellbeing", "Time", "Week 8", "Baseline", 0.6)], design={"type": "within"})
    df, md = e.generate()
    return e, df, md


@pytest.fixture(scope="module")
def mixed_run():
    e = EnhancedSimulationEngine(
        study_title="Training study", study_description="A training programme and later skill ratings.", sample_size=90,
        conditions=["Training", "Waitlist"], factors=[{"name": "Group", "levels": ["Training", "Waitlist"]}],
        scales=[_scale("Skill", 4)], additional_vars=[], demographics={"gender_quota": 50}, seed=12,
        effect_sizes=[EffectSizeSpec("Skill", "Group", "Training", "Waitlist", 0.7)],
        design={"type": "mixed", "within_factors": [{"name": "Time", "levels": ["Pre", "Post"]}]})
    df, md = e.generate()
    return e, df, md


BETWEEN_ONLY = ("Independent-samples", "pooled-variance", "Welch", "One-way ANOVA", "Tukey", "Mann-Whitney", "Levene",
                "Kruskal", "Condition Balance")


def _both(df, md):
    rep = ComprehensiveInstructorReport()
    mdv = rep.generate_comprehensive_report(df=df, metadata=md, schema_validation={}, prereg_text="", team_info={})
    html = ComprehensiveInstructorReport().generate_html_report(df=df, metadata=md, schema_validation={}, prereg_text="", team_info={})
    assert rep.section_errors == []
    return mdv, html


def test_within_report_has_paired_and_rm_statistics_and_no_between_tests(within_run):
    _, df, md = within_run
    text, html = _both(df, md)
    for doc in (text, html):
        assert "Repeated-measures ANOVA" in doc and "Paired comparisons" in doc
        assert "Greenhouse-Geisser" in doc and "Mauchly" in doc and "d_av" in doc and "d_z" in doc
        assert "Friedman" in doc
        for banned in BETWEEN_ONLY:
            assert banned not in doc, banned
    assert "Baseline - Week 4" in text and "Week 4 - Baseline" not in text      # first minus second, one convention


def test_within_report_numbers_agree_with_a_direct_computation(within_run):
    _, df, md = within_run
    text, _ = _both(df, md)
    a = df["Wellbeing_Baseline_mean"].to_numpy(float)
    b = df["Wellbeing_Week_8_mean"].to_numpy(float)
    r = WS.paired_t_test(a, b)
    assert f"{r['mean_diff']:.3f}" in text
    assert f"{r['d_av']:+.2f}" in text and f"{r['d_z']:+.2f}" in text


def test_within_report_checks_the_requested_effect_in_d_av(within_run):
    _, df, md = within_run
    text, html = _both(df, md)
    assert "Requested effects against this sample" in text
    assert "+0.60" in text and "Week 8 - Baseline" in text
    assert "Requested effects" in html


def test_mixed_report_has_the_mixed_anova_and_the_interaction_follow_up(mixed_run):
    _, df, md = mixed_run
    text, html = _both(df, md)
    for doc in (text, html):
        assert "Mixed ANOVA" in doc and "Group x Time" in doc
        assert "difference scores" in doc
        for banned in ("Independent-samples", "pooled-variance", "One-way ANOVA", "Tukey"):
            assert banned not in doc, banned
    assert "Group differences within each condition" in text


def test_html_report_is_hardened_and_escaped(within_run):
    _, df, md = within_run
    _, html = _both(df, md)
    assert "<script" not in html.lower() and "onerror" not in html.lower()
    assert "p &lt; .001" in html or "p = 0." in html or "&lt; .001" in html


def test_student_summary_explains_the_layout_and_gives_paired_code(within_run):
    _, df, md = within_run
    text = InstructorReportGenerator().generate_markdown_report(df=df, metadata=md, schema_validation={}, team_info={})
    assert "Data layout (repeated measures)" in text and "Simulated_Data_Long.csv" in text
    assert "ttest_rel" in text and "AnovaRM" in text and "paired = TRUE" in text
    assert "Power Analysis Estimates (paired design)" in text
    for banned in ("Independent-samples t-test", "Welch's t-test", "One-way ANOVA", "Tukey HSD", "per group"):
        assert banned not in text, banned


def test_between_report_is_still_the_between_report():
    e = EnhancedSimulationEngine(
        study_title="Framing", study_description="Message versions.", sample_size=60, conditions=["A", "B"],
        factors=[], scales=[_scale("Attitude", 3)], additional_vars=[], demographics={"gender_quota": 50}, seed=3)
    df, md = e.generate()
    text, html = _both(df, md)
    assert "Repeated-measures ANOVA" not in text and "Paired comparisons" not in text
    assert "Welch" in text or "pooled-variance" in text or "t-test" in text


# ---------------------------------------------------------------------------------------------
# mixed ANOVA with several between factors / more than two groups (Type III, sum-to-zero coding)
# ---------------------------------------------------------------------------------------------
def _factorial_data(seed=14, n_cell=10, K=3):
    rng = np.random.RandomState(seed)
    a = np.repeat(["a1", "a2"], 2 * n_cell)
    b = np.tile(np.repeat(["b1", "b2"], n_cell), 2)
    eff_a = np.where(a == "a1", 0.5, 0.0)[:, None] * np.array([0, 0.4, 0.8])
    eff_b = np.where(b == "b1", 0.3, 0.0)[:, None] + 0.4 * (a == "a1")[:, None] * (b == "b2")[:, None] * np.array([0, 1, 0])
    Y = rng.randn(len(a), 1) + rng.randn(len(a), K) + eff_a + eff_b
    return Y, a, b


def test_two_between_factors_match_the_classical_balanced_decomposition(stats_mode):
    Y, a, b = _factorial_data()
    n, K = Y.shape
    m = Y.mean(1)                                  # person means (between part)
    gm = m.mean()
    cell = {(x, y): m[(a == x) & (b == y)] for x in ("a1", "a2") for y in ("b1", "b2")}
    n_c = 10
    ma = {x: np.mean([cell[(x, y)].mean() for y in ("b1", "b2")]) for x in ("a1", "a2")}
    mb = {y: np.mean([cell[(x, y)].mean() for x in ("a1", "a2")]) for y in ("b1", "b2")}
    ss_a = 2 * n_c * sum((ma[x] - gm) ** 2 for x in ma)
    ss_b = 2 * n_c * sum((mb[y] - gm) ** 2 for y in mb)
    ss_ab = n_c * sum((cell[(x, y)].mean() - ma[x] - mb[y] + gm) ** 2 for x in ma for y in mb)
    ss_err = sum(((cell[k] - cell[k].mean()) ** 2).sum() for k in cell)
    df_e = n - 4
    res = WS.mixed_anova(Y, np.column_stack([a, b]), [K], ["Time"], between_names=["A", "B"])
    rows = {t["term"]: t for t in res["terms"]}
    for name, ss in (("A", ss_a), ("B", ss_b), ("A x B", ss_ab)):
        assert rows[name]["f"] == pytest.approx((ss / 1) / (ss_err / df_e), rel=1e-9), name
        assert (rows[name]["df1"], rows[name]["df2"]) == (1, df_e)
    # within part: Time and its interactions share the error of the full between model
    assert {"Time", "A x Time", "B x Time", "A x B x Time"} <= set(rows)
    zt = Y @ WS.helmert_contrasts(K).T             # orthonormal time contrasts
    ss_time = n * float((zt.mean(0) ** 2).sum())
    resid = sum(((zt[(a == x) & (b == y)] - zt[(a == x) & (b == y)].mean(0)) ** 2).sum() for x in ("a1", "a2") for y in ("b1", "b2"))
    assert rows["Time"]["f"] == pytest.approx((ss_time / (K - 1)) / (resid / (df_e * (K - 1))), rel=1e-9)
    assert (rows["A x Time"]["df1"], rows["A x Time"]["df2"]) == (K - 1, (K - 1) * df_e)
    if HAS_SCIPY:
        assert rows["A"]["p"] == pytest.approx(sp_stats.f.sf(rows["A"]["f"], 1, df_e), rel=1e-9)


def test_unbalanced_type_iii_main_effect_equals_the_unweighted_marginal_contrast(stats_mode):
    rng = np.random.RandomState(3)
    sizes = {("a1", "b1"): 18, ("a1", "b2"): 7, ("a2", "b1"): 6, ("a2", "b2"): 15}
    a, b = [], []
    for (x, y), k in sizes.items():
        a += [x] * k
        b += [y] * k
    a, b = np.array(a), np.array(b)
    Y = rng.randn(len(a), 1) + rng.randn(len(a), 2) + np.where(a == "a1", 0.6, 0.0)[:, None] + np.where(b == "b1", 0.4, 0.0)[:, None]
    m = Y.mean(1)
    cm = {k: m[(a == k[0]) & (b == k[1])].mean() for k in sizes}
    mse = sum(((m[(a == k[0]) & (b == k[1])] - cm[k]) ** 2).sum() for k in sizes) / (len(a) - 4)
    psi = (cm[("a1", "b1")] + cm[("a1", "b2")]) / 2 - (cm[("a2", "b1")] + cm[("a2", "b2")]) / 2
    var_unit = 0.25 * sum(1.0 / k for k in sizes.values())
    f_hand = psi ** 2 / (var_unit * mse)
    res = WS.mixed_anova(Y, np.column_stack([a, b]), [2], ["Time"], between_names=["A", "B"])
    row = next(t for t in res["terms"] if t["term"] == "A")
    assert row["f"] == pytest.approx(f_hand, rel=1e-9)
    assert res["method"].startswith("Type III")


def test_three_groups_are_one_factor_with_two_degrees_of_freedom(stats_mode):
    rng = np.random.RandomState(11)
    g = np.repeat(["g1", "g2", "g3"], 12)
    Y = rng.randn(36, 1) + rng.randn(36, 2) + np.array([[0.0], [0.5], [1.0]])[np.repeat([0, 1, 2], 12)]
    res = WS.mixed_anova(Y, g, [2], ["Time"], between_name="Dose")
    row = next(t for t in res["terms"] if t["term"] == "Dose")
    assert (row["df1"], row["df2"]) == (2, 33)
    m = Y.mean(1)
    f_hand = (12 * sum((m[g == k].mean() - m.mean()) ** 2 for k in ("g1", "g2", "g3")) / 2) / (
        sum(((m[g == k] - m[g == k].mean()) ** 2).sum() for k in ("g1", "g2", "g3")) / 33)
    assert row["f"] == pytest.approx(f_hand, rel=1e-9)


def test_report_for_two_between_factors_lists_every_term():
    conds = ["Training x Online", "Training x Live", "Waitlist x Online", "Waitlist x Live"]
    e = EnhancedSimulationEngine(
        study_title="Training study", study_description="A training programme, online or live, and skill ratings.",
        sample_size=120, conditions=conds,
        factors=[{"name": "Programme", "levels": ["Training", "Waitlist"]}, {"name": "Mode", "levels": ["Online", "Live"]}],
        scales=[_scale("Skill", 3)], additional_vars=[], demographics={"gender_quota": 50}, seed=21,
        design={"type": "mixed", "within_factors": [{"name": "Time", "levels": ["Pre", "Post"]}]})
    df, md = e.generate()
    text, html = _both(df, md)
    for term in ("Programme x Mode", "Programme x Time", "Mode x Time", "Programme x Mode x Time"):
        assert term in text and term in html, term
    assert "full factors" in text
