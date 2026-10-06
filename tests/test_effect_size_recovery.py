#!/usr/bin/env python3
"""Recovered-effect-size regression test.

The simulator must produce between-condition effects close to the Cohen's d the
user configures. Historically it inflated them ~4x (target d=0.5 -> observed ~2.1)
because the d->scale-shift conversion was applied to BOTH levels with a constant
that ignored the real within-condition SD, and because the persona interaction
multiplier averaged ~1.12 instead of 1.0.

Configured d refers to what the user analyses: the scale mean (multi-item) or the
single item. The engine scales the shift by item count so composites are not
inflated by averaging. Bounds tolerate bounded-scale effects (ceiling compression,
ERS) and sampling noise (SE of d ~0.06 at n=1000) but are tight enough that the
old 2-4x inflation fails by a large margin.
"""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))
warnings.filterwarnings("ignore")

from utils.enhanced_simulation_engine import EnhancedSimulationEngine, EffectSizeSpec  # noqa: E402

N_PER_RUN = 1000

# (label, scale_min, scale_max, num_items)
DESIGNS = [
    ("7pt_4items", 1, 7, 4),
    ("7pt_8items", 1, 7, 8),
    ("7pt_single", 1, 7, 1),
    ("5pt_3items", 1, 5, 3),
    ("slider_0_100", 0, 100, 1),
]


#: Seeds averaged by `_recovered_d`. Recovered d is a random variable: at
#: n=1000 its standard error is roughly 0.065, so a single draw tested against a
#: +/-30% band around d=0.5 (a half-width of 0.15, about 2.3 SE) fails by chance
#: often enough to be a flaky gate rather than a measurement. Averaging three
#: seeds cuts the SE to about 0.037 and makes the assertion a statement about the
#: engine rather than about seed 7. See the note on item-count overshoot below.
RECOVERY_SEEDS = (7, 13, 21)


def _recovered_d(target_d, lo, hi, items, seeds=RECOVERY_SEEDS, n=N_PER_RUN):
    """Mean recovered d over `seeds`.

    KNOWN LIMIT, pre-existing and also present on main: the engine over-recovers
    a configured d on long composites. Averaged over seeds, an 8-item 7-point
    scale returns about 1.14x the configured value. That is inside the +/-30%
    band this file asserts, but it is a real bias and not noise: a longer
    composite averages away more item-level noise than the effect scaling
    anticipates. Worth narrowing; not narrowed here.
    """
    return float(np.mean([
        _recovered_d_once(target_d, lo, hi, items, seed=s, n=n) for s in seeds
    ]))


def _recovered_d_once(target_d, lo, hi, items, seed=7, n=N_PER_RUN):
    scales = [{
        "name": "Attitude", "variable_name": "Attitude", "num_items": items,
        "scale_points": hi - lo + 1, "scale_min": lo, "scale_max": hi,
        "reverse_items": [], "type": "likert" if items > 1 else "single_item",
    }]
    eng = EnhancedSimulationEngine(
        study_title="Message Framing and Attitudes",
        study_description="A survey study of attitudes under two message versions.",
        sample_size=n, conditions=["Version A", "Version B"], factors=[],
        scales=scales, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        effect_sizes=[EffectSizeSpec(
            variable="Attitude", factor="condition",
            level_high="Version A", level_low="Version B",
            cohens_d=target_d, direction="positive")],
        seed=seed,
    )
    df, _ = eng.generate()
    col = "Attitude_mean" if "Attitude_mean" in df.columns else "Attitude_1"
    a = df.loc[df["CONDITION"] == "Version A", col].astype(float)
    b = df.loc[df["CONDITION"] == "Version B", col].astype(float)
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled)


@pytest.mark.parametrize("label,lo,hi,items", DESIGNS)
@pytest.mark.parametrize("target", [0.5, 0.8])
def test_recovered_d_tracks_target(label, lo, hi, items, target):
    """Observed d lies within [0.70x, 1.30x] of the target (old code: 2-4x)."""
    d = _recovered_d(target, lo, hi, items)
    assert 0.70 * target <= d <= 1.30 * target, (
        f"{label}: target d={target}, recovered d={d:.2f} "
        f"(allowed {0.70 * target:.2f}-{1.30 * target:.2f})")


@pytest.mark.parametrize("label,lo,hi,items", DESIGNS[:2])
def test_null_effect_stays_null(label, lo, hi, items):
    """With d=0 there must be no spurious treatment effect."""
    d = _recovered_d(0.0, lo, hi, items)
    assert abs(d) < 0.20, f"{label}: d=0 produced recovered d={d:.2f}"


def test_effect_is_monotonic_in_target():
    """Larger configured d must give larger recovered d, with no runaway inflation."""
    ds = [_recovered_d(t, 1, 7, 4) for t in (0.2, 0.5, 0.8)]
    assert ds[0] < ds[1] < ds[2], f"recovered d not monotonic: {ds}"
    assert ds[2] < 1.3 * 0.8, f"recovered d at target 0.8 inflated: {ds[2]:.2f}"


# ---------------------------------------------------------------------------
# Reverse-keyed scales: the composite must be a SCORED scale (reverse items
# recoded), otherwise opposite-keyed items cancel and effects are under-recovered.
# ---------------------------------------------------------------------------
def _reverse_engine(target_d=0.5, n=1000, seed=7, **kw):
    scales = [{
        "name": "Trust", "variable_name": "Trust", "num_items": 5,
        "scale_points": 7, "scale_min": 1, "scale_max": 7,
        "reverse_items": [2, 4], "type": "likert",
    }]
    return EnhancedSimulationEngine(
        study_title="AI vs Human Advisor and Trust",
        study_description="Participants rate trust in an AI or human advisor.",
        sample_size=n, conditions=["AI advisor", "Human advisor"], factors=[],
        scales=scales, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        effect_sizes=[EffectSizeSpec(
            variable="Trust", factor="condition",
            level_high="Human advisor", level_low="AI advisor",
            cohens_d=target_d, direction="positive")],
        seed=seed, **kw)


def test_composite_recodes_reverse_items():
    df, _ = _reverse_engine(n=300).generate()
    items = df[[f"Trust_{i}" for i in range(1, 6)]].astype(float).copy()
    for i in (2, 4):
        items[f"Trust_{i}"] = 8 - items[f"Trust_{i}"]
    assert np.allclose(df["Trust_mean"].astype(float), items.mean(axis=1), atol=0.011)


def test_composite_recodes_reverse_items_with_missing_data():
    df, _ = _reverse_engine(n=400, missing_data_rate=0.05, missing_data_mechanism="mcar").generate()
    items = df[[f"Trust_{i}" for i in range(1, 6)]].astype(float).copy()
    for i in (2, 4):
        items[f"Trust_{i}"] = 8 - items[f"Trust_{i}"]
    expected = items.mean(axis=1, skipna=True)
    ok = df["Trust_mean"].astype(float).notna() & expected.notna()
    assert ok.sum() > 300
    assert np.allclose(df.loc[ok, "Trust_mean"].astype(float), expected[ok], atol=0.011)


def test_reverse_keyed_composite_recovers_effect():
    """Composite d on a scale with reverse items tracks the target (was ~0.4x)."""
    df, _ = _reverse_engine(target_d=0.5).generate()
    a = df.loc[df["CONDITION"] == "Human advisor", "Trust_mean"].astype(float)
    b = df.loc[df["CONDITION"] == "AI advisor", "Trust_mean"].astype(float)
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    d = float((a.mean() - b.mean()) / pooled)
    assert 0.6 * 0.5 <= d <= 1.3 * 0.5, f"reverse-keyed composite d={d:.2f} vs target 0.5"


# ---------------------------------------------------------------------------
# Condition NAMES must not create effects of their own.
# ---------------------------------------------------------------------------
def _named_run(cond_hi, cond_lo, target_d, extra_conds=(), n=1200, seed=3, with_spec=True):
    scales = [{"name": "Attitude", "variable_name": "Attitude", "num_items": 4,
               "scale_points": 7, "scale_min": 1, "scale_max": 7,
               "reverse_items": [], "type": "likert"}]
    specs = [EffectSizeSpec(variable="Attitude", factor="condition", level_high=cond_hi,
                            level_low=cond_lo, cohens_d=target_d, direction="positive")] if with_spec else []
    eng = EnhancedSimulationEngine(
        study_title="Message Framing and Attitudes",
        study_description="A survey study of attitudes under different message versions.",
        sample_size=n, conditions=[cond_hi, cond_lo, *extra_conds], factors=[],
        scales=scales, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        effect_sizes=specs, seed=seed)
    df, _ = eng.generate()
    return df


def _pair_d(df, c1, c2, col="Attitude_mean"):
    a = df.loc[df["CONDITION"] == c1, col].astype(float)
    b = df.loc[df["CONDITION"] == c2, col].astype(float)
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled)


@pytest.mark.parametrize("hi,lo", [("High stakes", "Low stakes"),
                                   ("Paid", "Free"),
                                   ("Maintain", "Keep"),
                                   ("Positive framing", "Negative framing")])
def test_condition_names_do_not_create_effects(hi, lo):
    """d=0 configured: keywords such as high/low/positive/negative/'ai' in the
    condition names must not shift the means (old calibration added ~0.24 d)."""
    d = _pair_d(_named_run(hi, lo, 0.0), hi, lo)
    assert abs(d) < 0.20, f"{hi} vs {lo}: d=0 configured but recovered d={d:.2f}"


def test_control_sits_between_levels_of_configured_effect():
    """A condition matching neither level of a configured effect is the reference
    level: it must sit near the midpoint, not receive keyword effects."""
    df = _named_run("Version A", "Version B", 0.6, extra_conds=("Control",))
    ac, cb = _pair_d(df, "Version A", "Control"), _pair_d(df, "Control", "Version B")
    assert abs(ac - cb) < 0.22, f"control not centred: A-C={ac:.2f}, C-B={cb:.2f}"
    assert ac > 0.1 and cb > 0.1


def test_automatic_valence_effect_is_literature_sized():
    """With no d configured, a positive-vs-negative valence manipulation should give
    a moderate effect (~0.6; Balliet/valence literature), not the old ~1.3."""
    df = _named_run("Positive feedback", "Negative feedback", 0.0, with_spec=False)
    d = _pair_d(df, "Positive feedback", "Negative feedback")
    assert 0.3 <= d <= 0.9, f"automatic valence effect d={d:.2f}"


@pytest.mark.parametrize("target", [0.5, 0.8])
def test_binary_dv_recovers_effect(target):
    """0/1 DVs: configured d must map to a realistic proportion gap (d=0.5 is a
    ~20-point gap, not the ~44 points the inflated pipeline produced)."""
    scales = [{"name": "Choice", "variable_name": "Choice", "num_items": 1,
               "scale_points": 2, "scale_min": 0, "scale_max": 1,
               "reverse_items": [], "type": "binary"}]
    eng = EnhancedSimulationEngine(
        study_title="Message Framing and Choice",
        study_description="Participants choose whether to accept an offer after reading a message.",
        sample_size=2000, conditions=["Version A", "Version B"], factors=[],
        scales=scales, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        effect_sizes=[EffectSizeSpec(variable="Choice", factor="condition",
                                     level_high="Version A", level_low="Version B",
                                     cohens_d=target, direction="positive")],
        seed=2)
    df, _ = eng.generate()
    a = df.loc[df["CONDITION"] == "Version A", "Choice_1"].astype(float)
    b = df.loc[df["CONDITION"] == "Version B", "Choice_1"].astype(float)
    d = float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
    assert set(df["Choice_1"].astype(int).unique()) <= {0, 1}
    assert 0.7 * target <= d <= 1.3 * target, f"binary d={d:.2f} vs target {target}"
    assert (a.mean() - b.mean()) < 0.5 * target + 0.15, "proportion gap implausibly large"


# ---------------------------------------------------------------------------
# Economic-game DVs: published outcome distributions and baselines.
# ---------------------------------------------------------------------------
def _game_df(title, desc, col, conds=("Ingroup partner", "Outgroup partner"), target=None, n=2500, seed=4):
    scales = [{"name": col, "variable_name": col, "num_items": 1, "scale_points": 101,
               "scale_min": 0, "scale_max": 100, "reverse_items": [], "type": "slider"}]
    specs = [] if target is None else [EffectSizeSpec(
        variable=col, factor="condition", level_high=conds[0], level_low=conds[1],
        cohens_d=target, direction="positive")]
    eng = EnhancedSimulationEngine(
        study_title=title, study_description=desc, sample_size=n, conditions=list(conds),
        factors=[], scales=scales, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        effect_sizes=specs, seed=seed)
    return eng.generate()[0]


def test_dictator_game_matches_published_distribution():
    """Engel (2011): mean giving ~28%, ~36% give nothing, ~17% split 50/50, giving
    more than half is rare. The old pipeline gave 37% mean, 9% zeros, 32% > half."""
    df = _game_df("Dictator game giving",
                  "Participants decide how much of a $100 endowment to give to an anonymous partner in a dictator game.",
                  "Dictator_Giving", conds=("Control", "Control B"))
    v = df["Dictator_Giving_1"].astype(float)
    assert 22 <= v.mean() <= 34, f"mean giving {v.mean():.1f}%"
    assert 0.28 <= (v == 0).mean() <= 0.42, f"share giving 0: {(v == 0).mean():.2f}"
    assert 0.12 <= (v == 50).mean() <= 0.22, f"share giving 50: {(v == 50).mean():.2f}"
    assert (v > 50).mean() <= 0.15, f"share giving >50%: {(v > 50).mean():.2f}"
    assert v.min() >= 0 and v.max() <= 100


def test_trust_game_baseline():
    """Berg et al. (1995) / Johnson & Mislin (2011): mean amount sent ~50%."""
    df = _game_df("Trust game",
                  "Participants decide how much of a $100 endowment to send to a trustee who receives triple in a trust game.",
                  "Trust_Sent", conds=("Control", "Control B"))
    m = df["Trust_Sent_1"].astype(float).mean()
    assert 43 <= m <= 57, f"mean amount sent {m:.1f}%"


@pytest.mark.parametrize("target", [0.5, 0.8])
def test_game_dv_recovers_configured_effect(target):
    df = _game_df("Dictator game giving",
                  "Participants decide how much of a $100 endowment to give to an anonymous partner in a dictator game.",
                  "Dictator_Giving", target=target)
    d = _pair_d(df, "Ingroup partner", "Outgroup partner", col="Dictator_Giving_1")
    assert 0.7 * target <= d <= 1.3 * target, f"game d={d:.2f} vs target {target}"


def test_game_automatic_political_discrimination():
    """No d configured: ingroup partners must receive more than outgroup partners
    (Iyengar & Westwood 2015) with a literature-sized effect."""
    df = _game_df("Political ingroup bias in the dictator game",
                  "Partisans allocate money to co-partisans or opposing partisans in a dictator game.",
                  "Dictator_Giving")
    d = _pair_d(df, "Ingroup partner", "Outgroup partner", col="Dictator_Giving_1")
    assert 0.4 <= d <= 1.1, f"automatic intergroup discrimination d={d:.2f}"


# ---------------------------------------------------------------------------
# Cross-scale correlation, reliability and response-distribution realism.
# ---------------------------------------------------------------------------
def _two_scale_df(r, n=1500, seed=3, k=4, names=("Alpha_Scale", "Beta_Scale")):
    scales = [{"name": nm, "variable_name": nm, "num_items": k, "scale_points": 7,
               "scale_min": 1, "scale_max": 7, "reverse_items": [], "type": "likert"} for nm in names]
    eng = EnhancedSimulationEngine(
        study_title="Message Framing and Attitudes",
        study_description="A survey study of attitudes.",
        sample_size=n, conditions=["Version A", "Version B"], factors=[],
        scales=scales, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        effect_sizes=[], seed=seed, correlation_matrix=np.array([[1.0, r], [r, 1.0]]))
    return eng.generate()[0], names


@pytest.mark.parametrize("target", [-0.5, -0.3, 0.0, 0.15, 0.5, 0.8])
def test_cross_scale_correlation_reproduces_target(target):
    """Configured between-scale correlations (incl. negative ones) are reproduced.
    Old pipeline: realised r ~= 0.22 + 0.42 * target, so -0.4 came out ~ +0.04."""
    df, (a, b) = _two_scale_df(target)
    r = float(np.corrcoef(df[a + "_mean"].astype(float), df[b + "_mean"].astype(float))[0, 1])
    assert abs(r - target) <= 0.09, f"target r={target:+.2f}, realised r={r:+.2f}"


def test_multi_item_alpha_is_realistic():
    """A 4-item scale should have a believable Cronbach's alpha: not the 0.95+ the
    injection overshoot used to produce, and not unreliable."""
    df, (a, _) = _two_scale_df(0.3, n=1000)
    X = df[[f"{a}_{i}" for i in range(1, 5)]].astype(float)
    k = X.shape[1]
    alpha = k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))
    assert 0.70 <= alpha <= 0.93, f"alpha={alpha:.2f}"


def test_reverse_keyed_scale_alpha_after_recoding():
    df, _ = _reverse_engine(n=1000).generate()
    X = df[[f"Trust_{i}" for i in range(1, 6)]].astype(float).copy()
    for i in (2, 4):
        X[f"Trust_{i}"] = 8 - X[f"Trust_{i}"]
    alpha = 5 / 4 * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))
    assert 0.70 <= alpha <= 0.93, f"alpha after recoding reverse items = {alpha:.2f} (old: -0.4)"
    assert X.corr().values[np.triu_indices(5, 1)].min() > 0.15


def test_likert_distribution_has_no_ceiling_spike():
    """The 7 bin must not exceed the 6 bin by a wide margin (old ERS endpoint snap
    gave 19.6% at 7 vs 14.9% at 6) and the floor must stay small."""
    df = _named_run("Version A", "Version B", 0.0, n=3000)
    v = df["Attitude_1"].astype(int)
    p = v.value_counts(normalize=True).reindex(range(1, 8)).fillna(0)
    assert p[7] <= p[6] + 0.03, f"ceiling spike: P(7)={p[7]:.3f} vs P(6)={p[6]:.3f}"
    assert p[1] <= 0.06, f"floor too heavy: P(1)={p[1]:.3f}"


def test_effect_spec_matches_display_name_with_underscored_column():
    """A spec written as 'Perceived Quality' must drive the 'Perceived_Quality' column."""
    eng = EnhancedSimulationEngine(
        study_title="Annotation", study_description="Product annotation study",
        sample_size=40, conditions=["Human-curated", "AI-generated"], factors=[],
        scales=[{"name": "Perceived Quality", "variable_name": "Perceived_Quality", "num_items": 3,
                 "scale_points": 7, "scale_min": 1, "scale_max": 7, "reverse_items": [], "type": "likert"}],
        additional_vars=[], demographics={"gender_quota": 50, "age_mean": 30, "age_sd": 8},
        effect_sizes=[EffectSizeSpec(variable="Perceived Quality", factor="condition",
                                     level_high="Human-curated", level_low="AI-generated",
                                     cohens_d=0.5, direction="positive")], seed=1)
    hi = eng._compute_effect_for_condition("Human-curated", "Perceived_Quality")
    lo = eng._compute_effect_for_condition("AI-generated", "Perceived_Quality")
    assert hi > 0 > lo and abs(hi + lo) < 1e-9


# ---------------------------------------------------------------------------
# Automatic effects are anchored to META_ANALYTIC_DB when a paradigm is named.
# ---------------------------------------------------------------------------
def _auto_d(title, conds, dv, n=1600, seed=7):
    name = dv.replace(" ", "_")
    scales = [{"name": dv, "variable_name": name, "num_items": 4, "scale_points": 7,
               "scale_min": 1, "scale_max": 7, "reverse_items": [], "type": "likert"}]
    eng = EnhancedSimulationEngine(
        study_title=title, study_description=title, sample_size=n, conditions=conds, factors=[],
        scales=scales, additional_vars=[], demographics={"gender_quota": 50, "age_mean": 30, "age_sd": 8},
        seed=seed)
    df, _ = eng.generate()
    a = df.loc[df["CONDITION"] == conds[0], f"{name}_mean"].astype(float)
    b = df.loc[df["CONDITION"] == conds[1], f"{name}_mean"].astype(float)
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))


@pytest.mark.parametrize("title,conds,dv,expected", [
    ("Anchoring effect on price estimates", ["High anchor", "Low anchor"], "Estimated price", 0.80),
    ("Default effect in organ donation", ["Opt-out default", "Opt-in default"], "Donation intention", 0.68),
    # 0.17, not the 0.32 this entry carried before the 2026-10-06 recall audit:
    # Epton et al. (2015) found small effects throughout (acceptance .17).
    ("Self-affirmation and health intentions", ["Self-affirmation", "Control"], "Intention", 0.17),
    ("Social proof marketing study", ["Many others bought", "Few others bought"], "Purchase", 0.38),
    ("Mindfulness-based intervention and distress", ["Mindfulness", "Waitlist control"], "Distress", -0.55),
])
def test_meta_anchored_effect_magnitude(title, conds, dv, expected):
    """A named paradigm gets its published magnitude (and the right sign)."""
    d = _auto_d(title, conds, dv)
    assert 0.7 * abs(expected) <= abs(d) <= 1.3 * abs(expected), f"d={d:.2f} vs meta {expected}"
    assert np.sign(d) == np.sign(expected)


def test_unmatched_paradigm_keeps_generic_scaling():
    """No paradigm named -> no anchoring (the generic path is unchanged)."""
    from utils.enhanced_simulation_engine import _match_meta_effect
    assert _match_meta_effect("a survey about everyday things option a option b") is None


def test_explicit_effect_overrides_meta_anchor():
    d = _recovered_d_for_title("Anchoring effect on price estimates", 0.3)
    assert 0.2 <= d <= 0.42, f"explicit d=0.3 must win over the meta value 0.8 (got {d:.2f})"


def _recovered_d_for_title(title, target):
    scales = [{"name": "Price", "variable_name": "Price", "num_items": 4, "scale_points": 7,
               "scale_min": 1, "scale_max": 7, "reverse_items": [], "type": "likert"}]
    eng = EnhancedSimulationEngine(
        study_title=title, study_description=title, sample_size=2000,
        conditions=["High anchor", "Low anchor"], factors=[], scales=scales, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 30, "age_sd": 8},
        effect_sizes=[EffectSizeSpec(variable="Price", factor="condition", level_high="High anchor",
                                     level_low="Low anchor", cohens_d=target, direction="positive")], seed=5)
    df, _ = eng.generate()
    a = df.loc[df["CONDITION"] == "High anchor", "Price_mean"].astype(float)
    b = df.loc[df["CONDITION"] == "Low anchor", "Price_mean"].astype(float)
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))


def test_inferred_correlation_matrix_is_reproduced():
    """With no user matrix, the literature-inferred correlations (6 scales) are
    realised in the composites to within sampling + repair error."""
    from utils.correlation_matrix import infer_correlation_matrix
    names = ["Trust", "Satisfaction", "Anxiety", "Purchase Intention", "Loneliness", "Life Satisfaction"]
    scales = [{"name": n, "variable_name": n.replace(" ", "_"), "num_items": 4, "scale_points": 7,
               "scale_min": 1, "scale_max": 7, "reverse_items": [], "type": "likert"} for n in names]
    target, _ = infer_correlation_matrix(scales)
    eng = EnhancedSimulationEngine(
        study_title="Customer experience survey", study_description="Customer experience survey",
        sample_size=2000, conditions=["A", "B"], factors=[], scales=scales, additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 30, "age_sd": 8}, seed=3)
    df, _ = eng.generate()
    real = np.corrcoef(np.array([df[n.replace(" ", "_") + "_mean"].astype(float) for n in names]))
    iu = np.triu_indices(len(names), 1)
    assert np.abs(real[iu] - target[iu]).max() <= 0.09, np.round(real - target, 2)
    assert abs(float(np.mean(real[iu] - target[iu]))) <= 0.03


@pytest.mark.parametrize("text", [
    "participants see a product page with the default shipping option selected",
    "students in a tutoring session complete homework",
    "dictator game with an ingroup partner",
    "a survey about attitudes toward everyday things",
])
def test_meta_match_has_no_false_positives(text):
    from utils.enhanced_simulation_engine import _match_meta_effect
    assert _match_meta_effect(text) is None


@pytest.mark.parametrize("text,kb_key", [
    ("anchoring effect on price estimates", "anchoring_effect"),
    ("default effect in organ donation: opt-out versus opt-in", "default_effect"),
    ("self-affirmation and health intentions", "self_affirmation_meta"),
])
def test_meta_match_finds_named_paradigms(text, kb_key):
    """The matcher must land on the right knowledge-base entry.

    The expectation is read from the entry rather than frozen here: this test is
    about which entry a title resolves to, and a recalibration of the entry's own
    magnitude is not a matching bug.
    """
    from utils import scientific_knowledge_base as skb
    from utils.enhanced_simulation_engine import _match_meta_effect

    assert _match_meta_effect(text) == pytest.approx(
        skb.META_ANALYTIC_DB[kb_key].effect_d)


def test_meta_anchor_sign_for_consumption_dv():
    """Norms REDUCE energy use: the treatment arm must move the DV down.

    The magnitude band is wide and small on purpose. The 2026-10-06 recall audit
    cut this entry from d=0.35 to 0.08 — Allcott (2011) is a 2% reduction, and the
    entry's own notes already said so — so anything near the old 0.15 floor would
    now be the bug.
    """
    d = _auto_d("Social norms and energy conservation", ["Descriptive norm", "Control"], "Energy use")
    assert -0.25 < d < -0.02, f"d={d:.2f}"
