"""What the instructor and student reports SAY must be true (v1.2.9.1).

Covers: reverse-keyed items in the instructor composites, the effects tables (orientation, sources, caps),
claims the reports used to make without computing them, the key-test table in the markdown, the trait table,
the student summary's settings / seed / data dictionary / speed flag, and HTML escaping of every
user-controlled string with the sanitizer bypassed. Small DataFrames and metadata are built directly, so the
file is fast; one test runs the real engine to pin the report to its output. The statistics-dependent tests
run with scipy and with the numpy fallbacks the deployed app uses (scipy is not in requirements.txt).
"""
import html as html_lib
import itertools
import re
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "simulation_app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import utils.instructor_report as ir  # noqa: E402
from utils.instructor_report import ComprehensiveInstructorReport, InstructorReportGenerator  # noqa: E402


@pytest.fixture(params=["scipy", "numpy-fallback"])
def stats_mode(request, monkeypatch):
    """Run the test with scipy and with the pure-numpy fallbacks."""
    if request.param == "numpy-fallback":
        monkeypatch.setattr(ir, "SCIPY_AVAILABLE", False)
        monkeypatch.setattr(ir, "scipy_stats", None)
    elif not ir.SCIPY_AVAILABLE:
        pytest.skip("scipy is not installed")
    return request.param


# --------------------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------------------
def _study(conditions=("Control", "Treatment"), shifts=None, n=60, k=4, reverse=(), name="Trust", lo=1, hi=7,
           seed=3, with_mean=True):
    """Engine-shaped data: raw item columns (reverse-keyed items stored raw), the engine's rounded scored
    composite ``<name>_mean``, diagnostics columns, and metadata with a scale_generation_log."""
    rng = np.random.RandomState(seed)
    shifts = shifts or {c: 0.9 * i for i, c in enumerate(conditions)}
    rows = []
    for cond in conditions:
        for _ in range(n):
            z = rng.normal(0, 1.0)
            scored = [int(np.clip(np.round(4.0 + shifts[cond] + z + rng.normal(0, 0.8)), lo, hi)) for _ in range(k)]
            raw = [(lo + hi - v) if (j + 1) in reverse else v for j, v in enumerate(scored)]
            row = {"PARTICIPANT_ID": len(rows) + 1, "CONDITION": cond, "Age": int(rng.randint(18, 70)),
                   "Gender": str(rng.choice(["Male", "Female"])), "Attention_Check_1": 2, "Attention_Pass_Rate": 1.0,
                   "Completion_Time_Seconds": int(rng.randint(300, 900)), "Flag_Speed": 0, "Exclude_Recommended": 0}
            row.update({f"{name}_{j + 1}": v for j, v in enumerate(raw)})
            row[f"{name}_mean"] = round(float(np.mean(scored)), 2)
            rows.append(row)
    df = pd.DataFrame(rows)
    if not with_mean:
        df = df.drop(columns=[f"{name}_mean"])
    cols = [f"{name}_{j + 1}" for j in range(k)]
    scale = {"name": name, "variable_name": name, "num_items": k, "scale_points": hi, "scale_min": lo,
             "scale_max": hi, "reverse_items": list(reverse), "type": "likert"}
    meta = {"study_title": "Truthfulness study", "study_description": "Small synthetic study.",
            "conditions": list(conditions), "sample_size": len(df), "factors": [], "open_ended_questions": [],
            "scales": [scale], "run_id": "R1", "generation_timestamp": "2026-10-06T10:00:00",
            "effect_sizes_configured": [], "effect_sizes_observed": [], "seed": seed,
            "scale_generation_log": [{"name": name, "scale_min": lo, "scale_max": hi, "num_items": k,
                                      "reverse_items": sorted(reverse), "columns_generated": cols}]}
    return df, meta


def _pooled_d(a, b):
    n1, n2 = len(a), len(b)
    var = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    return float((a.mean() - b.mean()) / np.sqrt(var))


def _engine_effects(df, meta, high, low, d, direction="positive", source="user"):
    """Add effect metadata shaped like the engine's: configured spec, observed rows (items + composite for
    every pair of conditions, in condition order) and applied contrasts with source / intended d."""
    conds = meta["conditions"]
    sign = 1.0 if direction == "positive" else -1.0
    if d:
        meta["effect_sizes_configured"] = [{"variable": meta["scales"][0]["name"], "factor": "CONDITION", "cohens_d": d,
                                            "direction": direction, "level_high": high, "level_low": low}]
    observed, applied = [], []
    for entry in meta["scale_generation_log"]:
        items = entry["columns_generated"]
        prefix = items[0].rsplit("_", 1)[0]
        cols = items + ([f"{prefix}_mean"] if f"{prefix}_mean" in df.columns else [])
        if len(items) > 1 and cols[-1] != f"{prefix}_mean":
            continue  # typed DV without a composite: nothing at scale level
        for c1, c2 in itertools.combinations(conds, 2):
            for col in cols:
                a, b = df.loc[df.CONDITION == c1, col], df.loc[df.CONDITION == c2, col]
                observed.append({"variable": col, "condition_1": c1, "condition_2": c2, "mean_1": round(float(a.mean()), 3),
                                 "mean_2": round(float(b.mean()), 3), "cohens_d": round(_pooled_d(a, b), 3),
                                 "n_1": len(a), "n_2": len(b)})
            arm = {high: sign * d, low: -sign * d}
            intended = (arm.get(c1, 0.0) - arm.get(c2, 0.0)) / 2.0 if source == "user" else (0.0 if source == "none" else None)
            obs = next(o["cohens_d"] for o in observed if o["variable"] == cols[-1] and o["condition_1"] == c1 and o["condition_2"] == c2)
            applied.append({"variable": prefix, "condition_1": c1, "condition_2": c2, "source": source,
                            "intended_d": None if intended is None else round(intended + 0.0, 3), "observed_d": obs})
    meta["effect_sizes_observed"] = observed
    meta["effect_sizes_applied"] = {"inferred_effects_enabled": source != "none", "contrasts": applied}


def _reports(df, meta, prereg=""):
    comp = ComprehensiveInstructorReport()
    md = comp.generate_comprehensive_report(df=df, metadata=meta, schema_validation=None, prereg_text=prereg, team_info={})
    html = comp.generate_html_report(df=df, metadata=meta, schema_validation=None, prereg_text=prereg, team_info={})
    assert not comp.section_errors, comp.section_errors
    return md, html


def _text(markup: str) -> str:
    markup = re.sub(r"<style.*?</style>", " ", markup, flags=re.S)
    return html_lib.unescape(re.sub(r"(?:\s*\|\s*)+", " | ", re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " | ", markup))))


def _section(markup: str, start: str, end: str) -> str:
    i = markup.index(start)
    j = markup.find(end, i + len(start))
    return markup[i: j if j >= 0 else len(markup)]


# --------------------------------------------------------------------------------------------------
# 1. reverse-keyed items
# --------------------------------------------------------------------------------------------------
def test_composite_recodes_reverse_keyed_items_and_prefers_the_engine_column():
    df, meta = _study(reverse=(1, 3), n=40)
    scale = meta["scales"][0]
    cols = [f"Trust_{i}" for i in range(1, 5)]
    raw_mean = df[cols].mean(axis=1)
    assert abs(raw_mean - df["Trust_mean"]).max() > 0.5, "the fixture must make raw and scored composites differ"
    out = ir._scale_composite(df, scale, cols, meta)
    assert out.equals(pd.to_numeric(df["Trust_mean"]))  # the engine's own (rounded) column is used
    # without the engine column, or with a stale one, the composite is rebuilt from the recoded items
    rebuilt = ir._scale_composite(df.drop(columns=["Trust_mean"]), scale, cols, meta)
    assert abs(rebuilt - df["Trust_mean"]).max() <= 0.0051
    stale = df.copy()
    stale["Trust_mean"] = raw_mean.round(2)
    assert abs(ir._scale_composite(stale, scale, cols, meta) - df["Trust_mean"]).max() <= 0.0051
    # the scale spec alone (no generation log) is enough; a one-item scale is the item itself
    no_log = {k: v for k, v in meta.items() if k != "scale_generation_log"}
    assert abs(ir._scale_composite(df.drop(columns=["Trust_mean"]), scale, cols, no_log) - df["Trust_mean"]).max() <= 0.0051
    single = ir._scale_composite(df, scale, ["Trust_1"], meta)
    assert (single == df["Trust_1"]).all()


def test_composite_item_numbers_come_from_column_names_not_list_position():
    """With 10+ items a lexicographic column order ("_1", "_10", "_2") must not recode the wrong items."""
    df, meta = _study(reverse=(2,), k=11, n=20, with_mean=False)
    cols = sorted(c for c in df.columns if c.startswith("Trust_"))  # Trust_1, Trust_10, Trust_11, Trust_2 ...
    assert cols[1] == "Trust_10"
    scored = df[[f"Trust_{i}" for i in range(1, 12)]].copy()
    scored["Trust_2"] = 8 - scored["Trust_2"]
    out = ir._scale_composite(df, meta["scales"][0], cols, meta)
    assert abs(out - scored.mean(axis=1)).max() < 1e-9


def test_instructor_markdown_and_html_use_the_scored_composite(stats_mode):
    df, meta = _study(reverse=(1, 3), n=80, shifts={"Control": 0.0, "Treatment": 1.2})
    md, html = _reports(df, meta)
    by_cond = df.groupby("CONDITION")["Trust_mean"].mean()
    raw_by_cond = df[[f"Trust_{i}" for i in range(1, 5)]].mean(axis=1).groupby(df["CONDITION"]).mean()
    assert abs((by_cond["Treatment"] - by_cond["Control"]) - (raw_by_cond["Treatment"] - raw_by_cond["Control"])) > 0.4
    block = _section(html, "<h3>Trust</h3>", "<h4>Visualizations</h4>")
    for cond in ("Control", "Treatment"):
        cell = re.search(rf"<tr><td>{cond}</td><td>\d+</td><td>(-?[\d.]+)</td>", block)
        assert cell and abs(float(cell.group(1)) - by_cond[cond]) < 0.0015, (cond, cell and cell.group(1), by_cond[cond])
    md_block = _section(md, "### Trust\n", "### Single") if "### Single" in md else md[md.index("### Trust\n"):]
    mean_line = re.search(r"#### Composite Score \(Mean\)\s*\n\s*- Mean: (-?[\d.]+)", md_block)
    assert mean_line and abs(float(mean_line.group(1)) - df["Trust_mean"].mean()) < 0.0015
    assert "Reverse-keyed items 1, 3" in md_block and "recodes them" in md_block
    d_engine = _pooled_d(df.loc[df.CONDITION == "Treatment", "Trust_mean"], df.loc[df.CONDITION == "Control", "Trust_mean"])
    d_html = float(re.search(r"Effect Size \(Cohen's d\):</strong> (-?[\d.]+)", html).group(1))
    # the HTML's d is one of (control - treatment) or its mirror; its size is the engine's, not the raw-item one
    assert abs(abs(d_html) - abs(d_engine)) < 0.01


def test_reports_equal_the_engines_numbers_for_reverse_keyed_items_end_to_end(stats_mode):
    from utils.enhanced_simulation_engine import EffectSizeSpec, EnhancedSimulationEngine

    scale = {"name": "Trust", "variable_name": "Trust", "type": "likert", "num_items": 5, "scale_points": 7,
             "scale_min": 1, "scale_max": 7, "reverse_items": [1, 3, 5], "_validated": True}
    spec = [EffectSizeSpec(variable="Trust", factor="CONDITION", level_high="Treatment", level_low="Control",
                           cohens_d=0.8, direction="positive")]
    eng = EnhancedSimulationEngine(
        study_title="Reverse", study_description="A study of trust", sample_size=300, conditions=["Control", "Treatment"],
        factors=[], scales=[scale], additional_vars=[], demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        open_ended_questions=[], effect_sizes=spec, seed=3)
    eng.llm_generator.disable_permanently("test")
    df, meta = eng.generate()
    md, html = _reports(df, meta)
    student = InstructorReportGenerator().generate_markdown_report(df=df, metadata=meta, schema_validation=None, prereg_text="", team_info={})
    by_cond = df.groupby("CONDITION")["Trust_mean"].mean()
    block = _section(html, "<h3>Trust</h3>", "<h4>Visualizations</h4>")
    for cond in ("Control", "Treatment"):
        cell = re.search(rf"<tr><td>{cond}</td><td>\d+</td><td>(-?[\d.]+)</td>", block)
        assert abs(float(cell.group(1)) - by_cond[cond]) < 0.0015
    d_engine = _pooled_d(df.loc[df.CONDITION == "Treatment", "Trust_mean"], df.loc[df.CONDITION == "Control", "Trust_mean"])
    d_html = abs(float(re.search(r"Effect Size \(Cohen's d\):</strong> (-?[\d.]+)", html).group(1)))
    d_student = float(re.search(r"\| Trust \| Treatment \| Control \| \+0\.80 \| ([+-][\d.]+) \|", student).group(1))
    assert abs(d_html - abs(d_engine)) < 0.005 and abs(d_student - d_engine) < 0.011 and d_engine > 0.3
    assert "Reverse-keyed items 1, 3, 5" in md


# --------------------------------------------------------------------------------------------------
# 2. effects tables
# --------------------------------------------------------------------------------------------------
def test_html_effects_table_names_the_comparison_keeps_the_sign_and_orients_high_minus_low():
    df, meta = _study(conditions=("Control", "Treatment"), n=70, shifts={"Control": 0.0, "Treatment": 0.9}, name="Rating")
    _engine_effects(df, meta, high="Treatment", low="Control", d=0.8)
    _md, html = _reports(df, meta)
    s6 = _section(html, "<h2>6. Effect Size Verification</h2>", "<h2>7.")
    assert "<td> vs </td>" not in html and "vs </td>" not in s6
    text = _text(s6)
    assert "Rating | Treatment | Control | +0.80 |" in text  # configured: high minus low, positive
    observed = [o for o in meta["effect_sizes_observed"] if o["variable"] == "Rating_mean"][0]["cohens_d"]
    assert observed < 0  # engine orientation: condition 1 (Control) minus condition 2 (Treatment)
    assert f"Control − Treatment | Your specified effect(s) | -0.80 | {observed:.3f}" in text
    assert f"+{-observed:.2f}" in text  # the same effect, oriented high minus low
    assert "item-level rows" in text and "Rating_1" not in s6  # no item-by-pair rows


def test_markdown_effects_section_matches_configured_effects_to_composites():
    df, meta = _study(conditions=("Control", "Treatment"), n=70, shifts={"Control": 0.0, "Treatment": 0.9}, name="Rating")
    _engine_effects(df, meta, high="Treatment", low="Control", d=0.6, direction="negative")
    md, _html = _reports(df, meta)
    s7 = _section(md, "## 7. EFFECT SIZE QUALITY ASSESSMENT", "## 8.")
    assert "Not computed" not in s7 and "N/A" not in s7 and "calibrated from published meta-analyses" not in s7
    row = [ln for ln in s7.splitlines() if ln.startswith("| Rating | Treatment | Control |")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    obs_high_minus_low = -meta["effect_sizes_observed"][-1]["cohens_d"]
    assert cells[3] == "-0.60" and cells[4] == f"{obs_high_minus_low:+.2f}"
    assert cells[5] in ("Within sampling error", "Outside 2 SE of the intended d")  # a verdict, not "n/a"


def test_every_applied_contrast_is_listed_with_its_source_including_implied_ones():
    df, meta = _study(conditions=("Control", "Treatment A", "Treatment B"), n=50,
                      shifts={"Control": 0.0, "Treatment A": 0.8, "Treatment B": 0.3}, name="Rating")
    _engine_effects(df, meta, high="Treatment A", low="Control", d=0.5)
    md, html = _reports(df, meta)
    s7 = _section(md, "### Effects Built Into the Data, by Source", "## 8.")
    rows = [ln for ln in s7.splitlines() if ln.startswith("| Rating |")]
    assert len(rows) == 3 and all("Your specified effect(s)" in r for r in rows)
    got = {tuple(c.strip() for c in r.strip("|").split("|"))[1]: r.strip("|").split("|")[3].strip() for r in rows}
    assert got == {"Control - Treatment A": "-0.50", "Control - Treatment B": "-0.25", "Treatment A - Treatment B": "+0.25"}
    assert "reference level" in s7  # why a contrast with an unnamed condition carries half the effect
    assert "Control &minus; Treatment B" in html and "-0.25" in html


def test_inferred_and_switched_off_effects_are_described_truthfully():
    df, meta = _study(conditions=("Gain frame", "Loss frame"), n=40, name="Rating")
    _engine_effects(df, meta, high="", low="", d=0, source="inferred")
    md, html = _reports(df, meta)
    s7 = _section(md, "## 7. EFFECT SIZE QUALITY ASSESSMENT", "## 8.")
    assert "inferred from the condition names" in s7 and "not calibrated" in s7 and "Inferred from condition names" in s7
    assert "Configured vs Observed" not in s7
    _engine_effects(df, meta, high="", low="", d=0, source="none")
    md2, _ = _reports(df, meta)
    assert "inferred effects were switched off" in md2 and "None built in" in md2


def test_effects_tables_are_capped_at_scale_level_and_state_how_many_rows_were_left_out():
    conds = tuple(f"Group {chr(65 + i)}" for i in range(12))
    df, meta = _study(conditions=conds, n=20, name="S1")
    extra = []
    for name in ("S2", "S3"):
        d2, m2 = _study(conditions=conds, n=20, name=name, seed=9)
        for col in d2.columns:
            if col.startswith(name):
                df[col] = d2[col].to_numpy()
        extra.append(m2["scales"][0])
        meta["scale_generation_log"].extend(m2["scale_generation_log"])
    meta["scales"].extend(extra)
    _engine_effects(df, meta, high="", low="", d=0, source="inferred")
    observed = meta["effect_sizes_observed"]
    scale_rows = len([o for o in observed if o["variable"].endswith("_mean") and np.isfinite(o["cohens_d"])])
    item_rows = len(observed) - scale_rows
    assert scale_rows > 150 and item_rows > 500
    md, html = _reports(df, meta)
    cap = ir._EFFECT_TABLE_CAP
    s7 = _section(md, "### Effects Built Into the Data, by Source", "## 8.")
    assert len([ln for ln in s7.splitlines() if re.match(r"\| S[123] \|", ln)]) == cap
    assert f"Showing the {cap} contrasts with the largest |d| out of {scale_rows} scale-level contrasts; {scale_rows - cap} omitted." in s7
    assert f"{item_rows} item-level rows" in s7
    s6 = _section(html, "<h2>6. Effect Size Verification</h2>", "<h2>7.")
    assert s6.count("<tr><td>S") == cap and f"{scale_rows - cap} omitted" in s6
    # the student summary is capped too and says so
    student = InstructorReportGenerator().generate_markdown_report(df=df, metadata=meta, schema_validation=None, prereg_text="", team_info={})
    obs = _section(student, "### Observed Effects in Generated Data", "**Effect Size Interpretation")
    assert len(re.findall(r"^\| S[123] \|", obs, flags=re.M)) == 12 and f"out of {scale_rows}" in obs


def test_configured_effects_match_levels_case_insensitively_and_factor_levels_use_the_data():
    df, meta = _study(conditions=("Control", "Treatment"), n=70, shifts={"Control": 0.0, "Treatment": 0.9}, name="Rating")
    _engine_effects(df, meta, high="Treatment", low="Control", d=0.6)
    meta["effect_sizes_configured"][0].update(variable="rating", level_high="treatment", level_low="CONTROL")
    row = ir._configured_effect_summary(df, meta)[0]
    assert row["scale"] == "Rating" and row["basis"] == "contrast" and row["intended"] == 0.6
    assert row["observed"] > 0  # high minus low, although the engine stores Control minus Treatment
    # a factor-level effect in a 2x2 has no single contrast: the observed d is computed between the arms
    conds = ("No AI x Hedonic", "No AI x Utilitarian", "AI x Hedonic", "AI x Utilitarian")
    df2, meta2 = _study(conditions=conds, n=60, name="Rating", shifts={conds[0]: 0.0, conds[1]: 0.0, conds[2]: 0.9, conds[3]: 0.9})
    _engine_effects(df2, meta2, high="AI", low="No AI", d=0.8)
    meta2["effect_sizes_configured"][0]["factor"] = "Agent"
    r2 = ir._configured_effect_summary(df2, meta2)[0]
    assert r2["basis"] == "groups" and r2["observed"] > 0.4 and r2["n_1"] == 120 and r2["n_2"] == 120
    assert ir._effect_level_groups(list(conds), "AI", "No AI") == (["AI x Hedonic", "AI x Utilitarian"],
                                                                  ["No AI x Hedonic", "No AI x Utilitarian"])


def test_student_effects_table_keeps_its_columns_and_shows_what_was_built_in():
    df, meta = _study(conditions=("Control", "Treatment"), n=70, shifts={"Control": 0.0, "Treatment": 0.9}, name="DV")
    _engine_effects(df, meta, high="Treatment", low="Control", d=0.8)
    student = InstructorReportGenerator().generate_markdown_report(df=df, metadata=meta, schema_validation=None, prereg_text="", team_info={})
    first = [ln for ln in student.splitlines() if ln.startswith("| DV |") and "Treatment" in ln][0]
    cells = [c.strip() for c in first.strip("|").split("|")]
    assert cells[:4] == ["DV", "Treatment", "Control", "+0.80"] and float(cells[4]) > 0.3
    obs = _section(student, "### Observed Effects in Generated Data", "**Effect Size Interpretation")
    assert "DV_1" not in obs and "Built in as" in obs and "Your specified effect(s)" in obs
    assert "condition 1 minus condition 2" in obs


# --------------------------------------------------------------------------------------------------
# 3. claims
# --------------------------------------------------------------------------------------------------
def _unequal_skewed_study():
    rng = np.random.RandomState(4)
    rows = []
    for cond, scale in (("Control", 0.4), ("Treatment", 3.0)):
        for _ in range(120):
            v = 1.0 + rng.exponential(scale)
            rows.append({"PARTICIPANT_ID": len(rows) + 1, "CONDITION": cond, "Age": 30, "Gender": "Male", "Attention_Pass_Rate": 1.0,
                         "Completion_Time_Seconds": 600, "Exclude_Recommended": 0, "DV_1": float(np.clip(round(v), 1, 7)),
                         "DV_2": float(np.clip(round(v + rng.normal(0, 0.3)), 1, 7))})
    df = pd.DataFrame(rows)
    meta = {"study_title": "Skew", "conditions": ["Control", "Treatment"], "sample_size": len(df), "factors": [], "run_id": "R",
            "scales": [{"name": "DV", "variable_name": "DV", "num_items": 2, "scale_points": 7, "scale_min": 1, "scale_max": 7}]}
    return df, meta


def test_html_states_what_is_computed_not_welch_or_rank_based_tests(stats_mode):
    df, meta = _unequal_skewed_study()
    md, html = _reports(df, meta)
    assert "Variance homogeneity: not met" in html and "Normality (pooled across conditions): not met" in html
    for false_claim in ("Welch's correction applied", "Non-parametric tests also reported", "Mann-Whitney", "Kruskal"):
        assert false_claim not in html and false_claim not in md, false_claim
    assert "pooled-variance t-test shown above assumes equal variances" in html
    assert "no rank-based test is shown in this report" in html


def test_markdown_has_the_key_test_table_with_t_df_p_and_d(stats_mode):
    df, meta = _study(conditions=("Control", "Treatment"), n=70, shifts={"Control": 0.0, "Treatment": 0.7})
    md, html = _reports(df, meta)
    row = [ln for ln in md.splitlines() if ln.startswith("| Trust | Control - Treatment")][0]
    a, b = df.loc[df.CONDITION == "Control", "Trust_mean"], df.loc[df.CONDITION == "Treatment", "Trust_mean"]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    t = (a.mean() - b.mean()) / (sp * np.sqrt(1 / len(a) + 1 / len(b)))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert "pooled-variance t-test" in cells[1]
    assert cells[2] == f"t = {t:+.2f}" and cells[3] == str(len(a) + len(b) - 2)
    assert cells[5] == f"d = {_pooled_d(a, b):+.2f}" and cells[6] in {"negligible", "very small", "small", "medium", "large", "very large"}
    p_cell = cells[4]
    assert p_cell == "< .001" or 0.0 <= float(p_cell) <= 1.0
    if ir.SCIPY_AVAILABLE:
        from scipy import stats as sps
        p = float(sps.ttest_ind(a, b, equal_var=True).pvalue)
        assert p_cell == "< .001" if p < 0.001 else abs(float(p_cell) - p) < 0.006  # numpy fallback is approximate
    assert "not corrected for the number of DVs" in md
    # three conditions: the omnibus test with its degrees of freedom
    df3, meta3 = _study(conditions=("A", "B", "C"), n=40)
    md3, _ = _reports(df3, meta3)
    row3 = [ln for ln in md3.splitlines() if ln.startswith("| Trust | omnibus, 3 conditions")][0]
    assert "one-way ANOVA" in row3 and "F = " in row3 and "| 2, 117 |" in row3 and "η² = " in row3


def _trait_block(html: str) -> str:
    start = html.index("<h4>Personality Profiles by Condition</h4>")
    end = re.search(r"<h[24]>|<a id=", html[start + 10:])
    return html[start: start + 10 + end.start()] if end else html[start:]


def test_trait_table_compares_only_traits_recorded_for_every_condition():
    df, meta = _study(conditions=("Alpha", "Beta"), n=30)
    meta["trait_averages_by_condition"] = {
        "Alpha": {"acquiescence": 0.52, "extremity": 0.31, "brand_attachment": 0.64},
        "Beta": {"acquiescence": 0.55, "extremity": 0.29, "price_sensitivity": 0.44}}
    _md, html = _reports(df, meta)
    block = _trait_block(html)
    assert "Acquiescence" in block and "Extremity" in block and "0.520" in block and "0.290" in block
    assert "Brand Attachment" not in block and "Price Sensitivity" not in block and "0.000" not in block
    assert "2 further trait(s) were recorded for only some conditions" in block
    meta["trait_averages_by_condition"] = {"Alpha": {"a": 0.0}, "Beta": {"a": 0.0}}
    _md, html2 = _reports(df, meta)
    assert "0.000" not in _trait_block(html2) and "No trait was recorded for every condition" in _trait_block(html2)


# --------------------------------------------------------------------------------------------------
# 4. student summary
# --------------------------------------------------------------------------------------------------
def _student(df, meta):
    return InstructorReportGenerator().generate_markdown_report(df=df, metadata=meta, schema_validation=None, prereg_text="", team_info={})


def test_settings_block_prints_only_what_was_recorded():
    df, meta = _study(n=40)
    report = _student(df, meta)
    block = _section(report, "## Simulation Settings (Transparency)", "## Experimental Design Summary")
    for invented in ("Gender quota (% male) | 50%", "Age mean | 35", "Age standard deviation | 12", "| 85% |", "| 5% |"):
        assert invented not in block, invented
    assert block.count("_Not recorded in this run's metadata._") == 2
    assert re.search(r"\| Age \| M = \d+\.\d, SD = \d+\.\d, range \d+ to \d+ \|", block)  # observed, labelled as such
    assert "| **Random Seed** | 3 |" in report and "Auto" not in report.split("## Study Overview")[0]
    meta.update(demographics={"gender_quota": 30, "age_mean": 22, "age_sd": 3, "age_min": 18, "age_max": 30},
                attention_rate=0.7, random_responder_rate=0.2,
                exclusion_criteria={"completion_time_min_seconds": 120, "completion_time_max_seconds": 900,
                                    "straight_line_threshold": 6, "duplicate_ip_check": False})
    block2 = _section(_student(df, meta), "## Simulation Settings (Transparency)", "## Experimental Design Summary")
    for shown in ("| Gender quota (% male) | 30% |", "| Age mean | 22 |", "| Age standard deviation | 3 |", "| Age range | 18 to 30 |",
                  "| Attention check pass rate (setting) | 70% |", "| Random responder rate (setting) | 20% |",
                  "| Min completion time | 120 seconds |", "| Max completion time | 900 seconds |",
                  "| Straight-line threshold | 6 items |", "| Duplicate IP check | No |"):
        assert shown in block2, shown
    assert "_Not recorded" not in block2.split("### Exclusion Criteria")[0]


def _attention_frame(conds, keyed, fail_every=5):
    rows = []
    for cond in conds:
        for i in range(20):
            passed = (i % fail_every) != 0
            rows.append({"CONDITION": cond, "Attention_Check_1": keyed[cond] if passed else 3 - keyed[cond],
                         "Attention_Pass_Rate": 1.0 if passed else 0.0})
    return pd.DataFrame(rows)


def test_data_dictionary_states_the_real_attention_check_coding():
    engine_text = "Manipulation/attention check: 1=Correct, 2=Incorrect"
    df = _attention_frame(["Control", "AI"], {"Control": 2, "AI": 1})
    text = ir._attention_check_description(df)
    assert "not a pass/fail code" in text and "option 1 is correct in AI; option 2 is correct in Control" in text
    assert "Attention_Pass_Rate holds the result (1 = passed, 0 = failed)" in text
    assert "1=Correct" not in ir._dictionary_description("Attention_Check_1", engine_text, df)
    only2 = ir._attention_check_description(_attention_frame(["A", "B"], {"A": 2, "B": 2}))
    assert "2 = correct, 1 = incorrect" in only2 and "NOT 'correct'" in only2
    only1 = ir._attention_check_description(_attention_frame(["A", "B"], {"A": 1, "B": 1}))
    assert "1 = correct, 2 = incorrect" in only1
    # a column that is not the simulator's two-option check is left alone
    odd = df.copy()
    odd["Attention_Check_1"] = np.arange(len(odd)) % 7
    assert ir._attention_check_description(odd) is None
    assert ir._dictionary_description("Attention_Check_1", engine_text, odd) == engine_text
    # both documents use it
    study, meta = _study(conditions=("Control", "AI"), n=20)
    study["Attention_Check_1"] = np.where(study["CONDITION"] == "AI", 1, 2)
    meta["column_descriptions"] = {"Attention_Check_1": engine_text}
    assert "option 1 is correct in AI" in _student(study, meta)
    _md, html = _reports(study, meta)
    assert "1=Correct" not in html and "option 1 is correct in AI" in html


def test_speed_flag_count_agrees_with_the_flag_column():
    times = [480.0, 520.0, 600.0, 640.0, 700.0, 706.9, 760.0, 800.0, 900.0, 1200.0]
    df = pd.DataFrame({"Completion_Time_Seconds": times, "Flag_Speed": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]})
    text = "\n".join(ir._speed_flag_lines(df, {}))
    assert "Speed-flagged (`Flag_Speed` = 1): 1" in text and "below the minimum or above the maximum" in text
    assert "below 60 s: 0; above 1800 s: 0" in text and "(default window)" in text
    assert "1 flagged participant(s) have a recorded time (707 to 707 s)" in text  # provably not produced by the rule
    assert "Suspiciously fast (<60s): 0" not in text
    # with recorded thresholds the rule is quoted and checked against them
    cfg = {"exclusion_criteria": {"completion_time_min_seconds": 100, "completion_time_max_seconds": 650}}
    df2 = df.copy()
    df2["Flag_Speed"] = [0, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    text2 = "\n".join(ir._speed_flag_lines(df2, cfg))
    assert "(100 s to 650 s)" in text2 and "by default" not in text2 and "above 650 s: 6" in text2
    assert "1 flagged participant(s) have a recorded time (520 to 520 s)" in text2
    assert "Flag_Speed` / `Exclude_Recommended` as the exclusion rule" in text2
    consistent = df.copy()
    consistent["Flag_Speed"] = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # exactly the participants above 650 s
    assert "Note:" not in "\n".join(ir._speed_flag_lines(consistent, cfg))
    clean = pd.DataFrame({"Completion_Time_Seconds": [30, 90, 2000], "Flag_Speed": [1, 0, 1]})
    assert "Note:" not in "\n".join(ir._speed_flag_lines(clean, {}))
    d, meta = _study(n=20)
    assert "Speed-flagged (`Flag_Speed` = 1): 0" in _reports(d, meta)[0]


# --------------------------------------------------------------------------------------------------
# 5. HTML escaping (sanitizer bypassed so the escaping itself is tested)
# --------------------------------------------------------------------------------------------------
HOSTILE = {
    "title": "<script>alert('t')</script> Title & </title><b>x</b>",
    "team": "<img src=x onerror=alert(2)> Team \"Q\" & Co",
    "members": "Ann <b>A</b>\nBob <iframe src=//evil></iframe>",
    "scale": "Trust <i onmouseover=alert(3)>scale</i> & more",
    "scale2": "Rank <script>alert(9)</script> items",
    "cond": ["Cond <A> & \"x\"", "B <img src=x onerror=alert(4)>", "C'D </td><td>", "E <svg onload=alert(5)>"],
    "factor": "Fac <u>tor</u> & 1",
    "oe": "Why <script>alert(6)</script> & more? " + "word " * 40,
    "level": "Hi <b>gh</b>",
}


class _Tags(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags, self.attrs, self.text = Counter(), [], []

    def handle_starttag(self, tag, attrs):
        self.tags[tag] += 1
        self.attrs.extend(k for k, _ in attrs)

    def handle_data(self, data):
        self.text.append(data)


def _parse(markup: str) -> _Tags:
    p = _Tags()
    p.feed(markup)
    return p


def _hostile_study(hostile: bool):
    fl1 = ["F1 <b>a</b>", "F2 & b"] if hostile else ["F1 a", "F2 b"]
    fl2 = ["G1 <img src=x onerror=alert(4)>", "G2 'd' \"q\""] if hostile else ["G1 c", "G2 d"]
    conds = [f"{a} x {b}" for a, b in itertools.product(fl1, fl2)]
    name = HOSTILE["scale"] if hostile else "Trust scale"
    name2 = HOSTILE["scale2"] if hostile else "Rank items"
    df, meta = _study(conditions=conds, n=25, name="Trust", seed=11)
    meta["scales"][0]["name"] = name
    meta["scale_generation_log"][0]["name"] = name
    # a constant-sum DV exercises the "no variation" block
    for j in (1, 2, 3):
        df[f"Alloc_{j}"] = [40, 35, 25][j - 1]
    meta["scales"].append({"name": name2, "variable_name": "Alloc", "num_items": 3, "scale_points": 100, "scale_min": 0,
                           "scale_max": 100, "type": "constant_sum", "reverse_items": []})
    meta["scale_generation_log"].append({"name": name2, "scale_min": 0, "scale_max": 100, "num_items": 3, "reverse_items": [],
                                         "columns_generated": ["Alloc_1", "Alloc_2", "Alloc_3"]})
    meta["factors"] = [{"name": HOSTILE["factor"] if hostile else "Factor 1", "levels": fl1},
                       {"name": "Second", "levels": fl2}]
    meta["study_title"] = HOSTILE["title"] if hostile else "Title"
    meta["study_description"] = HOSTILE["oe"] if hostile else "Description"
    meta["open_ended_questions"] = [{"name": "Q1", "variable_name": "Q1", "question_text": HOSTILE["oe"] if hostile else "Why? " + "word " * 40}]
    meta["llm_response_stats"] = {}
    meta["llm_init_error"] = "boom <script>alert(7)</script>" if hostile else "boom"
    meta["generation_method_label"] = "Method <b>x</b>" if hostile else "Method x"
    meta["generation_warnings"] = ["Warn <img src=x onerror=alert(8)>" if hostile else "Warn"]
    meta["column_descriptions"] = {"Trust_1": "item <script>alert(1)</script> & \"q\"" if hostile else "item",
                                   "Col <b>x</b>" if hostile else "ColX": "desc"}
    meta["persona_by_condition"] = {"counts": {c: {"engaged responder": 5, "satisficer <b>x</b>" if hostile else "satisficer": 2}
                                               for c in conds}, "proportions": {}}
    meta["persona_distribution"] = {"counts": {"engaged responder": 5}, "proportions": {"engaged responder": 0.7, "weird <b>p</b>" if hostile else "weird": 0.3}}
    meta["trait_averages_by_condition"] = {c: {"acquiescence": 0.5, "x <b>y</b>" if hostile else "xy": 0.4} for c in conds}
    _engine_effects(df, meta, high=conds[0], low=conds[1], d=0.5)
    meta["effect_sizes_configured"][0]["variable"] = name
    meta["exclusion_summary"] = {"flagged_speed": 0, "flagged_attention": 0, "flagged_straightline": 0, "total_excluded": 0}
    return df, meta


def test_hostile_text_is_escaped_in_every_html_section_without_the_sanitizer(monkeypatch, stats_mode):
    monkeypatch.setattr(ir, "_harden_report_html", lambda document: document)  # the safety net is not what is tested
    team = {"team_name": HOSTILE["team"], "team_members": HOSTILE["members"]}
    comp = ComprehensiveInstructorReport()
    docs = {}
    for hostile in (False, True):
        df, meta = _hostile_study(hostile)
        docs[hostile] = comp.generate_html_report(df=df, metadata=meta, schema_validation=None, prereg_text="",
                                                  team_info=team if hostile else {"team_name": "Team", "team_members": "A\nB"})
        assert not comp.section_errors, comp.section_errors
    control, attack = _parse(docs[False]), _parse(docs[True])
    # hostile strings add no element and no attribute that the same report with harmless names does not have
    assert not (set(attack.tags) - set(control.tags)), set(attack.tags) - set(control.tags)
    assert not [a for a in attack.attrs if a.lower().startswith("on")]
    assert not [t for t in ("script", "iframe", "img", "object", "embed", "form") if attack.tags[t] > control.tags[t]]
    assert set(attack.attrs) <= set(control.attrs), set(attack.attrs) - set(control.attrs)
    visible = " ".join(attack.text)
    for needle in (HOSTILE["title"], "F1 <b>a</b> x G1 <img src=x onerror=alert(4)>", "F2 & b x G2 'd' \"q\"", HOSTILE["scale"], HOSTILE["factor"],
                   "<script>alert(6)</script>", "boom <script>alert(7)</script>", "Method <b>x</b>", "Warn <img src=x onerror=alert(8)>",
                   HOSTILE["team"], "satisficer <b>x</b>".title(), "weird <b>p</b>".title()):
        assert needle in visible, needle
    assert "<title>Instructor Report: &lt;script&gt;" in docs[True]
    # the same strings are plain text in the <title>, not a way out of it
    assert _parse(docs[True]).tags["title"] == 1


def test_open_ended_text_is_truncated_as_text_and_then_escaped():
    df, meta = _study(n=20)
    qualtrics = '<p style="margin:0"><span class="x">' + "Please describe &amp; explain " * 8 + "</span></p><br>"
    meta["open_ended_questions"] = [{"name": "Q1", "variable_name": "Q1", "question_text": qualtrics}]
    _md, html = _reports(df, meta)
    item = re.search(r"<li>(&lt;p style.*?)<code>", html, flags=re.S).group(1)
    assert item.startswith("&lt;p style=&quot;margin:0&quot;&gt;")  # escaped, not interpreted
    assert not re.findall(r"&(?!(?:[a-z]+|#\d+|#x[0-9a-f]+);)", item), "an entity was cut in half or left bare"
    assert len(_parse("<p>" + item + "</p>").text[0].rstrip()) == 120  # 120 characters of the question text, whole
