"""The analysis snippets/scripts embedded in the instructor report (User_Study_Summary)
must run against the delivered files: Simulated_Data.csv (item columns only) and
Simulation_Diagnostics.csv (Exclude_Recommended, keyed on ResponseId)."""

import re
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from utils.enhanced_simulation_engine import EnhancedSimulationEngine
from utils.instructor_report import InstructorReportGenerator
from utils.qualtrics_export import build_qualtrics_export, export_to_csv_bytes

SCALES = [
    {"name": "Trust scale", "num_items": 4, "scale_points": 7, "reverse_items": [2]},
    {"name": "Satisfaction", "num_items": 3, "scale_points": 5, "reverse_items": [1, 3]},
]
DEMO = {"gender_quota": 50, "age_mean": 35, "age_sd": 12}


def _run(conditions: List[str], factors: List[Dict], n: int = 60, seed: int = 7):
    eng = EnhancedSimulationEngine(
        study_title="Script Study",
        study_description="Effect of transparency on trust and satisfaction",
        sample_size=n,
        conditions=conditions,
        factors=factors,
        scales=[dict(s) for s in SCALES],
        additional_vars=[],
        demographics=DEMO,
        open_ended_questions=[],
        seed=seed,
        missing_data_rate=0.03,
        dropout_rate=0.0,
    )
    eng.llm_generator.disable_permanently("test")
    df, md = eng.generate()
    export_df, diag_df = build_qualtrics_export(df, md)
    md_text = InstructorReportGenerator().generate_markdown_report(df=df, metadata=md)
    return df, md, export_df, diag_df, md_text


@pytest.fixture(scope="module")
def two_group():
    return _run(["Control", "Treatment"], [{"name": "Group", "levels": ["Control", "Treatment"]}])


@pytest.fixture(scope="module")
def three_group():
    return _run(["Low", "Medium", "High"], [{"name": "Dose", "levels": ["Low", "Medium", "High"]}])


@pytest.fixture(scope="module")
def factorial():
    return _run(
        ["A x X", "A x Y", "B x X", "B x Y"],
        [{"name": "Group", "levels": ["A", "B"]}, {"name": "Mode", "levels": ["X", "Y"]}],
        n=80,
    )


def _blocks(md_text: str, lang: str) -> List[str]:
    return re.findall(r"```%s\n(.*?)```" % lang, md_text, flags=re.S)


def _write_files(tmp_path, export_df, diag_df):
    (tmp_path / "Simulated_Data.csv").write_bytes(export_to_csv_bytes(export_df))
    diag_df.to_csv(tmp_path / "Simulation_Diagnostics.csv", index=False)


def _run_code(code: str, tmp_path, monkeypatch) -> dict:
    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "snippet"}
    exec(compile(code, "<snippet>", "exec"), ns)
    return ns


def _prep_only(code: str) -> str:
    """numpy/pandas-only part: drop scipy/statsmodels imports and the tests after the prep."""
    keep = []
    for line in code.splitlines():
        if re.match(r"\s*(from|import)\s+(scipy|statsmodels)", line) or "scipy" in line.split("#")[0] and "import" in line:
            continue
        if line.startswith("# --- SECTION 5"):
            break
        keep.append(line)
    return "\n".join(keep)


def _expected_composite(export_df: pd.DataFrame, items: List[str], flip=None, rev=()) -> pd.Series:
    cols = []
    for i, it in enumerate(items, start=1):
        s = export_df[it].astype(float)
        cols.append(flip - s if i in rev else s)
    return pd.concat(cols, axis=1).mean(axis=1)


# ---------------------------------------------------------------- execution

def test_python_script_runs_and_recodes(two_group, tmp_path, monkeypatch):
    _, md, export_df, diag_df, md_text = two_group
    _write_files(tmp_path, export_df, diag_df)
    code = _prep_only(_blocks(md_text, "python")[-1])
    ns = _run_code(code, tmp_path, monkeypatch)
    df = ns["df"]
    exp_trust = _expected_composite(export_df, [f"Trust_scale_{i}" for i in range(1, 5)], 8, (2,))
    exp_sat = _expected_composite(export_df, [f"Satisfaction_{i}" for i in range(1, 4)], 6, (1, 3))
    np.testing.assert_allclose(df["Trust_scale_composite"].to_numpy(float), exp_trust.to_numpy(float), equal_nan=True)
    np.testing.assert_allclose(df["Satisfaction_composite"].to_numpy(float), exp_sat.to_numpy(float), equal_nan=True)
    # reversing changed the composite (i.e. it is not the naive mean)
    naive = export_df[[f"Trust_scale_{i}" for i in range(1, 5)]].astype(float).mean(axis=1)
    assert not np.allclose(naive.to_numpy(), exp_trust.to_numpy(), equal_nan=True)
    # exclusions come from the diagnostics file, joined on ResponseId
    n_keep = int((diag_df["Exclude_Recommended"] == 0).sum())
    assert len(ns["df_clean"]) == n_keep
    assert set(ns["df_clean"]["ResponseId"]) == set(diag_df.loc[diag_df["Exclude_Recommended"] == 0, "ResponseId"])


def test_python_script_without_diagnostics_file(two_group, tmp_path, monkeypatch):
    _, _, export_df, _, md_text = two_group
    (tmp_path / "Simulated_Data.csv").write_bytes(export_to_csv_bytes(export_df))
    ns = _run_code(_prep_only(_blocks(md_text, "python")[-1]), tmp_path, monkeypatch)
    assert len(ns["df_clean"]) == len(export_df)


def test_full_python_script_runs_with_scipy(two_group, tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    _, _, export_df, diag_df, md_text = two_group
    _write_files(tmp_path, export_df, diag_df)
    ns = _run_code(_blocks(md_text, "python")[-1], tmp_path, monkeypatch)
    assert "p_val" in ns


@pytest.mark.parametrize("fixture_name", ["two_group", "three_group"])
def test_quickstart_python_runs_when_scipy_available(fixture_name, request, tmp_path, monkeypatch):
    pytest.importorskip("scipy")
    _, _, export_df, diag_df, md_text = request.getfixturevalue(fixture_name)
    _write_files(tmp_path, export_df, diag_df)
    quick = _blocks(md_text, "python")[0]  # quick-start precedes the full script
    ns = _run_code(quick, tmp_path, monkeypatch)
    assert "p_val" in ns and np.isfinite(ns["p_val"])


@pytest.mark.parametrize("fixture_name", ["two_group", "three_group", "factorial"])
def test_quickstart_prep_runs_with_numpy_pandas_only(fixture_name, request, tmp_path, monkeypatch):
    _, _, export_df, diag_df, md_text = request.getfixturevalue(fixture_name)
    _write_files(tmp_path, export_df, diag_df)
    quick = _blocks(md_text, "python")[0]
    prep = []
    for line in quick.splitlines():
        if re.match(r"\s*(from|import)\s+.*(scipy|statsmodels)", line):
            line = "import os; import pandas as pd; import numpy as np" if line.startswith("import os") else ""
        if line.startswith(("g1 =", "model =", "groups =")):
            break
        prep.append(line)
    ns = _run_code("\n".join(prep), tmp_path, monkeypatch)
    assert "Trust_scale_composite" in ns["df_clean"].columns
    assert ns["df_clean"]["Trust_scale_composite"].notna().any()
    assert len(ns["df_clean"]) == int((diag_df["Exclude_Recommended"] == 0).sum())


def test_factorial_factor_columns_derived(factorial, tmp_path, monkeypatch):
    _, _, export_df, diag_df, md_text = factorial
    _write_files(tmp_path, export_df, diag_df)
    quick = _blocks(md_text, "python")[0]
    assert "C(Group) * C(Mode)" in quick
    prep = "\n".join(l for l in quick.splitlines()
                     if not re.match(r"\s*(from|import)\s+.*(scipy|statsmodels)", l) or l.startswith("import os"))
    prep = prep.split("model =")[0].replace("import os; import pandas as pd; import statsmodels.api as sm",
                                           "import os; import pandas as pd")
    ns = _run_code(prep, tmp_path, monkeypatch)
    assert set(ns["df"]["Group"].dropna()) == {"A", "B"}
    assert set(ns["df"]["Mode"].dropna()) == {"X", "Y"}


# ------------------------------------------------- static checks, all languages

LANGS = ["r", "python", "spss", "stata"]


@pytest.mark.parametrize("fixture_name", ["two_group", "three_group", "factorial"])
def test_no_nonexistent_columns_anywhere(fixture_name, request):
    _, _, export_df, _, md_text = request.getfixturevalue(fixture_name)
    cols = {c.lower() for c in export_df.columns}
    blocks = [(lang, b) for lang in LANGS for b in _blocks(md_text, lang)]
    assert len(blocks) >= 6
    for lang, code in blocks:
        assert not re.search(r"\b\w+_mean\b", code), (lang, code)
        assert "Outcome_" not in code and "Name_mean" not in code
        if re.search(r"exclude_recommended", code, flags=re.I):
            assert "Simulation_Diagnostics.csv" in code, (lang, code)
            assert re.search(r"responseid", code, flags=re.I), (lang, code)
        # every item column referenced exists in the delivered CSV (or is a recode of one)
        for tok in set(re.findall(r"\b(?:Trust_scale|Satisfaction|trust_scale|satisfaction)_\d+(?:_[Rr])?\b", code)):
            base = re.sub(r"_[Rr]$", "", tok)
            assert base.lower() in cols, (lang, tok)
            if tok != base:
                assert re.search(r"[-]\s*\S*" + re.escape(base), code, flags=re.I) or f"{tok} =" in code or f"{tok}'] =" in code \
                    or re.search(re.escape(tok) + r"\s*(<-|=)", code), (lang, tok)
        # nothing reads Simulation_Diagnostics-only columns from the main CSV
        for forbidden in ("Completion_Time_Seconds", "Flag_Speed", "PARTICIPANT_ID"):
            assert forbidden not in code, (lang, forbidden)


def test_reverse_keyed_items_recoded_in_every_language(two_group):
    _, _, _, _, md_text = two_group
    py = _blocks(md_text, "python")[-1]
    assert "df['Trust_scale_2_R'] = 8 - df['Trust_scale_2']" in py
    assert "df['Satisfaction_1_R'] = 6 - df['Satisfaction_1']" in py
    assert "'Trust_scale_2_R'" in py and "'Trust_scale_3'" in py
    r = _blocks(md_text, "r")[-1]
    assert "df$Trust_scale_2_R <- 8 - df$Trust_scale_2" in r
    assert "df$Trust_scale_2_R" in r.split("rowMeans(")[1].split(")")[0]
    spss = _blocks(md_text, "spss")[-1]
    assert "COMPUTE Trust_scale_2_R = 8 - Trust_scale_2." in spss
    assert "MEAN(Trust_scale_1 Trust_scale_2_R Trust_scale_3 Trust_scale_4)" in spss
    stata = _blocks(md_text, "stata")[-1]
    assert "gen trust_scale_2_r = 8 - trust_scale_2" in stata
    assert "rowmean(trust_scale_1 trust_scale_2_r trust_scale_3 trust_scale_4)" in stata
    assert "merge 1:1 responseid" in stata and "satisfaction_composite" in stata


@pytest.mark.parametrize("fixture_name", ["two_group", "three_group", "factorial"])
def test_all_python_blocks_compile(fixture_name, request):
    _, _, _, _, md_text = request.getfixturevalue(fixture_name)
    for code in _blocks(md_text, "python"):
        compile(code, "<snippet>", "exec")


def test_quickstart_uses_computed_composite_name(two_group, three_group):
    for fx in (two_group, three_group):
        md_text = fx[4]
        quick_py = _blocks(md_text, "python")[0]
        quick_r = _blocks(md_text, "r")[0]
        assert "Trust_scale_composite" in quick_py and "Trust_scale_composite" in quick_r
        assert "Outcome_mean" not in md_text
