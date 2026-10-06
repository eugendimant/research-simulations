"""Tests for the Qualtrics-style export (utils/qualtrics_export.py) and the analysis
scripts that read the delivered files.

The engine dataframe is never altered by the export; only the delivered files change.
"""

import csv
import io
import json
import re
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pytest

from utils import qualtrics_export as qx
from utils.enhanced_simulation_engine import EnhancedSimulationEngine
from utils.qualtrics_export import (
    QUALTRICS_METADATA_COLUMNS,
    build_qualtrics_export,
    export_to_csv_bytes,
    to_qualtrics_raw_csv,
)

SEED = 123
OTHER_SEED = 99
FORBIDDEN = [
    "RUN_ID", "SIMULATION_MODE", "SIMULATION_SEED", "PARTICIPANT_ID", "Trust_mean",
    "_Generation_Source", "Attention_Pass_Rate", "Max_Straight_Line", "Flag_Speed",
    "Flag_Attention", "Flag_StraightLine", "Exclude_Recommended", "Mean_Item_RT_ms",
    "Total_Scale_RT_ms", "Completion_Time_Seconds",
]


def _run_engine(seed: int):
    eng = EnhancedSimulationEngine(
        study_title="Trust Study",
        study_description="Effect of transparency on trust and satisfaction",
        sample_size=60,
        conditions=["Control", "Treatment"],
        factors=[{"name": "Group", "levels": ["Control", "Treatment"]}],
        scales=[
            {"name": "Trust", "num_items": 4, "scale_points": 7, "reverse_items": [2]},
            {"name": "Satisfaction", "num_items": 1, "scale_points": 7},
        ],
        additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        open_ended_questions=[
            {"name": "Feedback", "question_text": "Describe your thoughts on transparency.", "type": "text"}
        ],
        seed=seed,
        missing_data_rate=0.03,
        dropout_rate=0.06,
    )
    try:
        eng.llm_generator.disable_permanently("test: no network")
    except Exception:
        pass
    df, md = eng.generate()
    return eng, df, md


@pytest.fixture(scope="module")
def run():
    eng, df, md = _run_engine(SEED)
    export_df, diag_df = build_qualtrics_export(df, md)
    return eng, df, md, export_df, diag_df


@pytest.fixture(scope="module")
def run_other():
    eng, df, md = _run_engine(OTHER_SEED)
    return df, md


# ---------------------------------------------------------------- structure

def test_forbidden_columns_absent(run):
    _, _, _, export_df, _ = run
    for col in FORBIDDEN:
        assert col not in export_df.columns, col
    assert not [c for c in export_df.columns if c.startswith("ABE3_")]
    assert not [c for c in export_df.columns if c.endswith("_mean")]


def test_required_columns_present_in_order(run):
    _, df, _, export_df, _ = run
    cols = list(export_df.columns)
    assert cols[: len(QUALTRICS_METADATA_COLUMNS)] == QUALTRICS_METADATA_COLUMNS
    assert QUALTRICS_METADATA_COLUMNS[:3] == ["StartDate", "EndDate", "Status"]
    assert "Duration (in seconds)" in cols and "ResponseId" in cols
    rest = cols[len(QUALTRICS_METADATA_COLUMNS):]
    expected = ["CONDITION", "Age", "Gender", "Attention_Check_1", "Trust_1", "Trust_2",
                "Trust_3", "Trust_4", "Satisfaction_1", "Feedback"]
    assert rest == expected
    # original relative order is preserved
    orig = [c for c in df.columns if c in expected]
    assert orig == expected


def test_constant_and_blank_metadata(run):
    _, _, _, e, _ = run
    assert (e["Status"] == 0).all()
    assert (e["DistributionChannel"] == "anonymous").all()
    assert (e["UserLanguage"] == "EN").all()
    for c in ["IPAddress", "RecipientLastName", "RecipientFirstName", "RecipientEmail",
              "ExternalReference", "LocationLatitude", "LocationLongitude"]:
        assert e[c].isin(["", None]).all() or e[c].isna().all(), c


def test_response_id_format_and_uniqueness(run):
    _, _, _, e, _ = run
    assert e["ResponseId"].is_unique
    assert e["ResponseId"].map(lambda s: bool(re.fullmatch(r"R_[A-Za-z0-9]{15}", s))).all()


def test_timestamps_and_duration(run):
    _, _, _, e, _ = run
    fmt = "%Y-%m-%d %H:%M:%S"
    start = pd.to_datetime(e["StartDate"], format=fmt)
    end = pd.to_datetime(e["EndDate"], format=fmt)
    rec = pd.to_datetime(e["RecordedDate"], format=fmt)
    assert start.is_monotonic_increasing, "rows must be sorted by StartDate"
    assert (end >= start).all()
    assert (rec >= end).all()
    assert ((rec - end).dt.total_seconds() <= 5).all()
    diff = (end - start).dt.total_seconds()
    assert (np.abs(diff - e["Duration (in seconds)"].astype(float)) <= 1).all()
    # study period: within roughly the 2 years before the fixed reference date
    assert start.min() >= pd.Timestamp("2023-12-01") and end.max() < pd.Timestamp("2026-01-05")
    assert (start.max() - start.min()) <= pd.Timedelta(days=5)


def test_progress_finished_and_dropouts(run):
    _, _, _, e, _ = run
    fin = e["Finished"].astype(int)
    prog = e["Progress"].astype(int)
    assert set(fin.unique()) <= {0, 1}
    assert (prog[fin == 1] == 100).all()
    assert (prog[fin == 0] < 100).all() and (prog[fin == 0] >= 1).all()
    drop = e[fin == 0]
    assert len(drop) >= 1, "test config (seed 123, dropout_rate 0.06) must produce a visible dropout"
    # dropouts are faster than the median completer and have blank cells after the stop point
    assert drop["Duration (in seconds)"].max() < e.loc[fin == 1, "Duration (in seconds)"].median()
    assert drop["Satisfaction_1"].isna().all() or drop["Trust_4"].isna().all()
    assert (drop["Feedback"].fillna("") == "").all()


def test_diagnostics_alignment(run):
    _, df, _, export_df, diag = run
    assert len(diag) == len(export_df) == len(df)
    assert list(diag["ResponseId"]) == list(export_df["ResponseId"])
    assert list(diag.columns[:5]) == ["ResponseId", "PARTICIPANT_ID", "RUN_ID", "SIMULATION_MODE", "SIMULATION_SEED"]
    assert "Exclude_Recommended" in diag.columns
    for col in ["Trust_mean", "Flag_Speed", "Flag_Attention", "Flag_StraightLine", "Mean_Item_RT_ms",
                "Total_Scale_RT_ms", "_Generation_Source", "ABE3_Education"]:
        assert col in diag.columns, col
    assert not set(diag.columns) & {"Trust_1", "CONDITION", "Feedback"}


def test_response_values_unchanged(run):
    """The export must not alter any response value (only formatting of 6.0 -> 6)."""
    _, df, _, export_df, diag = run
    orig = df.set_index("PARTICIPANT_ID")
    aligned = orig.loc[diag["PARTICIPANT_ID"].to_numpy()].reset_index(drop=True)
    finished = export_df["Finished"].to_numpy() == 1
    for col in ["CONDITION", "Age", "Gender", "Attention_Check_1", "Trust_1", "Trust_2", "Trust_3",
                "Trust_4", "Satisfaction_1", "Feedback"]:
        if col == "Feedback":
            assert (export_df.loc[finished, col].to_numpy() == aligned.loc[finished, col].to_numpy()).all()
        elif pd.api.types.is_numeric_dtype(aligned[col]):
            a = aligned[col].to_numpy(dtype=float)
            b = export_df[col].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
            assert np.array_equal(a, b, equal_nan=True), col
        else:
            assert (aligned[col].to_numpy() == export_df[col].to_numpy()).all(), col
    # the engine dataframe itself is untouched
    assert "Trust_mean" in df.columns and "RUN_ID" in df.columns


def test_csv_prints_integers_without_decimals(run):
    _, _, _, export_df, _ = run
    text = export_to_csv_bytes(export_df).decode("utf-8")
    parsed = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    vals = parsed["Trust_1"][parsed["Trust_1"] != ""]
    assert vals.str.fullmatch(r"\d").all()


# ---------------------------------------------------------------- determinism

def test_deterministic_and_seed_sensitive(run, run_other):
    _, df, md, export_df, diag = run
    again, diag2 = build_qualtrics_export(df.copy(), dict(md))
    assert export_to_csv_bytes(again) == export_to_csv_bytes(export_df)
    assert diag2.to_csv(index=False) == diag.to_csv(index=False)
    other_df, other_md = run_other
    other_export, _ = build_qualtrics_export(other_df, other_md)
    assert export_to_csv_bytes(other_export) != export_to_csv_bytes(export_df)
    assert not set(other_export["ResponseId"]) & set(export_df["ResponseId"])


def test_no_wall_clock_dependence(run, monkeypatch):
    _, df, md, export_df, _ = run
    import datetime as real_dt

    class _NoNow(real_dt.datetime):
        @classmethod
        def now(cls, *a, **k):
            raise AssertionError("wall clock used")

        @classmethod
        def utcnow(cls):
            raise AssertionError("wall clock used")

        @classmethod
        def today(cls):
            raise AssertionError("wall clock used")

    monkeypatch.setattr(qx, "datetime", _NoNow)
    again, _ = build_qualtrics_export(df, md)
    assert export_to_csv_bytes(again) == export_to_csv_bytes(export_df)


# ---------------------------------------------------------------- raw 3-row csv

def test_raw_three_row_csv(run):
    _, _, md, export_df, _ = run
    raw = to_qualtrics_raw_csv(export_df, md["column_descriptions"])
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    assert len(rows) == len(export_df) + 3
    assert rows[0] == list(export_df.columns)
    assert len(rows[1]) == len(rows[2]) == len(rows[0])
    ids = [json.loads(c)["ImportId"] for c in rows[2]]
    assert ids[:3] == ["startDate", "endDate", "status"]
    assert "_recordId" in ids and "userLanguage" in ids
    qids = dict(zip(rows[0], ids))
    assert qids["Trust_1"].startswith("QID") and qids["Trust_4"].startswith("QID")
    assert qids["Trust_1"].split("_")[0] == qids["Trust_4"].split("_")[0]
    assert qids["Feedback"].startswith("QID")
    assert rows[1][rows[0].index("StartDate")] == "Start Date"
    # skipping the two extra header rows gives back the single-header file
    body = pd.read_csv(io.StringIO(raw.decode("utf-8")), skiprows=[1, 2], dtype=str, keep_default_na=False)
    single = pd.read_csv(io.StringIO(export_to_csv_bytes(export_df).decode("utf-8")), dtype=str, keep_default_na=False)
    assert body.equals(single)


# ---------------------------------------------------------------- analysis scripts

def _scripts(eng, df) -> Dict[str, str]:
    return {
        "R": eng.generate_r_export(df),
        "Python": eng.generate_python_export(df),
        "Julia": eng.generate_julia_export(df),
        "SPSS": eng.generate_spss_export(df),
        "Stata": eng.generate_stata_export(df),
    }


def _referenced_columns(lang: str, script: str) -> Set[str]:
    names: Set[str] = set()
    q = r"""['"]([A-Za-z_][A-Za-z0-9_]*)['"]"""
    if lang == "R":
        names |= set(re.findall(r"\b(?:data|diagnostics)\$([A-Za-z_]\w*)", script))
        for blk in re.findall(r"diagnostics\[, c\((.*?)\)\]", script):
            names |= set(re.findall(q, blk))
        names |= set(re.findall(r'by = "(\w+)"', script))
    elif lang == "Python":
        names |= set(re.findall(r"\b(?:data|data_clean|diagnostics)\['([^']+)'\]", script))
        for blk in re.findall(r"\[\[(.*?)\]\]", script):
            names |= set(re.findall(q, blk))
        names |= set(re.findall(r"on='(\w+)'", script))
        names |= set(re.findall(r"\)\['(\w+)'\]\.mean", script))
    elif lang == "Julia":
        names |= set(re.findall(r"\b(?:data|row)\.([A-Za-z_]\w*)", script))
        for blk in re.findall(r"\[:(\w+(?:, :\w+)*)\]", script):
            names |= set(re.findall(r"(\w+)", blk))
        names |= set(re.findall(r"on = :(\w+)", script))
    elif lang == "SPSS":
        for blk in re.findall(r"MEAN\((.*?)\)", script):
            names |= set(blk.split())
        names |= set(re.findall(r"COMPUTE \w+_R = \d+ - (\w+)\.", script))
        names |= set(re.findall(r"AUTORECODE VARIABLES=(\w+)", script))
        names |= set(re.findall(r"SORT CASES BY (\w+)", script))
        names |= set(re.findall(r"BY (\w+)\.", script))
        names |= set(re.findall(r"filter_\$=\((\w+) =", script))
    elif lang == "Stata":
        for blk in re.findall(r"rowmean\((.*?)\)", script):
            names |= set(blk.split())
        names |= set(re.findall(r"gen \w+_r = \d+ - (\w+)", script))
        names |= set(re.findall(r"encode (\w+),", script))
        for blk in re.findall(r"^\s*keep (?!if)(.*)$", script, flags=re.M):
            names |= set(blk.split())
        names |= set(re.findall(r"keep if (\w+) ==", script))
        names |= set(re.findall(r"merge 1:1 (\w+) using", script))
    derived = ("_R", "_r", "_composite", "_num")
    return {n for n in names if not n.endswith(derived) and n not in ("data_clean",)}


def test_scripts_reference_only_existing_columns(run):
    eng, df, _, export_df, diag = run
    existing: List[str] = list(export_df.columns) + list(diag.columns)
    lower = {c.lower() for c in existing}
    scripts = _scripts(eng, df)
    for lang, script in scripts.items():
        refs = _referenced_columns(lang, script)
        assert refs, f"{lang}: no column references extracted (extractor broken)"
        for name in refs:
            if lang == "Stata":
                assert name.lower() in lower, (lang, name)
            else:
                assert name in existing, (lang, name)
    # every script computes the composites itself and joins diagnostics on ResponseId
    assert "Trust_composite" in scripts["R"] and "ResponseId" in scripts["R"]
    assert "Trust_composite" in scripts["Python"] and "ResponseId" in scripts["Python"]
    assert "Trust_composite" in scripts["Julia"] and "ResponseId" in scripts["Julia"]
    assert "Trust_composite" in scripts["SPSS"] and "ResponseId" in scripts["SPSS"]
    assert "trust_composite" in scripts["Stata"] and "responseid" in scripts["Stata"]
    # no script touches the removed composite column
    for lang, script in scripts.items():
        assert "Trust_mean" not in script and "trust_mean" not in script, lang


def test_scripts_read_delivered_filenames(run):
    eng, df, _, _, _ = run
    for lang, script in _scripts(eng, df).items():
        assert "Simulated.csv" not in script, lang
        assert "Simulated_Data.csv" in script, lang
        assert "Simulation_Diagnostics.csv" in script, lang
        assert "preregistered exclusion rules" in script, lang


def test_python_script_runs_against_delivered_files(run, tmp_path):
    eng, df, _, export_df, diag = run
    (tmp_path / "Simulated_Data.csv").write_bytes(export_to_csv_bytes(export_df))
    (tmp_path / "Simulation_Diagnostics.csv").write_bytes(diag.to_csv(index=False).encode("utf-8"))
    script = eng.generate_python_export(df)
    ns: dict = {"__name__": "analysis"}
    import os
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        exec(compile(script, "Python_Prepare_Data.py", "exec"), ns)
    finally:
        os.chdir(cwd)
    data, clean = ns["data"], ns["data_clean"]
    assert "Trust_composite" in data.columns and "Satisfaction_composite" in data.columns
    assert len(data) == len(export_df) and 0 < len(clean) <= len(data)
    # Trust_2 is reverse-keyed on a 7-point scale: the composite must average the recoded item.
    items = export_df[["Trust_1", "Trust_2", "Trust_3", "Trust_4"]].astype("Float64").copy()
    items["Trust_2"] = 8 - items["Trust_2"]
    expected = items.mean(axis=1)
    got = data["Trust_composite"]
    # merge preserves row order; compare on ResponseId to be safe
    got = pd.Series(got.to_numpy(), index=data["ResponseId"])
    exp = pd.Series(expected.to_numpy(dtype=float, na_value=np.nan), index=export_df["ResponseId"])
    assert np.allclose(got.reindex(exp.index).to_numpy(), exp.to_numpy(), equal_nan=True)


def test_explainer_describes_delivered_files(run):
    eng, _, _, _, _ = run
    text = eng.generate_explainer()
    for needle in ["Simulated_Data.csv", "Simulation_Diagnostics.csv", "Simulated_Data_Qualtrics_Raw.csv",
                   "StartDate", "Duration (in seconds)", "Finished", "ResponseId", "Exclude_Recommended"]:
        assert needle in text, needle
    survey_part = text.split("SURVEY COLUMNS (Simulated_Data.csv)")[1].split("DIAGNOSTICS COLUMNS")[0]
    assert "Trust_1" in survey_part and "RUN_ID" not in survey_part and "Trust_mean" not in survey_part


def test_script_composites_use_reverse_recoded_columns(run):
    """Composite lines must average Trust_2_R (Stata: trust_2_r), never the raw reverse item."""
    eng, df, _, _, _ = run
    scripts = _scripts(eng, df)
    for lang, script in scripts.items():
        low = lang == "Stata"
        rec = "Trust_2_r" if low else "Trust_2_R"
        raw = "Trust_2"
        line = next(l for l in script.splitlines()
                    if ("trust_composite" if low else "Trust_composite") in l
                    and ("mean" in l.lower()))
        l2 = line.lower() if low else line
        assert rec.lower() in l2.lower() if low else rec in line, (lang, line)
        import re as _re
        assert not _re.search(r"(?<![\w])" + _re.escape(raw if not low else raw.lower()) + r"(?![\w])", line if not low else line.lower()), (lang, line)
