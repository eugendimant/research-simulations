#!/usr/bin/env python3
"""Deep output-data inspection for the 10 most recently uploaded student QSFs.
Parses → bridges → simulates → inspects the OUTPUT DATA for the kinds of bugs
students report: missing/fabricated DVs, NaN/blank cells, out-of-range values,
poor open-text, condition imbalance, straightlining, duplicate columns.

Run:  python3 tests/student_qsf_inspect.py [N]   (default N=60)
"""
import os, sys, glob, traceback
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from utils.qsf_preview import QSFPreviewParser
from utils.enhanced_simulation_engine import EnhancedSimulationEngine
import app

HERE = os.path.dirname(os.path.abspath(__file__))
QSF_DIR = os.path.join(HERE, "..", "simulation_app", "example_files")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

STUDENT_FILES = [
    "2026_05_05_Morality_follow_up_II_1_.qsf", "2026_05_05_Morality_follow_up_II.qsf",
    "2026_05_01_5010_group_8.qsf", "2026_04_30_5010_group_8.qsf",
    "2026_04_27_BDS5010_group02.qsf", "2026_04_25_Brand_Name_and_Consumer_Attitude_5010.qsf",
    "2026_04_25_The_Interactive_Eects_of_Brand_Nickname_Usage_and_Source_on.qsf",
    "2026_04_24_5010_Pilot_Testing_3_.qsf", "2026_04_23_5010_Pilot_Testing_3_.qsf",
    "2026_04_23_Group6_Charity_Review_Experiment.qsf",
]

BARE = {"it", "this", "that", "this topic", "the topic", ""}
META_TELLS = ["the survey", "this survey", "well-designed", "well designed", "the questionnaire"]

issues_global = []

def inspect(name, df, inp, preview):
    local = []
    cols = list(df.columns)
    # 1. duplicate column names
    dupes = [c for c in set(cols) if cols.count(c) > 1]
    if dupes:
        local.append(f"DUPLICATE columns: {dupes[:5]}")
    # 2. fabricated DV (real DVs dropped)
    fab = len(inp["scales"]) == 1 and inp["scales"][0].get("detected_from_qsf") is False
    n_slider = len(preview.slider_questions or [])
    n_text = len(preview.text_entry_questions or [])
    if fab and (n_slider or n_text or preview.total_questions > 10):
        local.append(f"DV FABRICATED (generic Main_DV) though survey has "
                     f"{n_slider} sliders / {n_text} text-entry — real DV likely missing from output")
    # 3. CONDITION column + balance
    cond_col = next((c for c in cols if c.lower() in ("condition", "conditions", "_condition")), None)
    if cond_col is None:
        local.append("no CONDITION column in output")
    else:
        vc = df[cond_col].value_counts()
        if len(vc) >= 2:
            ratio = vc.max() / max(1, vc.min())
            if ratio > 3.0:
                local.append(f"condition imbalance {dict(vc)} (max/min={ratio:.1f})")
    # 4. REAL DV columns only (matched to engine scale inputs) — NaN density,
    # straightlining, and out-of-range vs each scale's DECLARED min/max. Metadata,
    # timing, demographic and flag columns are intentionally excluded so we don't
    # false-flag constant SIMULATION_SEED, *_RT_ms timing, or -3..3 ideology cols.
    dv_ranges = {}
    for sc in inp["scales"]:
        vn = str(sc.get("variable_name") or sc.get("name") or "").strip().lower()
        if vn:
            try:
                dv_ranges[vn] = (float(sc.get("scale_min", 1)), float(sc.get("scale_max", 7)))
            except (TypeError, ValueError):
                dv_ranges[vn] = (1.0, 7.0)
    META = ("simulation_seed", "_seed", "_rt", "rt_ms", "_ms", "duration", "timing",
            "flag_", "exclude", "straight_line", "straightline", "abe3_", "response_id",
            "responseid", "progress", "finished", "speed", "attention", "manipulation",
            "comprehension", "age", "gender", "education", "income", "ethnicity", "race",
            "ideology", "party", "_id")

    def _dv_range(col):
        cl = col.lower()
        if any(t in cl for t in META):
            return None
        base = cl.rsplit("_", 1)[0] if ("_" in cl and cl.rsplit("_", 1)[-1].isdigit()) else cl
        if cl in dv_ranges:
            return dv_ranges[cl]
        if base in dv_ranges:
            return dv_ranges[base]
        for d, rng in dv_ranges.items():
            if cl.startswith(d):
                return rng
        return None

    nan_cols, const_cols, oor_cols = [], [], []
    for c in cols:
        rng = _dv_range(c)
        if rng is None or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        arr = pd.to_numeric(df[c], errors="coerce")
        nn = arr.notna().sum()
        if nn == 0:
            continue
        if 0.05 < arr.isna().mean() < 1.0:
            nan_cols.append(f"{c}({arr.isna().mean():.0%})")
        if nn >= 5 and arr.nunique(dropna=True) == 1:
            const_cols.append(c)
        lo, hi = rng
        tol = max(1.0, (hi - lo) * 0.05)
        mn, mx = float(np.nanmin(arr.values)), float(np.nanmax(arr.values))
        if mn < lo - tol or mx > hi + tol:
            oor_cols.append(f"{c}[{mn:.0f},{mx:.0f}]vs[{lo:.0f},{hi:.0f}]")
    if nan_cols:
        local.append(f"partial-NaN DV cols: {nan_cols[:6]}")
    if const_cols:
        local.append(f"ZERO-variance DV cols: {const_cols[:6]}")
    if oor_cols:
        local.append(f"OUT-OF-RANGE DV cols: {oor_cols[:6]}")
    # 5. open-text quality
    oe_cols = [c for c in cols if df[c].dtype == object and
               any(k in c.lower() for k in ["why", "open", "text", "explain", "comment", "_oe", "response", "elaborat"])]
    for c in oe_cols:
        vals = [str(v).strip() for v in df[c].tolist() if isinstance(v, str) and str(v).strip()]
        if not vals:
            continue
        filled = len(vals) / len(df)
        uniq = len(set(vals)) / len(vals)
        bare = sum(1 for v in vals if v.lower() in BARE)
        meta = sum(1 for v in vals if any(m in v.lower() for m in META_TELLS))
        probs = []
        if uniq < 0.3:
            probs.append(f"low-diversity({uniq:.0%} unique)")
        if bare:
            probs.append(f"{bare} bare-pronoun")
        if meta > len(vals) * 0.1:
            probs.append(f"{meta} survey-meta")
        if probs:
            local.append(f"OE '{c[:24]}': " + ", ".join(probs))
    return local

print(f"Inspecting {len(STUDENT_FILES)} student QSFs (N={N})\n" + "=" * 72)
for nm in STUDENT_FILES:
    path = os.path.join(QSF_DIR, nm)
    print(f"\n### {nm}")
    if not os.path.exists(path):
        print("  (file not found)"); continue
    try:
        with open(path, "rb") as f:
            preview = QSFPreviewParser().parse(f.read())
        inp = app._preview_to_engine_inputs(preview)
    except Exception as e:
        print(f"  PARSE/BRIDGE CRASH: {type(e).__name__}: {e}")
        issues_global.append((nm, "parse crash")); traceback.print_exc(); continue
    print(f"  parsed: conds={len(inp['conditions'])} scales={len(inp['scales'])} "
          f"oe={len(inp.get('open_ended_questions') or [])} "
          f"sliders={len(preview.slider_questions or [])} text_entry={len(preview.text_entry_questions or [])} "
          f"total_q={preview.total_questions}")
    try:
        eng = EnhancedSimulationEngine(
            study_title=preview.survey_name or nm,
            study_description=(preview.study_context or {}).get("description", "") or nm,
            sample_size=N, conditions=inp["conditions"], factors=inp["factors"],
            scales=inp["scales"], additional_vars=[],
            demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
            open_ended_questions=inp.get("open_ended_questions"),
            study_context=inp.get("study_context"), seed=11)
        if getattr(eng, "llm_generator", None) is not None:
            eng.llm_generator.disable_permanently("inspect - no network")
        df, meta = eng.generate()
    except Exception as e:
        print(f"  SIMULATE CRASH: {type(e).__name__}: {e}")
        issues_global.append((nm, f"sim crash: {e}")); traceback.print_exc(); continue
    print(f"  simulated: rows={len(df)} cols={len(df.columns)}")
    local = inspect(nm, df, inp, preview)
    if local:
        for li in local:
            print(f"    ⚠ {li}")
            issues_global.append((nm, li))
    else:
        print("    ✓ no output-data issues detected")

print("\n" + "=" * 72)
print(f"TOTAL flagged issues across {len(STUDENT_FILES)} student QSFs: {len(issues_global)}")
# crashes are hard failures; data-quality flags are warnings
crashes = [i for i in issues_global if "crash" in i[1]]
sys.exit(1 if crashes else 0)
