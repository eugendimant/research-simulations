#!/usr/bin/env python3
"""Randomly pick >=10 QSF files and simulate each at N=200, validating output.
Run: python3 tests/random_qsf_n200.py [n_files] [seed]
"""
import os, sys, glob, random, traceback
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from utils.qsf_preview import QSFPreviewParser
from utils.enhanced_simulation_engine import EnhancedSimulationEngine
import app

HERE = os.path.dirname(os.path.abspath(__file__))
QSF_DIR = os.path.join(HERE, "..", "simulation_app", "example_files")
N = 200
n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 10
pick_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260531

files = sorted(glob.glob(os.path.join(QSF_DIR, "*.qsf")))
rng = random.Random(pick_seed)
chosen = rng.sample(files, min(n_files, len(files)))

fails = []
def check(name, cond, detail=""):
    if not cond:
        fails.append(f"{name}: {detail}")
    print(f"   [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if (detail and not cond) else ''}")

print(f"Simulating {len(chosen)} randomly-chosen QSFs at N={N} (pick seed={pick_seed})\n" + "=" * 72)
for path in chosen:
    nm = os.path.basename(path)
    print(f"\n### {nm}")
    try:
        preview = QSFPreviewParser().parse(open(path, "rb").read())
        inp = app._preview_to_engine_inputs(preview)
    except Exception as e:
        check(f"{nm}: parse/bridge", False, f"{type(e).__name__}: {e}")
        traceback.print_exc(); continue
    n_scales = len(inp["scales"]); n_oe = len(inp.get("open_ended_questions") or [])
    print(f"   parsed: conds={len(inp['conditions'])} scales={n_scales} oe={n_oe}")
    try:
        eng = EnhancedSimulationEngine(
            study_title=preview.survey_name or nm,
            study_description=(preview.study_context or {}).get("description", "") or nm,
            sample_size=N, conditions=inp["conditions"], factors=inp["factors"],
            scales=inp["scales"], additional_vars=[],
            demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
            open_ended_questions=inp.get("open_ended_questions"),
            study_context=inp.get("study_context"), seed=2024)
        if getattr(eng, "llm_generator", None) is not None:
            eng.llm_generator.disable_permanently("validation - offline")
        df, meta = eng.generate()
    except Exception as e:
        check(f"{nm}: simulate", False, f"{type(e).__name__}: {e}")
        traceback.print_exc(); continue

    check(f"{nm}: simulate N={N}", True)
    check(f"{nm}: row count == {N}", len(df) == N, f"got {len(df)}")
    # unique participant ids
    pid = next((c for c in df.columns if c.upper() in ("PARTICIPANT_ID", "RESPONSE_ID")), None)
    if pid:
        check(f"{nm}: unique participant ids", df[pid].nunique() == len(df), f"{df[pid].nunique()}/{len(df)}")
    # no duplicate column names
    check(f"{nm}: no duplicate columns", len(df.columns) == len(set(df.columns)),
          str([c for c in df.columns if list(df.columns).count(c) > 1][:4]))
    # numeric DV columns: no inf, in declared range
    dv_ranges = {}
    for sc in inp["scales"]:
        vn = str(sc.get("variable_name") or sc.get("name") or "").strip()
        try:
            dv_ranges[vn] = (float(sc.get("scale_min", 1)), float(sc.get("scale_max", 7)))
        except (TypeError, ValueError):
            pass
    inf_cols, oor_cols = [], []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        arr = pd.to_numeric(df[c], errors="coerce")
        if np.isinf(arr.to_numpy(dtype=float, na_value=0.0)).any():
            inf_cols.append(c)
        base = c.rsplit("_", 1)[0] if ("_" in c and c.rsplit("_", 1)[-1].isdigit()) else c
        rng_ = dv_ranges.get(c) or dv_ranges.get(base)
        if rng_:
            lo, hi = rng_; tol = max(1.0, (hi - lo) * 0.02)
            nn = arr.dropna()
            if len(nn) and (nn.min() < lo - tol or nn.max() > hi + tol):
                oor_cols.append(f"{c}[{nn.min():.0f},{nn.max():.0f}]vs[{lo:.0f},{hi:.0f}]")
    check(f"{nm}: no inf in numeric DVs", not inf_cols, ", ".join(inf_cols[:5]))
    check(f"{nm}: DV values within declared range", not oor_cols, ", ".join(oor_cols[:4]))
    # CONDITION present + balanced-ish (if >1 condition)
    ccol = next((c for c in df.columns if c.lower() == "condition"), None)
    if ccol and df[ccol].nunique() > 1:
        vc = df[ccol].value_counts()
        ratio = vc.max() / max(1, vc.min())
        check(f"{nm}: condition balance", ratio <= 3.0, f"{dict(vc)} ratio={ratio:.1f}")
    # at least one real DV (not just a fabricated Main_DV when survey had content)
    fab = n_scales == 1 and inp["scales"][0].get("variable_name") == "Main_DV" \
          and inp["scales"][0].get("detected_from_qsf") is False
    if fab and preview.total_questions > 8:
        print(f"   note: fabricated Main_DV (survey had {preview.total_questions} questions but no detected scale)")

print("\n" + "=" * 72)
if fails:
    print(f"RANDOM N={N} VALIDATION: {len(fails)} FAILURES")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print(f"RANDOM N={N} VALIDATION: ALL CHECKS PASSED ({len(chosen)} files)")
sys.exit(0)
