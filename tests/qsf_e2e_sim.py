#!/usr/bin/env python3
"""End-to-end simulation on REAL QSF files using the actual app bridge
(_preview_to_engine_inputs) → EnhancedSimulationEngine. Catches runtime bugs and
quantifies which DVs/OE actually get simulated for diverse survey types.

Run:  python3 tests/qsf_e2e_sim.py [N]   (N = sample size per sim, default 40)
"""
import os, sys, glob, traceback
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from utils.qsf_preview import QSFPreviewParser
from utils.enhanced_simulation_engine import EnhancedSimulationEngine
import app  # for _preview_to_engine_inputs (headless OK)

HERE = os.path.dirname(os.path.abspath(__file__))
QSF_DIR = os.path.join(HERE, "..", "simulation_app", "example_files")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40

# Diverse sample across experiment types
SAMPLE = [
    # behavioral economics
    "DG & PGG (Information Nudge).qsf", "Dice_Game_T_Control.qsf",
    "Players_Liars_07_Mar05.qsf", "KW Elicitation (Penn).qsf",
    "Dimant (2023) norm elicitation builder.qsf", "MTurk_Norms_Questionnaire.qsf",
    "Default Nudge (V1, with explicit announcements).qsf",
    # consumer / marketing
    "Coffee_Shop_Loyalty_Programs.qsf",
    # political identity
    "Hate_Trumps_Love.qsf", "Experiment (MGP-Trump second).qsf",
    # student group projects (varied)
    "BDS_5010_Project.qsf", "2026_04_16_BDS5010_group2.qsf",
    "Final - Group 7.qsf", "Group #11 - Survey.qsf",
    "BDS5010 Group 5 Qualtrics Survey File.qsf",
    "2026_04_18_Group6_Charity_Review_Experiment.qsf",
    "Requiem_for_a_nudge_experiment_file.qsf",
    "2026_04_15_5010_with_WTP_revised_44.qsf",
]

fails = []
def check(name, cond, detail=""):
    if not cond:
        fails.append(f"{name}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if (detail and not cond) else ''}")

files = []
for nm in SAMPLE:
    p = os.path.join(QSF_DIR, nm)
    if os.path.exists(p):
        files.append(p)
print(f"Running e2e simulation on {len(files)} real QSFs (N={N} each)\n")

for path in files:
    name = os.path.basename(path)
    print(f"--- {name} ---")
    try:
        with open(path, "rb") as f:
            preview = QSFPreviewParser().parse(f.read())
    except Exception as e:
        check(f"{name}: parse", False, f"{type(e).__name__}: {e}"); continue

    try:
        inp = app._preview_to_engine_inputs(preview)
    except Exception as e:
        check(f"{name}: bridge", False, f"{type(e).__name__}: {e}")
        traceback.print_exc(); continue

    n_scales = len(inp["scales"]); n_oe = len(inp.get("open_ended_questions") or [])
    n_cond = len(inp["conditions"])
    fabricated = (n_scales == 1 and inp["scales"][0].get("detected_from_qsf") is False)
    print(f"    conds={n_cond} scales={n_scales} oe={n_oe} "
          f"sliders={len(preview.slider_questions or [])} "
          f"text_entry={len(preview.text_entry_questions or [])}"
          f"{'  [DV FABRICATED - real DVs not simulated]' if fabricated else ''}")

    try:
        eng = EnhancedSimulationEngine(
            study_title=preview.survey_name or name,
            study_description=(preview.study_context or {}).get("description", "") or name,
            sample_size=N, conditions=inp["conditions"], factors=inp["factors"],
            scales=inp["scales"], additional_vars=[],
            demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
            open_ended_questions=inp.get("open_ended_questions"),
            study_context=inp.get("study_context"), seed=7)
        if getattr(eng, "llm_generator", None) is not None:
            eng.llm_generator.disable_permanently("e2e test - no network")
        df, meta = eng.generate()
    except Exception as e:
        check(f"{name}: simulate", False, f"{type(e).__name__}: {e}")
        traceback.print_exc(); continue

    check(f"{name}: simulate", True)
    check(f"{name}: rows==N", len(df) == N, f"got {len(df)}")
    # NaN check in numeric DV columns (scale columns)
    nan_cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            arr = pd.to_numeric(df[c], errors="coerce").to_numpy()
            if np.isinf(arr).any():
                nan_cols.append(c + "(inf)")
    check(f"{name}: no inf in numeric", not nan_cols, ", ".join(nan_cols[:5]))

print("\n" + "=" * 70)
if fails:
    print(f"E2E SIM: {len(fails)} FAILURES")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("E2E SIM: ALL CHECKS PASSED")
sys.exit(0)
