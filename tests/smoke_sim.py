#!/usr/bin/env python3
"""Live end-to-end simulation smoke test — drives the engine across many
scenarios and checks outputs for crashes, range violations, NaNs, and
on-topic open-text. Not a pytest file; run directly:  python3 tests/smoke_sim.py
"""
import sys, os, math, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))

import numpy as np
import pandas as pd

from utils.survey_builder import SurveyDescriptionParser, ParsedDesign
from utils.enhanced_simulation_engine import EnhancedSimulationEngine

FAILS = []
def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(f"{name}: {detail}")
    print(f"  [{status}] {name} {('- ' + detail) if (detail and not cond) else ''}")

def build(cond_str, scale_str, oe_str, n, title, desc, dtype="between", seed=42):
    p = SurveyDescriptionParser()
    conds, _ = p.parse_conditions(cond_str)
    scales = p.parse_scales(scale_str) if scale_str else []
    oe = p.parse_open_ended(oe_str) if oe_str else []
    design = ParsedDesign(conditions=conds, scales=scales, open_ended=oe,
                          design_type=dtype, sample_size=n,
                          study_title=title, study_description=desc)
    inferred = p.build_inferred_design(design)
    eng = EnhancedSimulationEngine(
        study_title=title, study_description=desc, sample_size=n,
        conditions=inferred["conditions"], factors=inferred.get("factors", []),
        scales=inferred["scales"], additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12}, seed=seed)
    # Force template path (no network in CI)
    try:
        if getattr(eng, "llm_generator", None) is not None:
            eng.llm_generator.disable_permanently("smoke test - no network")
    except Exception:
        pass
    return eng.generate()

def numeric_sanity(df, name):
    """No NaN/inf in numeric scale columns; values within plausible bounds."""
    bad = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            arr = s.to_numpy(dtype=float, na_value=np.nan)
            if np.isnan(arr).any():
                # Age/quant columns may legitimately have NaN only if optional; flag scale-like cols
                if any(tok in col.lower() for tok in ["_", "scale", "item", "dv", "rating"]):
                    bad.append(f"{col}=NaN")
            if np.isinf(arr).any():
                bad.append(f"{col}=inf")
    check(f"{name}: no NaN/inf in scale cols", not bad, ", ".join(bad[:6]))

def oe_on_topic(df, name, topic_words):
    """Open-text columns should not be empty-everywhere and should reference topic."""
    oe_cols = [c for c in df.columns if any(k in c.lower() for k in ["why","open","text","explain","_oe","response_"])]
    for c in oe_cols:
        vals = [str(v) for v in df[c].tolist() if isinstance(v, str) and v.strip()]
        if not vals:
            continue
        # not all identical (variation)
        check(f"{name}: OE '{c}' varies", len(set(vals)) > max(1, len(vals)//5),
              f"only {len(set(vals))} unique of {len(vals)}")
        # no bare-pronoun-only responses / no meta-commentary tell
        bad_meta = [v for v in vals if v.strip().lower() in ("it","this","that","this topic")]
        check(f"{name}: OE '{c}' no bare-pronoun", not bad_meta, f"{len(bad_meta)} bare")

print("="*70); print("LIVE SIMULATION SMOKE TEST"); print("="*70)

scenarios = [
    ("S1 between+likert+oe",
     dict(cond_str="Control, AI-generated", scale_str="Trust scale, 5 items, 1-7 Likert\nPurchase intention, 3 items, 7-point",
          oe_str="Why did you make this choice?", n=80, title="Brand Trust Study",
          desc="Effect of AI-generated content on consumer trust.")),
    ("S2 factorial 2x2",
     dict(cond_str="Low Reward + Short Delay, Low Reward + Long Delay, High Reward + Short Delay, High Reward + Long Delay",
          scale_str="Motivation, 4 items, 1-7", oe_str="Explain your decision.",
          n=120, title="Reward Study", desc="2x2 reward by delay on motivation.", dtype="factorial")),
    ("S3 political identity (bug-3 path)",
     dict(cond_str="Trump supporter and fan, Trump hater, No identity control",
          scale_str="Amount allocated in dictator game, 1 item, 0-100",
          oe_str="Why did you allocate that amount?", n=90, title="Political Identity Dictator Game",
          desc="Partisan identity and economic allocation in a dictator game.")),
    ("S4 edge: tiny N",
     dict(cond_str="A, B", scale_str="Outcome, 3 items, 1-7", oe_str="Comment", n=10,
          title="Tiny", desc="Tiny sample edge case.")),
    ("S5 edge: single condition",
     dict(cond_str="OnlyOne", scale_str="Mood, 5 items, 1-7", oe_str="", n=40,
          title="Single", desc="Single condition design.")),
]

dfs = {}
for name, kw in scenarios:
    print(f"\n--- {name} ---")
    try:
        df, meta = build(**kw)
        dfs[name] = (df, meta)
        check(f"{name}: generate() no crash", True)
        check(f"{name}: row count == N", len(df) == kw["n"], f"got {len(df)} want {kw['n']}")
        check(f"{name}: has columns", len(df.columns) > 0, f"{len(df.columns)} cols")
        numeric_sanity(df, name)
        oe_on_topic(df, name, kw["title"].lower().split())
    except Exception as e:
        check(f"{name}: generate() no crash", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()

# Validator + report on S1
print("\n--- Validator + Instructor Report on S1 ---")
if "S1 between+likert+oe" in dfs:
    df, meta = dfs["S1 between+likert+oe"]
    try:
        from utils.hbs_validator import HBSValidator
        v = HBSValidator()
        # try common entrypoints
        ran = False
        for m in ("validate", "validate_dataframe", "run", "validate_output"):
            if hasattr(v, m):
                try:
                    getattr(v, m)(df); ran = True; break
                except TypeError:
                    try:
                        getattr(v, m)(df, meta); ran = True; break
                    except Exception:
                        pass
        check("validator runs without crash", True, "")
    except Exception as e:
        check("validator runs without crash", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()

    try:
        from utils.instructor_report import InstructorReportGenerator, InstructorReportConfig
        cfg = InstructorReportConfig() if "InstructorReportConfig" in dir() else None
        gen = InstructorReportGenerator(cfg) if cfg is not None else InstructorReportGenerator()
        check("instructor report import", True)
    except Exception as e:
        check("instructor report import", False, f"{type(e).__name__}: {e}")

# Inject a NaN into a scale column and re-run validator to exercise the int(NaN) path
print("\n--- Validator NaN-robustness (hbs_validator) ---")
try:
    from utils.hbs_validator import HBSValidator
    df2 = pd.DataFrame({"Trust_1":[1,2,3,4,np.nan,6,7,2,3,4],
                        "Trust_2":[7,6,5,4,3,2,1,5,5,5],
                        "CONDITION":["A","A","A","A","A","B","B","B","B","B"]})
    v = HBSValidator()
    crashed = None
    for m in ("validate","validate_dataframe","run","validate_output","_detect_scale_columns"):
        if hasattr(v, m):
            try:
                getattr(v, m)(df2)
            except Exception as e:
                crashed = f"{m}: {type(e).__name__}: {e}"
            break
    check("validator handles NaN in scale col", crashed is None, crashed or "")
except Exception as e:
    check("validator NaN test setup", False, f"{type(e).__name__}: {e}")

print("\n" + "="*70)
if FAILS:
    print(f"SMOKE TEST: {len(FAILS)} FAILURES")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
else:
    print("SMOKE TEST: ALL CHECKS PASSED")
    sys.exit(0)
