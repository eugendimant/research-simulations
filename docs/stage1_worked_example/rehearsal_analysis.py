"""Worked example: rehearse a pre-registered analysis on a synthetic export.

Input: the Qualtrics-style file that `make reproduce` delivers (seed 7, 100 rows, offline).
Planned analysis (written before any data exist): one-way ANOVA of the mean loyalty rating across the four
conditions, after excluding participants who fail the attention check.

Run:  make reproduce && python docs/stage1_worked_example/rehearsal_analysis.py [--typo | --wrong-code]
Two errors that a rehearsal catches before fielding:
  --typo        a misspelled column name (fails with KeyError)
  --wrong-code  the exclusion rule keeps answer 1 as "passed" when the engine records the pass as 2 (Attention_Pass_Rate is 1 for answer 2 in the diagnostics file).
                (This was my first draft of the rule. The synthetic export showed it keeping 8 of 100 rows.)
The numbers printed are properties of SYNTHETIC data. They are not findings.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

PATH = os.environ.get("REHEARSAL_CSV", os.path.join("reproducibility_output", "reference_delivered_export.csv"))
df = pd.read_csv(PATH)

item_cols = [c for c in df.columns if c.startswith("LoyaltyQuestions_")]
outcome = "LoyaltyQuestions_mean" if "--typo" not in sys.argv else "Loyalty_mean"   # the typo run fails here
df["LoyaltyQuestions_mean"] = df[item_cols].mean(axis=1)

PASS_CODE = 1 if "--wrong-code" in sys.argv else 2
kept = df[df["Attention_Check_1"] == PASS_CODE].copy()   # planned exclusion rule
print(f"rows delivered: {len(df)}   rows kept by the planned exclusion rule: {len(kept)}")
if len(kept) < 0.5 * len(df):
    sys.exit(f"STOP: the exclusion rule removes {len(df) - len(kept)} of {len(df)} rows. Check the answer coding.")
groups = [g[outcome].to_numpy() for _, g in kept.groupby("CONDITION")]
print({k: int(len(v)) for k, v in kept.groupby("CONDITION")[outcome]})
F, p = stats.f_oneway(*groups)
print(f"one-way ANOVA on {outcome}: F = {F:.2f}, p = {p:.3f}   (synthetic data, not evidence)")
