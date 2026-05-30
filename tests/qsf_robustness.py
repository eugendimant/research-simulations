#!/usr/bin/env python3
"""Parse every QSF in example_files/ to verify the parser is robust across all
survey types. Reports crashes, parse failures, and detection anomalies.
Run:  python3 tests/qsf_robustness.py
"""
import os, sys, glob, traceback
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_app"))

from utils.qsf_preview import QSFPreviewParser

HERE = os.path.dirname(os.path.abspath(__file__))
QSF_DIR = os.path.join(HERE, "..", "simulation_app", "example_files")

files = sorted(glob.glob(os.path.join(QSF_DIR, "*.qsf")))
print(f"Found {len(files)} QSF files\n")

crashes, parse_fail, zero_cond, zero_dv, ok = [], [], [], [], []
results = []

for path in files:
    name = os.path.basename(path)
    try:
        with open(path, "rb") as f:
            content = f.read()
    except Exception as e:
        crashes.append((name, f"read error: {e}"))
        continue
    try:
        parser = QSFPreviewParser()
        res = parser.parse(content)
    except Exception as e:
        crashes.append((name, f"{type(e).__name__}: {e}"))
        traceback.print_exc()
        continue

    n_cond = len(res.detected_conditions or [])
    n_dv = len(res.detected_scales or [])
    n_oe = len(res.open_ended_questions or [])
    results.append((name, res.success, n_cond, n_dv, n_oe, res.total_questions))
    if not res.success:
        parse_fail.append((name, (res.validation_errors or ["?"])[0]))
    elif n_dv == 0:
        zero_dv.append(name)
    else:
        ok.append(name)
    if n_cond == 0:
        zero_cond.append(name)

print("=" * 72)
print(f"PARSED OK (success + >=1 DV): {len(ok)}/{len(files)}")
print(f"CRASHES:                      {len(crashes)}")
print(f"PARSE success=False:          {len(parse_fail)}")
print(f"ZERO DV detected:             {len(zero_dv)}")
print(f"ZERO conditions detected:     {len(zero_cond)}")
print("=" * 72)

if crashes:
    print("\n### CRASHES (parser raised) ###")
    for n, e in crashes:
        print(f"  - {n}: {e}")

if parse_fail:
    print("\n### success=False ###")
    for n, e in parse_fail[:30]:
        print(f"  - {n}: {e}")

if zero_dv:
    print(f"\n### ZERO DV detected ({len(zero_dv)}) — may be legitimate (e.g. pure-behavioral) or a miss ###")
    for n in zero_dv[:40]:
        print(f"  - {n}")

# Summary stats
if results:
    dvs = [r[3] for r in results]
    conds = [r[2] for r in results]
    print(f"\nDV detection: min={min(dvs)} max={max(dvs)} mean={sum(dvs)/len(dvs):.1f}")
    print(f"Conditions:   min={min(conds)} max={max(conds)} mean={sum(conds)/len(conds):.1f}")

# Exit non-zero only on hard crashes (robustness failures)
sys.exit(1 if crashes else 0)
