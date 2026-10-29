"""Retrospective benchmark runner.

    python -m benchmarks.run_benchmark brader_valentino_suhay_2008 --seeds 40 --out benchmarks/results

The engine receives the design only (a QSF built from study.json). Human outcomes are read after generation and
used only to score. The same scoring is applied to two baselines that use no behavioral knowledge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from benchmarks import metrics as M  # noqa: E402
from benchmarks.baselines import naive_dataset  # noqa: E402
from benchmarks.fetch_human_data import fetch  # noqa: E402

ARMS = ["ENGINE_DEFAULT", "ENGINE_DECLARED", "ENGINE_NULL", "ENGINE_PRIOR", "BASELINE_NULL", "BASELINE_PRIOR"]
ARM_TEXT = {
    "ENGINE_DEFAULT": "Engine, nothing configured (declare-first default: no direction declared, so a true null)",
    "ENGINE_DECLARED": "Engine, direction declared (higher = more opposition to immigration), effects inferred from condition wording",
    "ENGINE_NULL": "Engine, inference switched off (true null)",
    "ENGINE_PRIOR": "Engine, analyst-stated effect d = {d} on the focal cell",
    "BASELINE_NULL": "Naive latent-normal simulation, no effect",
    "BASELINE_PRIOR": "Naive latent-normal simulation, analyst-stated effect d = {d}",
}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(HERE), text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def load_human(spec: Dict, path: str):
    df = pd.read_csv(path)
    table = {d["name"]: df[d["name"]].to_numpy(float) for d in spec["dvs"]}
    tone, eth = df["tone"].to_numpy(int), df["eth"].to_numpy(int)
    return table, tone, eth


def engine_table(df: pd.DataFrame, dvs: List[M.DV]):
    cond = df["CONDITION"].astype(str)
    tone = cond.str.startswith("Negative").to_numpy(int)
    eth = cond.str.contains("Latino").to_numpy(int)
    table = {}
    for dv in dvs:
        col = f"{dv.name}_1" if f"{dv.name}_1" in df.columns else dv.name
        table[dv.name] = df[col].to_numpy(float)
    return table, tone, eth


def baseline_table(d: Dict[str, np.ndarray]):
    cell = d["cell"]
    tone = np.array([c.startswith("Negative") for c in cell], dtype=int)
    eth = np.array(["Latino" in c for c in cell], dtype=int)
    return {k: v for k, v in d.items() if k != "cell"}, tone, eth


def pipeline_rehearsal(df: pd.DataFrame, dvs: List[M.DV], human_table) -> Dict:
    """Could the analysis script run on this table without edits beyond a declared column map?"""
    checks = {}
    for dv in dvs:
        col = f"{dv.name}_1"
        present = col in df.columns
        y = df[col].to_numpy(float) if present else np.array([])
        checks[dv.name] = {
            "column_present": present,
            "all_finite": bool(np.isfinite(y).all()) if present else False,
            "within_human_range": bool(y.min() >= human_table[dv.name].min() and y.max() <= human_table[dv.name].max()) if present and len(y) else False,
            "integer_valued": bool(np.allclose(y, np.round(y))) if present and len(y) else False,
            "levels_observed": sorted({float(v) for v in y}) if present else [],
        }
    prov = ["RUN_ID", "SIMULATION_MODE", "SIMULATION_SEED", "_Generation_Source"]
    checks["_provenance_columns_present"] = all(c in df.columns for c in prov)
    checks["_condition_labels"] = sorted(df["CONDITION"].astype(str).unique().tolist())
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--human-dir", default=os.path.join(HERE, "_human_data"))
    a = ap.parse_args()

    spec = json.load(open(os.path.join(HERE, "studies", a.study, "study.json")))
    dvs = [M.DV(d["name"], d["lo"], d["hi"], [float(x) for x in range(d["lo"], d["hi"] + 1)]) for d in spec["dvs"]]
    human_path = fetch(a.study, a.human_dir)
    h_table, h_tone, h_eth = load_human(spec, human_path)
    h_focal = ((h_tone == 1) & (h_eth == 1)).astype(int)
    human = M.summarize(h_table, dvs, h_focal, h_tone, h_eth)

    from benchmarks import engine_adapter as EA
    inputs = EA.build_inputs(spec)
    d = spec["prior_d"]
    t0 = time.time()
    arm_runs: Dict[str, List[Dict]] = {k: [] for k in ARMS}
    rehearsal = None
    eff_inferred = None
    domains = None
    warnings_seen: List[str] = []
    for i in range(a.seeds):
        seed = a.seed0 + i
        for arm in ARMS:
            if arm.startswith("ENGINE"):
                auto = arm in ("ENGINE_DEFAULT", "ENGINE_DECLARED")
                pd_ = d if arm == "ENGINE_PRIOR" else None
                pol = spec.get("declared_polarity") if arm == "ENGINE_DECLARED" else None
                df, meta = EA.run_engine(spec, inputs, seed, auto_effects=auto, prior_d=pd_, dv_polarity=pol)
                table, tone, eth = engine_table(df, dvs)
                if arm == "ENGINE_DEFAULT" and i == 0:
                    rehearsal = pipeline_rehearsal(df, dvs, h_table)
                    domains = meta.get("detected_domains")
                    eff_inferred = meta.get("effect_sizes_applied")
                    warnings_seen = list(meta.get("generation_warnings") or [])
            else:
                dd = d if arm == "BASELINE_PRIOR" else 0.0
                table, tone, eth = baseline_table(naive_dataset(spec["n_per_cell"], dvs, spec["focal_cell"], dd, seed))
            focal = ((tone == 1) & (eth == 1)).astype(int)
            s = M.summarize(table, dvs, focal, tone, eth)
            arm_runs[arm].append({"summary": s, "cmp": M.compare_to_human(human, s, dvs, h_table, table)})
        if i % 5 == 0:
            print(f"seed {i + 1}/{a.seeds}  {time.time() - t0:.0f}s", flush=True)

    result = {
        "study": a.study, "citation": spec["citation"], "seeds": a.seeds, "seed0": a.seed0, "prior_d": d,
        "git_sha": git_sha(), "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "human_data_sha256": hashlib.sha256(open(human_path, "rb").read()).hexdigest(),
        "human": human, "adapter_notes": inputs["notes"], "engine_detected_domains": domains,
        "engine_generation_warnings": warnings_seen, "pipeline_rehearsal": rehearsal,
        "arms": {k: M.aggregate([r["cmp"] for r in v], dvs) for k, v in arm_runs.items()},
        "null_calibration": {
            dv.name: {
                "engine_null_focal_reject_rate": float(np.mean([r["summary"]["dv"][dv.name]["focal_p"] < 0.05 for r in arm_runs["ENGINE_NULL"]])),
                "baseline_null_focal_reject_rate": float(np.mean([r["summary"]["dv"][dv.name]["focal_p"] < 0.05 for r in arm_runs["BASELINE_NULL"]])),
            } for dv in dvs},
        "synthetic_focal_g_over_seeds": {
            arm: {dv.name: {"mean": float(np.mean([r["summary"]["dv"][dv.name]["focal_g"] for r in v])),
                            "sd": float(np.std([r["summary"]["dv"][dv.name]["focal_g"] for r in v], ddof=1))} for dv in dvs}
            for arm, v in arm_runs.items()},
        "synthetic_means_over_seeds": {
            arm: {dv.name: float(np.mean([r["summary"]["dv"][dv.name]["mean"] for r in v])) for dv in dvs} for arm, v in arm_runs.items()},
        "engine_default_effect_record": eff_inferred,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"{a.study}.json")
    json.dump(result, open(out, "w"), indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
