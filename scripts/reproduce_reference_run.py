#!/usr/bin/env python3
"""One-command reproducibility demonstration (make reproduce).

Runs the engine offline on a reference QSF, then checks these things and writes a report:

  1. the run finishes and every provenance column is present in the engine output (RUN_ID, SIMULATION_MODE,
     SIMULATION_SEED, _Generation_Source), and the run metadata carries one calibration-provenance row per outcome;
  2. the user's download (utils/user_package.py) names the data file Simulated_Data.csv, states in its README that the
     responses are simulated, and contains none of the engine's internals;
  3. the same seed gives byte-identical output in a second process started with a different PYTHONHASHSEED;
  4. a different seed gives different output.

Exit status 0 only if all checks pass. No network, no API key, no Streamlit server.

    python3 scripts/reproduce_reference_run.py --qsf simulation_app/example_files/Coffee_Shop_Loyalty_Programs.qsf --seed 7 --outdir reproducibility_output
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "simulation_app"))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def generate(qsf_path: str, seed: int, n: int):
    """Return (engine_df, metadata, delivered_export_df) using only public entry points."""
    from utils.qsf_preview import QSFPreviewParser
    import app  # noqa: F401  (needs streamlit installed; no server is started)
    from utils.enhanced_simulation_engine import EnhancedSimulationEngine
    from utils.qualtrics_export import build_qualtrics_export

    preview = QSFPreviewParser().parse(open(qsf_path, "rb").read())
    inp = app._preview_to_engine_inputs(preview)
    eng = EnhancedSimulationEngine(
        study_title=os.path.basename(qsf_path), study_description="reference run", sample_size=n,
        conditions=inp["conditions"], factors=inp["factors"], scales=inp["scales"], additional_vars=[],
        demographics={"gender_quota": 50, "age_mean": 35, "age_sd": 12},
        open_ended_questions=inp.get("open_ended_questions"), seed=seed)
    eng.llm_generator.disable_permanently("offline")
    df, meta = eng.generate()
    export_df, _diag = build_qualtrics_export(df, meta)
    return df, meta, export_df


def child() -> None:
    """Child mode: print the output hashes as JSON and nothing else on stdout."""
    a = _args()
    df, _meta, export_df = generate(a.qsf, a.seed, a.n)
    print("HASHES " + json.dumps({"engine_csv": sha256_bytes(df.to_csv(index=False).encode()),
                                   "delivered_csv": sha256_bytes(export_df.to_csv(index=False).encode())}))


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qsf", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--outdir", default="reproducibility_output")
    ap.add_argument("--child", action="store_true")
    return ap.parse_args()


def run_child(qsf: str, seed: int, n: int, hashseed: str) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", "--qsf", qsf, "--seed", str(seed), "--n", str(n)],
                         capture_output=True, text=True, env=env, check=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("HASHES ")][-1]
    return json.loads(line[len("HASHES "):])


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    a = _args()
    if a.child:
        child()
        return 0
    os.makedirs(a.outdir, exist_ok=True)
    from utils import __version__ as version
    from utils.user_package import build_user_package, find_simulation_internals

    df, meta, export_df = generate(a.qsf, a.seed, a.n)
    engine_csv = df.to_csv(index=False).encode()
    delivered_csv = export_df.to_csv(index=False).encode()
    open(os.path.join(a.outdir, "reference_synthetic_data.csv"), "wb").write(engine_csv)
    open(os.path.join(a.outdir, "reference_delivered_export.csv"), "wb").write(delivered_csv)

    checks = {}
    prov = ["RUN_ID", "SIMULATION_MODE", "SIMULATION_SEED", "_Generation_Source"]
    checks["provenance_columns_in_engine_output"] = all(c in df.columns for c in prov)
    prov_rows = (meta.get("calibration_provenance") or {}).get("variables") or []
    checks["calibration_provenance_recorded"] = bool(prov_rows) and all("effect_source" in r for r in prov_rows)
    user = build_user_package({"Simulated_Data.csv": delivered_csv}, meta)
    readme = user.get("README_Download_Contents.txt", b"").decode("utf-8")
    checks["user_download_labelled_simulated"] = "Simulated_Data.csv" in user and "simulated, not collected from people" in readme
    checks["user_download_has_no_engine_internals"] = not any(
        find_simulation_internals(v.decode("utf-8", "replace")) for k, v in user.items() if not k.endswith(".csv"))
    checks["seed_recorded_matches_request"] = bool((df["SIMULATION_SEED"] == a.seed).all())
    same = run_child(a.qsf, a.seed, a.n, hashseed="12345")
    checks["same_seed_new_process_identical_engine_csv"] = same["engine_csv"] == sha256_bytes(engine_csv)
    checks["same_seed_new_process_identical_delivered_csv"] = same["delivered_csv"] == sha256_bytes(delivered_csv)
    other = run_child(a.qsf, a.seed + 1, a.n, hashseed="0")
    checks["different_seed_changes_output"] = other["engine_csv"] != sha256_bytes(engine_csv)
    ok = all(checks.values())

    report = {
        "status": "PASS" if ok else "FAIL", "checks": checks,
        "simulator_version": version, "commit_sha": git_sha(),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "input_qsf": a.qsf, "input_sha256": sha256_bytes(open(a.qsf, "rb").read()),
        "seed": a.seed, "n": a.n, "generation_mode": "offline (open-ended text from the built-in template engine)",
        "rows": int(df.shape[0]), "columns": int(df.shape[1]),
        "engine_csv_sha256": sha256_bytes(engine_csv), "delivered_csv_sha256": sha256_bytes(delivered_csv),
        "run_id": meta.get("run_id"), "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    json.dump(report, open(os.path.join(a.outdir, "reference_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
