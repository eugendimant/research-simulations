from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np

from .registry import BenchmarkRegistry
from .run import run_benchmarks
from .calibrate import calibrate_priors_means

def leave_one_out_cv(
    registry_path: Path,
    targets_dir: Path,
    specs_dir: Path,
    out_dir: Path,
    priors_path: Path,
    latent_classes_path: Path,
    evidence_store_path: Path,
    n: int = 2000,
    seed: int = 0,
    corpus_path: Optional[Path] = None,
) -> Dict[str, Any]:
    reg = BenchmarkRegistry.load(registry_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for b in reg.benchmarks:
        held = b.id
        train = {"version": reg.version, "benchmarks": []}
        for bb in reg.benchmarks:
            if bb.id != held:
                train["benchmarks"].append({
                    "id": bb.id, "game": bb.game, "source": bb.source,
                    "context_features": bb.context_features, "notes": bb.notes,
                })
        train_path = out_dir / f"_train__{held}.json"
        train_path.write_text(json.dumps(train, indent=2), encoding="utf-8")

        fold_dir = out_dir / f"fold__{held}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        calibrate_priors_means(
            registry_path=train_path,
            targets_dir=targets_dir,
            specs_dir=specs_dir,
            out_dir=fold_dir / "calibration",
            priors_path=priors_path,
            latent_classes_path=latent_classes_path,
            evidence_store_path=evidence_store_path,
            n=n,
            seed=seed,
            corpus_path=corpus_path,
        )
        pri_cal = fold_dir / "calibration" / "priors_calibrated.json"

        test_reg = {"version": reg.version, "benchmarks": [{
            "id": b.id, "game": b.game, "source": b.source,
            "context_features": b.context_features, "notes": b.notes,
        }]}
        test_path = fold_dir / "_test.json"
        test_path.write_text(json.dumps(test_reg, indent=2), encoding="utf-8")

        runs = run_benchmarks(
            registry_path=test_path,
            targets_dir=targets_dir,
            specs_dir=specs_dir,
            out_dir=fold_dir / "runs",
            priors_path=pri_cal,
            latent_classes_path=latent_classes_path,
            evidence_store_path=evidence_store_path,
            n=n,
            seed=seed,
            corpus_path=corpus_path,
        )
        errs = runs[0].errors if runs else {}
        rmse = float(np.sqrt(np.mean(np.asarray(list(errs.values()), dtype=float)**2))) if errs else float("nan")
        folds.append({"held_out": held, "rmse": rmse, "n_moments": len(errs)})
    summary = {"folds": folds, "mean_rmse": float(np.nanmean([f["rmse"] for f in folds])) if folds else float("nan")}
    (out_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
