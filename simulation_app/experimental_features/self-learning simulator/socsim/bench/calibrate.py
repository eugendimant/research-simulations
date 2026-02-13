from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
from datetime import datetime, timezone

from .run import run_benchmarks

def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Fit y ~ b0 + X b with ridge penalty on b (not intercept)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n, k = X.shape
    Xc = np.concatenate([np.ones((n,1)), X], axis=1)
    # penalty only for coefficients (not intercept)
    I = np.eye(k+1)
    I[0,0] = 0.0
    beta = np.linalg.solve(Xc.T @ Xc + alpha * I, Xc.T @ y)
    b0 = float(beta[0,0])
    b = beta[1:,0]
    return b, b0

def calibrate_global_means_via_ridge(registry_path: Path, targets_dir: Path, specs_dir: Path, out_dir: Path,
                                    priors_path: Path, latent_classes_path: Path, evidence_store_path: Path,
                                    n: int = 4000, seed: int = 0, alpha: float = 1.0) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pri = json.loads(priors_path.read_text(encoding="utf-8"))
    base_means = dict(pri.get("global_means", {}))
    if not base_means:
        raise ValueError("priors.json missing global_means")

    results = run_benchmarks(registry_path, targets_dir, specs_dir, out_dir/"runs",
                             priors_path, latent_classes_path, evidence_store_path,
                             n=n, seed=seed)

    # engineered features: mean_abs_error, n_moments
    X = []
    ys = []
    for r in results:
        errs = list(r.errors.values())
        mean_abs = float(np.mean(np.abs(errs))) if errs else 0.0
        nm = float(len(errs))
        X.append([mean_abs, nm])
        # pseudo target: direction based on first behavior-related moment
        s = 0.0
        for k, e in r.errors.items():
            if ("give" in k) or ("invest" in k) or ("contrib" in k):
                s = -float(e)  # if pred below obs -> increase
                break
        ys.append(s)

    X = np.asarray(X, dtype=float)
    y = np.asarray(ys, dtype=float)

    updated = json.loads(json.dumps(pri))
    for p in ["prosociality", "conditional_coop", "reciprocity"]:
        if p in base_means:
            b, b0 = _ridge_fit(X, y, float(alpha))
            xbar = np.asarray([float(np.mean(X[:,0])), float(np.mean(X[:,1]))], dtype=float)
            adj = float(b0 + float(np.dot(xbar, b)))
            adj = float(np.clip(adj, -0.25, 0.25))
            updated["global_means"][p] = float(base_means[p]) + adj

    updated.setdefault("_bench_calibration", {})
    updated["_bench_calibration"].update({
        "method":"ridge_pseudo_target",
        "alpha": float(alpha),
        "n_benchmarks": int(len(results)),
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })

    (out_dir/"priors_calibrated.json").write_text(json.dumps(updated, indent=2), encoding="utf-8")
    return updated


def optimize_holdout_alpha(registry_path: Path, targets_dir: Path, specs_dir: Path, out_dir: Path,
                           priors_path: Path, latent_classes_path: Path, evidence_store_path: Path,
                           n: int = 4000, seed: int = 0,
                           holdout_frac: float = 0.2,
                           alpha_grid: list[float] | None = None) -> Dict[str, Any]:
    """Fit ridge calibration on train benchmarks and choose alpha that maximizes held-out score.

    Score is negative RMSE across available moment keys.
    """
    if alpha_grid is None:
        alpha_grid = [0.1, 0.3, 1.0, 3.0, 10.0]
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    reg = BenchmarkRegistry.load(registry_path)
    ids = [b.id for b in reg.benchmarks]
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    k = max(1, int(round(len(ids) * holdout_frac)))
    test_ids = set(ids[:k])
    train_ids = [i for i in ids if i not in test_ids]

    def run_subset(sub_ids: list[str], out_sub: Path) -> Dict[str, Any]:
        sub_reg_path = out_sub / "registry_subset.json"
        sub = {"version": reg.version, "benchmarks": [b.__dict__ for b in reg.benchmarks if b.id in sub_ids]}
        sub_reg_path.write_text(json.dumps(sub, indent=2), encoding="utf-8")
        return run_benchmarks(sub_reg_path, targets_dir, specs_dir, out_sub / "runs",
                              priors_path, latent_classes_path, evidence_store_path, n=n, seed=seed)

    train_results = run_subset(train_ids, out_dir / "train")

    # Fit for each alpha and evaluate on test
    best = None
    all_scores = []
    for alpha in alpha_grid:
        calib = calibrate_global_means(registry_path=out_dir/"train"/"registry_subset.json",
                                       targets_dir=targets_dir, specs_dir=specs_dir,
                                       out_dir=out_dir/"tmp",
                                       priors_path=priors_path, latent_classes_path=latent_classes_path,
                                       evidence_store_path=evidence_store_path, n=n, seed=seed, alpha=float(alpha))
        # Evaluate
        test_results = run_subset(list(test_ids), out_dir / f"test_alpha_{alpha}")
        # compute RMSE over moment keys that exist
        # test_results structure: per benchmark errors in out_dir/runs; we can compute from returned dict
        errs = []
        for b in test_results.get("benchmarks", []):
            for _, v in (b.get("errors") or {}).items():
                try:
                    errs.append(float(v))
                except Exception:
                    pass
        rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
        score = -rmse
        all_scores.append({"alpha": float(alpha), "rmse": rmse, "score": score})
        if best is None or (score > best["score"]):
            best = {"alpha": float(alpha), "rmse": rmse, "score": score}

    payload = {"best": best, "scores": all_scores, "holdout_frac": holdout_frac, "train_ids": train_ids, "test_ids": sorted(list(test_ids))}
    (out_dir / "holdout_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
