from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
import json
import pandas as pd

from .registry import BenchmarkRegistry
from ..simulator import load_experiment_spec, simulate

@dataclass
class BenchRunResult:
    benchmark_id: str
    pred: Dict[str, float]
    obs: Dict[str, float]
    errors: Dict[str, float]

def _pred_from_df(df: pd.DataFrame, game: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if game in ("ultimatum","dictator"):
        if "act::give" in df.columns and "ctx::endowment" in df.columns:
            frac = (df["act::give"].astype(float) / df["ctx::endowment"].astype(float)).clip(0,1)
            out["pred::give_frac:mean"] = float(frac.mean())
            out["pred::give_frac:var"] = float(frac.var())
    if game in ("repeated_public_goods","public_goods_punishment"):
        if "act::contrib_r1" in df.columns:
            out["pred::contrib_r1:mean"] = float(df["act::contrib_r1"].astype(float).mean())
        if "act::contrib_rlast" in df.columns:
            out["pred::contrib_rlast:mean"] = float(df["act::contrib_rlast"].astype(float).mean())
    return out

def run_benchmarks(registry_path: Path, targets_dir: Path, specs_dir: Path, out_dir: Path,
                   priors_path: Path, latent_classes_path: Path, evidence_store_path: Path,
                   n: int = 4000, seed: int = 0, corpus_path: Path | None = None) -> List[BenchRunResult]:
    reg = BenchmarkRegistry.load(registry_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[BenchRunResult] = []

    for b in reg.benchmarks:
        spec_path = specs_dir / f"{b.game}.json"
        spec = load_experiment_spec(spec_path)
        spec.context.update(b.context_features)
        res = simulate(
            spec=spec, n=int(n), seed=int(seed),
            priors_path=priors_path, latent_classes_path=latent_classes_path,
            evidence_store_path=evidence_store_path,
            schema_path=Path("socsim/schema/evidence_unit_schema.json"),
            return_traces=False,
            corpus_path=corpus_path,
        )
        df = pd.DataFrame(res.rows)
        pred = _pred_from_df(df, b.game)

        targ = json.loads((targets_dir / f"{b.id}.targets.json").read_text(encoding="utf-8"))
        obs = {m["key"].replace("obs::","obs::"): float(m["value"]) for m in targ.get("moments", [])}
        errors: Dict[str, float] = {}
        for ok, ov in obs.items():
            pk = ok.replace("obs::", "pred::")
            if pk in pred:
                errors[ok] = float(pred[pk] - ov)

        (out_dir / f"{b.id}.pred_moments.json").write_text(json.dumps({"benchmark_id":b.id,"pred":pred,"obs":obs,"errors":errors}, indent=2), encoding="utf-8")
        results.append(BenchRunResult(b.id, pred, obs, errors))

    return results
