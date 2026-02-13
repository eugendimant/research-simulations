from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np

def build_scoreboard(run_dir: Path, out_json: Path, out_md: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(run_dir.glob("*.pred_moments.json")):
        it = json.loads(p.read_text(encoding="utf-8"))
        bid = it.get("benchmark_id")
        errs = it.get("errors", {})
        rmse = None
        if errs:
            xs = np.asarray(list(errs.values()), dtype=float)
            rmse = float(np.sqrt(np.mean(xs * xs)))
        rows.append({"benchmark_id": bid, "rmse": rmse, "n_moments": len(errs)})
    payload = {"rows": rows}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(["# Scoreboard", ""] + [f"- {r['benchmark_id']}: rmse={r['rmse']}, n={r['n_moments']}" for r in rows]), encoding="utf-8")
    return payload
