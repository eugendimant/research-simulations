from __future__ import annotations
from pathlib import Path
import json

def write_benchmark_report(run_dir: Path, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Benchmark report", ""]
    for p in sorted(run_dir.glob("*.pred_moments.json")):
        it = json.loads(p.read_text(encoding="utf-8"))
        bid = it.get("benchmark_id")
        lines.append(f"## {bid}")
        errs = it.get("errors", {})
        if not errs:
            lines.append("")
            lines.append("No comparable moments found (check target extraction or spec mapping).")
            lines.append("")
            continue
        lines.append("")
        for k, e in errs.items():
            lines.append(f"- {k}: error={float(e):+.4f}")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
