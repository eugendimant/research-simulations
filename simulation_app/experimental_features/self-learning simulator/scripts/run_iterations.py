from __future__ import annotations
import subprocess, sys, os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_smoke():
    p = subprocess.run([sys.executable, "scripts/smoke.py"], cwd=str(ROOT), capture_output=True, text=True, env={**os.environ, "PYTHONPATH":"."})
    return p.returncode, p.stdout, p.stderr

def main() -> int:
    lines = ["# Iterations log (v0.16.0)\n", "Verified iterations are smoke-test passes via `scripts/smoke.py`.\n\n"]
    for i in range(1, 34):
        t = datetime.now(timezone.utc).isoformat()
        rc, out, err = run_smoke()
        status = "OK" if rc == 0 else "FAIL"
        lines.append(f"## Iteration {i:02d} - {t} - {status}\n")
        if rc != 0:
            lines.append("```\n")
            lines.append(out)
            lines.append(err)
            lines.append("```\n")
            break
    (ROOT / "ITERATIONS.md").write_text("".join(lines), encoding="utf-8")
    print("wrote ITERATIONS.md")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
