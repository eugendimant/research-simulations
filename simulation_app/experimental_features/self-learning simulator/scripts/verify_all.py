from __future__ import annotations
import subprocess, sys, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(cmd):
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env={**os.environ, "PYTHONPATH":"."})
    return p.returncode, p.stdout, p.stderr

def main() -> int:
    cmds = [
        [sys.executable, "scripts/verify.py"],
        [sys.executable, "scripts/verify_surveys.py"],
        [sys.executable, "scripts/verify_games_extra.py"],
        [sys.executable, "-m", "pytest", "-q"],
    ]
    for c in cmds:
        rc, out, err = run(c)
        if rc != 0:
            print("FAILED:", c)
            print(out)
            print(err)
            return rc
    print("verify_all_ok")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
