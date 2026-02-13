from __future__ import annotations
import sys, json
from pathlib import Path

def main() -> int:
    sys.path.insert(0, ".")
    from socsim.bench.pack_runner import run_benchmark_pack
    out = Path("scripts/_pack_out")
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    res = run_benchmark_pack(Path("benchmarks/_fixture_ultimatum"), out, cache_dir=Path("scripts/_cache"), timeout_s=10)
    t = json.loads((out/"targets.json").read_text(encoding="utf-8"))
    assert abs(t["mean_offer"] - 4.5) < 1e-9
    print("verify_pack_ok")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
