from pathlib import Path
import json
from socsim.bench.pack_runner import run_benchmark_pack

def test_pack_runner_fixture(tmp_path: Path):
    bench_dir = Path("benchmarks/_fixture_ultimatum")
    out_dir = tmp_path / "out"
    cache = tmp_path / "cache"
    res = run_benchmark_pack(bench_dir=bench_dir, out_dir=out_dir, cache_dir=cache, timeout_s=10)
    targets = json.loads((out_dir / "targets.json").read_text(encoding="utf-8"))
    assert abs(targets["mean_offer"] - 4.5) < 1e-9
