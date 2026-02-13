from pathlib import Path
from socsim.bench.registry import BenchmarkRegistry

def test_load():
    r = BenchmarkRegistry.load(Path('socsim/data/benchmark_registry.json'))
    assert len(r.benchmarks) >= 1
