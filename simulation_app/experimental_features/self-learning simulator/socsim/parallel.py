from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import math
import multiprocessing as mp

from .simulator import simulate, SimulationResult
from .specs import ExperimentSpec

def _simulate_worker(args):
    return simulate(*args)


def simulate_parallel(
    spec: ExperimentSpec,
    n: int,
    seed: int,
    priors_path: Path,
    latent_classes_path: Path,
    evidence_store_path: Path,
    schema_path: Path | None,
    corpus_path: Path | None = None,
    processes: int = 4,
    return_traces: bool = False,
    safe_mode: bool = False,
    min_coverage: float = 0.0,
) -> SimulationResult:
    n = int(n)
    processes = max(1, int(processes))
    chunks = [n // processes] * processes
    for i in range(n % processes):
        chunks[i] += 1
    args = []
    base_seed = int(seed)
    for i, cn in enumerate(chunks):
        if cn <= 0:
            continue
        args.append((spec, cn, base_seed + i, priors_path, latent_classes_path, evidence_store_path, schema_path, return_traces, corpus_path, safe_mode, min_coverage))

    with mp.get_context("spawn").Pool(processes=processes) as pool:
        results = pool.map(_simulate_worker, args)

    rows = []
    summary = {"parallel": {"processes": processes, "chunks": chunks}}
    for r in results:
        rows.extend(r.rows)
        # last summary wins but keep conflict/coverage from first
        summary.update(r.summary)

    return SimulationResult(rows=rows, summary=summary)
