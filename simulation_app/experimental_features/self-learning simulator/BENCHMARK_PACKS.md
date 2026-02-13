# Benchmark packs (v0.15)

A benchmark pack is a directory with:
- `manifest.json` (dataset URL + optional SHA256)
- `adapter.json` (format + column mapping)
- `moments.json` (explicit target moments)

Run:
```bash
python -m socsim bench_run_pack --bench-dir benchmarks/<id> --out-dir bench_targets/<id>
```
This writes `targets.json` + `provenance.json`.

Moment kinds supported:
- mean
- quantile
- diff_in_means (two-arm)
- trajectory_mean (mean by group key)
