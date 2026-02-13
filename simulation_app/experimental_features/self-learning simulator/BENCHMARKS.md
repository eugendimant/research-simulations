# Benchmark packs

A benchmark pack is a folder containing:
- manifest.json (url, license, citation, optional sha256)
- adapter.json (format, column mapping, optional added constants)
- moments.json (moment definitions to compute `targets.json`)

## Generating targets from packs
Use:
```bash
python -m socsim bench_make_targets --bench_root benchmarks --out_targets bench_targets --cache .cache/socsim
```

## Notes on reproducibility
- If `sha256` is present, downloads are verified.
- If `sha256` is null, the pack is still usable, but the remote file may change. Pin it when possible.

## Included templates
`benchmarks/probstats_*` are templates referencing datasets listed on probstats4econ.com.
They are included for breadth across fields; the actual download happens at runtime.
