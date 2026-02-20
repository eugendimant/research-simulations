# Simulation Run Self-Improvement Workflow Plan (v1.0.7.3)

## Coding-side plan
1. Add a run-archive module that writes every completed simulation into a unique per-run folder.
2. Store required artifacts only: `Simulated_Data.csv`, `Instructor_Copy.md`, metadata, validation output, and engine log.
3. Add a run-folder naming convention with timestamp + run id to avoid collisions.
4. Add an audit scanner that checks only *new* run folders the software has not seen before.
5. Persist scanner state (`seen_runs`) in a state JSON file.
6. Add quality rules for empty datasets, blank OE cells, low OE uniqueness, generic OE text, and LLM activity mismatches.
7. Save per-run findings to `Quality_Audit.json` in each run folder.
8. Append all detected issues with timestamps to a global `continuous_improvement_log.txt`.
9. Surface archive + audit paths in the UI metadata so users/instructors can inspect outputs quickly.
10. Add tests for persistence and scanner behavior to prevent regressions.

## Implementation iterations completed
### Iteration 1
- Implemented `persist_simulation_run()` and `audit_run_directory()` primitives.

### Iteration 2
- Added `audit_new_runs()` with seen-run tracking and append-only improvement log.

### Iteration 3
- Integrated archive + audit workflow in Streamlit generation path.

### Iteration 4
- Added metadata/UI surfacing for archive path and continuous improvement log path.

### Iteration 5
- Hardened workflow with `Run_Manifest.json` checksums, short-response detector (`oe_too_short`), hidden-folder scanner exclusion, aggregate `audit_summary.json`, manifest-integrity audit checks, and expanded regression tests.
