# Stage 1 artifact guide (Organization Science special issue)

What to run, what to expect, and where each number in the Stage 1 proposal comes from. Everything below was run on a clean checkout of v1.3.1.2 on 2026-10-29 (Python 3.13.17, numpy 2.5.3, pandas 3.0.5).

## Run it

| Step | Command | Expected | Time |
|---|---|---|---|
| Install pinned versions | `make install` | pip installs `requirements-reproduce.txt` | 1 to 2 min |
| Reproduce a reference run | `make reproduce` | `"status": "PASS"`, eight checks true, files in `reproducibility_output/` | about 11 s |
| Version consistency | `make audit-version` | `OK: all 11 version locations are 1.3.1.2` | 1 s |
| Release smoke gates | `make smoke` | all pass | under 2 min |
| Retrospective benchmark | `make benchmark SEEDS=100` | writes `benchmarks/results/brader_valentino_suhay_2008.json` and prints the report (needs network once) | about 8 min |
| Full fast test suite | `make test-fast` | same selection as the PR job in CI | 30 to 60 min |
| QSF corpus | `python tests/qsf_robustness.py` | 303 files, 0 crashes, 281 with an outcome | under 1 min |

`make reproduce` generates 100 synthetic participants from `simulation_app/example_files/Coffee_Shop_Loyalty_Programs.qsf` with seed 7, offline, and checks that (1) the engine output carries its provenance columns, (2) the run records one calibration-provenance row per outcome, (3) the user's download names the data `Simulated_Data.csv` and states that the responses are simulated, (4) the download carries no engine internals, (5) the recorded seed equals the requested seed, (6) the engine file and (7) the delivered file are byte-identical across processes started with different hash seeds, and (8) a different seed changes the output.

## The benchmark

`benchmarks/` scores six arms against the human data of Brader, Valentino and Suhay (2008): nothing configured (declare-first default), direction declared, effects off, analyst d = 0.30, and two naive latent-normal baselines. The human data are fetched at run time from the Rdatasets mirror and checked against the pinned SHA-256 in `benchmarks/studies/brader_valentino_suhay_2008/human_data.sha256`; they are not redistributed. The design file (`study.json`) holds the analyst d and the direction declaration, both fixed before the runs reported in `benchmarks/results/`.

## Worked example

`make reproduce`, then `python docs/stage1_worked_example/rehearsal_analysis.py` (planned analysis: keeps 92 of 100 rows), `--typo` (stops on a misspelled column) and `--wrong-code` (stops because a miscoded exclusion rule would keep 8 of 100 rows).

## Where the rest is documented

`docs/CALIBRATION_STATUS.md` (the audit items, their fixes and measurements), `docs/EMPIRICAL_PROVENANCE.md` (registry status), `docs/ACCEPTABLE_USE.md` (permitted and prohibited uses).
