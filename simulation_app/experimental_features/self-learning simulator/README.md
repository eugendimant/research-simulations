# socsim (v0.3): evidence-traceable behavioral simulation scaffold

[Unverified] This package does not and cannot "make no mistake" or perfectly predict human behavior.
What it does provide:
- A reproducible pipeline to encode published findings as atomic evidence units with provenance.
- A causal-structure-oriented parameter generator (SCM-style) that supports interventions.
- Heterogeneous personas (mixture classes + continuous traits).
- Simulation engines for common behavioral games and Likert survey blocks.
- Full trace logs: which evidence matched, which parameters shifted, and why.

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

## Quickstart
```bash
python -m socsim simulate --spec examples/spec_dictator_identity.json --n 500 --seed 7
python -m socsim simulate --spec examples/spec_staghunt_identity.json --n 500 --seed 7
python -m socsim simulate --spec examples/spec_time_mpl.json --n 500 --seed 7
python -m socsim simulate --spec examples/spec_survey_likert.json --n 200 --seed 7 --traces
```

## Evidence store workflow
Validate and append an evidence unit to your local evidence store:
```bash
python -m socsim evidence_validate --evidence examples/evidence_units/eu_norm_salience.json
python -m socsim evidence_add --evidence examples/evidence_units/eu_norm_salience.json --store socsim/data/evidence_store.json
```

List evidence units that match a spec:
```bash
python -m socsim evidence_match --spec examples/spec_dictator_identity.json --store socsim/data/evidence_store.json
```

## Files you will care about
- `SOCSIM_INNER_WORKINGS.md` : detailed explanation of the architecture and causal flow
- `socsim/data/evidence_store.json` : atomic evidence units (with provenance fields)
- `socsim/data/priors.json` : parameter priors
- `socsim/schema/evidence_unit_schema.json` : strict schema that forces citations/provenance
- `socsim/schema/experiment_schema.json` : recommended experiment spec schema

## What is "causal-structure-oriented" here?
We treat the simulator as a structural model:

context + topic + population  ->  latent parameters (mechanisms)  ->  actions

The "latent parameters" step is where evidence units act as causal modifiers.
You can also intervene: do(param = value) to simulate policy/treatment counterfactuals.

## New in v0.11.0

- Added additional games: ultimatum, trust, public_goods, die_roll, sender_receiver, gift_exchange.
- Added a **corpus** subsystem to store *atomic insight units* as metadata-only records with verifiable identifiers (DOI/URL), created automatically from seed queries via Crossref.
- Evidence store defaults to **empty** to avoid applying unverified numeric effects.

### Build/extend the corpus (metadata only)

```bash
python -m socsim corpus_expand --rows 1 --mailto you@example.edu
python -m socsim corpus_validate
```

This writes/updates `socsim/data/corpus.json`.

### Run a simulation

```bash
python -m socsim simulate --spec examples/spec_trust_identity.json --n 500 --seed 1 --out out_trust
```

```bash
python -m socsim simulate --spec examples/spec_risk_holt_laury.json --n 200 --seed 7 --traces
```

## Autonomously expand corpus (next 100)

```bash
python -m socsim autocorpus --n 100 --store socsim/data/evidence_store.json
```

## New in v0.11.0
- Added games: holt_laury, mpl_time, repeated_pd
- Added conservative moment-evaluation utility: eval_moments
- Added a minimal pytest smoke test
- Added coverage report to every simulation summary

## Run tests
```bash
pytest -q
```


## Registered games (v0.11.0)
- dictator
- pd
- stag_hunt
- ultimatum
- trust
- public_goods (supports simulate_group via context.simulate_group)
- sender_receiver
- gift_exchange
- die_roll
- time_mpl (alias: mpl_time)
- risk_holt_laury (alias: holt_laury)
- repeated_pd


## Web access absolutes
See `WEB_ACCESS_ABSOLUTES.md`.

## Web-expand bibliography into atomic units (metadata-only)
```bash
python -m socsim corpus_expand_web "trust game meta-analysis" "public goods punishment meta-analysis" --mailto you@domain.edu
```
This writes metadata-only units (verifiable DOIs/URLs) into `socsim/data/corpus.json`.


## Top 10 next features identified (roadmap)
1. Audited effect-size extractors for narrow, well-specified formats.
2. Evidence quality scoring and weighting that affects context shifts.
3. Conflict detection and resolution for opposing evidence.
4. Transportability checks (population, incentives, stakes, anonymity, culture).
5. Structural causal model layer (explicit nodes, interventions).
6. Dataset ingestion and calibration loop (fit priors and context shift parameters).
7. Survey response models (IRT/GRM) for Likert batteries (actions only).
8. Repeated-game reputation module (trust with histories and beliefs).
9. Network formation and peer effects module.
10. Benchmark suite with canonical replications and regression tests.


## Implemented feature upgrades (v0.11.0)
- Audited CSV effect-size extractor + CLI ingest command (`ingest_csv_effects`).
- Evidence quality scoring and weighting (explicit metadata only).
- Conflict detection and reporting in run summary.
- Transportability scoring + attenuation using explicit expected context features.
- SCM skeleton with `context.interventions` applied as do-overrides.
- Calibration command (`calibrate`) using auditable ridge-per-moment fits.
- Survey battery simulation (GRM, actions-only) via `context.survey` -> `sv::*` columns.
- Benchmark runner (`benchmark`) with a suite file.


## Benchmarks and calibration
See BENCHMARKS.md.


## Roadmap
See `NEXT_STEPS_PLAN.md`.


## Specs
```bash
python -m socsim spec_template --game ultimatum > my_spec.json
python -m socsim spec_lint --spec my_spec.json
```


## Insights
```bash
python -m socsim insight_add --item path/to/insight.json
python -m socsim insight_search --game ultimatum
```


## Diagnostics
```bash
python -m socsim doctor
python -m socsim version
```


## Benchmark packs
See `BENCHMARK_PACKS.md`.


## Predictive roadmap
See `PREDICTIVE_ROADMAP.md`.


## Survey primitives
See `SURVEYS.md`.


## Field taxonomy
See `FIELDS.md`.


## Smoke test
Run `python scripts/smoke.py`.
