# SocSim v0.5.0 Technical Overview

This document is intentionally conservative: it describes what the software **actually does today**, what it **refuses to do without verified inputs**, and the workflow to extend it.

## 1) Goal and boundary conditions

SocSim is a *behavior simulator* for social-science experiments that is meant to be:

- **Context-aware** (game, parameters, framing, population).
- **Persona-based** (latent classes + individual parameter draws).
- **Evidence-traceable** (all numeric effects must come from explicit evidence units with provenance).
- **Corpus-backed** (a large, self-updating index of literature metadata that can be verified via DOI/URL).

It is **not** an automatic “perfect predictor of humans.” It is a modular scaffold designed to make *incremental* progress while keeping numeric claims auditable.

## 2) Core data structures

### 2.1 Experiment specification

`ExperimentSpec` (see `socsim/specs.py`) is a JSON-serializable spec that includes:

- `game`: `{ name, params }`
- `topic`: `{ name, tags, notes }`
- `population`: `{ label, country, sampling_frame }`
- `context`: arbitrary key/value fields (booleans, numbers, strings)

`ExperimentSpec.to_feature_dict()` converts the spec into a flat feature map used for evidence matching:
- game name at `game`
- topic at `topic`
- tags at `topic_tag::<tag>`
- context at `ctx::<key>`

### 2.2 Personas (latent agents)

A `Persona` is a parameter vector `params` plus a `latent_class`.
- Priors are in `socsim/data/priors.json` (truncated normals).
- Latent-class mixture weights and mean shifts are in `socsim/data/latent_classes.json`.

`PersonaGenerator.sample(...)` draws a persona by:
1. Sampling a latent class.
2. Drawing each parameter from its prior.
3. Applying mean shifts (from matched evidence and class shifts).
4. Applying optional interventions (`do()` operators) if provided.

### 2.3 Evidence units (numeric effects, strictly opt-in)

Evidence units are stored in `socsim/data/evidence_store.json`.
- Default store is **empty**.
- Each evidence unit must validate against `socsim/schema/evidence_unit_schema.json`.

Supported evidence types:
- `param_shift`: additive deltas to parameter means (applied when feature pattern matches).
- `moment_target`: reserved (not used in v0.5.0).
- `structural_estimate`: reserved (not used in v0.5.0).

**Important:** SocSim will never invent numeric effects. If you want behavior to change due to a manipulation, you add a corresponding evidence unit with provenance and coding notes.

### 2.4 Corpus (atomic insight units, metadata only)

The corpus lives in `socsim/data/corpus.json`. Units are validated against `socsim/corpus/schema.json`.

A corpus unit is *metadata only*:
- `source`: `{type: doi|url, ref: ...}` so existence can be verified.
- `bibliographic`: year, venue, authors (as available).
- `quality.status`: `metadata_only` by default.

Corpus units do not affect simulations unless you later create *evidence units* that encode numeric effects.

## 3) Simulation pipeline

`simulate(...)` in `socsim/simulator.py` does:

1. Load priors, latent classes, evidence store.
2. Convert experiment spec to features.
3. Match evidence units by feature patterns and aggregate mean shifts + extra SD.
4. Sample personas.
5. Simulate the selected game by calling `Game.simulate_one(...)`.
6. Return:
   - A row-level dataset (`rows`) with params, actions, payoffs.
   - A summary (`summary`) with action means, class counts, matched evidence ids, corpus size.

## 4) Games implemented (v0.5.0)

Games are in `socsim/games/` and are discrete-choice approximations using a logit/softmax rule.

- `dictator`
- `ultimatum`
- `trust`
- `public_goods`
- `pd`
- `stag_hunt`
- `gift_exchange`
- `sender_receiver`
- `die_roll`
- `time_mpl`
- `survey_likert`

Each game produces:
- `actions`: the chosen decision variables
- `payoffs`: resulting material payoffs
- `trace`: internal decision grids and choice probabilities (optional to persist)

## 5) “Atomic insights” workflow (verifiable units)

**Goal:** store insights as atomic, reusable building blocks.

SocSim splits this into two layers:

1) **Atomic Insight Unit (AIU):** verifiable bibliographic anchor (DOI/URL + tags).
2) **Evidence unit:** a *coded, quantitative* mapping from some condition(s) to a parameter shift or moment target.

### 5.1 Build a corpus from seed queries (fully automatic)

The seed query list is in `socsim/data/corpus_seed_queries.json` (100 queries).

Command:
```bash
python -m socsim.cli corpus_expand --rows 1 --mailto you@example.edu
```

This uses Crossref to fetch metadata and adds `metadata_only` AIUs.

### 5.2 Promote AIUs into evidence units (semi-automatic by design)

To avoid hallucination, the “promotion step” requires a coder:
- Identify the relevant manipulation and outcome(s).
- Choose a parameterization target (e.g., shift `baseline_prosocial`, `honesty_cost`, `belief_return_frac`).
- Encode effect size with uncertainty.
- Provide provenance (DOI/URL) and extraction notes.

Then:
```bash
python -m socsim.cli evidence_validate --evidence path/to/unit.json
python -m socsim.cli evidence_add --evidence path/to/unit.json
```

## 6) Roadmap (59 iteration loops)

This repository intentionally includes the scaffold needed to run many incremental iterations. Examples of iteration units:

1. Add a new game with a minimal decision model.
2. Add a new parameter with bounded prior.
3. Add a new evidence type (moment targets → parameter calibration).
4. Add an extractor pipeline that proposes candidate evidence units from AIUs, but keeps them “pending” until validated.
5. Add hierarchical population priors (country, platform, lab vs field).
6. Add cross-game parameter linking and measurement models.
7. Add robust causal-graph hooks for treatments and moderators.

If you want, the next step is to add:
- a `moment_calibration` module (simulate, compute moments, optimize parameter shifts),
- a reproducible “coder UI” for evidence promotion,
- and a benchmark suite against public datasets.

