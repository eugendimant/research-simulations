# socsim v0.3 inner workings (design notes)

## 1) What the simulator is (and is not)
- It is a mechanism-based generative model for behavior in experiments.
- It is not a guarantee of accuracy in novel domains.

The simulator aims for:
- traceability (what evidence was used),
- modularity (swap mechanisms, games, survey models),
- calibration support (fit a meta-model from your own datasets),
- explicit uncertainty (via priors, evidence uncertainty, and choice stochasticity).

## 2) Architectural layers

### Layer A: Spec ingestion
Input is an ExperimentSpec (JSON) with:
- game name + parameters (payoffs, choice grids)
- topic tags (identity, discrimination, honesty, etc.)
- context fields (anonymity, group salience, norm salience, stakes, repetition, etc.)
- population fields (country, sampling frame)

The spec is converted to a flat feature dict, for matching evidence units.

### Layer B: Atomic evidence units (EvidenceUnit)
An EvidenceUnit is an atomic claim extracted from published work, meant to be machine-actionable.

Types supported:
- param_shift: shifts a latent parameter distribution (mean and/or sd) under conditions.
- moment_target: targets an observable moment (mean give rate, cooperation rate, acceptance rate).
- structural_estimate: stores an estimated structural parameter with uncertainty.

Each unit includes:
- `when`: feature match pattern (game, topic tags, context, population)
- `effect`: what it changes (e.g., norm_weight +0.25)
- `uncertainty`: optional standard error / interval
- `provenance`: required citation fields (doi or url), plus extraction notes.

The schema forces provenance: it rejects units without a citation.

### Layer C: Causal parameter generator (SCM-style)
We use a structural form:
param = prior + sum(evidence_shifts) + class_shift + meta_model_shift + intervention_override

- priors are distributions (mean, sd, min, max)
- evidence_shifts are additive shifts (with optional extra variance)
- class_shift captures latent types (self-interested, fairness-minded, etc.)
- meta_model_shift comes from your own fitted ridge meta-weights (optional)
- intervention_override is a direct do-operator: replace a parameter with a fixed value

This is not a full general SCM language. It is a practical, explicit causal pipeline
for counterfactuals at the level social scientists typically manipulate: context and incentives.

### Layer D: Personas
A persona is a draw of all latent parameters (fairness, reciprocity, risk, time discounting, etc.)
plus a latent class label. Personas are independent draws unless you choose multi-task coupling.

### Layer E: Task engines (games and surveys)
Each task is a "Game" with a simulate_one() method.
- Games compute utilities under candidate actions and pick stochastically via logit choice.
- The Likert survey block uses a graded response model (GRM) with a latent trait theta
  and item discrimination/threshold parameters.

### Layer F: Traces and outputs
Each simulated row can include a JSON trace with:
- matched evidence IDs
- parameter shifts applied
- utilities / choice probabilities
- any task-specific decomposition

This is intended to support debugging, replication, and later LLM-based free text generation.

## 3) What you must do to approach high predictive accuracy
1) Build a curated evidence store (atomic units) with real citations.
2) Add per-domain priors and population priors.
3) Add evaluation: posterior predictive checks and out-of-domain warnings.
4) Fit a meta-model on your own compilation datasets (feature -> parameter shift).
5) Keep evidence unit scope tight to avoid over-transfer.

## 4) Minimal reproducible workflow
- Start with a small set of games you care about.
- Encode 50-200 evidence units from solid, replicated sources.
- Calibrate priors against benchmark datasets (lab + online).
- Run out-of-sample checks by holding out whole contexts.
- Expand games and survey types, but only when you can evaluate them.



## Atomic Insight Units (AIUs)

SocSim separates **(1) verifiable literature metadata** from **(2) quantitative effect coding**.

- **Corpus (AIUs):** Metadata-only records keyed by DOI/URL that can be checked externally. These do *not* change behavior parameters.
- **Evidence units:** Quantitative objects that shift priors or target moments. These must include provenance and should be added only after manual coding/verification.

### Why split these?

- Prevents accidental use of unverified numeric effects.
- Lets the system stay "self-updating" on the bibliographic layer while keeping the causal/quant layer auditable.

### Corpus workflow

1. `socsim/data/corpus_seed_queries.json` contains 100 seed queries spanning behavioral games, frames, and social-science mechanisms.
2. `socsim.cli corpus_expand` queries Crossref and stores results in `socsim/data/corpus.json`.
3. Use `corpus_validate` to validate against `socsim/corpus/schema.json`.

### Upgrading AIUs to Evidence Units

AIUs are *candidates*. A separate, auditable coding step should:

- read the paper/appendix,
- extract estimands (treatment effects, structural estimates, or targeted moments),
- map them to SocSim parameters or moments,
- write an `EvidenceUnit` with DOI/URL provenance.

Only then does the simulator use it.


## Corpus seed queries
`socsim/data/corpus_seed_queries.json` now contains 200 seed queries (first 100 + next 100). Run `python -m socsim.cli corpus_expand --rows 25 --sleep 0.25 --corpus socsim/data/corpus.json` to fetch metadata-only units.

## 8. Autocorpus (next 100 metadata-only references)

Run:
`python -m socsim autocorpus --n 100 --out ignored`

This updates `socsim/data/evidence_store.json` with metadata-only Crossref references using an internal query pack.
## Recursive improvement loops (66)
1. Add coverage report in simulation summary. [Implemented]
2. Add metadata-only web bibliography expansion. [Implemented]
3. Add web access absolutes document. [Implemented]
4. Add SQLite HTTP cache with SHA256. [Implemented]
5. Add rate limiting and bounded retries. [Implemented]
6. Add Crossref/OpenAlex/Semantic Scholar harvesters. [Implemented]
7. Add corpus store with schema validation. [Implemented]
8. Add CLI command corpus_expand_web. [Implemented]
9. Repair CLI and `python -m socsim` entrypoint. [Implemented]
10. Stabilize game registry + aliases. [Implemented]
11. Add risk task (Holt-Laury). [Implemented]
12. Add time MPL task. [Implemented]
13. Add repeated PD. [Implemented]
14. Add public goods group simulation. [Implemented]
15. Add pred vs obs moment comparison. [Implemented]
16. Add pytest smoke tests. [Implemented]
17. Use evidence quality weighting in ContextEngine. [Planned]
18. Conflict detection (opposing effects). [Planned]
19. Conflict resolution hierarchy. [Planned]
20. Explicit mapping from evidence units to parameter shifts. [Planned]
21. Transportability scorer based on context overlap. [Planned]
22. Stakes/anonymity/repetition moderators across games. [Planned]
23. Identity/discrimination moderators. [Planned]
24. Experimenter-demand module. [Planned]
25. Observability/social image module. [Planned]
26. Belief updating module. [Planned]
27. Trust with reputation/history. [Planned]
28. Punishment/reward in PGG. [Planned]
29. Partner choice/assortative matching. [Planned]
30. Network formation/diffusion. [Planned]
31. Discrete choice/conjoint. [Planned]
32. Auction/BDM tasks. [Planned]
33. Battle-of-the-sexes / coordination. [Planned]
34. Corruption/bribery game. [Planned]
35. Tax compliance game. [Planned]
36. Norm elicitation task. [Planned]
37. SVO measure. [Planned]
38. IRT/GRM for Likert batteries. [Planned]
39. Fatigue/satisficing model. [Planned]
40. Response-time model. [Planned]
41. Calibration pipeline (fit priors + shifts). [Planned]
42. Bayesian hierarchical pooling across studies. [Planned]
43. Meta-analytic ingestion format (structured). [Planned]
44. Audited PDF extractor (optional). [Planned]
45. Citation graph dedup (DOI resolution). [Planned]
46. Append-only provenance ledger. [Planned]
47. Model cards + audit reports. [Planned]
48. Drift detection. [Planned]
49. Qualtrics QSF parser (metadata only). [Planned]
50. Plugin interface for custom games/evidence mappers. [Planned]
51. Parallel simulation runner. [Planned]
52. Deterministic replay bundles (spec + cache snapshot). [Planned]
53. CI workflow for tests/lint. [Planned]
54. Benchmark suite (no auto-claims). [Planned]
55. Multiple population priors scaffolding. [Planned]
56. Multilingual survey scaffolding. [Planned]
57. Fairness audit mode (opt-in). [Planned]
58. Evidence-linked explanation generator (no free text). [Planned]
59. Uncertainty decomposition. [Planned]
60. Robust missingness handling in obs comparison. [Planned]
61. Effect attenuation with context mismatch. [Planned]
62. Dose-response intervention hooks. [Planned]
63. Attention check failure probability. [Planned]
64. Human-vs-AI partner framing moderator. [Planned]
65. Culture dimension moderator (curated inputs only). [Planned]
66. Publication bias sensitivity in aggregation. [Planned]


## Implemented top-10 features (v0.11.0)
1. Audited CSV effect extractor (strict format only).
2. Evidence-quality weighting (explicit metadata only).
3. Conflict detection and reporting.
4. Transportability scoring and attenuation.
5. SCM skeleton with do-style context overrides.
6. Calibration command (ridge per moment).
7. Survey battery GRM simulator (actions-only).
8. Reputation belief-updating module scaffolding.
9. Benchmark suite and benchmark runner.
10. End-to-end integration into simulation summary and traces.


## v0.11.0 additions

### Reproducibility primitives
- Every run has a `run_id`, optional JSONL logs, and produces a `MODEL_CARD.md`.
- `export_bundle` produces a zip that includes `MANIFEST.json` with SHA256 hashes for every file included.

### Web expansion discipline
- Harvesting remains metadata-only.
- Stable bibliography IDs are derived from SHA256 of (title, stable ref, source) after DOI normalization.

### Safe-mode
- `--safe-mode` can block runs when:
  - coverage_score < min_coverage
  - conflicts exist (optional with --abort-on-conflict)

### New behavioral tasks
- repeated trust (belief updating)
- public goods with punishment stage
- discrete choice (MNL with Gumbel noise)
- BDM mechanism


## Benchmarks and calibration
See BENCHMARKS.md.


### bench_scout
Uses OpenAlex search to propose candidate benchmark entries and likely repository links.
