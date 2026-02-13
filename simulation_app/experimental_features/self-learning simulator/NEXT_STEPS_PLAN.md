# Roadmap to Make socsim More Useful and Predictive (v0.14+)

This plan prioritizes predictive power, auditability, and benchmark realism.

## A. Inputs that make prediction feasible (ExperimentSpec+)
1. Canonical fields
- game: name + parameters (stakes, repetition, info structure, partner matching)
- topic: name + tags + manipulation descriptors
- population: label + country + sample frame (lab/online/student)
- context: anonymity, observability, identity salience, framing, timing, payoff scale

2. Parsers
- QSF parser: extract page flow, randomization, question types, payoff text, treatments
- Text parser (metadata-only): map natural language setup → fields, with uncertainty tags

3. Validation + lint
- strict schema + helpful errors
- lint rules: missing keys, impossible params, inconsistent repetition fields

## B. Atomic scientific insights (auditable and composable)
1. Insight Unit = minimal causal statement
- scope: games/topics/populations
- mechanism: mediator label(s)
- effect distribution: mean/sd/bounds and target parameter(s)
- moderators: how context shifts effect
- provenance: DOI/URL, fetch timestamp, checksum if available
- quality: design strength / risk-of-bias

2. Retrieval + composition
- retrieve by scope overlap
- conservative composition: quality-weighted shrinkage + uncertainty inflation outside scope

## C. Mechanism layer for extrapolation
- mediator state per agent: prosociality, reciprocity, conformity, image concern, risk
- context → mediator shifts (via insights)
- choice policy: stochastic mapping from mediators → actions

## D. Benchmarks: real datasets drive predictive power
1. Benchmark pack format
- manifest: URLs + sha256 + timestamps
- adapter: explicit column mapping
- moments: explicit definitions (not only heuristics)
- metadata: population, manipulations, exclusions

2. Targets beyond means
- quantiles, distribution distances
- interaction effects
- dynamic moments (round-by-round)
- subgroup moments (heterogeneity)

## E. Training loop
- hierarchical calibration with partial pooling as default
- LOO-CV and holdout evaluation as first class
- calibration reports: uncertainty + sensitivity

## F. Engineering + UX
- stable CLI commands: bench_cv, bench_scoreboard, bench_pack, spec_lint, spec_template, insight_add/search
- caching, polite rate limiting, reproducibility meta everywhere
- docs: end-to-end "add a real benchmark" tutorial
