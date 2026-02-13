# 10x quality plan (v0.16)

This is an implementation-oriented roadmap, not a promise of accuracy.

## 1. Benchmark coverage and rigor
- Add many benchmark packs with explicit license and citation metadata.
- Add richer moment libraries per domain (distributional, heterogeneity, dynamics).
- Pin downloads by sha256; store provenance for reproducibility.

## 2. Predictive training loops
- Held-out selection for calibration hyperparameters (implemented scaffolding).
- Multi-objective optimization (fit core moments while not overfitting tails).
- Hierarchical priors per latent class and domain.

## 3. Mechanism libraries
- Add mechanism components: social norms, identity, reciprocity, learning, attention, affect.
- Compose mechanisms by domain templates (e.g., political persuasion vs consumer choice).

## 4. Expanded experimental methods
- More games (network games, contests, principal-agent, bribery variants).
- More survey methods (vignettes, RRT variants, item-count, endorsement, conjoint).

## 5. Evidence store
- Store and query structured evidence units (paper -> parameters -> uncertainty).
- Track which mechanisms are supported by which evidence units.

## 6. Diagnostics and reporting
- Calibration reports with error decomposition, sensitivity, and ablations.
- Consistency checks (parameter identifiability, posterior predictive checks).

## 7. Engineering reliability
- Smoke tests + unit tests + reproducible seeds (smoke iterations implemented).
- Stable schemas for packs, targets, and results.

## 8. Extensibility
- Plugin API for new games and surveys.
- Dataset adapters as declarative transforms.

## 9. Documentation
- Clear end-to-end guide: dataset -> pack -> targets -> calibration -> evaluation.

## 10. Guardrails
- Prevent unsupported claims of predictive performance; report uncertainty.
