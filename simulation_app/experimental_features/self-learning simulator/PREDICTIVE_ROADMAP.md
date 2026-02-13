# Predictive Roadmap (v0.15+)

Goal: maximize out-of-sample predictive accuracy for behavioral outcomes in experimental games and surveys,
while keeping the system auditable, reproducible, and conservative when extrapolating.

This version implements:
- benchmark packs with integrity verification (SHA256)
- an explicit moments DSL that produces reproducible targets
- a pack runner + CLI to compute targets from a pack
- tests and verification scripts

Next priority (not fully implemented here):
- real benchmark ingestion (licensed) + more moment types + hierarchical calibration.
