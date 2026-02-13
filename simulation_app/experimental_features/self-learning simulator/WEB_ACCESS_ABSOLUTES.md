# Web access absolutes (socsim)

These constraints exist to avoid inventing evidence and to keep runs reproducible.

1. Web harvesting is metadata-only unless an explicit, audited extractor is run.
2. Any stored unit must include a resolvable reference (DOI or stable URL) in `source`.
3. All fetched bodies are timestamped and SHA256-hashed; cache is persisted (SQLite).
4. Rate limiting is mandatory; bounded retries with backoff are used.
5. Every stored unit includes provenance (who/when/how).
6. Bibliography (atomic units) is kept separate from causal evidence (evidence units).
7. Causal shifts are applied only from evidence units that pass JSON schema validation.
8. Quality weighting uses only explicitly provided metadata (no inference).
9. Transportability scoring uses only explicit `applicability.context_features` (no inference).
10. Conflicts are detected and reported; they are not silently ignored.

11. Stable bibliography IDs are deterministic (SHA256), not Python hash().
