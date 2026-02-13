# Memory

## 2026-02-12 — LLM-first simulation reliability overhaul (v1.0.6.8)

- Enforced explicit fallback consent behavior for open-ended generation:
  - If LLM providers are unavailable, users must either provide a key or explicitly allow one-run final fallback.
  - Without either, generation is blocked in LLM-first mode.
- Added runtime plumbing so `allow_template_fallback` flows from UI → engine → `LLMResponseGenerator`.
- Hardened `LLMResponseGenerator` to raise an error when all providers fail and fallback is disabled.
- Fixed critical condition corruption bug:
  - OE duplicate post-processing previously ran on all string-list columns and mutated `CONDITION` / `_PERSONA`.
  - Dedup now runs only on known open-ended columns.
- Added instructor report condition canonicalization to map contaminated labels (e.g., prefixed text) back to metadata conditions when unambiguous.
- Updated synchronized app/utils versions to `1.0.6.8`.


## 2026-02-12 — v1.0.6.9 follow-up
- Fixed LLM init regression (`os` import) that caused admin to report template-only runs.
- Reordered provider chain to put Google AI Studio first.
- Added OE quality filters + topical relevance checks to reduce gibberish.
- Added DV micro-dynamics (fatigue drift, streak inertia, endpoint correction) for more human-like item patterns.
- Added detailed roadmap doc: `docs/llm_and_behavior_overhaul_plan.md`.


## 2026-02-12 — v1.0.7.1
- Enforced stricter LLM-first integrity path with post-run API-activity assertion for OE runs.
- Removed default fallback UX; fallback now explicit and failure-time only.
- Added llm_init_error propagation to metadata and reporting surfaces.
- Added system-wide 10x roadmap: `docs/software_10x_improvements.md`.
- Upgraded versions to 1.0.7.1 and refreshed build id.
- Removed optional-skip blanking for visible OE items; simulations now enforce non-empty OE outputs.
- Added adaptive batch retries (20/10/5/1) in LLM response generation to avoid batch-size-induced sparse outputs.
- Added an admin provider quota/cost calculator for planning request/token demand per run.
- Improved emergency fallback wording to produce contextual, non-generic OE text when providers and templates both fail.
- Updated README provider-chain notes to reflect Google-first ordering.

## 2026-02-13 — Streamlit import KeyError hotfix (v1.0.7.2)
- Fixed startup crash path caused by deleting `utils*` from `sys.modules` during app boot/version checks.
- Removed runtime module-purge/re-import behavior; version mismatch now shows warning-only and asks for restart/redeploy.
- Added regression guard test to prevent reintroducing `sys.modules` purge logic in `simulation_app/app.py`.
- Workflow rule added: do not mutate `sys.modules` for hot-reload in Streamlit entrypoints; use `BUILD_ID` bump + clean restart for cache invalidation.

## 2026-02-13 — Run archive + continuous improvement workflow (v1.0.7.3)
- Added mandatory per-simulation archival folders in `data/simulation_runs/` containing:
  - `Simulated_Data.csv`
  - `Instructor_Copy.md`
  - `Metadata.json`
  - `Validation_Results.json` (when available)
  - `Engine_Log.txt` (when available)
  - `Quality_Audit.json`
- Added unseen-run scanner workflow:
  - Tracks previously audited folders with `.run_audit_state.json`
  - Audits only newly created run folders
  - Appends timestamped findings/recommendations to `continuous_improvement_log.txt`
- Added workflow rule: use `continuous_improvement_log.txt` as the first debugging source for future coding passes (Codex/Claude), then patch and add regression tests for the detected issue signatures.
- Added tests for run persistence and new-run-only auditing behavior to prevent regressions.

- Refinement pass: added `Run_Manifest.json` with per-file SHA-256 checksums for each run folder.
- Refinement pass: audit now flags short/gibberish-like OE outputs (`oe_too_short` for <5-word responses over threshold).
- Refinement pass: scanner now ignores hidden folders and writes aggregate `audit_summary.json` each pass.
- Refinement pass: audit now verifies run manifest presence/parseability and logs `manifest_missing`/`manifest_parse_failed` for trace integrity.
- Refinement pass: audit now writes cumulative `issue_trends.json` and exposes top recurring issue codes for faster triage across runs.
