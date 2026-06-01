# Response to the Code Audit (v1.2.7.5)

This document records how each item in `research_simulations_code_audit.md` was
handled. Each item was **verified against the actual code before any change**;
several "potential" issues turned out to be already-safe and are marked as such.

Scope decision (confirmed with the maintainer): fix all verified correctness,
reproducibility, security, and hygiene defects; **defer the large architectural
refactors** (splitting `app.py`, breaking up the 2,500-line functions, JSON→SQLite)
to a separate staged effort — doing them in a single pass would risk regressions
across a 15k-line app. Those remain tracked below as "Deferred (roadmap)".

## 1. Blocking failure
| Item | Status | Notes |
|------|--------|-------|
| 1.1 `socsim/cli.py` IndentationError | ✅ Fixed | The parser-setup block (lines 334–399) had been dedented out of `main()`; re-indented it back in. `python -m compileall -q simulation_app` now exits 0 over the whole tree. |
| 1.2 tests miss compile failure | ✅ Addressed | Added `test_socsim_cli_compiles_and_has_main` (compiles cli.py via `py_compile`, asserts `main` exists). |

## 2. Security & privacy
| Item | Status | Notes |
|------|--------|-------|
| 2.1 Embedded XOR LLM keys | ⚠️ Partially / by-design | These are *free-tier, rate-limited* built-in keys for zero-config demo use, obfuscated (not "secured"). Real fix = revoke + move to secrets/proxy, which is an operational/ownership decision. Code already supports env-var / session keys as the preferred path. Flagged for the maintainer; not silently "fixed" by deletion (would break the documented zero-config mode). |
| 2.2 Plaintext group API key | ✅ Fixed | `group_management` keeps the live key in runtime memory only; persists just the hash + timestamp; strips any legacy raw key from disk on load. |
| 2.3 Session key "encryption" is masking | ⚠️ Documented | It is masking, not encryption; keeping user keys in session is the existing design. Left as-is (renaming/over-engineering deferred); no key is logged. |
| 2.4 Silent QSF upload | ✅ Already gated | Disabled by default (`GITHUB_COLLECTION_ENABLED=false`); requires an explicit secret to enable. (Redaction/dry-run = roadmap.) |
| 2.5 Token-prefix echo | ✅ Fixed | Validation message no longer includes `Got: {token[:10]}...`. |
| 2.6 LLM content leakage | ⚠️ Roadmap | Offline/template mode already exists as the privacy default; explicit consent + redaction before LLM calls = roadmap. |

## 3. Correctness bugs
| Item | Status | Notes |
|------|--------|-------|
| 3.1 Open-ended dict dropped | ✅ Fixed + test | `_normalize_open_ended` now accepts `variable_name`/`export_tag`/`question_id` and preserves `variable_name`. |
| 3.2 Scale item lists ignored | ✅ Fixed (both fns) + test | List-valued `items` now sets `num_items=len` and populates `item_names`. |
| 3.3 Pandas dtype mutation | ✅ Fixed | `_set_cell` widens an int column to float before writing a float (no FutureWarning, even under `-W error`). |
| 3.4 Salted `hash()` | ✅ Fixed (3 sites) | Replaced with SHA-256 `_stable_int_hash` in hbs_engine, enhanced_simulation_engine, socsim_adapter. |
| 3.5 Global RNG mutation | ✅ Fixed (verified safe) | Removed global `np.random.seed`/`random.seed` from engine/persona_library/response_library; generation already uses per-call seeded `RandomState` (proven unaffected). `text_generator` still seeds globally **by design** (57 bare calls; converting it is a deferred refactor). |
| 3.6 `_Generation_Source` overwrite | ✅ Fixed | HBS now writes `_Base_Generation_Source` (preserved) + `_Simulation_Method`. |
| 3.7 Hard-coded "current" facts | ⚠️ Roadmap | In `hbs_tool_dispatcher` factual lookup; low impact (offline helper). Noted. |
| 3.8 Timezone DST offsets | ⚠️ Roadmap | Fixed-offset table can be off by 1h during DST; `zoneinfo` migration noted. |

**Reproducibility result:** same seed → byte-identical output across fresh
processes and `PYTHONHASHSEED` values (verified end-to-end, incl. paradata).

## 4. Packaging & dependencies
| Item | Status | Notes |
|------|--------|-------|
| 4.1 Incomplete requirements | ✅ Addressed | Added `requirements-optional.txt` documenting the **guarded** optional deps (plotly, scipy, matplotlib, pdfplumber/PyMuPDF, requests, openpyxl, jsonschema). All are lazy imports with graceful fallbacks — the core app runs without them. |
| 4.2 socsim metadata conflict | ⚠️ Roadmap | Experimental package; version literals disagree. Noted. |
| 4.3 Module version drift | ✅ Improved | De-conflicted the engine docstring and llm_response_generator versions; the 9 mandated sync locations remain consistent (1.2.7.5). |

## 5–10. Architecture / concurrency / scientific framing
- **5.1–5.3** (split `app.py`, break giant functions, narrow exceptions): **Deferred** — large refactors; tracked in `docs/COVERAGE_ROADMAP.md`.
- **6.1** atomic writes: ✅ added for usage counter + group storage. **6.2/6.3** (SQLite for group state) **deferred**.
- **7.x** golden QSF tests / BeautifulSoup HTML / identifier blocking: identifier exclusion + attention-check filtering already shipped; golden-fixtures **roadmap**.
- **8.x / 9.x / 10.x**: SMTP error sanitization ✅; rule-trace metadata, validation datasets, upload limits = **roadmap**.

## New regression tests (this pass)
`tests/test_bugfixes_v1264.py`: open-ended `variable_name`, scale list-items,
stable-hash determinism, no-global-RNG-mutation, CLI-compiles-and-has-main —
plus the existing 17. All 22 pass.

## Hygiene
Untracked 32 accidentally-committed `.pyc` files; `.gitignore` now excludes all
bytecode.

## Validation (every change)
22 regression tests · effect fuzz (2,592 combos) · **0 crashes across all 291
example QSFs** · **0 issues across the 10 student QSFs** · e2e all-pass ·
`compileall` clean · version synced (1.2.7.5).
