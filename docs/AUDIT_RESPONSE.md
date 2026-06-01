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

---

# Deep adversarial re-audit (v1.2.7.9)

After the v1.2.7.8 concurrency fixes, an independent adversarial agent traced the
generation/parsing/LLM/state surface end-to-end and **reproduced** four defects the
earlier passes missed. Each was verified against the code before fixing. It also
confirmed the v1.2.7.8 fixes (lock, per-instance OE state, non-cached parser) are
correct, and that `GroupManager`/`APIKeyManager` sharing is out of the data path.

| ID | Sev | Issue | Fix | Test |
|----|-----|-------|-----|------|
| H1 | HIGH | LLM pool key drift: prefill omitted the `\nCondition:`/`\nAdditional context:` suffixes the per-participant path adds, so every prefilled draw missed when `question_context` was set (prefill budget silently wasted; anti-pattern #29). | Single shared `_build_enriched_question_text()` builds the key for BOTH paths → byte-identical keys. | `test_h1_prefill_runtime_pool_key_match` |
| M1 | MED | `free_tier_exhausted_now` latched permanently for the run (its counter only reset inside a function every caller then skipped), contradicting its "recoverable" docstring. | Reset `_consecutive_transient_batches` in `reset_providers()` and per OE question. | `test_m1_transient_latch_resets_on_reset_providers` |
| M2 | MED | Reused engine produced different OE text on the same seed — only `text_generator` was reset, not the primary `comprehensive_generator`. | Added `reset()` to `ComprehensiveResponseGenerator` + ABE-v2 wrapper; called each run. | `test_m2_reused_engine_is_reproducible_non_llm` |
| L1 | LOW | `_used_responses`/`_used_sentences` unbounded at N=10,000 × many OE questions. | FIFO-bounded via companion `deque`s with **deterministic** oldest-first eviction (never `set.pop()`). | covered by cross-process determinism battery |

**Conscious trade-off (not a defect):** the `_GLOBAL_RNG_LOCK` is held across LLM
network I/O, so concurrent users serialize. Eliminating global seeding (routing ~70
RNG sites — engine `rng = np.random` ×15 + `text_generator` `random.*` ×58 — through
injected per-run RNGs) is deferred as its own verified change; it risks the
reproducibility just confirmed, and real-world impact is low (free-tier rate limits
serialize concurrent LLM users anyway; non-LLM N=10,000 ≈ 60s). Recorded in the
Self-Audit Bug-Class Catalog alongside new classes #13–16.

## Validation (v1.2.7.9)
29 bugfix regressions + 6 e2e = **35 pass** · cross-process determinism
byte-identical across `PYTHONHASHSEED` ∈ {0,1,42,31337} · random **N=200** across 10
QSFs all-pass · `compileall` exit 0 · version synced (1.2.7.9).
