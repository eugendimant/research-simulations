# Continuous Learning and Calibration System

The simulator improves with every run. This file specifies how simulation outputs are archived, compared against benchmarks, and used to calibrate future simulations.

## Core Principle

Every simulation run is both a deliverable AND a training sample. The system accumulates evidence about what works and what fails, then uses this evidence to make future simulations better.

## Architecture Overview

```
SIMULATION RUN
     │
     ├──→ DELIVER output to user
     │
     └──→ AUTO-ARCHIVE to GitHub
          │
          ├── Raw output (CSV, metadata)
          ├── Quality metrics (computed automatically)
          ├── Prompt snapshots (exact prompts used)
          └── Configuration (personas, parameters, condition specs)
                    │
                    ▼
          CALIBRATION DATABASE (grows over time)
                    │
                    ├── Compare against real human data (when available)
                    ├── Track quality metrics across runs
                    ├── Identify prompt patterns that produce best output
                    └── Update template parameters and covariance matrices
                              │
                              ▼
                    IMPROVED DEFAULTS for next simulation
```

## Auto-Archive Protocol

After every simulation run that produces output, automatically archive:

### What to Store

```
simulation_archive/
├── runs/
│   └── {timestamp}_{experiment_name}/
│       ├── config.json              # Full simulation configuration
│       │   ├── n_participants
│       │   ├── persona_library (types, mixture priors, trait distributions)
│       │   ├── condition_specs
│       │   ├── target_population
│       │   └── survey_structure (item list, skip logic, scales)
│       │
│       ├── prompts/                 # Exact prompts used (snapshots)
│       │   ├── subject_profile_template.txt
│       │   ├── item_prompts/        # One file per item type
│       │   └── prompt_version.txt   # Hash or version identifier
│       │
│       ├── output/
│       │   ├── simulated_data.csv   # The actual output
│       │   └── paradata.csv         # Timing, metadata (if generated)
│       │
│       ├── quality/
│       │   ├── metrics.json         # Automated quality metrics
│       │   ├── consistency_audit.json  # Intra-subject audit results
│       │   ├── distribution_stats.json # Means, SDs, skewness per item
│       │   └── flags.json           # Items/subjects that failed checks
│       │
│       └── generation_log.json      # Per-item: method used (LLM/template/hybrid),
│                                    # attempts needed, fallbacks triggered
│
├── training_data/                   # Real human data for calibration
│   └── {dataset_name}/
│       ├── data.csv
│       ├── metadata.json            # Source, N, population, measures
│       └── benchmarks.json          # Computed benchmarks from real data
│
├── calibration/
│   ├── covariance_matrices/         # Estimated construct correlations
│   │   └── {experiment_type}.json
│   ├── distribution_targets/        # Target distributions from real data
│   │   └── {item_type}.json
│   └── prompt_performance.json      # Which prompt versions produce best quality
│
└── response_library/                # Template response library (see template-system.md)
    └── ...
```

### When to Archive

- **Always**: After any simulation run that produces ≥50 participant rows
- **Quality metrics**: Computed automatically during the validation step — no extra work needed
- **Prompt snapshots**: Capture the exact prompt text used, including any per-run customizations
- **Generation log**: Track which generation method (LLM vs template vs hybrid) was used for each item, how many attempts were needed, and whether fallbacks were triggered

### Git Workflow for Archives

```bash
# After simulation completes and passes quality checks:
git add simulation_archive/runs/{latest_run}/
git commit -m "archive: {experiment_name} N={n} quality_score={score}"
git push origin main
```

Archives should be committed to the repository automatically. They are lightweight (CSVs + JSON) and invaluable for calibration.

## Calibration Against Real Human Data

When real human data is available for the same or similar experiment, use it as a benchmark following the Manning & Horton train/test methodology.

### Train/Test Protocol

1. **Split real data**: If you have one dataset, split 70/30 train/test. If you have multiple datasets from similar experiments, use one for training and another for testing.

2. **Calibrate on training data**:
   - Compare simulated response distributions (means, SDs, skewness) to training data distributions
   - Compute item-level KS statistics (continuous items) or chi-square statistics (categorical items)
   - Identify items where simulated output deviates most from human data
   - Adjust: persona parameters, noise levels, acquiescence bias, covariance matrix entries
   - Re-simulate and compare again (iterate until KS p-values > 0.05 on training set)

3. **Validate on testing data**:
   - Run the calibrated simulation against the held-out test data
   - If test-set quality metrics are comparable to training-set metrics → calibration is generalizing
   - If test-set metrics are substantially worse → overfitting to training data; simplify the calibration

4. **Store calibrated parameters** in `calibration/` for reuse in future simulations of similar experiments.

### What to Compare

| Metric | Computation | Target |
|--------|-------------|--------|
| Item means | Mean per item, simulated vs real | Within 0.5 SD of real mean |
| Item SDs | SD per item | Ratio between 0.7 and 1.3 of real SD |
| Scale reliability | Cronbach's alpha per scale | Within 0.1 of real alpha |
| Construct correlations | Inter-scale correlations | Within 0.15 of real correlations |
| Effect sizes (between conditions) | Cohen's d per condition contrast | Within CI of real effect size |
| Open-text length distribution | Mean, median, SD of word counts | Within 20% of real |
| Optional field completion rate | Proportion non-missing | Within 10pp of real rate |
| Attention check failure rate | Proportion failing | Within 3pp of real rate |
| Response time distribution (if applicable) | Median, IQR | Within 20% of real |

### When No Real Data Exists

If no real human data is available for calibration:
- Search the literature for published experiments with similar designs (this is where the Step 0c Research Scan is critical)
- Use published norms for standard scales (e.g., Big Five means and SDs are well-documented across many populations)
- Use general benchmarks: Likert SD typically 1.2–1.8 on 7-point scales, Cronbach's alpha typically 0.7–0.9 for validated scales, attention check failure rates of 3–8%
- Flag the simulation as "uncalibrated" in the archive and prioritize collecting real data for future calibration

## Quality Tracking Across Runs

Maintain a running quality log that tracks how the simulator performs over time.

### Quality Score Computation

Compute a composite quality score after each run:

```python
def compute_quality_score(simulated_data, config, real_data=None):
    scores = {}
    
    # 1. Distribution realism (0-1)
    scores['distribution'] = assess_distribution_realism(simulated_data, real_data)
    
    # 2. Intra-subject consistency (0-1)
    scores['consistency'] = assess_intra_subject_consistency(simulated_data, config)
    
    # 3. Open-text quality (0-1)
    scores['open_text'] = assess_open_text_quality(simulated_data, config)
    
    # 4. Human-likeness (0-1) — AI detection heuristics inverted
    scores['human_likeness'] = assess_human_likeness(simulated_data)
    
    # 5. Schema compliance (0-1)
    scores['schema'] = assess_schema_compliance(simulated_data, config)
    
    # Weighted composite
    weights = {
        'distribution': 0.25,
        'consistency': 0.30,  # highest weight — this is the hardest to get right
        'open_text': 0.20,
        'human_likeness': 0.15,
        'schema': 0.10
    }
    composite = sum(scores[k] * weights[k] for k in scores)
    
    return {'composite': composite, 'components': scores}
```

### Regression Detection

After each run, compare quality scores against the running average:
- If composite score drops > 10% from the 5-run moving average → **ALERT**: quality regression detected
- Identify which component(s) regressed
- Trace back to what changed (prompt modification, parameter change, new template)
- Revert the change if it caused the regression

### Performance Tracking Dashboard Data

Store time-series data for each component score across runs. This enables:
- Visualization of quality trends over time
- Identification of which changes improved vs degraded quality
- Evidence-based decisions about prompt engineering and template evolution

## Prompt Evolution

The prompts used to generate responses are themselves improvable artifacts.

### Prompt Versioning

- Every distinct prompt configuration gets a version identifier
- When modifying prompts, create a new version (never overwrite)
- Track which prompt version was used in each run
- Compare quality scores across prompt versions to identify improvements

### Prompt Improvement Triggers

Improve prompts when:
1. A specific item type consistently fails quality checks across multiple runs
2. Real human data becomes available and reveals a systematic bias in simulated output
3. A new detection method is published that the current output would fail
4. The template fallback is being triggered too frequently (>20% of items for a given type)

### Prompt Improvement Protocol

1. Identify the weakest quality component from the most recent runs
2. Search for recent literature or best practices relevant to that weakness
3. Draft an improved prompt version
4. Run a small comparison (N=50): old prompt vs new prompt on the same config
5. Compare quality scores — adopt the new version only if it improves the target metric WITHOUT degrading other metrics
6. Archive the comparison results for future reference

## Feedback Integration

### User Feedback Loop

When the user identifies quality issues in simulation output:
1. Log the specific issue (what was wrong, which item/subject)
2. Classify the issue (distribution | consistency | open-text | realism | schema)
3. Trace to root cause (prompt | template | parameter | architecture)
4. Implement fix
5. Add a regression test that would catch this issue in future runs
6. Update quality checks to detect this class of issue automatically

### Automated Feedback

The quality metrics system generates automated feedback:
- Per-item quality flags → inform which items need better prompts or templates
- Per-subject quality flags → inform which persona configurations produce poor output
- Aggregate trend data → inform whether recent changes are improvements or regressions

## Bootstrapping (First Runs)

When the simulator has no accumulated data:

1. **Use published norms as initial calibration** — Big Five scales, standard attitude measures, etc. have well-documented distributions.
2. **Start with conservative persona parameters** — moderate trait values, moderate noise, moderate correlations. Undershoot rather than overshoot on variance.
3. **Run initial simulations at small N** (N=50) and manually audit before scaling up.
4. **Actively seek real data for calibration** — use Step 0c Research Scan to find published datasets.
5. **After 5-10 runs**, the accumulated archive provides enough data to begin self-calibration.

## Degradation Prevention

Over time, accumulated fixes and optimizations can interact in unexpected ways. Prevent degradation by:

1. **Never remove quality checks** — only add new ones. Checks are monotonically increasing.
2. **Run the full quality battery on every run** — not just the checks relevant to the latest change.
3. **Maintain a "golden set"** — a small set of simulation configs (3-5) that are re-run periodically as regression tests. If quality drops on these, something broke.
4. **Version everything** — prompts, templates, parameters, covariance matrices. Any change can be reverted.

---

## Self-Audit Bug-Class Catalog (learn from every external finding)

**Goal: catch these classes myself before any commit, so external review (Codex,
users) finds nothing.** Every time an external reviewer or a later iteration finds
a bug I missed, add its *class* here and add a matching automated check. This list
is append-only and is the single most important regression-prevention asset.

### Pre-commit self-audit checklist (run mentally + with grep/tests on EVERY change)

1. **Reproducibility (salted hash).** Bare `hash()` is salted per process
   (`PYTHONHASHSEED`), so any seeded path using it is non-reproducible across
   restarts. → `grep -nE "[^_a-zA-Z]hash\(" <changed files>` ; replace with a
   SHA-256 / ordinal stable hash in ANY seeding/selection context.
   *Found by: Codex audit 3.4 (and a follow-up: 3 more sites I missed the first time).*

2. **Reproducibility (global RNG).** `np.random.seed()` / `random.seed()` mutate
   GLOBAL state (cross-session pollution in multi-user Streamlit). But REMOVING
   them breaks any downstream code that uses bare `random.*`/`np.random.*` and
   relied on that seed. → Before removing a global seed, grep the WHOLE call tree
   reached by `generate()` for bare `random.`/`np.random.` (not just the file you
   edited): text generators, validators, repair/audit passes. If any exist, use
   the **seed-then-save/restore** pattern (seed globals at the top of the run,
   restore prior state in a `finally`/before return) instead of deleting the seed.
   *Found by: Codex audit 3.5 + the P1 follow-up — my first fix only checked the
   numeric path and missed the OE/fallback/validator paths.*

3. **Verification scope.** When claiming "X is unaffected/identical", the test must
   exercise EVERY code path, not the convenient one. A matrix-only determinism test
   does NOT prove the open-ended/fallback/repair paths are deterministic. → Make
   the verifying test hit the broadest realistic config (multi-item scale + numeric
   DV + open-ended question + conditions), and diff a HASH of the full output, not
   one column.
   *Found by: Codex P1 — my "output identical" claim was scoped too narrowly.*

4. **Normalization drops valid input.** Input-normalizing functions that read only
   one key (`name`) silently drop valid specs keyed differently (`variable_name`,
   `export_tag`, `question_id`). And `int()` on a list-valued field silently
   defaults instead of using the list. → For every normalizer, test the dict/list
   variants of each field, assert nothing is dropped, assert list→count+names.
   *Found by: Codex audit 3.1, 3.2.*

5. **pandas dtype mutation.** Writing a float into an int column
   (`df.iat[...] = float`) raises a FutureWarning now and will hard-fail later.
   → Widen the column dtype first; test under `python -W error::FutureWarning`.
   *Found by: Codex audit 3.3.*

6. **Provenance overwrite.** Post-processing that overwrites a column the base
   layer set (e.g. `_Generation_Source`) erases row-level provenance. → Add a
   `_Base_*` column or a separate `_Method` column; never overwrite.
   *Found by: Codex audit 3.6.*

7. **Compile the WHOLE tree.** A syntax/indentation error in a sub-package (even
   experimental) ships in the release. → `python -m compileall -q simulation_app`
   must exit 0 as a pre-commit gate. Add an import/compile test for CLI entry points.
   *Found by: Codex audit 1.1.*

8. **Secret/PII exposure.** Never echo token/key prefixes, never persist raw keys
   to JSON, never surface raw SMTP/host errors to users. → grep changed files for
   `token[:`, `api_key'] =`, `st.error(.*exc`, key/print/log of secrets.
   *Found by: Codex audit 2.2, 2.5, 9.1.*

9. **Scratch-file hygiene.** Never `git add -A` after analysis — agents/probes
   leave `_*.py`/`.pyc` scratch. → `git status` review; `.gitignore` covers
   `/_*` and all bytecode; stage explicit paths.
   *Found by: my own repeated mistake (committed 59 scratch + 32 .pyc files).*

10. **Silent-corruption from downstream passes.** A correct transform can be
    re-broken by a LATER pass that's unaware of its invariant (e.g. the
    consistency-audit re-correlating constant-sum columns). → After adding any
    joint/structured output, grep every post-processing pass that mutates those
    columns and exempt them; test the invariant on the FINAL dataframe, at the
    item counts where the later pass actually fires (k≥4, not just k=3).
    *Found by: an earlier adversarial review (constant-sum 2–7% invalid at k≥4).*

### The self-audit work loop (run before every "done")

```
FOR each change set, before committing:
  1. grep the changed files for each bug-class signature above.
  2. For RNG/seed/hash changes: run a CROSS-PROCESS determinism test
     (two subprocesses, different PYTHONHASHSEED) hashing the FULL output,
     AND a global-state-restored test.
  3. Run the full battery: compileall (exit 0), pytest regressions,
     effect fuzz, qsf_robustness (0 crashes / all example QSFs),
     random_qsf_n200 (N=200), e2e sim.
  4. Spawn an independent adversarial review agent on the diff; treat any
     reproduced finding as real; verify-then-fix; re-run the battery.
  5. If the agent (or a user) finds a class NOT in this catalog, ADD IT here
     and add an automated check, so it can never be missed again.
```

The catalog grows monotonically. The target end-state: the adversarial agent and
external reviewers find **nothing**, because every class they would find is already
gated by a grep signature + an automated test in step 2–3.

11. **Question text bleeding into open-ended responses.** When subject/topic
    extraction falls back to the raw `question_text`, the literal (often
    interrogative) question gets interpolated into a template ("...I see both
    sides of Why did you rate your trust in the advisor that way?"). An
    interrogative phrase is NOT a usable topic. → Topic extractors must reject
    candidates that start with an interrogative (why/what/how/when/where/which/
    who/do/did/is/are/...) or contain `?`, and never use `question_text[:N]`
    verbatim as the topic — distill clean keywords instead. Test: generate OE for
    several interrogative questions and assert NO response contains the literal
    question phrasing.
    *Found by: my own N=100 free-LLM smoke test (the offline template fallback
    leaked the question into 54/100 responses) — exactly the kind of realism bug
    the self-audit should catch before shipping.*

12. **Thread-unsafe global state under concurrent Streamlit sessions.** Two
    sessions calling generate() at once race on (a) the process-global RNG
    temporary-seeding region and (b) any CLASS-LEVEL or MODULE-LEVEL mutable state
    (sets/dicts) in the generation path. One session consumes another's RNG
    stream, or sees another's accumulated fingerprints, breaking same-seed
    reproducibility. → (1) Serialize any global-RNG seed/restore region with a
    process lock. (2) NEVER keep run-scoped mutable state at class/module level —
    use per-instance attributes set in __init__. Grep signatures:
    `grep -nE '^    _[a-z_]+ *(:.*)?= *(set\(\)|\{\}|\[\])'` for class-level mutable
    decls; `grep -nE '[A-Z][A-Za-z]+\._[a-z_]+\.(add|append|update|clear)\('` for
    cross-instance mutation. Test: run N concurrent same-seed generations in
    threads (+ perturbing different-seed threads) and assert ONE identical output.
    *Found by: Codex PR review P1 — my seed-restore fix was exception-safe but not
    thread-safe, and ComprehensiveResponseGenerator kept _used_responses at class
    level. I should have caught both: "multi-user Streamlit" is in the threat model.*

13. **Two code paths that MUST agree, maintained separately, drift.** A cache key
    (or any value that two paths must compute identically) is built by two
    copy-pasted blocks; one path later gains an extra suffix the other doesn't.
    The keys silently stop matching → the cache always misses, defeating an
    optimization with NO error. Here: the LLM response-pool key is
    `md5(question_text[:200]|condition|sentiment)`; the prefill path built
    `question_text` WITHOUT the `\nCondition:`/`\nAdditional context:` suffixes the
    per-participant path appended, so every prefilled draw missed (prefill budget
    silently wasted — anti-pattern #29). → ANY value two paths must agree on
    (cache keys, hashes, IDs, signatures) MUST come from ONE shared helper, never
    two parallel blocks. Grep for the same f-string skeleton in 2+ places;
    extract a helper. Test: assert the two paths produce byte-identical keys AND
    that a value stored by path A is retrievable by path B. This class hides in
    NON-default paths (only fires when `question_context` is set), so the test must
    exercise the enriched path, not the bare one.
    *Found by: deep adversarial audit H1 — highest-value find because it produced
    no error and only triggered on the (encouraged) question-context path.*

14. **A "recoverable" latch with no path to reset.** A flag documented as
    "resets on success" only resets inside a function that, once the flag is set,
    every caller SKIPS. So the flag is effectively permanent for the rest of the
    run, contradicting its docstring. Here: `free_tier_exhausted_now` reads
    `_consecutive_transient_batches`, which only reset inside `_generate_batch` —
    but once exhausted, prefill fast-bails and the OE loop sets `_llm_throttled_now`,
    so `_generate_batch` is never called again → the free tier is abandoned for the
    whole run even if it recovers seconds later. → For every "recoverable"/"resets
    on X" flag, trace whether X is still REACHABLE once the flag is set. If the
    flag gates the only path to X, add an explicit reset at a genuine retry
    boundary (here: `reset_providers()` and per-OE-question). Test: set the latch,
    call the documented recovery boundary, assert the latch cleared.
    *Found by: deep adversarial audit M1 — the docstring asserted an invariant the
    control flow couldn't deliver.*

15. **Incomplete "fresh run" reset on a reused stateful object.** The engine reset
    ONE OE generator's dedup state (`text_generator`) but not the PRIMARY one
    (`comprehensive_generator`), so a REUSED engine produced different OE text on
    the same seed — a reproducibility-contract break that's latent only because the
    UI happens to rebuild the engine each run (SDK/batch/"regenerate" reuse would
    hit it). → When you reset run-scoped state "for a fresh dataset", enumerate
    EVERY object that accumulates run-scoped state (all OE generators, pools,
    recent-start buffers) and reset all of them — and give each a single `reset()`
    method so the list can't go stale. Test: call generate() TWICE on the same
    engine + once on a fresh engine, assert all three OE hashes equal.
    *Found by: deep adversarial audit M2 — I reset the fallback generator but
    overlooked the primary one right next to it.*

16. **Unbounded run-scoped accumulator at max N.** Dedup `set`s grew one
    fingerprint + N sentences per response with no cap; at `MAX_SIMULATED_N=10,000`
    × multiple OE questions that's hundreds of thousands of strings. → Bound any
    per-run accumulator that scales with N. CRITICAL: evict DETERMINISTICALLY
    (FIFO via a companion `deque`, oldest-first) — never `set.pop()` an arbitrary
    element, because str hash order depends on `PYTHONHASHSEED` and would
    reintroduce cross-process non-determinism (bug-class #1/#2). Test: the bounded
    structure stays ≤ cap at large N AND same-seed output is byte-identical across
    `PYTHONHASHSEED` values with eviction active.
    *Found by: deep adversarial audit L1.*

17. **Mechanical artifacts from text-mutation stages, and instruction text in the
    extracted topic (offline OE realism).** Two classes my OWN N=200 offline scan
    caught (the kind a reviewer/user would flag as "this data looks fake"):
    (a) the verbal-tic / filler / hedge inserters append punctuation next to
    existing punctuation, producing `",,"` (~14% of responses) — a mechanical glitch,
    NOT a realistic human typo. (b) topic extraction from `question_context` grabbed
    the instruction, not the subject: "Explain your view of the candidate **and
    why**" → topic "the candidate and why" → "I feel good about the candidate and
    why is straightforward" (68% of responses); and leading imperatives with no
    preposition ("Describe the tax plan") kept the verb. → (a) Add a DETERMINISTIC
    final-pass punctuation normalizer (pure function, no RNG, preserves `...`/`!!`/
    typos) at the single OE return point. (b) Strip instruction TAILS ("and why/how/
    explain...") and leading imperative PREFIXES ("describe/explain/list/rate/in your
    own words...") from the extracted topic — but conservatively, so real conjunctive
    topics ("crime and punishment", "crime ... in modern society") and embedded
    prepositions ("opini-on") are preserved. Test: scan offline OE at N≥120 for `,,`,
    space-before-punct, and instruction-tail leakage (assert 0); unit-test the topic
    extractor on tail/prefix/conjunctive/embedded-prep cases; confirm same-seed output
    stays byte-identical across PYTHONHASHSEED (the fixes are deterministic).
    *Found by: my own offline OE quality scan (",," at 14%, "and why" leak at 68%) —
    the self-audit loop catching a realism bug class before any external reviewer.*

18. **Default-bounded post-processing violates NARROW scales; uniform structure is a
    realism tell.** Two numeric-side classes from one self-audit pass:
    (a) **Bounds bug** — a post-generation perturbation (anti-straight-line jitter)
    inferred the scale ceiling with a hard floor (`scale_hi = max(observed, 5)`) and,
    in the engine's copy, from a LEAKED loop variable (the last scale). A 2-point
    item became `2+1=3`. → Per-item post-processing MUST clamp to EACH column's own
    range (a per-column observed-min/max map), never a default (5/7) or another
    scale's bounds. Test narrow scales explicitly (2- and 3-point) for
    `min>=1 and max<=scale_points` AND `unique ⊆ {1..points}` — a 7-point-only test
    would never have caught it. (b) **Uniform structure** — mixing every scale item
    toward the common factor with the SAME weight makes all inter-item correlations
    ≈ identical, a structural "looks generated" tell (Xie et al. 2026). → Vary
    per-item loadings (scale-stable, applied column-wise so they don't average out;
    centered on the uniform value so the MEAN correlation / Cronbach's α and the
    condition effect are preserved). Test: inter-item correlation SPREAD exceeds a
    threshold while the MEAN stays realistic and the effect direction survives; and
    same-seed output stays byte-identical across PYTHONHASHSEED.
    *Found by: my own statistical-realism probe (validated against Xie et al. 2026)
    + the scale-generation suite's 2-point binary test — exactly the kind of bug a
    7-point-only happy path hides.*

19. **A single top-level import crashes the WHOLE app (P0, production-down).**
    `app.py` did `from utils.group_management import GroupManager, APIKeyManager,
    _atomic_write_json`. The instant `app.py` and `group_management` drift out of
    sync (partial deploy, stale build, version skew, a rename), that ONE line
    raises `ImportError` and Streamlit shows a redacted broken-red page — the
    entire tool is DOWN, and the user can't even see why (Streamlit redacts the
    message). Importing a `_private` cross-module symbol made it worse: underscore
    names are internal and can vanish without notice. → (1) NEVER hard-import a
    `_private` symbol from another module at top level — wrap in
    `try/except ImportError` with a local fallback so the app still loads. (2) Ship
    an app-load SMOKE TEST that execs `app.py` exactly as Streamlit does and fails
    on ANY unresolved top-level import (`tests/test_app_import_safety.py`) — run it
    before every push; it's release-blocking. (3) A symbol an entry point imports
    is part of the contract: rename/move/delete it and ALL import sites in the SAME
    commit, never split across commits/branches. Grep signature for the offender:
    `grep -nE "^from [\w.]+ import .*\b_[a-z]" simulation_app/app.py`. Tests:
    `test_app_imports_without_error`, `test_app_survives_missing_private_atomic_helper`,
    `test_app_has_no_toplevel_private_cross_module_imports`.
    *Found by: the USER, on the live site (broken-red ImportError page) — the worst
    way to find out. This is now a CLAUDE.md ABSOLUTE RULE ("Import Resilience").
    Production-down is the highest-severity class; the app-load smoke test is the
    gate that must catch it before deploy.*

#### RESOLVED (v1.2.8.4): global-RNG lock vs. multi-user throughput
The `_GLOBAL_RNG_LOCK` used to be held across the WHOLE generation body (incl. LLM
network I/O), serializing concurrent users. Codex re-flagged it (PR #352 P2), so it
was **removed entirely** rather than left as a trade-off. The feared "~70 sites"
turned out to be a mis-estimate: the engine's numeric draws ALREADY use per-call
`np.random.RandomState` (23 sites, seeded from `self.seed`/`participant_seed`), and
`text_generator`'s 58 bare-`random.*` calls are in `OpenEndedTextGenerator`, which
the engine does NOT use (it uses `persona_library.TextResponseGenerator`, 0 global).
The ONLY global-RNG consumers in the generation path were `HBSValidator` (7 sites)
and the LLM backoff jitter (1) — both migrated to per-instance RNGs
(`HBSValidator(seed=...)`, `LLMResponseGenerator._rng`). Then global seeding + the
lock were deleted. Verified: same-seed output byte-identical across
`PYTHONHASHSEED ∈ {0,1,42,31337,271828}` with global seeding REMOVED (complete proof
no global RNG affects data) + `test_v1284_generation_is_not_globally_serialized`.
> **Lesson:** before deferring a fix as a "big refactor", MEASURE the actual call
> sites in the real path — a comprehensive grep (`grep -rE "random\.<draw>|np\.random\.<draw>"`
> across every module, then checking which are reachable from `generate()`) showed
> the scope was 8 sites, not 70. The "conscious trade-off" was real but the cost
> estimate that justified deferring it was wrong; an external reviewer was right to
> push. Don't let a guessed-high estimate excuse leaving a flagged issue unfixed.
