# Calibration status

What this tool's numbers are, what has been checked, and what has not. Written for a reader who
has to decide whether to trust a simulated dataset, and kept short enough to stay true.

Last updated: 2026-10-24 (v1.3.0.9), in response to the Stage 1 audit of 2026-10-23.

## The one-sentence version

**The tool recovers an analyst's assumption; it does not predict behaviour.** When you give it a
Cohen's *d*, it builds data that shows about that *d*. When you give it nothing, it does not know
what your conditions do, and as of v1.3.0.9 it says so instead of guessing.

## What changed in v1.3.0.8 and v1.3.0.9

| Audit item | Was | Is |
|---|---|---|
| P0-1 default inference | A condition named "negative" lowered **every** outcome, whatever higher values of that outcome meant. On the audited design the focal contrast came out at g = −0.19 (SD 0.17) against a human +0.46, and the sign agreed in 12% of 100 seeds. | **Declare-first.** A wording-derived effect is applied only once the analyst declares what higher values mean (per-DV radio on the Design page, or `dv_polarity` on the engine). Undeclared is a true null. The old behaviour is still available as an explicit opt-in labelled a heuristic guess. **Measured on the audited design, 40 seeds from 1000, N = 600:** the declare-first default gives mean g = −0.010 (SD 0.093), sign positive in **48%** of seeds, max |g| 0.250 — the true-null behaviour the audit asked for; the opt-in path reproduces the old confidently wrong sign (mean g = −0.292, positive in 0%). |
| P0-2 provenance | Every non-user effect exported as `"source": "inferred"`, whether it came from a recalled meta-analytic row, a valence keyword, or nothing but the label hash. `honesty_notice()` had no caller outside the tests. | `metadata["calibration_provenance"]` carries one row per outcome: `effect_source`, `marginal_source`, `evidence_tier`, `applied_d`, `polarity`, `polarity_source`, `marginal_reference` and a plain-language `status`. The same table appears on the results page and in the instructor report, with the registry's verification notice. |
| P0-3 game detection | Study-wide: a Likert `Brand_Satisfaction` in a study mentioning a "charity donation appeal" was classified as a dictator game, moving its item-1 mean from 5.60 to 2.81. | A DV qualifies only when **its own** name, question wording or item labels name the game, and only when its scale geometry fits. Detection evidence is recorded in metadata. The experimental game model is opt-in and off by default. |
| P1-4 numeric DVs | A DV declared 60–80 returned 60 for all 400 participants. | The declared `scale_max` is read instead of `scale_points`. Measured: (60, 80) → mean 68.0, SD 6.1, 20 distinct values; (−10, 10), (0, 1000), (1, 7), (0, 100) all non-constant with the mass inside the range. |
| P1-5 binary marginals | The base rate sat near 0.62 whatever the study (human 0.11 on the audited design: 1.63 human SDs out). | The analyst declares it. Measured recovery: 0.11 → 0.095, 0.50 → 0.495, 0.80 → 0.816. Undeclared is an explicit, flagged, uninformed 0.50 and a generation warning. |
| P1-6 parser | A 2 × 2 run as two sibling randomizers was reported as one factor with four levels; binary single-choice outcomes were not detected at all; a survey with no recognisable outcome got a fabricated 5-item 7-point `Main_DV` and a page announcing "1 DV(s) detected". | Crossed factors are read off the flow and **offered** for confirmation (never switched silently). Binary single-choice outcomes are detected and coded 0/1. "No outcome found" is stated prominently. Corpus (303 example surveys, 0 crashes before and after): surveys with no detected outcome 48 → 22; binary DVs 0 → 94; `single_item` DVs 299 → 483; 17 surveys expose a structural crossing, 13 of them 2 × 2. |
| P1-7 tier damping | Every tier was damped by the same 0.45; the anchor path looked the tier up with `key=""`; the 19 rule-only paradigms were discounted twice. | The anchor key is passed, the shrinkage is applied once (a rule-only match now carries `shrinkage=1.0` and its published magnitude times one policy factor), and `tier_cap_binds()` reports whether the tier moves the magnitude at all. Under the installed 0.45 shrinkage **no tier does**; the tier scales the floor. The README says this rather than the old claim. |
| P1-8 routing | `Willingness_to_Share`, `Payment_Satisfaction` and `Offer_Attractiveness` were read as economic allocations; `Behavior_Score` matched the social-comparison norm through the fragment "sco"; `Sleep_Quality` got the PSQI clinical norm; 29 of 103 keyword targets named norms that are not in the table. | A content match needs two shared tokens, or one that is distinctive and not a generic word (`behavior`, `score`, `quality`, `attitude`, …). A deficit-scored norm (PSQI) is used only once the DV's direction is declared. 20 broken targets were repointed or removed. All six audit-named variables now fall through to a flagged generic default; `PSS4_1`, `bfi_extra_1`, `stress_total` and `panas_pos_4` still reach their norms. |

## What is not calibrated, stated plainly

* **No knowledge-base row has been checked against its primary source.** The registry holds 503
  rows. `coverage_summary()['sourced_entries']` returns 1, and that 1 is the single `corrected`
  row, whose correction was an internal-consistency finding. 502 rows carry a recall audit only.
  The container cannot reach a journal or a repository (`docs/EMPIRICAL_PROVENANCE.md` lists the
  refused hosts).
* **The hand-set literals are unsourced.** The 43 STEP 2 effect-detection domains, STEP 0's
  relational values (+0.30, −0.35 to −0.40) and the 373 `EMPIRICAL_CORRELATIONS` construct pairs
  carry no citation, tier or registry entry. Moving them into `empirical_registry` with tiers is
  open work; until then no document may call them literature-grounded.
* **Published-but-unused knowledge.** `CULTURAL_ADJUSTMENTS` / `get_cultural_adjustment`,
  `get_meta_analytic_effect`, `empirical_marginals.marginal_for`, `audit_table` and
  `tier_cap_binds` are reachable for a reader or a script and are **not** consulted while data is
  generated. The inventory is pinned by `tests/test_unwired_knowledge_v1309.py`, so it cannot
  drift from the documents silently.
* **No recursive strategic reasoning runs.** For a beauty contest the guess is the closed form
  `p^k × max/2` plus noise. `StrategyRouter`, `LevelKStrategy` and `CognitiveHierarchyStrategy`
  exist in the experimental tree and nothing reaches them.
* **Offline open-ended text has never been tested against human text.** It is template based and
  audited for topicality and coherence, not for indistinguishability.
* **One benchmark study, not three.** The acceptance numbers above come from the audited design
  (Brader, Valentino & Suhay 2008) and from synthetic fixtures. A second and third benchmark, a
  prospective registered forecast, a comparison with language-model respondents, and a
  corpus-wide misrouting frequency are all still missing. Do not claim them.

## Restored game effects: the 1.13x was seed noise, not bias (v1.3.1.1)

v1.3.1.0 recorded that the step restoring a requested effect after the game model runs landed at
1.13x the request (d = 0.565 for 0.50, seeds 1-12, N = 600) and blamed a pre-clipping SD. Tracing the
restoration showed that explanation was wrong: the per-arm moves hit their targets exactly, and the
12-seed excess came from the seeded chance draw (+0.047 on the target gap, 2.3 standard errors) plus a
small arm gap already present in the game output (+0.017). Measured on 60 further seeds (13-72) the
realised d is **0.484 for a request of 0.50 (0.97x, SE 0.013, SD 0.097)**; over all 72 seeds it is
0.498. No engine change was needed. The test band stays at 0.9-1.15x because seeds 1-12 are fixed and
sit at 1.13x by chance; a longer sample is the honest check, not a tighter band on a short one.

## How to check any of this yourself

```bash
python3 -m pytest tests/test_declare_first_v1308.py tests/test_p1_fixes_v1309.py \
                  tests/test_unwired_knowledge_v1309.py -q     # the audit's regressions
python3 scripts/check_version_sync.py                          # all 11 version locations
python3 -m pytest tests -m "not slow" -q                       # the fast suite
```

Every run writes its own provenance: read `calibration_provenance` in the run's archived metadata
or the instructor report, or the "Where each outcome's numbers came from" table on the results page.
It is not in the user's download (see `utils/user_package.py`).
