# Coverage Roadmap — Simulating *any* reasonable human-behavior experiment

This document is the synthesized output of a 4-agent parallel coverage audit
(designs/DV-formats, topical domains, populations/individual-differences,
paradigms/output) plus adversarial critique. It answers: **does the current
design cover everything reasonable for anyone who wants to simulate real human
behavior, and where are the gaps?**

The tool is genuinely mature: **43 effect-detection domains**, a scientific
knowledge base of **468 calibration entries** (187 meta-analytic effects, 68
game calibrations, 201 construct norms spanning clinical/personality/affect/
well-being scales, 12 cultural adjustments), **78 personas** (6 response-style + 72 domain-specific), census-weighted demographics, ex-Gaussian
response-time realism, optional MCAR and trait/position-dependent missingness
+ survival-skewed dropout (off by default), inter-item
α targeting, cross-DV correlation, and acquiescence/extremity/SD response styles.

The audit found the breadth of *reference knowledge* is excellent. The real gaps
are concentrated in two places: **(a) the generation layer not consuming that
knowledge** for several question types, and **(b) structural output modes**
(long/dyadic/per-trial) that the single-row-per-participant engine cannot yet
express.

---

## ✅ Fixed — closing the detection↔generation seam (v1.2.7.0–1.2.7.4)

The root issue: **the parser detects rich DV types, but the generation engine
funneled almost everything through one numeric Likert pathway** (it never
branched on `scale["type"]`). So question types the tool *detects* produced
silently-invalid data.

**v1.2.7.3–1.2.7.4 (from Codex review + a 3-agent hyper-audit) — additional
data-validity fixes:**
- **Numeric skew is classified on DV-specific text only** (the DV's own
  name/question_text/description/anchors), never the study title/description —
  so a money mention at the study level no longer reshapes *unrelated* numeric
  DVs, and a money cue living in a numeric input's own question text is honored
  (preserved end-to-end through QSF→bridge→engine).
- **Cue matching is word-boundary** (not substring): `'times'`∉"sometimes",
  `'cost'`∉"costly", `'invest'`∉"investment attitude" (also guarded by a
  rating/attitude-context exclusion) — neutral rating DVs stay symmetric.
- **Scale-bound derivation fixed** for two corpus defects that produced
  constant/implausible columns: (a) non-1-based Qualtrics choice IDs (14–18,
  40–44) are normalized to 1..N instead of emitting "40" on a 5-point scale;
  (b) fractional (0–0.25) and huge (0–100000) slider ranges no longer collapse
  to a constant — they fill a clean integer grid with realistic spread.
- **Streamlined:** the type-aware post-processing was extracted from the giant
  `generate()` into three named helpers; classification regexes compile once.

### DV-type census

`_detect_scales()` can emit 11 types. Re-derived 2026-10-06 by parsing all 302
example QSFs (0 parse failures) and counting `detected_scales[*]["type"]`, the
types that actually occur are:
`matrix(991), slider(627), single_item(300), numbered_items(239),
constant_sum(32), rank_order(27), numeric_input(18)`.

The remaining four — `likert`, `best_worst`, `paired_comparison`, `hot_spot` —
have parser code paths but **0 occurrences** in the corpus. (`single_choice` and
`numbered` are not types `_detect_scales` emits at all; single-choice items are
grouped into `likert`/`single_item`.)

Regenerate this census rather than editing it — the corpus grows with every
auto-collect commit.

One sharp edge for callers driving the engine directly: the constant-sum total is
read from the scale's `scale_max` (`enhanced_simulation_engine.py:11390`), not
from a `total` field. QSF detection sets `scale_max: 100`, so the parser path is
fine, but a hand-built scale dict that passes `total` and omits `scale_max` falls
back to `k` and produces rows summing to the item count. Rank-order DVs were
re-verified the same day at 120/120 valid permutations.

| Area | Was | Now |
|------|-----|-----|
| **Constant-sum DVs** | Items generated independently — **0%** of rows summed to the total | Renormalized to sum **exactly** to the total (largest-remainder); re-verified 2026-10-06 at 100/100 rows for k=2,3,5,10 |
| **Rank-order DVs** | Independent integers — **0%** valid permutations (duplicate ranks) | Valid **1..k permutations** via latent-utility argsort (Plackett-Luce flavor) |
| **Numeric money/WTP DVs** | ~symmetric around the midpoint (skew≈0, no floor) | **Right-skewed** log-normal (skew ≈ +0.8 measured), 12% floor spike at $0, treatment effect preserved |
| **Numeric count/frequency DVs** | ~symmetric | **Right-skewed** (mode low, long tail) |
| **Joint-DV downstream safety** | consistency-audit + bounds-clip silently re-broke constant-sum 2–7% of the time | joint-constrained DVs exempted from alpha-repair, anti-straight-line jitter, and bounds-clipping |
| **Topical breadth** | 38 effect domains | **+5 domains**: emotion, misinformation/illusory-truth, aggression, negotiation, charitable giving — grounded, contested effects kept small, bounded by ±0.50 cap |
| **Dark Triad DVs** | no construct calibration | dormant SD3 norms wired into `_construct_map` |

The seam is now **correct for every DV type that occurs in real QSFs**. All
changes are **additive and gated on `type` + name cues**, so generic numeric
(age/temperature) and all Likert/matrix/slider DVs are **byte-identical**.
Validated: 38 regression tests in `tests/test_bugfixes_v1264.py`, crash/scoping fuzz
(2,592 context × condition × variable combos), 0 crashes across every example QSF, 0 issues across 10 student QSFs, e2e all-pass.

---

## 🗺️ Prioritized roadmap (not yet implemented)

Ordered by (frequency of need × value ÷ risk). These are larger, mostly
**structural** additions that need their own design + validation passes.

### Tier A — High value, medium risk (next)
1. **Within-subjects done properly** — `design_type="within"/"mixed"` is currently
   structurally simulated as between-subjects. Needs repeated DV columns
   (`DV_T1/T2…`) from a shared per-participant latent + level shift, giving
   realistic test-retest r≈0.5–0.7. *(Highest-value remaining design gap.)*
2. **Slider continuous realism** — sliders generate as bounded integers;
   feeling-thermometers/VAS could use finer granularity + endpoint heaping.
   Low risk. **Re-prioritized 2026-10-06:** this was ranked "modest value" on a
   corpus count of 29; the real count is **627**, the second most common DV type
   after matrix. Value is high, not modest — this belongs above item 1.
3. **WTP anchoring** — extend the new money right-skew to shift toward an explicit
   anchor value when one appears in the question text (Tversky & Kahneman 1974).

> ✅ **Done (v1.2.7.2):** WTP/money + count numeric realism (right-skew + floor
> spike). The other "joint DV types" (best-worst/paired-comparison/2AFC/
> multiple-response) were investigated and **do not occur in real QSFs**, so they
> are deprioritized — the seam is correct for every type that actually appears.

### Tier B — High value, high risk (structural output modes)
4. **Long-format / round-level data** for iterated games (PD/PGG/trust) with
   geometric contribution decay + conditional cooperation (Fehr & Gächter 2000;
   Fischbacher et al. 2001).
5. **Nested / multilevel IDs** (`Session_ID`, `Group_ID`, `Dyad_ID`) with a
   group-level random intercept (ICC ≈ 0.05–0.15).
6. **Per-trial cognitive tasks** (Stroop/flanker/go-no-go/IAT D-score) — define
   the advertised-but-missing `IMPLICIT_MEASURE_PARAMS`; emit congruency×RT trial
   matrices (ex-Gaussian).
7. **Risk & intertemporal titration** (Holt-Laury switch point, hyperbolic
   discounting k, BART) — KB constants already exist, unconsumed.
8. **Panel / wave long format** (T1/T2/T3) with within-person stability.
9. **Dyadic / strategic-pair output** (proposer↔responder, conditional
   acceptance) — unit of analysis is currently silently wrong for ultimatum/
   bargaining/gift-exchange.
10. **Sensitive-topic survey designs** — list experiment, randomized response,
    endorsement experiments.
11. **Conjoint / discrete-choice generation** — already *detected* in QSF, not
    generated; random-utility logit over profiles.

### Tier C — Population realism (activate dormant infrastructure)
12. **Per-participant cross-cultural response styles** — `CULTURAL_RESPONSE_STYLES`
    table + `_apply_cultural_response_style()` exist but are **never called**;
    wire nation→ARS/ERS offsets (Johnson et al. 2005; Harzing 2006).
12b. **Five more implemented-but-uncalled subsystems** (verified 2026-10-06 — each
    is a definition with zero callers repo-wide, so the behavior the docs used to
    claim does not run):
    - `_validate_effect_sizes()` (`enhanced_simulation_engine.py:1552`) — would
      compare achieved to configured *d* at a 0.15 tolerance.
    - `_validate_participant_responses()` (`:9480`) — longstring, IRV and
      endpoint-utilization checks per persona.
    - `_detect_careless_patterns()` (`:1456`) — the only implementation of
      midpoint-overuse detection, so that never reaches an output column.
      (Alternating-pattern detection is *not* lost with it: the live exclusion
      path detects it separately at `:11161` and folds it into
      `Max_Straight_Line` at `:11169`.)
    - `_detect_well_known_scale()` (`qsf_preview.py:1508`) and
      `_detect_reverse_coded_items()` (`:1530`) — so no QSF-parsed scale carries an
      instrument name or a `reverse_items` list (0 of 427 scales across 40 QSFs).
    - `_build_question_dependency_graph()` (`qsf_preview.py:1738`), plus
      `_parse_display_logic()` and `_parse_skip_logic()` — only the
      `has_display_logic` / `has_skip_logic` booleans are populated.
    Each is cheap to wire or to delete; leaving them defined invites the docs to
    drift back into describing them as live.
12c. **Knowledge-base tables with no callers** (verified 2026-10-06). Three of the
    imported tables in `scientific_knowledge_base.py` are never queried during
    generation, so their entries do not affect any output:
    - `META_ANALYTIC_DB` (187 entries) with `get_meta_analytic_effect`, imported
      at `enhanced_simulation_engine.py:273`/`:279`. The 43 STEP 2 domains use
      keyword-to-effect literals written into the engine instead, which is why
      automatic effects are not calibrated to any published magnitude.
    - `CULTURAL_ADJUSTMENTS` (12 entries) with `get_cultural_adjustment`
      (`:276`/`:282`), alongside the `CULTURAL_RESPONSE_STYLES` table in item 12.
    - `ORDER_EFFECTS` with `get_order_effect` (`:278`/`:284`).
    The tables that *are* queried: `GAME_CALIBRATIONS` (`:7109`),
    `CONSTRUCT_NORMS` (`:7298`), `RESPONSE_TIME_NORMS` (`:11260`, `:11312`) and
    `compute_fatigue_adjustment` (`:8777`, `:11290`). Wiring `META_ANALYTIC_DB`
    into the automatic-effect path is the highest-value item in this cluster —
    it is the obvious fix for the uncalibrated-magnitude characteristic noted
    below.
12d. **Ten orphaned open-ended template sets.** `DOMAIN_TEMPLATES` holds 116
    keys but lookup goes through `domain.value` (`response_library.py:8622`), so
    the 10 keys that are not `StudyDomain` values can never be selected:
    `artificial_intelligence`, `climate_change`, `ethical_dilemma`,
    `forgiveness`, `gratitude_experience`, `gratitude_intervention`,
    `moral_cleansing`, `narrative_transportation`, `nostalgia`, `sleep_quality`.
    Several are domains students plausibly study. Either add the matching
    `StudyDomain` members or alias the keys.
13. **Sample-source profiles** (MTurk / Prolific / undergrad / nat-rep) — careless
    base-rate, attention-pass, demographic skew, effect-size attenuation. Meta-DB
    `sample` moderators exist, unused.
14. **Big Five / HEXACO with realistic inter-correlations + trait→DV links** —
    trait names exist but are drawn independently and don't drive responses (van
    der Linden et al. 2010 GFP matrix; Soto et al. 2011 norms).
15. **Numeracy latent → numeric scale behavior** (round-number heaping, extremes).
16. **Special-population age profiles** (children/adolescents/older adults:
    reading speed, comprehension, scale-use) — the minimum age is user-settable (default 18, lower bound 13 in both UIs; `enhanced_simulation_engine.py:10853`) but nothing behavioral keys off it.
17. **Length-/demographic-conditioned attrition** (Galesic & Bosnjak 2009).
18. **Fraud subpopulation** (bots/duplicates/speeders) sized by sample source.

### Known characteristics (pre-existing, not regressions; noted for transparency)
- **Automatically-inferred condition effects are directional, not magnitude-calibrated.**
  When no `cohens_d` is configured for a DV, the effect is derived from condition
  wording via the STEP 0-4 detection pipeline and capped at ±0.50 before the Cohen's-*d*
  conversion — which bounds the shift that actually reaches generation at roughly
  ±0.12 normalized units — but it is not fitted to a target d — a contrast such as positive vs.
  negative feedback can land well above typical literature effects. A condition
  named "Control" also carries a small automatic effect rather than exactly zero.
  Configure an explicit effect size for any DV whose effect magnitude matters.
- **Cross-scale correlations run low.** Correlations between distinct constructs
  (e.g. Trust–Satisfaction r ≈ 0.40 measured, against a 0.52 target) are below what multi-construct survey data
  typically shows.
- **Construct norms apply a small ±0.15 calibration nudge, not a mean anchor** —
  a depression/anxiety/Machiavellianism DV with no manipulation centers near the
  scale's default, not at the published norm mean. Anchoring generated means to
  published construct norms is a worthwhile, separate, deeper change.
- The **`HBSParticipantFactory` census demographics** are appended as seven
  descriptive `ABE3_*` columns (education, income, party ID, ideology, state,
  region, response style) indexed by `i % n_states` — position-misaligned with
  the persona that generated each row — and do not drive DVs. There is no
  `ABE3_Age`: the `Age` column is an independent normal draw from the
  user-supplied mean/SD. Worth aligning + activating.

---

## How coverage was assessed (method)
Four independent agents mapped current vs. reasonable coverage on orthogonal
axes, each producing a frequency/risk-rated gap list with concrete, literature-
grounded implementation sketches; an adversarial critic then verified the
highest-value claims (catching, e.g., that several "unwired" constructs were
actually wired, and that the construct-norm system nudges rather than anchors).
Every claimed gap above was code-verified before inclusion.
