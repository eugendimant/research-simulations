# Coverage Roadmap — Simulating *any* reasonable human-behavior experiment

This document is the synthesized output of a 4-agent parallel coverage audit
(designs/DV-formats, topical domains, populations/individual-differences,
paradigms/output) plus adversarial critique. It answers: **does the current
design cover everything reasonable for anyone who wants to simulate real human
behavior, and where are the gaps?**

The tool is genuinely mature: **38 effect-detection domains**, a scientific
knowledge base of **~255 calibration entries** (187 meta-analytic effects, 68
game calibrations, construct norms spanning clinical/personality/affect/
well-being scales), **161 personas**, census-weighted demographics, ex-Gaussian
response-time realism, MCAR/MAR missingness + survival-skewed dropout, inter-item
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
  `generate()` into three named helpers; classification regexes compile once. Verified across the 291-QSF corpus, the DV types that
actually occur are: `matrix(1808), single_item(921), numbered_items(154),
numeric_input(52), slider(29), constant_sum(8), rank_order(5), likert(4),
numbered(1)`. (Types like `single_choice`/`best_worst`/`paired_comparison`/
`hot_spot` have parser code paths but **0 occurrences** in real QSFs — non-issues.)

| Area | Was | Now |
|------|-----|-----|
| **Constant-sum DVs** | Items generated independently — **0%** of rows summed to the total | Renormalized to sum **exactly** to the total (largest-remainder); 100% valid k=2..10 |
| **Rank-order DVs** | Independent integers — **0%** valid permutations (duplicate ranks) | Valid **1..k permutations** via latent-utility argsort (Plackett-Luce flavor) |
| **Numeric money/WTP DVs** | ~symmetric around the midpoint (skew≈0, no floor) | **Right-skewed** log-normal (skew≈+1.0), ~12% floor spike at $0, treatment effect preserved |
| **Numeric count/frequency DVs** | ~symmetric | **Right-skewed** (mode low, long tail) |
| **Joint-DV downstream safety** | consistency-audit + bounds-clip silently re-broke constant-sum 2–7% of the time | joint-constrained DVs exempted from alpha-repair, anti-straight-line jitter, and bounds-clipping |
| **Topical breadth** | 38 effect domains | **+5 domains**: emotion, misinformation/illusory-truth, aggression, negotiation, charitable giving — grounded, contested effects kept small, bounded by ±0.50 cap |
| **Dark Triad DVs** | no construct calibration | dormant SD3 norms wired into `_construct_map` |

The seam is now **correct for every DV type that occurs in real QSFs**. All
changes are **additive and gated on `type` + name cues**, so generic numeric
(age/temperature) and all Likert/matrix/slider DVs are **byte-identical**.
Validated: 15 regression tests, effect fuzz (2,592 combos), 0 crashes across 291
QSFs, 0 issues across 10 student QSFs, e2e all-pass.

---

## 🗺️ Prioritized roadmap (not yet implemented)

Ordered by (frequency of need × value ÷ risk). These are larger, mostly
**structural** additions that need their own design + validation passes.

### Tier A — High value, medium risk (next)
1. **Within-subjects done properly** — `design_type="within"/"mixed"` is currently
   structurally simulated as between-subjects. Needs repeated DV columns
   (`DV_T1/T2…`) from a shared per-participant latent + level shift, giving
   realistic test-retest r≈0.5–0.7. *(Highest-value remaining design gap.)*
2. **Slider continuous realism** — sliders (29 in corpus) generate as bounded
   integers; feeling-thermometers/VAS could use finer granularity + endpoint
   heaping. Low risk, modest value.
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
13. **Sample-source profiles** (MTurk / Prolific / undergrad / nat-rep) — careless
    base-rate, attention-pass, demographic skew, effect-size attenuation. Meta-DB
    `sample` moderators exist, unused.
14. **Big Five / HEXACO with realistic inter-correlations + trait→DV links** —
    trait names exist but are drawn independently and don't drive responses (van
    der Linden et al. 2010 GFP matrix; Soto et al. 2011 norms).
15. **Numeracy latent → numeric scale behavior** (round-number heaping, extremes).
16. **Special-population age profiles** (children/adolescents/older adults:
    reading speed, comprehension, scale-use) — age floor currently 18.
17. **Length-/demographic-conditioned attrition** (Galesic & Bosnjak 2009).
18. **Fraud subpopulation** (bots/duplicates/speeders) sized by sample source.

### Known characteristics (pre-existing, not regressions; noted for transparency)
- **Construct norms apply a small ±0.15 calibration nudge, not a mean anchor** —
  a depression/anxiety/Machiavellianism DV with no manipulation centers near the
  scale's default, not at the published norm mean. Anchoring generated means to
  published construct norms is a worthwhile, separate, deeper change.
- The **`HBSParticipantFactory` census demographics** are appended as `ABE3_*`
  columns indexed by `i % n_states` (position-misaligned with the persona that
  generated each row) and do not drive DVs — worth aligning + activating.

---

## How coverage was assessed (method)
Four independent agents mapped current vs. reasonable coverage on orthogonal
axes, each producing a frequency/risk-rated gap list with concrete, literature-
grounded implementation sketches; an adversarial critic then verified the
highest-value claims (catching, e.g., that several "unwired" constructs were
actually wired, and that the construct-norm system nudges rather than anchors).
Every claimed gap above was code-verified before inclusion.
