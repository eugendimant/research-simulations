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

## ✅ Fixed in this pass (v1.2.7.0)

| Area | Was | Now |
|------|-----|-----|
| **Constant-sum DVs** | Items generated independently — rows summed to anything (0% summed to 100) | Renormalized to sum **exactly** to the total (Dirichlet-style allocation) |
| **Rank-order DVs** | Independent integers — duplicate ranks (0% valid permutations) | Valid **1..k permutations** via latent-utility argsort (Plackett-Luce flavor) |
| **Topical breadth** | 38 effect domains | **+5 domains**: emotion induction/regulation, misinformation/illusory-truth, aggression/provocation, negotiation, charitable giving — literature-grounded, contested effects kept small, bounded by the ±0.50 cap |
| **Dark Triad DVs** | Mach/psychopathy DVs got no construct calibration | Dormant SD3 norms wired into `_construct_map` (small downward calibration nudge) |

All changes are **additive and gated** — every existing survey/DV produces
byte-identical output (validated: 12 regression tests, effect fuzz of 2,592
condition×variable combos, 0 crashes across 291 real QSFs, 0 issues across 10
student QSFs).

---

## 🗺️ Prioritized roadmap (not yet implemented)

Ordered by (frequency of need × value ÷ risk). These are larger, mostly
**structural** additions that need their own design + validation passes.

### Tier A — High value, medium risk (next)
1. **WTP / numeric DV realism** — right-skewed (log-normal) with a point-mass at
   the floor (~$0) and anchoring; today numeric inputs are bounded-uniform-ish.
   *(Distribution model ready; localized change.)*
2. **More joint DV types** — best-worst/MaxDiff counts, paired-comparison &
   2AFC (logistic latent-difference), multiple-response check-all (correlated
   Bernoullis). Detection already exists for several; only generation is missing.
3. **Within-subjects done properly** — `design_type="within"/"mixed"` is currently
   structurally simulated as between-subjects. Needs repeated DV columns
   (`DV_T1/T2…`) from a shared per-participant latent + level shift, giving
   realistic test-retest r≈0.5–0.7.

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
