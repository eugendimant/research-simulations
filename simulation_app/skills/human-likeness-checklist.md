# Human-Likeness Checklist for Simulation Realism Audits

Use this checklist when reviewing simulation output for realism. Derived from known AI detection methods in survey research (Nature 2025, Manning & Horton 2025, and practitioner reports on fake respondent identification).

## Quick Audit (Run on Every Output)

### 1. Distribution Shape
- [ ] Response means are NOT all clustered at scale midpoints
- [ ] Standard deviations are realistic (typically 1.2–1.8 on a 7-point scale for attitudinal items)
- [ ] Scale extremes (1 and 7) are used at realistic rates (typically 5–20% per extreme per item)
- [ ] Distribution shapes vary by item (some skewed, some roughly normal, some bimodal)

### 2. Within-Subject Coherence
- [ ] Related construct items correlate within subjects (Cronbach's alpha > 0.6 for established scales)
- [ ] Reverse-coded items show expected negative correlations with forward-coded items
- [ ] Open-text explanations match the quantitative response they reference
- [ ] Demographics are internally consistent (no married + single, no 18yo PhD)
- [ ] Behavioral choices across tasks are directionally consistent with stated preferences

### 3. Open-Text Quality
- [ ] Lexical diversity: no more than 10% of responses share distinctive phrases
- [ ] Length distribution is right-skewed (many short, few long)
- [ ] Optional fields have 30–60% completion rate (not 100%)
- [ ] No AI-characteristic phrases ("resilience and determination," "navigate challenges," "fostering")
- [ ] Some responses contain typos, abbreviations, or informal language
- [ ] Emotional tone matches quantitative anchor (low scores → negative text, high scores → positive text)

### 4. Attention and Engagement
- [ ] 2–8% of subjects fail attention checks
- [ ] Response variance decreases slightly in later survey sections (fatigue)
- [ ] Open-text responses get shorter toward end of survey
- [ ] Not all subjects complete every optional item

### 5. Response Patterns
- [ ] Some acquiescence bias present (slight agreement tendency)
- [ ] 2–5% straightlining rate on multi-item scales
- [ ] "No" responses appear at natural rates on screening questions (not 0%)
- [ ] A small number of contradictory responses exist (realistic noise)

## Deep Audit (Run on COMPLEX Changes or New Simulation Modes)

### 6. Between-Subject Patterns
- [ ] Demographic distributions match target population (not perfect quotas — slight imbalances expected)
- [ ] No demographic clustering (blocks of identical demographics)
- [ ] Condition assignment is balanced but not perfectly alternating
- [ ] Between-condition effect sizes are plausible for the domain

### 7. Temporal Patterns (if paradata is generated)
- [ ] Submission times are naturally spaced (not clustered in tight bursts)
- [ ] Completion duration is right-skewed (mean appropriate for survey length)
- [ ] Per-item response times correlate with item complexity
- [ ] Open-text items show longer response times than closed items

### 8. Cross-Validation Against Training Data
Following the Manning & Horton methodology:
- [ ] If training data (real human responses) is available, compare distributions statistically
- [ ] KS tests or similar for continuous measures — p > 0.05 for key items
- [ ] Chi-square tests for categorical items — p > 0.05
- [ ] Effect sizes in simulated data are within confidence interval of training data effects
- [ ] Correlation matrices (simulated vs. real) show similar structure

### 9. Anti-Detection Robustness
- [ ] Run a simple AI detection heuristic on open-text responses (perplexity, n-gram analysis)
- [ ] Verify that response patterns would not trigger paradata-based fraud detection
- [ ] Check that no single participant's profile would be flagged by standard quality filters (speeders, straightliners, contradictions)
- [ ] Verify email domains (if simulated) are not clustered unrealistically

## Known Failure Modes by Simulation Component

| Component | Common Failure | Detection Signal | Fix |
|-----------|---------------|------------------|-----|
| Likert scales | Compressed variance | SD < 1.0 on 7-point scale | Increase persona response noise; use full scale range |
| Open-text | AI phrasing | "Navigate," "foster," "resilience" | Add anti-AI-phrasing constraints; inject colloquialisms |
| Open-text | Matches wrong anchor | Positive text for low score | Enforce text generation conditioned on quantitative response |
| Demographics | Internal contradiction | Married + single | Generate demographics as coherent profile first, then reference throughout |
| Skip logic | Filled when should be NA | Values in skipped sections | Enforce skip logic as hard constraint post-generation |
| Attention checks | 0% failure rate | Perfect attention | Randomly assign 3–8% of subjects to fail attention checks |
| Optional fields | 100% completion | All optional fields filled | Set per-field completion probability (30–60%) |
| Response timing | Identical durations | Low variance in completion time | Sample from right-skewed distribution calibrated to survey length |
| Scale extremes | Avoided | No 1s or 7s | Calibrate persona noise to produce extremes at realistic rates |
| Acquiescence | None or too much | Flat or extreme agreement bias | Add slight positive bias to response generation |
| Straightlining | Never occurs | 0% straightlining | Allow 2–5% of subjects to straightline on at least one scale block |

## ABE 3.0 Advanced Checks (v1.2.5.0+)

When using the Adaptive Behavioral Engine 3.0, these additional checks apply:

| Component | Check | Benchmark |
|-----------|-------|-----------|
| Census demographics | Age, education, income distributions match US census weights | ABE 3.0 ParticipantFactory distributions |
| Party ID | 7-point scale with realistic partisan lean | 30% D, 25% R, 35% I, 10% other (approx.) |
| Stylometric consistency | Same participant's OE responses share voice (vocab, sentence length, filler rate) | ABE 3.0 StylometricEngine fingerprint |
| Error calibration | Typo rate, reading speed vary by education | Frederick 2005, ABE 3.0 benchmarks |
| Completion time | Right-skewed, N(840,120)s clipped [480,1500] | ABE 3.0 ParticipantState defaults |
| Self-validation | Validator passes all 5 checks (timing, OE uniqueness, straightlining, OE length, rating-text coherence) | ABE 3.0 benchmark battery |

---

## Population-Level Statistical Realism (Xie et al. 2026, PNAS — SSDataBench)

A PNAS benchmark of LLM-generated social-science data found that LLMs **compress
real-world heterogeneity into simplified "typological" profiles**. Whether or not
generation is LLM-based, the simulator MUST avoid these three failure modes
(verify them in the self-validation battery and in any new feature):

1. **Univariate distributions collapse toward typical profiles** — the single
   biggest tell. Synthetic values cluster near the modal/typical response and
   under-use the full range. → Preserve heterogeneity: realistic SD, genuine use
   of scale extremes, multimodality where real data is bimodal. Check: a DV whose
   real-world analog has wide spread must NOT come out with compressed variance.
   (This is why the persona variance layer, extremity trait, and the numeric
   right-skew realism exist — and why "data that looks the same across
   participants" is unacceptable.)

2. **Bivariate associations are systematically EXAGGERATED** — LLMs produce
   relationships stronger than reality (correlations/effect sizes too large).
   → Keep condition effects and inter-construct correlations within published
   bounds; never let an effect or a cross-DV correlation exceed plausible meta-
   analytic magnitudes. The ±0.50 effect cap and the construct-independent
   tendency model guard this. When adding effects, prefer conservative sizes and
   flag contested ones.

3. **Life-course / sequential dependencies are under-represented** — sequences of
   life events and their covariate associations are hard to reproduce. → For any
   longitudinal/repeated-measures feature, model realistic within-person
   dependency (test-retest r, autoregressive drift), not independent draws.

Roadmap directions the paper validates (already partially implemented here):
richer per-participant input/conditioning, qualitative context, and
domain-specific calibration against published norms — continue grounding every
calibration in a cited benchmark rather than scaling generic generation.

**Self-audit hook:** when validating output, compute each DV's variance and
extreme-use rate; flag "heterogeneity compression" if variance is implausibly
low or extremes are unused. Treat exaggerated condition effects / inter-item
correlations as a realism regression, not a feature.
