# Realism Benchmark

How we know whether simulated data looks like real data: measure both the same way
and publish the numbers.

## What is being compared

`simulation_app/utils/realism_benchmark.py` computes a fixed profile of
statistics for any wide table of item responses, simulated or real, and compares
the simulated profile to a reference profile computed from **real human
responses**. Fourteen metrics in four families:

| Family | Metrics | Why it separates real from generated data |
|---|---|---|
| Marginal shape | item mean, SD, skew, excess kurtosis | generated items are usually too tightly packed |
| Scale use | share at floor / ceiling / midpoint, distinct options used per respondent, round-number heaping | real respondents pile on endpoints and round numbers |
| Careless structure | longest run of identical answers, share of fully identical rows, within-person SD | real data has a careless and a ceiling-pegged tail; perfectly clean data is a tell |
| Covariance | mean inter-item correlation, Cronbach's alpha | the single biggest giveaway — see below |

## The reference data

`simulation_app/utils/reference_profiles.json` holds aggregate statistics
(no participant-level data is redistributed) computed from:

* **`psych::bfi`** — 25 IPIP personality items from the Synthetic Aperture
  Personality Assessment project, 2,800 respondents on a 6-point agreement scale,
  2,436 complete cases used. Distributed with the GPL-licensed R package `psych`
  and mirrored by the Rdatasets project; retrieved 2026-10-06. The item wordings
  on the package documentation page are what identify the seven reverse-keyed
  items (A1, C4, C5, E1, E2, O2, O5).

Two variants ship, because they genuinely differ: with the reverse-keyed items
aligned to the construct direction the mean inter-item correlation is **0.363**
and alpha **0.729**; left as administered they are **0.096** and **0.023**.
Comparing an all-positively-keyed simulated block against the mixed-keying
reference would demand an alpha no instrument should hit, so
`reference_for_scale(..., direction_aligned=...)` picks the right one.

### How the tolerances were chosen

Each tolerance is **twice the between-block standard deviation** of that metric
across the five real BFI blocks (floored at a small minimum). "Within tolerance"
therefore means *as close to real data as two real blocks of the same instrument
are to each other* — not an arbitrary pass mark.

### Honest limits of the reference

One self-selected web sample of one instrument family, on one scale length.
It pins down the shape of 6-point agreement data; it is **not** a reference for
other scale lengths (the benchmark widens tolerances and says so when it has to
substitute), other response formats, or economic-game allocations. Adding 5- and
7-point references needed OSF / Zenodo, which this build's network policy blocks;
5- and 7-point response-process aggregates were since measured from GitHub-mirrored
instruments (`registry/evidence_v1306.json`, see "Evidence gathered 2026-10-08" in
`EMPIRICAL_PROVENANCE.md`) but are **not yet wired into this benchmark**.

## Result: before and after

**Read this as a best case, not as a typical run.** The block below is synthetic
and its target reliability was set to the reference's own value, so alpha and
mean inter-item *r* move to the reference because they were aimed at it. On the
default path, with a real QSF and no stated reliability, the engine draws
`target_alpha ~ U(0.80, 0.90)` and hits that instead: measured across twelve real
Qualtrics designs, alpha is 0.842 before the realism layer and 0.843 after — it
does not move, because the engine is matching its own target rather than the
reference's 0.729. The rows marked "calibrated" below are arithmetic under a
matched target; the emergent rows are the evidence.

A 5-item, 6-point agreement block, N = 2,436, two conditions, template path
(no LLM), seed 11, **with the target reliability set to the reference's** — the
same run with the realism layer off and on:

| metric | engine before | engine after | real data | tolerance |
|---|---|---|---|---|
| mean inter-item r *(calibrated)* | 0.601 | **0.350** | 0.363 | 0.150 |
| Cronbach's alpha *(calibrated)* | 0.883 | **0.727** | 0.729 | 0.139 |
| within-person SD | 0.715 | **0.990** | 1.020 | 0.250 |
| identical-answer share | 0.002 | **0.052** | 0.052 | 0.030 |
| item SD | 1.204 | **1.396** | 1.409 | 0.216 |
| share at floor | 0.018 | **0.041** | 0.070 | 0.118 |
| distinct options used | 2.486 | **2.481** | 2.808 | 0.500 |
| **metrics within tolerance** | **10 / 14** | **14 / 14** | | |
| composite Cohen's d | +0.326 | **+0.341** | | |

The treatment effect survives: 0.326 → 0.341, a 5% change, because the transform
only rescales the item-specific component and pre-compensates the
between-condition component for the attenuation that added measurement error
causes.

### Which of the 14 are independent evidence

Three of the fourteen are **calibrated to the target**, so matching them is
arithmetic, not validation: mean inter-item r, alpha, and the identical-answer
share. The other eleven are **emergent consequences** of that calibration, and
they land inside real-data tolerance without being aimed at — item SD, skew,
kurtosis, floor / ceiling / midpoint / endpoint shares, distinct options used,
long-string length, within-person SD and item mean. That is the part that counts
as evidence.

### Within-person SD is a mixed-keying statistic

`item.likert.any.within_person_sd_fraction_of_span` (0.319 of span) is measured
on raw, un-recoded items, and every instrument behind it is mixed-keyed. A
consistent respondent answering 5 to the positive items and 1 to the reverse ones
contributes the whole keying gap to their own spread, which is why the real value
exceeds the item SD (0.295) — impossible for recoded items on one construct.

The entry therefore applies to mixed-keyed blocks only, and says so. Compared
against same-keyed blocks it reads as a 30% shortfall that is not there:
measured on this engine over four block shapes (8, 10 and 12 items; 5- and
7-point) and two seeds, a mixed-keyed block sits at **0.314 ± 0.002** of span
against the benchmark's 0.319 ± 0.045, while a same-keyed block sits at 0.238.
Item SD is keying-invariant by the same measurement (0.2978 same-keyed vs 0.2966
mixed), so its entry stays scale- and keying-free.

## Running it

```python
from utils import realism_benchmark as rb

out = rb.benchmark(df, item_columns, scale_min, scale_max,
                   direction_aligned=True)
print(out["verdict"], out["comparison"]["score"], out["comparison"]["failures"])
```

`compute_profile` works on any DataFrame or dict of columns, so the same code
profiles a real dataset for comparison. Missing values are deleted **listwise**:
dropping them per column shifts respondents against each other and silently
corrupts every row-wise metric — measured as 0.5% identical rows where the truth
was 11.5%.

## What the engine now does about it

1. **Reliability verification** (`utils/item_realism.py`, called from the scale
   loop). After generation, the achieved mean inter-item correlation is measured
   and, if it exceeds what the scale's target alpha implies, the item-specific
   variance is scaled up until it matches. The scale factor is solved by bisection
   against the measured correlation after rounding and clipping, because the
   analytic factor from the variance decomposition overshoots.

   This was needed because the existing correlation injection can only ever *raise*
   internal consistency, and it overshoots: measured 0.58 → 0.88 while aiming at
   0.75 on a 5-item block.

2. **Identical-answer realism**, run deliberately last. Generation leaves ~11% of
   respondents answering identically across a 5-item block; the ABE 3.0
   consistency audit then repairs nearly all of them away, leaving 0.3%. Real data
   sits between, at 5.2%, because people at the ceiling of a construct genuinely
   answer "6,6,6,6,6". The pass converts the respondents already closest to doing
   so — smallest spread first, ties broken by distance from an endpoint — to their
   own rounded block mean, so nobody's rank changes.

## The trade-off, stated plainly

Lower reliability attenuates the effect size observed on a composite score. That
is measurement error, not a bug, and it is why real studies need more
participants than a power calculation on true scores implies. It cannot be
avoided while also fixing reliability: scaling the shared and unique components
by the same factor leaves the inter-item correlation exactly where it was, so no
transform lowers alpha and leaves the observed composite *d* untouched.

`decouple_block(..., preserve_effect=True)` — the default — pre-compensates the
between-condition component by exactly the factor that cancels the attenuation,
so a user still observes the effect they specified. Passing `preserve_effect=False`
leaves the attenuation in, which is what you want when the question is "how much
power would this design really have?".
