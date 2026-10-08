# Empirical Provenance

Which numbers in this tool are actually sourced, and which only look sourced.

## The problem

`simulation_app/utils/scientific_knowledge_base.py` holds 484 calibration
entries — 187 meta-analytic effects, 68 economic-game calibrations, 201 construct
norms, 12 cultural adjustments, 10 response-time norms, 6 order effects. Each one
carries a citation, a 95% confidence interval, a study count *k*, a participant
count *N*, a heterogeneity estimate τ and an I². Presented that way, every entry
reads as if it were transcribed from a paper.

Most were not. The values carry the statistical fingerprint of bulk generation
rather than extraction:

* **79%** of the 187 meta-analytic *N* values are exact multiples of 1000.
* **98%** of the 201 construct-norm *N* values are exact multiples of 100;
  **96%** of the game calibrations likewise.
* τ takes only **8 distinct values** across all 187 entries, clustered on 0.08,
  0.10, 0.12, 0.14, 0.15, 0.18, 0.20.
* **173 of 187** entries are labelled `replication_status="replicated"`.

Some entries also contradict themselves. `dictator_standard` states
`sd_proportion = 0.18` while simultaneously documenting subpopulation shares of
36% giving nothing, 17% giving exactly half and 5% giving everything. A
distribution with those point masses and a mean near 0.28 has an SD close to
**0.31**; 0.18 cannot produce the very bimodality the same entry documents. That
one is established without leaving the repository, and is recorded as a
correction.

A tool whose selling point is that it is grounded in the literature must not
present unverified numbers as published fact. That does not mean the entries are
worthless — they are a reasonable prior, assembled by something that has read a
great deal of this literature — but a prior and a published measurement are
different things and the difference has to be visible.

## What this release adds

`simulation_app/utils/empirical_registry.py`:

* **Provenance records** — per entry: status, citation, DOI, the URL that was
  fetched, the quoted sentence, the verification date, which fields the quote
  actually supports, and any correction. Entries with no record report as
  `UNVERIFIED`, which is the honest default rather than silence.
* **Verification tiers** and a weight for each: `verified` / `corrected` 1.00,
  `partial` 0.85, `cited_unchecked` 0.70, `unverified` 0.55. Where the simulator
  consults an entry through this layer, an unverified number's deviation from
  neutral is damped, so an invented value can never drive a run as hard as a
  sourced one.
* **`audit_table()` and `coverage_summary()`** — one row per entry with its tier
  and provenance, so the app can answer "where does this number come from?" for
  any value behind a run.
* **`honesty_notice()`** — a paragraph stating the current counts, suitable for
  showing verbatim next to any literature claim in the UI.
* **Replication adjustment** — `EffectPolicy` + `adjust_effect()` apply
  publication-bias shrinkage and a between-study heterogeneity draw, so each run
  can differ the way two real labs differ. **Active since v1.3.0.5 on a recalled
  figure**: see "Consequences".

## State of verification: 1 of 484 source-verified

> **As of 2026-10-08 (v1.3.0.6):** the knowledge base holds 503 rows (the 484 above plus
> 19 paradigm entries added in v1.3.0.5). **1 of 503 is sourced** (`corrected`), **0 are
> `verified`**, and the headline did not move in the evidence pass below. Separately,
> the registry *store* now holds **146 MEASURED entries**: 43 in `item_process.json`
> and 103 in `evidence_v1306.json`. Those are computed from data we hold, are not rows of
> the knowledge base, and are not counted in `sourced_entries`.

**No entry has been verified against a primary source**, because this build cannot
reach one. Every scholarly host and data repository attempted was refused by the
environment's network egress policy with a 403 to CONNECT:

```
nature.com                science.org              journals.sagepub.com
psycnet.apa.org           link.springer.com        pmc.ncbi.nlm.nih.gov
royalsocietypublishing.org  econtent.hogrefe.com   osf.io
zenodo.org                dataverse.harvard.edu    openpsychometrics.org
api.crossref.org          api.openalex.org         doi.org
pubmed.ncbi.nlm.nih.gov   europepmc.org            arxiv.org
```

Web *search* works. A search engine's summary of a paper is not the paper: it
cannot be quoted, its metric definitions cannot be checked, and it cannot
distinguish an article's own figure from a correction to it or a commentary on it.
Recording such a figure as "verified" would be precisely the failure this layer
exists to prevent, so nothing is filled in from snippets or from recall.

The one record present is the `dictator_standard` SD correction above, and it is
labelled as an internal-consistency finding, not a literature check.

## The recall audit, 2026-10-06: 483 of 484

Since sources cannot be read here, the entries were instead audited against what a
model that has read a great deal of this literature recalls of it — entry by entry,
checking the citation, the design, the direction of the effect, the magnitude
against the published or meta-analytic estimate, and internal consistency. The
verdicts live in `simulation_app/utils/registry/recall_audit.json`; the per-entry
log is `summaries/literature-recall-audit-2026-10-06.md`.

| Verdict | Entries |
|---|---|
| `recall_consistent` — citation recognised, value consistent with the literature | 200 |
| `recall_corrected` — a field was wrong and was changed | 125 |
| `recall_uncertain` — citation plausible, number not judgeable from memory | 157 |
| `unrecognized` — citation could not be placed at all | 1 |

169 fields across 122 entries were changed, and every one keeps the value it
replaced in its record, so the pass is reversible from its own data.

**This does not move the verification count, by construction.** Recall is not a
source check, and the two are kept structurally apart: `register_recall()` refuses
any tier above the recall band, refuses a verdict with no note, and refuses to
overwrite a record that rests on evidence; `doi`, `url`, `quote` and `verified_on`
are empty by construction; every recall tier weighs less than `CITED_UNCHECKED`;
none grants `may_set_magnitude`; and `sourced_entries` still counts only
`VERIFIED`/`CORRECTED`/`PARTIAL`. See `docs/REGISTRY.md`, "The recall band".

### What the audit found, in kind

The defects cluster, and the clusters matter more than any single number:

* **Meta-analytic apparatus attached to sources that are not meta-analyses.** The
  largest class by far, and the main driver of the 157 `recall_uncertain`
  verdicts: a real citation — a book, a narrative review, a handbook chapter, a
  single experiment — carrying a pooled `effect_d`, `n_studies`, `ci_95`, τ and
  I² that the cited work does not report.
* **Replication-crisis magnitudes carried at their original size.** Ego depletion,
  moral reminders, facial feedback, the decoy effect, deindividuation and the ELM
  were labelled `replicated` while citing the very work whose verdict was
  negative.
* **Clinical-level means presented as general-population norms.** Nine construct
  norms had a per-item mean that, converted back to the instrument's total, lands
  past that instrument's own clinical cutoff (PHQ-9, GAD-7, PSS-10, AUDIT, PSQI,
  PHQ-15, LSAS, MBI-EE, SCL).
* **`scale_points` holding something other than a response-option count** — an
  item count, a score range, or a Likert count for an instrument that has no
  Likert metric (the SVO slider's angle, IPAQ's minutes, the NPI's forced-choice
  dyads). This field drives response generation, so it is the costliest to get
  wrong.
* **Internally impossible heterogeneity** — `i_squared=0.0` beside a positive τ,
  in entries across several domains.
* **Near-duplicate entries with disagreeing values**, which will double-count or
  conflict at calibration time (`social_proof_meta`/`conformity_asch_meta`,
  `contact_hypothesis`/`intergroup_contact_extended`, the two inoculation keys,
  the two misinformation-correction keys, `growth_mindset_meta`/
  `growth_mindset_intervention_meta`, `common_pool_resource_standard`/
  `tragedy_of_commons_standard`, `social_exclusion_nts`/`ostracism_cyberball`).
  These are reported, not resolved: dropping a key changes what the engine looks
  up and belongs in its own change.

The synthesized-statistics finding above is confirmed from the other direction:
where a `k`/`N` pair *does* match a real meta-analysis (Rowland, Cepeda, Durlak,
Hughes, Nickow, Sisk, Guilbault, Roseth, Engel, Johnson & Mislin, Jachimowicz,
Khoury, Ekers, Humphrey), the τ and I² beside it still do not match what those
papers report. Every τ and I² in the knowledge base should be treated as
unsourced regardless of tier.

### Consequences, deliberately

* `shrinkage_factor()` returns **0.45** (0.60 until v1.3.0.5; see "Evidence gathered" below) and `default_tau()` **0.15**, but
  both rest on **recall, not on a source** (`policy:publication_bias` is tiered
  `recall_uncertain`; `shrinkage_verified()` is False and `honesty_notice()` says
  so). They are installed through `set_recalled_shrinkage()`, which writes only to the
  recall band and refuses to overwrite a verified record; `set_shrinkage()` still
  accepts only a checked source. The figures below were the recalled evidence behind
  0.60 and are kept as the record of that judgement; the 2026-10-08 search-summary figures
  that moved it to 0.45 are in the next section (ratio = replication / original effect):
  Open Science Collaboration 2015 about 0.5; Camerer et al. 2018 about 0.5 (0.45-0.75
  by summary); Camerer et al. 2016 (economics) about 0.66; Many Labs 1 and 2 about
  half; Kvarven et al. 2020 meta-analyses exceed preregistered replications by d
  0.2-0.3, a ratio near 0.3-0.6; Mertens et al. 2022 nudges 0.43 before and near zero
  after bias correction. 0.60 was the top of that range on purpose, so that a real
  effect is not erased. The per-evidence-type table (`single_original_study` 0.50,
  `meta_analysis` 0.45, `preregistered_replication` 1.00) is documentation; only the
  meta-analysis figure is applied, because every inferred path draws on
  `META_ANALYTIC_DB`. τ 0.15 d is the recalled 0.1-0.25 range for social-psychology
  meta-analyses and is used where an entry reports none (entries report 0.05-0.20).
* **Where it applies.** Only to effects the tool *infers*: the paradigm anchor
  (`_match_meta_entry` -> `_shrink_inferred_meta_effect`) and the literature
  fallback (`literature_effects.lookup`). Never to a user-specified effect, the
  `auto_effects=False` null, economic-game calibrations or relational / intergroup
  (STEP 0) effects. The generic keyword (STEP 1-2) effects are hand-set, not
  published d values, and since v1.3.0.6 take the same 0.45 factor directly (no
  heterogeneity draw, no tier weight) because their magnitudes were set from original
  rather than replicated findings. `adjust_effect` keeps
  the sign, and in replication-adjusted mode the result never falls below
  `min_retained` (0.35) of the published magnitude. Each run's draw is seeded by the
  engine's stable RNG, so a seeded run reproduces. The fallback also has a
  per-entry tier weight (0.55-0.68 for recall tiers). Until v1.3.0.5 it was multiplied
  with the 0.45 shrinkage, so a fallback effect landed near the 0.35 floor: two
  discounts for one doubt. Since v1.3.0.6 the STRONGER of the two applies
  (`min(tier weight, shrinkage)`; the factor is 0.45 since the evidence pass): doubt about the value is two-sided, so it is a reason
  for an unchecked number never to push harder than a checked one (a verified entry gets
  the shrinkage alone, an unverified one at most that), not a second reason to shrink.
  The anchor path has no tier weighting. The content matcher (`literature_effects._is_confident`)
  also needs, since v1.3.0.6, a DISTINCTIVE token from the condition label (at most three paradigms
  of the table, not a generic word such as brand/control/product/health), and a label with a plain or
  reference qualifier (generic, standard, basic, regular, plain, "default brand") never matches: on the
  example labels this removed 27 spurious matches and corrected one. A literature-routed arm is applied against the
  zero-point reference arm with a gap of `2 x 0.109 x d` on every route, so the applied d
  is what the composite shows (the fallback used to deliver about d/2).
* The verified distribution shapes in `empirical_marginals.py` are gated on
  provenance, so `marginal_for()` returns `None` for everything unsourced and **no
  economic-game marginal is active**. The machinery, the mixture solver and its
  tests all ship; activating a shape needs a source.

### What would finish it

In order of preference:

1. **Egress to `pmc.ncbi.nlm.nih.gov`, `osf.io`, `nature.com` and
   `royalsocietypublishing.org`.** Between them these cover most of the relevant
   open-access material, including the publication-bias and heterogeneity
   literature the shrinkage factor depends on.
2. **The dozen key papers on a local path**, which needs no network at all and is
   the most reliable route.

Once either is in place the work is mechanical: fetch, quote, register. The
priority order is the entries the engine actually consults — the four economic-game
meta-analyses (dictator, trust, ultimatum, public goods), the careless-responding
and response-style norms; the publication-bias shrinkage factor, which is
installed from recall and still needs a source check.

## What *is* verified

The realism benchmark's reference data. `utils/reference_profiles.json` was
computed from real human item-level responses that **were** fetched (a GitHub-hosted
mirror, which the egress policy allows), with the source page, the data file URL,
the licence, the retrieval date and a quotation recorded alongside. See
`REALISM_BENCHMARK.md`.

That asymmetry is worth stating plainly: this release can demonstrate that the
simulator's *distributions* now match real human data, measured. It cannot yet
demonstrate that its *effect sizes* match the published literature, because it
could not read the literature.

## Evidence gathered 2026-10-08

The sandbox still cannot read a journal, OSF, Zenodo, Crossref or PubMed. It can read
raw files on GitHub (and the Rdatasets mirror), and it can run web searches. This pass
used both, and keeps them in different tiers.

* **Data we hold -> `MEASURED`.** `tools/derive_evidence_benchmarks.py` recomputes
  everything from the downloaded files; the SHA-256 of every file is recorded in
  `registry/evidence_v1306.json`, and re-running the script on the same files reproduces
  that file byte for byte (`EVIDENCE_DATA_ROOT=<dir> pytest tests/test_evidence_v1306.py`).
* **Search-engine summaries -> `registry/corroboration_v1306.json`.** Each item records
  the claim, our value, what the summary states (25 words or fewer), the URLs and a
  verdict. A summary is not the paper, so **no tier changed and nothing is `verified`**.

### What was measured

All entries are `T0_MEASURED`. The `evidence.` ids are not consulted by any engine path.
The one family that is: the 7-point straight-lining cells are republished as
`item.likert.<keying>.k<n>.sp7.straightlined_share` in `item_process_7pt_v1306.json`, scoped to
seven response options, and the identical-answer pass reaches them through its existing
registry lookup (see "Wired in" below). Nothing else the engine does reads this pass.

| Dataset | n (rows in file) | Used for | SHA-256 |
|---|---|---|---|
| CFCS, 12 items, 5-pt | 15,035 | 5-pt marginals, reliability, straight-lining | `1dfbbf32ba04...c23f58a` |
| MACH-IV, 20 items, 5-pt, per-item page times | 73,489 | 5-pt block; **response times** | `52b830731aea...f882fdaec` |
| Protestant Work Ethic, 19 items, 5-pt | 1,350 | 5-pt block | `457a27a112c9...c21` |
| Nerdy Personality Attributes, 26 items, 5-pt | 25,226 | 5-pt block; test seconds | `286019040c1f...373ef` |
| Hypersensitive Narcissism, 10 items, 5-pt | 53,981 | 5-pt block | `8d42609e61e2...fd` |
| RIASEC, 6 x 8 items, 5-pt, all keyed one way | 145,828 | **same-keyed** 5-pt blocks | `738c0db516ac...dfa06` |
| HEXACO-IPIP, 24 facets x 10 items, **7-pt** | 22,786 | 7-pt blocks | `c57020fb7b4a...f99` |
| TIPI (7-pt, 10 items) appended to four of the above | 1,327-143,748 | 7-pt marginals | (host files above) |
| Ultimatum responder study (university students) | 111 | acceptance by offer share | `b1681e8bf1d9...5eac2` |
| Binary dictator choices (Bruhin, Fehr & Schunk replication data, course copy) | 6,786 choices, 174 people | prosocial choice share | `a6cb429c5854...5b9` |
| Rdatasets `carData::Guyer` | 20 groups | n-person PD cooperation | `5874ec2f6b10...d917` |
| Rdatasets `psych::Garcia`, `psych::Tal_Or` | 129, 123 | observed experimental effects | `c348699e39bf...`, `0ed07116a2e5...` |

The openpsychometrics files are the same GitHub mirror and the same stated terms as
`item_process.json` ("solely for research purpose"); no licence is declared, so only
aggregates are committed. The ultimatum and binary-dictator files are public repositories
with no stated licence and the Rdatasets files come from R packages whose licence was not
checked here; for all of them only derived aggregates and hashes are kept. Keying is
empirical (sign of the first-principal-component loading inside a single-construct
block), not a recalled key.

What came out (value +- SD across blocks, or the entry's stated width):

| Quantity | 5-point | 7-point | Registry today |
|---|---|---|---|
| Item SD as fraction of span | 0.328 +- 0.019 (11 blocks) | 0.283 +- 0.018 (28 blocks) | 0.295 (5- and 9-pt) |
| Endpoint occupancy | 0.420 +- 0.068 | 0.234 +- 0.054 | 0.338 (5-pt, 8-10 items) |
| Midpoint share | 0.177 +- 0.043 | 0.120 +- 0.021 | none |
| Straight-lining, 5 items, same-keyed | 0.101 +- 0.065 | **0.031** +- 0.013 | 0.081 (5-pt) |
| Straight-lining, 5 items, mixed-keyed | 0.010 +- 0.003 | 0.006 +- 0.003 | 0.013 |
| Aligned alpha, 3 / 5 / 8 items | 0.62 / 0.72 / 0.81 | 0.62 / 0.73 / 0.80 | none |
| Aligned mean inter-item r | 0.37 | 0.36 | 0.363 (bfi, 6-pt) |

* The existing 5-point straight-lining rates are **corroborated by independent data**
  (mixed 0.010 vs 0.013, same-keyed 0.101 vs 0.081; the 5-point same-keyed rate has a
  wide spread because it rests on eight same-keyed blocks: NPAS, HSNS and six RIASEC scales).
* **7-point same-keyed blocks straight-line about a third as often as 5-point ones**
  (0.031 vs 0.101 at five items). `item.likert.same.k*` was keyed on width and keying only,
  so it applied the 5-point rate to 7-point scales. This is now corrected (see "Wired in").
* Item SD as a share of span is 0.283 at seven points against 0.328 at five; the scale-free
  0.295 is within one across-block SD of the 7-point value and 1.7 SD below the 5-point one.
* Response time (MACH-IV, 73,473 complete respondents): median 7.0 s per page, 10th-90th
  percentile of per-person medians 4.5-11.7 s; 0.6% of pages under 1 s; 0.2% of people
  with a median under 2 s. NPAS and RIASEC give 4.0 and 4.9 s per item on a single
  page. These are volunteers who agreed to a research follow-up, so fast responding is
  likely under-stated for paid panels.
* Careless-responding proxies in these volunteer data are low: 0.1% answered all 20
  mixed-keyed MACH items identically. The engine's 5% default is therefore not contradicted
  by these samples and not supported by them either (Meade & Craig-type 10-12% rests on
  undergraduates and different indicators).
* **Ultimatum responder acceptance by responder's share of the pie** (n = 111 students,
  per-person rates over a few trials): 100% at 50%, 90% at 45%, 79% at 40%, 38% at 25%,
  33% at 20%, 28% at 15%, 23% at 10%, 18% at 5%. A step between 40% and 25%, not a smooth
  curve; the stakes are not stated in the file.
* **Binary dictator choices**: 21% of trade-off games end with the option that pays the
  receiver more at the dictator's cost; 37% when the dictator is ahead in both options
  against 12% when behind in both; 34% at below-median cost against 8% above. Binary
  choices, not a continuous give-away: not comparable to the 28% mean of `dictator_standard`.
* **n-person prisoner's dilemma (Fox & Guyer 1978)**: cooperation 0.34 anonymous vs 0.46
  public, 10 groups per cell, d = 1.19 +- 0.49 (group level, repeated).
* **Two single experiments**, observed d with its standard error: Garcia et al. liking,
  individual protest vs control +0.48 +- 0.22 and collective +0.39 +- 0.22; anger,
  collective vs control -0.72 +- 0.22; Tal-Or presumed media influence +0.37 +- 0.18 and
  reaction +0.32 +- 0.18. These are single published studies and carry winner's-curse risk.

What was **not** found: no continuous dictator, public-goods or trust-game participant
data (the searches surfaced only code, LLM benchmarks and a compiled "dataset" repository that appears to hold
study-level summaries rather than participant rows, which was skipped); no 5/7-point instrument with a
published key available to compare against the empirical key; a `BSE` PsychoPy ultimatum
set (33 students) was downloaded and dropped because its `offer`/`Total_Pot` columns are
inconsistent (pots of 0, shares above 1) and its frame labels are ambiguous; the
openpsychometrics ECR file has no codebook, so its item wording is unknown.

### What the searches corroborated or contradicted

51 items in `corroboration_v1306.json`: **27 corroborated, 6 contradicted, 18
inconclusive** (the sixth contradiction is the 0.60 shrinkage factor itself). Condition names of the 303 example QSFs matched only 10 older knowledge-base
entries at all (so "the 15 most frequent" could not be filled); all 10 were searched, plus
the 19 paradigm entries from v1.3.0.5, the replication figures and the game baselines.

* **Replication shrinkage (changed, 0.60 -> 0.45).** Stated by the summaries: Open
  Science Collaboration about 0.5 (100 studies); Camerer 2018 about 0.5 (21; 0.71 for
  true positives); Camerer 2016 0.66 (18); Many Labs 2 median d 0.60 -> 0.15 (**0.25**,
  not "about half": the half was the share that replicated; 28); Kvarven 2020
  meta-analyses "almost three times" the replications (0.33; 15 pairs); Mertens 0.45 raw,
  0.04 (Maier) to 0.08-0.31 (Szaszi) corrected. Weighting each source by its number of
  studies gives **0.46**; the plain median is 0.50; the two sources that compare
  meta-analyses (what the inferred path draws on) with replications are the lowest. The
  active factor is therefore **0.45**, with the caveats that the definitions differ (ratio
  of means, of medians, of significant subsets), the lowest figures come from selected
  controversial effects, and Lewis et al. call the Kvarven gap "less dramatic". The
  factor stays a **recall-tier** figure (`recall_uncertain`, `shrinkage_verified()`
  False) because summaries are not the papers; the old value is kept in the policy's
  `corrected` record as `factor_was`. `min_retained` stays at 0.35 of the published d.
  Consequence: the fallback path, which multiplies by a recall-tier weight of 0.55-0.68,
  now sits on that floor (0.35 x published) rather than near it; the paradigm-anchor path
  follows the factor (0.45 x published, with the between-study draw above the floor).
* **Game baselines.** Corroborated: dictator mean 28.3%, 36% giving nothing, bimodal at 0
  and one half; ultimatum mean offer about 40% (37 papers); trust-game sending about 50%
  (secondhand). Inconclusive: public-goods mean (summary gives group efficiency), Sally's
  PD rate, Dal Bo & Frechette, the "rejected about half of the time below 20%" rule (our
  measured students reject 67-82% there), Engel's study count (the summaries disagree).
  Contradicted but not applied: Johnson & Mislin "eighty-four iterations" against our
  k = 162 (one abstract).
* **Paradigm entries.** Corroborated on value and k: identifiable victim (r = .05, 41
  studies), expressive writing (r = .075, 146), positive psychology (39 studies, 0.20-0.34),
  active learning (0.47, 225), feedback (0.41), tailored messages (r = .074, 57), illusory
  truth (0.39 within items; 0.50 between), gamification (0.25-0.49), imagined contact
  (d+ 0.35 vs our 0.25, inside our interval). Contradicted: ostracism -0.45 against a Cyberball
  meta-analysis reporting d > |1.4| on need/affect outcomes (one abstract, measure-specific:
  flagged, not applied). Inconclusive: the rest, usually because the summary gives counts
  but no pooled effect.
* **Older entries.** Corroborated: anchoring (g = 0.825 in a 2023 meta-analysis), placebo
  (-0.23, 158 trials against our k = 202), defaults (0.68), ingroup cooperation (0.32),
  self-affirmation (0.17 for message acceptance). Contradicted: `loyalty_program_meta` and
  `personalization_meta` cite no meta-analysis at all (an agenda paper, a review and two
  primary experiments), which the recall audit had already suspected; the pooled d and k
  have no source. Inconclusive: endowment, scarcity (k = 131 confirmed), zero price.

**What was changed on this evidence** (all through the recall band, old values kept as
`<field>_was` in the record): the shrinkage factor (0.60 -> 0.45, above); `loyalty_program_meta`
and `personalization_meta`, which cite no meta-analysis, now say so in their source and notes
and carry k = 0 and N = 0 (were 35 / 100,000 and 40 / 35,000) with the d unchanged and
labelled a judgement; `social_exclusion_ostracism_meta` keeps -0.45 but is scoped to a
downstream belonging or mood outcome (the Cyberball meta-analysis reports |d| above 1.4 for
need-satisfaction checks); `trust_standard` gains a note on the 84-iteration count (value
unchanged, one abstract). The records are in `registry/recall_evidence_v1306.json`.
No knowledge-base magnitude was changed.

### What remains unchecked

Everything in the knowledge base is still unsourced except `dictator_standard`'s internal
SD correction. This pass adds confidence, not sourcing: a summary agreed with 27 of 51
claims, but none of those summaries is the paper. Still needed: the primary text of the
four game meta-analyses (the pooled public-goods and PD rates), Gerber & Wheeler 2009 and
the source of the `-0.45`, the Mertens and Maier tables behind `nudge_general_meta`, and
a continuous-giving dictator/trust data set. The 7-point straight-lining rates are wired in (below); the other measured families are not.

### Wired in (v1.3.0.6)

`registry/item_process_7pt_v1306.json` holds 16 cells (same- and mixed-keyed, widths 3-10, 7-point
only). `_apply_identical_answer_realism` calls `lookup_best("item.likert.<keying>.k<n>", ...)` as
before; the new cells carry `scale_points [7]` and are narrower, so they win for 7-point blocks and
decline for every other scale length. No engine code changed.

Validation (corpus benchmark harness, 10 random corpus inputs containing 7-point blocks, N = 500,
seed 1, this registry file removed vs present; 37 blocks of 3+ items): identical-answer share on
7-point blocks 0.102 -> 0.047 on average (5-item same-keyed 0.082 -> 0.030 against the measured
0.031; 4-item 0.118 -> 0.054; 3-item 0.194 -> 0.112; 9-item 0.034 -> 0.004); mean alpha
0.814 -> 0.809 (largest block change 0.016), mean inter-item r 0.477 -> 0.468, item SD fraction
of span 0.299 -> 0.298, endpoint share 0.257 -> 0.250. Three further inputs with 38 five-point
blocks and the 2- and 4-point blocks in the first ten: every metric bit-identical (max absolute
difference 0.0). The recovery and fidelity suites (86 tests) pass with the 0.45 factor.

## Reaching the audit from code

```python
from utils import empirical_registry as reg

reg.coverage_summary()          # headline counts by tier and by table
reg.audit_table()               # one row per entry, worst-sourced first
reg.honesty_notice()            # the paragraph to show users
reg.status_of("game", "dictator_standard")
reg.provenance_of("game", "dictator_standard").note
```
