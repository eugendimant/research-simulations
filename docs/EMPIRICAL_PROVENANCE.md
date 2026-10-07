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
  can differ the way two real labs differ. **Currently inert**: see below.

## State of verification: 1 of 484 source-verified

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

* `shrinkage_factor()` returns **1.0** — no publication-bias correction is
  applied. The candidate figures for how far published effects exceed replication
  effects differ by definition (ratio of pooled means vs median of per-pair ratios
  vs regression slope) by enough to change every effect in the system, and the
  comparison baseline differs too (single original studies vs published
  meta-analyses). Applying an unverified correction to every effect would be worse
  than applying none.
* `default_tau()` returns **0.0** — no heterogeneity draw, for the same reason:
  τ is reported on different scales (*d*, Fisher's *z*, log odds) and is not
  comparable across them, and an I² is a proportion of observed variance, not an
  absolute magnitude that can be used as a perturbation SD.
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
and response-style norms, and the publication-bias shrinkage factor.

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

## Reaching the audit from code

```python
from utils import empirical_registry as reg

reg.coverage_summary()          # headline counts by tier and by table
reg.audit_table()               # one row per entry, worst-sourced first
reg.honesty_notice()            # the paragraph to show users
reg.status_of("game", "dictator_standard")
reg.provenance_of("game", "dictator_standard").note
```
