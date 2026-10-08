"""
Empirical Provenance Data — the verification record, kept separate from the logic.
=================================================================================

`empirical_registry.py` imports `install()` from here and calls it with the
registration helpers. Keeping the records in their own module means the
verification pass can grow without touching the registry logic, and a mistake in
the data cannot take the engine down (the import is guarded).

STATE OF VERIFICATION AS OF 2026-10-06
--------------------------------------
Zero of the 484 knowledge-base entries have been verified against a primary
source, because this build cannot reach one. Every scholarly host and data
repository attempted was refused by the environment's network egress policy:
nature.com, science.org, journals.sagepub.com, psycnet.apa.org, link.springer.com,
pmc.ncbi.nlm.nih.gov, royalsocietypublishing.org, econtent.hogrefe.com, osf.io,
zenodo.org, dataverse.harvard.edu, openpsychometrics.org, api.crossref.org,
api.openalex.org and doi.org all answered 403 to CONNECT. Web SEARCH works, but a
search engine's summary of a paper is not the paper: it cannot be quoted, its
metric definitions cannot be checked, and it cannot distinguish an article's
figure from a correction to it or a commentary on it. Recording such a figure as
"verified" would be exactly the failure this module exists to prevent, so nothing
here is filled in from search snippets or from recall.

What that means in practice:
  * `status_of()` reports UNVERIFIED for effectively every entry, which is the
    honest answer, and the engine damps unverified entries accordingly.
  * The publication-bias shrinkage is installed from RECALL, not from a source
    (see `RECALLED_SHRINKAGE` below and `set_recalled_shrinkage`). It is tiered
    `recall_uncertain`, `shrinkage_verified()` is False, and the notice says so.
  * The one record below is an INTERNAL-CONSISTENCY finding, derived from the
    knowledge base's own numbers rather than from a paper, and is labelled as such.

To finish the job, the environment needs egress to pmc.ncbi.nlm.nih.gov, osf.io,
nature.com and royalsocietypublishing.org (which between them cover most of the
open-access material), or the dozen papers placed on a local path. The
verification itself is then mechanical.
"""
from __future__ import annotations

__version__ = "1.0.0"


def install(register, set_shrinkage, Provenance,
            VERIFIED, PARTIAL, CITED_UNCHECKED, CORRECTED, UNVERIFIED) -> None:
    """Register every provenance record. Called by `empirical_registry` on import."""

    # ── Internal-consistency findings ────────────────────────────────────────
    # Not a literature check: this is the knowledge base contradicting itself, and
    # that can be established without leaving the repository.
    register("game:dictator_standard", Provenance(
        status=CORRECTED,
        source="Engel (2011) as cited by the knowledge base entry",
        doi="",                     # not confirmed: the source could not be fetched
        url="",
        quote="",
        verified_on="2026-10-06",
        checked_fields=("sd_proportion",),
        corrected={"sd_proportion": 0.31},
        note=(
            "INTERNAL CONSISTENCY ONLY — the cited paper could not be fetched, so the "
            "mean, k and N remain unverified. What is established is that the entry "
            "contradicts itself: it states sd_proportion=0.18 while also stating "
            "subpopulation shares of 36% giving nothing, 17% giving half and 5% giving "
            "everything. A distribution with those point masses and a mean near 0.28 has "
            "an SD close to 0.31 — simulating 0.18 cannot reproduce the very bimodality "
            "the same entry documents. 0.31 is the SD of the mixture that reproduces the "
            "entry's own stated masses and mean (see empirical_marginals.PointMassMixture)."
        ),
    ))

    # ── Publication-bias shrinkage ───────────────────────────────────────────
    # Not installed here: the figure rests on recall, so it goes through
    # `set_recalled_shrinkage` (see `RECALLED_SHRINKAGE`), never `set_shrinkage`,
    # which is reserved for a checked source.
    _ = set_shrinkage   # referenced so the signature stays honest about the hook


#: Publication-bias shrinkage and default heterogeneity for effects the tool INFERS
#: (never for an effect the user specifies). Every figure here is recalled from the
#: replication literature and has not been checked against a source in this build.
#: Recalled evidence and its spread (ratio = replication effect / original effect):
#:   * Open Science Collaboration (2015, Science), 100 psychology studies:
#:     replication effects about half the original (r about .20 vs .40); the social
#:     psychology subset did worse than the cognitive subset.
#:   * Camerer et al. (2018, Nature Human Behaviour), 21 social-science experiments
#:     from Nature/Science: replication effects about 0.5 of the original (roughly
#:     0.45-0.75 depending on the summary used). Camerer et al. (2016, Science),
#:     18 laboratory economics experiments: about 0.66.
#:   * Klein et al. (2014; 2018), Many Labs 1 and 2: about half of the selected
#:     effects replicated in 2, with replication effects well below the originals.
#:     Search summaries of the Many Labs 2 abstract (2026-10-08) give median d 0.60 for
#:     the originals and 0.15 for the replications, a ratio near 0.25.
#:   * Kvarven, Stromland and Johannesson (2020, Nature Human Behaviour): meta-
#:     analytic estimates exceeded matched preregistered multi-lab replications by
#:     roughly d 0.2-0.3 on average, i.e. a meta-analytic d near 0.4-0.5 corresponds
#:     to a replication d near 0.1-0.2 (ratio roughly 0.3-0.6).
#:   * Schafer and Schwarz (2019, Frontiers in Psychology): preregistered or large
#:     studies report effects about half the size of the rest of the literature.
#:   * Mertens et al. (2022, PNAS) found nudge d about 0.43 before correction; Maier
#:     et al. (2022) found little to no evidence of an effect after bias correction.
#:   * Heterogeneity: typical between-study SD in social-psychology meta-analyses is
#:     recalled at about d 0.1-0.25; the knowledge base's own entry tau values run
#:     0.05-0.20 (median 0.12), so 0.15 is used where an entry reports none.
#: The active factor is the meta-analysis one because every inferred path draws on
#: META_ANALYTIC_DB. v1.3.0.6 lowers it from 0.60 to 0.45. Figures stated by search
#: summaries on 2026-10-08 (docs/EMPIRICAL_PROVENANCE.md, "Evidence gathered";
#: registry/corroboration_v1306.json): OSC 2015 about 0.5 (k=100), Camerer 2018 about
#: 0.5 (k=21; 0.71 for true positives), Camerer 2016 0.66 (k=18), Many Labs 2 median
#: d 0.60 -> 0.15 = 0.25 (k=28), Kvarven 2020 meta-analyses almost 3x the preregistered
#: replications = 0.33 (k=15). Weighted by the number of studies that is 0.46; the plain
#: median is 0.50; the two sources that compare META-ANALYSES with replications
#: (Kvarven, Many Labs 2) are the lowest. 0.45 is the study-weighted centre, rounded
#: down because the inferred path draws on meta-analytic entries and the knowledge
#: base was already recalibrated toward meta-analytic values in places. The floor
#: (EffectPolicy.min_retained = 0.35 of the published d) is unchanged and now binds in
#: the lower tail of the between-study draw. `by_evidence` documents the other evidence
#: types; only "meta_analysis" is applied today. Still recall-tier: the summaries are
#: not the papers.
RECALLED_SHRINKAGE = {
    "factor": 0.45,
    "corrected": {"factor_was": 0.60, "meta_analysis_was": 0.60},
    "default_tau": 0.15,
    "by_evidence": {
        "single_original_study": 0.50,
        "meta_analysis": 0.45,
        "preregistered_replication": 1.00,
    },
    "source": ("Open Science Collaboration 2015; Camerer et al. 2016, 2018; Klein et al. "
               "2014, 2018; Kvarven et al. 2020; Schafer & Schwarz 2019; Mertens et al. 2022"),
    "note": ("Inferred effects are multiplied by 0.45 (stated range about 0.25-0.71 across "
             "OSC 2015, Camerer 2016/2018, Many Labs 2, Kvarven 2020) and given a between-study "
             "draw with SD 0.15 d where the entry reports no tau (recalled range 0.1-0.25). "
             "Estimates differ by definition (ratio of pooled means, median of per-pair "
             "ratios, regression slope), so the figure is a judgement, not a measurement. "
             "Lowered from 0.60 in v1.3.0.6 after search-summary figures (OSC 0.5, Camerer 0.5/0.66, "
             "Many Labs 2 0.25, Kvarven 0.33) gave a study-weighted centre of about 0.46. "
             "Never applied to an effect the user specifies."),
}
