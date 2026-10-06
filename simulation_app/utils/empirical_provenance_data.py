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
  * `shrinkage_factor()` returns 1.0 — NO publication-bias correction is applied,
    because the correction factor itself could not be verified. Applying an
    unverified shrinkage to every effect in the system would be worse than
    applying none.
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
    # Deliberately NOT installed. `set_shrinkage` is left uncalled, so
    # `shrinkage_factor()` returns 1.0 and `adjust_effect` applies no correction.
    # The literature on how far published effects exceed replication effects could
    # not be read here, and the candidate figures differ by definition (ratio of
    # pooled means vs median of per-pair ratios vs regression slope) by enough to
    # change every effect in the system.
    _ = set_shrinkage   # referenced so the signature stays honest about the hook
