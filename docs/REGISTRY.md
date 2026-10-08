# The empirical registry

Where the simulator's numbers come from, what each one is allowed to do, and how to
promote one.

## Why tiers

`scientific_knowledge_base.py` holds 484 calibration entries, each carrying a citation,
a 95% CI, k, N, τ and I². They *look* transcribed from papers. They were not: the values
carry the fingerprint of generated numbers (79% of meta-analytic N values are exact
multiples of 1000, τ takes only 8 distinct values across 187 entries, 173/187 are marked
"replicated"). As of 2026-10-06, **zero of them have been checked against a primary
source**, because this container's network policy blocks every journal, OSF, Zenodo,
Crossref and doi.org itself.

A tool whose selling point is "grounded in the literature" must not present unverified
numbers as published fact. So every value that changes output carries a tier, and the
tier — not the prose around it — governs what the value may do.

| Tier | Evidence | May set a magnitude? |
|---|---|---|
| `MEASURED` | Computed by us from a dataset we hold. Records the script, the data file and its SHA-256. | **Yes** |
| `VERIFIED` | Primary source read, with the verbatim source sentence and a DOI. | **Yes** |
| `CORRECTED` | Checked and found wrong; the right value is in `Provenance.corrected`. | **Yes** |
| `PARTIAL` | The source supports part of the entry. | No |
| `CITED_UNCHECKED` | The citation is real; the numbers were not confirmed. | No |
| `RECALL_CONSISTENT` | Audited from recalled knowledge of the literature; citation recognised and the value is consistent with it. Nothing was read. | No |
| `RECALL_CORRECTED` | Recall says a field was wrong; the entry was changed and the old value is kept in `Provenance.corrected` as `<field>_was`. | No |
| `RECALL_UNCERTAIN` | Citation plausible, number not judgeable from memory. | No |
| `UNVERIFIED` | No record at all. | No |
| `UNRECOGNIZED` | The citation could not be placed at all. Ranked *below* `UNVERIFIED`: a specific-looking citation nobody can place is itself a warning sign. | No |

Entries below `CORRECTED` are not discarded — they remain the best available prior and
keep driving the engine exactly as before — but `lookup()` will not hand one back as a
magnitude, and `confidence_weight()` damps how hard it may push the data.

**A tier claim cannot outrun its evidence.** `_parse_entry` downgrades at load time: a
`MEASURED` entry with no script or no source files becomes `UNVERIFIED`; a `VERIFIED`
entry with no quote becomes `CITED_UNCHECKED`.

### The recall band

The four recall tiers record a different kind of evidence from the five above them: a
model that has read a great deal of this literature was asked, entry by entry, whether it
recognises the citation and whether the stored number matches the published or
meta-analytic estimate. That is a judgement from memory. It catches real defects — a
replication failure still carrying its original magnitude, a meta-analytic *k*/*N* pasted
onto a primary study, a scale whose stated response-option count is not the instrument's
— and it is worth having. It is **not** source verification, and the two are kept
structurally apart so they can never be confused:

* `register_recall()` is the only way in, and it refuses any tier outside the recall band
  and any verdict with no note. A data file cannot promote an entry past it.
* `doi`, `url`, `quote` and `verified_on` are empty by construction, and every note is
  prefixed `RECALL, NOT SOURCE-VERIFIED`.
* Every recall tier weighs less than `CITED_UNCHECKED`, and none grants
  `may_set_magnitude`.
* `coverage_summary()["sourced_entries"]` and the percentage in `honesty_notice()` count
  only `VERIFIED`/`CORRECTED`/`PARTIAL`, so a recall pass cannot move the headline number.

The records live in `utils/registry/recall_audit.json` (data only) and are installed by
`load_recall_audit()`. A `RECALL_CORRECTED` entry's **old** value is kept alongside the new
one, so every change a recall pass made to the knowledge base is reversible from the
record.

## Why applicability is a hard guard

Nearly every misuse of a real number has the same shape: a value that is true under the
conditions it was measured under, applied where those conditions do not hold.

- Straight-lining runs **3.3%** in a same-keyed 9-item block and **0.3%** in a
  mixed-keyed one. A 10× error if the keying is ignored.
- A 5-point attitude scale is **flat** (excess kurtosis ≈ −0.5); a 9-point polarised
  political scale is **bimodal** (+0.8) with 3–5% on the midpoint.
- Answering identically across a whole 50-item instrument (**0.1%**) is far rarer than
  within any one block of it.
- Public-goods contributions decay across rounds, so a one-shot pooled mean does not
  describe a repeated game.
- Ultimatum rejection is conditional on offer size, so a flat pooled rate is a misuse
  even when the rate itself is correct.

So each entry declares an `Applicability`, and a lookup whose design does not satisfy it
**declines**. A declined lookup leaves the engine's existing behaviour untouched, which is
explicit and inspectable; a plausible wrong number is neither.

An unknown design dimension never satisfies a stated condition. Silence is not agreement.

## Deriving the measured entries

```bash
python3 -I tools/derive_item_benchmarks.py <data_root> -o simulation_app/utils/registry/item_process.json
```

`<data_root>` holds the downloaded instruments, one directory each. Nothing in the output
file is typed by hand; run the script and it is reproduced. Only key-free statistics are
derived there (marginal shape and row patterns), because α depends on the instrument's
reverse-keying and these mirrors do not all publish a key — the α reference lives in
`utils/reference_profiles.json`, which uses published keys.

**Raw data is never committed.** The openpsychometrics instruments declare no licence
(the site states a research-reuse intent), and they were fetched from a third-party GitHub
mirror verified only by row-count agreement with the source index. Derived summary
statistics only, with the source's own terms quoted in the provenance record.

## Promoting an entry

1. Read the primary source and copy the sentence that states the number.
2. Set `tier` to `verified`, fill `provenance.quote`, `provenance.doi`,
   `provenance.retrieved`.
3. State `effect_scale`. There is no default: τ is reported on *d*, on Fisher's *z* and on
   log odds and is not comparable across them, and I² is a proportion of observed variance
   rather than a magnitude at all.
4. State the `applicability` the source's own design supports — not the one you wish it
   supported.
5. Run `pytest tests/test_registry_layer.py`.

Promoting a recall record is the same procedure: read the source, then replace the
`recall_*` record with a `verified` one. Do not edit the tier in
`recall_audit.json` — `register_recall()` will refuse it.

## The one constant that touches everything

`set_shrinkage()` installs the publication-bias correction applied to **every** literature
effect the simulator produces. It refuses:

- provenance below `MEASURED`/`VERIFIED`/`CORRECTED`;
- a publication-derived record with no verbatim quote;
- a τ on any scale other than *d*.

The recalled-figure route is separate: `set_recalled_shrinkage()` installs a factor
through `register_recall()`, so it is tiered `recall_uncertain`, can never claim a source,
and cannot overwrite a verified record. Since v1.3.0.5 it is how the active default is
installed: `shrinkage_factor()` is **0.60** and `default_tau()` **0.15**, applied to
effects the tool infers (paradigm anchor, literature fallback) and never to an effect you
specify. `coverage_summary()["shrinkage_verified"]` stays False and `honesty_notice()`
says the factor is recalled and unchecked. See `docs/EMPIRICAL_PROVENANCE.md` for the
evidence behind the figure. Promote it by reading the sources and calling `set_shrinkage()`
with a quote.

## Reading a run

`RunLedger` records every registry value that actually changed a number, so
`ledger.notice()` reports the mix for *that dataset* rather than the mix across the whole
table: "7 of 9 calibrations used in this run come from measured data or a checked source;
2 rest on literature assertions that have not been verified."
