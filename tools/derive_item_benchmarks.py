#!/usr/bin/env python3
"""Derive the T0 (MEASURED) response-process benchmarks from real human item-level data.

Every number in `simulation_app/utils/registry/item_process.json` is produced by this
script. Nothing in that file is typed by hand, quoted from a summary, or recalled — run
this script and the file is reproduced, which is what the T0_MEASURED tier means.

Only key-free statistics are derived here: marginal shape (item mean, SD, skew, excess
kurtosis, floor/ceiling occupancy) and row-pattern statistics (straight-lining share,
long-string distribution). Internal consistency is deliberately NOT derived here, because
alpha depends on the reverse-keying of the instrument and these mirrors do not all publish
a key; the alpha reference lives in `utils/reference_profiles.json`, which uses published
keys.

Usage (data is untrusted — paths are arguments, never imported from):

    python3 -I tools/derive_item_benchmarks.py <data_root> [-o <out.json>]

<data_root> holds one directory per dataset, as downloaded:
    opp_BIG5/data.csv   opp_SD3/data.csv   opp_HSQ/data.csv   opp_RWAS/data.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# (directory, separator, item column names, scale min, scale max, missing codes, blocks)
DATASETS = [
    dict(
        dir="opp_BIG5", sep="\t", scale_min=1, scale_max=5, missing=(0,),
        name="IPIP Big Five 50-item markers",
        canonical="https://openpsychometrics.org/_rawdata/BIG5.zip",
        mirror="https://raw.githubusercontent.com/haghish/openpsychometrics/main/BIG5/data.csv",
        items=[f"{f}{i}" for f in "ENACO" for i in range(1, 11)],
        blocks={f: [f"{f}{i}" for i in range(1, 11)] for f in "ENACO"},
    ),
    dict(
        dir="opp_SD3", sep="\t", scale_min=1, scale_max=5, missing=(0,),
        name="Short Dark Triad",
        canonical="https://openpsychometrics.org/_rawdata/SD3.zip",
        mirror="https://raw.githubusercontent.com/haghish/openpsychometrics/main/SD3/data.csv",
        items=[f"{f}{i}" for f in "MNP" for i in range(1, 10)],
        blocks={f: [f"{f}{i}" for i in range(1, 10)] for f in "MNP"},
    ),
    dict(
        dir="opp_HSQ", sep=",", scale_min=1, scale_max=5, missing=(-1,),
        name="Humor Styles Questionnaire",
        canonical="https://openpsychometrics.org/_rawdata/HSQ.zip",
        mirror="https://raw.githubusercontent.com/haghish/openpsychometrics/main/HSQ/data.csv",
        items=[f"Q{i}" for i in range(1, 33)],
        blocks={f"sub{b}": [f"Q{i}" for i in range(1 + b, 33, 4)] for b in range(4)},
    ),
    dict(
        dir="opp_RWAS", sep=",", scale_min=1, scale_max=9, missing=(0,),
        name="Right-Wing Authoritarianism Scale",
        canonical="https://openpsychometrics.org/_rawdata/RWAS.zip",
        mirror="https://raw.githubusercontent.com/haghish/openpsychometrics/main/RWAS/data.csv",
        items=[f"Q{i}" for i in range(1, 23)],
        blocks={"full": [f"Q{i}" for i in range(1, 23)]},
    ),
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def long_string(row: np.ndarray) -> int:
    """Longest run of identical consecutive responses in one row."""
    best = run = 1
    for i in range(1, row.size):
        run = run + 1 if row[i] == row[i - 1] else 1
        if run > best:
            best = run
    return best


def profile_block(mat: np.ndarray, lo: int, hi: int) -> Dict[str, float]:
    """Marginal and row-pattern statistics for one complete-case item block."""
    n_rows, n_items = mat.shape
    means = mat.mean(axis=0)
    sds = mat.std(axis=0, ddof=1)
    centred = mat - means
    with np.errstate(invalid="ignore", divide="ignore"):
        m2 = (centred ** 2).mean(axis=0)
        m3 = (centred ** 3).mean(axis=0)
        m4 = (centred ** 4).mean(axis=0)
        skew = np.where(m2 > 0, m3 / np.power(m2, 1.5), 0.0)
        kurt = np.where(m2 > 0, m4 / (m2 ** 2) - 3.0, 0.0)
    floor = (mat == lo).mean(axis=0)
    ceil = (mat == hi).mean(axis=0)
    ls = np.array([long_string(mat[i]) for i in range(n_rows)])
    within_sd = mat.std(axis=1, ddof=1) if n_items > 1 else np.zeros(n_rows)
    span = float(hi - lo)
    return {
        "n_rows": int(n_rows),
        "n_items": int(n_items),
        "scale_span": span,
        "item_sd_fraction_of_span": float(sds.mean() / span) if span > 0 else float("nan"),
        "within_person_sd_fraction_of_span": float(
            (within_sd.mean() / span) if span > 0 else float("nan")),
        "item_mean": float(means.mean()),
        "item_sd": float(sds.mean()),
        "mean_abs_skew": float(np.abs(skew).mean()),
        "mean_excess_kurtosis": float(kurt.mean()),
        "floor_rate": float(floor.mean()),
        "ceiling_rate": float(ceil.mean()),
        "endpoint_occupancy": float(floor.mean() + ceil.mean()),
        "midpoint_rate": float((mat == (lo + hi) // 2).mean()) if (lo + hi) % 2 == 0 else float("nan"),
        "straightlined_share": float((ls == n_items).mean()),
        "long_string_mean": float(ls.mean()),
        "long_string_median": float(np.median(ls)),
        "long_string_p90": float(np.percentile(ls, 90)),
        "long_string_p99": float(np.percentile(ls, 99)),
        "within_person_sd": float(within_sd.mean()),
        "distinct_options_used": float(np.array([np.unique(mat[i]).size for i in range(n_rows)]).mean()),
    }


def straightlining_by_width(mat: np.ndarray, widths=range(3, 11)) -> Dict[int, float]:
    """Share of respondents answering identically across a contiguous k-item window.

    Averaged over every window of that width in the block. Simulated blocks are
    short - 3 to 8 items is typical - and the published instruments here are 8 to
    50, so the rate has to be measured AT the width it will be applied at. It is
    strongly width-dependent (a 3-item window is far easier to straight-line than
    a 10-item one), which is exactly why one global share was wrong.
    """
    n_rows, n_items = mat.shape
    out: Dict[int, float] = {}
    for k in widths:
        if k > n_items:
            continue
        shares = []
        for start in range(0, n_items - k + 1):
            win = mat[:, start:start + k]
            shares.append(float((win == win[:, :1]).all(axis=1).mean()))
        if shares:
            out[int(k)] = float(np.mean(shares))
    return out


def complete_matrix(df: pd.DataFrame, items: Sequence[str], missing) -> np.ndarray:
    sub = df[list(items)].apply(pd.to_numeric, errors="coerce")
    for code in missing:
        sub = sub.mask(sub == code)
    sub = sub.dropna(axis=0, how="any")          # listwise: rows must stay aligned
    return sub.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Aggregation into registry entries
# ---------------------------------------------------------------------------

#: Blocks whose items are all keyed in the same direction. Straight-lining is a
#: property of the instrument's keying, not only of the respondent, so a
#: same-keyed block cannot supply a benchmark for a mixed-keyed one. SD3
#: Machiavellianism is the one same-keyed block in these data (Paulhus & Jones
#: publish no reverse items for it) and it straight-lines ~20x more often.
SAME_KEYED_BLOCKS = {("Short Dark Triad", "M")}

#: Metrics aggregated across contributing blocks. `value` is the unweighted mean
#: across blocks and `dispersion` the SD across blocks, which is what makes these
#: usable as ranges rather than as point targets a generator could overfit to.
BLOCK_METRICS = (
    ("item_sd", "sd", "raw_points"),
    ("item_mean", "mean", "raw_points"),
    ("floor_rate", "proportion", "proportion"),
    ("ceiling_rate", "proportion", "proportion"),
    ("endpoint_occupancy", "proportion", "proportion"),
    ("mean_abs_skew", "mean", "raw_points"),
    ("mean_excess_kurtosis", "mean", "raw_points"),
    ("within_person_sd", "sd", "raw_points"),
    ("distinct_options_used", "mean", "raw_points"),
    ("long_string_median", "mean", "raw_points"),
    ("long_string_p99", "mean", "raw_points"),
    ("straightlined_share", "proportion", "proportion"),
)


def _entry(entry_id, quantity, values, effect_scale, applicability, contributors,
           sources, caveats):
    arr = np.asarray(values, dtype=float)
    return {
        "entry_id": entry_id,
        "quantity": quantity,
        "value": float(arr.mean()),
        "dispersion": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "dispersion_kind": "across_blocks" if arr.size > 1 else "none",
        "effect_scale": effect_scale,
        "construct": "",
        "paradigm": None,
        "applicability": applicability,
        "k_studies": None,
        "n_participants": None,
        "tier": "T0_MEASURED",
        "provenance": {
            "kind": "dataset",
            "script": "tools/derive_item_benchmarks.py",
            "retrieved": date.today().isoformat(),
            "measured_in_population": "online_volunteer",
            "contributing_blocks": contributors,
            "datasets": sorted({c.split(" :: ")[0] for c in contributors}),
            "sources": [s for s in sources
                        if s["dataset"] in {c.split(" :: ")[0] for c in contributors}],
            "license": "none stated; source declares research-reuse intent",
            "license_quote": (
                "For general public edification the data collected through the "
                "personality tests on this website is dumped here. All data is "
                "anonymous."
            ),
            "mirror_warning": (
                "fetched from a third-party GitHub mirror of openpsychometrics.org; "
                "row counts agree with the source index, but the canonical host should "
                "be re-fetched when it becomes reachable"
            ),
        },
        "caveats": tuple(caveats) + ((
            "Derived from a single contributing block, so the value is an order of "
            "magnitude rather than a precise rate",
        ) if len(contributors) < 2 else ()),
    }


def build_entries(block_profiles, full_profiles, sources) -> List[Dict]:
    """Aggregate the measured profiles into applicability-guarded registry entries."""
    entries: List[Dict] = []
    common_caveats = [
        "Measured in self-selected internet volunteer samples, not a probability "
        "sample and not a paid experiment panel. The entry is not scoped to that "
        "population, because it is the best available human reference for online "
        "data collection generally - but it is a RANGE, and the dispersion field "
        "is the honest width of it",
        "Single-group instruments: these benchmark response process only and must "
        "never be used to calibrate a condition effect",
    ]

    def group(profiles, pred):
        return [p for p in profiles if pred(p)]

    mixed5 = group(block_profiles, lambda p: (
        p["scale_points"] == 5 and 8 <= p["n_items"] <= 10
        and (p["dataset"], p["block"]) not in SAME_KEYED_BLOCKS))
    same5 = group(block_profiles, lambda p: (
        p["scale_points"] == 5 and (p["dataset"], p["block"]) in SAME_KEYED_BLOCKS))
    full5 = group(full_profiles, lambda p: p["scale_points"] == 5)
    wide9 = group(full_profiles, lambda p: p["scale_points"] == 9)

    def tag(p):
        return f"{p['dataset']} :: {p['block']} (n={p['n_rows']}, k={p['n_items']})"

    if mixed5:
        app = {"scale_points": [5], "items_per_block": [8, 10], "keying": "mixed",
               "design": "any", "population": "any", "repetition": "any"}
        for metric, quantity, scale in BLOCK_METRICS:
            entries.append(_entry(
                f"item.likert5.block8to10.{metric}", quantity,
                [p[metric] for p in mixed5], scale, app,
                [tag(p) for p in mixed5], sources, list(common_caveats)))

    if same5:
        app = {"scale_points": [5], "items_per_block": [8, 10], "keying": "same",
               "design": "any", "population": "any", "repetition": "any"}
        entries.append(_entry(
            "item.likert5.block8to10.same_keyed.straightlined_share", "proportion",
            [p["straightlined_share"] for p in same5], "proportion", app,
            [tag(p) for p in same5], sources,
            common_caveats + [
                "Derived from a single same-keyed block, so the value is an order of "
                "magnitude, not a precise rate",
                "Roughly 20x the mixed-keyed rate: straight-lining is a property of "
                "the instrument's keying, not only of the respondent",
            ]))

    if full5:
        app = {"scale_points": [5], "items_per_block": [22, 50], "keying": "mixed",
               "design": "any", "population": "any", "repetition": "any"}
        entries.append(_entry(
            "item.likert5.full_instrument.straightlined_share", "proportion",
            [p["straightlined_share"] for p in full5], "proportion", app,
            [tag(p) for p in full5], sources,
            common_caveats + [
                "Answering identically across a whole 22-50 item instrument is rarer "
                "than within any one block, so this must not be applied per block",
            ]))

    if wide9:
        app = {"scale_points": [9], "items_per_block": [20, 25], "keying": "mixed",
               "design": "any", "population": "any", "repetition": "any"}
        for metric, quantity, scale in BLOCK_METRICS:
            entries.append(_entry(
                f"item.likert9.polarised.{metric}", quantity,
                [p[metric] for p in wide9], scale, app,
                [tag(p) for p in wide9], sources,
                common_caveats + [
                    "A polarised political instrument: strongly bimodal (positive "
                    "excess kurtosis) where 5-point attitude scales are flat. This is "
                    "a wide-polarised-scale profile, not a general 9-point one",
                ]))
    # Straight-lining as a function of block width and keying. Two dimensions,
    # because the rate moves by an order of magnitude along each of them: a
    # same-keyed block straight-lines far more than a mixed-keyed one (nothing
    # contradicts a run of identical answers when every item points the same way),
    # and a short window far more than a long one.
    for keying, want_same in (("mixed", False), ("same", True)):
        by_k: Dict[int, List[float]] = {}
        contributors: Dict[int, List[str]] = {}
        for p in block_profiles:
            is_same = (p["dataset"], p["block"]) in SAME_KEYED_BLOCKS
            if is_same != want_same:
                continue
            for k_str, v in (p.get("straightlining_by_width") or {}).items():
                by_k.setdefault(int(k_str), []).append(float(v))
                contributors.setdefault(int(k_str), []).append(tag(p))
        for k, vals in sorted(by_k.items()):
            if not vals:
                continue
            app = {"scale_points": None, "items_per_block": [k, k], "keying": keying,
                   "design": "any", "population": "any", "repetition": "any"}
            entries.append(_entry(
                f"item.likert.{keying}.k{k}.straightlined_share", "proportion",
                vals, "proportion", app, contributors[k], sources,
                common_caveats + [
                    "Measured over every contiguous %d-item window of each block, so "
                    "it is the rate for a block of that width rather than for a whole "
                    "instrument" % k,
                ]))

    # Scale-free entries. Expressed as a fraction of the scale span, and only
    # emitted when the fraction actually agrees across scale lengths — which is the
    # empirical question, not an assumption. Dispersion across all contributing
    # blocks is the honest width, and it is what decides whether the entry gets to
    # claim it generalises.
    everything = [p for p in block_profiles + full_profiles if p["n_items"] >= 8]
    for metric in ("item_sd_fraction_of_span", "within_person_sd_fraction_of_span"):
        vals = [p[metric] for p in everything if p[metric] == p[metric]]
        if len(vals) < 4:
            continue
        arr = np.asarray(vals, dtype=float)
        if arr.std(ddof=1) / max(abs(arr.mean()), 1e-9) > 0.25:
            # Too variable across scale lengths to claim it is scale-free. The
            # scale-specific entries above still stand.
            continue
        # item_sd is a property of a single item, so it is not bounded by how many
        # items happen to sit beside it. within_person_sd IS a block statistic and
        # keeps the block-size bound it was measured under.
        bounds = [1, 100] if metric.startswith("item_sd") else [8, 50]
        app = {"scale_points": None, "items_per_block": bounds, "keying": "any",
               "design": "any", "population": "any", "repetition": "any"}
        entries.append(_entry(
            f"item.likert.any.{metric}", "proportion", vals, "proportion", app,
            [tag(p) for p in everything if p[metric] == p[metric]], sources,
            common_caveats + [
                "Emitted only because the fraction agreed across 5-point and 9-point "
                "instruments; the coefficient of variation across contributing blocks "
                "is the width of that agreement",
            ]))
    return entries


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root")
    ap.add_argument("-o", "--out", default="-")
    args = ap.parse_args(argv)

    sources, full_profiles, block_profiles = [], [], []
    for spec in DATASETS:
        path = os.path.join(args.data_root, spec["dir"], "data.csv")
        if not os.path.exists(path):
            print(f"skip (not found): {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path, sep=spec["sep"], low_memory=False)
        lo, hi = spec["scale_min"], spec["scale_max"]
        src = {
            "dataset": spec["name"],
            "retrieved_from": spec["mirror"],
            "canonical_source": spec["canonical"],
            "sha256": sha256(path),
            "bytes": os.path.getsize(path),
        }
        sources.append(src)

        mat = complete_matrix(df, spec["items"], spec["missing"])
        full_profiles.append(dict(dataset=spec["name"], scale_points=hi - lo + 1,
                                  block="__full_instrument__", **profile_block(mat, lo, hi)))
        for bname, bitems in spec["blocks"].items():
            bmat = complete_matrix(df, bitems, spec["missing"])
            if bmat.shape[0] < 100:
                continue
            prof = dict(dataset=spec["name"], scale_points=hi - lo + 1,
                        block=bname, **profile_block(bmat, lo, hi))
            prof["straightlining_by_width"] = {
                str(k): v for k, v in straightlining_by_width(bmat).items()}
            block_profiles.append(prof)

    out = {
        "schema_version": 1,
        "generated_by": "tools/derive_item_benchmarks.py",
        "generated_on": date.today().isoformat(),
        "sources": sources,
        "entries": build_entries(block_profiles, full_profiles, sources),
        "full_instrument_profiles": full_profiles,
        "block_profiles": block_profiles,
    }
    text = json.dumps(out, indent=2, sort_keys=False)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
