#!/usr/bin/env python3
"""Derive MEASURED evidence entries (v1.3.0.6) from open datasets we actually hold.

Everything in `simulation_app/utils/registry/evidence_v1306.json` is produced by this
script from the files named in `DATASETS`. Nothing is typed by hand or recalled. Each
entry records the script, the source URL, the retrieval date and the SHA-256 of every
file it draws on, so `tests/test_evidence_v1306.py` can check the hashes against the
committed aggregates and anyone holding the files can re-run it.

Four families:
  * Likert response process at 5 and 7 response options (open online personality
    instruments mirrored on GitHub): item SD as a fraction of span, endpoint share,
    midpoint share, direction-aligned inter-item r and alpha by block width,
    straight-lining by block width and keying.
  * Per-item response time (MACH-IV administers one item per page and logs it).
  * Economic games: ultimatum responder acceptance by offer share (n=111 students),
    a binary dictator-game choice set (Bruhin, Fehr & Schunk replication data, n=174),
    n-person prisoner's dilemma cooperation (Fox & Guyer 1978, 20 groups).
  * Two small randomised experiments shipped in R packages (Garcia et al. 2010;
    Tal-Or et al. 2010): observed condition effects as Cohen's d.

Keying is NOT taken from a recalled key. For single-construct blocks the sign of an
item's loading on the first principal component of the block's correlation matrix is
the empirical key (every item in a same-keyed block loads positively). That is a
measurement we can reproduce; a published key is not.

Usage (data is untrusted - paths are arguments, never imported from):

    python3 tools/derive_evidence_benchmarks.py <data_root> \
        -o simulation_app/utils/registry/evidence_v1306.json

<data_root> layout, as downloaded:
    opp_<NAME>/data.csv                       openpsychometrics mirror files
    ug_social_anxiety/database_UG_social_anxiety.sav
    bruhin_binary_dictator/choices.csv
    rdatasets/{Guyer,Garcia,Tal_Or}.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

RETRIEVED = "2026-10-08"
SCRIPT = "tools/derive_evidence_benchmarks.py"
OPP_MIRROR = "https://raw.githubusercontent.com/haghish/openpsychometrics/main"
OPP_CANON = "https://openpsychometrics.org/_rawdata"
OPP_LICENSE = ("none stated; the mirror states 'solely for research purpose, allowing "
               "students to access the data for statistical courses'")
OPP_QUOTE = ("For general public edification the data collected through the personality "
             "tests on this website is dumped here. All data is anonymous.")
RDAT = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv"

# name, mirror dir, canonical zip, file path under data_root, sep, scale (lo, hi)
OPP = {
    "CFCS": ("CFCS", "CFCS.zip", "opp_CFCS/data.csv", "\t"),
    "MACH-IV": ("MACH_data", "MACH_data.zip", "opp_MACH_data/data.csv", "\t"),
    "PWE": ("PWE_data", "PWE_data.zip", "opp_PWE_data/data.csv", "\t"),
    "NPAS": ("NPAS-data-16December2018", "NPAS-data-16December2018.zip",
             "opp_NPAS-data-16December2018/data.csv", "\t"),
    "HSNS": ("HSNS%2BDD", "HSNS+DD.zip", "opp_HSNS+DD/data.csv", "\t"),
    "RIASEC": ("RIASEC_data12Dec2018", "RIASEC_data12Dec2018.zip",
               "opp_RIASEC_data12Dec2018/data.csv", "\t"),
    "HEXACO": ("HEXACO", "HEXACO.zip", "opp_HEXACO/data.csv", "\t"),
}

#: (instrument, scale_min, scale_max, [(block name, [columns])], tipi_only)
def _likert_specs(frames: Dict[str, pd.DataFrame]) -> List[dict]:
    specs = [
        dict(inst="CFCS", lo=1, hi=5, blocks={"CFCS": [f"Q{i}" for i in range(1, 13)]}),
        dict(inst="MACH-IV", lo=1, hi=5, blocks={"MACH": [f"Q{i}A" for i in range(1, 21)]}),
        dict(inst="PWE", lo=1, hi=5, blocks={"PWE": [f"Q{i}A" for i in range(1, 20)]}),
        dict(inst="NPAS", lo=1, hi=5, blocks={"NPAS": [f"Q{i}" for i in range(1, 27)]}),
        dict(inst="HSNS", lo=1, hi=5, blocks={"HSNS": [f"HSNS{i}" for i in range(1, 11)]}),
        dict(inst="RIASEC", lo=1, hi=5,
             blocks={f: [f"{f}{i}" for i in range(1, 9)] for f in "RIASEC"}),
    ]
    hexaco = frames.get("HEXACO")
    if hexaco is not None:
        facets: Dict[str, List[str]] = {}
        for c in list(hexaco.columns)[:240]:
            m = re.match(r"^([A-Za-z]+?)(\d+)$", c)
            if m and m.group(1) != "V":
                facets.setdefault(m.group(1), []).append(c)
        facets = {k: sorted(v, key=lambda s: int(re.sub(r"\D", "", s)))
                  for k, v in facets.items() if len(v) == 10}
        specs.append(dict(inst="HEXACO", lo=1, hi=7, blocks=facets))
    return specs


# --------------------------------------------------------------------------- helpers
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def complete(df: pd.DataFrame, cols: Sequence[str], lo: int, hi: int) -> np.ndarray:
    sub = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    sub = sub.where((sub >= lo) & (sub <= hi))          # 0 / out-of-range = missing
    return sub.dropna(axis=0, how="any").to_numpy(dtype=float)


def empirical_key(mat: np.ndarray) -> np.ndarray:
    """+1/-1 per item: sign of the loading on the first principal component,
    oriented so that the majority is +1. Valid for single-construct blocks."""
    corr = np.corrcoef(mat, rowvar=False)
    w, v = np.linalg.eigh(corr)
    first = v[:, int(np.argmax(w))]
    key = np.where(first >= 0, 1.0, -1.0)
    if key.sum() < 0:
        key = -key
    return key


def window_stats(mat: np.ndarray, key: np.ndarray, widths=range(3, 9)) -> Dict[str, Dict[int, dict]]:
    """Straight-line share and aligned reliability over every contiguous k-window,
    split by whether the window is same- or mixed-keyed (empirical key)."""
    n, p = mat.shape
    aligned = np.where(key > 0, mat, (mat.max() + mat.min()) - mat)
    out: Dict[str, Dict[int, dict]] = {"same": {}, "mixed": {}, "any": {}}
    for k in widths:
        if k > p:
            continue
        acc: Dict[str, List[tuple]] = {"same": [], "mixed": [], "any": []}
        for s in range(0, p - k + 1):
            win = mat[:, s:s + k]
            sl = float((win == win[:, :1]).all(axis=1).mean())
            wa = aligned[:, s:s + k]
            c = np.corrcoef(wa, rowvar=False)
            rbar = float((c.sum() - k) / (k * (k - 1)))
            alpha = k * rbar / (1 + (k - 1) * rbar) if (1 + (k - 1) * rbar) > 0 else float("nan")
            kind = "same" if abs(key[s:s + k].sum()) == k else "mixed"
            acc[kind].append((sl, rbar, alpha))
            acc["any"].append((sl, rbar, alpha))
        for kind, rows in acc.items():
            if rows:
                a = np.asarray(rows)
                out[kind][k] = dict(straightlined=float(a[:, 0].mean()),
                                    rbar=float(a[:, 1].mean()),
                                    alpha=float(np.nanmean(a[:, 2])), windows=len(rows))
    return out


def block_profile(mat: np.ndarray, lo: int, hi: int, keyed: bool) -> dict:
    n, p = mat.shape
    span = float(hi - lo)
    sds = mat.std(axis=0, ddof=1)
    cen = mat - mat.mean(axis=0)
    m2 = (cen ** 2).mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = np.where(m2 > 0, (cen ** 3).mean(axis=0) / m2 ** 1.5, 0.0)
        kurt = np.where(m2 > 0, (cen ** 4).mean(axis=0) / m2 ** 2 - 3.0, 0.0)
    mid = (lo + hi) / 2.0
    prof = {
        "n_rows": int(n), "n_items": int(p), "scale_points": int(hi - lo + 1),
        "item_sd_fraction_of_span": float(sds.mean() / span),
        "floor_rate": float((mat == lo).mean()), "ceiling_rate": float((mat == hi).mean()),
        "midpoint_rate": float((mat == mid).mean()),
        "mean_abs_skew": float(np.abs(skew).mean()),
        "mean_excess_kurtosis": float(kurt.mean()),
        "straightlined_share": float((mat == mat[:, :1]).all(axis=1).mean()),
        "within_person_sd_fraction_of_span": float(mat.std(axis=1, ddof=1).mean() / span),
    }
    prof["endpoint_occupancy"] = prof["floor_rate"] + prof["ceiling_rate"]
    if keyed:
        key = empirical_key(mat)
        prof["n_reverse_items_empirical"] = int((key < 0).sum())
        prof["block_keying"] = "same" if (key > 0).all() else "mixed"
        prof["windows"] = {kind: {str(k): v for k, v in d.items()}
                           for kind, d in window_stats(mat, key).items()}
        al = window_stats(mat, key, widths=[p])["any"].get(p)
        if al:
            prof["aligned_mean_interitem_r"] = al["rbar"]
            prof["aligned_alpha"] = al["alpha"]
    return prof


# --------------------------------------------------------------------------- entries
def make_entry(entry_id, quantity, values, scale, applicability, contributors, sources,
               caveats, kind="dataset", measured_in="online_volunteer", license_="",
               license_quote="", n_participants=None, dispersion_kind="across_blocks"):
    arr = np.asarray(values, dtype=float)
    caveats = list(caveats)
    if len(contributors) == 1:
        caveats.append("Derived from a single contributing block or study, so the value is "
                       "an order of magnitude rather than a precise rate")
    used = {c.split(" :: ")[0] for c in contributors}
    return {
        "entry_id": entry_id, "quantity": quantity, "value": float(arr.mean()),
        "dispersion": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "dispersion_kind": dispersion_kind if arr.size > 1 else "none",
        "effect_scale": scale, "construct": "", "paradigm": None,
        "applicability": applicability, "k_studies": None,
        "n_participants": n_participants, "tier": "T0_MEASURED",
        "provenance": {
            "kind": kind, "script": SCRIPT, "retrieved": RETRIEVED,
            "measured_in_population": measured_in,
            "contributing_blocks": contributors,
            "datasets": sorted(used),
            "sources": [s for s in sources if s["dataset"] in used],
            "license": license_, "license_quote": license_quote,
            "consumed_by_engine": False,
        },
        "caveats": caveats,
    }


def single(entry_id, quantity, value, scale, applicability, dataset, sources, caveats,
           n, license_, license_quote="", measured_in="student_sample", dispersion=None,
           dispersion_kind="none"):
    e = make_entry(entry_id, quantity, [value], scale, applicability, [f"{dataset} :: all"],
                   sources, caveats, license_=license_, license_quote=license_quote,
                   measured_in=measured_in, n_participants=n)
    if dispersion is not None:
        e["dispersion"] = float(dispersion)
        e["dispersion_kind"] = dispersion_kind
    return e


APP_ANY = dict(design="any", population="any", repetition="any")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp)


def d_se(d: float, na: int, nb: int) -> float:
    return float(np.sqrt((na + nb) / (na * nb) + d * d / (2 * (na + nb))))


# --------------------------------------------------------------------------- main
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root")
    ap.add_argument("-o", "--out", default="-")
    args = ap.parse_args(argv)
    root = args.data_root

    sources: List[dict] = []
    frames: Dict[str, pd.DataFrame] = {}
    for inst, (mdir, zipname, rel, sep) in OPP.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"skip (not found): {path}", file=sys.stderr)
            continue
        frames[inst] = pd.read_csv(path, sep=sep, low_memory=False)
        sources.append({"dataset": inst, "retrieved_from": f"{OPP_MIRROR}/{mdir}/data.csv",
                        "canonical_source": f"{OPP_CANON}/{zipname}",
                        "sha256": sha256(path), "bytes": os.path.getsize(path),
                        "n_rows_file": int(len(frames[inst]))})

    entries: List[dict] = []
    profiles: List[dict] = []
    common = [
        "Self-selected internet volunteers who took an online personality test, not a "
        "probability sample and not a paid experiment panel",
        "Single-group instruments: these benchmark response process only and must never "
        "calibrate a condition effect",
        "Items are 'scale blocks' (all items of one instrument or facet). Straight-lining "
        "is the share answering every item of a contiguous k-item window identically; "
        "column order is the instrument's item numbering, not necessarily page order",
        "Item keying is empirical (sign of the first-principal-component loading), not a "
        "published key",
    ]

    # ---------------------------------------------------------------- Likert blocks
    for spec in _likert_specs(frames):
        df = frames.get(spec["inst"])
        if df is None:
            continue
        for bname, cols in spec["blocks"].items():
            mat = complete(df, cols, spec["lo"], spec["hi"])
            if mat.shape[0] < 500:
                continue
            p = block_profile(mat, spec["lo"], spec["hi"], keyed=True)
            p.update(dataset=spec["inst"], block=bname)
            profiles.append(p)
    # TIPI: 7-point, 10 items, one follow-up survey attached to several tests.
    for inst in ("MACH-IV", "NPAS", "PWE", "RIASEC"):
        df = frames.get(inst)
        if df is None:
            continue
        mat = complete(df, [f"TIPI{i}" for i in range(1, 11)], 1, 7)
        if mat.shape[0] >= 500:
            p = block_profile(mat, 1, 7, keyed=False)
            p.update(dataset=f"TIPI after {inst}", block="TIPI", tipi=True)
            profiles.append(p)
    # TIPI blocks reuse the file of the host test, so their source is that file.
    tag = lambda p: f"{p['dataset']} :: {p['block']} (n={p['n_rows']}, k={p['n_items']})"  # noqa: E731
    host = lambda p: (p["dataset"].replace("TIPI after ", "") if p.get("tipi") else p["dataset"])  # noqa: E731

    def contrib(ps):
        return [f"{host(p)} :: {p['block']} (n={p['n_rows']}, k={p['n_items']})"
                + (" [TIPI]" if p.get("tipi") else "") for p in ps]

    for sp in (5, 7):
        grp = [p for p in profiles if p["scale_points"] == sp]
        if not grp:
            continue
        ks = sorted({p["n_items"] for p in grp})
        app = dict(scale_points=[sp], items_per_block=[ks[0], ks[-1]], keying="any", **APP_ANY)
        for metric, quantity, scale in (
                ("item_sd_fraction_of_span", "proportion", "proportion"),
                ("floor_rate", "proportion", "proportion"),
                ("ceiling_rate", "proportion", "proportion"),
                ("endpoint_occupancy", "proportion", "proportion"),
                ("midpoint_rate", "proportion", "proportion"),
                ("mean_abs_skew", "mean", "raw_points"),
                ("mean_excess_kurtosis", "mean", "raw_points"),
                ("within_person_sd_fraction_of_span", "proportion", "proportion")):
            vals = [p[metric] for p in grp]
            entries.append(make_entry(
                f"evidence.likert{sp}.blocks.{metric}", quantity, vals, scale, app,
                contrib(grp), sources, common, license_=OPP_LICENSE,
                license_quote=OPP_QUOTE))
        keyed = [p for p in grp if "windows" in p]
        # whole-block straight-lining and reliability, by block keying
        for kind in ("same", "mixed"):
            sub = [p for p in keyed if p["block_keying"] == kind]
            if not sub:
                continue
            kk = sorted({p["n_items"] for p in sub})
            a2 = dict(scale_points=[sp], items_per_block=[kk[0], kk[-1]], keying=kind, **APP_ANY)
            entries.append(make_entry(
                f"evidence.likert{sp}.{kind}.whole_block.straightlined_share", "proportion",
                [p["straightlined_share"] for p in sub], "proportion", a2, contrib(sub),
                sources, common, license_=OPP_LICENSE, license_quote=OPP_QUOTE))
        for k in range(3, 9):
            for kind in ("same", "mixed"):
                vals, who = [], []
                for p in keyed:
                    w = p["windows"][kind].get(str(k))
                    if w:
                        vals.append(w["straightlined"])
                        who.append(p)
                if vals:
                    a2 = dict(scale_points=[sp], items_per_block=[k, k], keying=kind, **APP_ANY)
                    entries.append(make_entry(
                        f"evidence.likert{sp}.{kind}.k{k}.straightlined_share", "proportion",
                        vals, "proportion", a2, contrib(who), sources,
                        common + ["Mean over every contiguous k-item window of each block"],
                        license_=OPP_LICENSE, license_quote=OPP_QUOTE))
            vals, who = [], []
            for p in keyed:
                w = p["windows"]["any"].get(str(k))
                if w:
                    vals.append(w)
                    who.append(p)
            if vals:
                a2 = dict(scale_points=[sp], items_per_block=[k, k], keying="any", **APP_ANY)
                for field, nm in (("rbar", "aligned_mean_interitem_r"), ("alpha", "aligned_alpha")):
                    entries.append(make_entry(
                        f"evidence.likert{sp}.k{k}.{nm}", "mean",
                        [v[field] for v in vals], "raw_points" if field == "rbar" else "raw_points",
                        a2, contrib(who), sources,
                        common + ["Direction-aligned using the empirical key; standardised "
                                  "alpha from the mean inter-item r of each k-item window"],
                        license_=OPP_LICENSE, license_quote=OPP_QUOTE))

    # ---------------------------------------------------------------- response time
    mach = frames.get("MACH-IV")
    if mach is not None:
        t = mach[[f"Q{i}E" for i in range(1, 21)]].apply(pd.to_numeric, errors="coerce")
        ans = mach[[f"Q{i}A" for i in range(1, 21)]].apply(pd.to_numeric, errors="coerce")
        ok = (ans.values >= 1).all(axis=1) & np.isfinite(t.values).all(axis=1) & (t.values > 0).all(axis=1)
        tt = t.values[ok] / 1000.0
        per_person = np.median(tt, axis=1)
        n = int(ok.sum())
        rt_app = dict(scale_points=[5], items_per_block=[20, 20], keying="any", **APP_ANY)
        rt_cav = ["One item per page, random order, agree/disagree statements of 8-25 words, "
                  "online volunteers; includes reading time",
                  "Per-item time is the logged page time in ms (converted to seconds)"]
        for nm, val in (
                ("item_seconds_median", float(np.median(tt))),
                ("item_seconds_p10", float(np.percentile(tt, 10))),
                ("item_seconds_p90", float(np.percentile(tt, 90))),
                ("person_median_seconds_p10", float(np.percentile(per_person, 10))),
                ("person_median_seconds_p50", float(np.percentile(per_person, 50))),
                ("person_median_seconds_p90", float(np.percentile(per_person, 90))),
                ("share_items_under_1s", float((tt < 1.0).mean())),
                ("share_persons_median_under_2s", float((per_person < 2.0).mean())),
                ("share_persons_median_under_1s", float((per_person < 1.0).mean())),
                ("first_item_over_later_items_ratio",
                 float(np.median(tt[:, 0]) / np.median(tt[:, 1:])))):
            entries.append(single(
                f"evidence.rt.mach5pt.{nm}", "mean", val,
                "raw_points" if nm.endswith(("median", "p10", "p90")) or "seconds" in nm else "proportion",
                rt_app, "MACH-IV", sources, rt_cav, n, OPP_LICENSE, OPP_QUOTE,
                measured_in="online_volunteer"))
        # The 'first item' ratio is about column Q1, not page 1; keep honest naming.
        entries = [e for e in entries if e["entry_id"] != "evidence.rt.mach5pt.first_item_over_later_items_ratio"]
    for inst, ni in (("NPAS", 26), ("RIASEC", 48), ("MACH-IV", 20)):
        df = frames.get(inst)
        if df is None or "testelapse" not in df:
            continue
        te = pd.to_numeric(df["testelapse"], errors="coerce")
        te = te[(te > 0) & (te < 7200)]
        entries.append(single(
            f"evidence.rt.{inst.lower().replace('-', '')}.test_seconds_per_item_median",
            "mean", float(te.median() / ni), "raw_points",
            dict(scale_points=[5], items_per_block=[ni, ni], keying="any", **APP_ANY),
            inst, sources,
            ["testelapse is the whole test page in seconds, divided by the item count; "
             "values above 7200 s were dropped as abandoned sessions"],
            int(len(te)), OPP_LICENSE, OPP_QUOTE, measured_in="online_volunteer"))

    # ---------------------------------------------------------------- games
    p_ug = os.path.join(root, "ug_social_anxiety", "database_UG_social_anxiety.sav")
    if os.path.exists(p_ug):
        import pyreadstat
        ug, _ = pyreadstat.read_sav(p_ug)
        src_ug = {"dataset": "UG social anxiety",
                  "retrieved_from": "https://raw.githubusercontent.com/acbica/UltimatumGame_SocialAnxiety/master/database_UG_social_anxiety.sav",
                  "canonical_source": "https://github.com/acbica/UltimatumGame_SocialAnxiety",
                  "sha256": sha256(p_ug), "bytes": os.path.getsize(p_ug),
                  "n_rows_file": int(len(ug))}
        sources.append(src_ug)
        # Column names say <proposer>_<responder> percent; the responder's share is the
        # second number, and acceptance falls monotonically across them (checked in tests).
        for resp in (50, 45, 40, 25, 20, 15, 10, 5):
            col = f"Acc_rates_{100 - resp}_{resp}splits"
            v = ug[col].dropna() / 100.0
            entries.append(single(
                f"evidence.game.ug_responder.accept_at_share{resp:02d}", "proportion",
                float(v.mean()), "proportion", dict(scale_points=None, **APP_ANY),
                "UG social anxiety", sources,
                ["University students (mean age %.1f) in a social-anxiety study; each "
                 "participant's rate is over a handful of trials per split, so the cell is "
                 "a mean of per-person acceptance rates" % ug["Age"].mean(),
                 "Stakes and whether payment was real are not stated in the data file",
                 "Measured responder behaviour at a FIXED offer share; do not apply as a flat "
                 "pooled rejection rate"],
                int(v.size), "none stated in the repository", measured_in="student_sample",
                dispersion=float(v.std(ddof=1)), dispersion_kind="across_participants"))
        fair = ug["Acceptance_rates_fair_splits"].dropna() / 100.0
        unfair = ug["Acceptance_rates_unfair_splits"].dropna() / 100.0
        for nm, s in (("fair_splits", fair), ("unfair_splits", unfair)):
            entries.append(single(
                f"evidence.game.ug_responder.accept_{nm}_pooled", "proportion", float(s.mean()),
                "proportion", dict(scale_points=None, **APP_ANY), "UG social anxiety",
                sources, ["Pooled over the study's own definition of fair/unfair splits; "
                          "see the per-share entries for the gradient"],
                int(s.size), "none stated in the repository", measured_in="student_sample",
                dispersion=float(s.std(ddof=1)), dispersion_kind="across_participants"))

    p_bd = os.path.join(root, "bruhin_binary_dictator", "choices.csv")
    if os.path.exists(p_bd):
        c = pd.read_csv(p_bd)
        sources.append({
            "dataset": "Binary dictator choices (Bruhin, Fehr & Schunk replication data, course copy)",
            "retrieved_from": "https://raw.githubusercontent.com/felixn95/tds_dictator_game/main/data/choices.csv",
            "canonical_source": "https://doi.org/10.1093/jeea/jvy018 (as named by the course notebook)",
            "sha256": sha256(p_bd), "bytes": os.path.getsize(p_bd),
            "n_rows_file": int(len(c))})
        bd = sources[-1]["dataset"]
        # Trade-off games: one option gives the dictator more, the other gives the
        # receiver more. Prosocial choice = the option with higher receiver payoff.
        sx, sy, ox, oy = c.self_x, c.self_y, c.other_x, c.other_y
        tradeoff = ((sx > sy) & (ox < oy)) | ((sx < sy) & (ox > oy))
        pro_is_x = ox > oy
        pro = np.where(pro_is_x, c.choice_x == 1, c.choice_x == 0)
        ahead_both = (sx > ox) & (sy > oy)
        behind_both = (sx < ox) & (sy < oy)
        cost = np.where(pro_is_x, sy - sx, sx - sy).astype(float)     # own money given up
        t = c[tradeoff].assign(pro=pro[tradeoff], ahead=ahead_both[tradeoff],
                               behind=behind_both[tradeoff])
        per_subject = lambda m: t[m].groupby("sid").pro.mean()   # noqa: E731
        bd_cav = ["Binary allocation choices (30-39 games per person, payoffs in experimental "
                  "currency), not a continuous give-away: this is the rate of choosing the "
                  "option that pays the receiver more at a cost to oneself",
                  "A course-assignment copy of the replication data of the cited paper; the "
                  "redistribution terms are not stated, so only aggregates are kept"]
        for nm, mask in (("all_tradeoff", np.ones(len(t), bool)),
                         ("dictator_ahead_in_both", t.ahead.values),
                         ("dictator_behind_in_both", t.behind.values)):
            s = per_subject(mask)
            entries.append(single(
                f"evidence.game.binary_dictator.prosocial_choice_share.{nm}", "proportion",
                float(t[mask].pro.mean()), "proportion", dict(scale_points=None, **APP_ANY),
                bd, sources, bd_cav, int(s.size), "none stated in the repository",
                measured_in="student_sample", dispersion=float(s.std(ddof=1)),
                dispersion_kind="across_participants"))
        for nm, q in (("cost_below_median", t.assign(cost=cost[tradeoff]).cost <= np.median(cost[tradeoff])),
                      ("cost_above_median", t.assign(cost=cost[tradeoff]).cost > np.median(cost[tradeoff]))):
            entries.append(single(
                f"evidence.game.binary_dictator.prosocial_choice_share.{nm}", "proportion",
                float(t[q.values].pro.mean()), "proportion", dict(scale_points=None, **APP_ANY),
                bd, sources, bd_cav + ["Split at the median own-payoff sacrifice of the trade-off games"],
                int(t[q.values].sid.nunique()), "none stated in the repository",
                measured_in="student_sample"))

    p_rd = os.path.join(root, "rdatasets")
    def rd(name):
        pth = os.path.join(p_rd, name + ".csv")
        if not os.path.exists(pth):
            return None
        sources.append({"dataset": f"Rdatasets {name}", "retrieved_from": f"{RDAT}/{'carData' if name == 'Guyer' else 'psych'}/{name}.csv",
                        "canonical_source": "https://vincentarelbundock.github.io/Rdatasets/",
                        "sha256": sha256(pth), "bytes": os.path.getsize(pth)})
        return pd.read_csv(pth)

    rd_lic = ("data file from the Rdatasets mirror of an R package; the package licence was not "
              "independently verified in this build, so only derived aggregates are kept")
    g = rd("Guyer")
    if g is not None:
        g["rate"] = g.cooperation / 120.0
        anon, pub = g[g.condition == "anonymous"].rate.values, g[g.condition == "public"].rate.values
        gd = "Rdatasets Guyer"
        gc = ["Fox & Guyer (1978) n-person prisoner's dilemma: 4-person groups, 30 trials, "
              "20 groups in all; the unit of observation is the group (cooperative choices "
              "out of 120)", "Repeated game: cooperation rates are for 30 rounds pooled"]
        for nm, a in (("anonymous", anon), ("public", pub)):
            entries.append(single(
                f"evidence.game.npd_cooperation.{nm}", "proportion", float(a.mean()), "proportion",
                dict(scale_points=None, design="between", population="any", repetition="repeated"),
                gd, sources, gc, int(a.size), rd_lic, measured_in="student_sample",
                dispersion=float(a.std(ddof=1)), dispersion_kind="across_groups"))
        d = cohens_d(pub, anon)
        e = single("evidence.game.npd_cooperation.public_minus_anonymous_d", "cohens_d", d, "d",
                   dict(scale_points=None, design="between", population="any", repetition="repeated"),
                   gd, sources, gc + ["Cohen's d on group cooperation rates; n=10 groups per cell"],
                   int(len(g)), rd_lic, measured_in="student_sample",
                   dispersion=d_se(d, len(pub), len(anon)), dispersion_kind="standard_error")
        entries.append(e)

    for name, dv_defs in (("Garcia", [("liking", "protest", (0, 1), "individual_protest_minus_control"),
                                      ("liking", "protest", (0, 2), "collective_protest_minus_control"),
                                      ("anger", "protest", (0, 2), "collective_protest_minus_control")]),
                          ("Tal_Or", [("pmi", "cond", (0, 1), "high_minus_low_importance"),
                                      ("reaction", "cond", (0, 1), "high_minus_low_importance")])):
        df = rd(name)
        if df is None:
            continue
        for dv, fac, (lo_l, hi_l), label in dv_defs:
            a = df.loc[df[fac] == hi_l, dv].dropna().values
            b = df.loc[df[fac] == lo_l, dv].dropna().values
            d = cohens_d(a, b)
            entries.append(single(
                f"evidence.effect.{name.lower()}.{dv}.{label}", "cohens_d", d, "d",
                dict(scale_points=None, design="between", population="any", repetition="one_shot"),
                f"Rdatasets {name}", sources,
                ["Observed effect in ONE small randomised experiment (n=%d and %d); the standard "
                 "error is the honest width. Not a meta-analytic estimate and carries the "
                 "winner's-curse risk of any single published study" % (len(a), len(b)),
                 "Cohen's d = (mean of the labelled condition - mean of the control/low condition) / pooled SD"],
                int(len(a) + len(b)), rd_lic, measured_in="student_sample",
                dispersion=d_se(d, len(a), len(b)), dispersion_kind="standard_error"))

    # Cross-check of the engine's current scale-free constant against what 5/7-point
    # data show. Informational: it is NOT an entry.
    cross = {}
    for sp in (5, 7):
        grp = [p for p in profiles if p["scale_points"] == sp]
        if grp:
            arr = np.array([p["item_sd_fraction_of_span"] for p in grp])
            cross[f"item_sd_fraction_of_span_{sp}pt"] = {
                "mean": float(arr.mean()), "sd_across_blocks": float(arr.std(ddof=1)),
                "blocks": int(arr.size)}
    out = {
        "schema_version": 1, "generated_by": SCRIPT, "generated_on": RETRIEVED,
        "note": ("Entries use the 'evidence.' prefix, which no engine path consults. They are "
                 "MEASURED but do not change generated data; wiring one in is a separate, "
                 "tested change."),
        "sources": sources, "entries": entries,
        "cross_checks": cross, "block_profiles": profiles,
    }
    text = json.dumps(out, indent=1, sort_keys=False)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}: {len(entries)} entries from {len(profiles)} blocks", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
