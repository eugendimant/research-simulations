"""Instructor-report content for within-subjects and mixed designs.

The between-subjects report compares groups of different people. On repeated-measures data that would
be wrong (the same people appear in every condition), so for a run whose ``Metadata.json`` carries a
``design`` of type ``within`` or ``mixed`` the report is built from this module instead:

* paired t-tests with d_z and d_av and their confidence intervals, Holm-adjusted for several pairs,
* repeated-measures ANOVA (Mauchly's test, Greenhouse-Geisser and Huynh-Feldt corrections),
* the mixed ANOVA for mixed designs, with the group x condition interaction also as a t-test on the
  difference scores,
* Wilcoxon signed-rank and Friedman as the non-parametric alternatives,
* a check of the requested effects against what this sample shows, in d_av.

The analysis (:func:`analyse`) is separate from the layout: both the Markdown and the HTML report are
produced from the same list of blocks, so the two documents print the same numbers.
"""
from __future__ import annotations

import html as _html
import itertools
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import within_stats as WS
from .within_design import _col_for, design_from_metadata

logger = logging.getLogger(__name__)

__all__ = ["analyse", "markdown_blocks", "html_blocks", "render_markdown", "render_html", "key_rows",
           "effect_check_rows"]

Block = Tuple[Any, ...]


# --------------------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------------------
def _ir() -> Any:
    from . import instructor_report as ir
    return ir


def _f(x: Any, digits: int = 2, signed: bool = False) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _p(x: Any) -> str:
    return _ir()._report_p_cell(x)


def _pt(x: Any) -> str:
    return _ir()._report_p_text(x)


def _ci(ci: Sequence[Any], digits: int = 2) -> str:
    try:
        lo, hi = float(ci[0]), float(ci[1])
    except (TypeError, ValueError, IndexError):
        return "n/a"
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "n/a"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def _dlabel(d: Any) -> str:
    return _ir()._cohens_d_label(d)


def _dv_scale(metadata: Dict[str, Any], dv: str) -> Dict[str, Any]:
    for s in metadata.get("scales") or []:
        if not isinstance(s, dict):
            continue
        for key in ("variable_name", "name"):
            raw = str(s.get(key, ""))
            if raw and _ir()._report_clean_column_name(raw) == dv:
                return s
    return {}


# --------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------
def _composite_matrix(df: pd.DataFrame, metadata: Dict[str, Any], dv: str, labels: Sequence[str]) -> Optional[np.ndarray]:
    cols = [_col_for(metadata, dv, lab) for lab in labels]
    if any(c is None or c not in df.columns for c in cols):
        return None
    return np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cols])


def _marginal(Y: np.ndarray, cells: Sequence[Dict[str, Any]], factor: str, level: str,
              at: Optional[Dict[str, Any]] = None) -> Optional[np.ndarray]:
    """Person-level mean over the cells at ``level`` of ``factor`` (optionally restricted by ``at``)."""
    idx = []
    for j, c in enumerate(cells):
        lv = c.get("levels") or {}
        if str(lv.get(factor, "")).strip().lower() != str(level).strip().lower():
            continue
        ok = True
        for k, want in (at or {}).items():
            wants = want if isinstance(want, (list, tuple, set)) else [want]
            have = next((v for kk, v in lv.items() if str(kk).strip().lower() == str(k).strip().lower()), None)
            if have is not None and str(have).strip().lower() not in {str(w).strip().lower() for w in wants}:
                ok = False
        if ok:
            idx.append(j)
    if not idx:
        return None
    return Y[:, idx].mean(1)


def analyse(df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Run every analysis of a within/mixed dataset; returns plain dicts (see the renderers)."""
    d = design_from_metadata(metadata)
    if not d:
        raise ValueError("analyse() needs the data of a within-subjects or mixed run")
    cells = d["cells"]
    labels = [c["label"] for c in cells]
    wf = d.get("within_factors") or []
    levels = [len(f["levels"]) for f in wf] or [len(cells)]
    names = [str(f["name"]) for f in wf] or ["Condition"]
    mixed = d["type"] == "mixed"
    groups = df["CONDITION"].astype(str).to_numpy() if (mixed and "CONDITION" in df.columns) else None
    group_labels = list(d.get("group_labels") or [])
    if mixed and groups is not None:
        present = list(dict.fromkeys(groups.tolist()))
        group_labels = [g for g in group_labels if g in present] + [g for g in present if g not in group_labels]

    res: Dict[str, Any] = {"design": d, "labels": labels, "levels": levels, "names": names, "mixed": mixed,
                           "group_labels": group_labels, "n": int(len(df)), "dvs": []}
    # order and completion
    if "Order" in df.columns:
        vc = df["Order"].astype(str).value_counts()
        res["orders"] = [(str(k), int(v)) for k, v in vc.items()]
    if "Conditions_Completed" in df.columns:
        vc = pd.to_numeric(df["Conditions_Completed"], errors="coerce").value_counts().sort_index()
        res["completed"] = [(int(k), int(v)) for k, v in vc.items() if math.isfinite(k)]
    if mixed and groups is not None:
        res["group_n"] = [(g, int(np.sum(groups == g))) for g in group_labels]

    # between-subjects factors as full factors when the group labels can be read as a crossing of them
    bfactors = [f for f in (d.get("between_factors") or []) if isinstance(f, dict) and f.get("levels")]
    between_matrix = None
    between_names = None
    if mixed and groups is not None and bfactors:
        from .within_design import _match_levels
        parsed = {g: _match_levels(g, bfactors) for g in group_labels}
        if all(v is not None for v in parsed.values()):
            between_names = [str(f["name"]) for f in bfactors]
            between_matrix = np.array([[parsed[g][nm] for nm in between_names] for g in groups.tolist()], dtype=object)
    res["between_names"] = between_names

    for dv in (d.get("wide_columns") or {}):
        Y = _composite_matrix(df, metadata, dv, labels)
        if Y is None:
            continue
        item: Dict[str, Any] = {"dv": dv, "scale": _dv_scale(metadata, dv), "n_rows": int(len(Y))}
        desc = []
        for j, lab in enumerate(labels):
            col = Y[:, j][np.isfinite(Y[:, j])]
            n = len(col)
            sd = float(col.std(ddof=1)) if n > 1 else float("nan")
            half = _ir()._t_ci_halfwidth(sd, n) if n > 1 and math.isfinite(sd) else float("nan")
            desc.append({"label": lab, "n": n, "mean": float(col.mean()) if n else float("nan"), "sd": sd,
                         "ci": (float(col.mean()) - half, float(col.mean()) + half) if n > 1 else (float("nan"),) * 2})
        item["desc"] = desc
        if mixed and groups is not None:
            gd = []
            for g in group_labels:
                m = groups == g
                for j, lab in enumerate(labels):
                    col = Y[m, j][np.isfinite(Y[m, j])]
                    gd.append({"group": g, "label": lab, "n": int(len(col)),
                               "mean": float(col.mean()) if len(col) else float("nan"),
                               "sd": float(col.std(ddof=1)) if len(col) > 1 else float("nan")})
            item["desc_groups"] = gd
        # within-person correlations (pairwise complete)
        K = len(labels)
        C = np.full((K, K), np.nan)
        for i in range(K):
            for j in range(K):
                ok = np.isfinite(Y[:, i]) & np.isfinite(Y[:, j])
                if ok.sum() > 3 and Y[ok, i].std() > 0 and Y[ok, j].std() > 0:
                    C[i, j] = np.corrcoef(Y[ok, i], Y[ok, j])[0, 1]
        item["corr"] = C
        off = C[np.triu_indices(K, 1)]
        item["r_mean"] = float(np.nanmean(off)) if np.isfinite(off).any() else float("nan")
        # omnibus
        if mixed and groups is not None and len(group_labels) >= 2:
            item["mixed_anova"] = (WS.mixed_anova(Y, between_matrix, levels, names, between_names=between_names)
                                   if between_matrix is not None else
                                   WS.mixed_anova(Y, groups, levels, names, between_name="Group"))
        else:
            item["rm_anova"] = WS.rm_anova(Y, levels, names)
        # pairwise paired t (Holm), only as many pairs as are readable
        if K <= 6:
            item["pairs"] = WS.pairwise_paired(Y, labels)
        # non-parametric
        if K == 2:
            item["wilcoxon"] = WS.wilcoxon_signed_rank(Y[:, 0], Y[:, 1])
        elif K >= 3:
            item["friedman"] = WS.friedman_test(Y)
        # mixed extras
        if mixed and groups is not None:
            if K == 2:
                item["interaction_diff"] = WS.interaction_by_difference_scores(Y[:, 0], Y[:, 1], groups)
            between = []
            for j, lab in enumerate(labels):
                col = Y[:, j]
                samples = [col[(groups == g) & np.isfinite(col)] for g in group_labels]
                if len(samples) == 2 and all(len(s) > 1 for s in samples):
                    a, b = samples
                    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
                    se = sp * math.sqrt(1 / len(a) + 1 / len(b))
                    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
                    dfree = len(a) + len(b) - 2
                    between.append({"label": lab, "g1": group_labels[0], "g2": group_labels[1],
                                    "diff": float(a.mean() - b.mean()), "t": float(t), "df": dfree,
                                    "p": _ir()._p_t_two_sided(t, dfree) if math.isfinite(t) else float("nan"),
                                    "d": float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")})
                elif len(samples) > 2 and all(len(s) > 1 for s in samples):
                    grand = np.concatenate(samples)
                    ssb = sum(len(s) * (s.mean() - grand.mean()) ** 2 for s in samples)
                    ssw = sum(((s - s.mean()) ** 2).sum() for s in samples)
                    d1, d2 = len(samples) - 1, len(grand) - len(samples)
                    f = (ssb / d1) / (ssw / d2) if ssw > 0 else float("nan")
                    between.append({"label": lab, "f": float(f), "df1": d1, "df2": d2,
                                    "p": _ir()._p_f_sf(f, d1, d2) if math.isfinite(f) else float("nan"),
                                    "eta2": float(ssb / (ssb + ssw)) if (ssb + ssw) > 0 else float("nan")})
            item["between_at_cell"] = between
        item["Y_shape"] = tuple(Y.shape)
        res["dvs"].append(item)
    res["effect_check"] = effect_check_rows(df, metadata)
    return res


def effect_check_rows(df: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Requested effects against this sample, in d_av (within contrasts) or d (between-group contrasts)."""
    d = design_from_metadata(metadata)
    if not d:
        return []
    cells = d["cells"]
    labels = [c["label"] for c in cells]
    rows: List[Dict[str, Any]] = []
    spec_rows = ((metadata.get("effect_sizes_applied") or {}).get("specs")) or []
    for sp in spec_rows:
        dv = str(sp.get("variable", ""))
        Y = _composite_matrix(df, metadata, dv, labels)
        row = {"variable": dv, "contrast": f"{sp.get('level_high')} - {sp.get('level_low')}",
               "kind": sp.get("kind"), "intended_d": sp.get("intended_d"), "scope": sp.get("scope"),
               "status": sp.get("status"), "observed_d_av": None, "observed_d_z": None, "se": None,
               "check": "not applied" if not sp.get("matched") else "n/a"}
        if Y is not None and sp.get("matched"):
            if sp.get("kind") == "within":
                fac = next((f["name"] for f in d["within_factors"]
                            if str(sp.get("level_high", "")).strip().lower() in {str(x).strip().lower() for x in f["levels"]}),
                           None)
                if fac:
                    hi = _marginal(Y, cells, fac, str(sp.get("level_high")), sp.get("scope"))
                    lo = _marginal(Y, cells, fac, str(sp.get("level_low")), sp.get("scope"))
                    if hi is not None and lo is not None:
                        t = WS.paired_t_test(hi, lo)
                        row["observed_d_av"], row["observed_d_z"] = t["d_av"], t["d_z"]
                        ci = t["d_av_ci"]
                        if all(math.isfinite(x) for x in ci):
                            row["se"] = (ci[1] - ci[0]) / (2 * 1.96)
            elif "CONDITION" in df.columns:
                from . import enhanced_simulation_engine as E
                groups = df["CONDITION"].astype(str).to_numpy()
                avg = np.mean([Y[:, j] for j, c in enumerate(cells)
                               if (not sp.get("scope")) or _scope_cell_ok(c, sp.get("scope"))], axis=0) \
                    if any((not sp.get("scope")) or _scope_cell_ok(c, sp.get("scope")) for c in cells) else None
                if avg is not None:
                    hi_m = np.array([E._spec_side(sp, E._label_norm(g)) > 0 for g in groups])
                    lo_m = np.array([E._spec_side(sp, E._label_norm(g)) < 0 for g in groups])
                    a, b = avg[hi_m & np.isfinite(avg)], avg[lo_m & np.isfinite(avg)]
                    if len(a) > 1 and len(b) > 1:
                        sp_ = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
                        if sp_ > 0:
                            row["observed_d_av"] = float((a.mean() - b.mean()) / sp_)
                            row["se"] = math.sqrt(1 / len(a) + 1 / len(b) + row["observed_d_av"] ** 2 / (2 * (len(a) + len(b))))
            intended, observed, se = row["intended_d"], row["observed_d_av"], row["se"]
            if _fin(intended) and _fin(observed):
                tol = max(0.15, 2.0 * se) if _fin(se) else 0.2
                row["check"] = "within sampling error" if abs(observed - intended) <= tol else "outside sampling error"
        rows.append(row)
    return rows


def _scope_cell_ok(cell: Dict[str, Any], scope: Any) -> bool:
    if not isinstance(scope, dict):
        return True
    lv = cell.get("levels") or {}
    for k, want in scope.items():
        wants = want if isinstance(want, (list, tuple, set)) else [want]
        have = next((v for kk, v in lv.items() if str(kk).strip().lower() == str(k).strip().lower()), None)
        if have is not None and str(have).strip().lower() not in {str(w).strip().lower() for w in wants}:
            return False
    return True


def _fin(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------------------
def key_rows(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One headline test per DV for the key-results table."""
    out: List[Dict[str, Any]] = []
    labels = analysis["labels"]
    for item in analysis["dvs"]:
        row: Dict[str, Any] = {"dv": item["dv"]}
        if analysis["mixed"] and item.get("mixed_anova", {}).get("ok"):
            terms = item["mixed_anova"]["terms"]
            inter = next((t for t in terms if t.get("type") == "interaction"), None) or terms[0]
            row.update(contrast=f"{inter['term']} (mixed ANOVA)", statistic=f"F = {_f(inter['f'])}",
                       df=f"{inter['df1']:.0f}, {inter['df2']:.0f}", p=inter["p_gg"] if inter["df1"] > 1 else inter["p"],
                       effect=f"partial eta² = {_f(inter['partial_eta2'], 3)}",
                       label=_eta_label(inter["partial_eta2"]))
            diff = item.get("interaction_diff")
            if diff and _fin(diff.get("t")):
                row["extra"] = f"t on difference scores = {_f(diff['t'])}, {_pt(diff['p'])}"
        elif len(labels) == 2 and item.get("pairs"):
            t = item["pairs"][0]
            row.update(contrast=f"{labels[0]} - {labels[1]} (paired t-test)", statistic=f"t = {_f(t['t'], 2, True)}",
                       df=f"{t['df']:.0f}", p=t["p"], effect=f"d_av = {_f(t['d_av'], 2, True)}, d_z = {_f(t['d_z'], 2, True)}",
                       label=_dlabel(t["d_av"]))
        elif item.get("rm_anova", {}).get("ok"):
            term = item["rm_anova"]["terms"][0]
            row.update(contrast=f"{term['term']} (repeated-measures ANOVA)", statistic=f"F = {_f(term['f'])}",
                       df=f"{term['df1']:.0f}, {term['df2']:.0f}",
                       p=term["p_gg"] if term["df1"] > 1 else term["p"],
                       effect=f"partial eta² = {_f(term['partial_eta2'], 3)}", label=_eta_label(term["partial_eta2"]))
        else:
            row["error"] = "too few complete cases for a repeated-measures test"
        out.append(row)
    return out


def _eta_label(eta: Any) -> str:
    try:
        v = float(eta)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    return "negligible" if v < 0.01 else "small" if v < 0.06 else "medium" if v < 0.14 else "large"


def _anova_rows(anova: Dict[str, Any]) -> List[List[str]]:
    rows = []
    for t in anova.get("terms", []):
        gg = t["df1"] > 1 and t.get("type") != "between"
        rows.append([
            t["term"], f"{t['df1']:.0f}, {t['df2']:.0f}", _f(t["f"]), _p(t["p"]),
            _p(t["p_gg"]) if gg else "-", _p(t["p_hf"]) if gg else "-",
            _f(t["partial_eta2"], 3), _f(t.get("generalized_eta2"), 3) if "generalized_eta2" in t else "-"])
    return rows


def _sphericity_text(anova: Dict[str, Any]) -> List[str]:
    out = []
    for t in anova.get("terms", []):
        if t["df1"] > 1 and t.get("type") != "between" and math.isfinite(t.get("eps_gg", float("nan"))):
            w = t.get("mauchly_w")
            ptxt = f"Mauchly's W = {_f(w, 3)}, {_pt(t.get('mauchly_p'))}; " if _fin(w) else ""
            out.append(f"{t['term']}: {ptxt}Greenhouse-Geisser epsilon = {_f(t['eps_gg'], 3)}, "
                       f"Huynh-Feldt epsilon = {_f(t['eps_hf'], 3)}.")
    return out


def _omnibus_sentence(anova: Dict[str, Any]) -> List[str]:
    out = []
    for t in anova.get("terms", []):
        use_gg = t["df1"] > 1 and t.get("type") != "between" and _fin(t.get("mauchly_p")) and t["mauchly_p"] < 0.05
        p = t["p_gg"] if use_gg else t["p"]
        sig = "is statistically significant" if _fin(p) and p < 0.05 else "is not statistically significant"
        txt = (f"The {t['term']} effect {sig}: F({t['df1']:.0f}, {t['df2']:.0f}) = {_f(t['f'])}, {_pt(p)}, "
               f"partial eta² = {_f(t['partial_eta2'], 3)} ({_eta_label(t['partial_eta2'])})")
        txt += (", Greenhouse-Geisser corrected because sphericity is rejected." if use_gg else ".")
        out.append(txt)
    return out


def _dv_blocks(analysis: Dict[str, Any], item: Dict[str, Any]) -> List[Block]:
    labels = analysis["labels"]
    sc = item.get("scale") or {}
    b: List[Block] = [("h3", item["dv"])]
    n_items = sc.get("num_items")
    pts = sc.get("scale_points")
    if n_items:
        b.append(("p", f"Configuration: {n_items} item(s), {pts}-point scale. The score analysed in each condition is "
                       f"the composite mean ({item['dv']}_<condition>_mean) or the single item."))
    b.append(("h4", "By condition (all participants with a score)"))
    b.append(("table", ["Condition", "N", "Mean", "SD", "95% CI"],
              [[d["label"], str(d["n"]), _f(d["mean"], 3), _f(d["sd"], 3), _ci(d["ci"], 3)] for d in item["desc"]]))
    if item.get("desc_groups"):
        b.append(("h4", "By group and condition"))
        b.append(("table", ["Group", "Condition", "N", "Mean", "SD"],
                  [[g["group"], g["label"], str(g["n"]), _f(g["mean"], 3), _f(g["sd"], 3)] for g in item["desc_groups"]]))
    C = item["corr"]
    if len(labels) <= 8:
        b.append(("h4", "Within-person correlations between conditions"))
        b.append(("table", [""] + list(labels),
                  [[labels[i]] + [("1.00" if i == j else _f(C[i, j], 2)) for j in range(len(labels))]
                   for i in range(len(labels))]))
        b.append(("p", f"Mean correlation across conditions: r = {_f(item['r_mean'], 2)} "
                       f"(requested {_f(analysis['design'].get('within_correlation'), 2)}). The correlation is what makes the "
                       "paired test more powerful than a test between groups of different people."))
    anova = item.get("mixed_anova") or item.get("rm_anova")
    if anova and anova.get("ok"):
        title = "Mixed ANOVA" if item.get("mixed_anova") else "Repeated-measures ANOVA"
        b.append(("h4", f"{title} (complete cases, N = {anova['n']})"))
        b.append(("table", ["Effect", "df", "F", "p", "p (Greenhouse-Geisser)", "p (Huynh-Feldt)", "partial eta²",
                            "generalised eta²"], _anova_rows(anova)))
        for line in _sphericity_text(anova):
            b.append(("note", line))
        for line in _omnibus_sentence(anova):
            b.append(("p", line))
        if item.get("mixed_anova"):
            b.append(("note", f"Group sizes: {', '.join(f'{g} = {n}' for g, n in zip(anova['groups'], anova['group_n']))}. "
                              "Type III sums of squares (sum-to-zero coding)" + ("; the between-subjects factors enter as full factors with their interactions. " if analysis.get("between_names") else "; the between-subjects groups are analysed as one grouping variable. ") + "p-values for within effects assume sphericity unless the corrected columns are shown."))
    elif anova is not None:
        b.append(("warn", "Too few participants with complete data in every condition for the omnibus test."))
    pairs = item.get("pairs") or []
    if pairs:
        b.append(("h4", "Paired comparisons (first minus second)"))
        rows = []
        for t in pairs:
            rows.append([f"{t['label_1']} - {t['label_2']}", str(t["n"]), _f(t["mean_diff"], 3), _ci(t["ci"], 3),
                         f"{_f(t['t'], 2, True)} ({t['df']:.0f})", _p(t["p"]), _p(t.get("p_holm")),
                         f"{_f(t['d_z'], 2, True)} {_ci(t['d_z_ci'])}", f"{_f(t['d_av'], 2, True)} {_ci(t['d_av_ci'])}"])
        b.append(("table", ["Contrast", "N pairs", "Mean diff.", "95% CI", "t (df)", "p", "p (Holm)", "d_z [95% CI]",
                            "d_av [95% CI]"], rows))
        b.append(("note", "d_av = mean difference / average of the two conditions' SDs, the effect size comparable to a "
                          "between-subjects d. d_z = mean difference / SD of the differences; d_z = d_av / sqrt(2(1 - r)), so "
                          "it is larger than d_av when the conditions are positively correlated. The Holm column corrects "
                          "for the number of pairs." if len(pairs) > 1 else
                          "d_av = mean difference / average of the two conditions' SDs, the effect size comparable to a "
                          "between-subjects d. d_z = mean difference / SD of the differences; d_z = d_av / sqrt(2(1 - r))."))
        if len(pairs) == 1:
            t = pairs[0]
            sig = "significant" if _fin(t["p"]) and t["p"] < 0.05 else "not significant"
            b.append(("p", f"{t['label_1']} minus {t['label_2']}: mean difference {_f(t['mean_diff'], 3)} "
                           f"(95% CI {_ci(t['ci'], 3)}), t({t['df']:.0f}) = {_f(t['t'], 2, True)}, {_pt(t['p'])}, "
                           f"d_av = {_f(t['d_av'], 2, True)} ({_dlabel(t['d_av'])}), d_z = {_f(t['d_z'], 2, True)}; "
                           f"the difference is {sig}."))
    if item.get("wilcoxon"):
        w = item["wilcoxon"]
        b.append(("h4", "Non-parametric alternative: Wilcoxon signed-rank"))
        b.append(("p", f"W+ = {_f(w['w_plus'], 1)}, W- = {_f(w['w_minus'], 1)}, z = {_f(w['z'], 2, True)}, {_pt(w['p'])}, "
                       f"r = {_f(w['r'], 2, True)} (N = {w['n']}, {w['n_nonzero']} non-zero differences; "
                       f"{'exact' if w['exact'] else 'normal approximation'})."))
    if item.get("friedman") and _fin(item["friedman"].get("chi2")):
        fr = item["friedman"]
        b.append(("h4", "Non-parametric alternative: Friedman test"))
        b.append(("p", f"chi-square({fr['df']:.0f}) = {_f(fr['chi2'])}, {_pt(fr['p'])}, Kendall's W = {_f(fr['kendall_w'], 3)} "
                       f"(N = {fr['n']}). Mean ranks: " + ", ".join(f"{l} {_f(r, 2)}" for l, r in zip(labels, fr["mean_ranks"])) + "."))
    if item.get("interaction_diff") and _fin(item["interaction_diff"].get("t")):
        idf = item["interaction_diff"]
        b.append(("h4", "Group x condition interaction as a test on the difference scores"))
        rows = [[str(g["group"]), str(g["n"]), _f(g["mean_diff"], 3), _ci(g["ci"], 3), f"{_f(g['t'], 2, True)} ({g['df']:.0f})",
                 _p(g["p"]), _f(g["d_av"], 2, True)] for g in idf["per_group"]]
        b.append(("table", ["Group", "N pairs", f"Change ({labels[0]} - {labels[1]})", "95% CI", "t (df)", "p", "d_av"], rows))
        b.append(("p", f"Difference of the changes between groups: {_f(idf['mean_diff_of_diffs'], 3)} (95% CI {_ci(idf['ci'], 3)}), "
                       f"t({idf['df']:.0f}) = {_f(idf['t'], 2, True)}, {_pt(idf['p'])}. This equals the F test of the interaction "
                       "in the mixed ANOVA (F = t squared)."))
    if item.get("between_at_cell"):
        b.append(("h4", "Group differences within each condition"))
        rows = []
        for r in item["between_at_cell"]:
            if "t" in r:
                rows.append([r["label"], f"{r['g1']} - {r['g2']}", _f(r["diff"], 3), f"{_f(r['t'], 2, True)} ({r['df']})",
                             _p(r["p"]), _f(r["d"], 2, True)])
            else:
                rows.append([r["label"], "all groups", "-", f"F({r['df1']}, {r['df2']}) = {_f(r['f'])}", _p(r["p"]),
                             f"eta² = {_f(r['eta2'], 3)}"])
        b.append(("table", ["Condition", "Contrast", "Mean diff.", "Test", "p", "Effect size"], rows))
    return b


def markdown_blocks(analysis: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, List[Block]]:
    """Blocks per report section: ``design``, ``key``, ``dv:<name>``, ``effects``, ``order``."""
    d = analysis["design"]
    sections: Dict[str, List[Block]] = {}
    rows = [["Design type", "Within-subjects (every participant sees every condition)" if d["type"] == "within"
             else "Mixed (between-subjects groups x within-subject conditions)"]]
    for f in d.get("within_factors") or []:
        rows.append([f"Within factor: {f['name']}", ", ".join(f["levels"])])
    if analysis["mixed"]:
        rows.append(["Between groups (N)", ", ".join(f"{g} ({n})" for g, n in analysis.get("group_n", []))])
    rows.append(["Order of conditions", f"{d.get('order_effective', d.get('order'))} (recorded per participant in the Order column)"])
    oe = d.get("order_effect") or {}
    rows.append(["Order / fatigue drift", (f"{_f(oe.get('d_per_position'), 3, True)} SD per later position"
                                           if oe.get("enabled") else "switched off")])
    rows.append(["Within-person correlation (requested)", f"{_f(d.get('within_correlation'), 2)} ({d.get('correlation_structure')})"])
    att = d.get("attrition") or {}
    rows.append(["Attrition", f"{att.get('dropped', 0)} participant(s) dropped out; they miss the conditions presented after "
                              "the last one they finished"])
    rows.append(["Effect size convention", "d_av for within contrasts (mean difference / average SD of the two conditions); "
                                           "d_z (paired) is reported next to it"])
    s_design: List[Block] = [("h3", "Design"), ("table", ["Property", "Value"], rows)]
    for note in d.get("notes") or []:
        s_design.append(("warn", note))
    if analysis.get("orders"):
        top = analysis["orders"][:12]
        s_design.append(("h4", "Presentation orders (counterbalancing)"))
        s_design.append(("table", ["Order", "N"], [[o, str(n)] for o, n in top]))
        if len(analysis["orders"]) > len(top):
            s_design.append(("note", f"{len(analysis['orders']) - len(top)} further orders not shown."))
    if analysis.get("completed"):
        s_design.append(("h4", "Conditions completed"))
        s_design.append(("table", ["Conditions completed", "N"], [[str(k), str(v)] for k, v in analysis["completed"]]))
        s_design.append(("note", "Participants with missing conditions are left out of the tests that involve those "
                                 "conditions (complete cases)."))
    sections["design"] = s_design
    kr = key_rows(analysis)
    if kr:
        trows = []
        for r in kr:
            if "error" in r:
                trows.append([r["dv"], f"not computed: {r['error']}", "n/a", "n/a", "n/a", "n/a", "n/a"])
            else:
                trows.append([r["dv"], r["contrast"], r["statistic"], r["df"], _p(r["p"]), r["effect"], r["label"]])
        sections["key"] = [("h3", "Key test results (one row per DV composite)"),
                           ("table", ["DV", "Contrast / test", "Statistic", "df", "p", "Effect size", "Magnitude"], trows),
                           ("note", "Two-sided tests on complete cases. Contrasts are first minus second condition. Where a "
                                    "Greenhouse-Geisser correction applies (3 or more levels) the corrected p is shown. p-values "
                                    "are not corrected for the number of DVs.")]
    for item in analysis["dvs"]:
        sections[f"dv:{item['dv']}"] = _dv_blocks(analysis, item)
    ec = analysis.get("effect_check") or []
    if ec:
        erows = []
        for r in ec:
            erows.append([r["variable"], r["contrast"], str(r.get("kind") or ""),
                          _f(r["intended_d"], 2, True), _f(r["observed_d_av"], 2, True), _f(r["observed_d_z"], 2, True), r["check"]])
        sections["effects"] = [
            ("h3", "Requested effects against this sample"),
            ("table", ["DV", "Contrast", "Kind", "Requested d", "Observed d_av", "Observed d_z", "Check"], erows),
            ("note", "Requested within effects are d_av; between-group effects of a mixed design are ordinary d. The observed "
                     "values are for this one sample and vary with sampling (the Check column allows two standard errors)."),
        ]
    return sections


html_blocks = markdown_blocks  # the block lists are format independent


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------
def _md_cell(x: Any) -> str:
    return str(x).replace("|", "/").replace("\n", " ")


def render_markdown(blocks: Sequence[Block], level_offset: int = 0) -> List[str]:
    out: List[str] = []
    for blk in blocks:
        kind = blk[0]
        if kind == "h3":
            out += [f"{'#' * (3 + level_offset)} {blk[1]}", ""]
        elif kind == "h4":
            out += [f"{'#' * (4 + level_offset)} {blk[1]}", ""]
        elif kind == "p":
            out += [str(blk[1]), ""]
        elif kind == "note":
            out += [f"*{blk[1]}*", ""]
        elif kind == "warn":
            out += [f"**Note:** {blk[1]}", ""]
        elif kind == "table":
            headers, rows = blk[1], blk[2]
            out.append("| " + " | ".join(_md_cell(h) for h in headers) + " |")
            out.append("|" + "|".join("---" for _ in headers) + "|")
            for r in rows:
                out.append("| " + " | ".join(_md_cell(c) for c in r) + " |")
            out.append("")
    return out


def render_html(blocks: Sequence[Block]) -> List[str]:
    esc = _html.escape
    out: List[str] = []
    for blk in blocks:
        kind = blk[0]
        if kind == "h3":
            out.append(f"<h3>{esc(str(blk[1]))}</h3>")
        elif kind == "h4":
            out.append(f"<h4>{esc(str(blk[1]))}</h4>")
        elif kind == "p":
            out.append(f"<p>{esc(str(blk[1]))}</p>")
        elif kind == "note":
            out.append(f"<p><em>{esc(str(blk[1]))}</em></p>")
        elif kind == "warn":
            out.append(f"<div class='warning-box'>{esc(str(blk[1]))}</div>")
        elif kind == "table":
            headers, rows = blk[1], blk[2]
            out.append("<table><tr>" + "".join(f"<th>{esc(str(h))}</th>" for h in headers) + "</tr>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in r) + "</tr>")
            out.append("</table>")
    return out


# --------------------------------------------------------------------------------------
# report assembly (mixed into ComprehensiveInstructorReport)
# --------------------------------------------------------------------------------------
def is_repeated_design(metadata: Any) -> bool:
    """True when the run's metadata describes a within-subjects or mixed design."""
    try:
        return design_from_metadata(metadata) is not None
    except Exception:  # noqa: BLE001 - a malformed design block means "not a repeated-measures run"
        return False


class WithinReportMixin:
    """Report methods for within/mixed runs; relies on the section builders of the host class."""

    def _within_prepare(self, ctx: Any) -> Optional[Dict[str, Any]]:
        try:
            analysis = analyse(ctx.df, ctx.metadata)
            ctx.within = analysis
            ctx.within_blocks = markdown_blocks(analysis, ctx.metadata)
            return analysis
        except Exception as exc:  # noqa: BLE001 - reported as a section note, never a crash
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning("Within-subjects analysis failed: %s", reason, exc_info=True)
            self.section_errors.append(f"'within-subjects analysis': {reason[:300]}")
            ctx.within, ctx.within_blocks = None, {}
            return None

    def _within_markdown(self, ctx: Any) -> str:
        ir = _ir()
        self._within_prepare(ctx)
        blocks = ctx.within_blocks

        def head(num: str, title: str) -> List[str]:
            return ["-" * 80, f"## {num}. {title}", "-" * 80, ""]

        for title, build in (("Study overview", self._md_overview),
                             ("Data quality assurance", self._md_quality_assurance),
                             ("1. Data quality summary", self._md_data_quality)):
            self._run_section(ctx, title, build)
        self._run_section(ctx, "2. Experimental design verification", lambda c: c.out.extend(
            head("2", "EXPERIMENTAL DESIGN VERIFICATION") + render_markdown(blocks.get("design", []))))
        self._run_section(ctx, "3. Dependent variable analysis", lambda c: c.out.extend(
            head("3", "DEPENDENT VARIABLE ANALYSIS (REPEATED MEASURES)")
            + render_markdown(blocks.get("key", []))))
        for item in (getattr(ctx, "within", None) or {}).get("dvs", []):
            self._run_section(ctx, f"3. Dependent variable analysis: {item['dv']}",
                              lambda c, key=f"dv:{item['dv']}": c.out.extend(render_markdown(blocks.get(key, []))))
        for title, build in (("5. Persona distribution and impact", self._md_persona),
                             ("6. Open-ended questions summary", self._md_open_ended)):
            self._run_section(ctx, title, build)
        self._run_section(ctx, "7. Effect size quality assessment", lambda c: c.out.extend(
            head("7", "EFFECT SIZE QUALITY ASSESSMENT") + render_markdown(blocks.get("effects", []))
            if blocks.get("effects") else []))
        self._run_section(ctx, "9. Instructor recommendations", self._md_recommendations)
        return ir._finalize_p_text("\n".join(map(str, ctx.out)), html=False)

    def _within_chart(self, item: Dict[str, Any]) -> Optional[str]:
        data = {d["label"]: (d["mean"], (d["ci"][1] - d["mean"]) if _fin(d["ci"][1]) else 0.0)
                for d in item["desc"] if _fin(d["mean"])}
        if len(data) < 2:
            return None
        try:
            return self._create_means_dot_plot(data, f"{item['dv']}: mean by condition (95% CI)", "Mean score")
        except Exception:  # noqa: BLE001 - the chart is optional
            return None

    def _within_html(self, ctx: Any) -> str:
        ir = _ir()
        esc = _html.escape
        ctx.out = self._html_document_head(ctx.metadata)
        analysis = self._within_prepare(ctx)
        blocks = ctx.within_blocks
        self._run_section(ctx, "Study overview", self._html_overview)

        def sample(c: Any) -> None:
            c.out.append("<a id='sample-overview'></a>")
            c.out.append("<h2>1. Sample Overview</h2>")
            k = len((analysis or {}).get("labels", [])) or len(c.conditions)
            c.out.append("<div class='metric-grid'>")
            for value, label in ((c.n_total, "Participants"), (k, "Conditions per person"),
                                 (f"{c.exclusion_rate:.1f}%", "Exclusion flags"),
                                 ((analysis or {}).get("design", {}).get("type", "").title(), "Design")):
                c.out.append(f"<div class='metric-card'><div class='metric-value'>{esc(str(value))}</div>"
                             f"<div class='metric-label'>{esc(label)}</div></div>")
            c.out.append("</div>")
            c.out.append("<a href='#top' class='back-to-top'>Back to top</a>")

        self._run_section(ctx, "1. Sample overview", sample)

        def summary(c: Any) -> None:
            c.out.append("<a id='exec-summary'></a>")
            c.out.append("<h2>2. Executive Summary</h2>")
            c.out.extend(render_html(blocks.get("design", [])))
            c.out.extend(render_html(blocks.get("key", [])))

        self._run_section(ctx, "2. Executive summary", summary)

        def dv_head(c: Any) -> None:
            c.out.append("<a id='statistical-analysis'></a>")
            c.out.append("<h2>3. Statistical Analysis by DV (repeated measures)</h2>")
            c.out.append("<p>Every participant contributes to every condition, so each DV is analysed with paired and "
                         "repeated-measures methods. Between-subjects tests are not used on these data.</p>")

        self._run_section(ctx, "3. Statistical analysis by DV", dv_head)
        for item in (analysis or {}).get("dvs", []):
            def one(c: Any, item: Dict[str, Any] = item) -> None:
                c.out.extend(render_html(blocks.get(f"dv:{item['dv']}", [])))
                img = self._within_chart(item)
                if img:
                    c.out.append(f"<div class='chart-container'><img src='data:image/png;base64,{img}' "
                                 f"alt='Mean {esc(item['dv'])} by condition'></div>")
            self._run_section(ctx, f"3. Statistical analysis by DV: {item['dv']}", one)
        self._run_section(ctx, "4-5. Persona and categorical analysis", self._html_persona)

        def effects(c: Any) -> None:
            if blocks.get("effects"):
                c.out.append("<a id='effect-verification'></a>")
                c.out.append("<h2>6. Effect Size Verification</h2>")
                c.out.extend(render_html(blocks["effects"]))

        self._run_section(ctx, "6. Effect size verification", effects)
        for title, build in (("7. Data quality and exclusions", self._html_exclusions),
                             ("Generation warnings", self._html_generation_warnings),
                             ("8. Instructor notes and methodology", self._html_methodology),
                             ("9. Data dictionary", self._html_data_dictionary),
                             ("Footer", self._html_footer)):
            self._run_section(ctx, title, build)
        return ir._harden_report_html(ir._finalize_p_text("\n".join(map(str, ctx.out)), html=True))


# --------------------------------------------------------------------------------------
# student study summary (User_Study_Summary.md): layout, tests, starter code, power
# --------------------------------------------------------------------------------------
def _paired_power(d_av: float, r: float, n: int) -> float:
    """Approximate power of a two-sided paired t-test (alpha = .05) for a given d_av, within-person r and N pairs."""
    if n < 3:
        return float("nan")
    d_z = abs(d_av) / math.sqrt(max(2.0 * (1.0 - r), 1e-6))
    ncp = d_z * math.sqrt(n)
    z_power = ncp - 1.96
    return float(min(0.99, max(0.05, 0.5 * (1 + math.erf(z_power / math.sqrt(2))))))


def summary_sections(metadata: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> List[str]:
    """Markdown lines for the 'Suggested analysis' part of the student summary of a within/mixed run."""
    d = design_from_metadata(metadata)
    if not d:
        return []
    cells = d["cells"]
    labels = [c["label"] for c in cells]
    K = len(labels)
    wide = d.get("wide_columns") or {}
    scales = {}
    for s in metadata.get("scales") or []:
        if isinstance(s, dict):
            scales[_ir()._report_clean_column_name(str(s.get("variable_name") or s.get("name") or ""))] = s
    mixed = d["type"] == "mixed"
    n = int(metadata.get("sample_size") or (len(df) if df is not None else 0))
    r = float(d.get("within_correlation") or 0.5)
    L: List[str] = ["### Data layout (repeated measures)", ""]
    L.append(f"This is a **{d['type']}-subjects** design: "
             + ("every participant answers every condition." if not mixed else
                "participants are in one between-subjects group and answer every within-subject condition."))
    L += ["", "- `Simulated_Data.csv`: one row per participant; each measure appears once per condition as "
              "`<Measure>_<Condition>_<item>`.",
          "- `Simulated_Data_Long.csv`: the same data with one row per participant and condition (use it for repeated-measures "
          "ANOVA and mixed models). Its `Condition` column and the within-factor column(s) identify the condition.",
          "- `Order` and `Position_<Condition>` record the counterbalanced presentation order; `Conditions_Completed` records attrition "
          "(a participant who dropped out is missing the conditions presented last).", ""]
    L += ["| Measure | Condition | Item columns |", "|---|---|---|"]
    for dv, per in wide.items():
        for lab in labels:
            info = per.get(lab) or {}
            cols = info.get("items") or []
            L.append(f"| {dv} | {lab} | {cols[0]}{' ... ' + cols[-1] if len(cols) > 1 else ''} |")
    L.append("")
    L += ["### Statistical Test Recommendations", ""]
    L += ["| Step | Analysis | Purpose | What to report |", "|---|---|---|---|",
          "| 1 | Missing data | Count participants missing a condition (`Conditions_Completed`) | N per condition, N complete |",
          "| 2 | Descriptives | Mean and SD per condition, and the correlation between conditions | M, SD, r |"]
    if K == 2 and not mixed:
        L += ["| 3 | Paired-samples t-test | Compare the two conditions within the same people | t, df, p, 95% CI of the difference |",
              "| 4 | Effect size | d_z (mean difference / SD of differences) and d_av (mean difference / average SD) | both, with CIs |",
              "| 5 | Non-parametric check | Wilcoxon signed-rank test | W, p, matched-pairs r |"]
    elif not mixed:
        L += ["| 3 | Repeated-measures ANOVA | Omnibus test of the within factor | F, df, p, partial eta-squared |",
              "| 4 | Sphericity | Mauchly's test; Greenhouse-Geisser or Huynh-Feldt correction if violated | W, epsilon, corrected p |",
              "| 5 | Post-hoc paired t-tests | All pairs with a Holm (or Bonferroni) correction | t, adjusted p, d_av / d_z |",
              "| 6 | Non-parametric check | Friedman test, then pairwise Wilcoxon signed-rank tests | chi-square, p, Kendall's W |"]
    else:
        L += ["| 3 | Mixed ANOVA | Group, within factor and Group x within interaction | F, df, p, partial eta-squared |",
              "| 4 | Interaction follow-up | Compare the change scores between groups (t-test on difference scores) and test the change inside each group | t, df, p |",
              "| 5 | Sphericity | Mauchly's test and Greenhouse-Geisser correction when the within factor has 3 or more levels | W, epsilon |"]
    L += [f"| {7 if K > 2 else 6} | Order | Does presentation order matter? Add `Order` or `Position` as a covariate / factor | F or beta, p |", ""]
    L += ["**Effect sizes.** Report d_av (comparable to a between-subjects d) and d_z. They are linked by d_z = d_av / sqrt(2(1 - r)), "
          "where r is the correlation between the conditions; a positive r makes the paired test more powerful than a between-groups test.", ""]
    dv0 = next(iter(wide), None)
    if dv0:
        sc = scales.get(dv0, {})
        per = wide[dv0]
        first, second = labels[0], labels[1]
        i1 = (per.get(first) or {}).get("items") or []
        i2 = (per.get(second) or {}).get("items") or []
        rev = sorted(int(x) for x in (sc.get("reverse_items") or []) if str(x).lstrip("-").isdigit())
        flip = float(sc.get("scale_min", 1)) + float(sc.get("scale_max", sc.get("scale_points", 7)))
        flip_s = str(int(flip)) if flip.is_integer() else str(flip)

        def py_comp(items: List[str], name: str) -> str:
            parts = [f"({flip_s} - df['{c}'])" if (k + 1) in rev else f"df['{c}']" for k, c in enumerate(items)]
            return f"df['{name}'] = pd.concat([{', '.join(parts)}], axis=1).mean(axis=1)" if len(parts) > 1 else f"df['{name}'] = {parts[0]}"

        def r_comp(items: List[str], name: str) -> str:
            parts = [f"({flip_s} - df${c})" if (k + 1) in rev else f"df${c}" for k, c in enumerate(items)]
            return f"df${name} <- rowMeans(cbind({', '.join(parts)}), na.rm = TRUE)"

        c1, c2 = f"{dv0}_{cells[0]['slug']}", f"{dv0}_{cells[1]['slug']}"
        measure_long = f"{dv0}_mean" if len(i1) > 1 else f"{dv0}_1"
        L += ["### Quick-Start Analysis Code", "",
              f"Paired comparison of the first two conditions of `{dv0}` (first minus second):", "", "**Python:**", "```python",
              "import pandas as pd, numpy as np", "from scipy import stats", "df = pd.read_csv('Simulated_Data.csv')",
              py_comp(i1, c1), py_comp(i2, c2),
              f"a, b = df['{c1}'], df['{c2}']", "ok = a.notna() & b.notna()", "diff = (a - b)[ok]",
              "t, p = stats.ttest_rel(a[ok], b[ok])",
              "d_z = diff.mean() / diff.std(ddof=1)", "d_av = diff.mean() / ((a[ok].std() + b[ok].std()) / 2)",
              "print(f't({ok.sum() - 1}) = {t:.3f}, p = {p:.4f}, d_z = {d_z:.3f}, d_av = {d_av:.3f}')", "```", "",
              "**R:**", "```r", "library(readr)", "df <- read_csv('Simulated_Data.csv', show_col_types = FALSE)",
              r_comp(i1, c1), r_comp(i2, c2), f"t.test(df${c1}, df${c2}, paired = TRUE)", "```", ""]
        if K > 2 or mixed:
            L += ["Repeated-measures / mixed ANOVA from the long file (complete cases):", "", "**Python (statsmodels):**", "```python",
                  "import pandas as pd", "from statsmodels.stats.anova import AnovaRM",
                  "long = pd.read_csv('Simulated_Data_Long.csv')",
                  f"y = '{measure_long}'", "complete = long.groupby('PARTICIPANT_ID')[y].count() == long['Condition'].nunique()",
                  "long = long[long['PARTICIPANT_ID'].isin(complete[complete].index)]",
                  "print(AnovaRM(long, y, 'PARTICIPANT_ID', within=['Condition']).fit())",
                  "# mixed designs: statsmodels MixedLM, or the pingouin / afex packages, with the group in the model", "```", "",
                  "**R (afex):**", "```r", "library(afex); library(readr)", "long <- read_csv('Simulated_Data_Long.csv', show_col_types = FALSE)",
                  f"aov_ez(id = 'PARTICIPANT_ID', dv = '{measure_long}', data = long, within = 'Condition'"
                  + (", between = 'CONDITION'" if mixed else "") + ")", "```", ""]
    L += ["### Power Analysis Estimates (paired design)", "",
          f"**Sample:** N = {n} participants, each measured in {K} conditions; assumed correlation between conditions r = {r:.2f}.", "",
          "| Effect size (d_av) | Implied d_z | Estimated power (paired t-test) |", "|---|---|---|"]
    for lab, dav in (("Small", 0.2), ("Medium", 0.5), ("Large", 0.8)):
        dz = dav / math.sqrt(max(2 * (1 - r), 1e-6))
        L.append(f"| {lab} ({dav:.1f}) | {dz:.2f} | {_paired_power(dav, r, n):.0%} |")
    L += ["", "*Power of the paired test for one contrast between two conditions; the correlation between conditions raises it "
              "(a between-groups design with the same N would have less). Mixed-design interactions and ANOVA omnibus tests need "
              "their own power calculation.*", ""]
    return L
