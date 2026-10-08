"""Statistics for repeated-measures data, numpy only (scipy is used for p-values when it is installed).

Everything the instructor report prints about a within-subjects or mixed design comes from here:

* ``paired_t_test``       paired t, mean difference with a t-based CI, d_z and d_av with approximate CIs
* ``wilcoxon_signed_rank`` exact for n <= 50 without ties or zeros, normal approximation otherwise
* ``friedman_test``       chi-square with the tie correction, Kendall's W
* ``rm_anova``            one-way or factorial repeated-measures ANOVA, Mauchly's test, Greenhouse-Geisser
                          and Huynh-Feldt corrections, partial and generalised eta squared
* ``mixed_anova``         between-groups factor x within factor(s), same corrections
* ``pairwise_paired``     all pairs, paired t with Holm adjustment
* ``interaction_by_difference_scores``  for a two-level within factor: the group x time interaction as a
                          t-test on the difference scores (identical to the mixed-ANOVA F with 1 df)

Complete cases only: a participant who dropped out of a condition is left out of the analyses that
involve it, and the number used is returned with every result.

The p-values come from the report's own distribution functions (``instructor_report._p_t_two_sided``,
``_p_f_sf``, ``_chi2_sf``), so with scipy missing they are the exact numpy versions the deployed app uses.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "paired_t_test", "wilcoxon_signed_rank", "friedman_test", "rm_anova", "mixed_anova",
    "pairwise_paired", "interaction_by_difference_scores", "helmert_contrasts", "complete_cases",
]

_NAN = float("nan")


def _ir() -> Any:
    from . import instructor_report as ir  # lazy: instructor_report imports this module too
    return ir


def _p_t(t: float, df: float) -> float:
    return float(_ir()._p_t_two_sided(t, df)) if math.isfinite(t) and df > 0 else _NAN


def _p_f(f: float, d1: float, d2: float) -> float:
    return float(_ir()._p_f_sf(f, d1, d2)) if math.isfinite(f) and d1 > 0 and d2 > 0 else _NAN


def _p_chi2(x: float, df: float) -> float:
    return float(_ir()._chi2_sf(x, df)) if math.isfinite(x) and df > 0 else _NAN


def _tcrit(df: float, conf: float = 0.95) -> float:
    return float(_ir()._t_crit(df, conf))


def complete_cases(*arrays: Any) -> Tuple[np.ndarray, ...]:
    """Rows where every array is finite (arrays may be 1-D or 2-D)."""
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    keep = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        keep &= np.all(np.isfinite(a.reshape(len(a), -1)), axis=1)
    return tuple(a[keep] for a in arrs)


# --------------------------------------------------------------------------------------
# paired comparison of two conditions
# --------------------------------------------------------------------------------------
def paired_t_test(x: Any, y: Any, conf: float = 0.95) -> Dict[str, Any]:
    """Paired t-test of ``x - y``.

    ``d_z`` = mean difference / SD of the differences; ``d_av`` = mean difference / mean of the two SDs
    (the quantity comparable to a between-subjects d). Their CIs are normal approximations: for d_z
    SE = sqrt(1/n + d_z^2/(2n)); for d_av SE = sqrt(2(1-r)/n + d_av^2 (1+r^2)/(4(n-1))).
    """
    xa, ya = complete_cases(x, y)
    n = len(xa)
    out: Dict[str, Any] = {"n": int(n), "mean_1": _NAN, "mean_2": _NAN, "mean_diff": _NAN, "sd_diff": _NAN,
                           "t": _NAN, "df": float(max(n - 1, 0)), "p": _NAN, "ci": (_NAN, _NAN), "d_z": _NAN,
                           "d_z_ci": (_NAN, _NAN), "d_av": _NAN, "d_av_ci": (_NAN, _NAN), "r": _NAN,
                           "sd_1": _NAN, "sd_2": _NAN, "degenerate": False}
    if n < 2:
        out["degenerate"] = True
        return out
    diff = xa - ya
    m, sd = float(diff.mean()), float(diff.std(ddof=1))
    s1, s2 = float(xa.std(ddof=1)), float(ya.std(ddof=1))
    out.update(mean_1=float(xa.mean()), mean_2=float(ya.mean()), mean_diff=m, sd_diff=sd, sd_1=s1, sd_2=s2)
    if s1 > 0 and s2 > 0:
        out["r"] = float(np.corrcoef(xa, ya)[0, 1])
    se = sd / math.sqrt(n)
    tc = _tcrit(n - 1, conf)
    if sd <= 1e-12:
        out["degenerate"] = True
        out["ci"] = (m, m)
        out["p"] = 1.0 if abs(m) <= 1e-12 else 0.0
        out["t"] = 0.0 if abs(m) <= 1e-12 else float("inf") * (1 if m > 0 else -1)
        return out
    t = m / se
    out.update(t=t, p=_p_t(t, n - 1), ci=(m - tc * se, m + tc * se), d_z=m / sd)
    z = 1.959964 if abs(conf - 0.95) < 1e-9 else float(_ir()._norm_ppf(0.5 + conf / 2.0))
    se_dz = math.sqrt(1.0 / n + out["d_z"] ** 2 / (2.0 * n))
    out["d_z_ci"] = (out["d_z"] - z * se_dz, out["d_z"] + z * se_dz)
    avg = (s1 + s2) / 2.0
    if avg > 0:
        d_av = m / avg
        r = out["r"] if math.isfinite(out["r"]) else 0.0
        var = 2.0 * (1.0 - r) / n + d_av ** 2 * (1.0 + r * r) / (4.0 * max(n - 1, 1))
        se_av = math.sqrt(max(var, 0.0))
        out["d_av"] = d_av
        out["d_av_ci"] = (d_av - z * se_av, d_av + z * se_av)
    return out


def pairwise_paired(Y: Any, labels: Sequence[str]) -> List[Dict[str, Any]]:
    """All pairs of columns, paired t with Holm-adjusted p (each pair uses its own complete cases)."""
    Y = np.asarray(Y, dtype=float)
    rows: List[Dict[str, Any]] = []
    for i, j in itertools.combinations(range(Y.shape[1]), 2):
        res = paired_t_test(Y[:, i], Y[:, j])
        res.update(label_1=str(labels[i]), label_2=str(labels[j]))
        rows.append(res)
    ps = [r["p"] for r in rows]
    if rows and all(math.isfinite(p) for p in ps):
        adj = _ir()._holm_adjust(ps)
        for r, a in zip(rows, adj):
            r["p_holm"] = float(a)
    else:
        for r in rows:
            r["p_holm"] = r["p"]
    return rows


# --------------------------------------------------------------------------------------
# non-parametric
# --------------------------------------------------------------------------------------
def _signed_rank_exact_two_sided(w_plus: float, n: int) -> float:
    """Exact two-sided p of the Wilcoxon signed-rank statistic (no ties, no zeros)."""
    total = n * (n + 1) // 2
    counts = np.zeros(total + 1)
    counts[0] = 1.0
    for rank in range(1, n + 1):
        counts[rank:] = counts[rank:] + counts[:-rank].copy() if rank <= total else counts[rank:]
    probs = counts / counts.sum()
    w = int(round(w_plus))
    p_low = float(probs[: min(w, total - w) + 1].sum())
    return float(min(1.0, 2.0 * p_low))


def wilcoxon_signed_rank(x: Any, y: Any) -> Dict[str, Any]:
    """Wilcoxon signed-rank test of ``x - y`` (zeros dropped; exact for n <= 50 without ties)."""
    xa, ya = complete_cases(x, y)
    diff = xa - ya
    n_all = len(diff)
    diff = diff[np.abs(diff) > 1e-12]
    n = len(diff)
    out: Dict[str, Any] = {"n": int(n_all), "n_nonzero": int(n), "w_plus": _NAN, "w_minus": _NAN, "z": _NAN,
                           "p": _NAN, "r": _NAN, "exact": False}
    if n == 0:
        out.update(w_plus=0.0, w_minus=0.0, p=1.0, z=0.0, r=0.0)
        return out
    ranks, tie_sizes = _ir()._rank_average(np.abs(diff))
    w_plus = float(ranks[diff > 0].sum())
    w_minus = float(ranks[diff < 0].sum())
    out.update(w_plus=w_plus, w_minus=w_minus)
    has_ties = bool(np.any(np.asarray(tie_sizes) > 1)) if len(tie_sizes) else False
    mean = n * (n + 1) / 4.0
    if n <= 50 and not has_ties:
        out["p"] = _signed_rank_exact_two_sided(min(w_plus, w_minus), n)
        out["exact"] = True
        var = n * (n + 1) * (2 * n + 1) / 24.0
        out["z"] = (w_plus - mean) / math.sqrt(var)
    else:
        var = n * (n + 1) * (2 * n + 1) / 24.0
        t_corr = sum(float(t) ** 3 - float(t) for t in tie_sizes) / 48.0 if has_ties else 0.0
        var -= t_corr
        cc = 0.5 * np.sign(w_plus - mean)
        z = (w_plus - mean - cc) / math.sqrt(var) if var > 0 else 0.0
        out["z"] = float(z)
        out["p"] = float(min(1.0, 2.0 * _ir()._normal_sf(abs(z))))
    out["r"] = float(out["z"] / math.sqrt(n_all)) if n_all else _NAN
    return out


def friedman_test(Y: Any) -> Dict[str, Any]:
    """Friedman test over the columns of ``Y`` (complete cases), with the tie correction and Kendall's W."""
    (Ya,) = complete_cases(np.asarray(Y, dtype=float))
    n, k = Ya.shape if Ya.ndim == 2 else (0, 0)
    out: Dict[str, Any] = {"n": int(n), "k": int(k), "chi2": _NAN, "df": float(max(k - 1, 0)), "p": _NAN,
                           "kendall_w": _NAN, "mean_ranks": []}
    if n < 2 or k < 3:
        return out
    ranks = np.vstack([_ir()._rank_average(row)[0] for row in Ya])
    col = ranks.sum(0)
    chi = 12.0 / (n * k * (k + 1)) * float(np.sum(col ** 2)) - 3.0 * n * (k + 1)
    tie_term = 0.0
    for row in Ya:
        _, counts = np.unique(row, return_counts=True)
        tie_term += float(np.sum(counts ** 3 - counts))
    denom = 1.0 - tie_term / (n * k * (k * k - 1))
    if denom <= 1e-12:
        out["chi2"], out["p"], out["kendall_w"] = 0.0, 1.0, 0.0
        out["mean_ranks"] = (col / n).tolist()
        return out
    chi /= denom
    out.update(chi2=float(chi), p=_p_chi2(chi, k - 1), kendall_w=float(chi / (n * (k - 1))),
               mean_ranks=(col / n).tolist())
    return out


# --------------------------------------------------------------------------------------
# repeated-measures and mixed ANOVA
# --------------------------------------------------------------------------------------
def helmert_contrasts(m: int) -> np.ndarray:
    """(m-1) x m orthonormal contrasts, each orthogonal to the constant."""
    rows = []
    for r in range(1, m):
        v = np.zeros(m)
        v[:r] = 1.0
        v[r] = -float(r)
        rows.append(v / np.linalg.norm(v))
    return np.array(rows)


def _term_matrix(levels: Sequence[int], term: Tuple[int, ...]) -> np.ndarray:
    """Orthonormal rows that carry the effect of the factors in ``term`` over the product of ``levels``."""
    mat = np.ones((1, 1))
    for f, m in enumerate(levels):
        part = helmert_contrasts(m) if f in term else np.ones((1, m)) / math.sqrt(m)
        mat = np.kron(mat, part)
    return mat


def _terms(n_factors: int) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for size in range(1, n_factors + 1):
        out.extend(itertools.combinations(range(n_factors), size))
    return out


def _term_name(term: Tuple[int, ...], names: Sequence[str]) -> str:
    return " x ".join(names[i] for i in term)


def _sphericity(S: np.ndarray, n_eff: int) -> Dict[str, float]:
    """Mauchly's W with its chi-square p, Greenhouse-Geisser and Huynh-Feldt epsilons for an orthonormal-contrast
    covariance matrix ``S`` ((p x p), p = df of the term) estimated with ``n_eff`` degrees of freedom + 1."""
    p = S.shape[0]
    res = {"mauchly_w": _NAN, "mauchly_chi2": _NAN, "mauchly_p": _NAN, "eps_gg": 1.0, "eps_hf": 1.0}
    if p < 2:
        return res
    tr = float(np.trace(S))
    tr2 = float(np.trace(S @ S))
    if tr <= 1e-12 or tr2 <= 1e-12:
        return res
    eps = tr * tr / (p * tr2)
    res["eps_gg"] = float(min(1.0, max(1.0 / p, eps)))
    det = float(np.linalg.det(S))
    if det > 0 and n_eff > p:
        w = det / ((tr / p) ** p)
        w = float(min(1.0, max(w, 1e-300)))
        f = 1.0 - (2 * p * p + p + 2) / (6.0 * p * n_eff)
        chi = -n_eff * f * math.log(w)
        res.update(mauchly_w=w, mauchly_chi2=float(chi), mauchly_p=_p_chi2(chi, p * (p + 1) / 2.0 - 1))
    denom = p * (n_eff - p * eps)
    if denom > 0:
        res["eps_hf"] = float(min(1.0, max(eps, (( n_eff + 1) * p * eps - 2) / denom)))
    return res


def rm_anova(Y: Any, levels: Optional[Sequence[int]] = None, names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Repeated-measures ANOVA over the columns of ``Y``.

    ``levels`` gives the number of levels of each within factor; the columns of ``Y`` are the product of
    the levels with the FIRST factor varying slowest (a 2 x 3 design has columns A1B1 A1B2 A1B3 A2B1 ...).
    The default is one factor with ``Y.shape[1]`` levels. One row per term (each factor and each
    interaction): F (sphericity assumed), uncorrected and Greenhouse-Geisser / Huynh-Feldt p, partial and
    generalised eta squared (generalised only for the one-factor case), and Mauchly's test.
    """
    (Ya,) = complete_cases(np.asarray(Y, dtype=float))
    n, K = Ya.shape if Ya.ndim == 2 else (0, 0)
    levels = list(levels) if levels else [K]
    names = list(names) if names else ([f"Factor {i + 1}" for i in range(len(levels))] if len(levels) > 1
                                       else ["Condition"])
    out: Dict[str, Any] = {"n": int(n), "levels": levels, "names": names, "terms": [], "ok": False}
    if n < 3 or K < 2 or int(np.prod(levels)) != K:
        return out
    grand_sub = Ya.mean(1)
    ss_subj = K * float(np.sum((grand_sub - grand_sub.mean()) ** 2))
    for term in _terms(len(levels)):
        L = _term_matrix(levels, term)
        Z = Ya @ L.T
        df1 = L.shape[0]
        zbar = Z.mean(0)
        ss_eff = n * float(np.sum(zbar ** 2))
        resid = Z - zbar[None, :]
        ss_err = float(np.sum(resid ** 2))
        df2 = df1 * (n - 1)
        row: Dict[str, Any] = {"term": _term_name(term, names), "df1": df1, "df2": df2, "ss": ss_eff,
                               "ss_error": ss_err, "ms": ss_eff / df1, "ms_error": ss_err / df2 if df2 else _NAN}
        if ss_err <= 1e-12:
            row.update(f=float("inf") if ss_eff > 1e-12 else 0.0, p=0.0 if ss_eff > 1e-12 else 1.0)
        else:
            f = (ss_eff / df1) / (ss_err / df2)
            row.update(f=float(f), p=_p_f(f, df1, df2))
        S = np.cov(Z.T, ddof=1).reshape(df1, df1) if df1 > 0 else np.zeros((1, 1))
        sph = _sphericity(S, n - 1)
        row.update(sph)
        if df1 > 1 and math.isfinite(row["f"]):
            row["p_gg"] = _p_f(row["f"], df1 * sph["eps_gg"], df2 * sph["eps_gg"])
            row["p_hf"] = _p_f(row["f"], df1 * sph["eps_hf"], df2 * sph["eps_hf"])
        else:
            row["p_gg"] = row["p_hf"] = row["p"]
        row["partial_eta2"] = float(ss_eff / (ss_eff + ss_err)) if (ss_eff + ss_err) > 0 else _NAN
        if len(levels) == 1:
            tot = ss_eff + ss_subj + ss_err
            row["generalized_eta2"] = float(ss_eff / tot) if tot > 0 else _NAN
        out["terms"].append(row)
    out["ok"] = True
    out["ss_subjects"] = ss_subj
    return out


def _sum_code(labels: np.ndarray) -> Tuple[np.ndarray, List[Any]]:
    """Sum-to-zero (effect) coding of one factor: (n x (m-1)) columns, last level = -1 on every column."""
    levels = list(dict.fromkeys(labels.tolist()))
    m = len(levels)
    X = np.zeros((len(labels), max(m - 1, 0)))
    for j in range(m - 1):
        X[labels == levels[j], j] = 1.0
        X[labels == levels[-1], j] = -1.0
    return X, levels


def _between_columns(factor_labels: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[Tuple[int, ...], slice]]]:
    """Design matrix of a full-factorial between-subjects model in sum coding (intercept first) and the column
    block of every term (a tuple of factor indices)."""
    n = len(factor_labels[0])
    mains = [_sum_code(lab)[0] for lab in factor_labels]
    cols = [np.ones((n, 1))]
    blocks: List[Tuple[Tuple[int, ...], slice]] = []
    start = 1
    for size in range(1, len(factor_labels) + 1):
        for combo in itertools.combinations(range(len(factor_labels)), size):
            M = mains[combo[0]]
            for c in combo[1:]:
                M = (M[:, :, None] * mains[c][:, None, :]).reshape(n, -1)
            cols.append(M)
            blocks.append((combo, slice(start, start + M.shape[1])))
            start += M.shape[1]
    return np.hstack(cols), blocks


def _rss(X: np.ndarray, Z: np.ndarray) -> float:
    beta = np.linalg.lstsq(X, Z, rcond=None)[0]
    return float(((Z - X @ beta) ** 2).sum())


def mixed_anova(Y: Any, groups: Any, levels: Optional[Sequence[int]] = None,
                names: Optional[Sequence[str]] = None, between_name: str = "Group",
                between_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Mixed ANOVA: between-subjects factor(s) x the within factor(s) in the columns of ``Y``.

    ``groups`` is one label per participant (one pooled grouping variable, named ``between_name``) or an
    ``n x B`` array / list of B label vectors, one per between-subjects FACTOR (named ``between_names``): the
    between factors then enter as full factors with all their interactions. Type III sums of squares with
    sum-to-zero coding (the test of each term adjusted for all others, as SPSS and R's ``car::Anova(type=3)`` report
    them); for balanced cells they equal the classical Type I values. Rows: every between term, every within term
    and every between x within interaction. Within terms share the pooled within-cell covariance, so Mauchly's
    test and the Greenhouse-Geisser / Huynh-Feldt corrections use it. Complete cases only.
    """
    Ya = np.asarray(Y, dtype=float)
    ga = np.asarray(groups, dtype=object)
    if ga.ndim == 1:
        ga = ga[:, None]
    keep = np.all(np.isfinite(Ya), axis=1)
    Ya, ga = Ya[keep], ga[keep]
    n, K = Ya.shape if Ya.ndim == 2 else (0, 0)
    B = ga.shape[1]
    bnames = list(between_names) if between_names else ([between_name] if B == 1 else [f"Between {i + 1}" for i in range(B)])
    levels = list(levels) if levels else [K]
    names = list(names) if names else ([f"Factor {i + 1}" for i in range(len(levels))] if len(levels) > 1
                                       else ["Condition"])
    factor_labels = [np.array([str(v) for v in ga[:, j]]) for j in range(B)] if n else []
    out: Dict[str, Any] = {"n": int(n), "groups": [], "group_n": [], "levels": levels, "names": names,
                           "between_names": bnames, "terms": [], "ok": False}
    if n == 0 or K < 2 or int(np.prod(levels)) != K:
        return out
    cell_keys = [" x ".join(r) for r in zip(*factor_labels)]
    cells = list(dict.fromkeys(cell_keys))
    out["groups"] = cells
    out["group_n"] = [cell_keys.count(c) for c in cells]
    G = len(cells)
    X, blocks = _between_columns(factor_labels)
    rank = int(np.linalg.matrix_rank(X))
    df_err_unit = n - rank
    if G < 2 or df_err_unit < 2 or min(out["group_n"]) < 2 or any(len(set(f.tolist())) < 2 for f in factor_labels):
        return out

    def term_name(combo: Tuple[int, ...]) -> str:
        return " x ".join(bnames[i] for i in combo)

    def rows_for(Z: np.ndarray, within_label: Optional[str], within_df: int) -> List[Dict[str, Any]]:
        beta = np.linalg.lstsq(X, Z, rcond=None)[0]
        resid = Z - X @ beta
        rss_full = float((resid ** 2).sum())
        d = Z.shape[1]
        dfe = d * df_err_unit
        rows: List[Dict[str, Any]] = []
        spec_terms = [((), slice(0, 1))] + blocks
        sph = {"mauchly_w": _NAN, "mauchly_chi2": _NAN, "mauchly_p": _NAN, "eps_gg": 1.0, "eps_hf": 1.0}
        if within_label is not None and d > 1:
            sph = _sphericity((resid.T @ resid) / df_err_unit, df_err_unit)
        for combo, sl in spec_terms:
            if within_label is None and combo == ():
                continue                                    # the grand mean is not reported for the between part
            keep_cols = [c for c in range(X.shape[1]) if not (sl.start <= c < sl.stop)]
            ss = _rss(X[:, keep_cols], Z) - rss_full
            df1 = (sl.stop - sl.start) * d if within_label is not None else (sl.stop - sl.start)
            if within_label is None:
                label, typ = term_name(combo), "between"
            elif combo == ():
                label, typ = within_label, "within"
            else:
                label, typ = f"{term_name(combo)} x {within_label}", "interaction"
            row: Dict[str, Any] = {"term": label, "type": typ, "df1": df1, "df2": dfe, "ss": ss, "ss_error": rss_full,
                                   "ms": ss / df1, "ms_error": rss_full / dfe}
            if rss_full <= 1e-12:
                row.update(f=float("inf") if ss > 1e-12 else 0.0, p=0.0 if ss > 1e-12 else 1.0)
            else:
                f = (ss / df1) / (rss_full / dfe)
                row.update(f=float(f), p=_p_f(f, df1, dfe))
            row.update(sph if typ != "between" else {"mauchly_w": _NAN, "mauchly_chi2": _NAN, "mauchly_p": _NAN,
                                                     "eps_gg": 1.0, "eps_hf": 1.0})
            if typ != "between" and within_df > 1 and math.isfinite(row["f"]):
                row["p_gg"] = _p_f(row["f"], df1 * sph["eps_gg"], dfe * sph["eps_gg"])
                row["p_hf"] = _p_f(row["f"], df1 * sph["eps_hf"], dfe * sph["eps_hf"])
            else:
                row["p_gg"] = row["p_hf"] = row["p"]
            row["partial_eta2"] = float(ss / (ss + rss_full)) if (ss + rss_full) > 0 else _NAN
            rows.append(row)
        return rows

    out["terms"].extend(rows_for(Ya.mean(1, keepdims=True), None, 1))
    for term in _terms(len(levels)):
        L = _term_matrix(levels, term)
        out["terms"].extend(rows_for(Ya @ L.T, _term_name(term, names), L.shape[0]))
    out["method"] = "Type III sums of squares, sum-to-zero coding"
    out["ok"] = True
    return out


def interaction_by_difference_scores(x: Any, y: Any, groups: Any) -> Dict[str, Any]:
    """Group x time interaction for a two-level within factor: a t-test of the difference scores ``x - y``
    between the groups (two groups: Student's t, which equals sqrt(F) of the mixed ANOVA interaction),
    plus the paired change inside each group."""
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ga = np.asarray(groups)
    keep = np.isfinite(xa) & np.isfinite(ya)
    xa, ya, ga = xa[keep], ya[keep], ga[keep]
    labels = list(dict.fromkeys(ga.tolist()))
    diff = xa - ya
    per_group = []
    for lab in labels:
        m = ga == lab
        res = paired_t_test(xa[m], ya[m])
        res["group"] = lab
        per_group.append(res)
    out: Dict[str, Any] = {"groups": labels, "per_group": per_group, "t": _NAN, "df": _NAN, "p": _NAN,
                           "mean_diff_of_diffs": _NAN, "ci": (_NAN, _NAN), "d": _NAN}
    if len(labels) != 2:
        return out
    a, b = diff[ga == labels[0]], diff[ga == labels[1]]
    if len(a) < 2 or len(b) < 2:
        return out
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    dd = float(a.mean() - b.mean())
    if sp2 <= 1e-12:
        return out
    se = math.sqrt(sp2 * (1.0 / na + 1.0 / nb))
    t = dd / se
    df = na + nb - 2
    tc = _tcrit(df)
    out.update(t=float(t), df=float(df), p=_p_t(t, df), mean_diff_of_diffs=dd, ci=(dd - tc * se, dd + tc * se),
               d=float(dd / math.sqrt(sp2)))
    return out
