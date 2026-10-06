# simulation_app/utils/instructor_report.py
from __future__ import annotations
"""
Instructor Report Generator for Behavioral Experiment Simulation Tool
Generates comprehensive instructor-facing reports for student simulations.
"""

# Version identifier to help track deployed code
__version__ = "1.2.9.1"  # v1.2.9.1: report describes the effects actually built in (report-facing version stamp)

from dataclasses import dataclass
from types import SimpleNamespace
from datetime import datetime
import json
import base64
import io
import warnings
import re
import math
import html as _html_lib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import logging
try:  # neutralises active content in the finished HTML report (v1.2.9.1); never blocks report generation
    from .html_safety import harden_report_html as _harden_report_html
except ImportError:  # imported as a top-level module (scripts) or a partial deploy
    try:
        from html_safety import harden_report_html as _harden_report_html  # type: ignore[no-redef]
    except ImportError:
        def _harden_report_html(document: str) -> str:  # type: ignore[misc]
            return document
logger = logging.getLogger(__name__)

# Try multiple import strategies for scipy
SCIPY_AVAILABLE = False
scipy_stats = None

try:
    from scipy import stats as _scipy_stats
    scipy_stats = _scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    pass

if not SCIPY_AVAILABLE:
    try:
        import scipy.stats as _scipy_stats
        scipy_stats = _scipy_stats
        SCIPY_AVAILABLE = True
    except ImportError:
        pass

# Matplotlib imports
MATPLOTLIB_AVAILABLE = False
plt = None

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as _plt
    plt = _plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

# Import SVG chart generators (guaranteed fallback - no external dependencies)
try:
    from . import svg_charts
    SVG_CHARTS_AVAILABLE = True
except ImportError:
    try:
        import svg_charts
        SVG_CHARTS_AVAILABLE = True
    except ImportError:
        SVG_CHARTS_AVAILABLE = False


def _report_clean_column_name(name: str) -> str:
    """Replicate engine's _clean_column_name for column matching.

    v1.0.6.3: The engine uses _clean_column_name(variable_name) to generate
    DataFrame column prefixes.  The report must use the same logic to find them.
    """
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    clean = re.sub(r'_+', '_', clean)
    return clean.strip('_') or "Variable"


def _numeric_column_names(df: "pd.DataFrame") -> List[str]:
    """Names of the numeric (non-boolean) columns of ``df``, in column order.

    Text answers (open-ended questions) must never reach a mean, SD or range: ``df[cols].min()`` on a
    text column raised a UFuncTypeError that used to replace both instructor analyses with a stub.
    """
    names: List[str] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            names.append(name)
    return names


def _scale_column_registry(metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    """Map each scale name to the item columns the engine generated for it (``scale_generation_log``)."""
    registry: Dict[str, List[str]] = {}
    for entry in (metadata.get("scale_generation_log") or []):
        if isinstance(entry, dict):
            registry[str(entry.get("name", ""))] = list(entry.get("columns_generated") or [])
    return registry


def _find_scale_columns(df: "pd.DataFrame", scale: Dict[str, Any],
                        col_registry: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Find DataFrame columns for a scale using multiple name strategies.

    v1.0.6.3: Fixes N=1 bug where report couldn't find scale columns because
    the engine uses _clean_column_name(variable_name) but the report was only
    trying name.replace(' ', '_').

    Tries in order:
    1. Column registry (from scale_generation_log metadata)
    2. variable_name cleaned with _report_clean_column_name
    3. name cleaned with _report_clean_column_name
    4. Legacy: name.replace(' ', '_')

    v1.2.9.1: only numeric columns are returned. A prefix match used to pick up an open-ended TEXT column
    whose name starts with the scale name (``Punitive_Pilot_03`` for scale ``Punitive_Pilot``), and the text
    then reached numeric statistics. When several columns share the prefix, the ones whose remainder is
    purely digits (``<prefix>_1``, ``<prefix>_2``, ...) win over longer names such as ``<prefix>_Pilot_1``.
    """
    scale_name = scale.get("name", "Scale")
    numeric = _numeric_column_names(df)
    numeric_set = set(numeric)

    def _by_prefix(prefix: str) -> List[str]:
        lead = f"{prefix}_"
        loose = [c for c in numeric if str(c).startswith(lead) and str(c)[-1:].isdigit()]
        strict = [c for c in loose if str(c)[len(lead):].isdigit()]
        return sorted(strict or loose, key=str)

    # Strategy 1: Column registry from engine's scale_generation_log
    if col_registry:
        cols = col_registry.get(scale_name, [])
        if cols:
            existing = [c for c in cols if c in numeric_set]
            if existing:
                return existing

    # Strategy 2: variable_name with clean column name (matches engine)
    var_name = str(scale.get("variable_name", "")).strip()
    if var_name:
        cols = _by_prefix(_report_clean_column_name(var_name))
        if cols:
            return cols

    # Strategy 3: display name with clean column name
    cols = _by_prefix(_report_clean_column_name(scale_name))
    if cols:
        return cols

    # Strategy 4: Legacy fallback — simple space-to-underscore
    return _by_prefix(str(scale_name).replace(' ', '_'))


def _hypothesis_text(hypothesis: Any) -> str:
    """Return the wording of one extracted hypothesis.

    ``_parse_prereg_hypotheses`` returns plain strings, but an older/other caller may hand over
    ``{"text": ...}`` dicts; both are accepted so the executive summary can never fail on the shape.
    """
    if isinstance(hypothesis, dict):
        for key in ("text", "hypothesis", "statement"):
            value = hypothesis.get(key)
            if value:
                return str(value).strip()
        return ""
    return "" if hypothesis is None else str(hypothesis).strip()


def _is_finite_number(value: Any) -> bool:
    """True for a real, finite number (bools, None, NaN and +/-inf are not)."""
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _report_p_cell(p: Any, html: bool = False) -> str:
    """p-value for a table cell: ``< .001`` below .001, ``0.0432`` otherwise, ``n/a`` if unusable."""
    if not _is_finite_number(p):
        return "n/a"
    value = float(p)
    if value < 0.001:
        return "&lt; .001" if html else "< .001"
    return f"{value:.4f}"


def _report_p_text(p: Any, html: bool = False) -> str:
    """Format a p-value for running text: ``p < .001`` below .001, ``p = 0.0432`` otherwise, ``p n/a`` if unusable.

    A p-value that rounds to 0.0000 must never be printed as ``p = 0.0000``, and NaN/inf never reach the page.
    ``html=True`` writes the less-than sign as ``&lt;``.
    """
    cell = _report_p_cell(p, html=html)
    if cell == "n/a":
        return "p n/a"
    return f"p {cell}" if cell.startswith(("<", "&lt;")) else f"p = {cell}"


def _fnum(value: Any, fmt: str = ".2f", na: str = "n/a") -> str:
    """Format ``value`` with ``fmt``; return ``na`` when it is missing or not finite (never prints nan/inf)."""
    if not _is_finite_number(value):
        return na
    return format(float(value), fmt)


# DV types whose item columns are tied together: a ranking uses each rank once, a constant-sum question splits
# a fixed total. The mean of such items is the same for every participant, so it says nothing about conditions.
_JOINT_ITEM_SCALE_TYPES = frozenset({"rank_order", "ranking", "rank_order_scale", "constant_sum", "constantsum"})


def _is_joint_item_scale(scale: Dict[str, Any]) -> bool:
    """True when the scale is a constant-sum or rank-order question (items sum to a constant)."""
    kind = re.sub(r"[\s\-]+", "_", str(scale.get("type", "") or "").strip().lower())
    return kind in _JOINT_ITEM_SCALE_TYPES


def _composite_has_no_variation(composite: Any, condition: Any = None) -> bool:
    """True when every condition group (with 2+ valid values) is constant, so no test can be computed.

    With zero variance in every group a t-test or ANOVA divides by zero: scipy answers NaN or +/-inf with
    p = NaN or 0, and the numpy fallbacks answer t = 0, p = 1; none of it is a result. The check uses the
    range of each group with a relative tolerance, because a constant-sum mean such as 100 / 3 carries float
    noise of about 1e-14 that would otherwise look like variation and give t = 2,500,000.
    """
    try:
        values = pd.to_numeric(pd.Series(composite), errors="coerce")
        valid = values.notna()
        if int(valid.sum()) < 2:
            return False

        def _flat(group: "pd.Series") -> bool:
            arr = group.to_numpy(dtype=float)
            spread = float(np.nanmax(arr) - np.nanmin(arr))
            return spread <= 1e-9 * max(1.0, float(np.nanmax(np.abs(arr))))

        if condition is None:
            return _flat(values[valid])
        cond = pd.Series(condition).reindex(values.index)
        groups = [values[valid & (cond == c)] for c in cond[valid].dropna().unique()]
        sized = [g for g in groups if len(g) >= 2]
        return bool(sized) and all(_flat(g) for g in sized)
    except (TypeError, ValueError):
        logger.warning("Could not check the composite for variation", exc_info=True)
        return False


def _finite_results_only(results: Any) -> Any:
    """Copy of a statistics result dict without any test, row or comparison whose numbers are not finite.

    A zero-variance pair gives t = NaN or +/-inf with p = NaN or 0, and +/-inf is not "significant".
    A test like that says nothing, so it is left out instead of printing ``t = nan`` or ``p = 0.0000``.
    Non-dict input is returned unchanged.
    """
    if not isinstance(results, dict):
        return results

    def _all_finite(node: Any) -> bool:
        if isinstance(node, dict):
            return all(_all_finite(v) for v in node.values())
        if isinstance(node, (list, tuple)):
            return all(_all_finite(v) for v in node)
        if isinstance(node, (float, np.floating)):
            return bool(np.isfinite(node))
        return True

    cleaned: Dict[str, Any] = {}
    for key, value in results.items():
        if isinstance(value, dict):
            nested = [v for v in value.values() if isinstance(v, dict)]
            if nested and len(nested) == len(value):  # rows keyed by name (coefficients, cell statistics, ...)
                kept = {k: v for k, v in value.items() if _all_finite(v)}
                if kept:
                    cleaned[key] = kept
            elif _all_finite(value):
                cleaned[key] = value
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            kept_rows = [v for v in value if _all_finite(v)]
            if kept_rows:
                cleaned[key] = kept_rows
        elif _all_finite(value):
            cleaned[key] = value
    return cleaned


def _report_context(df: pd.DataFrame, metadata: Dict[str, Any], prereg_text: str,
                    team_info: Optional[Dict[str, Any]], html: bool) -> SimpleNamespace:
    """Shared state of one report run: the output list plus the values that several sections need.

    These values used to be computed inside the first section that needed them, so a failure there left every
    later section without them. They are computed once, defensively, before any section runs.
    """
    n_total = len(df)
    try:
        n_excluded = int(df["Exclude_Recommended"].sum()) if "Exclude_Recommended" in df.columns else 0
    except (TypeError, ValueError):
        logger.warning("Exclude_Recommended could not be summed; assuming no recommended exclusions", exc_info=True)
        n_excluded = 0
    conditions = list(metadata.get("conditions") or [])
    try:
        df = _align_condition_values(df, [str(c) for c in conditions])
    except (TypeError, ValueError, KeyError):
        logger.warning("Condition labels could not be aligned with the metadata; using them as they are", exc_info=True)
    # v1.0.6.3: Fallback (HTML) - if the metadata conditions do not match the data, use the data's own values
    if html and conditions and "CONDITION" in df.columns:
        data_conditions = df["CONDITION"].unique().tolist()
        if data_conditions and sum(1 for c in conditions if c in data_conditions) == 0:
            conditions = data_conditions
    # v1.2.9.1: ONE collision-free display label per condition (Group_1 / Group_2 stay distinct) for every
    # table, test and contrast; the ordering used everywhere is the metadata order held in ``conditions``.
    cond_labels = _condition_display_labels(
        list(conditions) + (df["CONDITION"].dropna().unique().tolist() if "CONDITION" in df.columns else []))

    def _cond_label(cond: Any) -> str:
        return cond_labels.get(cond) or _clean_condition_name(cond)

    return SimpleNamespace(
        out=[], html=html, df=df, cond_labels=cond_labels, cond_label=_cond_label,
        # v1.2.4.0: the FULL dataset is analysed. Exclude_Recommended is informational (it teaches students about
        # data quality) and must NOT reduce the analysis N: instructors expect the full sample.
        df_clean=df,
        metadata=metadata, prereg_text=prereg_text, team_info=team_info,
        n_total=n_total, n_excluded=n_excluded, n_clean=n_total - n_excluded,
        exclusion_rate=(n_excluded / n_total * 100) if n_total > 0 else 0,
        conditions=conditions, factors=metadata.get("factors") or [], scales=metadata.get("scales") or [],
        col_registry={}, dv_batch=[], all_scale_results=[], exec_summary_index=None,
    )


_P_ZERO_TEXT = re.compile(r"\bp\s*=\s*0\.0{3,4}(?![0-9])")
_P_ZERO_CELL = re.compile(r"(<td class=['\"](?:sig|marginal|nonsig)['\"]>)\s*0\.0{3,4}\s*(</td>)")


def _finalize_p_text(text: str, html: bool) -> str:
    """Last-line guard for a finished report: a p-value that prints as 0.0000 (or 0.000) is written ``< .001``.

    The statistics boxes, chart annotations and p columns format p with a fixed number of decimals, which turns
    every p below .00005 into ``p = 0.0000``. Only the unambiguous forms are rewritten: the phrase ``p = 0.000(0)``
    and, in HTML, the significance-coloured p cells of the tables (``<td class='sig'>0.0000</td>``).
    """
    less = "&lt;" if html else "<"
    text = _P_ZERO_TEXT.sub(f"p {less} .001", text)
    if html:
        text = _P_ZERO_CELL.sub(r"\1&lt; .001\2", text)
    return text


def _no_variation_note(scale: Dict[str, Any], n_items: int, mean: Any) -> str:
    """One truthful sentence (plain text) for a measure whose composite cannot be tested."""
    kind = re.sub(r"[\s\-]+", "_", str(scale.get("type", "") or "").strip().lower())
    m = _fnum(mean, ".2f")
    if kind in ("constant_sum", "constantsum"):
        return (f"The {n_items} options of this question are shares of one fixed total, so their mean is fixed by design "
                f"({m} for everyone who answered all of them) and cannot differ between conditions. The composite is not "
                "analysed; each option is compared across conditions instead.")
    if _is_joint_item_scale(scale):
        return (f"The {n_items} items of this question are ranks, so their mean rank is fixed by design "
                f"({m} for everyone who ranked all of them) and cannot differ between conditions. The composite is not "
                "analysed; each item is compared across conditions instead.")
    return (f"Every participant in every condition has the same score on this measure (M = {m}), so there is no variation "
            "to analyse and no test can be computed.")


# ---------------------------------------------------------------------------
# Embedded analysis-script helpers (v1.2.8.9)
#
# The delivered Simulated_Data.csv is a faithful Qualtrics-style export: item
# columns only. It has NO "<Scale>_mean" and NO "Exclude_Recommended" column
# (those live in Simulation_Diagnostics.csv, keyed on ResponseId). Every snippet
# the report embeds therefore computes composites from the item columns
# (reverse-keyed items recoded first) and merges the diagnostics file before
# filtering. These helpers mirror the engine's export-script convention
# (flip = scale_min + scale_max; recoded columns are "<item>_R", Stata "<item>_r").
# ---------------------------------------------------------------------------

_SCRIPT_RESERVED_COLUMNS = {"CONDITION", "Age", "Gender", "ResponseId"}


def _script_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _script_scales(
    metadata: Dict[str, Any],
    df: Optional["pd.DataFrame"] = None,
    lowercase: bool = False,
) -> List[Dict[str, Any]]:
    """Scale specs for embedded scripts (item columns, reverse items, flip constant).

    When ``df`` is given, only items present as columns are referenced.
    """
    cols = set(df.columns) if df is not None and hasattr(df, "columns") else None
    out: List[Dict[str, Any]] = []
    gen_log = [e for e in (metadata.get("scale_generation_log") or []) if isinstance(e, dict)]
    used: set = set()
    for scale in (metadata.get("scales") or []):
        if not isinstance(scale, dict):
            continue
        raw = str(scale.get("name", "Scale")).strip() or "Scale"
        name = _report_clean_column_name(raw)
        num_items = max(1, _script_int(scale.get("num_items", 5), 5))
        points = _script_int(scale.get("scale_points", 7), 7)
        reverse = set()
        rev_raw = scale.get("reverse_items") or []
        if isinstance(rev_raw, (list, tuple, set)):
            for x in rev_raw:
                try:
                    reverse.add(int(x))
                except (TypeError, ValueError):
                    logging.getLogger(__name__).debug("Skipping invalid reverse item %r", x)
        smin = _script_int(scale.get("scale_min", 1), 1)
        smax = _script_int(scale.get("scale_max", points), points)
        items = [f"{name}_{i}" for i in range(1, num_items + 1)]
        # Prefer the columns the generator recorded: their prefix comes from variable_name
        # and is de-duplicated, so it can differ from the display name.
        k = next((i for i, e in enumerate(gen_log)
                  if i not in used and str(e.get("name", "")).strip() == raw), None)
        if k is not None:
            used.add(k)
            gen_cols = [str(c) for c in (gen_log[k].get("columns_generated") or [])]
            if gen_cols:
                name = gen_cols[0].rsplit("_", 1)[0]
                items = gen_cols
                reverse = {int(x) for x in (gen_log[k].get("reverse_items") or []) if str(x).lstrip("-").isdigit()} or reverse
        if cols is not None:
            items = [it for it in items if it in cols]
            if not items:
                continue
        rev = sorted(r for r in reverse if f"{name}_{r}" in items)
        if lowercase:
            name = name.lower()
            items = [it.lower() for it in items]
        sfx = "_r" if lowercase else "_R"
        rev_cols = {f"{name}_{r}" for r in rev}
        composite_items = [(f"{it}{sfx}" if it in rev_cols else it) for it in items]
        out.append({
            "raw": raw, "name": name, "items": items, "reverse": rev,
            "points": points, "flip": smin + smax,
            "composite_items": composite_items, "composite": f"{name}_composite",
        })
    return out


def _script_factor_map(
    conditions: List[Any], factors: List[Dict[str, Any]]
) -> Optional[List[Tuple[str, Dict[str, str]]]]:
    """Map each condition label to one level per factor (None if not resolvable).

    The delivered CSV only has CONDITION, so scripts derive factor columns from it.
    Returns [(factor_column_name, {condition: level})] or None.
    """
    factors = [f for f in (factors or []) if isinstance(f, dict)]
    if len(factors) < 2 or not conditions:
        return None
    result: List[Tuple[str, Dict[str, str]]] = []
    used: set = set()
    for idx, f in enumerate(factors):
        fname = _report_clean_column_name(f.get("name", f"Factor{idx + 1}"))
        if fname in _SCRIPT_RESERVED_COLUMNS or fname in used:
            fname = f"{fname}_factor"
        used.add(fname)
        levels = [str(l) for l in (f.get("levels") or [])]
        if len(levels) < 2:
            return None
        mapping: Dict[str, str] = {}
        for cond in conditions:
            c = str(cond)
            parts = [x.strip().lower() for x in re.split(
                r"\s+[x\u00d7]\s+|\s*[|,;/+&]\s*|\s+-\s+|\s*_\s*", c) if x.strip()]
            hits = [l for l in levels if l.lower() in parts]  # whole-part match first
            if not hits:
                hits = [l for l in levels
                        if re.search(r"(?<![A-Za-z0-9])" + re.escape(l) + r"(?![A-Za-z0-9])", c, flags=re.I)]
            if len(hits) > 1:  # prefer the longest match
                hits = [max(hits, key=len)]
            if len(hits) != 1:
                return None
            mapping[c] = hits[0]
        if len(set(mapping.values())) < 2:
            return None
        result.append((fname, mapping))
    cells = {tuple(m[str(c)] for _, m in result) for c in conditions}
    if len(cells) != len(conditions):
        return None
    return result


def _py_lit(x: Any) -> str:
    return repr(str(x))


def _r_lit(x: Any) -> str:
    s = str(x).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _py_prep_lines(
    scales: List[Dict[str, Any]],
    factor_map: Optional[List[Tuple[str, Dict[str, str]]]] = None,
    clean_name: str = "df_clean",
) -> List[str]:
    """Python: composites from items (reverse-keyed recoded), diagnostics merge, clean df."""
    L: List[str] = []
    for sc in scales:
        if sc["reverse"]:
            L.append(f"# {sc['raw']}: reverse-keyed items {sc['reverse']} are recoded first ({sc['flip']} - x)")
            for r in sc["reverse"]:
                it = f"{sc['name']}_{r}"
                L.append(f"df['{it}_R'] = {sc['flip']} - df['{it}']")
        items = ", ".join(f"'{i}'" for i in sc["composite_items"])
        L.append(f"df['{sc['composite']}'] = df[[{items}]].mean(axis=1)  # {sc['raw']} composite")
    if factor_map:
        L.append("# The CSV has one CONDITION column; derive the factor columns from it")
        for fname, mapping in factor_map:
            pairs = ", ".join(f"{_py_lit(k)}: {_py_lit(v)}" for k, v in mapping.items())
            L.append(f"df['{fname}'] = df['CONDITION'].astype(str).map({{{pairs}}})")
    L.extend([
        "# Exclude_Recommended lives in Simulation_Diagnostics.csv; join it on ResponseId",
        "if os.path.exists('Simulation_Diagnostics.csv'):",
        "    diagnostics = pd.read_csv('Simulation_Diagnostics.csv')",
        "    df = df.merge(diagnostics[['ResponseId', 'Exclude_Recommended']], on='ResponseId', how='left')",
        f"    {clean_name} = df[df['Exclude_Recommended'] == 0].copy()",
        "else:",
        "    print('Simulation_Diagnostics.csv not found; no exclusions applied.')",
        f"    {clean_name} = df.copy()",
    ])
    return L


def _r_prep_lines(
    scales: List[Dict[str, Any]],
    factor_map: Optional[List[Tuple[str, Dict[str, str]]]] = None,
    data: str = "df",
    clean: str = "df_clean",
) -> List[str]:
    """R: composites from items (reverse-keyed recoded), diagnostics join, clean data frame."""
    L: List[str] = []
    for sc in scales:
        if sc["reverse"]:
            L.append(f"# {sc['raw']}: reverse-keyed items {sc['reverse']} are recoded first ({sc['flip']} - x)")
            for r in sc["reverse"]:
                it = f"{sc['name']}_{r}"
                L.append(f"{data}${it}_R <- {sc['flip']} - {data}${it}")
        refs = ", ".join(f"{data}${i}" for i in sc["composite_items"])
        L.append(f"{data}${sc['composite']} <- rowMeans(cbind({refs}), na.rm = TRUE)  # {sc['raw']} composite")
    if factor_map:
        L.append("# The CSV has one CONDITION column; derive the factor columns from it")
        for fname, mapping in factor_map:
            pairs = ", ".join(f"{_r_lit(k)} = {_r_lit(v)}" for k, v in mapping.items())
            L.append(f"{data}${fname} <- factor(unname(c({pairs})[as.character({data}$CONDITION)]))")
    L.extend([
        "# Exclude_Recommended lives in Simulation_Diagnostics.csv; join it on ResponseId",
        'if (file.exists("Simulation_Diagnostics.csv")) {',
        '  diagnostics <- read_csv("Simulation_Diagnostics.csv", show_col_types = FALSE)',
        f'  {data} <- left_join({data}, diagnostics[, c("ResponseId", "Exclude_Recommended")], by = "ResponseId")',
        f"  {clean} <- {data}[!is.na({data}$Exclude_Recommended) & {data}$Exclude_Recommended == 0, ]",
        "} else {",
        '  message("Simulation_Diagnostics.csv not found; no exclusions applied.")',
        f"  {clean} <- {data}",
        "}",
    ])
    return L



def _extract_persona_proportions(persona_dist: Any) -> Dict[str, float]:
    """Extract a flat {persona_name: proportion} dict from persona_distribution metadata.

    v1.2.3: The persona_distribution metadata was changed to a nested dict with
    'counts', 'proportions', and 'total_participants' keys. This function handles
    both the new nested format and the old flat format for backwards compatibility.

    Args:
        persona_dist: Either a flat dict {name: proportion} or nested dict with
                      'proportions' key.

    Returns:
        Flat dict mapping persona names to float proportions (0-1 range).
    """
    if not persona_dist or not isinstance(persona_dist, dict):
        return {}

    # New nested format: {"counts": {...}, "proportions": {...}, "total_participants": N}
    if "proportions" in persona_dist and isinstance(persona_dist["proportions"], dict):
        raw = persona_dist["proportions"]
        result = {}
        for k, v in raw.items():
            try:
                result[str(k)] = float(v)
            except (ValueError, TypeError):
                pass
        return result

    # Old flat format: {"Engaged Responder": 0.35, "Satisficer": 0.22, ...}
    # Check if values are numeric (not dicts/lists)
    result = {}
    for k, v in persona_dist.items():
        if isinstance(v, (int, float)):
            result[str(k)] = float(v)
        elif isinstance(v, str):
            try:
                result[str(k)] = float(v)
            except (ValueError, TypeError):
                pass
        # Skip dict/list values (sub-keys of new format without 'proportions')
    return result


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, handling dicts, None, NaN, and other edge cases.

    v1.2.3: Added to prevent float() crashes on unexpected types.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return default
        return float(value)
    if isinstance(value, dict):
        # Try to extract a numeric value from common dict structures
        for key in ('value', 'proportion', 'mean', 'count'):
            if key in value:
                try:
                    return float(value[key])
                except (ValueError, TypeError):
                    pass
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _clean_condition_name(condition: str) -> str:
    """Remove common suffixes and clean up condition names for report display."""
    if not condition:
        return condition
    cleaned = re.sub(r'\s*\(new\)', '', str(condition), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*\(copy\)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*- copy\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*_\d+$', '', cleaned)
    cleaned = re.sub(r'\s*\(\d+\)\s*$', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def _align_condition_values(df: pd.DataFrame, metadata_conditions: List[str]) -> pd.DataFrame:
    """Canonicalize CONDITION values against known metadata conditions.

    Handles accidental condition contamination like "Yeah AI x Utilitarian"
    by mapping to the canonical suffix when unambiguous.
    """
    if "CONDITION" not in df.columns or not metadata_conditions:
        return df

    canon = [str(c).strip() for c in metadata_conditions if str(c).strip()]
    if not canon:
        return df

    # A cleaned name shared by several conditions (Group_1 / Group_2 -> "group") is ambiguous: never
    # use it to relabel a row; a value that already IS a canonical condition is kept as it is.
    canon_set = set(canon)
    clean_counts: Dict[str, int] = {}
    for c in canon:
        key = _clean_condition_name(c).lower()
        clean_counts[key] = clean_counts.get(key, 0) + 1
    clean_to_canon = {_clean_condition_name(c).lower(): c for c in canon
                      if clean_counts[_clean_condition_name(c).lower()] == 1}
    mapped: List[Any] = []
    changes = 0

    for raw in df["CONDITION"].astype(str):
        if raw in canon_set:
            mapped.append(raw)
            continue
        raw_clean = _clean_condition_name(raw)
        raw_low = raw_clean.lower()

        if raw_low in clean_to_canon:
            mapped.append(clean_to_canon[raw_low])
            continue

        matches = [c for c in canon if raw_low.endswith(_clean_condition_name(c).lower())]
        if len(matches) == 1:
            mapped.append(matches[0])
            changes += 1
        else:
            mapped.append(raw)

    if changes == 0:
        return df

    out = df.copy()
    out["CONDITION"] = mapped
    return out



# ============================================================================
# DISTRIBUTION FUNCTIONS AND NUMPY-ONLY STATISTICAL TESTS
#
# The deployed app does not ship scipy (simulation_app/requirements.txt lists
# numpy and pandas only), so every p-value the report prints has to be
# computable without it.  The functions below are exact to double precision:
# the regularized incomplete beta function (Student t and F) uses a modified-
# Lentz continued fraction, the regularized incomplete gamma function (chi-
# square) a series / continued fraction pair.  When scipy IS importable the
# report keeps using it through the _p_* / _t_crit wrappers; the two paths agree
# to better than 1e-9 on every p-value (tests/test_report_statistics_v1291.py
# compares them on a dense grid).
# ============================================================================

_CF_TINY = 1e-300        # Lentz floor that keeps the continued fractions finite
_CF_TOL = 1e-15          # relative convergence target of the continued fractions
_CF_MAX_ITER = 100000    # hard stop; the fractions converge in O(sqrt(df)) steps


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (modified Lentz, DLMF 8.17.22)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _CF_TINY:
        d = _CF_TINY
    d = 1.0 / d
    h = d
    for m in range(1, _CF_MAX_ITER):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = 1.0 + aa / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = 1.0 + aa / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_TOL:
            break
    return h


def _betainc(a: float, b: float, x: float, y: Optional[float] = None) -> float:
    """Regularized incomplete beta function I_x(a, b), accurate to double precision.

    ``y`` may carry ``1 - x`` computed without cancellation (callers that know it
    in closed form pass it, e.g. ``t^2 / (df + t^2)`` for the t distribution).
    """
    if not (a > 0.0 and b > 0.0) or x != x:
        return float("nan")
    if y is None:
        y = 1.0 - x
    if x <= 0.0:
        return 0.0
    if y <= 0.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log(y))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return min(1.0, front * _betacf(a, b, x) / a)
    return max(0.0, 1.0 - front * _betacf(b, a, y) / b)


def _gammainc_reg(a: float, x: float, upper: bool) -> float:
    """Regularized incomplete gamma function: Q(a, x) when ``upper`` else P(a, x)."""
    if not a > 0.0 or x != x:
        return float("nan")
    if x <= 0.0:
        return 1.0 if upper else 0.0
    if math.isinf(x):
        return 0.0 if upper else 1.0
    ln_prefix = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:                      # series for P(a, x)
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(_CF_MAX_ITER):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        p = total * math.exp(ln_prefix)
        return max(0.0, 1.0 - p) if upper else min(1.0, p)
    b = x + 1.0 - a                      # modified-Lentz continued fraction for Q(a, x)
    c = 1.0 / _CF_TINY
    d = 1.0 / b
    h = d
    for i in range(1, _CF_MAX_ITER):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = b + an / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_TOL:
            break
    q = math.exp(ln_prefix) * h
    return min(1.0, q) if upper else max(0.0, 1.0 - q)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF (erfc form: no cancellation in the lower tail)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _normal_sf(x: float) -> float:
    """Upper-tail probability of the standard normal distribution."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation plus one Halley step; ~1e-15)."""
    if p != p or p < 0.0 or p > 1.0:
        return float("nan")
    if p == 0.0:
        return float("-inf")
    if p == 1.0:
        return float("inf")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
             / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    elif p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
             / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    else:
        q = math.sqrt(-2.0 * math.log1p(-p))
        x = -((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
              / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    err = _normal_cdf(x) - p
    u = err * math.sqrt(2.0 * math.pi) * math.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


def _t_sf(t: float, df: float) -> float:
    """Upper-tail probability P(T > t) of Student's t distribution with ``df`` degrees of freedom."""
    if t != t or df != df or not df > 0.0:
        return float("nan")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    if math.isinf(df):
        return _normal_sf(t)
    t2 = t * t
    denom = df + t2
    if math.isinf(denom):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * _betainc(0.5 * df, 0.5, df / denom, t2 / denom)
    return tail if t > 0 else 1.0 - tail


def _t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value of a t statistic: P(|T| >= |t|)."""
    if t != t or df != df or not df > 0.0:
        return float("nan")
    if math.isinf(t):
        return 0.0
    if math.isinf(df):
        return min(1.0, 2.0 * _normal_sf(abs(t)))
    t2 = t * t
    denom = df + t2
    if math.isinf(denom):
        return 0.0
    return min(1.0, _betainc(0.5 * df, 0.5, df / denom, t2 / denom))


def _t_pdf(t: float, df: float) -> float:
    """Density of Student's t distribution."""
    ln_c = math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df) - 0.5 * math.log(df * math.pi)
    return math.exp(ln_c - 0.5 * (df + 1.0) * math.log1p(t * t / df))


def _t_isf(p: float, df: float) -> float:
    """Inverse survival function of Student's t: the t with P(T > t) = p (0 < p < 1)."""
    if p != p or df != df or not df > 0.0 or p <= 0.0 or p >= 1.0:
        return float("nan")
    if p == 0.5:
        return 0.0
    if p > 0.5:
        return -_t_isf(1.0 - p, df)
    z = -_norm_ppf(p)                    # normal quantile: a good start for large df
    lo, hi = 0.0, max(1.0, z)
    for _ in range(2000):                # bracket the root: sf(hi) < p <= sf(lo)
        if _t_sf(hi, df) < p:
            break
        lo = hi
        hi *= 2.0
    t = min(max(z, lo), hi)
    for _ in range(200):                 # safeguarded Newton iteration
        f = _t_sf(t, df) - p
        if f > 0.0:
            lo = t
        else:
            hi = t
        pdf = _t_pdf(t, df)
        t_new = t + f / pdf if pdf > 0.0 else 0.5 * (lo + hi)
        if not lo < t_new < hi:
            t_new = 0.5 * (lo + hi)
        if abs(t_new - t) <= 1e-14 * max(1.0, abs(t_new)):
            return t_new
        t = t_new
    return t


def _t_ppf(q: float, df: float) -> float:
    """Quantile function of Student's t distribution."""
    if q != q or q < 0.0 or q > 1.0:
        return float("nan")
    if q == 0.0:
        return float("-inf")
    if q == 1.0:
        return float("inf")
    if q == 0.5:
        return 0.0
    return _t_isf(1.0 - q, df) if q > 0.5 else -_t_isf(q, df)


def _f_sf(f: float, df1: float, df2: float) -> float:
    """Upper-tail probability P(F > f) of the F distribution (exact to double precision)."""
    if f != f or df1 != df1 or df2 != df2 or not (df1 > 0.0 and df2 > 0.0):
        return float("nan")
    if f <= 0.0:
        return 1.0
    if math.isinf(f):
        return 0.0
    num = df1 * f
    denom = df2 + num
    if math.isinf(denom):
        return 0.0
    return _betainc(0.5 * df2, 0.5 * df1, df2 / denom, num / denom)


def _chi2_sf(x: float, df: float) -> float:
    """Upper-tail probability P(X > x) of the chi-square distribution."""
    if x != x or df != df or not df > 0.0:
        return float("nan")
    return _gammainc_reg(0.5 * df, 0.5 * x, True)


def _chi2_cdf(x: float, df: float) -> float:
    """CDF of the chi-square distribution (exact to double precision)."""
    if x != x or df != df or not df > 0.0:
        return float("nan")
    return _gammainc_reg(0.5 * df, 0.5 * x, False)


# --- scipy-preferring wrappers: every p-value / critical value in the report goes through these ---

def _use_scipy() -> bool:
    return bool(SCIPY_AVAILABLE and scipy_stats is not None)


def _p_t_two_sided(t: float, df: float) -> float:
    """Two-sided p-value for a t statistic (scipy when importable, otherwise the exact numpy version)."""
    if _use_scipy():
        try:
            return float(2.0 * scipy_stats.t.sf(abs(t), df))
        except Exception as exc:  # pragma: no cover - defensive: never block a report on scipy quirks
            logger.warning("scipy t.sf failed (%s); using the numpy implementation", exc)
    return _t_two_sided_p(t, df)


def _p_f_sf(f: float, df1: float, df2: float) -> float:
    """Upper-tail p-value for an F statistic."""
    if _use_scipy():
        try:
            return float(scipy_stats.f.sf(f, df1, df2))
        except Exception as exc:  # pragma: no cover
            logger.warning("scipy f.sf failed (%s); using the numpy implementation", exc)
    return _f_sf(f, df1, df2)


def _p_chi2_sf(x: float, df: float) -> float:
    """Upper-tail p-value for a chi-square statistic."""
    if _use_scipy():
        try:
            return float(scipy_stats.chi2.sf(x, df))
        except Exception as exc:  # pragma: no cover
            logger.warning("scipy chi2.sf failed (%s); using the numpy implementation", exc)
    return _chi2_sf(x, df)


def _t_crit(df: float, conf: float = 0.95) -> float:
    """Two-sided critical value of Student's t for a ``conf`` interval (1.96 only as df -> infinity)."""
    if df != df or not df > 0.0:
        return float("nan")
    tail = 0.5 * (1.0 - conf)
    if _use_scipy():
        try:
            return float(scipy_stats.t.isf(tail, df))
        except Exception as exc:  # pragma: no cover
            logger.warning("scipy t.isf failed (%s); using the numpy implementation", exc)
    return _t_isf_cached(round(float(tail), 12), round(float(df), 9))


_T_ISF_CACHE: Dict[Tuple[float, float], float] = {}


def _t_isf_cached(tail: float, df: float) -> float:
    """Memoised ``_t_isf`` (a report asks for the same few degrees of freedom hundreds of times)."""
    key = (tail, df)
    val = _T_ISF_CACHE.get(key)
    if val is None:
        val = _t_isf(tail, df)
        if len(_T_ISF_CACHE) > 4096:
            _T_ISF_CACHE.clear()
        _T_ISF_CACHE[key] = val
    return val


# --- numpy implementations of the tests (used when scipy is missing; also directly testable) ---

def _is_constant(values: Any) -> bool:
    """True when a sample has no variation at all (max == min), robust to float noise in var()."""
    arr = np.asarray(values, dtype=float)
    return arr.size > 0 and float(np.max(arr)) == float(np.min(arr))


def _numpy_ttest_ind(group1: np.ndarray, group2: np.ndarray, equal_var: bool = True) -> Tuple[float, float]:
    """Independent-samples t-test (pooled or Welch) with an exact two-sided p; (nan, nan) if undefined."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2 or (_is_constant(g1) and _is_constant(g2)):
        return float("nan"), float("nan")
    m1, m2 = float(np.mean(g1)), float(np.mean(g2))
    v1, v2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))
    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = math.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
    else:
        s1, s2 = v1 / n1, v2 / n2
        se = math.sqrt(s1 + s2)
        denom = (s1 * s1) / (n1 - 1) + (s2 * s2) / (n2 - 1)
        df = (s1 + s2) ** 2 / denom if denom > 0 else float("nan")
    if not se > 0.0 or df != df:
        return float("nan"), float("nan")
    t_stat = (m1 - m2) / se
    return float(t_stat), float(_t_two_sided_p(t_stat, df))


def _numpy_f_oneway(*groups) -> Tuple[float, float]:
    """One-way ANOVA with an exact p-value; (nan, nan) when the within-group variance is zero."""
    arrays = [np.asarray(g, dtype=float) for g in groups]
    k = len(arrays)
    n_total = sum(len(g) for g in arrays)
    if k < 2 or n_total <= k or all(_is_constant(g) for g in arrays):
        return float("nan"), float("nan")
    grand_mean = float(np.mean(np.concatenate(arrays)))
    ss_between = sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in arrays)
    ss_within = sum(float(np.sum((g - np.mean(g)) ** 2)) for g in arrays)
    df_between, df_within = k - 1, n_total - k
    if not ss_within > 0.0:
        return float("nan"), float("nan")
    f_stat = (ss_between / df_between) / (ss_within / df_within)
    return float(f_stat), float(_f_sf(f_stat, df_between, df_within))


def _rank_average(values: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Mid-ranks (ties share the average rank) and the sizes of the tie groups."""
    arr = np.asarray(values, dtype=float)
    uniq, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    mid = cum - (counts - 1) / 2.0
    return mid[inverse.reshape(-1)], counts


def _mwu_exact_sf(u_obs: int, n1: int, n2: int) -> float:
    """P(U >= u_obs) for the Mann-Whitney U statistic under H0 without ties (exact).

    The number of arrangements giving each U is the coefficient list of the Gaussian binomial
    [n1+n2 choose min(n1, n2)]_q = prod_{i=1..m} (1 - q^(n+i)) / (1 - q^i), built with vector operations.
    """
    m, n = min(n1, n2), max(n1, n2)
    size = m * n + 1
    counts = np.zeros(size)
    counts[0] = 1.0
    for i in range(1, m + 1):
        shifted = np.zeros(size)
        if n + i < size:
            shifted[n + i:] = counts[:size - (n + i)]
        counts = counts - shifted                     # multiply by (1 - q^(n+i))
        pad = (-size) % i                             # divide by (1 - q^i): cumulative sum with stride i
        padded = np.concatenate([counts, np.zeros(pad)]).reshape(-1, i)
        counts = np.cumsum(padded, axis=0).reshape(-1)[:size]
    total = float(counts.sum())
    u_obs = max(0, int(u_obs))
    return float(counts[u_obs:].sum()) / total if u_obs < size and total > 0 else 0.0


def _numpy_mannwhitneyu(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Two-sided Mann-Whitney U test matching scipy.stats.mannwhitneyu (method='auto').

    Returns (U1, p): mid-ranks for ties, tie-corrected variance and continuity correction;
    the exact distribution when there are no ties and a sample has <= 8 values.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    ranks, tie_counts = _rank_average(np.concatenate([g1, g2]))
    u1 = float(np.sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0)
    u2 = float(n1 * n2 - u1)
    u_big = max(u1, u2)
    has_ties = bool(tie_counts.max() > 1)
    if not has_ties and not (n1 > 8 and n2 > 8):      # scipy's method='auto' rule
        p = min(1.0, 2.0 * _mwu_exact_sf(int(round(u_big)), n1, n2))
        return u1, float(p)
    n = n1 + n2
    tie_term = float(np.sum(tie_counts.astype(float) ** 3 - tie_counts))
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if not var > 0.0:
        return u1, 1.0
    z = (u_big - n1 * n2 / 2.0 - 0.5) / math.sqrt(var)
    return u1, float(min(1.0, 2.0 * _normal_sf(z)))


def _numpy_kruskal(*groups) -> Tuple[float, float]:
    """Kruskal-Wallis H test with tie correction and an exact chi-square p (matches scipy.stats.kruskal)."""
    arrays = [np.asarray(g, dtype=float) for g in groups]
    k = len(arrays)
    n = sum(len(g) for g in arrays)
    if k < 2 or n < 2:
        return float("nan"), float("nan")
    ranks, tie_counts = _rank_average(np.concatenate(arrays))
    h = 0.0
    start = 0
    for g in arrays:
        r = ranks[start:start + len(g)]
        start += len(g)
        if len(g):
            h += float(np.sum(r)) ** 2 / len(g)
    h = 12.0 / (n * (n + 1)) * h - 3.0 * (n + 1)
    correction = 1.0 - float(np.sum(tie_counts.astype(float) ** 3 - tie_counts)) / (n ** 3 - n)
    if not correction > 0.0:
        return float("nan"), float("nan")
    h /= correction
    return float(h), float(_chi2_sf(h, k - 1))


def _numpy_levene(*groups) -> Tuple[float, float]:
    """Levene / Brown-Forsythe test of equal variances (absolute deviations from the median)."""
    arrays = [np.asarray(g, dtype=float) for g in groups]
    return _numpy_f_oneway(*[np.abs(g - np.median(g)) for g in arrays])


def _swilk_poly(coefs: Tuple[float, ...], x: float) -> float:
    """Polynomial c0 + c1 x + c2 x^2 + ... (Horner)."""
    total = 0.0
    for c in reversed(coefs):
        total = total * x + c
    return total


def _numpy_shapiro(data: np.ndarray) -> Tuple[float, float]:
    """Shapiro-Wilk W test (Royston's AS R94 algorithm, as used by scipy.stats.shapiro).

    Valid for 3 <= n <= 5000.  Returns (1.0, 1.0) for fewer than 3 values or constant data.
    """
    x = np.sort(np.asarray(data, dtype=float).ravel())
    n = len(x)
    if n < 3 or not float(x[-1] - x[0]) > 0.0:
        return 1.0, 1.0
    nn2 = n // 2
    if n == 3:
        a = [math.sqrt(0.5)]
    else:
        an25 = n + 0.25
        m = [_norm_ppf((i - 0.375) / an25) for i in range(1, nn2 + 1)]
        summ2 = 2.0 * sum(v * v for v in m)
        ssumm2 = math.sqrt(summ2)
        rsn = 1.0 / math.sqrt(n)
        c1 = (0.0, 0.221157, -0.147981, -2.07119, 4.434685, -2.706056)
        c2 = (0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633)
        a1 = _swilk_poly(c1, rsn) - m[0] / ssumm2
        if n > 5:
            a2 = -m[1] / ssumm2 + _swilk_poly(c2, rsn)
            fac = math.sqrt((summ2 - 2.0 * m[0] ** 2 - 2.0 * m[1] ** 2) / (1.0 - 2.0 * a1 ** 2 - 2.0 * a2 ** 2))
            a = [a1, a2] + [-m[i] / fac for i in range(2, nn2)]
        else:
            fac = math.sqrt((summ2 - 2.0 * m[0] ** 2) / (1.0 - 2.0 * a1 ** 2))
            a = [a1] + [-m[i] / fac for i in range(1, nn2)]
    coef = np.zeros(n)                   # antisymmetric coefficient vector
    for i, ai in enumerate(a):
        coef[i] = -ai
        coef[n - 1 - i] = ai
    xs = x / (x[-1] - x[0])
    asa = coef - coef.mean()
    xsx = xs - xs.mean()
    ssa = float(np.sum(asa * asa))
    ssx = float(np.sum(xsx * xsx))
    sax = float(np.sum(asa * xsx))
    ssassx = math.sqrt(ssa * ssx)
    w1 = (ssassx - sax) * (ssassx + sax) / (ssa * ssx)       # 1 - W without cancellation
    w = 1.0 - w1
    if n == 3:
        pi6, stqr = 6.0 / math.pi, math.pi / 3.0
        p = pi6 * (math.asin(math.sqrt(min(1.0, max(0.0, w)))) - stqr)
        return float(w), float(min(1.0, max(0.0, p)))
    y = math.log(w1) if w1 > 0.0 else -745.0
    xx = math.log(n)
    if n <= 11:
        gamma = _swilk_poly((-2.273, 0.459), float(n))
        if y >= gamma:
            return float(w), 1e-19
        y = -math.log(gamma - y)
        mu = _swilk_poly((0.544, -0.39978, 0.025054, -6.714e-4), float(n))
        sigma = math.exp(_swilk_poly((1.3822, -0.77857, 0.062767, -0.0020322), float(n)))
    else:
        mu = _swilk_poly((-1.5861, -0.31082, -0.083751, 0.0038915), xx)
        sigma = math.exp(_swilk_poly((-0.4803, -0.082676, 0.0030302), xx))
    return float(w), float(_normal_sf((y - mu) / sigma))


def _fisher_exact_2x2(table: Any) -> float:
    """Two-sided Fisher exact p-value for a 2x2 table (probabilities <= the observed one are summed)."""
    t = np.asarray(table, dtype=float)
    a, b, c, d = int(round(t[0, 0])), int(round(t[0, 1])), int(round(t[1, 0])), int(round(t[1, 1]))
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2
    if n == 0:
        return 1.0
    lo, hi = max(0, c1 - r2), min(c1, r1)

    def _log_pmf(k: int) -> float:
        return (math.lgamma(r1 + 1) - math.lgamma(k + 1) - math.lgamma(r1 - k + 1)
                + math.lgamma(r2 + 1) - math.lgamma(c1 - k + 1) - math.lgamma(r2 - c1 + k + 1)
                - (math.lgamma(n + 1) - math.lgamma(c1 + 1) - math.lgamma(n - c1 + 1)))

    probs = [math.exp(_log_pmf(k)) for k in range(lo, hi + 1)]
    p_obs = probs[a - lo]
    return float(min(1.0, sum(p for p in probs if p <= p_obs * (1.0 + 1e-7))))


def _numpy_chi2_contingency(observed: Any, correction: bool = True) -> Tuple[float, float, int, np.ndarray]:
    """Pearson chi-square test of independence matching scipy.stats.chi2_contingency.

    Applies Yates' continuity correction on 2x2 tables (``correction=True``), exactly as scipy does.
    Returns (chi2, p, dof, expected).
    """
    obs = np.asarray(observed, dtype=float)
    row = obs.sum(axis=1, keepdims=True)
    col = obs.sum(axis=0, keepdims=True)
    total = float(obs.sum())
    expected = row * col / total
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    if dof == 0:
        return 0.0, 1.0, 0, expected
    adj = obs
    if dof == 1 and correction:
        diff = expected - obs
        adj = obs + np.sign(diff) * np.minimum(0.5, np.abs(diff))
    chi2 = float(np.sum((adj - expected) ** 2 / expected))
    return chi2, float(_chi2_sf(chi2, dof)), int(dof), expected


# ============================================================================
# REPORT-LEVEL STATISTICS HELPERS (v1.2.9.1)
#
# One ordering, one set of labels and one number format for every table the
# report prints, so the markdown and HTML versions can never disagree.
# ============================================================================

def _order_conditions(present: List[Any], preferred: Optional[List[Any]] = None) -> List[Any]:
    """The ONE condition order used by every test, table and contrast in the report.

    ``preferred`` (the metadata ``conditions`` list) wins; conditions it does not mention follow in
    order of first appearance in the data.  Without ``preferred`` the order is first appearance.
    Two-group quantities (t, Cohen's d, pairwise contrasts) are always "first minus second" in it.
    """
    seen: Dict[Any, None] = {}
    for c in present:
        if c is not None and c == c:
            seen.setdefault(c, None)
    if not preferred:
        return list(seen)
    ordered: List[Any] = []
    for c in preferred:
        if c in seen and c not in ordered:
            ordered.append(c)
    ordered.extend(c for c in seen if c not in ordered)
    return ordered


def _condition_display_labels(conditions: List[Any]) -> Dict[Any, str]:
    """Display label for each condition: the cleaned name, but never one label for two conditions.

    ``_clean_condition_name`` strips trailing "_1" / "(2)" artifacts, which would merge genuinely
    different conditions such as Group_1 / Group_2 / Group_3 into one "Group".  Whenever cleaning
    would make two labels identical, the affected conditions keep their original text.
    """
    raw = list(dict.fromkeys(c for c in conditions if c is not None and c == c))
    cleaned = {c: (_clean_condition_name(str(c)) or str(c)) for c in raw}
    counts: Dict[str, int] = {}
    for lab in cleaned.values():
        counts[lab] = counts.get(lab, 0) + 1
    labels: Dict[Any, str] = {}
    for c in raw:
        lab = cleaned[c]
        if counts[lab] > 1:
            lab = re.sub(r"\s+", " ", str(c)).strip() or str(c)
        labels[c] = lab
    return labels


def _fmt_p(p: Any) -> str:
    """p-value text for a table cell ('< .001' for tiny values, never '0.0000'); delegates to ``_report_p_cell``."""
    return _report_p_cell(p)


def _p_eq(p: Any) -> str:
    """Statement form ('p = 0.0123' / 'p < .001' / 'p n/a'); delegates to ``_report_p_text``."""
    return _report_p_text(p)


def _fmt_p_html(p: Any) -> str:
    """HTML-safe table-cell p-value ('&lt; .001' for tiny values)."""
    return _report_p_cell(p, html=True)


def _p_eq_html(p: Any) -> str:
    """HTML-safe statement form ('p &lt; .001')."""
    return _report_p_text(p, html=True)


def _fmt_stat(x: Any, digits: int = 3) -> str:
    """Number with fixed decimals; 'undefined' for NaN / inf (e.g. a t statistic with zero variance)."""
    return _fnum(x, f".{digits}f", na="undefined")


def _cohens_d_label(d: Any) -> str:
    """The ONE verbal label for a Cohen's d, used by every table and sentence in the report.

    |d| < 0.1 negligible, < 0.2 very small, < 0.5 small, < 0.8 medium, < 1.2 large, otherwise very large.
    """
    try:
        v = abs(float(d))
    except (TypeError, ValueError):
        return "undefined"
    if v != v:
        return "undefined"
    if v < 0.1:
        return "negligible"
    if v < 0.2:
        return "very small"
    if v < 0.5:
        return "small"
    if v < 0.8:
        return "medium"
    if v < 1.2:
        return "large"
    return "very large"


def _t_ci_halfwidth(sd: float, n: int, conf: float = 0.95) -> float:
    """Half-width of a t-based confidence interval of a mean (0.0 when it cannot be computed)."""
    if n is None or n < 2 or sd != sd or not sd >= 0.0:
        return 0.0
    return float(_t_crit(n - 1, conf) * sd / math.sqrt(n))


def _holm_adjust(p_values: List[float]) -> List[float]:
    """Holm step-down adjusted p-values (NaN entries are left out of the family and stay NaN)."""
    idx = [i for i, p in enumerate(p_values) if p == p]
    adjusted = [float("nan")] * len(p_values)
    order = sorted(idx, key=lambda i: p_values[i])
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _name_tokens(name: Any) -> List[str]:
    """Lower-case alphanumeric words of a column or control name ('Participant_Age' -> participant, age)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


_CONTROL_STOPWORDS = {"and", "or", "the", "a", "an", "for", "as", "of", "with", "by", "on", "in", "to", "at",
                      "control", "controls", "controlling", "covariate", "covariates", "variable", "variables"}


def _looks_derived_from_a_dv(column: Any) -> bool:
    """Survey item / composite columns (Trust_3, Trust_mean, _composite) are never covariates."""
    name = str(column)
    return name.startswith("_") or bool(re.search(r"_(\d+|mean|composite|sum|total)$", name, re.IGNORECASE))


def _match_control_columns(df: pd.DataFrame, control_name: Any, exclude: Any = ()) -> List[str]:
    """Numeric columns that ARE the named control, matched by whole words (never by substring).

    'age' matches Age / Participant_Age but not Message_1, Engagement_2, Image_1 or Percentage;
    columns derived from a DV (items, composites, anything in ``exclude``) never match.
    """
    wanted = [t for t in _name_tokens(control_name) if t not in _CONTROL_STOPWORDS]
    if not wanted:
        return []
    banned = set(str(c) for c in exclude)
    candidates = []
    for col in df.columns:
        if str(col) in banned or _looks_derived_from_a_dv(col):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        candidates.append(col)
    exact = [c for c in candidates if _name_tokens(c) == wanted]
    if exact:
        return exact
    return [c for c in candidates if set(wanted) <= set(_name_tokens(c))]


def _find_demographic_column(df: pd.DataFrame, words: Tuple[str, ...], exclude: Any = (),
                             numeric: bool = True) -> Optional[str]:
    """The demographic column (Age, Gender, ...) named by whole word, preferring an exact name match."""
    banned = set(str(c) for c in exclude)
    exact: List[str] = []
    partial: List[str] = []
    for col in df.columns:
        if str(col) in banned or _looks_derived_from_a_dv(col):
            continue
        is_num = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
        if numeric and not is_num:
            continue
        tokens = _name_tokens(col)
        if len(tokens) == 1 and tokens[0] in words:
            exact.append(col)
        elif any(w in tokens for w in words):
            partial.append(col)
    for group in (exact, partial):
        if group:
            return group[0]
    return None


_CONDITION_NOISE_RE = re.compile(r"\s*[\(\[]\s*(?:new|copy|old|unused|duplicate)\s*[\)\]]|\s*-\s*copy\s*$", re.IGNORECASE)
_LEVEL_FILLER_WORDS = {"x", "and", "with", "by", "vs"}


def _parse_condition_levels(condition: Any, factors: List[Dict[str, Any]]) -> Optional[Tuple[str, ...]]:
    """The exact level of every factor named inside a condition label, or None.

    Matching is by whole token (a level is never found inside another word) and, within a factor,
    the longest level wins, so 'No AI x Hedonic' gives ('No AI', 'Hedonic') whatever the order of
    the factor's levels, 'Male' is not found in 'Female' and 'Receptive' not in 'Unreceptive'.
    Every factor must be matched and nothing but separators may remain, otherwise None.
    """
    text = re.sub(r"\s+", " ", _CONDITION_NOISE_RE.sub("", str(condition))).strip().lower()
    if not text:
        return None
    found: List[str] = []
    for factor in factors:
        levels = [str(lv) for lv in (factor.get("levels") or []) if str(lv).strip()]
        matches: List[Tuple[int, int, str]] = []
        for lv in levels:
            norm = re.sub(r"\s+", " ", lv).strip().lower()
            pattern = r"(?<![^\W_])" + re.escape(norm) + r"(?![^\W_])"
            for m in re.finditer(pattern, text):
                matches.append((m.start(), m.end(), lv))
        maximal = [m for m in matches
                   if not any(o[0] <= m[0] and m[1] <= o[1] and (o[1] - o[0]) > (m[1] - m[0]) for o in matches)]
        if not maximal:
            return None
        start, end, level = sorted(maximal, key=lambda m: (m[0], -(m[1] - m[0])))[0]
        found.append(level)
        text = text[:start] + " " * (end - start) + text[end:]
    leftovers = re.findall(r"[^\W_]+", text)
    if any(tok not in _LEVEL_FILLER_WORDS for tok in leftovers):
        return None
    return tuple(found)


def _resolve_factorial_design(conditions: List[Any], factors: List[Dict[str, Any]],
                              max_factors: int = 3) -> Dict[str, Any]:
    """Pick the factors a factorial ANOVA can analyse and map every condition to its levels.

    Tries the first ``max_factors`` factors, then fewer, and keeps the largest prefix whose
    condition cells form a COMPLETE crossing (every combination of the observed levels occurs).
    Returns {"ok": True, "factors": [...], "level_map": {condition: (level, ...)}, "unparsed": [...]}
    or {"ok": False, "reason": "..."} (the reason is printed in the report).
    """
    from itertools import product
    usable = [f for f in factors if isinstance(f, dict) and len(f.get("levels") or []) >= 2]
    if len(usable) < 2:
        return {"ok": False, "reason": "fewer than two factors with at least two levels are defined"}
    reason = ""
    # names that spell out ALL usable factors can still be analysed on a prefix of them (the rest is pooled)
    parsed_all = {cond: _parse_condition_levels(cond, usable) for cond in conditions}
    for k in range(min(len(usable), max_factors), 1, -1):
        facs = usable[:k]
        level_map: Dict[Any, Tuple[str, ...]] = {}
        unparsed: List[Any] = []
        for cond in conditions:
            parsed = _parse_condition_levels(cond, facs)
            if parsed is None and parsed_all[cond] is not None:
                parsed = parsed_all[cond][:k]
            if parsed is None:
                unparsed.append(cond)
            else:
                level_map[cond] = parsed
        if not level_map:
            reason = reason or "the condition names do not spell out the factor levels"
            continue
        used_levels = []
        for i, fac in enumerate(facs):
            declared = [str(lv) for lv in fac.get("levels")]
            observed = {combo[i] for combo in level_map.values()}
            used_levels.append([lv for lv in declared if lv in observed])
        if any(len(lv) < 2 for lv in used_levels):
            reason = reason or "a factor has fewer than two levels among the conditions found in the data"
            continue
        combos = set(level_map.values())
        expected = len(list(product(*used_levels)))
        if len(combos) != expected:
            reason = (f"the conditions do not form a complete crossing of "
                      f"{' x '.join(str(f.get('name', 'factor')) for f in facs)} "
                      f"({len(combos)} of {expected} cells present)")
            continue
        return {"ok": True, "factors": facs, "levels": used_levels, "level_map": level_map,
                "unparsed": unparsed, "skipped_factors": usable[k:]}
    return {"ok": False, "reason": reason or "the factor structure could not be matched to the condition names"}


def _factorial_type3(y: np.ndarray, level_index: List[np.ndarray], n_levels: List[int]) -> Dict[str, Any]:
    """Type III sums of squares of a full-factorial between-subjects ANOVA (any number of factors).

    Sum-to-zero (effect) coding; the SS of a term is the rise in residual SS when its columns are
    removed from the saturated model.  Balanced or not, this is the SS that tests the term against
    the unweighted marginal means (SAS / car::Anova type III).  Returns the term list and the error SS.
    """
    from itertools import combinations
    n = len(y)
    k = len(n_levels)

    def _effect_columns(idx: np.ndarray, n_lv: int) -> np.ndarray:
        cols = np.zeros((n, n_lv - 1))
        for j in range(n_lv - 1):
            cols[idx == j, j] = 1.0
        cols[idx == n_lv - 1, :] = -1.0
        return cols

    main = [_effect_columns(level_index[f], n_levels[f]) for f in range(k)]
    terms: List[Tuple[Tuple[int, ...], np.ndarray]] = []
    for size in range(1, k + 1):
        for subset in combinations(range(k), size):
            cols = main[subset[0]]
            for f in subset[1:]:
                cols = (cols[:, :, None] * main[f][:, None, :]).reshape(n, -1)
            terms.append((subset, cols))

    def _rss(blocks: List[np.ndarray]) -> float:
        design = np.hstack(blocks)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        return float(np.sum(resid * resid))

    ones = np.ones((n, 1))
    full = [ones] + [cols for _, cols in terms]
    ss_error = _rss(full)
    n_cells = sum(c.shape[1] for c in full)
    out_terms = []
    for i, (subset, cols) in enumerate(terms):
        reduced = [ones] + [c for j, (_, c) in enumerate(terms) if j != i]
        ss = max(0.0, _rss(reduced) - ss_error)
        out_terms.append({"factors": list(subset), "ss": ss, "df": int(cols.shape[1])})
    return {"terms": out_terms, "ss_error": ss_error, "df_error": n - n_cells, "n_cells": n_cells}


def _association_test(contingency: pd.DataFrame) -> Dict[str, Any]:
    """Chi-square test of independence for a condition x category table, with validity checks.

    * a single condition (or a single category) has no between-condition comparison: no test is run;
    * 2x2 tables get Yates' continuity correction (scipy does the same by default);
    * Cochran's rule: the test is unreliable when more than 20% of the expected counts are below 5
      or any is below 1 -- the result is flagged ``valid=False`` and, for a 2x2 table, Fisher's exact p
      is added.
    """
    observed = contingency.to_numpy(dtype=float)
    out: Dict[str, Any] = {"n": int(observed.sum()), "n_conditions": int(observed.shape[0]),
                           "n_categories": int(observed.shape[1])}
    if observed.shape[0] < 2:
        out["status"] = "single_condition"
        return out
    if observed.shape[1] < 2:
        out["status"] = "single_category"
        return out
    if _use_scipy():
        try:
            chi2, p, dof, expected = scipy_stats.chi2_contingency(observed)
        except Exception as exc:  # pragma: no cover - fall back to the numpy implementation
            logger.warning("scipy chi2_contingency failed (%s); using the numpy implementation", exc)
            chi2, p, dof, expected = _numpy_chi2_contingency(observed)
    else:
        chi2, p, dof, expected = _numpy_chi2_contingency(observed)
    expected = np.asarray(expected, dtype=float)
    share_low = float(np.mean(expected < 5.0))
    min_expected = float(expected.min())
    out.update({
        "status": "ok", "chi2": float(chi2), "p_value": float(p), "dof": int(dof),
        "yates": bool(dof == 1), "share_expected_below_5": share_low, "min_expected": min_expected,
        "valid": bool(share_low <= 0.20 and min_expected >= 1.0),
    })
    if not out["valid"] and observed.shape == (2, 2):
        try:
            if _use_scipy():
                out["fisher_p"] = float(scipy_stats.fisher_exact(observed)[1])
            else:
                out["fisher_p"] = _fisher_exact_2x2(observed)
        except Exception as exc:  # pragma: no cover
            logger.warning("Fisher exact test failed: %s", exc)
    return out


def _safe_to_markdown(df: pd.DataFrame, **kwargs) -> str:
    """
    Safely convert DataFrame to markdown, falling back to string representation
    if tabulate is not available (required by pandas.to_markdown).
    """
    try:
        return df.to_markdown(**kwargs)
    except ImportError:
        return df.to_string(**kwargs)


@dataclass
class InstructorReportConfig:
    include_metadata: bool = True
    include_schema_validation: bool = True
    include_preview_stats: bool = True
    include_design_summary: bool = True
    include_attention_checks: bool = True
    include_exclusions: bool = True
    include_persona_summary: bool = True
    include_variables: bool = True
    include_r_script: bool = True
    include_python_script: bool = True  # v1.2.3: Added missing config field
    include_spss_syntax: bool = True  # v1.2.3: Added missing config field
    include_stata_script: bool = True  # v1.2.3: Added missing config field


class InstructorReportGenerator:
    """
    Generates instructor-facing documentation for a simulation run.
    This is intended to help instructors quickly assess whether a team's
    simulation is coherent, standardized, and internally consistent.
    """

    def __init__(self, config: Optional[InstructorReportConfig] = None):
        self.config = config or InstructorReportConfig()

    def _get_persona_impact(self, persona: str) -> str:
        """Get expected impact description for a persona type."""
        impacts = {
            "engaged responder": "Reliable data, realistic variance",
            "engaged": "Reliable data, realistic variance",
            "satisficer": "May inflate midpoint responses",
            "extreme responder": "Increases overall variance",
            "extreme": "Increases overall variance",
            "acquiescent": "May inflate agreement/positive responses",
            "skeptic": "May deflate agreement/positive responses",
            "random": "Adds noise, likely flagged for exclusion",
            "careless": "Straight-line patterns, likely excluded",
            "careful responder": "High quality responses, low exclusion rate",
            "moderate responder": "Reduces variance, avoids extremes",
        }
        return impacts.get(persona, "Standard response patterns")

    def generate_markdown_report(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        schema_validation: Optional[Dict[str, Any]] = None,
        prereg_text: str = "",
        team_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines: List[str] = []

        lines.append(f"# Study Summary: {metadata.get('study_title', 'Untitled Study')}")
        lines.append("")
        lines.append("## Generation Information")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| **Generated** | {metadata.get('generation_timestamp', datetime.now().isoformat())} |")
        lines.append(f"| **Run ID** | `{metadata.get('run_id', 'N/A')}` |")
        lines.append(f"| **Mode** | {metadata.get('simulation_mode', 'pilot').title()} |")
        lines.append(f"| **Tool Version** | {metadata.get('app_version', 'N/A')} |")
        lines.append(f"| **Random Seed** | {metadata.get('random_seed', 'Auto')} |")
        # v1.1.0.7: Show data source (QSF upload vs manual builder)
        _data_source_label = metadata.get('data_source_label', '')
        if _data_source_label:
            lines.append(f"| **Data Source** | {_data_source_label} |")
        # v1.1.0.4: Show which generation method was used (not just in admin dashboard)
        _gen_method_label = metadata.get('generation_method_label', '')
        if _gen_method_label:
            lines.append(f"| **Generation Method** | {_gen_method_label} |")
        lines.append("")

        # v1.1.0: Add Study Overview section
        lines.append("## Study Overview")
        lines.append("")
        study_desc = metadata.get('study_description', '')
        if study_desc:
            # v1.8.7: Show full description (no truncation)
            clean_desc = study_desc.strip()
            lines.append(f"**Description:** {clean_desc}")
            lines.append("")

        # Design summary
        conditions = metadata.get('conditions', [])
        scales = metadata.get('scales', [])
        sample_size = metadata.get('sample_size', 0)

        lines.append("### Design at a Glance")
        lines.append("")
        lines.append("| Element | Details |")
        lines.append("|---------|---------|")
        lines.append(f"| **Sample Size** | N = {sample_size} |")
        lines.append(f"| **Conditions** | {len(conditions)} ({', '.join(conditions[:4])}{'...' if len(conditions) > 4 else ''}) |")
        lines.append(f"| **Outcome Measures** | {len(scales)} scale(s) detected |")

        # Detected research domain
        domains = metadata.get('detected_domains', [])
        if domains:
            lines.append(f"| **Research Domain** | {domains[0] if domains else 'General'} |")
        lines.append("")

        # =====================================================================
        # v1.2.0: COMPREHENSIVE SIMULATION INTELLIGENCE REPORT
        # =====================================================================
        lines.append("## Simulation Intelligence Report")
        lines.append("")
        lines.append("This section explains how the simulation system analyzed and approached your study, providing transparency into the data generation process.")
        lines.append("")

        # --- TOPIC/DOMAIN ANALYSIS ---
        lines.append("### Topic & Domain Analysis")
        lines.append("")

        detected_domains = metadata.get('detected_domains', [])
        study_context = metadata.get('study_context', {})
        if not isinstance(study_context, dict):
            study_context = {}
        domain_keywords = study_context.get('detected_keywords', [])

        if detected_domains:
            lines.append(f"**Primary Research Domain:** {detected_domains[0].replace('_', ' ').title()}")
            if len(detected_domains) > 1:
                secondary = [d.replace('_', ' ').title() for d in detected_domains[1:5]]
                lines.append(f"**Secondary Domains:** {', '.join(secondary)}")
            lines.append("")

            # Domain explanation
            domain_explanations = {
                'behavioral_economics': 'Study involves economic decision-making, incentives, or behavioral biases',
                'social_psychology': 'Study examines social influence, attitudes, or interpersonal dynamics',
                'consumer_behavior': 'Study focuses on purchasing decisions, brand perceptions, or marketing',
                'ai_attitudes': 'Study explores perceptions of AI, algorithms, or automation',
                'technology_adoption': 'Study examines technology use, acceptance, or digital behaviors',
                'organizational_behavior': 'Study involves workplace dynamics, leadership, or employee attitudes',
                'health_psychology': 'Study addresses health behaviors, medical decisions, or wellbeing',
                'political_psychology': 'Study examines political attitudes, voting, or civic engagement',
                'environmental_psychology': 'Study focuses on environmental attitudes or sustainable behaviors',
                'moral_psychology': 'Study explores ethical judgments, moral reasoning, or values',
            }
            primary_domain = detected_domains[0].lower()
            if primary_domain in domain_explanations:
                lines.append(f"**Domain Interpretation:** {domain_explanations[primary_domain]}")
                lines.append("")
        else:
            lines.append("**Research Domain:** General (no specific domain detected)")
            lines.append("")

        if domain_keywords:
            lines.append(f"**Keywords Detected:** {', '.join(domain_keywords[:10])}")
            lines.append("")

        # --- SIMULATION APPROACH ---
        lines.append("### How the Simulation Approached This Study")
        lines.append("")
        # v1.4.3.1: Use source-agnostic language (works for both QSF upload and conversational builder)
        _data_source = metadata.get("data_source", "")
        if "builder" in str(_data_source).lower():
            _source_phrase = "your study description and condition structure"
        else:
            _source_phrase = "your QSF file, study description, and condition structure"
        lines.append(f"Based on the analysis of {_source_phrase}, the simulation:")
        lines.append("")

        conditions = metadata.get('conditions', [])
        scales = metadata.get('scales', [])
        effect_sizes = metadata.get('effect_sizes_configured', []) or metadata.get('effect_sizes', [])

        # Describe the approach
        approach_points = []

        if len(conditions) > 1:
            approach_points.append(f"Assigned **{len(conditions)} experimental conditions** with balanced allocation")

        _inferred_on = (metadata.get('effect_sizes_applied') or {}).get('inferred_effects_enabled', True)
        if effect_sizes:
            approach_points.append(f"Applied **{len(effect_sizes)} user-specified effect size(s)** to create systematic condition differences")
        elif _inferred_on:
            approach_points.append("Inferred **small condition differences from the condition names** (a heuristic; see Condition Effects Strategy)")
        else:
            approach_points.append("Built in **no condition differences** (inferred effects were switched off)")

        if scales:
            scale_types = set(s.get('type', 'likert') for s in scales)
            approach_points.append(f"Generated responses for **{len(scales)} DV(s)** ({', '.join(scale_types)})")

        open_ended = metadata.get('open_ended_questions', [])
        if open_ended:
            approach_points.append(f"Created **{len(open_ended)} unique open-ended responses** per participant based on detected topic context")

        for point in approach_points:
            lines.append(f"- {point}")
        lines.append("")

        # --- CONDITION EFFECT STRATEGY ---
        lines.append("### Condition Effects Strategy")
        lines.append("")

        applied = metadata.get('effect_sizes_applied') or {}
        contrasts = applied.get('contrasts') or []
        inferred_on = applied.get('inferred_effects_enabled', True)

        if effect_sizes:
            lines.append("**Effects you specified** (the Cohen's d on the scale mean is calibrated to land "
                         "near the intended value; the observed value varies with sampling):")
            lines.append("")
            lines.append("| Variable | High level | Low level | Intended d (high - low) | Observed d (high - low) |")
            lines.append("|----------|------------|-----------|-------------------------|-------------------------|")
            user_rows = [r for r in contrasts if r.get('source') == 'user']

            def _norm_name(x: Any) -> str:
                return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")

            for es in effect_sizes[:10]:
                var = es.get('variable', 'DV')
                high = es.get('level_high', '') or 'high level'
                low = es.get('level_low', '') or 'low level'
                d = _safe_float(es.get('cohens_d', 0.5))
                # The engine raises the high level for direction "positive" and lowers it otherwise
                intended = d if str(es.get('direction', 'positive')).lower() == 'positive' else -d
                observed = None
                for r in user_rows:
                    if _norm_name(r.get('variable', '')) != _norm_name(var):
                        continue
                    pair = (r.get('condition_1'), r.get('condition_2'))
                    if pair not in ((high, low), (low, high)) or r.get('observed_d') is None:
                        continue
                    # contrast rows are condition_1 minus condition_2, in condition order:
                    # orient them as high minus low
                    observed = r['observed_d'] if pair == (high, low) else -r['observed_d']
                    break
                obs_txt = f"{observed:+.2f}" if observed is not None else "n/a"
                lines.append(f"| {var} | {high} | {low} | {intended:+.2f} | {obs_txt} |")
            lines.append("")

        if inferred_on:
            lines.append("**Inferred differences:** for contrasts without a specified effect, the tool infers a "
                         "small difference from the wording of the condition names (for example, a gain vs a "
                         "loss frame). This is a heuristic, not a calibrated effect, and it can be larger than a "
                         "typical real effect. Specify your own effect size to override it, or turn inferred "
                         "effects off in Advanced Settings to get a true null.")
        else:
            lines.append("**Inferred effects were switched off:** contrasts you did not specify contain no "
                         "built-in difference apart from sampling noise.")
        lines.append("")

        # --- OBSERVED EFFECTS ---
        observed_effects = metadata.get('effect_sizes_observed', [])
        if observed_effects:
            lines.append("### Observed Effects in Generated Data")
            lines.append("")
            lines.append("| Variable | Condition 1 | Condition 2 | M₁ | M₂ | Cohen's d |")
            lines.append("|----------|-------------|-------------|-----|-----|-----------|")
            for obs in observed_effects[:10]:
                var = obs.get('variable', 'DV')
                c1 = obs.get('condition_1', 'C1')
                c2 = obs.get('condition_2', 'C2')
                m1 = obs.get('mean_1', 0)
                m2 = obs.get('mean_2', 0)
                d = obs.get('cohens_d', 0)
                lines.append(f"| {var} | {c1} | {c2} | {m1:.2f} | {m2:.2f} | {d:.2f} |")
            lines.append("")

            lines.append("**Effect Size Interpretation:**")
            lines.append("- d = 0.2: Small effect (subtle but potentially meaningful)")
            lines.append("- d = 0.5: Medium effect (moderate practical significance)")
            lines.append("- d = 0.8: Large effect (substantial difference)")
            lines.append("")

        # --- PERSONA RATIONALE ---
        lines.append("### Persona Selection Rationale")
        lines.append("")

        persona_dist_raw = metadata.get('persona_distribution', {})
        persona_dist = _extract_persona_proportions(persona_dist_raw)
        if persona_dist:
            # Explain why certain personas were chosen
            lines.append("The simulation assigned response style personas based on:")
            lines.append("")
            lines.append("1. **Base prevalence rates** from survey methodology research (Krosnick, 1991; Meade & Craig, 2012)")
            lines.append("2. **Domain-specific adjustments** based on detected research topic")
            lines.append("3. **Realistic online panel proportions** (~35% engaged, ~22% satisficers, ~5% careless)")
            lines.append("")

            # Show persona traits summary
            persona_traits = {
                'engaged responder': {'attention': 0.92, 'consistency': 0.78, 'extremity': 0.18},
                'satisficer': {'attention': 0.68, 'consistency': 0.55, 'extremity': 0.12},
                'extreme responder': {'attention': 0.80, 'consistency': 0.70, 'extremity': 0.88},
                'acquiescent': {'attention': 0.75, 'consistency': 0.65, 'extremity': 0.35},
                'careless': {'attention': 0.35, 'consistency': 0.28, 'extremity': 0.45},
            }

            lines.append("**Persona Trait Profiles:**")
            lines.append("")
            lines.append("| Persona | Attention Level | Response Consistency | Endpoint Use |")
            lines.append("|---------|-----------------|---------------------|--------------|")
            for persona, share in sorted(persona_dist.items(), key=lambda x: -_safe_float(x[1]))[:5]:
                traits = persona_traits.get(persona.lower(), {'attention': 0.7, 'consistency': 0.6, 'extremity': 0.3})
                att = traits['attention']
                con = traits['consistency']
                ext = traits['extremity']
                att_label = "High" if att > 0.8 else ("Medium" if att > 0.5 else "Low")
                con_label = "High" if con > 0.7 else ("Medium" if con > 0.5 else "Low")
                ext_label = "High" if ext > 0.6 else ("Medium" if ext > 0.3 else "Low")
                lines.append(f"| {persona.title()} | {att_label} ({att:.0%}) | {con_label} ({con:.0%}) | {ext_label} ({ext:.0%}) |")
            lines.append("")
        else:
            lines.append("_Persona distribution details not available._")
            lines.append("")

        # --- OPEN-ENDED RESPONSE GENERATION ---
        open_ended_details = metadata.get('open_ended_questions', [])
        if open_ended_details:
            lines.append("### Open-Ended Response Generation")
            lines.append("")
            lines.append(f"**{len(open_ended_details)} open-ended question(s)** were detected and populated with contextually appropriate text responses.")
            lines.append("")
            # Check if LLM was used (indicated by metadata)
            llm_stats = metadata.get('llm_stats', metadata.get('llm_response_stats', {}))
            llm_calls = llm_stats.get('llm_calls', 0) if llm_stats else 0
            llm_attempts = llm_stats.get('llm_attempts', llm_calls) if llm_stats else 0
            pool_size = llm_stats.get('pool_size', 0) if llm_stats else 0
            fallback_uses = llm_stats.get('fallback_uses', 0) if llm_stats else 0
            batch_failures = llm_stats.get('batch_failures', 0) if llm_stats else 0
            allow_template_fallback = bool(llm_stats.get('allow_template_fallback', True)) if llm_stats else True
            llm_init_error = str(metadata.get('llm_init_error', '') or '').strip()

            # v1.0.9.2: Read user-selected generation method for accurate labeling
            _gen_method = metadata.get('generation_method', '')
            _gen_method_label = metadata.get('generation_method_label', '')
            _is_adaptive_engine = _gen_method in ('experimental', 'abe_v2')
            _is_template_engine = _gen_method == 'template'
            _is_abe3 = _gen_method == 'abe3'
            # v1.2.5.4: Recognize free_llm and own_api as AI-powered methods
            _is_llm_method = _gen_method in ('free_llm', 'own_api')

            if pool_size > 0:
                # AI-generated responses were actually produced and used
                if _is_abe3 or _is_adaptive_engine or _is_llm_method:
                    # v1.2.5.4: All AI methods (free_llm, own_api, abe3, abe_v2) use ABE 3.0 engine
                    lines.append("**Generation approach:** Adaptive Behavioral Engine 3.0 + AI-Powered LLM")
                else:
                    lines.append("**Generation approach:** AI-Powered (Language Model)")
                lines.append("")
                if _is_abe3 or _is_llm_method:
                    lines.append("- **Engine:** Adaptive Behavioral Engine 3.0 with census-weighted demographics and deep persona coherence")
                    lines.append("- Open-text responses powered by real-time LLM generation with stylometric fingerprinting")
                elif _is_adaptive_engine:
                    lines.append("- **Engine:** Adaptive Behavioral Engine with domain-calibrated behavioral models")
                    lines.append("- Open-text responses powered by real-time LLM generation with behavioral coherence")
                lines.append(f"- A pool of {pool_size} base responses was pre-generated across sentiment categories")
                lines.append("- Each participant received a unique response via draw-with-replacement with persona variation")
                lines.append("- Variation layers: word-level edits, sentence restructuring, verbosity/formality/engagement modulation")
                lines.append("- Response style (length, tone, effort) matches each participant's assigned persona")
                if fallback_uses > 0:
                    total_oe_responses = pool_size + fallback_uses
                    pct = (fallback_uses / max(1, total_oe_responses)) * 100
                    if pct > 50:
                        lines.append(f"- **Note:** {fallback_uses} response(s) ({pct:.0f}%) used template fallback due to API limits")
                    else:
                        lines.append(f"- {fallback_uses} response(s) ({pct:.0f}%) used template fallback due to API rate limits")
                else:
                    lines.append("- All responses were successfully generated by the language model (no template fallback needed)")
            elif llm_calls > 0 or llm_attempts > 0:
                # v1.1.0.5: Consistent labeling — when AI was attempted but failed,
                # clearly state Template Engine was used. Match the badge label from
                # metadata['generation_method_label'] for zero confusion.
                if _is_abe3 or _is_adaptive_engine or _is_llm_method:
                    # v1.2.5.4: All AI methods fall back to ABE 3.0 template-backed
                    lines.append("**Generation approach:** Adaptive Behavioral Engine 3.0 (template-backed, AI unavailable)")
                elif _is_template_engine:
                    lines.append("**Generation approach:** Template Engine (225+ Research Domains)")
                else:
                    lines.append("**Generation approach:** Template Engine (AI providers were unavailable)")
                lines.append("")
                if llm_calls > 0:
                    lines.append(f"- {llm_calls} API request(s) were sent but all providers were unavailable or returned errors")
                else:
                    lines.append(f"- {llm_attempts} provider probe(s) were made but no usable connection was established")
                if batch_failures > 0:
                    lines.append(f"- {batch_failures} batch generation attempt(s) did not produce usable responses")
                lines.append("- All open-ended responses were generated using the built-in template engine (225+ research domains)")
                lines.append("- Template responses are unique per participant with topic-appropriate content")
                lines.append("- Response length and sentiment vary based on simulated persona")
                if not _is_template_engine:
                    lines.append("- **Tip:** Provide your own API key (Groq, Google AI, etc.) for AI-powered responses")
            elif llm_init_error:
                lines.append("**Generation approach:** LLM initialization failure (run integrity warning)")
                lines.append("")
                lines.append(f"- LLM generator failed to initialize: `{llm_init_error}`")
                lines.append("- Open-ended responses did not run through API calls in this run")
                lines.append("- This run should be treated as invalid for AI-based simulation")
            elif not allow_template_fallback:
                lines.append("**Generation approach:** LLM-first strict mode")
                lines.append("")
                lines.append("- Template fallback was disabled for this run")
                lines.append("- No successful API outputs were recorded; rerun with valid API connectivity")
            else:
                # v1.0.9.2: Correctly label based on user-selected method
                if _is_abe3 or _is_llm_method:
                    # v1.2.5.4: free_llm/own_api with no OE questions still uses ABE 3.0 for numeric data
                    lines.append("**Generation approach:** Adaptive Behavioral Engine 3.0")
                    lines.append("")
                    lines.append("- **Engine:** Adaptive Behavioral Engine 3.0 with census-weighted demographics and deep persona coherence")
                    lines.append("- Calibrated error rates by education level (Frederick 2005)")
                    lines.append("- Stylometric voice fingerprinting ensures consistent writing style per participant")
                    lines.append("- Self-validating output against human-realism benchmarks")
                    lines.append("- Cross-DV coherence: numeric ratings and open-text responses tell a consistent story")
                    if _is_llm_method:
                        lines.append(f"- **Selected method:** {_gen_method_label}")
                        if pool_size == 0 and llm_calls == 0 and llm_attempts == 0:
                            lines.append("- No open-ended questions detected; AI generation was not needed for this run")
                elif _is_adaptive_engine:
                    lines.append("**Generation approach:** Adaptive Behavioral Engine 3.0")
                    lines.append("")
                    lines.append("- **Engine:** Adaptive Behavioral Engine with domain-calibrated behavioral models")
                    lines.append("- 50+ persona archetypes with 7-dimensional trait profiles drive response generation")
                    lines.append("- 30+ experimental paradigm recognizers calibrate responses to published norms")
                    lines.append("- Cross-DV coherence: numeric ratings and open-text responses tell a consistent story")
                    lines.append("- Domain-specific behavioral calibration (e.g., Engel 2011, Iyengar & Westwood 2015, Dimant 2024)")
                elif _is_template_engine:
                    lines.append("**Generation approach:** Template Engine (225+ Research Domains)")
                    lines.append("")
                    lines.append("- Responses generated offline using the built-in template engine")
                else:
                    lines.append("**Generation approach:** Advanced Template Engine (225+ Research Domains)")
                    lines.append("")
                lines.append("- Coverage: 225+ research domains, 40 question types, domain-specific response patterns")
                lines.append("- Each response is unique per participant (no duplicate sentences within the dataset)")
                lines.append("- Topic context is extracted from question text to ensure topical relevance")
                lines.append("- Response characteristics (length, tone, effort level) match each participant's persona profile")
                lines.append("- Sentiment aligns with participant's scale responses for cross-measure consistency")
            lines.append("")

            # Show sample question types
            lines.append("**Questions detected:**")
            lines.append("")
            for i, q in enumerate(open_ended_details[:5], 1):
                q_name = q.get('variable_name', q.get('name', f'Q{i}'))
                q_text = q.get('question_text', '')[:60]
                q_ctx = q.get('question_context', '')
                if q_text:
                    lines.append(f"- **{q_name}**: \"{q_text}{'...' if len(q.get('question_text', '')) > 60 else ''}\"")
                else:
                    lines.append(f"- **{q_name}**")
                if q_ctx:
                    lines.append(f"  - *Context:* {q_ctx}")
            if len(open_ended_details) > 5:
                lines.append(f"- _...and {len(open_ended_details) - 5} more_")
            lines.append("")

        lines.append("---")
        lines.append("")

        if team_info:
            lines.append("## Team Information")
            lines.append("")
            team_name = team_info.get("team_name", "")
            team_members = team_info.get("team_members", "")
            if team_name:
                lines.append(f"- **Team Name:** {team_name}")
            if team_members:
                lines.append("- **Members:**")
                for member in team_members.strip().split('\n'):
                    if member.strip():
                        lines.append(f"  - {member.strip()}")
            lines.append("")

        # DATA QUALITY SUMMARY (Executive Overview)
        lines.append("## Data Quality Summary")
        lines.append("")
        lines.append("### Quick Assessment")
        lines.append("")

        n_total = len(df)
        n_excluded = int(df["Exclude_Recommended"].sum()) if "Exclude_Recommended" in df.columns else 0
        exclusion_rate = (n_excluded / n_total * 100) if n_total > 0 else 0

        # Calculate balance (coefficient of variation of condition sizes)
        balance_status = "N/A"
        if "CONDITION" in df.columns:
            cond_counts = df["CONDITION"].value_counts()
            if len(cond_counts) > 1:
                cv = cond_counts.std() / cond_counts.mean() * 100 if cond_counts.mean() > 0 else 0
                balance_status = "Balanced" if cv < 10 else ("Slightly unbalanced" if cv < 20 else "Unbalanced")

        lines.append("| Metric | Value | Status |")
        lines.append("|--------|-------|--------|")
        lines.append(f"| Sample Size | N = {n_total} | {'Adequate' if n_total >= 30 else 'Small'} |")
        lines.append(f"| Exclusion Rate | {exclusion_rate:.1f}% ({n_excluded}/{n_total}) | {'Normal' if exclusion_rate < 20 else 'High'} |")
        lines.append(f"| Condition Balance | {balance_status} | - |")
        lines.append(f"| Missing Data | {int(df.isna().sum().sum())} cells | {'Clean' if df.isna().sum().sum() == 0 else 'Some missing'} |")
        lines.append("")

        lines.append("### Interpretation Guide")
        lines.append("")
        lines.append("- **Exclusion Rate**: Simulated exclusion rates typically range 5-15%. Higher rates may indicate stricter criteria or more careless responders.")
        lines.append("- **Condition Balance**: Slight imbalances (<10%) are normal and won't affect most analyses.")
        lines.append("- **This is simulated data**: Results demonstrate what your analysis pipeline will produce with realistic-looking data structures.")
        lines.append("")

        # SIMULATION SETTINGS TRANSPARENCY SECTION
        lines.append("## Simulation Settings (Transparency)")
        lines.append("")
        lines.append("These settings were used to generate the simulated data:")
        lines.append("")

        # Demographics settings
        demo = metadata.get("demographics", {})
        lines.append("### Demographics Configuration")
        lines.append("")
        lines.append("| Setting | Value |")
        lines.append("|---------|-------|")
        lines.append(f"| Gender quota (% male) | {demo.get('gender_quota', 50)}% |")
        lines.append(f"| Age mean | {demo.get('age_mean', 35)} |")
        lines.append(f"| Age standard deviation | {demo.get('age_sd', 12)} |")
        lines.append("")

        # Response quality settings
        lines.append("### Response Quality Settings")
        lines.append("")
        lines.append("| Setting | Value |")
        lines.append("|---------|-------|")
        lines.append(f"| Attention check pass rate | {_safe_float(metadata.get('attention_rate', 0.85)):.0%} |")
        lines.append(f"| Random responder rate | {_safe_float(metadata.get('random_responder_rate', 0.05)):.0%} |")
        lines.append("")

        # Exclusion criteria
        exclusion = metadata.get("exclusion_criteria", {})
        if exclusion:
            lines.append("### Exclusion Criteria")
            lines.append("")
            lines.append("| Criterion | Threshold |")
            lines.append("|-----------|-----------|")
            lines.append(f"| Min completion time | {exclusion.get('completion_time_min_seconds', 60)} seconds |")
            lines.append(f"| Max completion time | {exclusion.get('completion_time_max_seconds', 1800)} seconds |")
            lines.append(f"| Straight-line threshold | {exclusion.get('straight_line_threshold', 10)} items |")
            lines.append(f"| Duplicate IP check | {'Yes' if exclusion.get('duplicate_ip_check', True) else 'No'} |")
            lines.append("")

        if prereg_text:
            lines.append("## Preregistration Notes (as provided)")
            lines.append("")
            lines.append("```")
            lines.append(prereg_text.strip()[:2000])
            if len(prereg_text) > 2000:
                lines.append("... [truncated]")
            lines.append("```")
            lines.append("")

        if self.config.include_design_summary:
            lines.append("## Experimental Design Summary")
            lines.append("")
            lines.append("| Element | Details |")
            lines.append("|---------|---------|")
            lines.append(f"| **Sample Size (N)** | {metadata.get('sample_size', 'N/A')} |")

            conditions = metadata.get("conditions", []) or []
            lines.append(f"| **Conditions** | {len(conditions)}: {', '.join(str(c) for c in conditions) if conditions else 'N/A'} |")

            factors = metadata.get("factors", []) or []
            if factors:
                for i, f in enumerate(factors):
                    fname = f.get("name", "Factor")
                    levels = f.get("levels", [])
                    lines.append(f"| **Factor {i+1}** | {fname}: {', '.join(str(l) for l in levels)} |")
            else:
                lines.append("| **Factors** | Single factor (Condition) |")

            # Randomization level
            design_review = metadata.get("design_review", {})
            rand_level = design_review.get("randomization_level", "Participant-level")
            lines.append(f"| **Randomization** | {rand_level} |")

            domains = metadata.get("detected_domains", []) or []
            if domains:
                lines.append(f"| **Detected Domains** | {', '.join(domains[:5])} |")
            lines.append("")

        if self.config.include_preview_stats:
            lines.append("## Quick data checks")
            lines.append("")
            lines.append(f"- Rows: {df.shape[0]}")
            lines.append(f"- Columns: {df.shape[1]}")
            lines.append(f"- Missing values (total): {int(df.isna().sum().sum())}")
            lines.append("")

            if "CONDITION" in df.columns:
                vc = df["CONDITION"].value_counts(dropna=False)
                lines.append("### Condition counts")
                lines.append("")
                lines.append(_safe_to_markdown(vc.to_frame("n")))
                lines.append("")

        if self.config.include_attention_checks:
            lines.append("## Attention checks")
            lines.append("")
            # v1.4.3: Support both old and new attention check column names
            _attn_col = "Attention_Check_1" if "Attention_Check_1" in df.columns else (
                "AI_Mentioned_Check" if "AI_Mentioned_Check" in df.columns else None
            )
            if _attn_col is not None:
                lines.append(f"- **{_attn_col}** distribution:")
                lines.append(_safe_to_markdown(df[_attn_col].value_counts(dropna=False).to_frame("n")))
                lines.append("")
            if "Attention_Pass_Rate" in df.columns:
                lines.append("- **Attention_Pass_Rate** summary:")
                lines.append(_safe_to_markdown(df["Attention_Pass_Rate"].describe().to_frame()))
                lines.append("")

        if self.config.include_exclusions:
            lines.append("## Exclusions")
            lines.append("")
            if "Exclude_Recommended" in df.columns:
                excl = df["Exclude_Recommended"].value_counts(dropna=False).to_frame("n")
                lines.append(_safe_to_markdown(excl))
                lines.append("")
            for col in ["Completion_Time_Seconds", "Max_Straight_Line"]:
                if col in df.columns:
                    lines.append(f"### {col} summary")
                    lines.append("")
                    lines.append(_safe_to_markdown(df[col].describe().to_frame()))
                    lines.append("")

        if self.config.include_persona_summary:
            lines.append("## Persona Distribution (Simulated Response Styles)")
            lines.append("")

            # Comprehensive theory section
            lines.append("### Theoretical Background")
            lines.append("")
            lines.append("Personas in this simulation are based on well-established survey methodology research on response styles and participant behaviors. Real survey data naturally contains participants who respond in systematically different ways - some are highly engaged and thoughtful, while others may satisfice (provide minimally acceptable responses), exhibit response biases, or respond carelessly.")
            lines.append("")
            lines.append("By incorporating these response patterns into simulated data, we create more realistic datasets that mirror what researchers encounter in actual studies. This allows students to practice identifying and handling data quality issues before collecting real data.")
            lines.append("")

            lines.append("### How Personas Were Selected")
            lines.append("")
            lines.append("Personas are automatically assigned to simulated participants based on:")
            lines.append("1. **Study domain detection**: The simulation analyzes your study title and description to identify relevant behavioral domains (e.g., consumer behavior, attitudes, decision-making)")
            lines.append("2. **Weighted random assignment**: Each persona has a base probability, adjusted by study characteristics")
            lines.append("3. **Realistic proportions**: The distribution aims to match what researchers typically observe in online panel data (e.g., ~5-15% satisficers, ~5% careless responders)")
            lines.append("")

            dist_raw = metadata.get("persona_distribution", {}) or {}
            dist = _extract_persona_proportions(dist_raw)
            if dist:
                lines.append("### Persona Breakdown for This Simulation")
                lines.append("")
                lines.append("| Persona Type | Description | Behavioral Characteristics | Share |")
                lines.append("|--------------|-------------|---------------------------|-------|")

                # Comprehensive persona descriptions with behavioral characteristics
                persona_info = {
                    "engaged responder": {
                        "desc": "Thoughtful, attentive participant",
                        "chars": "Reads questions carefully, uses full scale range appropriately, consistent with their attitudes"
                    },
                    "engaged": {
                        "desc": "Thoughtful, attentive participant",
                        "chars": "Reads questions carefully, uses full scale range appropriately, consistent with their attitudes"
                    },
                    "satisficer": {
                        "desc": "Minimally effortful responder",
                        "chars": "Gravitates to middle options, may skip reading full questions, faster completion times"
                    },
                    "extreme responder": {
                        "desc": "Uses scale endpoints frequently",
                        "chars": "Strong opinions, uses 1s and 7s (or max/min) more than average, high variance"
                    },
                    "extreme": {
                        "desc": "Uses scale endpoints frequently",
                        "chars": "Strong opinions, uses 1s and 7s (or max/min) more than average, high variance"
                    },
                    "acquiescent": {
                        "desc": "Agreement bias responder",
                        "chars": "Tendency to agree with statements regardless of content, inflated positive responses"
                    },
                    "skeptic": {
                        "desc": "Disagreement bias responder",
                        "chars": "Tendency to disagree or rate negatively, lower mean responses"
                    },
                    "random": {
                        "desc": "Inconsistent, inattentive responder",
                        "chars": "High variance, fails attention checks, no clear pattern"
                    },
                    "careless": {
                        "desc": "Pattern-based responder",
                        "chars": "Straight-lining, very fast completion, likely to be flagged for exclusion"
                    },
                    "careful responder": {
                        "desc": "Highly attentive, methodical",
                        "chars": "Longer completion times, passes all attention checks, low variance within scales"
                    },
                    "moderate responder": {
                        "desc": "Avoids extreme responses",
                        "chars": "Uses middle portion of scale, rarely selects endpoints"
                    },
                }

                for persona, share in sorted(dist.items(), key=lambda x: -_safe_float(x[1])):
                    persona_key = str(persona).lower() if persona else "unknown"
                    info = persona_info.get(persona_key, {"desc": "Standard response pattern", "chars": "Typical survey behavior"})
                    share_f = _safe_float(share)
                    pct = share_f * 100 if share_f <= 1 else share_f
                    lines.append(f"| **{str(persona).title()}** | {info['desc']} | {info['chars']} | {pct:.1f}% |")
                lines.append("")

                # Show total participants by persona
                lines.append("### Participant Counts by Persona")
                lines.append("")
                n_total = metadata.get("sample_size", len(df))
                lines.append("| Persona | Approximate Count | Expected Impact |")
                lines.append("|---------|-------------------|-----------------|")
                for persona, share in sorted(dist.items(), key=lambda x: -_safe_float(x[1])):
                    share_f = _safe_float(share)
                    share_val = share_f if share_f <= 1 else share_f / 100
                    count = int(round(n_total * share_val))
                    # Describe expected impact
                    impact = self._get_persona_impact(str(persona).lower() if persona else "unknown")
                    lines.append(f"| {str(persona).title()} | ~{count} | {impact} |")
                lines.append("")

                # Interpretation guidance
                lines.append("### Interpreting Simulated Data with Personas")
                lines.append("")
                lines.append("Understanding the persona distribution helps interpret your simulated results:")
                lines.append("")
                lines.append("1. **Scale means may be slightly inflated or deflated** depending on the balance of acquiescent vs. skeptic personas")
                lines.append("2. **Exclusion rates reflect realistic data cleaning** - participants flagged for exclusion are often those with 'careless' or 'random' personas")
                lines.append("3. **Effect sizes remain interpretable** - condition effects are applied consistently across personas, so treatment differences reflect your specified effect sizes")
                lines.append("4. **Variance patterns mirror real data** - some participants show more extreme responses (higher variance) while others cluster around the mean")
                lines.append("")
                lines.append("**Important**: The persona assigned to each simulated participant is NOT included in the exported data. This is intentional - in real research, you wouldn't know each participant's response style. The personas are used only to generate realistic response patterns.")
                lines.append("")
            else:
                lines.append("_Persona distribution data not available in metadata. This may indicate an older simulation or configuration issue._")
                lines.append("")

        if self.config.include_variables:
            lines.append("## Dependent Variables (Scales)")
            lines.append("")
            # v1.4.3.1: Source-agnostic phrasing
            _dv_source = metadata.get("data_source", "")
            if "builder" in str(_dv_source).lower():
                lines.append("The following scales/DVs were simulated based on your study description and configuration:")
            else:
                lines.append("The following scales/DVs were simulated based on your QSF and configuration:")
            lines.append("")

            scales = metadata.get("scales", []) or []
            if scales:
                lines.append("| Scale Name | Items | Scale Points | Source | Reverse Items |")
                lines.append("|------------|-------|--------------|--------|---------------|")

                for s in scales:
                    name = s.get("name", "Scale")
                    num_items = s.get("num_items", s.get("items", "?"))
                    scale_points = s.get("scale_points", "?")

                    # Determine source of scale points
                    detected = s.get("detected_from_qsf", None)
                    if detected is True:
                        # v1.4.3.1: Source-agnostic label
                        _scale_data_source = metadata.get("data_source", "")
                        if "builder" in str(_scale_data_source).lower():
                            source = "Auto-detected"
                        else:
                            source = "QSF (detected)"
                    elif detected is False:
                        source = "Default"
                    else:
                        source = "Config"

                    reverse = s.get("reverse_items", [])
                    reverse_str = ", ".join(str(r) for r in reverse) if reverse else "None"

                    lines.append(f"| {name} | {num_items} | {scale_points}-point | {source} | {reverse_str} |")

                lines.append("")

                # Add note about scale points
                lines.append("### Scale Points Note")
                lines.append("")
                # v1.4.3.1: Source-agnostic scale point notes
                _sp_source = metadata.get("data_source", "")
                if "builder" in str(_sp_source).lower():
                    lines.append("- **Auto-detected**: Scale points were inferred from your study description")
                    lines.append("- **Default**: Scale points defaulted to 7 when not explicitly specified")
                    lines.append("- **Config**: Scale points were set in the simulation configuration")
                    lines.append("")
                    lines.append("If the scale points don't match your preregistration, manually specify scale points in the tool's configuration.")
                else:
                    lines.append("- **QSF (detected)**: Scale points were automatically detected from your Qualtrics survey file")
                    lines.append("- **Default**: Scale points defaulted to 7 because they couldn't be detected from QSF")
                    lines.append("- **Config**: Scale points were set in the simulation configuration")
                    lines.append("")
                    lines.append("If the scale points don't match your preregistration, please verify your QSF has the correct response options defined, or manually specify scale points in the tool's configuration.")
                lines.append("")
            else:
                lines.append("_No scales/DVs listed in metadata. This may indicate a configuration issue._")
                lines.append("")

            # v1.2.5.8: Mediator/Moderator variables section
            _med_mod_vars = metadata.get("mediator_moderator_variables", [])
            if _med_mod_vars:
                lines.append("### Mediator / Moderator Variables")
                lines.append("")
                lines.append("| Variable | Role | Connected DV(s) | Items | Range | Description |")
                lines.append("|----------|------|----------------|-------|-------|-------------|")
                for _mm in _med_mod_vars:
                    _mm_name = _mm.get("name", "?")
                    _mm_role = _mm.get("role", "mediator").title()
                    _mm_dvs = ", ".join(_mm.get("connected_dvs", [])) or "—"
                    _mm_items = _mm.get("num_items", 1)
                    _mm_range = f"{_mm.get('scale_min', 1)}-{_mm.get('scale_max', 7)}"
                    _mm_desc = _mm.get("description", "")[:60]
                    lines.append(f"| {_mm_name} | {_mm_role} | {_mm_dvs} | {_mm_items} | {_mm_range} | {_mm_desc} |")
                lines.append("")
                lines.append("**Simulation approach:**")
                _n_mediators = sum(1 for m in _med_mod_vars if m.get("role") == "mediator")
                _n_moderators = sum(1 for m in _med_mod_vars if m.get("role") == "moderator")
                if _n_mediators > 0:
                    lines.append(f"- {_n_mediators} mediator(s): simulated with condition effects + r≈0.45 correlation with connected DVs")
                if _n_moderators > 0:
                    lines.append(f"- {_n_moderators} moderator(s): simulated with r≈0.15 correlation with connected DVs")
                lines.append("")

        # === v1.2.4: PRACTICAL DATA INTERPRETATION GUIDE ===
        lines.append("## How to Work With This Data")
        lines.append("")
        lines.append("### Quick Start Checklist")
        lines.append("")
        lines.append("1. **Load the CSV** into your preferred statistical analysis tool")
        lines.append("2. **Check the codebook** (Data_Codebook_Handbook.txt) for variable definitions")
        lines.append("3. **Examine exclusion flags** — filter out flagged participants before analysis")
        lines.append("4. **Verify your conditions** — check that CONDITION matches your expected groups")
        lines.append("5. **Run descriptive statistics** before hypothesis testing")
        lines.append("")

        conditions = metadata.get("conditions", [])
        if conditions and len(conditions) >= 2:
            lines.append("### Suggested Analysis Approach")
            lines.append("")
            if len(conditions) == 2:
                lines.append(f"With **2 conditions** ({', '.join(str(c) for c in conditions)}), consider:")
                lines.append("- **Independent samples t-test** for comparing condition means")
                lines.append("- **Mann-Whitney U test** as a non-parametric alternative")
                lines.append("- **Cohen's d** for effect size estimation")
            elif len(conditions) <= 4:
                lines.append(f"With **{len(conditions)} conditions** ({', '.join(str(c) for c in conditions)}), consider:")
                lines.append("- **One-way ANOVA** for comparing condition means")
                lines.append("- **Kruskal-Wallis test** as a non-parametric alternative")
                lines.append("- **Post-hoc pairwise comparisons** (Tukey HSD or Bonferroni)")
                lines.append("- **Eta-squared (η²)** for effect size")
            else:
                lines.append(f"With **{len(conditions)} conditions**, consider a structured ANOVA approach")
            lines.append("")

            factors = metadata.get("factors", [])
            if factors and len(factors) >= 2:
                factor_names = [f.get("name", "Factor") for f in factors if isinstance(f, dict)]
                if len(factor_names) >= 2:
                    lines.append(f"Your **factorial design** ({' × '.join(factor_names)}) allows testing:")
                    lines.append(f"- Main effect of {factor_names[0]}")
                    lines.append(f"- Main effect of {factor_names[1]}")
                    lines.append(f"- **Interaction** between {factor_names[0]} and {factor_names[1]}")
                    lines.append("- Use **two-way ANOVA** (or factorial ANOVA) for this analysis")
                    lines.append("")

        exclusion_summary = metadata.get("exclusion_summary", {})
        if exclusion_summary:
            lines.append("### Data Cleaning Guide")
            lines.append("")
            lines.append("Quality flags are in `Simulation_Diagnostics.csv` (join on `ResponseId`, then filter on `Exclude_Recommended`):")
            lines.append("")
            n = metadata.get("sample_size", "N/A")
            flagged_speed = exclusion_summary.get("flagged_speed", 0)
            flagged_attention = exclusion_summary.get("flagged_attention", 0)
            flagged_straight = exclusion_summary.get("flagged_straightline", 0)
            total_excluded = exclusion_summary.get("total_excluded", 0)
            lines.append(f"| Flag | Count | % of Sample | Action |")
            lines.append(f"|------|-------|-------------|--------|")
            if isinstance(n, (int, float)) and n > 0:
                lines.append(f"| Speed flags | {flagged_speed} | {flagged_speed/n*100:.1f}% | Too fast/slow completion |")
                lines.append(f"| Attention flags | {flagged_attention} | {flagged_attention/n*100:.1f}% | Failed attention checks |")
                lines.append(f"| Straight-lining | {flagged_straight} | {flagged_straight/n*100:.1f}% | Identical responses in sequence |")
                lines.append(f"| **Total excluded** | **{total_excluded}** | **{total_excluded/n*100:.1f}%** | **Recommended for removal** |")
            lines.append("")
            lines.append("**Tip:** After joining `Simulation_Diagnostics.csv` on `ResponseId`, filter with `Exclude_Recommended == 0` to keep only clean responses.")
            lines.append("")

        # v1.2.0: Enhanced Analysis Recommendations with statistical test recommendations and power analysis
        lines.append("## Analysis Recommendations")
        lines.append("")

        # Get design information
        conditions = metadata.get('conditions', [])
        factors = metadata.get('factors', [])
        factors = [f for f in factors if isinstance(f, dict)]
        scales = metadata.get('scales', [])
        scales = [s for s in scales if isinstance(s, dict)]
        sample_size = metadata.get('sample_size', len(df) if df is not None else 100)
        effect_sizes_cfg = metadata.get('effect_sizes_configured', [])
        is_factorial = len(factors) >= 2
        num_conditions = len(conditions)

        lines.append("### Suggested Analysis Steps")
        lines.append("")
        lines.append("1. **Data Cleaning**")
        lines.append("   - Review `Exclude_Recommended` (in `Simulation_Diagnostics.csv`, joined on `ResponseId`) for data quality issues")
        lines.append("   - Check `Duration (in seconds)` for speedy responders (< 60s suspicious)")
        lines.append("   - Examine `Max_Straight_Line` for response patterns (> 5 suggests inattention)")
        lines.append("   - Check `Attention_Pass_Rate` (< 50% warrants exclusion)")
        lines.append("")
        lines.append("2. **Descriptive Statistics**")
        lines.append("   - Calculate means and SDs by condition")
        lines.append("   - Check for outliers (beyond 3 SD from mean)")
        lines.append("   - Verify condition balance (N per group)")
        lines.append("   - Assess normality (Shapiro-Wilk test)")
        lines.append("")

        # Statistical Test Recommendations
        # v1.4.3.1: Enhanced Statistical Test Recommendations with design-specific detail
        lines.append("### Statistical Test Recommendations")
        lines.append("")
        has_ordinal = any(s.get('scale_points', 7) <= 5 for s in scales)
        # Determine if within-subjects
        _design_type = metadata.get("design_type", "between")
        is_within = str(_design_type).lower() in ("within", "within-subjects", "repeated")

        # Build a first DV name for code snippets
        _first_dv = "Outcome_composite"
        if scales:
            _first_dv = _report_clean_column_name(str(scales[0].get("name", "Outcome"))) + "_composite"

        if is_factorial:
            factor_str = " x ".join([str(len(f.get('levels', []))) for f in factors])
            factor_names = [f.get('name', 'Factor').replace(' ', '_') for f in factors if isinstance(f, dict)]
            lines.append(f"**Design:** {factor_str} Factorial (between-subjects)")
            lines.append("")
            lines.append("**Primary analysis: Factorial ANOVA**")
            lines.append("")
            lines.append("| Step | Analysis | Purpose | What to Report |")
            lines.append("|------|----------|---------|----------------|")
            if len(factor_names) >= 2:
                lines.append(f"| 1 | Factorial ANOVA | Test main effects and interaction | F, df, p, partial eta-squared |")
                lines.append(f"| 2 | Main effect of {factor_names[0]} | Does {factor_names[0]} influence the DV? | F, df, p |")
                lines.append(f"| 3 | Main effect of {factor_names[1]} | Does {factor_names[1]} influence the DV? | F, df, p |")
                lines.append(f"| 4 | {factor_names[0]} x {factor_names[1]} interaction | Do factors combine non-additively? | F, df, p |")
                lines.append(f"| 5 | Simple effects (if interaction p < .05) | Decompose the interaction | t or F, p, Cohen's d |")
            else:
                lines.append(f"| 1 | Factorial ANOVA | Main effects + interaction | F, df, p, partial eta-squared |")
                lines.append(f"| 2 | Simple effects | If interaction significant | t or F, p |")
            lines.append("")
            if has_ordinal:
                lines.append("**Non-parametric alternative:** Aligned Rank Transform (ART) ANOVA for ordinal DVs")
                lines.append("")
            lines.append("**Effect size:** Report partial eta-squared for each effect. Benchmarks: small = .01, medium = .06, large = .14")
            lines.append("")
        elif num_conditions == 2:
            lines.append("**Design:** Two-Group Between-Subjects Comparison")
            lines.append("")
            lines.append("**Primary analysis: Independent-samples t-test with Cohen's d**")
            lines.append("")
            lines.append("| Step | Analysis | Purpose | What to Report |")
            lines.append("|------|----------|---------|----------------|")
            lines.append("| 1 | Check assumptions | Normality (Shapiro-Wilk), homogeneity (Levene's) | W, p; F, p |")
            lines.append("| 2 | Welch's t-test | Compare group means (robust to unequal variances) | t, df, p (two-tailed) |")
            lines.append("| 3 | Cohen's d | Quantify effect magnitude | d with 95% CI |")
            lines.append("| 4 | Descriptives | Group means and SDs | M, SD per group |")
            lines.append("")
            if has_ordinal:
                lines.append("**Non-parametric alternative:** Mann-Whitney U test (report U, p, rank-biserial r)")
                lines.append("")
            lines.append("**Interpretation guide:**")
            lines.append("- p < .05 with d >= 0.20: Statistically significant and practically meaningful")
            lines.append("- p < .05 with d < 0.20: Statistically significant but trivial effect")
            lines.append("- p >= .05: No significant difference; report observed d to inform future power analyses")
            lines.append("")
        elif num_conditions > 2:
            lines.append(f"**Design:** {num_conditions}-Group Between-Subjects Comparison")
            lines.append("")
            lines.append("**Primary analysis: One-way ANOVA with post-hoc pairwise tests**")
            lines.append("")
            lines.append("| Step | Analysis | Purpose | What to Report |")
            lines.append("|------|----------|---------|----------------|")
            lines.append("| 1 | Check assumptions | Normality per group, Levene's test | Shapiro-Wilk W, Levene's F |")
            lines.append("| 2 | One-way ANOVA | Omnibus test of group differences | F(df_between, df_within), p |")
            lines.append("| 3 | Effect size | Overall effect magnitude | Eta-squared or omega-squared |")
            lines.append("| 4 | Post-hoc tests (if F is significant) | Pairwise group comparisons | Tukey HSD: mean diff, p, 95% CI |")
            lines.append("| 5 | Pairwise effect sizes | Effect for each pair | Cohen's d for each comparison |")
            lines.append("")
            if has_ordinal:
                lines.append("**Non-parametric alternative:** Kruskal-Wallis H test, followed by Dunn's test with Bonferroni correction")
                lines.append("")
            lines.append(f"**Post-hoc correction:** With {num_conditions} groups you have {num_conditions * (num_conditions - 1) // 2} pairwise comparisons. Use Tukey HSD (controls family-wise error) or Bonferroni correction.")
            lines.append("")

        if is_within:
            lines.append("**Within-subjects design detected:**")
            lines.append("")
            if num_conditions == 2:
                lines.append("- **Primary:** Paired-samples t-test (report t, df, p, Cohen's d_z)")
                lines.append("- **Non-parametric:** Wilcoxon signed-rank test (report W, p, matched-pairs rank-biserial r)")
            else:
                lines.append("- **Primary:** Repeated-measures ANOVA (report F, df, p, partial eta-squared)")
                lines.append("- Check sphericity with Mauchly's test; apply Greenhouse-Geisser correction if violated")
                lines.append("- **Non-parametric:** Friedman test followed by pairwise Wilcoxon signed-rank tests")
            lines.append("")

        # v1.4.3.1: Quick-start analysis code snippets
        lines.append("### Quick-Start Analysis Code")
        lines.append("")
        lines.append("Copy-paste starter code for your primary analysis:")
        lines.append("")

        # Snippets read the delivered files: Simulated_Data.csv has item columns only, so the
        # composite is computed from the items (reverse-keyed recoded) and Exclude_Recommended
        # is joined from Simulation_Diagnostics.csv on ResponseId.
        _qs_all = _script_scales(metadata, df)
        _qs_sc = _qs_all[:1]
        if _qs_sc:
            _first_dv = _qs_sc[0]["composite"]
        _fmap = _script_factor_map(conditions, factors) if is_factorial else None
        if not _qs_sc:
            lines.append("*No scale item columns were found, so no starter code is shown.*")
            lines.append("")
        elif _fmap:
            _f1, _f2 = _fmap[0][0], _fmap[1][0]
            # Factorial ANOVA snippets
            lines.append("**R:**")
            lines.append("```r")
            lines.append("library(readr); library(dplyr); library(effectsize)")
            lines.append("df <- read_csv('Simulated_Data.csv', show_col_types = FALSE)")
            lines.extend(_r_prep_lines(_qs_sc, _fmap))
            lines.append(f"model <- aov({_first_dv} ~ {_f1} * {_f2}, data = df_clean)")
            lines.append("summary(model)")
            lines.append("eta_squared(model, partial = TRUE)")
            lines.append("```")
            lines.append("")
            lines.append("**Python:**")
            lines.append("```python")
            lines.append("import os; import pandas as pd; import statsmodels.api as sm")
            lines.append("from statsmodels.formula.api import ols")
            lines.append("df = pd.read_csv('Simulated_Data.csv')")
            lines.extend(_py_prep_lines(_qs_sc, _fmap))
            lines.append(f"model = ols('{_first_dv} ~ C({_f1}) * C({_f2})', data=df_clean).fit()")
            lines.append("print(sm.stats.anova_lm(model, typ=2))")
            lines.append("```")
            lines.append("")
        elif num_conditions == 2 and conditions:
            # Two-group t-test snippets
            lines.append("**R:**")
            lines.append("```r")
            lines.append("library(readr); library(dplyr); library(effsize)")
            lines.append("df <- read_csv('Simulated_Data.csv', show_col_types = FALSE)")
            lines.extend(_r_prep_lines(_qs_sc))
            lines.append("df_clean$CONDITION <- factor(df_clean$CONDITION)")
            lines.append(f"t.test({_first_dv} ~ CONDITION, data = df_clean, var.equal = FALSE)")
            lines.append(f"effsize::cohen.d({_first_dv} ~ CONDITION, data = df_clean)")
            lines.append("```")
            lines.append("")
            lines.append("**Python:**")
            lines.append("```python")
            lines.append("import os; import pandas as pd; from scipy import stats; import numpy as np")
            lines.append("df = pd.read_csv('Simulated_Data.csv')")
            lines.extend(_py_prep_lines(_qs_sc))
            lines.append(f"g1 = df_clean[df_clean['CONDITION'] == {_py_lit(conditions[0])}]['{_first_dv}'].dropna()")
            lines.append(f"g2 = df_clean[df_clean['CONDITION'] == {_py_lit(conditions[1])}]['{_first_dv}'].dropna()")
            lines.append("t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)")
            lines.append("d = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)")
            lines.append("print(f't({len(g1)+len(g2)-2}) = {t_stat:.3f}, p = {p_val:.4f}, d = {d:.3f}')")
            lines.append("```")
            lines.append("")
        elif num_conditions > 2:
            # One-way ANOVA snippets
            lines.append("**R:**")
            lines.append("```r")
            lines.append("library(readr); library(dplyr); library(effectsize)")
            lines.append("df <- read_csv('Simulated_Data.csv', show_col_types = FALSE)")
            lines.extend(_r_prep_lines(_qs_sc))
            lines.append(f"model <- aov({_first_dv} ~ CONDITION, data = df_clean)")
            lines.append("summary(model)")
            lines.append("TukeyHSD(model)")
            lines.append("eta_squared(model)")
            lines.append("```")
            lines.append("")
            lines.append("**Python:**")
            lines.append("```python")
            lines.append("import os; import pandas as pd; from scipy import stats")
            lines.append("df = pd.read_csv('Simulated_Data.csv')")
            lines.extend(_py_prep_lines(_qs_sc))
            lines.append(f"groups = [df_clean[df_clean['CONDITION'] == c]['{_first_dv}'].dropna() for c in {[str(c) for c in conditions]!r}]")
            lines.append("f_stat, p_val = stats.f_oneway(*groups)")
            lines.append("print(f'F = {f_stat:.3f}, p = {p_val:.4f}')")
            lines.append("# Post-hoc: pip install scikit-posthocs, then sp.posthoc_ttest(df_clean, val_col, group_col)")
            lines.append("```")
            lines.append("")

        # Power Analysis
        lines.append("### Power Analysis Estimates")
        lines.append("")
        n_per_group = sample_size // max(num_conditions, 1) if num_conditions > 0 else sample_size
        lines.append(f"**Sample:** N = {sample_size} (~{n_per_group} per condition)")
        lines.append("")
        lines.append("| Effect Size | d | Estimated Power |")
        lines.append("|-------------|---|-----------------|")
        z_alpha = 1.96
        for label, d in [("Small", 0.2), ("Medium", 0.5), ("Large", 0.8)]:
            ncp = abs(d) * math.sqrt(n_per_group / 2)
            z_power = ncp - z_alpha
            power = 0.5 * (1 + math.erf(z_power / math.sqrt(2)))
            power = max(0.05, min(0.99, power))
            lines.append(f"| {label} | {d:.2f} | {power:.0%} |")
        lines.append("")
        lines.append("*80% power is generally considered adequate. If power < 80% for your expected effect, consider increasing sample size.*")
        lines.append("")

        # Effect Size Guide
        lines.append("### Effect Size Interpretation")
        lines.append("")
        lines.append("| Measure | Small | Medium | Large | When to Use |")
        lines.append("|---------|-------|--------|-------|-------------|")
        lines.append("| Cohen's d | 0.20 | 0.50 | 0.80 | Two-group comparisons |")
        lines.append("| Eta-squared (eta2) | 0.01 | 0.06 | 0.14 | ANOVA overall effect |")
        lines.append("| Partial eta-squared | 0.01 | 0.06 | 0.14 | Factorial ANOVA per factor |")
        lines.append("| Omega-squared | 0.01 | 0.06 | 0.14 | Less biased ANOVA estimate |")
        lines.append("")

        if self.config.include_schema_validation and schema_validation is not None:
            lines.append("## Schema validation")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(schema_validation, indent=2, ensure_ascii=False, default=str))
            lines.append("```")
            lines.append("")

        # Analysis Scripts Section
        lines.append("## Analysis Scripts")
        lines.append("")
        lines.append("Auto-generated scripts with explanatory comments for your statistical software:")
        lines.append("")

        if self.config.include_r_script:
            lines.append("### R Script")
            lines.append("")
            lines.append("```r")
            lines.append(self._generate_comprehensive_r_script(metadata, df))
            lines.append("```")
            lines.append("")

        if self.config.include_python_script:
            lines.append("### Python Script")
            lines.append("")
            lines.append("```python")
            lines.append(self._generate_python_script(metadata, df))
            lines.append("```")
            lines.append("")

        if self.config.include_spss_syntax:
            lines.append("### SPSS Syntax")
            lines.append("")
            lines.append("```spss")
            lines.append(self._generate_spss_syntax(metadata, df))
            lines.append("```")
            lines.append("")

        if self.config.include_stata_script:
            lines.append("### Stata Script")
            lines.append("")
            lines.append("```stata")
            lines.append(self._generate_stata_script(metadata, df))
            lines.append("```")
            lines.append("")

        # === DATA DICTIONARY (placed at end for reference) ===
        column_descriptions = metadata.get("column_descriptions", {})
        if column_descriptions and isinstance(column_descriptions, dict):
            lines.append("## Data Dictionary")
            lines.append("")
            lines.append("Complete reference for every column in the output CSV:")
            lines.append("")
            lines.append("| Column | Description |")
            lines.append("|--------|-------------|")
            for col_name, col_desc in column_descriptions.items():
                safe_name = str(col_name).replace("|", "/")
                safe_desc = str(col_desc).replace("|", "/")
                lines.append(f"| `{safe_name}` | {safe_desc} |")
            lines.append("")
        elif df is not None and len(df.columns) > 0:
            lines.append("## Data Dictionary")
            lines.append("")
            lines.append("Columns present in the output CSV:")
            lines.append("")
            lines.append("| Column | Type | Description |")
            lines.append("|--------|------|-------------|")
            _col_desc_map = {
                "PARTICIPANT_ID": "Unique participant identifier (1-N)",
                "RUN_ID": "Simulation run identifier",
                "CONDITION": "Experimental condition assignment",
                "Age": "Participant age in years",
                "Gender": "Participant gender (Male, Female, Non-binary, Prefer not to say)",
                "Attention_Check_1": "Attention/manipulation check (1=Correct, 2=Incorrect)",
                "Completion_Time_Seconds": "Survey completion time in seconds",
                "Attention_Pass_Rate": "Proportion of attention checks passed (0-1)",
                "Max_Straight_Line": "Longest run of identical consecutive responses",
                "Flag_Speeder": "Speed flag: 1=unusually fast completion",
                "Flag_StraightLine": "Straight-line flag: 1=repetitive pattern detected",
                "Exclude_Recommended": "Recommended exclusion: 1=exclude, 0=retain",
                "SIMULATION_MODE": "Simulation mode (pilot/full)",
                "SIMULATION_SEED": "Random seed used for reproducibility",
            }
            for col in df.columns:
                col_str = str(col)
                dtype_str = str(df[col].dtype)
                desc = _col_desc_map.get(col_str, "")
                if not desc:
                    if col_str.endswith("_mean"):
                        desc = "Composite mean score for scale"
                    elif "_" in col_str and col_str.split("_")[-1].isdigit():
                        desc = "Individual scale item response"
                    elif col_str.startswith("OE_") or col_str.startswith("OpenEnded_"):
                        desc = "Open-ended text response"
                lines.append(f"| `{col_str}` | {dtype_str} | {desc} |")
            lines.append("")

        return "\n".join(lines)

    def _generate_basic_r_script(self, metadata: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> str:
        conditions = metadata.get("conditions", [])
        scales = _script_scales(metadata, df)

        condition_levels = ", ".join([_r_lit(c) for c in conditions])

        r_lines = [
            "# ============================================================",
            f"# Basic R script for: {metadata.get('study_title', 'Untitled Study')}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Run ID: {metadata.get('run_id', 'N/A')}",
            "# ============================================================",
            "",
            "library(readr)",
            "library(dplyr)",
            "",
            "# Simulated_Data.csv holds item columns only; composites are computed below.",
            "df <- read_csv('Simulated_Data.csv', show_col_types = FALSE)",
            "",
            "# Set up factors",
            f"df$CONDITION <- factor(df$CONDITION, levels = c({condition_levels}))",
            "",
        ]
        r_lines.extend(_r_prep_lines(scales))
        r_lines.append("")
        r_lines.append("summary(df_clean)")
        return "\n".join(r_lines)

    def _generate_comprehensive_r_script(self, metadata: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> str:
        """Generate comprehensive R script with explanatory comments."""
        conditions = metadata.get("conditions", [])
        factors = [f for f in metadata.get("factors", []) if isinstance(f, dict)]
        scales = _script_scales(metadata, df)
        fmap = _script_factor_map(conditions, factors)
        num_conditions = len(conditions)

        condition_levels = ", ".join([_r_lit(c) for c in conditions])

        r_lines = [
            "# ============================================================================",
            f"# COMPREHENSIVE R ANALYSIS SCRIPT",
            f"# Study: {metadata.get('study_title', 'Untitled Study')}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "# ============================================================================",
            "",
            "# --- SECTION 1: SETUP ---",
            "# Load required packages (install if needed)",
            "library(readr)      # CSV reading",
            "library(dplyr)      # Data manipulation",
            "# library(effectsize) # Effect sizes (optional)",
            "# library(car)        # Levene's test (optional)",
            "",
            "# --- SECTION 2: DATA LOADING ---",
            "# Simulated_Data.csv holds the item columns; Simulation_Diagnostics.csv holds the flags.",
            "df <- read_csv('Simulated_Data.csv', show_col_types = FALSE)",
            "head(df)  # Verify data loaded correctly",
            "",
            "# --- SECTION 3: DATA PREPARATION ---",
            "# Set condition as factor with proper ordering",
            f"df$CONDITION <- factor(df$CONDITION, levels = c({condition_levels}))",
            "",
            "# --- SECTION 4: COMPUTE COMPOSITES, JOIN DIAGNOSTICS, APPLY EXCLUSIONS ---",
            "# Composites are the mean of the item columns (reverse-keyed items recoded first).",
            "# Document the exclusion criteria in your methods!",
        ]
        r_lines.extend(_r_prep_lines(scales, fmap))
        r_lines.extend([
            "cat('Total N:', nrow(df), '| Clean N:', nrow(df_clean), '\\n')",
            "",
            "# --- SECTION 5: DESCRIPTIVES BY CONDITION ---",
        ])
        if scales:
            r_lines.append("df_clean %>% group_by(CONDITION) %>%")
            r_lines.append("  summarise(n = n(),")
            for s in scales:
                r_lines.append(f"            {s['name']}_M = mean({s['composite']}, na.rm = TRUE),")
                r_lines.append(f"            {s['name']}_SD = sd({s['composite']}, na.rm = TRUE),")
            r_lines.append("  )")
        r_lines.append("")
        r_lines.append("# --- SECTION 6: STATISTICAL TESTS ---")

        if fmap:
            r_lines.append("# FACTORIAL ANOVA - tests main effects and interaction")
            formula = " * ".join(n for n, _ in fmap)
            for s in scales:
                r_lines.append(f"model_{s['name']} <- aov({s['composite']} ~ {formula}, data = df_clean)")
                r_lines.append(f"summary(model_{s['name']})")
                r_lines.append(f"# Effect sizes: effectsize::eta_squared(model_{s['name']}, partial = TRUE)")
        elif num_conditions == 2:
            r_lines.append("# TWO-GROUP T-TEST")
            r_lines.append("# Welch's t-test (var.equal=FALSE) is more robust")
            for s in scales:
                r_lines.append(f"t.test({s['composite']} ~ CONDITION, data = df_clean, var.equal = FALSE)")
                r_lines.append(f"# Effect size: effectsize::cohens_d({s['composite']} ~ CONDITION, data = df_clean)")
            r_lines.append("")
            r_lines.append("# NON-PARAMETRIC ALTERNATIVE (if non-normal):")
            for s in scales:
                r_lines.append(f"# wilcox.test({s['composite']} ~ CONDITION, data = df_clean)")
        elif num_conditions > 2:
            r_lines.append("# ONE-WAY ANOVA")
            for s in scales:
                r_lines.append(f"model_{s['name']} <- aov({s['composite']} ~ CONDITION, data = df_clean)")
                r_lines.append(f"summary(model_{s['name']})")
                r_lines.append(f"# Post-hoc: TukeyHSD(model_{s['name']})")
            r_lines.append("")
            r_lines.append("# NON-PARAMETRIC ALTERNATIVE:")
            for s in scales:
                r_lines.append(f"# kruskal.test({s['composite']} ~ CONDITION, data = df_clean)")

        r_lines.append("")
        r_lines.append("# ============================================================================")
        return "\n".join(r_lines)

    def _generate_python_script(self, metadata: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> str:
        """Generate Python analysis script with explanatory comments."""
        conditions = [str(c) for c in metadata.get("conditions", [])]
        factors = [f for f in metadata.get("factors", []) if isinstance(f, dict)]
        scales = _script_scales(metadata, df)
        fmap = _script_factor_map(conditions, factors)
        num_conditions = len(conditions)

        py_lines = [
            "# ============================================================================",
            f"# COMPREHENSIVE PYTHON ANALYSIS SCRIPT",
            f"# Study: {metadata.get('study_title', 'Untitled Study')}",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "# ============================================================================",
            "",
            "import os",
            "import pandas as pd",
            "import numpy as np",
            "from scipy import stats",
            "",
            "# --- SECTION 1: DATA LOADING ---",
            "# Simulated_Data.csv holds the item columns; Simulation_Diagnostics.csv holds the flags.",
            "df = pd.read_csv('Simulated_Data.csv')",
            "print('Dataset shape:', df.shape)",
            "",
            "# --- SECTION 2: COMPUTE COMPOSITES, JOIN DIAGNOSTICS, APPLY EXCLUSIONS ---",
            "# Composites are the mean of the item columns (reverse-keyed items recoded first).",
            "# Document the exclusion criteria in your methods!",
        ]
        py_lines.extend(_py_prep_lines(scales, fmap))
        py_lines.extend([
            "print(f'Total N: {len(df)} | Clean N: {len(df_clean)}')",
            "",
            "# --- SECTION 3: DATA PREPARATION ---",
            f"condition_order = {conditions}",
            "df_clean['CONDITION'] = pd.Categorical(df_clean['CONDITION'], categories=condition_order, ordered=True)",
            "",
            "# --- SECTION 4: DESCRIPTIVES ---",
        ])
        for s in scales:
            py_lines.append(f"print(df_clean.groupby('CONDITION', observed=True)['{s['composite']}'].agg(['count', 'mean', 'std']))")

        py_lines.append("")
        py_lines.append("# --- SECTION 5: STATISTICAL TESTS ---")

        if num_conditions == 2 and conditions:
            py_lines.append("# TWO-GROUP T-TEST")
            py_lines.append(f"cond1, cond2 = {conditions[0]!r}, {conditions[1]!r}")
            for s in scales:
                py_lines.append(f"g1 = df_clean[df_clean['CONDITION'] == cond1]['{s['composite']}'].dropna()")
                py_lines.append(f"g2 = df_clean[df_clean['CONDITION'] == cond2]['{s['composite']}'].dropna()")
                py_lines.append("t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)  # Welch's t-test")
                py_lines.append(f"print(f'{s['name']}: t={{t_stat:.3f}}, p={{p_val:.4f}}')")
                py_lines.append("# Cohen's d")
                py_lines.append("d = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)")
                py_lines.append("print(f\"Cohen's d = {d:.3f}\")")
        elif num_conditions > 2:
            py_lines.append("# ONE-WAY ANOVA")
            for s in scales:
                py_lines.append(f"groups = [df_clean[df_clean['CONDITION'] == c]['{s['composite']}'].dropna() for c in condition_order]")
                py_lines.append("f_stat, p_val = stats.f_oneway(*groups)")
                py_lines.append(f"print(f'{s['name']} ANOVA: F={{f_stat:.3f}}, p={{p_val:.4f}}')")

        py_lines.append("")
        py_lines.append("# ============================================================================")
        return "\n".join(py_lines)

    def _spss_diag_merge_lines(self) -> List[str]:
        return [
            "* Exclude_Recommended lives in Simulation_Diagnostics.csv; match it on ResponseId.",
            "* Import 'Simulation_Diagnostics.csv' the same way (File > Import Data > CSV Data...), then run:",
            "DATASET NAME diag WINDOW=FRONT.",
            "DATASET ACTIVATE diag.",
            "SORT CASES BY ResponseId (A).",
            "DATASET ACTIVATE data.",
            "SORT CASES BY ResponseId (A).",
            "MATCH FILES /FILE=* /TABLE=diag /BY ResponseId.",
            "EXECUTE.",
            "",
            "USE ALL.",
            "COMPUTE filter_$=(Exclude_Recommended = 0).",
            "VARIABLE LABELS filter_$ 'Exclude_Recommended = 0 (FILTER)'.",
            "VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'.",
            "FORMATS filter_$ (f1.0).",
            "FILTER BY filter_$.",
            "EXECUTE.",
        ]

    def _generate_spss_syntax(self, metadata: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> str:
        """Generate SPSS syntax with explanatory comments."""
        conditions = metadata.get("conditions", [])
        scales = _script_scales(metadata, df)
        num_conditions = len(conditions)

        spss_lines = [
            "* ============================================================================.",
            f"* COMPREHENSIVE SPSS ANALYSIS SYNTAX.",
            f"* Study: {metadata.get('study_title', 'Untitled Study')}.",
            f"* Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
            "* ============================================================================.",
            "",
            "* --- SECTION 1: DATA LOADING ---.",
            "* Use File > Import Data > CSV Data... to import Simulated_Data.csv",
            "* (one header row; text columns as strings).",
            "DATASET NAME data WINDOW=FRONT.",
            "",
            "* CONDITION is a string column; create a numeric version with value labels.",
            "AUTORECODE VARIABLES=CONDITION /INTO CONDITION_num /PRINT.",
            "",
            "* --- SECTION 2: COMPUTE COMPOSITES ---.",
            "* Composites are the mean of the item columns (reverse-keyed items recoded first).",
        ]

        for s in scales:
            if s["reverse"]:
                spss_lines.append(f"* {s['raw']} - reverse code items {s['reverse']}.")
                for r_item in s["reverse"]:
                    item_name = f"{s['name']}_{r_item}"
                    spss_lines.append(f"COMPUTE {item_name}_R = {s['flip']} - {item_name}.")
                spss_lines.append("EXECUTE.")
            spss_lines.append(f"* {s['raw']}: composite from the item columns.")
            spss_lines.append(f"COMPUTE {s['composite']} = MEAN({' '.join(s['composite_items'])}).")
            spss_lines.append("EXECUTE.")

        spss_lines.append("")
        spss_lines.append("* --- SECTION 3: DATA CLEANING ---.")
        spss_lines.extend(self._spss_diag_merge_lines())
        spss_lines.extend([
            "",
            "* Check sample size.",
            "FREQUENCIES VARIABLES=CONDITION.",
            "",
        ])

        spss_lines.append("* --- SECTION 4: DESCRIPTIVES ---.")
        if scales:
            dv_vars = " ".join(s["composite"] for s in scales)
            spss_lines.append(f"MEANS TABLES={dv_vars} BY CONDITION")
            spss_lines.append("  /CELLS=MEAN STDDEV COUNT.")
        spss_lines.append("")
        spss_lines.append("* --- SECTION 5: STATISTICAL TESTS ---.")

        if num_conditions == 2:
            spss_lines.append("* TWO-GROUP T-TEST (CONDITION_num is the numeric version of CONDITION).")
            spss_lines.append("* Levene's test included in output - check for equal variances.")
            for s in scales:
                spss_lines.append(f"T-TEST GROUPS=CONDITION_num(1 2)")
                spss_lines.append(f"  /VARIABLES={s['composite']}")
                spss_lines.append("  /MISSING=ANALYSIS.")
            spss_lines.append("")
            spss_lines.append("* NON-PARAMETRIC: Mann-Whitney U.")
            for s in scales:
                spss_lines.append(f"NPAR TESTS /M-W={s['composite']} BY CONDITION_num(1 2).")
        elif num_conditions > 2:
            spss_lines.append("* ONE-WAY ANOVA with post-hoc tests.")
            for s in scales:
                spss_lines.append(f"ONEWAY {s['composite']} BY CONDITION_num")
                spss_lines.append("  /STATISTICS DESCRIPTIVES HOMOGENEITY")
                spss_lines.append("  /POSTHOC=TUKEY ALPHA(0.05).")
            spss_lines.append("")
            spss_lines.append("* For effect size (eta-squared), use GLM.")
            for s in scales:
                spss_lines.append(f"UNIANOVA {s['composite']} BY CONDITION_num")
                spss_lines.append("  /PRINT=ETASQ.")

        spss_lines.append("")
        spss_lines.append("* ============================================================================.")
        return "\n".join(spss_lines)

    def _generate_stata_script(self, metadata: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> str:
        """Generate Stata script with explanatory comments."""
        conditions = metadata.get("conditions", [])
        scales = _script_scales(metadata, df, lowercase=True)
        num_conditions = len(conditions)

        stata_lines = [
            "/* ============================================================================",
            f"   COMPREHENSIVE STATA ANALYSIS SCRIPT",
            f"   Study: {metadata.get('study_title', 'Untitled Study')}",
            f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "   ============================================================================ */",
            "",
            "// --- SECTION 1: DATA LOADING ---",
            "// Note: import delimited lower-cases variable names (CONDITION -> condition).",
            "clear all",
            'import delimited "Simulated_Data.csv", clear varnames(1)',
            "describe",
            "",
            "// --- SECTION 2: DATA PREPARATION ---",
            "// Encode condition as numeric factor",
            "encode condition, generate(condition_num)",
            "",
            "// --- SECTION 3: COMPUTE COMPOSITES ---",
            "// Composites are the mean of the item columns (reverse-keyed items recoded first).",
        ]

        for s in scales:
            if s["reverse"]:
                stata_lines.append(f"// {s['raw']} - reverse code items {s['reverse']}")
                for r_item in s["reverse"]:
                    item_name = f"{s['name']}_{r_item}"
                    stata_lines.append(f"gen {item_name}_r = {s['flip']} - {item_name}")
            stata_lines.append(f"egen {s['composite']} = rowmean({' '.join(s['composite_items'])})")

        stata_lines.extend([
            "",
            "// --- SECTION 4: DATA CLEANING ---",
            "// Exclude_Recommended lives in Simulation_Diagnostics.csv; merge it on responseid.",
            'capture confirm file "Simulation_Diagnostics.csv"',
            "if _rc == 0 {",
            "    preserve",
            '    import delimited "Simulation_Diagnostics.csv", clear varnames(1)',
            "    keep responseid exclude_recommended",
            "    tempfile diag",
            "    save `diag'",
            "    restore",
            "    merge 1:1 responseid using `diag', keep(master match) nogenerate",
            "}",
            "else {",
            '    display "Simulation_Diagnostics.csv not found; no exclusions applied."',
            "    gen exclude_recommended = 0",
            "}",
            "// Keep only clean data (document criteria in methods!)",
            "keep if exclude_recommended == 0",
            "count",
            "tabulate condition",
            "",
            "// --- SECTION 5: DESCRIPTIVES ---",
        ])
        for s in scales:
            stata_lines.append(f"tabstat {s['composite']}, by(condition) statistics(n mean sd)")

        stata_lines.append("")
        stata_lines.append("// --- SECTION 6: STATISTICAL TESTS ---")

        if num_conditions == 2:
            stata_lines.append("// TWO-GROUP T-TEST")
            stata_lines.append("// Use 'unequal' for Welch's t-test (more robust)")
            for s in scales:
                stata_lines.append(f"ttest {s['composite']}, by(condition_num) unequal")
            stata_lines.append("")
            stata_lines.append("// NON-PARAMETRIC: Wilcoxon rank-sum (Mann-Whitney)")
            for s in scales:
                stata_lines.append(f"ranksum {s['composite']}, by(condition_num)")
        elif num_conditions > 2:
            stata_lines.append("// ONE-WAY ANOVA")
            for s in scales:
                stata_lines.append(f"oneway {s['composite']} condition_num, tabulate")
            stata_lines.append("")
            stata_lines.append("// Post-hoc tests")
            stata_lines.append("// Run after anova command:")
            stata_lines.append("// pwcompare condition_num, mcompare(tukey) effects")
            stata_lines.append("")
            stata_lines.append("// NON-PARAMETRIC: Kruskal-Wallis")
            for s in scales:
                stata_lines.append(f"kwallis {s['composite']}, by(condition_num)")

        stata_lines.append("")
        stata_lines.append("/* ============================================================================ */")
        return "\n".join(stata_lines)


class ComprehensiveInstructorReport:
    """
    Generates a detailed, comprehensive report for instructors ONLY.

    This report includes:
    - Statistical analyses with full interpretation (t-tests, ANOVA, effect sizes)
    - Data quality diagnostics (attention checks, completion times, exclusions)
    - Visualizations with automatic chart interpretation
    - Hypothesis testing based on preregistration information
    - Descriptive statistics by condition
    - Recommendations for student grading

    This is NOT shared with students - they should practice these analyses themselves.

    Report Sections:
    1. Data Quality Summary - Exclusions, attention, completion times
    2. Experimental Design Check - Condition balance, randomization
    3. Descriptive Statistics - By condition and overall
    4. Inferential Statistics - t-tests, ANOVA, effect sizes
    5. Visualization Gallery - Charts with interpretation
    6. Recommendations - Grading guidance for instructors

    Version: 2.2.1 - Enhanced interpretations and practical significance
    """

    # Report formatting constants
    SECTION_SEPARATOR = "=" * 80
    SUBSECTION_SEPARATOR = "-" * 80

    def __init__(self):
        self._warnings: List[str] = []
        self._insights: List[str] = []
        self.section_errors: List[str] = []  # sections the last report had to skip (see _run_section)

    def _run_section(self, ctx: Any, title: str, build: Any, slot: Optional[int] = None) -> bool:
        """Run one report section. If it raises, replace what it wrote by a one-line note and carry on.

        The note reads ``[section "<title>" could not be generated: <reason>]``; the failure is logged and added to
        ``self.section_errors`` so the caller can flag it. ``slot`` is the index of a reserved output entry (the
        executive-summary placeholder) that should hold the note instead of the end of the report.
        Returns True when the section ran cleanly.
        """
        mark = len(ctx.out)
        n_results = len(ctx.all_scale_results)
        try:
            build(ctx)
            return True
        except Exception as exc:  # noqa: BLE001 - one bad section must not cost the whole report
            reason = re.sub(r"\s+", " ", f"{type(exc).__name__}: {exc}").strip()[:300]
            logger.warning("Report section %r could not be generated: %s", title, reason, exc_info=True)
            self.section_errors.append(f"'{title}': {reason}")
            del ctx.out[mark:]
            del ctx.all_scale_results[n_results:]
            note = f'[section "{title}" could not be generated: {reason}]'
            if ctx.html:
                entries = [f"<div class='warning-box'>{_html_lib.escape(note)}</div>"]
            else:
                entries = [note, ""]
            if slot is not None and 0 <= slot < len(ctx.out):
                ctx.out[slot] = entries[0]
            else:
                ctx.out.extend(entries)
            return False

    def generate_comprehensive_report(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        schema_validation: Optional[Dict[str, Any]] = None,
        prereg_text: str = "",
        team_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate the comprehensive instructor-only report (Markdown).

        The report is built section by section, each behind a guard: a section that raises is replaced by one line,
        ``[section "<title>" could not be generated: <reason>]``, and the others are not affected. Skipped sections
        are logged and listed in ``self.section_errors``. Every dependent variable is its own guarded block.
        """
        self.section_errors = []
        ctx = _report_context(df, metadata, prereg_text, team_info, html=False)
        for title, build in (
            ("Study overview", self._md_overview),
            ("Data quality assurance", self._md_quality_assurance),
            ("1. Data quality summary", self._md_data_quality),
            ("2. Experimental design verification", self._md_design),
            ("3. Dependent variable analysis", self._md_dv_header),
        ):
            self._run_section(ctx, title, build)
        for scale in list(ctx.scales or []):
            ctx.dv_batch = [scale]
            name = scale.get("name", "Scale") if isinstance(scale, dict) else scale
            self._run_section(ctx, f"3. Dependent variable analysis: {name}", self._md_dv_scale)
        for title, build in (
            ("4. Preregistration alignment check", self._md_prereg),
            ("5. Persona distribution and impact", self._md_persona),
            ("6. Open-ended questions summary", self._md_open_ended),
            ("7. Effect size quality assessment", self._md_effect_sizes),
            ("8. Condition balance analysis", self._md_condition_balance),
            ("9. Instructor recommendations", self._md_recommendations),
        ):
            self._run_section(ctx, title, build)
        return _finalize_p_text("\n".join(map(str, ctx.out)), html=False)

    def _md_overview(self, ctx: Any) -> None:
        """Report header and study overview."""
        lines, metadata, team_info = ctx.out, ctx.metadata, ctx.team_info

        # Header
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE INSTRUCTOR REPORT (CONFIDENTIAL)")
        lines.append("=" * 80)
        lines.append("")
        lines.append("**This report is for instructor review only. Students receive a simpler version.**")
        lines.append("")

        # =============================================================
        # STUDY OVERVIEW SECTION (NEW - All first page info)
        # =============================================================
        lines.append("-" * 80)
        lines.append("## STUDY OVERVIEW")
        lines.append("-" * 80)
        lines.append("")

        # Study Title
        study_title = metadata.get('study_title', 'Untitled Study')
        lines.append(f"### {study_title}")
        lines.append("")

        # Team Information
        if team_info:
            team_name = team_info.get('team_name', '')
            team_members = team_info.get('team_members', '')
            if team_name:
                lines.append(f"**Team:** {team_name}")
            if team_members:
                # Format members nicely (handle newlines)
                members_formatted = team_members.replace('\n', ', ').replace(',,', ',').strip(', ')
                lines.append(f"**Team Members:** {members_formatted}")
            lines.append("")

        # Study Description / Abstract
        study_description = metadata.get('study_description', '')
        if study_description:
            lines.append("### Abstract / Study Description")
            lines.append("")
            lines.append(study_description)
            lines.append("")

        # v1.1.0.4: Generation method (visible in both user and instructor reports)
        _gen_method_label = metadata.get('generation_method_label', '')
        if _gen_method_label:
            lines.append(f"**Data Generation Method:** {_gen_method_label}")
            lines.append("")

        # Experimental Design
        lines.append("### Experimental Design")
        lines.append("")

        # Conditions
        conditions = metadata.get('conditions', [])
        if conditions:
            lines.append(f"**Conditions ({len(conditions)}):**")
            for i, cond in enumerate(conditions, 1):
                lines.append(f"  {i}. {cond}")
            lines.append("")

        # Factors (if factorial design)
        factors = metadata.get('factors', [])
        if factors:
            lines.append(f"**Factors ({len(factors)}):**")
            for factor in factors:
                factor_name = factor.get('name', 'Factor')
                levels = factor.get('levels', [])
                lines.append(f"  - {factor_name}: {', '.join(str(l) for l in levels)}")
            lines.append("")

        # Scales / DVs
        scales = metadata.get('scales', [])
        if scales:
            lines.append(f"**Dependent Variables / Scales ({len(scales)}):**")
            for scale in scales:
                scale_name = scale.get('name', 'Scale')
                scale_points = scale.get('scale_points', 7)
                num_items = scale.get('num_items', 1)
                lines.append(f"  - {scale_name} ({num_items} item{'s' if num_items > 1 else ''}, {scale_points}-point scale)")
            lines.append("")

        # Effect Sizes (hypotheses)
        effect_sizes = metadata.get('effect_sizes_configured', [])
        if effect_sizes:
            lines.append("**Hypothesized Effects:**")
            for effect in effect_sizes:
                var = effect.get('variable', '')
                factor = effect.get('factor', '')
                d = effect.get('cohens_d', 0)
                direction = effect.get('direction', 'higher')
                if d > 0:
                    lines.append(f"  - {var}: d = {d:.2f} ({direction} in treatment)")
            lines.append("")

        # Sample Size
        sample_size = metadata.get('sample_size', 0)
        lines.append(f"**Sample Size:** N = {sample_size}")
        lines.append("")

        # Generation Info
        lines.append("### Generation Details")
        lines.append("")
        lines.append(f"**Generated:** {metadata.get('generation_timestamp', datetime.now().isoformat())}")
        lines.append(f"**Run ID:** `{metadata.get('run_id', 'N/A')}`")
        lines.append(f"**Mode:** {metadata.get('simulation_mode', 'pilot').title()}")

        # Internal usage counter (for instructor tracking)
        usage_stats = metadata.get('usage_stats', {})
        total_simulations = usage_stats.get('total_simulations', 'N/A')
        lines.append(f"**Total Simulations Run (all time):** {total_simulations}")
        lines.append("")

    def _md_quality_assurance(self, ctx: Any) -> None:
        """Automated quality checks: scale ranges and open-ended response uniqueness."""
        lines, df, metadata = ctx.out, ctx.df, ctx.metadata

        # === v1.2.5: DATA QUALITY ASSURANCE SECTION ===
        lines.append("")
        lines.append("-" * 80)
        lines.append("## DATA QUALITY ASSURANCE")
        lines.append("-" * 80)
        lines.append("")
        lines.append("### Automated Quality Checks")
        lines.append("")

        # Scale range verification
        scales = metadata.get("scales", [])
        if scales:
            lines.append("| Scale | Items | Expected Range | Actual Range | Status |")
            lines.append("|-------|-------|---------------|--------------|--------|")
            # v1.2.9.1: same column lookup as the analysis below (registry first, numeric columns only). Matching
            # "<name>_<digits>" by hand picked up an open-ended TEXT column (UFuncTypeError, both analyses stubbed)
            # and missed names such as "1.9Q" whose columns are "1_9Q_1".
            _range_registry = _scale_column_registry(metadata)
            for scale in scales:
                s_name = str(scale.get("name", "Unknown")).strip().replace(" ", "_")
                n_items = scale.get("num_items", 5)
                s_min = scale.get("scale_min", 1)
                s_max = scale.get("scale_max", 7)
                cols = _find_scale_columns(df, scale, _range_registry)
                if cols:
                    actual_min = df[cols].min().min()
                    actual_max = df[cols].max().max()
                    if not (_is_finite_number(actual_min) and _is_finite_number(actual_max)):
                        lines.append(f"| {s_name} | {len(cols)} | [{s_min}-{s_max}] | no answers | ⚠️ Review |")
                        continue
                    status = "✅ Pass" if actual_min >= s_min and actual_max <= s_max else "⚠️ Review"
                    lines.append(f"| {s_name} | {len(cols)} | [{s_min}-{s_max}] | [{actual_min}-{actual_max}] | {status} |")
                else:
                    lines.append(f"| {s_name} | 0 | [{s_min}-{s_max}] | no columns found | ⚠️ Review |")
            lines.append("")

        # Response uniqueness check for open-ended (v1.2.5.1: centralized detection)
        try:
            from utils import detect_oe_columns as _detect_oe_uniq
            oe_cols = _detect_oe_uniq(df)
        except (ImportError, Exception):
            oe_cols = []
        if oe_cols:
            lines.append("### Open-Ended Response Uniqueness")
            lines.append("")
            for col in oe_cols:
                responses = df[col].dropna().tolist()
                unique_responses = len(set(responses))
                total = len(responses)
                pct = (unique_responses / total * 100) if total > 0 else 0
                status = "✅" if pct >= 95 else ("⚠️" if pct >= 80 else "❌")
                lines.append(f"- {col}: {unique_responses}/{total} unique ({pct:.1f}%) {status}")
            lines.append("")

    def _md_data_quality(self, ctx: Any) -> None:
        """Section 1: exclusions, attention checks and completion times."""
        lines, df = ctx.out, ctx.df
        n_total, n_excluded, n_clean, exclusion_rate = ctx.n_total, ctx.n_excluded, ctx.n_clean, ctx.exclusion_rate

        # =============================================================
        # SECTION 1: DATA QUALITY SUMMARY
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 1. DATA QUALITY SUMMARY")
        lines.append("-" * 80)
        lines.append("")

        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total N | {n_total} |")
        lines.append(f"| Excluded | {n_excluded} ({exclusion_rate:.1f}%) |")
        lines.append(f"| Clean N | {n_clean} |")
        lines.append(f"| Missing cells | {int(df.isna().sum().sum())} |")
        lines.append("")

        # Attention check analysis
        if "Attention_Pass_Rate" in df.columns:
            lines.append("### Attention Check Analysis")
            lines.append("")
            attention_stats = df["Attention_Pass_Rate"].describe()
            lines.append(f"- Mean pass rate: {attention_stats['mean']:.2%}")
            lines.append(f"- Median pass rate: {attention_stats['50%']:.2%}")
            lines.append(f"- Failed all checks (0%): {(df['Attention_Pass_Rate'] == 0).sum()} participants")
            lines.append(f"- Passed all checks (100%): {(df['Attention_Pass_Rate'] == 1).sum()} participants")
            lines.append("")

        # Completion time analysis
        if "Completion_Time_Seconds" in df.columns:
            lines.append("### Completion Time Analysis")
            lines.append("")
            time_stats = df["Completion_Time_Seconds"].describe()
            lines.append(f"- Mean: {time_stats['mean']:.1f} seconds ({time_stats['mean']/60:.1f} min)")
            lines.append(f"- Median: {time_stats['50%']:.1f} seconds")
            lines.append(f"- Min: {time_stats['min']:.1f} seconds")
            lines.append(f"- Max: {time_stats['max']:.1f} seconds")
            lines.append(f"- Suspiciously fast (<60s): {(df['Completion_Time_Seconds'] < 60).sum()}")
            lines.append(f"- Very slow (>30min): {(df['Completion_Time_Seconds'] > 1800).sum()}")
            lines.append("")

    def _md_design(self, ctx: Any) -> None:
        """Section 2: design type, condition distribution and factor structure."""
        lines, metadata, df, n_total = ctx.out, ctx.metadata, ctx.df, ctx.n_total
        conditions, factors, scales = ctx.conditions, ctx.factors, ctx.scales

        # =============================================================
        # SECTION 2: EXPERIMENTAL DESIGN CHECK
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 2. EXPERIMENTAL DESIGN VERIFICATION")
        lines.append("-" * 80)
        lines.append("")

        lines.append(f"**Design type:** {metadata.get('design_type', 'Between-subjects')}")
        lines.append(f"**Number of conditions:** {len(conditions)}")
        lines.append(f"**Number of factors:** {len(factors)}")
        lines.append(f"**Number of DVs:** {len(scales)}")
        lines.append("")

        # Condition distribution
        if "CONDITION" in df.columns:
            lines.append("### Condition Distribution")
            lines.append("")
            cond_counts = df["CONDITION"].value_counts()
            lines.append("| Condition | N | % |")
            lines.append("|-----------|---|---|")
            for cond, count in cond_counts.items():
                pct = count / n_total * 100
                lines.append(f"| {cond} | {count} | {pct:.1f}% |")
            lines.append("")

            # Balance check
            cv = cond_counts.std() / cond_counts.mean() * 100 if cond_counts.mean() > 0 else 0
            if cv < 5:
                lines.append("✅ **Excellent balance** (CV < 5%)")
            elif cv < 10:
                lines.append("✅ **Good balance** (CV < 10%)")
            elif cv < 20:
                lines.append("⚠️ **Slight imbalance** (CV 10-20%)")
            else:
                lines.append("❌ **Notable imbalance** (CV > 20%)")
            lines.append("")

        # Factor structure
        if factors:
            lines.append("### Factor Structure")
            lines.append("")
            for f in factors:
                fname = f.get("name", "Factor")
                levels = f.get("levels", [])
                lines.append(f"- **{fname}**: {', '.join(str(l) for l in levels)} ({len(levels)} levels)")
            lines.append("")

    def _md_dv_header(self, ctx: Any) -> None:
        """Section 3 heading; also prepares the data and column registry the per-DV blocks use."""
        lines, df, metadata = ctx.out, ctx.df, ctx.metadata

        # =============================================================
        # SECTION 3: DV ANALYSIS & STATISTICS
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 3. DEPENDENT VARIABLE ANALYSIS")
        lines.append("-" * 80)
        lines.append("")

        # v1.2.4.0: Use FULL dataset for instructor report analysis.
        # Exclude_Recommended is informational (teaches students about data quality)
        # but should NOT reduce the analysis N — instructors expect the full sample.
        df_clean = df

        # v1.4.11: Build column registry from scale_generation_log if available
        _col_registry: Dict[str, List[str]] = _scale_column_registry(metadata)
        ctx.df_clean, ctx.col_registry = df_clean, _col_registry

    def _md_dv_scale(self, ctx: Any) -> None:
        """Section 3: the block of one dependent variable (``ctx.dv_batch`` holds just that scale)."""
        lines, df_clean, conditions = ctx.out, ctx.df_clean, ctx.conditions
        _col_registry, scales = ctx.col_registry, ctx.dv_batch

        for scale in scales:
            scale_name = scale.get("name", "Scale")
            num_items = scale.get("num_items", 5)
            scale_points = scale.get("scale_points", 7)

            lines.append(f"### {scale_name}")
            lines.append("")
            lines.append(f"Configuration: {num_items} items, {scale_points}-point scale")
            lines.append("")

            # v1.0.6.3: Use unified column finder (fixes N=1 / missing DV analysis)
            scale_cols = _find_scale_columns(df_clean, scale, _col_registry)

            if scale_cols:
                # Item-level statistics
                lines.append("#### Item-level Statistics")
                lines.append("")
                lines.append("| Item | Mean | SD | Min | Max |")
                lines.append("|------|------|----|----|-----|")
                for col in scale_cols[:10]:  # Limit to first 10 items
                    if col in df_clean.columns:
                        stats = df_clean[col].describe()
                        lines.append(f"| {col} | {_fnum(stats['mean'], '.2f')} | {_fnum(stats['std'], '.2f')} | {_fnum(stats['min'], '.0f')} | {_fnum(stats['max'], '.0f')} |")
                lines.append("")

                # v1.0.5.4: Compute composite for multi-item scales, use raw values for single-item DVs
                if len(scale_cols) >= 2:
                    composite = df_clean[scale_cols].mean(axis=1)
                    lines.append("#### Composite Score (Mean)")
                    lines.append("")
                    comp_stats = composite.describe()
                    lines.append(f"- Mean: {_fnum(comp_stats['mean'], '.3f')}")
                    lines.append(f"- SD: {_fnum(comp_stats['std'], '.3f')}")
                    lines.append(f"- Range: [{_fnum(comp_stats['min'], '.2f')}, {_fnum(comp_stats['max'], '.2f')}]")
                    lines.append("")
                else:
                    # Single-item DV — use the column directly (no composite needed)
                    composite = df_clean[scale_cols[0]]

                # v1.2.9.1: a composite that cannot differ between participants (constant-sum or rank-order items,
                # or no variation at all) is not analysed: SD = 0 in every group makes the by-condition table and
                # Cohen's d meaningless. Say so and compare the items instead.
                if _is_joint_item_scale(scale) or _composite_has_no_variation(
                        composite, df_clean["CONDITION"] if "CONDITION" in df_clean.columns else None):
                    lines.extend(self._no_variation_block(scale, scale_cols, df_clean, composite, conditions, html=False))
                    lines.append("")
                    continue

                # v1.0.5.4: By-condition analysis runs for ALL DVs (single-item and multi-item)
                if "CONDITION" in df_clean.columns:
                    lines.append("#### By Condition")
                    lines.append("")
                    lines.append("| Condition | N | Mean | SD | 95% CI |")
                    lines.append("|-----------|---|------|----|---------| ")

                    df_clean_copy = df_clean.copy()
                    df_clean_copy["_composite"] = composite

                    _assigned_md: Dict[str, int] = {}
                    for cond in conditions:
                        cond_data = df_clean_copy[df_clean_copy["CONDITION"] == cond]["_composite"]
                        # v1.2.5.1: SD, SE and CI use valid N only.
                        # v1.2.9.1: the displayed N is that same analytic N (participants with a score),
                        # not the number assigned to the condition; the difference is disclosed below the table.
                        n_total_c = len(cond_data)
                        cond_valid = cond_data.dropna()
                        n_valid_c = len(cond_valid)
                        if n_valid_c > 0:
                            mean = cond_valid.mean()
                            n = n_valid_c  # Display analytic N
                            sd = cond_valid.std() if n_valid_c > 1 else 0.0
                            if np.isnan(sd):
                                sd = 0.0
                            ci_half = _t_ci_halfwidth(sd, n_valid_c)  # t critical value (1.96 is too narrow for small n)
                            ci_low = mean - ci_half
                            ci_high = mean + ci_half
                            if n_total_c != n_valid_c:
                                _assigned_md[str(cond)] = n_total_c
                            if n == 1:
                                lines.append(f"| {cond} | {n} | {mean:.3f} | — | N/A (single observation) |")
                            else:
                                lines.append(f"| {cond} | {n} | {mean:.3f} | {sd:.3f} | [{ci_low:.3f}, {ci_high:.3f}] |")
                    lines.append("")
                    if _assigned_md:
                        lines.append("*N is the number of participants with a score (the N every statistic uses). "
                                     "Assigned to the condition, including those without a score: "
                                     + ", ".join(f"{k} {v}" for k, v in _assigned_md.items()) + ".*")
                        lines.append("")

                    # Effect size (Cohen's d for first two conditions; first minus second, like every contrast in the report)
                    if len(conditions) >= 2:
                        cond1_data = df_clean_copy[df_clean_copy["CONDITION"] == conditions[0]]["_composite"].dropna()
                        cond2_data = df_clean_copy[df_clean_copy["CONDITION"] == conditions[1]]["_composite"].dropna()
                        if len(cond1_data) > 1 and len(cond2_data) > 1:
                            pooled_std = ((cond1_data.std()**2 + cond2_data.std()**2) / 2) ** 0.5
                            if pooled_std > 0 and not (_is_constant(cond1_data.to_numpy()) and _is_constant(cond2_data.to_numpy())):
                                cohens_d = (cond1_data.mean() - cond2_data.mean()) / pooled_std
                                lines.append(f"**Effect size (Cohen's d, {conditions[0]} - {conditions[1]}):** {cohens_d:.3f}")
                                _d_label = _cohens_d_label(cohens_d)
                                lines.append(f"  → {_d_label[:1].upper()}{_d_label[1:]} effect")
                                lines.append(f"  (first minus second: positive means {conditions[0]} scores higher"
                                             f"{'; first two conditions only' if len(conditions) > 2 else ''})")
                                lines.append("")

            lines.append("")

    def _md_prereg(self, ctx: Any) -> None:
        """Section 4: preregistration alignment check."""
        lines, prereg_text, scales, n_total = ctx.out, ctx.prereg_text, ctx.scales, ctx.n_total

        # =============================================================
        # SECTION 4: PREREGISTRATION CHECK
        # =============================================================
        if prereg_text:
            lines.append("-" * 80)
            lines.append("## 4. PREREGISTRATION ALIGNMENT CHECK")
            lines.append("-" * 80)
            lines.append("")

            # Sample size check
            prereg_lower = prereg_text.lower()
            lines.append("### Key Checks")
            lines.append("")

            # Look for sample size mentions
            import re
            size_matches = re.findall(r'n\s*=\s*(\d+)|sample.*?(\d+)|(\d+)\s*participants', prereg_lower)
            if size_matches:
                lines.append(f"**Sample size in preregistration:** Patterns found - check matches actual N={n_total}")

            # Look for scale mentions
            for scale in scales:
                scale_name = scale.get("name", "").lower()
                scale_points = scale.get("scale_points", 7)
                if scale_name in prereg_lower:
                    lines.append(f"**{scale_name}:** Mentioned in preregistration. Simulated with {scale_points}-point scale.")
                    # Check for scale point mentions
                    point_patterns = [f"{scale_points}-point", f"{scale_points} point", f"1-{scale_points}"]
                    if any(p in prereg_lower for p in point_patterns):
                        lines.append(f"  ✅ Scale points match preregistration")
                    else:
                        lines.append(f"  ⚠️ Verify scale points match preregistration")

            lines.append("")
            lines.append("### Preregistration Text (excerpt)")
            lines.append("```")
            lines.append(prereg_text[:1500] + ("..." if len(prereg_text) > 1500 else ""))
            lines.append("```")
            lines.append("")

    def _md_persona(self, ctx: Any) -> None:
        """Section 5: persona distribution and its impact on the data."""
        lines, metadata, n_total = ctx.out, ctx.metadata, ctx.n_total

        # =============================================================
        # SECTION 5: PERSONA DISTRIBUTION ANALYSIS
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 5. PERSONA DISTRIBUTION & IMPACT")
        lines.append("-" * 80)
        lines.append("")

        dist_raw = metadata.get("persona_distribution", {}) or {}
        dist = _extract_persona_proportions(dist_raw)
        if dist:
            lines.append("| Persona | % | Expected N | Impact on Data |")
            lines.append("|---------|---|------------|----------------|")
            for persona, share in sorted(dist.items(), key=lambda x: -_safe_float(x[1])):
                share_f = _safe_float(share)
                share_val = share_f if share_f <= 1 else share_f / 100
                pct = share_val * 100
                count = int(round(n_total * share_val))
                impact = self._get_detailed_impact(persona.lower())
                lines.append(f"| {persona.title()} | {pct:.1f}% | ~{count} | {impact} |")
            lines.append("")

            # Estimate impact on results
            lines.append("### Estimated Impact on Results")
            lines.append("")
            acquiescent_share = _safe_float(dist.get("acquiescent", 0))
            skeptic_share = _safe_float(dist.get("skeptic", 0))
            if acquiescent_share > 0.1:
                lines.append(f"⚠️ High acquiescence ({acquiescent_share:.0%}) may inflate positive responses")
            if skeptic_share > 0.1:
                lines.append(f"⚠️ High skepticism ({skeptic_share:.0%}) may deflate responses")
            careless_share = _safe_float(dist.get("careless", 0)) + _safe_float(dist.get("random", 0))
            if careless_share > 0.1:
                lines.append(f"⚠️ Notable careless/random ({careless_share:.0%}) - verify exclusion criteria are working")
            lines.append("")

    def _md_open_ended(self, ctx: Any) -> None:
        """Section 6: open-ended questions and how the responses were generated."""
        lines, df, metadata, conditions = ctx.out, ctx.df, ctx.metadata, ctx.conditions

        # =============================================================
        # SECTION 6: OPEN-ENDED QUESTIONS SUMMARY (NEW v2.4.4)
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 6. OPEN-ENDED QUESTIONS SUMMARY")
        lines.append("-" * 80)
        lines.append("")

        # v1.2.5.1: Use centralized OE detection to prevent CONDITION/metadata
        # columns from being misidentified as open-ended text
        try:
            from utils import detect_oe_columns
            open_ended_cols = detect_oe_columns(df)
        except (ImportError, Exception):
            open_ended_cols = []

        if open_ended_cols:
            lines.append(f"**{len(open_ended_cols)} open-ended question(s) simulated:**")
            lines.append("")
            for col in open_ended_cols[:10]:  # Limit to first 10
                lines.append(f"### {col}")
                lines.append("")
                # Show sample responses by condition if possible
                if "CONDITION" in df.columns and len(conditions) > 0:
                    for cond in conditions[:3]:  # First 3 conditions
                        cond_data = df[df["CONDITION"] == cond][col]
                        if len(cond_data) > 0:
                            sample = cond_data.iloc[0]
                            if isinstance(sample, str) and len(sample) > 0:
                                truncated = sample[:200] + "..." if len(sample) > 200 else sample
                                lines.append(f"**{cond}:** *\"{truncated}\"*")
                                lines.append("")
                else:
                    # Just show first response
                    sample = df[col].iloc[0] if len(df) > 0 else ""
                    if isinstance(sample, str) and len(sample) > 0:
                        truncated = sample[:300] + "..." if len(sample) > 300 else sample
                        lines.append(f"*Sample:* \"{truncated}\"")
                        lines.append("")

                # Response length statistics
                lengths = df[col].apply(lambda x: len(str(x)) if x else 0)
                lines.append(f"- **Response length:** Mean={lengths.mean():.0f} chars, Range=[{lengths.min()}-{lengths.max()}]")
                lines.append("")
        else:
            lines.append("No open-ended questions were simulated in this dataset.")
            lines.append("")

        # v1.0.9.2: LLM Generation Details (in comprehensive report) — with method differentiation
        llm_stats = metadata.get('llm_stats', metadata.get('llm_response_stats', {}))
        llm_calls = llm_stats.get('llm_calls', 0) if llm_stats else 0
        llm_attempts = llm_stats.get('llm_attempts', llm_calls) if llm_stats else 0
        pool_size = llm_stats.get('pool_size', 0) if llm_stats else 0
        fallback_uses = llm_stats.get('fallback_uses', 0) if llm_stats else 0
        open_ended_qs = metadata.get('open_ended_questions', [])
        _gen_method_comp = metadata.get('generation_method', '')
        _is_adaptive_comp = _gen_method_comp in ('experimental', 'abe_v2')
        _is_template_comp = _gen_method_comp == 'template'
        _is_abe3_comp = _gen_method_comp == 'abe3'
        _is_llm_comp = _gen_method_comp in ('free_llm', 'own_api')  # v1.2.5.4
        if open_ended_qs:
            lines.append("### Response Generation Method")
            lines.append("")
            if pool_size > 0:
                if _is_abe3_comp or _is_adaptive_comp or _is_llm_comp:
                    lines.append("**Generation approach:** Adaptive Behavioral Engine 3.0 + AI-Powered LLM")
                else:
                    lines.append("**Generation approach:** AI-Powered (Language Model)")
                lines.append("")
                lines.append(f"- **Total API calls:** {llm_calls}")
                lines.append(f"- **Response pool size:** {pool_size} base responses")
                if fallback_uses > 0:
                    total = pool_size + fallback_uses
                    pct = (fallback_uses / max(1, total)) * 100
                    lines.append(f"- **Template fallback:** {fallback_uses} response(s) ({pct:.0f}%)")
                else:
                    lines.append("- **Template fallback:** None needed (all AI-generated)")
            elif llm_calls > 0 or llm_attempts > 0:
                if _is_abe3_comp or _is_adaptive_comp or _is_llm_comp:
                    lines.append("**Generation approach:** Adaptive Behavioral Engine 3.0 (template-backed, AI unavailable)")
                else:
                    lines.append("**Generation approach:** Template Engine (AI providers were unavailable)")
                lines.append("")
                _api_req_count = max(llm_calls, llm_attempts)
                lines.append(f"- {_api_req_count} API request(s) were sent but all providers were unavailable or returned errors")
                lines.append("- All open-ended responses were generated using the built-in template engine")
            else:
                if _is_abe3_comp or _is_llm_comp:
                    lines.append("**Generation approach:** Adaptive Behavioral Engine 3.0")
                    lines.append("")
                    lines.append("Responses were generated using the Adaptive Behavioral Engine 3.0 with census-weighted demographics,")
                    lines.append("calibrated error rates, and stylometric voice fingerprinting for each participant.")
                    if _is_llm_comp:
                        _comp_label = metadata.get('generation_method_label', _gen_method_comp)
                        lines.append(f"**Selected method:** {_comp_label}")
                elif _is_adaptive_comp:
                    lines.append("**Generation approach:** Adaptive Behavioral Engine 3.0")
                elif _is_template_comp:
                    lines.append("**Generation approach:** Template Engine (225+ Research Domains)")
                else:
                    lines.append("**Generation approach:** Advanced Template Engine (225+ Research Domains)")
                lines.append("")
                lines.append("Responses were generated using the built-in behavioral simulation engine covering 225+ research domains.")
                lines.append("Each response is unique per participant with topic-grounded content matching persona profiles.")
            lines.append("")

    def _md_effect_sizes(self, ctx: Any) -> None:
        """Section 7: configured versus observed effect sizes."""
        lines, metadata = ctx.out, ctx.metadata

        # =============================================================
        # SECTION 7: EFFECT SIZE QUALITY ASSESSMENT (NEW v2.4.4)
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 7. EFFECT SIZE QUALITY ASSESSMENT")
        lines.append("-" * 80)
        lines.append("")

        configured_effects = metadata.get("effect_sizes_configured", [])
        observed_effects = metadata.get("effect_sizes_observed", [])

        if configured_effects:
            lines.append("### Configured vs Observed Effect Sizes")
            lines.append("")
            lines.append("| Variable | Configured d | Observed d | Match Quality |")
            lines.append("|----------|--------------|------------|---------------|")

            for cfg in configured_effects:
                var = cfg.get("variable", "Unknown")
                cfg_d = cfg.get("cohens_d", 0)
                # Find matching observed effect
                obs_d = None
                for obs in observed_effects:
                    if obs.get("variable") == var:
                        obs_d = obs.get("cohens_d", obs.get("d_observed"))
                        break

                if obs_d is not None:
                    diff = abs(cfg_d - obs_d)
                    if diff < 0.1:
                        quality = "✅ Excellent"
                    elif diff < 0.2:
                        quality = "✅ Good"
                    elif diff < 0.3:
                        quality = "⚠️ Acceptable"
                    else:
                        quality = "❌ Poor match"
                    lines.append(f"| {var} | {cfg_d:.3f} | {obs_d:.3f} | {quality} |")
                else:
                    lines.append(f"| {var} | {cfg_d:.3f} | N/A | ⚠️ Not computed |")
            lines.append("")

            # Overall assessment
            lines.append("### Quality Interpretation")
            lines.append("")
            lines.append("Effect sizes are calibrated from published meta-analyses and research findings.")
            lines.append("Good matches indicate the simulation faithfully reproduces expected effect magnitudes.")
            lines.append("")
        else:
            lines.append("No effect sizes were explicitly configured for this simulation.")
            lines.append("The simulation used domain-inferred defaults based on study context.")
            lines.append("")

    def _md_condition_balance(self, ctx: Any) -> None:
        """Section 8: participants per condition."""
        lines, df, conditions = ctx.out, ctx.df, ctx.conditions

        # =============================================================
        # SECTION 8: CONDITION BALANCE ANALYSIS (NEW v2.4.4)
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 8. CONDITION BALANCE ANALYSIS")
        lines.append("-" * 80)
        lines.append("")

        if "CONDITION" in df.columns:
            cond_counts = df["CONDITION"].value_counts()
            total = len(df)
            expected_per_cond = total / len(conditions) if conditions else total

            lines.append("### Participant Distribution by Condition")
            lines.append("")
            lines.append("| Condition | N | % | Deviation from Expected |")
            lines.append("|-----------|---|---|------------------------|")

            max_deviation = 0
            for cond in conditions:
                count = cond_counts.get(cond, 0)
                pct = (count / total * 100) if total > 0 else 0
                deviation = abs(count - expected_per_cond)
                deviation_pct = (deviation / expected_per_cond * 100) if expected_per_cond > 0 else 0
                max_deviation = max(max_deviation, deviation_pct)

                if deviation_pct < 5:
                    dev_indicator = "✅"
                elif deviation_pct < 10:
                    dev_indicator = "⚠️"
                else:
                    dev_indicator = "❌"

                lines.append(f"| {cond} | {count} | {pct:.1f}% | {dev_indicator} {deviation_pct:.1f}% |")

            lines.append("")

            # Balance assessment
            lines.append("### Balance Assessment")
            lines.append("")
            if max_deviation < 5:
                lines.append("✅ **Excellent balance** - Conditions are evenly distributed.")
            elif max_deviation < 10:
                lines.append("✅ **Good balance** - Minor deviations within acceptable range.")
            elif max_deviation < 15:
                lines.append("⚠️ **Acceptable balance** - Some imbalance, but unlikely to affect analysis.")
            else:
                lines.append("❌ **Imbalanced** - Consider adjusting condition allocation for future runs.")
            lines.append("")

            # Check for empty conditions
            empty_conds = [c for c in conditions if cond_counts.get(c, 0) == 0]
            if empty_conds:
                lines.append(f"⚠️ **Warning:** {len(empty_conds)} condition(s) have no participants: {', '.join(empty_conds)}")
                lines.append("")
        else:
            lines.append("No CONDITION column found in data.")
            lines.append("")

    def _md_recommendations(self, ctx: Any) -> None:
        """Section 9: recommendations for instructors, and the report footer."""
        lines, conditions, n_clean, exclusion_rate = ctx.out, ctx.conditions, ctx.n_clean, ctx.exclusion_rate

        # =============================================================
        # SECTION 9: RECOMMENDATIONS
        # =============================================================
        lines.append("-" * 80)
        lines.append("## 9. INSTRUCTOR RECOMMENDATIONS")
        lines.append("-" * 80)
        lines.append("")

        lines.append("### For Student Evaluation")
        lines.append("")
        lines.append("1. **Data cleaning**: Have students identify and justify exclusions")
        lines.append("2. **Descriptive statistics**: Verify they compute means/SDs by condition")
        lines.append("3. **Effect sizes**: Check if they interpret Cohen's d correctly")
        lines.append("4. **Scale reliability**: If they compute Cronbach's alpha, it should be reasonable")
        lines.append("5. **Visualization**: Check for appropriate choice of plots")
        lines.append("")

        lines.append("### Things to Watch For")
        lines.append("")
        if exclusion_rate > 15:
            lines.append(f"- Exclusion rate is {exclusion_rate:.1f}% - ask students to justify their criteria")
        if n_clean < 30:
            lines.append(f"- Clean N is only {n_clean} - discuss power implications")
        if len(conditions) > 4:
            lines.append(f"- {len(conditions)} conditions may be complex for analysis")
        lines.append("")

        # Footer
        lines.append("-" * 80)
        lines.append("END OF COMPREHENSIVE INSTRUCTOR REPORT")
        lines.append("-" * 80)

    def _get_detailed_impact(self, persona: str) -> str:
        """Get detailed impact description for instructor understanding."""
        impacts = {
            "engaged responder": "High quality data, typical variance patterns",
            "engaged": "High quality data, typical variance patterns",
            "satisficer": "Central tendency bias, may reduce effect detection",
            "extreme responder": "Inflated variance, potential outlier effects",
            "extreme": "Inflated variance, potential outlier effects",
            "acquiescent": "Positively skewed responses, inflated agreement",
            "skeptic": "Negatively skewed responses, deflated agreement",
            "random": "Noise injection, should be caught by exclusion criteria",
            "careless": "Straight-lining patterns, should trigger exclusion",
            "careful responder": "Consistent, low-variance responses",
            "moderate responder": "Restricted range, may reduce variance",
        }
        return impacts.get(persona, "Standard patterns")

    @staticmethod
    def _p_significance(p: float) -> dict:
        """Return significance classification for a p-value.

        Returns dict with keys: significant (bool), marginally_significant (bool),
        sig_label (str: '***', '**', '*', '†', 'ns').
        - p < 0.001: significant, label '***'
        - p < 0.01:  significant, label '**'
        - p < 0.05:  significant, label '*'
        - p < 0.10:  marginally significant, label '†'
        - p >= 0.10: not significant, label 'ns'
        """
        # Handle NaN / inf p-values gracefully
        if p is None or pd.isna(p):  # NaN check (v1.2.8.8)
            return {"significant": False, "marginally_significant": False, "sig_label": "ns"}
        if p < 0.001:
            return {"significant": True, "marginally_significant": False, "sig_label": "***"}
        elif p < 0.01:
            return {"significant": True, "marginally_significant": False, "sig_label": "**"}
        elif p < 0.05:
            return {"significant": True, "marginally_significant": False, "sig_label": "*"}
        elif p < 0.10:
            return {"significant": False, "marginally_significant": True, "sig_label": "†"}
        else:
            return {"significant": False, "marginally_significant": False, "sig_label": "ns"}

    def _run_statistical_tests(
        self,
        df: pd.DataFrame,
        dv_column: str,
        condition_column: str = "CONDITION",
        condition_order: Optional[List[Any]] = None,
        condition_labels: Optional[Dict[Any, str]] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive statistical tests on the data.

        Uses scipy if available, falls back to the exact numpy implementations otherwise (both give
        the same p-values to better than 1e-9).

        v1.2.9.1 orientation: conditions are ordered ONCE -- ``condition_order`` (the metadata order)
        when given, otherwise first appearance in the data -- and t, Cohen's d and every pairwise
        contrast are "first minus second" in that order; the contrast text is stored under ``contrast``
        so the report can print it next to the statistic.  Groups with fewer than 2 valid observations
        cannot be tested; they are listed under ``groups_excluded`` instead of silently disappearing.
        Zero-variance groups give NaN ("undefined"), never an infinite t, and the pairwise verdicts use
        Holm-adjusted p-values (the raw p stays in ``p_value``).
        """
        results: Dict[str, Any] = {}
        results["scipy_used"] = SCIPY_AVAILABLE

        # The ONE canonical condition order (metadata order when supplied, else first appearance)
        conditions = _order_conditions(df[condition_column].dropna().unique().tolist(), condition_order)

        if len(conditions) < 2:
            return {"error": "Need at least 2 conditions for comparison"}

        labels = {**_condition_display_labels(conditions), **(condition_labels or {})}

        def _label(cond: Any) -> str:
            return labels.get(cond) or _clean_condition_name(str(cond))

        # Get data by condition — dropna is needed for statistical computations
        # but we track both total N and valid N per condition.
        groups = {
            cond: pd.to_numeric(df.loc[df[condition_column] == cond, dv_column], errors="coerce").dropna().to_numpy(dtype=float)
            for cond in conditions
        }
        # v1.2.5.0: Track total N per condition (including NaN rows) for accurate reporting
        total_n_per_cond = {cond: int((df[condition_column] == cond).sum()) for cond in conditions}

        # Groups with fewer than two valid observations cannot be tested -- say so instead of dropping silently
        valid_groups = {c: g for c, g in groups.items() if len(g) >= 2}
        results["groups_excluded"] = [
            {"condition": _label(c), "n": int(len(groups[c])), "n_assigned": total_n_per_cond.get(c, 0)}
            for c in conditions if c not in valid_groups
        ]
        if len(valid_groups) < 2:
            return {"error": "Need at least 2 groups with 2+ observations each",
                    "groups_excluded": results["groups_excluded"]}

        conditions = list(valid_groups.keys())
        groups = valid_groups
        results["conditions"] = [_label(c) for c in conditions]

        # Basic descriptive stats
        # v1.2.9.1: "n" is the ANALYTIC N (observations actually used); the assigned N stays available
        results["descriptives"] = {}
        for cond, data in groups.items():
            n_valid = len(data)
            sd = float(np.std(data, ddof=1))
            results["descriptives"][_label(cond)] = {
                "n": n_valid,
                "n_assigned": total_n_per_cond.get(cond, n_valid),
                "n_valid": n_valid,  # Participants with valid DV data
                "mean": float(np.mean(data)),
                "std": sd,
                "median": float(np.median(data)),
                "se": sd / math.sqrt(n_valid),
                "ci_halfwidth": _t_ci_halfwidth(sd, n_valid),
            }

        # Define test functions (scipy or numpy fallback)
        if _use_scipy():
            ttest_func = lambda g1, g2, eq_var: scipy_stats.ttest_ind(g1, g2, equal_var=eq_var)
            anova_func = scipy_stats.f_oneway
            mannwhitney_func = lambda g1, g2: scipy_stats.mannwhitneyu(g1, g2, alternative='two-sided')
            levene_func = scipy_stats.levene
            shapiro_func = scipy_stats.shapiro
            kruskal_func = scipy_stats.kruskal
        else:
            ttest_func = lambda g1, g2, eq_var: _numpy_ttest_ind(g1, g2, equal_var=eq_var)
            anova_func = _numpy_f_oneway
            mannwhitney_func = _numpy_mannwhitneyu
            levene_func = _numpy_levene
            shapiro_func = _numpy_shapiro
            kruskal_func = _numpy_kruskal

        nan = float("nan")

        def _finite_pair(pair: Any) -> Tuple[float, float]:
            stat, p_val = float(pair[0]), float(pair[1])
            if not (math.isfinite(stat) and math.isfinite(p_val)):
                return nan, nan  # e.g. zero variance: t would be +/-inf and p = 0 ("significant") -- undefined instead
            return stat, p_val

        def _t_test(a: np.ndarray, b: np.ndarray, equal_var: bool) -> Tuple[float, float]:
            if _is_constant(a) and _is_constant(b):
                return nan, nan
            return _finite_pair(ttest_func(a, b, equal_var))

        def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
            mean_diff = float(np.mean(a) - np.mean(b))
            pooled = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2))
            if pooled > 0 and not (_is_constant(a) and _is_constant(b)):
                return mean_diff / pooled
            return 0.0 if mean_diff == 0 else nan

        try:
            # Two-group comparisons
            if len(conditions) == 2:
                c1, c2 = conditions[0], conditions[1]
                g1, g2 = groups[c1], groups[c2]
                lab1, lab2 = _label(c1), _label(c2)
                contrast = f"{lab1} - {lab2}"
                mean_diff = float(np.mean(g1) - np.mean(g2))

                # Independent samples t-test (pooled variance)
                t_stat, t_p = _t_test(g1, g2, True)
                results["t_test"] = {
                    "statistic": t_stat,
                    "p_value": t_p,
                    "df": len(g1) + len(g2) - 2,
                    **self._p_significance(t_p),
                    "groups": [lab1, lab2],
                    "contrast": contrast,
                    "mean_difference": mean_diff,
                    "defined": bool(math.isfinite(t_stat)),
                }

                # Welch's t-test (unequal variances)
                welch_t, welch_p = _t_test(g1, g2, False)
                s1, s2 = float(np.var(g1, ddof=1)) / len(g1), float(np.var(g2, ddof=1)) / len(g2)
                welch_den = s1 ** 2 / (len(g1) - 1) + s2 ** 2 / (len(g2) - 1)
                results["welch_t_test"] = {
                    "statistic": welch_t,
                    "p_value": welch_p,
                    "df": float((s1 + s2) ** 2 / welch_den) if welch_den > 0 else nan,
                    **self._p_significance(welch_p),
                    "contrast": contrast,
                }

                # Mann-Whitney U test (non-parametric)
                try:
                    u_stat, u_p = mannwhitney_func(g1, g2)
                    results["mann_whitney"] = {
                        "statistic": float(u_stat),
                        "p_value": float(u_p),
                        **self._p_significance(float(u_p)),
                        "contrast": contrast,
                    }
                except Exception as mw_err:
                    logger.warning("Mann-Whitney test could not be computed: %s", mw_err)

                # Cohen's d effect size (first minus second, same orientation as the t statistic)
                cohens_d = _cohens_d(g1, g2)
                results["cohens_d"] = {
                    "value": float(cohens_d),
                    "interpretation": _cohens_d_label(cohens_d),
                    "contrast": contrast,
                }

            # Multi-group comparisons (2+ conditions)
            group_list = [groups[c] for c in conditions]
            n_valid_total = sum(len(g) for g in group_list)

            # One-way ANOVA
            if all(_is_constant(g) for g in group_list):
                f_stat, anova_p = nan, nan
            else:
                f_stat, anova_p = _finite_pair(anova_func(*group_list))
            results["anova"] = {
                "f_statistic": f_stat,
                "p_value": anova_p,
                **self._p_significance(anova_p),
                "num_groups": len(conditions),
                "df_between": len(conditions) - 1,
                "df_within": n_valid_total - len(conditions),
            }

            # Kruskal-Wallis (non-parametric); exact numpy implementation when scipy is absent
            all_data = np.concatenate(group_list)
            if not _is_constant(all_data):
                try:
                    h_stat, kw_p = _finite_pair(kruskal_func(*group_list))
                    results["kruskal_wallis"] = {
                        "h_statistic": h_stat,
                        "p_value": kw_p,
                        **self._p_significance(kw_p),
                    }
                except Exception as kw_err:
                    logger.warning("Kruskal-Wallis test could not be computed: %s", kw_err)

            # Eta-squared effect size for ANOVA
            grand_mean = np.mean(all_data)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_list)
            ss_total = np.sum((all_data - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            results["eta_squared"] = {
                "value": float(eta_squared),
                "interpretation": self._interpret_eta_squared(eta_squared),
            }

            # Levene's test for homogeneity of variances (undefined when every group is constant:
            # all variances are then zero, i.e. trivially equal)
            if all(_is_constant(g) for g in group_list):
                levene_stat, levene_p = nan, nan
                homogeneous = True
            else:
                levene_stat, levene_p = _finite_pair(levene_func(*group_list))
                homogeneous = bool(levene_p > 0.05) if math.isfinite(levene_p) else True
            results["levene_test"] = {
                "statistic": levene_stat,
                "p_value": levene_p,
                "homogeneous": homogeneous,
            }

            # Normality test (on pooled data, limited to 5000).  Shapiro-Wilk in BOTH modes: the numpy
            # fallback is Royston's algorithm (the one scipy uses), not a skewness/kurtosis rule.
            if 3 <= len(all_data) <= 5000 and not _is_constant(all_data):
                shapiro_stat, shapiro_p = shapiro_func(all_data)
                results["normality_test"] = {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "normal": bool(shapiro_p > 0.05),
                    "test_name": "Shapiro-Wilk",
                }

            # Pairwise comparisons for 3+ groups (first minus second in the canonical order)
            if len(conditions) >= 3:
                pairwise = []
                for i in range(len(conditions)):
                    for j in range(i + 1, len(conditions)):
                        c1, c2 = conditions[i], conditions[j]
                        g1, g2 = groups[c1], groups[c2]
                        t_stat, t_p = _t_test(g1, g2, True)
                        pairwise.append({
                            "comparison": f"{_label(c1)} vs {_label(c2)}",
                            "contrast": f"{_label(c1)} - {_label(c2)}",
                            "t_stat": t_stat,
                            "p_value": t_p,
                            "cohens_d": float(_cohens_d(g1, g2)),
                            "mean_difference": float(np.mean(g1) - np.mean(g2)),
                        })
                # Holm step-down adjustment over this DV's family of comparisons; verdicts use it
                adjusted = _holm_adjust([comp["p_value"] for comp in pairwise])
                for comp, p_adj in zip(pairwise, adjusted):
                    comp["p_adjusted"] = float(p_adj)
                    comp.update(self._p_significance(float(p_adj)))
                results["pairwise_comparisons"] = pairwise
                results["pairwise_adjustment"] = "Holm"

        except Exception as e:
            logger.warning("Statistical test error: %s", e)
            results["error"] = f"Statistical test error: {str(e)}"

        return results

    def _parse_prereg_hypotheses(self, prereg_text: str) -> Dict[str, Any]:
        """Parse pre-registration text to extract hypotheses and analysis plans.

        Returns structured information about:
        - Hypotheses (H1, H2, etc.)
        - Mentioned DVs/outcomes
        - Control variables mentioned
        - Analysis methods mentioned

        Enhanced pattern matching for common pre-registration formats.
        """
        result = {
            "hypotheses": [],
            "mentioned_dvs": [],
            "control_variables": [],
            "analysis_methods": [],
            "interactions_mentioned": False,
        }

        if not prereg_text:
            return result

        prereg_lower = prereg_text.lower()
        prereg_original = prereg_text  # Keep original case for better extraction

        # ========================================
        # ENHANCED HYPOTHESIS EXTRACTION
        # ========================================

        # Pattern 1: Explicit hypothesis labels (H1:, H2:, Hypothesis 1, etc.)
        explicit_hyp_patterns = [
            r'h\s*(\d+)[:\.\s-]+([^.!?\n]+[.!?]?)',  # H1: text, H1. text, H1 - text
            r'hypothesis\s*(\d*)[:\.\s-]+([^.!?\n]+[.!?]?)',  # Hypothesis 1: text
            r'prediction\s*(\d*)[:\.\s-]+([^.!?\n]+[.!?]?)',  # Prediction 1: text
        ]
        for pattern in explicit_hyp_patterns:
            matches = re.findall(pattern, prereg_original, re.IGNORECASE)
            for match in matches:
                # match is (number, text) or just (text,) depending on pattern
                text = match[1] if len(match) > 1 else match[0]
                text = text.strip()
                if len(text) > 10 and text not in result["hypotheses"]:
                    result["hypotheses"].append(text)

        # Pattern 2: Prediction statements (we hypothesize/predict/expect/anticipate/propose)
        prediction_patterns = [
            r'we\s+(?:hypothesize|predict|expect|anticipate|propose)\s+(?:that\s+)?([^.!?\n]+[.!?]?)',
            r'it\s+is\s+(?:hypothesized|predicted|expected|anticipated)\s+(?:that\s+)?([^.!?\n]+[.!?]?)',
            r'our\s+(?:hypothesis|prediction)\s+is\s+(?:that\s+)?([^.!?\n]+[.!?]?)',
        ]
        for pattern in prediction_patterns:
            matches = re.findall(pattern, prereg_original, re.IGNORECASE)
            for m in matches:
                text = m.strip()
                if len(text) > 10 and text not in result["hypotheses"]:
                    result["hypotheses"].append(text)

        # Pattern 3: Effect direction statements (common in pre-regs)
        effect_patterns = [
            r'(?:participants|those|individuals)\s+(?:in|assigned\s+to|who\s+receive)\s+(?:the\s+)?(\w+\s+)?(?:condition|group)\s+will\s+([^.!?\n]+[.!?]?)',
            r'(?:the\s+)?(\w+\s+)?(?:condition|group|treatment)\s+will\s+(?:result\s+in|lead\s+to|show|demonstrate|have)\s+([^.!?\n]+[.!?]?)',
            r'(?:there\s+will\s+be\s+)?(?:a\s+)?(?:significant|positive|negative)?\s*(?:difference|effect|relationship|correlation)\s+(?:between|in)\s+([^.!?\n]+[.!?]?)',
            r'(\w+)\s+will\s+be\s+(?:higher|lower|greater|less|more|stronger|weaker)\s+(?:in|for|among)\s+([^.!?\n]+[.!?]?)',
        ]
        for pattern in effect_patterns:
            # Keep the whole matched sentence. Joining only the capture groups dropped the words between them
            # ("gamified  see a tier badge ...", "Trust the AI condition") and read as garbled text in the report.
            for found in re.finditer(pattern, prereg_original, re.IGNORECASE):
                text = re.sub(r"\s+", " ", found.group(0)).strip()
                if len(text) > 15 and text not in result["hypotheses"]:
                    result["hypotheses"].append(text)

        # Pattern 4: Look for hypotheses in sections/headers
        # Find text after "Hypotheses:" or "Predictions:" headers
        section_patterns = [
            r'hypothes[ie]s?[:\s]*\n+([^\n]+(?:\n[^\n#]+)*)',  # After "Hypotheses:" header
            r'predictions?[:\s]*\n+([^\n]+(?:\n[^\n#]+)*)',  # After "Predictions:" header
        ]
        for pattern in section_patterns:
            matches = re.findall(pattern, prereg_original, re.IGNORECASE | re.MULTILINE)
            for section_text in matches:
                # Split by numbered items or bullet points
                items = re.split(r'\n\s*(?:\d+[\.\)]\s*|\*\s*|-\s*|•\s*)', section_text)
                for item in items:
                    item = item.strip()
                    if len(item) > 15 and item not in result["hypotheses"]:
                        result["hypotheses"].append(item)

        # Pattern 5: Bullet points or numbered lists that look like hypotheses
        list_items = re.findall(r'(?:^|\n)\s*(?:\d+[\.\)]\s*|\*\s*|-\s*|•\s*)([^.!?\n]*(?:will|should|expect|predict|hypothesize|higher|lower|greater|less|more|significant)[^.!?\n]*[.!?]?)', prereg_original, re.IGNORECASE)
        for item in list_items:
            item = item.strip()
            if len(item) > 15 and item not in result["hypotheses"]:
                result["hypotheses"].append(item)

        # Deduplicate and clean hypotheses
        seen = set()
        unique_hypotheses = []
        for hyp in result["hypotheses"]:
            hyp_clean = hyp.strip().lower()
            if hyp_clean not in seen and len(hyp) > 10:
                seen.add(hyp_clean)
                unique_hypotheses.append(hyp.strip())
        result["hypotheses"] = unique_hypotheses[:10]  # Cap at 10 hypotheses

        # Identify control variables mentioned
        control_indicators = [
            r'control(?:ling)?\s+(?:for\s+)?(\w+(?:\s+\w+)?)',
            r'covariat(?:e|es)[:\s]+([^.!?\n]+)',
            r'(?:age|gender|sex|education|income)\s+(?:as\s+)?(?:a\s+)?control',
        ]
        for pattern in control_indicators:
            matches = re.findall(pattern, prereg_lower)
            result["control_variables"].extend([m.strip() for m in matches])

        # Check for specific control variables -- whole words only ("age" is not in "message",
        # "engagement", "percentage" or "average"; "sex" is not in "Essex")
        if re.search(r'\bage\b', prereg_lower):
            if 'age' not in result["control_variables"]:
                result["control_variables"].append('age')
        if re.search(r'\b(?:gender|sex)\b', prereg_lower):
            if 'gender' not in result["control_variables"]:
                result["control_variables"].append('gender')

        # Check for interaction effects
        interaction_terms = ['interaction', 'moderat', 'x ', ' × ', 'cross-over', 'crossover']
        result["interactions_mentioned"] = any(term in prereg_lower for term in interaction_terms)

        # Analysis methods mentioned
        method_keywords = {
            't-test': ['t-test', 't test', 'ttest'],
            'ANOVA': ['anova', 'analysis of variance'],
            'regression': ['regression', 'ols', 'ordinary least squares'],
            'chi-squared': ['chi-squared', 'chi-square', 'χ²'],
            'mediation': ['mediation', 'mediator', 'indirect effect'],
            'moderation': ['moderation', 'moderator', 'interaction effect'],
        }
        for method, keywords in method_keywords.items():
            if any(kw in prereg_lower for kw in keywords):
                result["analysis_methods"].append(method)

        return result

    def _run_regression_analysis(
        self,
        df: pd.DataFrame,
        dv_column: str,
        condition_column: str = "CONDITION",
        include_controls: bool = True,
        prereg_controls: Optional[List[str]] = None,
        condition_order: Optional[List[Any]] = None,
        exclude_columns: Optional[List[str]] = None,
        condition_labels: Optional[Dict[Any, str]] = None,
    ) -> Dict[str, Any]:
        """Run regression analysis with condition as predictor and optional control variables.

        Args:
            df: DataFrame with data
            dv_column: Name of dependent variable column
            condition_column: Name of condition column
            include_controls: Whether to include Age/Gender controls if available
            prereg_controls: Additional control variables from pre-registration
            condition_order: Conditions in the report's one canonical order; the first one is the
                reference level of the dummy coding (default: first appearance in the data)
            exclude_columns: Columns derived from a DV (scale items, composites) -- never covariates
            condition_labels: Display label per raw condition value

        v1.2.9.1: the reference level is reported (``reference_category`` / ``reference_levels``) instead
        of being the alphabetical first level that get_dummies dropped silently; covariates are matched
        by WHOLE WORD on column names (a pre-registered "age" control can no longer pull Message_1..5 or
        Engagement_1..5 into the model, which gave R2 = 1.000 and every p = 1.0) and a column derived
        from a DV is never a covariate; adjusted R2 uses n - k with k INCLUDING the intercept.

        Works with or without scipy (exact numpy p-values).
        """
        results: Dict[str, Any] = {}
        results["scipy_used"] = SCIPY_AVAILABLE
        results["controls_included"] = []

        try:
            conditions = _order_conditions(df[condition_column].dropna().unique().tolist(), condition_order)
            if len(conditions) < 2:
                return {"error": "Need at least 2 conditions"}
            labels = {**_condition_display_labels(conditions), **(condition_labels or {})}

            def _label(cond: Any) -> str:
                return labels.get(cond) or _clean_condition_name(str(cond))

            # Reference category = first condition in the canonical order; every other condition is a dummy
            reference = _label(conditions[0])
            X = pd.DataFrame(index=df.index)
            for cond in conditions[1:]:
                X[_label(cond)] = (df[condition_column] == cond).astype(float)
            results["reference_levels"] = {"Condition": reference}

            banned = {str(c) for c in (exclude_columns or [])} | {str(dv_column), str(condition_column)}
            used_sources: set = set()

            def _unique_name(name: str) -> str:
                while name in X.columns:
                    name = f"{name} (control)"
                return name

            if include_controls:
                # Age control: the demographic Age column (whole-word match), standardized
                age_col = _find_demographic_column(df, ("age",), banned, numeric=True)
                if age_col is not None:
                    age_raw = pd.to_numeric(df[age_col], errors="coerce")
                    age_data = age_raw.fillna(age_raw.mean())
                    X[_unique_name('Age')] = (age_data - age_data.mean()) / (age_data.std() + 1e-10)
                    results["controls_included"].append('Age')
                    used_sources.add(str(age_col))

                # Gender control (dummy coded; the alphabetical first level is the reference)
                gender_col = _find_demographic_column(df, ("gender", "sex"), banned, numeric=False)
                if gender_col is not None:
                    gender_series = df[gender_col]
                    gender_levels = sorted(gender_series.dropna().astype(str).unique().tolist())
                    if len(gender_levels) >= 2:
                        for level in gender_levels[1:]:
                            X[_unique_name(f"Gender_{level}")] = (gender_series.astype(str) == level).astype(float)
                        results["controls_included"].append('Gender')
                        results["reference_levels"]["Gender"] = gender_levels[0]
                    used_sources.add(str(gender_col))

            # Add pre-registered control variables if specified (whole-word column match, never DV-derived)
            if prereg_controls:
                for ctrl_name in prereg_controls:
                    for ctrl_col in _match_control_columns(df, ctrl_name, banned):
                        if str(ctrl_col) in used_sources:
                            continue
                        ctrl_raw = pd.to_numeric(df[ctrl_col], errors="coerce")
                        ctrl_data = ctrl_raw.fillna(ctrl_raw.mean())
                        X[_unique_name(str(ctrl_col))] = (ctrl_data - ctrl_data.mean()) / (ctrl_data.std() + 1e-10)
                        used_sources.add(str(ctrl_col))
                        results["controls_included"].append(str(ctrl_col))

            y_series = pd.to_numeric(df[dv_column], errors="coerce")
            valid_rows = y_series.notna()
            y = y_series[valid_rows].to_numpy(dtype=np.float64)
            X = X.loc[valid_rows].fillna(0.0)

            if len(y) < 3:
                return {"error": "Insufficient data for regression"}

            X_values = X.to_numpy(dtype=np.float64)

            # Add constant
            X_with_const = np.column_stack([np.ones(len(X_values)), X_values])
            n = len(y)
            k = X_with_const.shape[1]  # number of columns INCLUDING the intercept
            df_resid = n - k
            if df_resid <= 0:
                return {"error": "Insufficient residual degrees of freedom for regression"}

            # OLS regression: beta = (X'X)^-1 X'y
            XtX_inv = np.linalg.pinv(X_with_const.T @ X_with_const)
            beta = XtX_inv @ X_with_const.T @ y

            # Predictions and residuals
            y_pred = X_with_const @ beta
            residuals = y - y_pred

            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Standard errors
            mse = ss_res / df_resid
            se = np.sqrt(np.maximum(np.diag(XtX_inv) * mse, 0.0))

            # t-statistics and p-values (NaN when the standard error is zero: perfect fit / no variation)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stats = np.where(se > 0, beta / se, np.nan)
            p_values = np.array([
                _p_t_two_sided(float(t), df_resid) if np.isfinite(t) else float("nan") for t in t_stats
            ])

            results["coefficients"] = {
                "intercept": {
                    "estimate": float(beta[0]),
                    "std_error": float(se[0]),
                    "t_stat": float(t_stats[0]),
                    "p_value": float(p_values[0]),
                }
            }

            # Predictor coefficients (labels are already display names; they are not cleaned again)
            predictor_names = X.columns.tolist()
            results["reference_category"] = reference
            for i, name in enumerate(predictor_names):
                results["coefficients"][str(name)] = {
                    "estimate": float(beta[i + 1]),
                    "std_error": float(se[i + 1]),
                    "t_stat": float(t_stats[i + 1]),
                    "p_value": float(p_values[i + 1]),
                    **self._p_significance(float(p_values[i + 1])),
                }

            results["model_fit"] = {
                "r_squared": float(r_squared),
                # adjusted R2 = 1 - (1 - R2)(n - 1)/(n - p - 1) with p = k - 1 predictors, i.e. n - k
                "adj_r_squared": float(1 - (1 - r_squared) * (n - 1) / (n - k)),
                "n": n,
                "df_residual": df_resid,
                "n_predictors": k - 1,
            }

            # F-test for overall model
            if k > 1:
                f_stat = (r_squared / (k - 1)) / ((1 - r_squared) / df_resid) if r_squared < 1 else float('inf')
                f_p = _p_f_sf(f_stat, k - 1, df_resid)

                results["f_test"] = {
                    "f_statistic": float(f_stat),
                    "p_value": float(f_p),
                    **self._p_significance(float(f_p)),
                }

        except Exception as e:
            logger.warning("Regression analysis failed: %s", e)
            results["error"] = f"Regression error: {str(e)}"

        return results

    def _run_factorial_anova(
        self,
        df: pd.DataFrame,
        dv_column: str,
        factors: List[Dict[str, Any]],
        condition_column: str = "CONDITION",
    ) -> Dict[str, Any]:
        """Run a between-subjects factorial ANOVA (2 or 3 factors, all main effects and interactions).

        Parses the factorial structure from the condition names and computes Type III sums of squares
        (sum-to-zero coding; identical to the classical SS in balanced designs and still valid when
        cells differ in size).  Works with or without scipy (exact numpy p-values).

        v1.2.9.1:
        - Factor levels are read from the condition names by exact whole-token matching, longest level
          first -- never by substring ('AI' inside 'No AI', 'Male' inside 'Female').
        - The residual row is stored under ``residual``; ``error`` is reserved for failures, so the
          report's render guard no longer hides every successful analysis.
        - Up to three factors are analysed; ``design_note`` says which factors were analysed, which were
          not, and which rows were left out.  Conditions that do not form a complete crossing of the
          factors return ``error`` (with ``not_crossed``) and a reason the report prints.
        - Total df is the analytic N - 1.
        """
        results: Dict[str, Any] = {}
        results["scipy_used"] = SCIPY_AVAILABLE

        try:
            # Need at least 2 factors for factorial ANOVA
            if len(factors) < 2:
                return {"error": "Need at least 2 factors for factorial ANOVA", "single_factor": True}

            conditions = df[condition_column].dropna().unique().tolist()
            if len(conditions) < 4:
                return {"error": "Need at least 4 conditions for 2x2 factorial", "conditions": len(conditions)}

            # v1.2.5.0: Track actual sample size BEFORE any filtering
            n_actual = len(df)

            design = _resolve_factorial_design(conditions, factors)
            if not design.get("ok"):
                return {"error": f"Factorial ANOVA not run: {design.get('reason')}", "not_crossed": True,
                        "n_actual": n_actual}

            facs = design["factors"]
            level_map: Dict[Any, Tuple[str, ...]] = design["level_map"]
            names = [str(f.get("name", f"Factor{i + 1}")) for i, f in enumerate(facs)]

            cell_per_row = [level_map.get(c) for c in df[condition_column]]
            y_all = pd.to_numeric(df[dv_column], errors="coerce").to_numpy(dtype=float)
            parsed = np.array([c is not None for c in cell_per_row], dtype=bool)
            has_dv = np.isfinite(y_all)
            valid = parsed & has_dv
            n_analysis = int(valid.sum())
            n_outside = int((df[condition_column].notna().to_numpy() & ~parsed).sum())
            n_missing_dv = int((parsed & ~has_dv).sum())

            if n_analysis < 10:
                return {
                    "error": "Insufficient data after factor parsing",
                    "n_actual": n_actual,
                    "n_parsed": int(parsed.sum()),
                    "n_dv_valid": int(has_dv.sum()),
                }

            y = y_all[valid]
            cells = [cell_per_row[i] for i in np.flatnonzero(valid)]

            # Levels actually present among the analysed rows (declared order), and a complete-crossing check
            from itertools import product
            level_lists = []
            for i, fac in enumerate(facs):
                present = {c[i] for c in cells}
                level_lists.append([str(lv) for lv in fac.get("levels") if str(lv) in present])
            if any(len(lv) < 2 for lv in level_lists):
                return {"error": "Need at least 2 levels per factor. Found "
                                 + " and ".join(str(len(lv)) for lv in level_lists)}
            expected_cells = list(product(*level_lists))
            observed_cells = set(cells)
            if len(observed_cells) != len(expected_cells):
                return {"error": f"Factorial ANOVA not run: the conditions with data do not form a complete "
                                 f"crossing of {' x '.join(names)} ({len(observed_cells)} of {len(expected_cells)} "
                                 f"cells have scores)", "not_crossed": True, "n_actual": n_actual}

            n_levels = [len(lv) for lv in level_lists]
            level_index = []
            for i in range(len(facs)):
                position = {lv: j for j, lv in enumerate(level_lists[i])}
                level_index.append(np.array([position[c[i]] for c in cells], dtype=int))

            n_cells = len(expected_cells)
            df_within = n_analysis - n_cells
            if df_within <= 0:
                return {"error": "Insufficient degrees of freedom for error term"}

            results["factors_analysed"] = [{"name": names[i], "levels": level_lists[i]} for i in range(len(facs))]
            results["factors_not_analysed"] = [str(f.get("name", "factor")) for f in design.get("skipped_factors", [])]
            results["factor1"] = {"name": names[0], "levels": level_lists[0]}
            results["factor2"] = {"name": names[1], "levels": level_lists[1]}
            results["level_map"] = level_map
            results["n_total"] = n_actual
            results["n_analysis"] = n_analysis
            if n_analysis < n_actual:
                results["n_dropped"] = n_actual - n_analysis
                results["n_drop_note"] = (
                    f"{n_actual - n_analysis} rows excluded from ANOVA "
                    f"(missing DV or unparseable condition)"
                )

            # Cell statistics (cell sizes may differ)
            cell_stats: Dict[str, Any] = {}
            cell_sizes = []
            for combo in expected_cells:
                mask = np.ones(n_analysis, dtype=bool)
                for i, lv in enumerate(combo):
                    mask &= level_index[i] == level_lists[i].index(lv)
                vals = y[mask]
                cell_sizes.append(len(vals))
                cell_stats[" × ".join(combo)] = {
                    "n": int(len(vals)),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0,
                }
            results["cell_statistics"] = cell_stats
            balanced = len(set(cell_sizes)) == 1

            # Marginal means (observed, unweighted by design)
            results["marginal_means"] = {
                names[i]: {lv: float(y[level_index[i] == j].mean()) for j, lv in enumerate(level_lists[i])}
                for i in range(len(facs))
            }

            # Type III sums of squares for every term
            fit = _factorial_type3(y, level_index, n_levels)
            ss_within = fit["ss_error"]
            ms_within = ss_within / df_within
            ss_total = float(np.sum((y - y.mean()) ** 2))
            nan = float("nan")

            terms = []
            by_index: Dict[Tuple[int, ...], Dict[str, Any]] = {}
            for t in fit["terms"]:
                idx = t["factors"]
                ss_t, df_t = t["ss"], t["df"]
                ms_t = ss_t / df_t if df_t > 0 else 0.0
                f_t = ms_t / ms_within if ms_within > 0 else nan
                p_t = _p_f_sf(f_t, df_t, df_within) if math.isfinite(f_t) else nan
                eta_t = ss_t / (ss_t + ss_within) if (ss_t + ss_within) > 0 else 0.0
                entry = {
                    "term": " × ".join(names[i] for i in idx),
                    "factors": [names[i] for i in idx],
                    "order": len(idx),
                    "ss": float(ss_t),
                    "df": int(df_t),
                    "ms": float(ms_t),
                    "f_statistic": float(f_t),
                    "p_value": float(p_t),
                    "partial_eta_squared": float(eta_t),
                    **self._p_significance(float(p_t)),
                    "interpretation": self._interpret_eta_squared(eta_t),
                }
                terms.append(entry)
                by_index[tuple(idx)] = entry
            results["terms"] = terms

            # Legacy keys (first two factors) kept for existing consumers
            _plain = ("term", "factors", "order")
            results["main_effect_1"] = {"factor": names[0], **{k: v for k, v in by_index[(0,)].items() if k not in _plain}}
            results["main_effect_2"] = {"factor": names[1], **{k: v for k, v in by_index[(1,)].items() if k not in _plain}}
            results["interaction"] = {"factors": f"{names[0]} × {names[1]}",
                                      **{k: v for k, v in by_index[(0, 1)].items() if k not in _plain}}

            results["residual"] = {
                "ss": float(ss_within),
                "df": int(df_within),
                "ms": float(ms_within),
            }
            results["total"] = {
                "ss": ss_total,
                "df": n_analysis - 1,
            }

            # One-line disclosure of exactly what was analysed
            note = (f"Factors analysed: {' × '.join(f'{names[i]} ({n_levels[i]} levels)' for i in range(len(facs)))}"
                    f" on N = {n_analysis}")
            if results["factors_not_analysed"]:
                note += (f"; not analysed (their levels are pooled): {', '.join(results['factors_not_analysed'])}")
            if n_outside:
                outside = sorted({_clean_condition_name(str(c)) for c in design.get("unparsed", [])})
                note += (f"; {n_outside} participants in conditions outside the factorial crossing "
                         f"({', '.join(outside[:5])}{', ...' if len(outside) > 5 else ''}) are excluded")
            if n_missing_dv:
                note += f"; {n_missing_dv} with a missing score are excluded"
            if not balanced:
                note += (f"; cell sizes differ (n = {min(cell_sizes)}-{max(cell_sizes)}), "
                         f"so Type III sums of squares are used")
            results["design_note"] = note + "."

        except Exception as e:
            logger.warning("Factorial ANOVA failed: %s", e)
            results = {"scipy_used": SCIPY_AVAILABLE, "error": f"Factorial ANOVA error: {str(e)}"}

        return results

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d (the one label helper shared by every table and sentence)."""
        return _cohens_d_label(d)

    def _interpret_eta_squared(self, eta2: float) -> str:
        """Interpret eta-squared effect size with detailed thresholds."""
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.02:
            return "very small"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        elif eta2 < 0.26:
            return "large"
        else:
            return "very large"

    def _interpret_omega_squared(self, omega2: float) -> str:
        """Interpret omega-squared effect size (less biased than eta-squared)."""
        if omega2 < 0.01:
            return "negligible"
        elif omega2 < 0.06:
            return "small"
        elif omega2 < 0.14:
            return "medium"
        else:
            return "large"

    def _interpret_r_squared(self, r2: float) -> str:
        """Interpret R-squared (coefficient of determination)."""
        if r2 < 0.02:
            return "negligible"
        elif r2 < 0.13:
            return "small"
        elif r2 < 0.26:
            return "medium"
        else:
            return "large"

    def _interpret_correlation(self, r: float) -> str:
        """Interpret Pearson correlation coefficient."""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "weak"
        elif abs_r < 0.5:
            return "moderate"
        elif abs_r < 0.7:
            return "strong"
        else:
            return "very strong"

    def _get_practical_significance(self, effect_size: float, effect_type: str) -> str:
        """Generate practical significance statement based on effect size."""
        if effect_type == "cohens_d":
            interpretation = self._interpret_cohens_d(effect_size)
        elif effect_type == "eta_squared":
            interpretation = self._interpret_eta_squared(effect_size)
        elif effect_type == "r":
            interpretation = self._interpret_correlation(effect_size)
        else:
            interpretation = "unknown"

        significance_statements = {
            "negligible": "This effect is too small to have practical importance.",
            "very small": "This effect is minimal and unlikely to be noticeable in practice.",
            "small": "This effect is small but may be meaningful in some contexts.",
            "weak": "This relationship is weak and has limited practical value.",
            "moderate": "This effect is moderate and likely meaningful in practice.",
            "medium": "This effect is of medium magnitude and practically meaningful.",
            "strong": "This effect is strong and has clear practical implications.",
            "large": "This effect is large and has substantial practical importance.",
            "very strong": "This effect is very strong with major practical implications.",
            "very large": "This effect is very large and highly practically significant.",
        }
        return significance_statements.get(interpretation, "")

    def _generate_chart_interpretation(
        self,
        chart_data: Dict[str, Tuple[float, float]],
        stats_results: Dict[str, Any],
        scale_name: str
    ) -> str:
        """
        Generate a 1-2 sentence interpretation of the chart results.

        Args:
            chart_data: Dict mapping condition names to (mean, std_error) tuples
            stats_results: Statistical test results
            scale_name: Name of the scale being analyzed

        Returns:
            HTML string with interpretation
        """
        if not chart_data:
            return ""

        # v1.2.9.1: describe only the conditions the tests actually compared (groups with fewer than
        # 2 scores are left out of the tests, so they must not be "highest" / "lowest" or counted)
        _compared = stats_results.get("conditions") if isinstance(stats_results, dict) else None
        if _compared:
            chart_data = {c: v for c, v in chart_data.items() if c in set(_compared)} or chart_data

        conditions = list(chart_data.keys())
        means = {c: chart_data[c][0] for c in conditions}
        n_conditions = len(conditions)

        interpretation_parts = []

        # Find highest and lowest scoring conditions
        sorted_conds = sorted(means.items(), key=lambda x: x[1], reverse=True)
        highest = sorted_conds[0]
        lowest = sorted_conds[-1]
        mean_diff = highest[1] - lowest[1]

        # Get significance and effect size info
        p_value = None
        effect_size = None
        effect_interpretation = None
        is_significant = False
        is_marginal = False

        if "t_test" in stats_results:
            p_value = stats_results["t_test"]["p_value"]
            is_significant = stats_results["t_test"]["significant"]
            is_marginal = stats_results["t_test"].get("marginally_significant", False)
        elif "anova" in stats_results:
            p_value = stats_results["anova"]["p_value"]
            is_significant = stats_results["anova"]["significant"]
            is_marginal = stats_results["anova"].get("marginally_significant", False)

        if "cohens_d" in stats_results:
            effect_size = stats_results["cohens_d"]["value"]
            effect_interpretation = stats_results["cohens_d"]["interpretation"]
        elif "eta_squared" in stats_results:
            effect_size = stats_results["eta_squared"]["value"]
            effect_interpretation = stats_results["eta_squared"]["interpretation"]

        # Generate interpretation
        if n_conditions == 2:
            # Two-group comparison
            if is_significant and p_value is not None:
                interpretation_parts.append(
                    f"<strong>{highest[0]}</strong> scored significantly higher (M = {highest[1]:.2f}) than "
                    f"<strong>{lowest[0]}</strong> (M = {lowest[1]:.2f}), with a difference of {mean_diff:.2f} points "
                    f"({_p_eq_html(p_value)})."
                )
                if effect_interpretation:
                    interpretation_parts.append(
                        f" This represents a <strong>{effect_interpretation} effect</strong>"
                        f"{f' (d = {effect_size:.2f})' if effect_size else ''}."
                    )
            elif is_marginal and p_value is not None:
                interpretation_parts.append(
                    f"A marginally significant difference was found between <strong>{highest[0]}</strong> "
                    f"(M = {highest[1]:.2f}) and <strong>{lowest[0]}</strong> (M = {lowest[1]:.2f}), "
                    f"with a difference of {mean_diff:.2f} points ({_p_eq_html(p_value)})."
                )
                if effect_interpretation and effect_size:
                    interpretation_parts.append(
                        f" The effect size was {effect_interpretation} (d = {effect_size:.2f})."
                    )
            else:
                interpretation_parts.append(
                    f"No statistically significant difference was found between <strong>{highest[0]}</strong> "
                    f"(M = {highest[1]:.2f}) and <strong>{lowest[0]}</strong> (M = {lowest[1]:.2f})"
                    f"{f' ({_p_eq_html(p_value)})' if p_value is not None else ''}."
                )
                if effect_interpretation and effect_size:
                    interpretation_parts.append(
                        f" The effect size was {effect_interpretation} (d = {effect_size:.2f})."
                    )
        else:
            # Multi-group comparison
            if is_significant and p_value is not None:
                interpretation_parts.append(
                    f"Significant differences were found across conditions ({_p_eq_html(p_value)}). "
                    f"<strong>{highest[0]}</strong> showed the highest mean (M = {highest[1]:.2f}), "
                    f"while <strong>{lowest[0]}</strong> showed the lowest (M = {lowest[1]:.2f})."
                )
                if effect_interpretation:
                    interpretation_parts.append(
                        f" The overall effect was <strong>{effect_interpretation}</strong>"
                        f"{f' (η² = {effect_size:.3f})' if effect_size else ''}."
                    )
            elif is_marginal and p_value is not None:
                interpretation_parts.append(
                    f"Marginally significant differences were found across the {n_conditions} conditions "
                    f"({_p_eq_html(p_value)}). "
                    f"<strong>{highest[0]}</strong> showed the highest mean (M = {highest[1]:.2f}), "
                    f"while <strong>{lowest[0]}</strong> showed the lowest (M = {lowest[1]:.2f})."
                )
                if effect_interpretation:
                    interpretation_parts.append(
                        f" The overall effect was <strong>{effect_interpretation}</strong>"
                        f"{f' (η² = {effect_size:.3f})' if effect_size else ''}."
                    )
            else:
                interpretation_parts.append(
                    f"No significant differences were found across the {n_conditions} conditions"
                    f"{f' ({_p_eq_html(p_value)})' if p_value is not None else ''}. "
                    f"Means ranged from {lowest[1]:.2f} to {highest[1]:.2f}."
                )

        return "".join(interpretation_parts)

    def _generate_executive_summary(
        self,
        all_scale_results: List[Dict[str, Any]],
        prereg_text: Optional[str],
        n_total: int,
        conditions: List[str]
    ) -> str:
        """
        Generate detailed executive summary with key takeaways and hypothesis evaluation.

        Args:
            all_scale_results: List of dicts containing scale analysis results
            prereg_text: Pre-registration text if available
            n_total: Total sample size
            conditions: List of condition names

        Returns:
            HTML string with executive summary
        """
        # Measures whose composite could not be tested (constant-sum or rank-order items, or no variation at all)
        # are listed below, not counted as null results.
        untested = [str(r.get("scale_name", "Unknown Scale")) for r in all_scale_results if r.get("not_tested")]
        all_scale_results = [r for r in all_scale_results if not r.get("not_tested")]

        html = ["<h2>2. Executive Summary</h2>"]
        html.append("<div class='section-block' style='background:#f0f7ff;border-left:4px solid #3498db;'>")

        # Collect detailed findings
        sig_findings = []
        marginal_findings = []
        nonsig_findings = []
        largest_effect = None
        largest_effect_size = 0
        all_effects = []

        for result in all_scale_results:
            scale_name = result.get("scale_name", "Unknown Scale")
            stats = result.get("stats_results", {})
            chart_data = result.get("chart_data", {})
            is_sig = False
            is_marginal = False
            effect_val = 0
            effect_type = None
            p_val = None

            # Check significance and get p-value
            if "t_test" in stats:
                p_val = stats["t_test"]["p_value"]
                is_sig = stats["t_test"]["significant"]
                is_marginal = stats["t_test"].get("marginally_significant", False)
            elif "anova" in stats:
                p_val = stats["anova"]["p_value"]
                is_sig = stats["anova"]["significant"]
                is_marginal = stats["anova"].get("marginally_significant", False)
            if p_val is not None and not _is_finite_number(p_val):
                # an undefined test (for example zero variance in every group) is no finding either way
                p_val, is_sig, is_marginal = None, False, False

            # Get effect size
            if "cohens_d" in stats:
                effect_val = abs(stats["cohens_d"]["value"])
                effect_type = "d"
            elif "eta_squared" in stats:
                effect_val = stats["eta_squared"]["value"]
                effect_type = "η²"
            if effect_type is not None and not _is_finite_number(effect_val):
                effect_val, effect_type = 0, None

            # Get means for condition comparison
            if chart_data:
                sorted_conds = sorted(chart_data.items(), key=lambda x: x[1][0], reverse=True)
                highest_cond = sorted_conds[0][0] if sorted_conds else None
                highest_mean = sorted_conds[0][1][0] if sorted_conds else None
                lowest_cond = sorted_conds[-1][0] if sorted_conds else None
                lowest_mean = sorted_conds[-1][1][0] if sorted_conds else None
            else:
                highest_cond = lowest_cond = highest_mean = lowest_mean = None

            finding_info = {
                "scale": scale_name,
                "effect_size": effect_val,
                "effect_type": effect_type,
                "p_value": p_val,
                "stats": stats,
                "highest_cond": highest_cond,
                "highest_mean": highest_mean,
                "lowest_cond": lowest_cond,
                "lowest_mean": lowest_mean,
                "significant": is_sig,
                "marginally_significant": is_marginal,
            }

            if is_sig:
                sig_findings.append(finding_info)
                if effect_val > largest_effect_size:
                    largest_effect_size = effect_val
                    largest_effect = finding_info
            elif is_marginal:
                marginal_findings.append(finding_info)
            else:
                nonsig_findings.append(finding_info)

            all_effects.append(finding_info)

        # Generate summary text
        n_scales = len(all_scale_results)
        n_sig = len(sig_findings)
        n_conditions = len(conditions)

        html.append("<p style='font-size:14px;line-height:1.8;margin:0;'>")

        # Opening - Study Overview
        html.append("<strong style='color:#2c3e50;font-size:15px;'>Study Overview:</strong><br>")
        html.append(
            f"This simulation generated data for <strong>{n_total} participants</strong> randomly assigned to "
            f"<strong>{n_conditions} experimental condition{'s' if n_conditions > 1 else ''}</strong>: "
            f"{', '.join(str(c) for c in conditions)}. "
        )
        html.append(f"The analysis examined {n_scales} dependent variable{'s' if n_scales > 1 else ''}.")
        if untested:
            html.append(
                f" {len(untested)} further measure{'s' if len(untested) > 1 else ''} could not be tested as a composite "
                f"(a constant-sum or rank-order question, or no variation at all): {_html_lib.escape(', '.join(untested[:8]))}"
                f"{' and more' if len(untested) > 8 else ''}. See the Statistical Analysis section for the item-by-item comparison where one applies."
            )
        if not all_scale_results:
            html.append("</p></div>")
            return "\n".join(html)

        # Main Findings
        html.append("<br><br><strong style='color:#2c3e50;font-size:15px;'>Key Results:</strong><br>")

        if n_sig > 0:
            html.append(
                f"<span style='color:#27ae60;'>✓</span> <strong>{n_sig} of {n_scales} dependent variable{'s' if n_sig > 1 else ''} showed statistically significant differences</strong> between conditions:<br>"
            )

            for i, finding in enumerate(sig_findings):
                # v1.2.9.1: the same verbal label as every other table (d: _cohens_d_label, eta-squared: its own scale)
                effect_desc = (_cohens_d_label(finding["effect_size"]) if finding["effect_type"] == "d"
                               else self._interpret_eta_squared(finding["effect_size"]))

                html.append(f"&nbsp;&nbsp;• <strong>{finding['scale']}</strong>: ")
                if finding['highest_cond'] and finding['lowest_cond']:
                    html.append(
                        f"{finding['highest_cond']} (M = {finding['highest_mean']:.2f}) > {finding['lowest_cond']} (M = {finding['lowest_mean']:.2f}), "
                    )
                _effect_txt = f", {effect_desc} effect ({finding['effect_type']} = {_fmt_stat(finding['effect_size'], 2)})" if finding['effect_type'] else ""
                html.append(f"{_report_p_text(finding['p_value'], html=True)}{_effect_txt}<br>")

            if largest_effect:
                html.append(
                    f"<br>The <strong>strongest effect</strong> was observed for <strong>{largest_effect['scale']}</strong>. "
                )

        # Show marginally significant findings
        if marginal_findings:
            html.append(
                f"<br><span style='color:#f39c12;'>†</span> <strong>{len(marginal_findings)} DV{'s' if len(marginal_findings) > 1 else ''} showed marginally significant differences</strong> (p &lt; .10):<br>"
            )
            for finding in marginal_findings:
                html.append(f"&nbsp;&nbsp;• <strong>{finding['scale']}</strong>: ")
                if finding['highest_cond'] and finding['lowest_cond']:
                    html.append(
                        f"{finding['highest_cond']} (M = {finding['highest_mean']:.2f}) > {finding['lowest_cond']} (M = {finding['lowest_mean']:.2f}), "
                    )
                html.append(f"{_report_p_text(finding['p_value'], html=True)}<br>")

        if n_sig == 0 and not marginal_findings:
            html.append(
                f"<span style='color:#e74c3c;'>✗</span> <strong>No statistically significant differences</strong> were found between conditions "
                f"on any of the {n_scales} dependent variable{'s' if n_scales > 1 else ''}. "
            )
            # Show the closest to significance
            if all_effects:
                closest = min(all_effects, key=lambda x: x['p_value'] if (x['p_value'] is not None and x['p_value'] == x['p_value']) else 1.0)
                if closest['p_value'] is not None and closest['p_value'] == closest['p_value']:
                    html.append(f"The closest to significance was <strong>{closest['scale']}</strong> ({_report_p_text(closest['p_value'], html=True)}).")
        elif n_sig == 0:
            # Had marginal but no sig findings
            html.append(
                f"<span style='color:#e74c3c;'>✗</span> No differences reached conventional significance (p &lt; .05). "
            )

        # Pre-registration hypotheses. The match below uses words in the measure names only: it never tests the
        # predicted direction, so it can point to a related result but must never call a hypothesis "supported".
        if prereg_text:
            prereg_info = self._parse_prereg_hypotheses(prereg_text)
            hypotheses = [h for h in (_hypothesis_text(x) for x in prereg_info.get("hypotheses", [])) if h]

            html.append("<br><br><strong style='color:#2c3e50;font-size:15px;'>Pre-Registration Hypotheses:</strong><br>")

            if hypotheses:
                html.append(
                    f"{len(hypotheses)} hypothesis statement(s) were picked out of the pre-registration text automatically (the wording may be "
                    "incomplete). Each one is matched to the dependent variables by words in the variable name only, so the lines below point "
                    "to related results. They do not test the predicted direction and do not show that a hypothesis is supported:<br>"
                )
                for h_text in hypotheses:
                    h_lower = h_text.lower()
                    related = []
                    for finding in sig_findings:
                        words = [w for w in re.findall(r"[a-z0-9]+", str(finding["scale"]).lower()) if len(w) > 3]
                        if any(w in h_lower for w in words):
                            related.append(str(finding["scale"]))
                    shown = _html_lib.escape(h_text[:80] + ("..." if len(h_text) > 80 else ""))
                    if related:
                        names = _html_lib.escape(", ".join(related[:3]))
                        html.append(f"&nbsp;&nbsp;• <em>\"{shown}\"</em> — a significant result on a related measure was found ({names}); the direction was not checked<br>")
                    else:
                        html.append(f"&nbsp;&nbsp;• <em>\"{shown}\"</em> — no significant related result<br>")
            else:
                html.append("No specific hypotheses were extracted from the pre-registration document. Review the document manually to compare predictions with results.")

        # Practical Implications
        html.append("<br><strong style='color:#2c3e50;font-size:15px;'>Interpretation:</strong><br>")
        if n_sig > 0:
            html.append(
                f"These results suggest that the experimental manipulation had a measurable effect on participant responses. "
                f"Students should examine the pattern of means to understand the direction of effects and consider whether "
                f"these findings align with theoretical predictions."
            )
        else:
            html.append(
                f"The lack of significant findings could indicate that: (1) the manipulation was not strong enough, "
                f"(2) the sample size was insufficient to detect small effects, or (3) there is genuinely no effect of the "
                f"experimental conditions on the measured outcomes. Students should consider these possibilities in their discussion."
            )

        # Note about simulation
        html.append("<br><br><em style='color:#7f8c8d;font-size:12px;'>")
        html.append("Note: These are simulated results generated for pedagogical purposes. The patterns reflect the simulation parameters chosen, ")
        html.append("and actual experimental results will depend on real participant responses.</em>")
        html.append("</p></div>")

        return "\n".join(html)

    def _generate_stat_test_interpretation(
        self,
        test_type: str,
        stats_results: Dict[str, Any],
        chart_data: Dict[str, Tuple[float, float]],
        scale_name: str
    ) -> str:
        """Generate plain-language interpretation for a statistical test."""
        # v1.2.9.1: describe only the conditions the tests actually compared (groups with fewer than
        # 2 scores are left out of the tests and must not be counted or named as highest / lowest)
        _compared = stats_results.get("conditions") if isinstance(stats_results, dict) else None
        if _compared:
            chart_data = {c: v for c, v in chart_data.items() if c in set(_compared)} or chart_data
        conditions = list(chart_data.keys())
        means = {c: chart_data[c][0] for c in conditions}
        sorted_conds = sorted(means.items(), key=lambda x: x[1], reverse=True)
        highest = sorted_conds[0]
        lowest = sorted_conds[-1]

        if test_type == "t_test" and "t_test" in stats_results:
            t = stats_results["t_test"]
            if not t.get("defined", True):
                return "The t-test cannot be computed because the scores do not vary within the conditions (zero variance), so no significance claim is made."
            if t["significant"]:
                return f"The t-test indicates a statistically significant difference between conditions. Participants in the <strong>{highest[0]}</strong> condition scored higher (M = {highest[1]:.2f}) than those in <strong>{lowest[0]}</strong> (M = {lowest[1]:.2f})."
            elif t.get("marginally_significant"):
                return f"The t-test found a marginally significant difference (p &lt; .10) between conditions. <strong>{highest[0]}</strong> scored higher (M = {highest[1]:.2f}) than <strong>{lowest[0]}</strong> (M = {lowest[1]:.2f}), though this did not reach conventional significance (p &lt; .05)."
            else:
                return f"The t-test did not find a statistically significant difference between the two conditions, suggesting that {scale_name} scores were similar regardless of experimental condition."

        elif test_type == "anova" and "anova" in stats_results:
            a = stats_results["anova"]
            if not math.isfinite(a.get("f_statistic", float("nan"))):
                return "The ANOVA cannot be computed because the scores do not vary within the conditions (zero variance), so no significance claim is made."
            if a["significant"]:
                return f"The ANOVA reveals significant variation in {scale_name} across conditions. <strong>{highest[0]}</strong> showed the highest scores while <strong>{lowest[0]}</strong> showed the lowest. Post-hoc comparisons (below) identify which specific pairs differ."
            elif a.get("marginally_significant"):
                return f"The ANOVA found marginally significant variation (p &lt; .10) in {scale_name} across the {len(conditions)} conditions compared. <strong>{highest[0]}</strong> showed the highest scores while <strong>{lowest[0]}</strong> showed the lowest. This trend may warrant further investigation with a larger sample."
            else:
                return f"The ANOVA did not detect significant differences in {scale_name} across the {len(conditions)} conditions compared, suggesting the experimental manipulation may not have affected this outcome."

        elif test_type == "effect_size":
            if "cohens_d" in stats_results:
                d = stats_results["cohens_d"]
                label = _cohens_d_label(d["value"])
                value = _fmt_stat(d["value"], 2)
                sentences = {
                    "undefined": "The effect size is undefined because the scores do not vary within the conditions.",
                    "negligible": f"The effect size is <strong>negligible</strong> (d = {value}), suggesting minimal practical difference between conditions.",
                    "very small": f"The effect size is <strong>very small</strong> (d = {value}), a difference too small to matter in most practical settings.",
                    "small": f"The effect size is <strong>small</strong> (d = {value}), indicating a modest difference that may have limited practical significance.",
                    "medium": f"The effect size is <strong>medium</strong> (d = {value}), suggesting a moderate and potentially meaningful difference.",
                    "large": f"The effect size is <strong>large</strong> (d = {value}), indicating a substantial and practically meaningful difference between conditions.",
                    "very large": f"The effect size is <strong>very large</strong> (d = {value}), indicating a very substantial difference between conditions.",
                }
                return sentences.get(label, "")
            elif "eta_squared" in stats_results:
                e = stats_results["eta_squared"]
                return f"The effect size (η² = {e['value']:.3f}) indicates that {e['value']*100:.1f}% of variance in {scale_name} is explained by condition assignment ({e['interpretation']} effect)."

        elif test_type == "regression":
            return f"The regression analysis examines condition effects while controlling for demographic variables, providing a more precise estimate of the experimental effect."

        return ""

    def _get_prereg_requested_analyses(self, prereg_text: Optional[str]) -> Dict[str, bool]:
        """Parse pre-registration to determine which analyses were requested."""
        requested = {
            "t_test": False,
            "anova": False,
            "regression": False,
            "factorial": False,
            "chi_squared": False,
            "mann_whitney": False,
            "correlation": False
        }

        if not prereg_text:
            return requested

        text_lower = prereg_text.lower()

        # Check for specific analysis mentions
        if any(term in text_lower for term in ["t-test", "t test", "independent samples", "two-sample"]):
            requested["t_test"] = True
        if any(term in text_lower for term in ["anova", "analysis of variance", "f-test", "between-subjects"]):
            requested["anova"] = True
        if any(term in text_lower for term in ["regression", "linear model", "glm", "control variable", "covariate"]):
            requested["regression"] = True
        if any(term in text_lower for term in ["factorial", "interaction", "2x2", "2x3", "3x3", "two-way", "main effect"]):
            requested["factorial"] = True
        if any(term in text_lower for term in ["chi-square", "chi square", "χ²", "contingency"]):
            requested["chi_squared"] = True
        if any(term in text_lower for term in ["mann-whitney", "wilcoxon", "non-parametric", "nonparametric"]):
            requested["mann_whitney"] = True
        if any(term in text_lower for term in ["correlation", "pearson", "spearman"]):
            requested["correlation"] = True

        return requested

    def _item_comparisons(
        self,
        df: pd.DataFrame,
        item_cols: List[str],
        conditions: List[Any],
    ) -> List[Dict[str, Any]]:
        """Compare each item column across conditions (the analysis for a composite that cannot vary).

        Returns one dict per item: ``item``, ``cells`` (clean condition name -> (mean, sd, n)), ``varies``
        and ``test`` (the finite results of ``_run_statistical_tests``; empty when no test is defined).
        """
        rows: List[Dict[str, Any]] = []
        if "CONDITION" not in df.columns:
            return rows
        for col in item_cols:
            frame = pd.DataFrame({
                "CONDITION": df["CONDITION"].to_numpy(),
                "_item": pd.to_numeric(df[col], errors="coerce").to_numpy(),
            })
            cells: Dict[str, Tuple[float, float, int]] = {}
            for cond in conditions:
                group = frame.loc[frame["CONDITION"] == cond, "_item"].dropna()
                if len(group) > 0:
                    cells[_clean_condition_name(cond)] = (
                        float(group.mean()), float(group.std()) if len(group) > 1 else float("nan"), int(len(group)))
            varies = not _composite_has_no_variation(frame["_item"], frame["CONDITION"])
            test: Dict[str, Any] = {}
            if varies and len(cells) >= 2:
                test = _finite_results_only(self._run_statistical_tests(
                    frame, "_item", "CONDITION", condition_order=list(conditions)))  # same orientation as the composite
            rows.append({"item": str(col), "cells": cells, "varies": varies, "test": test})
        return rows

    @staticmethod
    def _item_test_cells(row: Dict[str, Any], html: bool = False) -> Tuple[str, str, str, str]:
        """(test statistic, p-value, effect size, significance class) for one ``_item_comparisons`` row."""
        res = row.get("test") or {}
        if len(row.get("cells", {})) == 2 and "t_test" in res:
            t = res["t_test"]
            d = res.get("cohens_d", {}).get("value")
            effect = f"d = {_fnum(d, '.2f')}" if _is_finite_number(d) else "n/a"
            return (f"t = {_fnum(t.get('statistic'), '.2f')}", _report_p_cell(t.get("p_value"), html=html), effect,
                    "sig" if t.get("significant") else ("marginal" if t.get("marginally_significant") else "nonsig"))
        if "anova" in res:
            a = res["anova"]
            eta = res.get("eta_squared", {}).get("value")
            effect = f"η² = {_fnum(eta, '.3f')}" if _is_finite_number(eta) else "n/a"
            return (f"F = {_fnum(a.get('f_statistic'), '.2f')}", _report_p_cell(a.get("p_value"), html=html), effect,
                    "sig" if a.get("significant") else ("marginal" if a.get("marginally_significant") else "nonsig"))
        return ("no variation within conditions" if not row.get("varies", True) else "not computed", "n/a", "n/a", "nonsig")

    def _item_comparison_html(self, rows: List[Dict[str, Any]]) -> List[str]:
        """HTML table for ``_item_comparisons`` rows (all text escaped)."""
        esc = lambda value: _html_lib.escape(str(value), quote=True)  # noqa: E731
        out = ["<table><tr><th>Item</th><th>Mean (SD) by condition</th><th>Test</th><th>p</th><th>Effect size</th></tr>"]
        for row in rows:
            cells = "<br>".join(f"{esc(c)}: {_fnum(m, '.2f')} ({_fnum(sd, '.2f')})" for c, (m, sd, _n) in row["cells"].items())
            test, p_cell, effect, cls = self._item_test_cells(row, html=True)
            out.append(f"<tr><td>{esc(row['item'])}</td><td>{cells}</td><td>{test}</td><td class='{cls}'>{p_cell}</td><td>{effect}</td></tr>")
        out.append("</table>")
        return out

    def _item_comparison_markdown(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Markdown table for ``_item_comparisons`` rows."""
        cell = lambda value: str(value).replace("|", "\\|")  # noqa: E731
        out = ["| Item | Mean (SD) by condition | Test | p | Effect size |", "|------|------------------------|------|---|-------------|"]
        for row in rows:
            cells = "; ".join(f"{cell(c)}: {_fnum(m, '.2f')} ({_fnum(sd, '.2f')})" for c, (m, sd, _n) in row["cells"].items())
            test, p_cell, effect, _cls = self._item_test_cells(row)
            out.append(f"| {cell(row['item'])} | {cells} | {test} | {p_cell} | {effect} |")
        return out

    def _no_variation_block(
        self,
        scale: Dict[str, Any],
        scale_cols: List[str],
        df: pd.DataFrame,
        composite: Any,
        conditions: List[Any],
        html: bool,
    ) -> List[str]:
        """Lines/HTML for a DV whose composite cannot be tested (constant-sum, rank-order, zero variance).

        Says why in one sentence and, when the measure has several items, compares each item across
        conditions instead. Never computes a test on the constant composite.
        """
        mean = float(np.nanmean(pd.to_numeric(pd.Series(composite), errors="coerce"))) if len(df) else float("nan")
        note = _no_variation_note(scale, len(scale_cols), mean)
        rows: List[Dict[str, Any]] = []
        if len(scale_cols) >= 2:
            rows = [r for r in self._item_comparisons(df, scale_cols, conditions) if r["cells"]]
        out: List[str] = []
        if html:
            out.append(f"<div class='warning-box'>{_html_lib.escape(note)}</div>")
            if rows:
                out.extend(self._item_comparison_html(rows))
                out.append(f"<p style='font-size:0.85em;color:#64748b;'>p-values are not corrected for the {len(rows)} comparisons.</p>")
            return out
        out.append(f"**Note:** {note}")
        out.append("")
        if rows:
            out.extend(self._item_comparison_markdown(rows))
            out.append("")
            out.append(f"*p-values are not corrected for the {len(rows)} comparisons.*")
        return out

    def _create_bar_chart(
        self,
        data: Dict[str, Tuple[float, float]],
        title: str,
        ylabel: str,
        effect_size: Optional[float] = None,
        p_value: Optional[float] = None,
    ) -> Optional[str]:
        """Create an enhanced bar chart with error bars, annotations, and styling."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        if not data:
            return None

        try:
            # Reset matplotlib state and use default style as fallback
            plt.close('all')
            try:
                plt.style.use('seaborn-v0_8-whitegrid')
            except Exception:
                try:
                    plt.style.use('seaborn-whitegrid')
                except Exception:
                    plt.style.use('default')

            fig, ax = plt.subplots(figsize=(10, 6))

            conditions = list(data.keys())
            means = [data[c][0] for c in conditions]
            errors = [data[c][1] for c in conditions]

            # Modern color palette (colorblind-friendly)
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']
            bar_colors = colors[:len(conditions)]

            # Create bars with gradient effect
            bars = ax.bar(conditions, means, yerr=errors, capsize=8,
                         color=bar_colors, edgecolor='white', linewidth=2,
                         alpha=0.85, error_kw={'linewidth': 2, 'capthick': 2, 'ecolor': '#2c3e50'})

            # Style improvements
            ax.set_ylabel(ylabel, fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
            ax.tick_params(axis='x', rotation=30, labelsize=11)
            ax.tick_params(axis='y', labelsize=10)

            # Add value labels on bars with better formatting
            for bar, mean, error in zip(bars, means, errors):
                height = bar.get_height()
                ax.annotate(f'{mean:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height + error),
                           xytext=(0, 8), textcoords="offset points",
                           ha='center', va='bottom', fontsize=11, fontweight='bold',
                           color='#2c3e50',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

            # Add significance and effect size annotation if provided
            annotation_text = []
            if p_value is not None:
                sig_symbol = self._p_significance(p_value)["sig_label"]
                annotation_text.append(f"{_p_eq(p_value)} {sig_symbol}")
            if effect_size is not None:
                annotation_text.append(f"d = {effect_size:.2f}")

            if annotation_text:
                ax.text(0.98, 0.98, "\n".join(annotation_text),
                       transform=ax.transAxes, fontsize=11,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.9, edgecolor='#bdc3c7'))

            # Remove top and right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')

            # Add subtle gridlines
            ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#ecf0f1')
            ax.set_axisbelow(True)

            plt.tight_layout()

            # Save to base64 with higher DPI
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception as e:
            # Fallback: try a simpler chart
            try:
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, 5))
                conditions = list(data.keys())
                means = [data[c][0] for c in conditions]
                ax.bar(conditions, means, color='steelblue', alpha=0.7)
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close(fig)
                return img_base64
            except Exception:
                return None

    def _create_distribution_plot(
        self,
        df: pd.DataFrame,
        column: str,
        condition_column: str = "CONDITION",
        title: str = "Distribution by Condition",
    ) -> Optional[str]:
        """Create an enhanced violin/box plot with individual data points."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Clean condition names for display
            df_plot = df.copy()
            df_plot['_clean_condition'] = df_plot[condition_column].map(_condition_display_labels(df_plot[condition_column].dropna().unique().tolist()))
            conditions = df_plot['_clean_condition'].unique().tolist()

            # Modern color palette
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']

            positions = range(len(conditions))
            box_data = [df_plot[df_plot['_clean_condition'] == c][column].dropna().values for c in conditions]

            # Create violin plots for density visualization
            parts = ax.violinplot(box_data, positions=positions, showmeans=False,
                                  showmedians=False, showextrema=False)

            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_edgecolor('white')
                pc.set_alpha(0.3)

            # Overlay box plots
            bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.3,
                           showfliers=False)

            for i, (patch, median) in enumerate(zip(bp['boxes'], bp['medians'])):
                patch.set_facecolor(colors[i % len(colors)])
                patch.set_alpha(0.7)
                patch.set_edgecolor('white')
                patch.set_linewidth(2)
                median.set_color('white')
                median.set_linewidth(2)

            # Style whiskers and caps
            for whisker in bp['whiskers']:
                whisker.set_color('#7f8c8d')
                whisker.set_linewidth(1.5)
            for cap in bp['caps']:
                cap.set_color('#7f8c8d')
                cap.set_linewidth(1.5)

            # Add individual data points with jitter
            # v1.2.8.4: local RandomState (not global np.random) so the plot is
            # deterministic without relying on the engine seeding the global RNG.
            _jit_rng = np.random.RandomState(20260601)
            for i, (pos, data) in enumerate(zip(positions, box_data)):
                if len(data) > 0:
                    jitter = _jit_rng.normal(0, 0.04, len(data))
                    ax.scatter(pos + jitter, data, alpha=0.4, s=20,
                              color=colors[i % len(colors)], edgecolor='white', linewidth=0.5,
                              zorder=3)

            # Add mean markers
            for i, (pos, data) in enumerate(zip(positions, box_data)):
                if len(data) > 0:
                    mean_val = np.mean(data)
                    ax.scatter(pos, mean_val, marker='D', s=80, color='white',
                              edgecolor=colors[i % len(colors)], linewidth=2, zorder=4)

            # Styling
            ax.set_xticks(positions)
            ax.set_xticklabels(conditions, rotation=30, ha='right', fontsize=11)
            ax.set_ylabel("Score", fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)

            # Add legend for mean marker
            ax.scatter([], [], marker='D', s=80, color='white', edgecolor='#2c3e50',
                      linewidth=2, label='Mean')
            ax.legend(loc='upper right', framealpha=0.9)

            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')

            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#ecf0f1')
            ax.set_axisbelow(True)

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception as e:
            # Fallback: try a simpler box plot
            try:
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, 5))
                df_plot = df.copy()
                df_plot['_clean_condition'] = df_plot[condition_column].map(_condition_display_labels(df_plot[condition_column].dropna().unique().tolist()))
                conditions = df_plot['_clean_condition'].unique().tolist()
                box_data = [df_plot[df_plot['_clean_condition'] == c][column].dropna().values for c in conditions]

                ax.boxplot(box_data, labels=conditions)
                ax.set_title(title)
                ax.set_ylabel("Score")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close(fig)
                return img_base64
            except Exception:
                return None

    def _create_interaction_plot(
        self,
        df: pd.DataFrame,
        column: str,
        factor1_col: str,
        factor2_col: str,
        factor1_name: str,
        factor2_name: str,
        title: str = "Interaction Plot",
    ) -> Optional[str]:
        """Create an interaction plot for factorial designs."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get unique levels
            f1_levels = df[factor1_col].dropna().unique().tolist()
            f2_levels = df[factor2_col].dropna().unique().tolist()

            # Colors and markers
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
            markers = ['o', 's', '^', 'D']

            # Calculate means and SEs for each cell
            for i, f2 in enumerate(f2_levels):
                means = []
                errors = []
                for f1 in f1_levels:
                    cell_data = df[(df[factor1_col] == f1) & (df[factor2_col] == f2)][column].dropna()
                    if len(cell_data) > 0:
                        means.append(cell_data.mean())
                        errors.append(_t_ci_halfwidth(float(cell_data.std()), len(cell_data)))
                    else:
                        means.append(np.nan)
                        errors.append(0)

                # Plot line with error bars
                x_positions = range(len(f1_levels))
                ax.errorbar(x_positions, means, yerr=errors,
                           marker=markers[i % len(markers)], markersize=12,
                           color=colors[i % len(colors)], linewidth=2.5,
                           capsize=6, capthick=2, label=f"{factor2_name}: {f2}",
                           markeredgecolor='white', markeredgewidth=2)

            # Styling
            ax.set_xticks(range(len(f1_levels)))
            ax.set_xticklabels([str(l) for l in f1_levels], fontsize=11)
            ax.set_xlabel(factor1_name, fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_ylabel("Mean Score", fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)

            # Legend
            ax.legend(loc='best', framealpha=0.95, fontsize=10)

            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')

            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#ecf0f1')
            ax.set_axisbelow(True)

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception:
            return None

    def _create_effect_size_forest_plot(
        self,
        comparisons: List[Dict[str, Any]],
        title: str = "Effect Sizes (Cohen's d) with 95% CI",
    ) -> Optional[str]:
        """Create a forest plot showing effect sizes for all pairwise comparisons."""
        if not MATPLOTLIB_AVAILABLE or not comparisons:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, max(4, len(comparisons) * 0.6 + 1)))

            y_positions = range(len(comparisons))
            effects = [c['cohens_d'] for c in comparisons]
            labels = [c['comparison'] for c in comparisons]
            significant = [c['significant'] for c in comparisons]

            # Approximate CI for Cohen's d (rough estimate)
            ci_widths = [0.4 for _ in comparisons]  # Simplified

            # Colors based on significance (green=sig, orange=marginal, gray=ns)
            marginals = [c.get('marginally_significant', False) for c in comparisons]
            colors = ['#2ecc71' if sig else ('#f39c12' if marg else '#95a5a6') for sig, marg in zip(significant, marginals)]

            # Plot effect sizes
            for i, (effect, label, sig, color) in enumerate(zip(effects, labels, significant, colors)):
                # Horizontal line for CI
                ax.hlines(i, effect - ci_widths[i], effect + ci_widths[i],
                         color=color, linewidth=3, alpha=0.7)
                # Diamond marker for point estimate
                ax.scatter(effect, i, marker='D', s=150, color=color,
                          edgecolor='white', linewidth=2, zorder=3)

            # Reference line at 0
            ax.axvline(x=0, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7,
                      label='No effect')

            # Effect size interpretation zones
            ax.axvspan(-0.2, 0.2, alpha=0.1, color='#f39c12', label='Negligible')
            ax.axvspan(0.2, 0.5, alpha=0.1, color='#f1c40f')
            ax.axvspan(-0.5, -0.2, alpha=0.1, color='#f1c40f')
            ax.axvspan(0.5, 0.8, alpha=0.1, color='#e67e22')
            ax.axvspan(-0.8, -0.5, alpha=0.1, color='#e67e22')

            # Styling
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels, fontsize=11)
            ax.set_xlabel("Cohen's d", fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)

            # Add interpretation text
            xlim = ax.get_xlim()
            ax.text(xlim[1], -0.7, "Green = Significant (p < .05)\nOrange = Marginally significant (p < .10)\nGray = Non-significant",
                   fontsize=9, ha='right', va='top', style='italic', color='#7f8c8d')

            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')

            ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#ecf0f1')
            ax.set_axisbelow(True)

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception as e:
            # Fallback (v1.2.8.8): simpler plot built only from `comparisons`.
            # (Previously referenced an undefined `pairwise_results`.)
            logger.warning("Forest plot primary path failed (%s); using simple fallback", e)
            try:
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, max(3, len(comparisons) * 0.5 + 1)))
                _d_vals = [float(c.get('cohens_d', 0.0)) for c in comparisons]
                _lbls = [str(c.get('comparison', f'Comparison {k + 1}')) for k, c in enumerate(comparisons)]
                ax.barh(list(range(len(_d_vals))), _d_vals, color='#95a5a6')
                ax.set_yticks(list(range(len(_d_vals))))
                ax.set_yticklabels(_lbls)
                ax.axvline(0, color='#e74c3c', linestyle='--')
                ax.set_xlabel("Cohen's d")
                ax.set_title(title)
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close(fig)
                return img_base64
            except Exception:
                return None

    def _create_histogram_by_condition(
        self,
        df: pd.DataFrame,
        column: str,
        condition_column: str = "CONDITION",
        title: str = "Distribution Histogram by Condition",
    ) -> Optional[str]:
        """Create overlapping histograms for each condition."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6))

            # Clean condition names for display
            df_plot = df.copy()
            df_plot['_clean_condition'] = df_plot[condition_column].map(_condition_display_labels(df_plot[condition_column].dropna().unique().tolist()))
            conditions = df_plot['_clean_condition'].unique().tolist()

            # Modern color palette with transparency
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']

            # Create histograms for each condition
            for i, cond in enumerate(conditions):
                data = df_plot[df_plot['_clean_condition'] == cond][column].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=15, alpha=0.5, label=cond,
                           color=colors[i % len(colors)], edgecolor='white', linewidth=1)

            # Styling
            ax.set_xlabel("Score", fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_ylabel("Frequency", fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
            ax.legend(loc='upper right', framealpha=0.95, fontsize=10)

            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')

            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#ecf0f1')
            ax.set_axisbelow(True)

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception as e:
            # Fallback: try a simpler histogram
            try:
                plt.close('all')
                fig, ax = plt.subplots(figsize=(8, 5))
                df_plot = df.copy()
                data = df_plot[column].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=15, alpha=0.7, color='steelblue', edgecolor='white')
                ax.set_title(title)
                ax.set_xlabel("Score")
                ax.set_ylabel("Frequency")
                plt.tight_layout()

                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close(fig)
                return img_base64
            except Exception:
                return None

    def _create_means_dot_plot(
        self,
        data: Dict[str, Tuple[float, float]],
        title: str,
        ylabel: str,
        grand_mean: Optional[float] = None,
    ) -> Optional[str]:
        """Create a dot plot with means and error bars - cleaner alternative to bar chart."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            conditions = list(data.keys())
            means = [data[c][0] for c in conditions]
            errors = [data[c][1] for c in conditions]

            # Modern color palette
            colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']

            y_positions = range(len(conditions))

            # Plot dots with error bars (horizontal)
            for i, (cond, mean, error) in enumerate(zip(conditions, means, errors)):
                color = colors[i % len(colors)]
                # Error bar
                ax.errorbar(mean, i, xerr=error, fmt='o', markersize=15,
                           color=color, ecolor=color, capsize=8, capthick=2,
                           markeredgecolor='white', markeredgewidth=2, elinewidth=2)
                # Value label
                ax.text(mean + error + 0.05, i, f'{mean:.2f}', va='center', ha='left',
                       fontsize=11, fontweight='bold', color='#2c3e50')

            # Add grand mean line if provided
            if grand_mean is not None:
                ax.axvline(x=grand_mean, color='#e74c3c', linestyle='--', linewidth=2,
                          alpha=0.7, label=f'Grand Mean: {grand_mean:.2f}')
                ax.legend(loc='lower right', framealpha=0.95)

            # Styling
            ax.set_yticks(y_positions)
            ax.set_yticklabels(conditions, fontsize=11)
            ax.set_xlabel(ylabel, fontsize=13, fontweight='bold', color='#2c3e50')
            ax.set_title(title, fontsize=14, fontweight='bold', color='#2c3e50', pad=20)

            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#bdc3c7')
            ax.spines['bottom'].set_color('#bdc3c7')

            ax.xaxis.grid(True, linestyle='--', alpha=0.5, color='#ecf0f1')
            ax.set_axisbelow(True)

            # Adjust x-axis to show full range
            ax.set_xlim(left=min(means) - max(errors) - 0.5)

            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return img_base64
        except Exception:
            return None

    def generate_html_report(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        schema_validation: Optional[Dict[str, Any]] = None,
        prereg_text: str = "",
        team_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a comprehensive HTML report with visualizations and statistical tests.

        The report is built section by section, each behind a guard: a section that raises is replaced by one note,
        ``[section "<title>" could not be generated: <reason>]``, and the others are not affected. Skipped sections
        are logged and listed in ``self.section_errors``. Every dependent variable is its own guarded block.
        """
        self.section_errors = []
        ctx = _report_context(df, metadata, prereg_text, team_info, html=True)
        ctx.out = self._html_document_head(metadata)
        self._run_section(ctx, "Study overview", self._html_overview)
        self._run_section(ctx, "1. Sample overview", self._html_sample_overview)
        self._run_section(ctx, "Executive summary slot", self._html_exec_slot)
        self._run_section(ctx, "3. Statistical analysis by DV", self._html_dv_header)
        for scale in list(ctx.scales or []):
            ctx.dv_batch = [scale]
            name = scale.get("name", "Scale") if isinstance(scale, dict) else scale
            self._run_section(ctx, f"3. Statistical analysis by DV: {name}", self._html_dv_scale)
        self._run_section(ctx, "4-5. Persona and categorical analysis", self._html_persona)
        self._run_section(ctx, "2. Executive summary", self._html_exec_summary, slot=ctx.exec_summary_index)
        for title, build in (
            ("6. Effect size verification", self._html_effect_verification),
            ("7. Data quality and exclusions", self._html_exclusions),
            ("Generation warnings", self._html_generation_warnings),
            ("8. Instructor notes and methodology", self._html_methodology),
            ("9. Data dictionary", self._html_data_dictionary),
            ("Footer", self._html_footer),
        ):
            self._run_section(ctx, title, build)
        # Safety net: whatever user-controlled text reached the markup, the finished report contains no
        # script, iframe, form, event handler, external resource or javascript: link, and the browser
        # that opens the file is told to run no script and load nothing from the network.
        return _harden_report_html(_finalize_p_text("\n".join(map(str, ctx.out)), html=True))

    def _html_document_head(self, metadata: Dict[str, Any]) -> List[str]:
        """Open the HTML document: styles, contents sidebar and the start of the report container."""

        # CSS styles for the report (v1.3.4: improved layout with TOC and sections)
        css = """
        <style>
            * { box-sizing: border-box; }
            body { font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; color: #1a1a2e; line-height: 1.6; }
            .page-wrapper { max-width: 1200px; margin: 0 auto; display: flex; gap: 30px; align-items: flex-start; }
            .toc-sidebar { position: sticky; top: 20px; width: 220px; min-width: 220px; background: white; border-radius: 10px; padding: 20px 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); font-size: 0.82em; max-height: calc(100vh - 40px); overflow-y: auto; }
            .toc-sidebar h3 { color: #1e3a5f; font-size: 0.95em; margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #2563eb; }
            .toc-sidebar a { display: block; color: #475569; text-decoration: none; padding: 5px 8px; border-radius: 4px; margin-bottom: 2px; transition: all 0.15s; }
            .toc-sidebar a:hover { background: #eef2ff; color: #2563eb; }
            .toc-sidebar a.toc-h2 { font-weight: 600; color: #1e3a5f; }
            .toc-sidebar a.toc-h3 { padding-left: 18px; font-size: 0.92em; }
            .report-container { flex: 1; min-width: 0; background: white; padding: 40px 45px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
            h1 { color: #1a1a2e; border-bottom: 3px solid #2563eb; padding-bottom: 12px; font-weight: 700; font-size: 1.8em; }
            h2 { color: #1e3a5f; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-top: 40px; font-weight: 600; font-size: 1.4em; scroll-margin-top: 20px; }
            h3 { color: #475569; font-weight: 600; font-size: 1.15em; margin-top: 24px; scroll-margin-top: 20px; }
            h4 { color: #64748b; font-weight: 500; font-size: 1.05em; margin-top: 18px; }
            p { margin: 8px 0; }
            table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.92em; }
            th, td { border: 1px solid #e2e8f0; padding: 10px 12px; text-align: left; }
            th { background-color: #1e3a5f; color: white; font-weight: 500; letter-spacing: 0.02em; }
            tr:nth-child(even) { background-color: #f8fafc; }
            tr:hover { background-color: #eef2ff; }
            .section-block { background: #fafbfc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 24px 28px; margin: 20px 0; }
            .stat-box { background: #f0f7ff; padding: 18px 20px; border-radius: 8px; margin: 12px 0; border-left: 4px solid #2563eb; }
            .warning-box { background: #fffbeb; padding: 18px 20px; border-radius: 8px; margin: 12px 0; border-left: 4px solid #f59e0b; }
            .success-box { background: #f0fdf4; padding: 18px 20px; border-radius: 8px; margin: 12px 0; border-left: 4px solid #22c55e; }
            .error-box { background: #fef2f2; padding: 18px 20px; border-radius: 8px; margin: 12px 0; border-left: 4px solid #ef4444; }
            .chart-container { text-align: center; margin: 20px 0; }
            .chart-container img { max-width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
            .confidential { background: #dc2626; color: white; padding: 6px 18px; border-radius: 4px; display: inline-block; margin-bottom: 20px; font-weight: 600; letter-spacing: 0.05em; font-size: 0.85em; }
            .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }
            .metric-card { background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%); color: white; padding: 22px 16px; border-radius: 10px; text-align: center; box-shadow: 0 3px 12px rgba(37,99,235,0.15); }
            .metric-value { font-size: 2em; font-weight: 700; letter-spacing: -0.02em; }
            .metric-label { font-size: 0.85em; opacity: 0.9; margin-top: 4px; font-weight: 400; }
            .back-to-top { display: inline-block; margin-top: 10px; font-size: 0.8em; color: #94a3b8; text-decoration: none; }
            .back-to-top:hover { color: #2563eb; }
            code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; color: #475569; }
            .sig { color: #16a34a; font-weight: 600; }
            .marginal { color: #f39c12; font-weight: 600; }
            .nonsig { color: #94a3b8; }
            ol, ul { padding-left: 24px; }
            li { margin-bottom: 4px; }
            em { color: #64748b; }
            @media print {
                body { background: white; padding: 0; }
                .page-wrapper { display: block; }
                .toc-sidebar { display: none; }
                .report-container { box-shadow: none; padding: 20px; }
                .metric-card { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
                .confidential { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
                th { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
            }
            @media (max-width: 900px) {
                .page-wrapper { flex-direction: column; }
                .toc-sidebar { position: static; width: 100%; min-width: 0; }
            }
        </style>
        """

        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>Instructor Report: {metadata.get('study_title', 'Study')}</title>",
            css,
            "</head>",
            "<body>",
            "<div class='page-wrapper'>",
            # Table of Contents sidebar
            "<nav class='toc-sidebar'>",
            "<h3>Contents</h3>",
            "<a href='#top' class='toc-h2'>Report Header</a>",
            "<a href='#study-overview' class='toc-h2'>Study Overview</a>",
            "<a href='#sample-overview' class='toc-h2'>1. Sample Overview</a>",
            "<a href='#exec-summary' class='toc-h2'>2. Executive Summary</a>",
            "<a href='#statistical-analysis' class='toc-h2'>3. Statistical Analysis</a>",
            "<a href='#persona-analysis' class='toc-h2'>4. Persona &amp; Response</a>",
            "<a href='#categorical-analysis' class='toc-h2'>5. Categorical Analysis</a>",
            "<a href='#effect-verification' class='toc-h2'>6. Effect Verification</a>",
            "<a href='#data-quality' class='toc-h2'>7. Data Quality</a>",
            "<a href='#methodology' class='toc-h2'>8. Methodology</a>",
            "<a href='#data-dictionary' class='toc-h2'>9. Data Dictionary</a>",
            "</nav>",
            # Main report content
            "<div class='report-container'>",
            "<a id='top'></a>",
        ]
        return html_parts

    def _html_overview(self, ctx: Any) -> None:
        """Report header and study overview (design, DVs, generation method and details)."""
        html_parts, metadata, team_info = ctx.out, ctx.metadata, ctx.team_info

        # Header
        html_parts.append("<span class='confidential'>CONFIDENTIAL &mdash; INSTRUCTOR ONLY</span>")
        _e = lambda value: _html_lib.escape(str(value), quote=True)  # noqa: E731 - user-controlled text -> HTML
        html_parts.append("<h1>Comprehensive Simulation &amp; Statistical Report</h1>")
        html_parts.append(f"<p style='color:#64748b;margin-top:-8px;font-size:1.05em;'>Behavioral Experiment Simulation Tool v{__version__}</p>")

        # =============================================================
        # STUDY OVERVIEW SECTION (All first page info)
        # =============================================================
        html_parts.append("<a id='study-overview'></a>")
        html_parts.append("<div class='section-block'>")
        html_parts.append("<h2>Study Overview</h2>")

        # Study Title
        study_title = metadata.get('study_title', 'Untitled Study')
        html_parts.append(f"<h3>{_e(study_title)}</h3>")

        # Team Information
        if team_info:
            team_name = team_info.get('team_name', '')
            team_members = team_info.get('team_members', '')
            if team_name:
                html_parts.append(f"<p><strong>Team:</strong> {_e(team_name)}</p>")
            if team_members:
                members_formatted = team_members.replace('\n', ', ').replace(',,', ',').strip(', ')
                html_parts.append(f"<p><strong>Team Members:</strong> {_e(members_formatted)}</p>")

        # Study Description / Abstract
        study_description = metadata.get('study_description', '')
        if study_description:
            html_parts.append(f"<p><strong>Abstract:</strong> {_e(study_description)}</p>")

        # Experimental Design (inside same section-block)
        html_parts.append("<h3>Experimental Design</h3>")

        # Conditions
        conditions = metadata.get('conditions', [])
        if conditions:
            html_parts.append(f"<p><strong>Conditions ({len(conditions)}):</strong></p>")
            html_parts.append("<ul>")
            for cond in conditions:
                html_parts.append(f"<li>{_e(cond)}</li>")
            html_parts.append("</ul>")

        # Factors
        factors = metadata.get('factors', [])
        if factors:
            html_parts.append(f"<p><strong>Factors ({len(factors)}):</strong></p>")
            html_parts.append("<ul>")
            for factor in factors:
                factor_name = factor.get('name', 'Factor')
                levels = factor.get('levels', [])
                html_parts.append(f"<li>{_e(factor_name)}: {_e(', '.join(str(l) for l in levels))}</li>")
            html_parts.append("</ul>")

        # Scales / DVs
        scales_meta = metadata.get('scales', [])
        if scales_meta:
            html_parts.append(f"<p><strong>Dependent Variables ({len(scales_meta)}):</strong></p>")
            html_parts.append("<ul>")
            for scale in scales_meta:
                scale_name = scale.get('name', 'Scale')
                scale_points = scale.get('scale_points', 7)
                num_items = scale.get('num_items', 1)
                html_parts.append(f"<li>{_e(scale_name)} ({_e(num_items)} item{'s' if num_items > 1 else ''}, {_e(scale_points)}-point)</li>")
            html_parts.append("</ul>")

        # NOTE: Data Dictionary moved to bottom of report (after Methodology) per user request

        # Effect Sizes
        effect_sizes = metadata.get('effect_sizes_configured', [])
        if effect_sizes and any(e.get('cohens_d', 0) > 0 for e in effect_sizes):
            html_parts.append("<p><strong>Hypothesized Effects:</strong></p>")
            html_parts.append("<ul>")
            for effect in effect_sizes:
                d = effect.get('cohens_d', 0)
                if d > 0:
                    var = effect.get('variable', '')
                    direction = effect.get('direction', 'higher')
                    html_parts.append(f"<li>{_e(var)}: d = {d:.2f} ({_e(direction)} in treatment)</li>")
            html_parts.append("</ul>")

        # ── Study Context / Domain ─────────────────────────────────────
        study_context = metadata.get("study_context", {})
        detected_domains = metadata.get("detected_domains", [])
        if study_context or detected_domains:
            html_parts.append("<h3>Research Context</h3>")
            _domain = study_context.get("study_domain", study_context.get("domain", ""))
            if _domain:
                html_parts.append(f"<p><strong>Research Domain:</strong> {_e(_domain.title())}</p>")
            if detected_domains:
                html_parts.append(f"<p><strong>Detected Topic Areas:</strong> {_e(', '.join(str(x) for x in detected_domains[:10]))}</p>")
            _source = study_context.get("source", "")
            if _source:
                _source_label = "Conversational Builder" if "builder" in _source else "QSF Upload"
                html_parts.append(f"<p><strong>Input Method:</strong> {_source_label}</p>")
            _participant_chars = study_context.get("participant_characteristics", "")
            if _participant_chars:
                html_parts.append(f"<p><strong>Target Participants:</strong> {_e(_participant_chars)}</p>")
            _persona_domains = study_context.get("persona_domains", [])
            if _persona_domains:
                html_parts.append(f"<p><strong>Persona Domains Activated:</strong> {_e(', '.join(str(d).replace('_', ' ').title() for d in _persona_domains))}</p>")
        # Open-ended questions summary
        oe_questions = metadata.get("open_ended_questions", [])
        if oe_questions:
            html_parts.append(f"<h3>Open-Ended Questions ({len(oe_questions)})</h3>")
            html_parts.append("<ul>")
            for oe in oe_questions:
                q_text = oe.get("question_text", oe.get("name", "")) if isinstance(oe, dict) else str(oe)
                var_name = oe.get("variable_name", "") if isinstance(oe, dict) else ""
                q_ctx = oe.get("question_context", "") if isinstance(oe, dict) else ""
                if q_text:
                    _var_tag = f" <code>({_e(var_name)})</code>" if var_name else ""
                    _ctx_tag = f"<br><small style='color:#666;'>Context: {_e(q_ctx)}</small>" if q_ctx else ""
                    html_parts.append(f"<li>{_e(str(q_text)[:120])}{_var_tag}{_ctx_tag}</li>")
            html_parts.append("</ul>")

        # v1.8.7: LLM Generation Details in HTML report
        llm_stats_html = metadata.get('llm_stats', metadata.get('llm_response_stats', {}))
        llm_calls_html = llm_stats_html.get('llm_calls', 0) if llm_stats_html else 0
        llm_attempts_html = llm_stats_html.get('llm_attempts', llm_calls_html) if llm_stats_html else 0
        pool_size_h = llm_stats_html.get('pool_size', 0) if llm_stats_html else 0
        fallback_h = llm_stats_html.get('fallback_uses', 0) if llm_stats_html else 0
        batch_failures_h = llm_stats_html.get('batch_failures', 0) if llm_stats_html else 0
        allow_template_fallback_h = bool(llm_stats_html.get('allow_template_fallback', True)) if llm_stats_html else True
        llm_init_error_h = str(metadata.get('llm_init_error', '') or '').strip()
        # v1.0.9.2: Read user-selected generation method for accurate labeling
        _gen_method_h = metadata.get('generation_method', '')
        _is_adaptive_h = _gen_method_h in ('experimental', 'abe_v2')
        _is_template_h = _gen_method_h == 'template'
        _is_abe3_h = _gen_method_h == 'abe3'
        # v1.2.5.4: Recognize free_llm and own_api as AI-powered methods
        _is_llm_method_h = _gen_method_h in ('free_llm', 'own_api')

        if oe_questions:
            html_parts.append("<h3 style='margin-top:20px;'>Response Generation Method</h3>")
            if pool_size_h > 0:
                if _is_abe3_h or _is_adaptive_h or _is_llm_method_h:
                    html_parts.append("<p><strong>Generation approach:</strong> Adaptive Behavioral Engine 3.0 + AI-Powered LLM</p>")
                else:
                    html_parts.append("<p><strong>Generation approach:</strong> AI-Powered (Language Model)</p>")
                html_parts.append(f"<p><strong>Total API calls:</strong> {llm_calls_html}</p>")
                html_parts.append(f"<p><strong>Response pool size:</strong> {pool_size_h} base responses</p>")
                if fallback_h > 0:
                    total_h = pool_size_h + fallback_h
                    pct_h = (fallback_h / max(1, total_h)) * 100
                    html_parts.append(f"<p><strong>Template fallback:</strong> {fallback_h} response(s) ({pct_h:.0f}%)</p>")
                else:
                    html_parts.append("<p><strong>Template fallback:</strong> None needed (all AI-generated)</p>")
            elif llm_calls_html > 0 or llm_attempts_html > 0:
                if _is_abe3_h or _is_adaptive_h or _is_llm_method_h:
                    html_parts.append("<p><strong>Generation approach:</strong> Adaptive Behavioral Engine 3.0 (template-backed, AI unavailable)</p>")
                elif _is_template_h:
                    html_parts.append("<p><strong>Generation approach:</strong> Template Engine (225+ Research Domains)</p>")
                else:
                    html_parts.append("<p><strong>Generation approach:</strong> Template Engine (AI providers were unavailable)</p>")
                if llm_calls_html > 0:
                    html_parts.append(f"<p>{llm_calls_html} API request(s) were sent but all providers were unavailable or returned errors.</p>")
                else:
                    html_parts.append(f"<p>{llm_attempts_html} provider probe(s) were made but no usable connection was established.</p>")
                if batch_failures_h > 0:
                    html_parts.append(f"<p>{batch_failures_h} batch generation attempt(s) did not produce usable responses.</p>")
                html_parts.append("<p>All open-ended responses were generated using the built-in template engine.</p>")
                if not _is_template_h:
                    html_parts.append("<p><em><strong>Tip:</strong> Provide your own API key (Groq, Google AI, etc.) for AI-powered responses.</em></p>")
            elif llm_init_error_h:
                html_parts.append("<p><strong>Generation approach:</strong> LLM initialization failure (run integrity warning)</p>")
                html_parts.append(f"<p>LLM generator failed to initialize: <code>{llm_init_error_h}</code></p>")
                html_parts.append("<p>Open-ended responses did not run through API calls in this run.</p>")
            elif not allow_template_fallback_h:
                html_parts.append("<p><strong>Generation approach:</strong> LLM-first strict mode</p>")
                html_parts.append("<p>Template fallback was disabled and no successful API output was recorded; rerun with valid API connectivity.</p>")
            else:
                if _is_abe3_h or _is_llm_method_h:
                    # v1.2.5.4: free_llm/own_api with no OE still use ABE 3.0 for numeric
                    html_parts.append("<p><strong>Generation approach:</strong> Adaptive Behavioral Engine 3.0</p>")
                    html_parts.append("<p>Census-weighted demographics with deep persona coherence. "
                                     "Calibrated error rates by education level (Frederick 2005). "
                                     "Stylometric voice fingerprinting ensures consistent writing style per participant.</p>")
                    if _is_llm_method_h:
                        _gen_label_h = metadata.get('generation_method_label', _gen_method_h)
                        html_parts.append(f"<p><strong>Selected method:</strong> {_gen_label_h}</p>")
                elif _is_adaptive_h:
                    html_parts.append("<p><strong>Generation approach:</strong> Adaptive Behavioral Engine 3.0</p>")
                    html_parts.append("<p>50+ persona archetypes with 7-dimensional trait profiles. "
                                     "30+ paradigm recognizers calibrated to published norms. "
                                     "Cross-DV coherence ensures numeric ratings and text responses tell a consistent story.</p>")
                elif _is_template_h:
                    html_parts.append("<p><strong>Generation approach:</strong> Template Engine (225+ Research Domains)</p>")
                    html_parts.append("<p>Responses generated offline using the built-in template engine.</p>")
                else:
                    html_parts.append("<p><strong>Generation approach:</strong> Advanced Template Engine (225+ Research Domains)</p>")
                    html_parts.append("<p>Responses were generated using the built-in behavioral simulation engine "
                                     "covering 225+ research domains. Each response is unique per participant with topic-grounded content.</p>")

        # Generation Details (still inside section-block)
        html_parts.append("<h3 style='margin-top:28px;padding-top:16px;border-top:1px solid #e2e8f0;'>Generation Details</h3>")
        html_parts.append(f"<p><strong>Generated:</strong> {metadata.get('generation_timestamp', datetime.now().isoformat())}</p>")
        html_parts.append(f"<p><strong>Run ID:</strong> <code>{metadata.get('run_id', 'N/A')}</code></p>")
        _html_gen_method = metadata.get('generation_method_label', metadata.get('generation_method', 'N/A'))
        html_parts.append(f"<p><strong>Generation Method:</strong> {_html_gen_method}</p>")
        html_parts.append(f"<p><strong>Mode:</strong> {metadata.get('simulation_mode', 'pilot').title()}</p>")
        html_parts.append(f"<p><strong>Seed:</strong> <code>{metadata.get('seed', 'N/A')}</code></p>")
        html_parts.append(f"<p><strong>App Version:</strong> {metadata.get('app_version', __version__)}</p>")

        # Internal usage counter (for instructor tracking)
        usage_stats = metadata.get('usage_stats', {})
        total_simulations = usage_stats.get('total_simulations', 'N/A')
        html_parts.append(f"<p><strong>Total Simulations Run (all time):</strong> {total_simulations}</p>")
        html_parts.append("</div>")  # close Study Overview section-block
        html_parts.append("<a href='#top' class='back-to-top'>Back to top</a>")

    def _html_sample_overview(self, ctx: Any) -> None:
        """Section 1: sample size, exclusions and condition distribution."""
        html_parts, df, n_total, n_clean, exclusion_rate, conditions = (
            ctx.out, ctx.df, ctx.n_total, ctx.n_clean, ctx.exclusion_rate, ctx.conditions)
        _cond_label = ctx.cond_label  # v1.2.9.1: one collision-free label per condition

        html_parts.append("<a id='sample-overview'></a>")
        html_parts.append("<h2>1. Sample Overview</h2>")
        html_parts.append("<div class='metric-grid'>")
        html_parts.append(f"<div class='metric-card'><div class='metric-value'>{n_total}</div><div class='metric-label'>Total N</div></div>")
        html_parts.append(f"<div class='metric-card'><div class='metric-value'>{n_clean}</div><div class='metric-label'>Clean N</div></div>")
        html_parts.append(f"<div class='metric-card'><div class='metric-value'>{exclusion_rate:.1f}%</div><div class='metric-label'>Exclusion Rate</div></div>")
        html_parts.append(f"<div class='metric-card'><div class='metric-value'>{len(conditions)}</div><div class='metric-label'>Conditions</div></div>")
        html_parts.append("</div>")

        # Condition distribution (with cleaned names)
        if "CONDITION" in df.columns:
            html_parts.append("<h3>Condition Distribution</h3>")
            html_parts.append("<table><tr><th>Condition</th><th>N</th><th>%</th></tr>")
            cond_counts = df["CONDITION"].value_counts()
            for cond, count in cond_counts.items():
                pct = count / n_total * 100
                clean_cond = _cond_label(cond)
                html_parts.append(f"<tr><td>{clean_cond}</td><td>{count}</td><td>{pct:.1f}%</td></tr>")
            html_parts.append("</table>")

        html_parts.append("<a href='#top' class='back-to-top'>Back to top</a>")

    def _html_exec_slot(self, ctx: Any) -> None:
        """Reserve the place of the executive summary; it is filled in after the DV analysis."""
        html_parts = ctx.out

        # v1.3.4: Executive Summary placeholder — will be computed after DV analysis and inserted here
        html_parts.append("<a id='exec-summary'></a>")
        exec_summary_index = len(html_parts)
        html_parts.append("<!-- EXEC_SUMMARY_PLACEHOLDER -->")
        ctx.exec_summary_index = exec_summary_index

    def _html_dv_header(self, ctx: Any) -> None:
        """Section 3 heading; also prepares the column registry the per-DV blocks use."""
        html_parts, metadata = ctx.out, ctx.metadata

        # DV Analysis with statistical tests
        scales = metadata.get("scales", [])
        html_parts.append("<a id='statistical-analysis'></a>")
        html_parts.append("<h2>3. Statistical Analysis by DV</h2>")

        # Track all scale analysis results for executive summary
        all_scale_results = []

        # v1.0.6.3: Build column registry for HTML report (same as markdown)
        _html_col_registry: Dict[str, List[str]] = _scale_column_registry(metadata)
        ctx.scales, ctx.all_scale_results, ctx.col_registry = scales, all_scale_results, _html_col_registry

    def _html_dv_scale(self, ctx: Any) -> None:
        """Section 3: the block of one dependent variable (``ctx.dv_batch`` holds just that scale)."""
        html_parts, df_clean, metadata, conditions, prereg_text = ctx.out, ctx.df_clean, ctx.metadata, ctx.conditions, ctx.prereg_text
        all_scale_results, _html_col_registry, scales = ctx.all_scale_results, ctx.col_registry, ctx.dv_batch
        _cond_labels, _cond_label = ctx.cond_labels, ctx.cond_label  # v1.2.9.1: one label per condition

        # v1.2.9.1: every column that belongs to ANY DV (its items and composites) -- never a covariate
        _dv_item_columns: set = set()
        for _reg_cols in _html_col_registry.values():
            _dv_item_columns.update(str(c) for c in _reg_cols)
        for _sc in (ctx.scales or scales):
            if not isinstance(_sc, dict):
                continue  # a malformed scale entry is handled (and reported) by its own DV block
            try:
                _dv_item_columns.update(str(c) for c in _find_scale_columns(df_clean, _sc, _html_col_registry))
            except (AttributeError, TypeError, ValueError):
                logger.warning("DV columns of %r could not be listed for the covariate exclusion", _sc, exc_info=True)

        for scale in scales:
            scale_name = scale.get("name", "Scale")
            # v1.0.6.3: Use unified column finder (fixes N=1 / missing DV analysis)
            scale_cols = _find_scale_columns(df_clean, scale, _html_col_registry)

            if not scale_cols:
                continue

            html_parts.append(f"<h3>{scale_name}</h3>")

            # Calculate composite
            if len(scale_cols) >= 1:
                composite = df_clean[scale_cols].mean(axis=1)
                df_analysis = df_clean.copy()
                df_analysis["_composite"] = composite

                # v1.2.9.1: a composite that cannot differ between participants (constant-sum or rank-order items,
                # or no variation at all) gets no tests and no charts: with SD = 0 in every group scipy answers
                # t = nan / p = nan (or t = +/-inf, p = 0) and the numpy fallbacks t = 0, p = 1. Say so and compare
                # the items instead.
                if _is_joint_item_scale(scale) or _composite_has_no_variation(
                        df_analysis["_composite"], df_analysis["CONDITION"] if "CONDITION" in df_analysis.columns else None):
                    html_parts.extend(self._no_variation_block(scale, scale_cols, df_analysis, df_analysis["_composite"], conditions, html=True))
                    all_scale_results.append({"scale_name": scale_name, "chart_data": {}, "stats_results": {}, "not_tested": True})
                    continue

                # Descriptive stats table
                html_parts.append("<h4>Descriptive Statistics</h4>")
                html_parts.append("<table><tr><th>Condition</th><th>N</th><th>Mean</th><th>SD</th><th>95% CI</th></tr>")

                chart_data = {}
                _assigned_note: Dict[str, int] = {}
                _no_score_conds: List[str] = []
                for cond in conditions:
                    cond_data = df_analysis[df_analysis["CONDITION"] == cond]["_composite"].dropna()
                    # v1.2.5.1: SD, SE and CI use valid observations only.
                    # v1.2.9.1: the displayed N is that same analytic N (observations with a score),
                    # not the number assigned to the condition; the difference is disclosed below the table.
                    n_total_cond = int((df_analysis["CONDITION"] == cond).sum())
                    n_valid = len(cond_data)
                    if n_valid > 0:
                        mean = cond_data.mean()
                        n = n_valid  # Display analytic N
                        # v1.2.5.1: SD and SE MUST use valid N, not total N
                        sd = cond_data.std() if n_valid > 1 else 0.0
                        if np.isnan(sd):
                            sd = 0.0
                        se = sd / np.sqrt(n_valid) if n_valid > 1 else 0.0
                        ci_half = _t_ci_halfwidth(sd, n_valid)  # t critical value (1.96 is too narrow for small n)
                        ci_low = mean - ci_half
                        ci_high = mean + ci_half
                        clean_cond = _cond_label(cond)
                        if n_total_cond != n_valid:
                            _assigned_note[clean_cond] = n_total_cond
                        if n == 1:
                            html_parts.append(f"<tr><td>{clean_cond}</td><td>{n}</td><td>{mean:.3f}</td><td>—</td><td>N/A (single observation)</td></tr>")
                        else:
                            html_parts.append(f"<tr><td>{clean_cond}</td><td>{n}</td><td>{mean:.3f}</td><td>{sd:.3f}</td><td>[{ci_low:.3f}, {ci_high:.3f}]</td></tr>")
                        chart_data[clean_cond] = (mean, ci_half)
                    else:
                        _no_score_conds.append(_cond_label(cond))

                html_parts.append("</table>")
                if _assigned_note or _no_score_conds:
                    _note_bits = []
                    if _assigned_note:
                        _note_bits.append(
                            "N is the number of participants with a score (the N every statistic uses). "
                            "Assigned to the condition, including those without a score: "
                            + ", ".join(f"{_html_lib.escape(str(k))} {v}" for k, v in _assigned_note.items()) + "."
                        )
                    if _no_score_conds:
                        _note_bits.append("No valid scores in: " + ", ".join(_html_lib.escape(str(c)) for c in _no_score_conds) + ".")
                    html_parts.append("<p style='font-size:0.85em;color:#64748b;'>" + " ".join(_note_bits) + "</p>")

                # Add interpretation for descriptive statistics
                if chart_data:
                    means = [(c, m[0]) for c, m in chart_data.items()]
                    if len(means) >= 2:
                        sorted_means = sorted(means, key=lambda x: x[1], reverse=True)
                        highest = sorted_means[0]
                        lowest = sorted_means[-1]
                        diff = highest[1] - lowest[1]
                        grand_mean = sum(m[1] for m in means) / len(means)

                        html_parts.append("<div class='interpretation-box' style='background:#f8f9fa;padding:12px;border-radius:6px;margin:10px 0;border-left:3px solid #3498db;'>")
                        html_parts.append(f"<strong>Summary:</strong> The <em>{highest[0]}</em> condition showed the highest mean ({highest[1]:.2f}), while <em>{lowest[0]}</em> showed the lowest ({lowest[1]:.2f}). ")
                        html_parts.append(f"The difference between highest and lowest conditions is {diff:.2f} scale points. ")
                        html_parts.append(f"The grand mean across conditions is {grand_mean:.2f}.")
                        html_parts.append("</div>")

                # Run statistical tests first to get effect size and p-value for chart
                stats_results = {}
                if len(conditions) >= 2 and "CONDITION" in df_analysis.columns:
                    stats_results = self._run_statistical_tests(
                        df_analysis, "_composite", "CONDITION",
                        condition_order=list(conditions), condition_labels=_cond_labels,
                    )
                    stats_results = _finite_results_only(stats_results)  # v1.2.9.1: no nan/inf test ever reaches the page

                # Extract effect size and p-value for bar chart annotation
                effect_size_val = stats_results.get("cohens_d", {}).get("value") if "cohens_d" in stats_results else None
                p_value_val = stats_results.get("t_test", {}).get("p_value") if "t_test" in stats_results else (
                    stats_results.get("anova", {}).get("p_value") if "anova" in stats_results else None
                )

                # ========================================
                # VISUALIZATIONS - GUARANTEED to produce charts
                # Strategy: Try matplotlib first, always fall back to SVG
                # ========================================
                html_parts.append("<h4>Visualizations</h4>")
                viz_count = 0
                matplotlib_worked = False

                # Prepare data for SVG fallbacks (condition -> list of values)
                svg_dist_data = {}
                if "CONDITION" in df_analysis.columns:
                    for cond in conditions:
                        mask = df_analysis["CONDITION"] == cond  # exact match: Group_1 / Group_2 stay separate
                        svg_dist_data[_cond_label(cond)] = df_analysis.loc[mask, "_composite"].dropna().tolist()

                # chart_data keys are already the collision-free display labels
                clean_chart_data = dict(chart_data)

                # === TRY MATPLOTLIB CHARTS FIRST ===
                if MATPLOTLIB_AVAILABLE:
                    # Visualization 1: Bar chart with error bars
                    try:
                        chart_img = self._create_bar_chart(
                            chart_data,
                            f"{scale_name}: Means with 95% CI",
                            "Mean Score",
                            effect_size=effect_size_val,
                            p_value=p_value_val
                        )
                        if chart_img:
                            html_parts.append("<div class='chart-container'>")
                            html_parts.append(f"<img src='data:image/png;base64,{chart_img}' alt='Bar chart'>")
                            html_parts.append("</div>")
                            viz_count += 1
                            matplotlib_worked = True
                    except Exception:
                        pass

                    # Visualization 2: Distribution plot (violin + box)
                    try:
                        dist_img = self._create_distribution_plot(
                            df_analysis, "_composite", "CONDITION",
                            f"{scale_name}: Distribution by Condition"
                        )
                        if dist_img:
                            html_parts.append("<div class='chart-container'>")
                            html_parts.append(f"<img src='data:image/png;base64,{dist_img}' alt='Distribution'>")
                            html_parts.append("</div>")
                            viz_count += 1
                            matplotlib_worked = True
                    except Exception:
                        pass

                    # Visualization 3: Histogram by condition
                    try:
                        hist_img = self._create_histogram_by_condition(
                            df_analysis, "_composite", "CONDITION",
                            f"{scale_name}: Score Distribution Histogram"
                        )
                        if hist_img:
                            html_parts.append("<div class='chart-container'>")
                            html_parts.append(f"<img src='data:image/png;base64,{hist_img}' alt='Histogram'>")
                            html_parts.append("</div>")
                            viz_count += 1
                            matplotlib_worked = True
                    except Exception:
                        pass

                # === SVG FALLBACK - GUARANTEED TO WORK ===
                # If matplotlib failed or produced no charts, use SVG
                if viz_count == 0 and SVG_CHARTS_AVAILABLE:
                    try:
                        # SVG Bar Chart - ALWAYS works
                        svg_bar = svg_charts.create_bar_chart_svg(
                            clean_chart_data,
                            title=f"{scale_name}: Means with 95% CI",
                            ylabel="Mean Score",
                            effect_size=effect_size_val,
                            p_value=p_value_val
                        )
                        html_parts.append("<div class='chart-container'>")
                        html_parts.append(svg_bar)
                        html_parts.append("</div>")
                        viz_count += 1
                    except Exception:
                        pass

                    try:
                        # SVG Distribution Plot - ALWAYS works
                        svg_dist = svg_charts.create_distribution_svg(
                            svg_dist_data,
                            title=f"{scale_name}: Distribution by Condition",
                            xlabel="Score"
                        )
                        html_parts.append("<div class='chart-container'>")
                        html_parts.append(svg_dist)
                        html_parts.append("</div>")
                        viz_count += 1
                    except Exception:
                        pass

                    try:
                        # SVG Histogram - ALWAYS works
                        svg_hist = svg_charts.create_histogram_svg(
                            svg_dist_data,
                            title=f"{scale_name}: Score Distribution Histogram",
                            xlabel="Score"
                        )
                        html_parts.append("<div class='chart-container'>")
                        html_parts.append(svg_hist)
                        html_parts.append("</div>")
                        viz_count += 1
                    except Exception:
                        pass

                    try:
                        # SVG Means Comparison - ALWAYS works
                        grand_mean = df_analysis["_composite"].mean() if "_composite" in df_analysis.columns else None
                        svg_means = svg_charts.create_means_comparison_svg(
                            clean_chart_data,
                            title=f"{scale_name}: Condition Means",
                            xlabel="Mean Score",
                            grand_mean=grand_mean
                        )
                        html_parts.append("<div class='chart-container'>")
                        html_parts.append(svg_means)
                        html_parts.append("</div>")
                        viz_count += 1
                    except Exception:
                        pass

                # === ULTIMATE FALLBACK: Generate inline SVG directly ===
                # If even SVG module failed, generate basic inline SVG
                if viz_count == 0 and clean_chart_data:
                    try:
                        # Create a simple inline SVG bar chart directly
                        conds = list(clean_chart_data.keys())
                        means = [clean_chart_data[c][0] for c in conds]
                        max_mean = max(means) if means else 1
                        if not max_mean or max_mean <= 0:
                            max_mean = 1  # avoid div-by-zero when all means are 0/negative

                        svg_lines = [
                            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300" style="max-width:100%;height:auto;background:#fff;font-family:Arial,sans-serif;">',
                            '<rect width="500" height="300" fill="white"/>',
                            f'<text x="250" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#2c3e50">{scale_name}: Condition Means</text>',
                        ]

                        bar_width = 60
                        spacing = 400 / len(conds)
                        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']

                        for i, (cond, mean) in enumerate(zip(conds, means)):
                            x = 50 + spacing * i + spacing/2 - bar_width/2
                            bar_height = max(0, (mean / max_mean) * 180)  # clamp: no negative-height rects
                            y = 250 - bar_height
                            color = colors[i % len(colors)]
                            label = cond[:10] if len(cond) > 10 else cond

                            svg_lines.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" opacity="0.8"/>')
                            svg_lines.append(f'<text x="{x + bar_width/2}" y="{y - 5}" text-anchor="middle" font-size="11" fill="#2c3e50">{mean:.2f}</text>')
                            svg_lines.append(f'<text x="{x + bar_width/2}" y="270" text-anchor="middle" font-size="9" fill="#2c3e50">{label}</text>')

                        svg_lines.append('</svg>')

                        html_parts.append("<div class='chart-container'>")
                        html_parts.append('\n'.join(svg_lines))
                        html_parts.append("</div>")
                        viz_count += 1
                    except Exception:
                        pass

                # Show info message only if we have visualizations
                if viz_count > 0 and not matplotlib_worked:
                    html_parts.append("<p style='font-size:10px;color:#7f8c8d;margin-top:10px;'><em>Charts rendered using SVG visualization engine.</em></p>")

                # Add chart interpretation summary
                if viz_count > 0 and chart_data:
                    interpretation = self._generate_chart_interpretation(chart_data, stats_results, scale_name)
                    if interpretation:
                        html_parts.append("<div class='interpretation-box' style='background:#f8f9fa;padding:15px;border-radius:6px;margin:15px 0;border-left:3px solid #3498db;'>")
                        html_parts.append(f"<strong>Key Finding:</strong> {interpretation}")
                        html_parts.append("</div>")

                # Statistical tests
                if len(conditions) >= 2 and "CONDITION" in df_analysis.columns:

                    # Determine which analyses were requested in pre-registration
                    prereg_analyses = self._get_prereg_requested_analyses(prereg_text)

                    html_parts.append("<h4>Statistical Tests</h4>")

                    # v1.2.9.1: groups with fewer than 2 scores cannot be tested -- say so, and say how many were compared
                    _excl_groups = stats_results.get("groups_excluded") or []
                    if _excl_groups:
                        _n_compared = len(stats_results.get("conditions") or [])
                        html_parts.append(
                            "<p style='font-size:0.9em;color:#b45309;'>Not included in the tests below (fewer than 2 scores): "
                            + ", ".join(f"{_html_lib.escape(str(g.get('condition', '')))} (n = {g.get('n', 0)})" for g in _excl_groups)
                            + f". The tests compare the remaining {_n_compared} condition{'s' if _n_compared != 1 else ''}.</p>"
                        )

                    # Two-group tests (t-test)
                    if "t_test" in stats_results:
                        t = stats_results["t_test"]
                        sig_class = "sig" if t["significant"] else ("marginal" if t.get("marginally_significant") else "nonsig")
                        prereg_badge = " <span style='background:#27ae60;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>PRE-REGISTERED</span>" if prereg_analyses.get("t_test") else " <span style='background:#95a5a6;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>ADDITIONAL</span>"
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>Independent Samples t-test:</strong>{prereg_badge}<br>")
                        html_parts.append(f"t = {_fmt_stat(t['statistic'])}, <span class='{sig_class}'>{_p_eq_html(t['p_value'])}</span>")
                        html_parts.append(f" <span style='color:#64748b;font-size:0.9em;'>(df = {t.get('df', '')}; difference = {_html_lib.escape(str(t.get('contrast', '')))})</span>")
                        # Add interpretation
                        interp = self._generate_stat_test_interpretation("t_test", stats_results, chart_data, scale_name)
                        if interp:
                            html_parts.append(f"<br><em style='color:#666;'>{interp}</em>")
                        html_parts.append("</div>")

                    # Effect size for two groups
                    if "cohens_d" in stats_results:
                        d = stats_results["cohens_d"]
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>Effect Size (Cohen's d):</strong> {_fmt_stat(d['value'])} <span style='color:#64748b;font-size:0.9em;'>({_html_lib.escape(str(d.get('contrast', '')))}; positive = the first-named condition scores higher)</span>")
                        # Add interpretation
                        interp = self._generate_stat_test_interpretation("effect_size", stats_results, chart_data, scale_name)
                        if interp:
                            html_parts.append(f"<br><em style='color:#666;'>{interp}</em>")
                        html_parts.append("</div>")

                    # ANOVA for 3+ groups
                    if "anova" in stats_results:
                        a = stats_results["anova"]
                        sig_class = "sig" if a["significant"] else ("marginal" if a.get("marginally_significant") else "nonsig")
                        prereg_badge = " <span style='background:#27ae60;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>PRE-REGISTERED</span>" if prereg_analyses.get("anova") else " <span style='background:#95a5a6;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>ADDITIONAL</span>"
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>One-way ANOVA:</strong>{prereg_badge}<br>")
                        html_parts.append(f"F = {_fmt_stat(a['f_statistic'])}, <span class='{sig_class}'>{_p_eq_html(a['p_value'])}</span>")
                        html_parts.append(f" <span style='color:#64748b;font-size:0.9em;'>(df = {a.get('df_between', '')}, {a.get('df_within', '')}; {a.get('num_groups', '')} conditions compared)</span>")
                        # Add interpretation
                        interp = self._generate_stat_test_interpretation("anova", stats_results, chart_data, scale_name)
                        if interp:
                            html_parts.append(f"<br><em style='color:#666;'>{interp}</em>")
                        html_parts.append("</div>")

                    # Effect size for ANOVA
                    if "eta_squared" in stats_results:
                        e = stats_results["eta_squared"]
                        html_parts.append("<div class='stat-box'>")
                        html_parts.append(f"<strong>Effect Size (η²):</strong> {e['value']:.4f}")
                        # Add interpretation
                        interp = self._generate_stat_test_interpretation("effect_size", stats_results, chart_data, scale_name)
                        if interp:
                            html_parts.append(f"<br><em style='color:#666;'>{interp}</em>")
                        html_parts.append("</div>")

                    # Assumption checks (simplified, no warnings)
                    assumption_notes = []
                    if "levene_test" in stats_results:
                        lev = stats_results["levene_test"]
                        if lev["homogeneous"]:
                            assumption_notes.append(f"Variance homogeneity: ✓ Met (Levene's {_p_eq_html(lev['p_value'])})")
                        else:
                            assumption_notes.append(f"Variance homogeneity: Welch's correction applied (Levene's {_p_eq_html(lev['p_value'])})")

                    if "normality_test" in stats_results:
                        sw = stats_results["normality_test"]
                        _sw_name = _html_lib.escape(str(sw.get("test_name", "Shapiro-Wilk")))
                        if sw["normal"]:
                            assumption_notes.append(f"Normality: ✓ Met ({_p_eq_html(sw['p_value'])}, {_sw_name} on the pooled scores)")
                        else:
                            assumption_notes.append(f"Normality: Non-parametric tests also reported ({_p_eq_html(sw['p_value'])}, {_sw_name} on the pooled scores)")

                    if assumption_notes:
                        html_parts.append("<div class='stat-box' style='background:#f8f9fa;'>")
                        html_parts.append("<strong>Assumption Checks:</strong> " + " | ".join(assumption_notes))
                        html_parts.append("</div>")

                    # Pairwise comparisons for 3+ groups
                    if "pairwise_comparisons" in stats_results and len(stats_results["pairwise_comparisons"]) > 0:
                        html_parts.append("<h4>Pairwise Comparisons</h4>")
                        html_parts.append("<table><tr><th>Comparison (first - second)</th><th>t</th><th>p (raw)</th><th>p (Holm)</th><th>Cohen's d</th><th>Significant (Holm)</th></tr>")

                        sig_pairs = []
                        marginal_pairs = []
                        largest_effect_pair = None
                        largest_d = 0

                        for comp in stats_results["pairwise_comparisons"]:
                            _is_marginal = comp.get("marginally_significant", False)
                            sig_class = "sig" if comp["significant"] else ("marginal" if _is_marginal else "nonsig")
                            sig_text = "Yes" if comp["significant"] else ("Marginal" if _is_marginal else "No")
                            html_parts.append(f"<tr><td>{comp['comparison']}</td><td>{_fmt_stat(comp['t_stat'])}</td><td>{_fmt_p_html(comp['p_value'])}</td><td class='{sig_class}'>{_fmt_p_html(comp.get('p_adjusted', comp['p_value']))}</td><td>{_fmt_stat(comp['cohens_d'])}</td><td class='{sig_class}'>{sig_text}</td></tr>")

                            if comp["significant"]:
                                sig_pairs.append(comp['comparison'])
                            elif _is_marginal:
                                marginal_pairs.append(comp['comparison'])
                            if abs(comp['cohens_d']) > largest_d:
                                largest_d = abs(comp['cohens_d'])
                                largest_effect_pair = comp

                        html_parts.append("</table>")
                        html_parts.append(
                            f"<p style='font-size:0.85em;color:#64748b;'>t and d are first minus second condition in the order of the descriptive table "
                            f"(positive = the first-named condition scores higher). p (raw) is uncorrected; p (Holm) is adjusted for the "
                            f"{len(stats_results['pairwise_comparisons'])} comparisons made for this DV and decides the Significant column.</p>"
                        )

                        # Interpretation instead of warning
                        html_parts.append("<div class='interpretation-box' style='background:#f8f9fa;padding:15px;border-radius:6px;margin:15px 0;border-left:3px solid #3498db;'>")
                        html_parts.append("<strong>Key Finding:</strong> ")
                        if sig_pairs:
                            html_parts.append(f"{len(sig_pairs)} of {len(stats_results['pairwise_comparisons'])} pairwise comparisons reached statistical significance after Holm adjustment. ")
                            if largest_effect_pair:
                                html_parts.append(f"The largest effect was between {largest_effect_pair['comparison']} (d = {_fmt_stat(largest_effect_pair['cohens_d'], 2)}).")
                        elif marginal_pairs:
                            html_parts.append(f"No pairwise comparisons reached conventional significance (Holm-adjusted p &lt; .05), but {len(marginal_pairs)} showed marginally significant differences (Holm-adjusted p &lt; .10): {', '.join(marginal_pairs)}.")
                        else:
                            html_parts.append("No pairwise comparisons reached statistical significance after Holm adjustment, suggesting the overall ANOVA effect may be driven by subtle differences across multiple groups rather than any single pair.")
                        html_parts.append("</div>")

                        # Forest plot for effect sizes
                        forest_img = self._create_effect_size_forest_plot(
                            stats_results["pairwise_comparisons"],
                            f"{scale_name}: Effect Sizes (Cohen's d)"
                        )
                        if forest_img:
                            html_parts.append("<div class='chart-container'>")
                            html_parts.append(f"<img src='data:image/png;base64,{forest_img}' alt='Forest plot'>")
                            html_parts.append("</div>")

                    # Regression analysis with control variables (only show if successful)
                    prereg_info = self._parse_prereg_hypotheses(prereg_text) if prereg_text else {}
                    prereg_controls = prereg_info.get("control_variables", [])

                    reg_results = self._run_regression_analysis(
                        df_analysis, "_composite", "CONDITION",
                        include_controls=True,
                        prereg_controls=prereg_controls,
                        condition_order=list(conditions),
                        exclude_columns=sorted(_dv_item_columns),
                        condition_labels=_cond_labels,
                    )
                    reg_results = _finite_results_only(reg_results)

                    # Only show regression if it worked (no warnings for failures)
                    if "error" not in reg_results and "model_fit" in reg_results:
                        prereg_badge = " <span style='background:#27ae60;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>PRE-REGISTERED</span>" if prereg_analyses.get("regression") else " <span style='background:#95a5a6;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>ADDITIONAL</span>"
                        html_parts.append(f"<h4>Regression Analysis (with Controls){prereg_badge}</h4>")

                        controls_used = reg_results.get("controls_included", [])
                        if controls_used:
                            html_parts.append(f"<div class='stat-box'><strong>Control variables included:</strong> {', '.join(controls_used)}</div>")
                        # v1.2.9.1: say which level every dummy-coded factor is measured against
                        _ref_levels = reg_results.get("reference_levels") or {}
                        if _ref_levels:
                            html_parts.append(
                                "<div class='stat-box'><strong>Reference levels:</strong> "
                                + "; ".join(f"{_html_lib.escape(str(k))} = {_html_lib.escape(str(v))}" for k, v in _ref_levels.items())
                                + ". Each condition coefficient is that condition's difference from the reference condition, "
                                  "holding the controls constant.</div>"
                            )

                        html_parts.append("<div class='stat-box'>")
                        fit = reg_results["model_fit"]
                        html_parts.append(f"<strong>Model Fit:</strong> R² = {fit['r_squared']:.4f}, Adj. R² = {fit['adj_r_squared']:.4f} (N = {fit['n']})<br>")

                        if "f_test" in reg_results:
                            f = reg_results["f_test"]
                            sig_class = "sig" if f["significant"] else ("marginal" if f.get("marginally_significant") else "nonsig")
                            html_parts.append(f"<strong>F-test:</strong> F = {_fmt_stat(f['f_statistic'])}, <span class='{sig_class}'>{_p_eq_html(f['p_value'])}</span> (df = {fit.get('n_predictors', '')}, {fit['df_residual']})<br>")

                        # Coefficients table
                        if "coefficients" in reg_results:
                            html_parts.append("<br><strong>Coefficients:</strong>")
                            html_parts.append("<table><tr><th>Predictor</th><th>B</th><th>SE</th><th>t</th><th>p</th></tr>")
                            for pred, coef in reg_results["coefficients"].items():
                                sig_class = "sig" if coef.get("significant") else ("marginal" if coef.get("marginally_significant") else "nonsig")
                                html_parts.append(f"<tr><td>{pred}</td><td>{coef['estimate']:.3f}</td><td>{coef['std_error']:.3f}</td><td>{_fmt_stat(coef['t_stat'])}</td><td class='{sig_class}'>{_fmt_p_html(coef['p_value'])}</td></tr>")
                            html_parts.append("</table>")

                        # Regression interpretation
                        html_parts.append("<br><em style='color:#666;'>")
                        html_parts.append(f"The regression model explains {fit['r_squared']*100:.1f}% of variance in {scale_name}. ")
                        if "f_test" in reg_results and reg_results["f_test"]["significant"]:
                            html_parts.append("The overall model is statistically significant.")
                        html_parts.append("</em>")
                        html_parts.append("</div>")

                    # Factorial ANOVA for 2x2+ designs
                    factors = metadata.get("factors", [])
                    if len(factors) >= 2 and len(conditions) >= 4:
                        factorial_results = self._run_factorial_anova(df_analysis, "_composite", factors, "CONDITION")
                        factorial_results = _finite_results_only(factorial_results)

                        # v1.2.9.1: render whenever the analysis produced its terms ("error" is reserved for
                        # failures, the residual row lives under "residual"); when it cannot run, say why
                        if "error" not in factorial_results and factorial_results.get("terms"):
                            prereg_badge = " <span style='background:#27ae60;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>PRE-REGISTERED</span>" if prereg_analyses.get("factorial") else " <span style='background:#95a5a6;color:white;padding:2px 6px;border-radius:3px;font-size:10px;'>ADDITIONAL</span>"
                            html_parts.append(f"<h4>Factorial ANOVA (Main Effects & Interaction){prereg_badge}</h4>")
                            if factorial_results.get("design_note"):
                                html_parts.append(f"<p style='font-size:0.9em;color:#64748b;'>{_html_lib.escape(str(factorial_results['design_note']))}</p>")
                            # ANOVA summary table (one row per term: main effects first, then interactions)
                            html_parts.append("<table><tr><th>Source</th><th>SS</th><th>df</th><th>MS</th><th>F</th><th>p</th><th>η²<sub>p</sub></th></tr>")
                            for term in factorial_results["terms"]:
                                sig_class = "sig" if term["significant"] else ("marginal" if term.get("marginally_significant") else "nonsig")
                                html_parts.append(f"<tr><td><strong>{_html_lib.escape(str(term['term']))}</strong></td><td>{term['ss']:.2f}</td><td>{term['df']}</td><td>{term['ms']:.2f}</td><td>{_fmt_stat(term['f_statistic'])}</td><td class='{sig_class}'>{_fmt_p_html(term['p_value'])}</td><td>{term['partial_eta_squared']:.4f}</td></tr>")

                            # Residual (error) row
                            resid = factorial_results.get("residual")
                            if isinstance(resid, dict):
                                html_parts.append(f"<tr><td>Residual</td><td>{resid['ss']:.2f}</td><td>{resid['df']}</td><td>{resid['ms']:.2f}</td><td>-</td><td>-</td><td>-</td></tr>")

                            html_parts.append("</table>")

                            # Interpretation with key finding box
                            html_parts.append("<div class='interpretation-box' style='background:#f8f9fa;padding:15px;border-radius:6px;margin:15px 0;border-left:3px solid #3498db;'>")
                            html_parts.append("<strong>Key Finding:</strong> ")
                            findings = []
                            for term in factorial_results["terms"]:
                                if term["order"] == 1:
                                    _what = f"main effect of <strong>{_html_lib.escape(str(term['term']))}</strong>"
                                    if term["significant"]:
                                        findings.append(f"significant {_what} ({term['interpretation']} effect)")
                                    elif term.get("marginally_significant"):
                                        findings.append(f"marginally significant {_what} (p &lt; .10)")
                                else:
                                    _names = " and ".join(_html_lib.escape(str(n)) for n in term["factors"])
                                    if term["significant"]:
                                        findings.append(f"<strong>significant interaction</strong> between {_names} ({term['interpretation']} effect)")
                                    elif term.get("marginally_significant"):
                                        findings.append(f"<strong>marginally significant interaction</strong> between {_names} (p &lt; .10)")

                            if findings:
                                html_parts.append("The factorial analysis revealed " + ", ".join(findings) + ".")
                            else:
                                html_parts.append("No significant main effects or interactions were detected in this factorial design.")
                            html_parts.append("</div>")

                            # Cell means table
                            if "cell_statistics" in factorial_results:
                                html_parts.append("<strong>Cell Means:</strong>")
                                html_parts.append("<table><tr><th>Cell</th><th>N</th><th>Mean</th><th>SD</th></tr>")
                                for cell, stats in factorial_results["cell_statistics"].items():
                                    html_parts.append(f"<tr><td>{_html_lib.escape(str(cell))}</td><td>{stats['n']}</td><td>{stats['mean']:.3f}</td><td>{stats['std']:.3f}</td></tr>")
                                html_parts.append("</table>")

                            # Interaction plot (two analysed factors only; levels come from the exact parse)
                            _analysed = factorial_results.get("factors_analysed") or []
                            if len(_analysed) == 2 and factorial_results.get("level_map"):
                                f1_name = _analysed[0]["name"]
                                f2_name = _analysed[1]["name"]
                                _lvl_map = factorial_results["level_map"]

                                # Create factor columns for plotting
                                df_plot = df_analysis.copy()
                                df_plot["_f1"] = df_plot["CONDITION"].map(lambda x, _m=_lvl_map: (_m.get(x) or (None, None))[0])
                                df_plot["_f2"] = df_plot["CONDITION"].map(lambda x, _m=_lvl_map: (_m.get(x) or (None, None))[1])
                                df_plot = df_plot.dropna(subset=["_f1", "_f2"])

                                interaction_img = self._create_interaction_plot(
                                    df_plot, "_composite", "_f1", "_f2",
                                    f1_name, f2_name,
                                    f"{scale_name}: Interaction Plot"
                                )
                                if interaction_img:
                                    html_parts.append("<div class='chart-container'>")
                                    html_parts.append(f"<img src='data:image/png;base64,{interaction_img}' alt='Interaction plot'>")
                                    html_parts.append("</div>")
                        elif factorial_results.get("error") and not factorial_results.get("single_factor"):
                            html_parts.append(
                                f"<p style='font-size:0.9em;color:#64748b;'><em>Factorial ANOVA ({_html_lib.escape(' x '.join(str(f.get('name', 'factor')) for f in factors[:3]))}) "
                                f"could not be computed: {_html_lib.escape(str(factorial_results['error']).replace('Factorial ANOVA not run: ', ''))}.</em></p>"
                            )

                # Track this scale's results for executive summary
                all_scale_results.append({
                    "scale_name": scale_name,
                    "chart_data": chart_data,
                    "stats_results": stats_results
                })

    def _html_persona(self, ctx: Any) -> None:
        """Sections 4-5: persona distribution and response styles, then the categorical analysis."""
        html_parts, df_clean, metadata, conditions, n_total = ctx.out, ctx.df_clean, ctx.metadata, ctx.conditions, ctx.n_total
        _cond_label = ctx.cond_label  # v1.2.9.1: one collision-free label per condition

        # Chi-squared test for categorical associations
        if "CONDITION" in df_clean.columns and "Gender" in df_clean.columns:
            html_parts.append("<a id='persona-analysis'></a>")
            html_parts.append("<h2>4. Persona Analysis &amp; Response Styles</h2>")

            # ── Persona Distribution ──────────────────────────────────────
            persona_dist_raw = metadata.get("persona_distribution", {}) or {}
            persona_dist = _extract_persona_proportions(persona_dist_raw)
            personas_used = metadata.get("personas_used", [])

            if persona_dist:
                html_parts.append("<div class='stat-box'>")
                html_parts.append("<h3>Simulated Participant Personas</h3>")
                html_parts.append("<p>Each simulated participant was assigned a response style persona based on decades of survey methodology research. "
                                  "These personas reflect patterns observed in real survey respondents, producing data with realistic "
                                  "statistical properties (varying attention, response styles, and biases).</p>")
                html_parts.append("</div>")

                # Build persona info lookup
                _persona_info = {
                    "engaged responder": ("Thoughtful, attentive participant", "High attention, full scale range, consistent with attitudes"),
                    "engaged": ("Thoughtful, attentive participant", "High attention, full scale range, consistent with attitudes"),
                    "satisficer": ("Minimally effortful responder", "Gravitates to middle options, faster completion, may skip reading"),
                    "extreme responder": ("Uses scale endpoints frequently", "Strong opinions, uses 1s and 7s, high within-scale variance"),
                    "extreme": ("Uses scale endpoints frequently", "Strong opinions, uses 1s and 7s, high within-scale variance"),
                    "acquiescent": ("Agreement bias responder", "Tends to agree regardless of content, inflated positive responses"),
                    "skeptic": ("Disagreement bias responder", "Tends to disagree or rate negatively, lower mean responses"),
                    "random": ("Inconsistent, inattentive responder", "High variance, fails attention checks, no clear pattern"),
                    "careless": ("Pattern-based responder", "Straight-lining, very fast completion, flagged for exclusion"),
                    "careful responder": ("Highly attentive, methodical", "Longer completion, passes all attention checks, low variance"),
                    "moderate responder": ("Avoids extreme responses", "Uses middle portion of scale, rarely selects endpoints"),
                }

                html_parts.append("<h4>Persona Distribution</h4>")
                html_parts.append("<table>")
                html_parts.append("<tr><th>Persona Type</th><th>Description</th><th>Behavioral Pattern</th><th>Share</th><th>~Count</th></tr>")
                _n_total_pers = metadata.get("sample_size", n_total)
                for persona, share in sorted(persona_dist.items(), key=lambda x: -_safe_float(x[1])):
                    share_f = _safe_float(share)
                    pct = share_f * 100 if share_f <= 1 else share_f
                    share_val = share_f if share_f <= 1 else share_f / 100
                    count = int(round(_n_total_pers * share_val))
                    pkey = persona.lower()
                    info = _persona_info.get(pkey, ("Standard response pattern", "Typical survey behavior"))
                    html_parts.append(
                        f"<tr><td><strong>{persona.title()}</strong></td>"
                        f"<td>{info[0]}</td>"
                        f"<td><em>{info[1]}</em></td>"
                        f"<td>{pct:.1f}%</td>"
                        f"<td>{count}</td></tr>"
                    )
                html_parts.append("</table>")

                # Scientific references for personas
                html_parts.append("<div class='stat-box' style='font-size:0.9em;'>")
                html_parts.append("<strong>Scientific Basis:</strong> Engaged responders based on Krosnick's (1991) 'optimizers'; "
                                  "satisficers per Krosnick (1991); extreme responders per Greenleaf (1992); "
                                  "acquiescent responders per Billiet &amp; McClendon (2000); "
                                  "careless responders per Meade &amp; Craig (2012).")
                html_parts.append("</div>")

            # ── Persona by Condition ──────────────────────────────────────
            persona_by_cond = metadata.get("persona_by_condition", {})
            pbc_counts = persona_by_cond.get("counts", {}) if isinstance(persona_by_cond, dict) else {}
            if pbc_counts and conditions:
                html_parts.append("<h4>Persona Distribution by Condition</h4>")
                html_parts.append("<p>This table shows how personas were distributed across experimental conditions, "
                                  "verifying that response style composition is balanced across groups.</p>")

                # Collect all persona types across conditions
                all_ptypes = set()
                for cond_dict in pbc_counts.values():
                    if isinstance(cond_dict, dict):
                        all_ptypes.update(cond_dict.keys())
                all_ptypes_sorted = sorted(all_ptypes)

                if all_ptypes_sorted:
                    html_parts.append("<table>")
                    html_parts.append("<tr><th>Persona</th>")
                    for cond in conditions:
                        clean_c = _cond_label(cond)
                        html_parts.append(f"<th>{clean_c}</th>")
                    html_parts.append("</tr>")

                    for ptype in all_ptypes_sorted:
                        html_parts.append(f"<tr><td><strong>{ptype.title()}</strong></td>")
                        for cond in conditions:
                            cond_dict = pbc_counts.get(cond, {})
                            count = cond_dict.get(ptype, 0) if isinstance(cond_dict, dict) else 0
                            html_parts.append(f"<td>{count}</td>")
                        html_parts.append("</tr>")
                    html_parts.append("</table>")

            # ── Trait Averages (Overall) ──────────────────────────────────
            trait_avg_overall = metadata.get("trait_averages_overall", {})
            if trait_avg_overall and isinstance(trait_avg_overall, dict):
                html_parts.append("<h4>Simulated Personality Profile (Sample Averages)</h4>")
                html_parts.append("<p>Average trait values across all simulated participants, derived from persona assignments.</p>")
                html_parts.append("<div class='metric-grid'>")
                for trait, val in sorted(trait_avg_overall.items()):
                    trait_display = trait.replace("_", " ").title()
                    html_parts.append(
                        f"<div class='metric-card' style='background:linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: #1a1a2e;'>"
                        f"<div class='metric-value'>{val:.2f}</div>"
                        f"<div class='metric-label'>{trait_display}</div></div>"
                    )
                html_parts.append("</div>")

            # ── Trait Averages by Condition ────────────────────────────────
            trait_avg_by_cond = metadata.get("trait_averages_by_condition", {})
            if trait_avg_by_cond and isinstance(trait_avg_by_cond, dict) and conditions:
                # Check if any condition has trait data
                has_traits = any(
                    isinstance(v, dict) and len(v) > 0
                    for v in trait_avg_by_cond.values()
                )
                if has_traits:
                    html_parts.append("<h4>Personality Profiles by Condition</h4>")
                    html_parts.append("<p>Average trait values per condition. Balanced profiles across conditions indicates "
                                      "that persona assignment did not confound the experimental manipulation.</p>")

                    # Collect all traits
                    all_traits = set()
                    for cond_traits in trait_avg_by_cond.values():
                        if isinstance(cond_traits, dict):
                            all_traits.update(cond_traits.keys())
                    all_traits_sorted = sorted(all_traits)

                    if all_traits_sorted:
                        html_parts.append("<table>")
                        html_parts.append("<tr><th>Trait</th>")
                        for cond in conditions:
                            clean_c = _cond_label(cond)
                            html_parts.append(f"<th>{clean_c}</th>")
                        html_parts.append("</tr>")
                        for trait in all_traits_sorted:
                            trait_display = trait.replace("_", " ").title()
                            html_parts.append(f"<tr><td>{trait_display}</td>")
                            for cond in conditions:
                                cond_traits = trait_avg_by_cond.get(cond, {})
                                val = cond_traits.get(trait, 0) if isinstance(cond_traits, dict) else 0
                                html_parts.append(f"<td>{val:.3f}</td>")
                            html_parts.append("</tr>")
                        html_parts.append("</table>")

            # ── Validation Issues Corrected ────────────────────────────────
            validation_corrected = metadata.get("validation_issues_corrected", 0)
            if validation_corrected and validation_corrected > 0:
                html_parts.append("<div class='stat-box'>")
                html_parts.append(f"<strong>Data Validation:</strong> {validation_corrected} response value(s) were "
                                  f"automatically corrected during generation to stay within valid scale ranges.")
                html_parts.append("</div>")

            html_parts.append("<a id='categorical-analysis'></a>")
            html_parts.append("<h2>5. Categorical Analysis</h2>")
            html_parts.append("<h3>Condition × Gender</h3>")

            try:
                # Rows follow the report's one condition order and the collision-free labels
                _row_conds = _order_conditions(df_clean["CONDITION"].dropna().unique().tolist(), list(conditions))
                _row_labels = [_cond_label(c) for c in _row_conds]
                _label_of = dict(zip(_row_conds, _row_labels))
                contingency = pd.crosstab(df_clean["CONDITION"].map(_label_of), df_clean["Gender"])
                contingency = contingency.reindex([lab for lab in _row_labels if lab in contingency.index])

                # Chi-squared test (scipy or exact numpy implementation; Yates-corrected for 2x2) with validity checks
                assoc = _association_test(contingency)

                if assoc.get("status") == "single_condition":
                    html_parts.append("<div class='stat-box'>")
                    html_parts.append("<strong>Chi-squared test:</strong> not run &mdash; single condition: no between-condition comparison.")
                    html_parts.append("</div>")
                elif assoc.get("status") == "single_category":
                    html_parts.append("<div class='stat-box'>")
                    html_parts.append("<strong>Chi-squared test:</strong> not run &mdash; only one gender category is present in the data.")
                    html_parts.append("</div>")
                else:
                    chi2, p, dof = assoc["chi2"], assoc["p_value"], assoc["dof"]
                    _chi2_sig = self._p_significance(float(p))
                    sig_class = "sig" if _chi2_sig["significant"] else ("marginal" if _chi2_sig["marginally_significant"] else "nonsig")
                    html_parts.append("<div class='stat-box'>")
                    html_parts.append(f"<strong>Chi-squared test:</strong> χ² = {chi2:.3f}, df = {dof}, ")
                    html_parts.append(f"<span class='{sig_class}'>{_p_eq_html(p)}</span>")
                    html_parts.append(f" <span style='color:#64748b;font-size:0.9em;'>(N = {assoc['n']}{'; Yates continuity correction for the 2x2 table' if assoc.get('yates') else ''})</span>")
                    if not assoc.get("valid", True):
                        html_parts.append(
                            f"<br><em>Too few expected counts for the chi-square test: {assoc['share_expected_below_5']:.0%} of the cells have an expected count below 5 "
                            f"(the usual rule is at most 20% and none below 1), so this p-value is unreliable and says nothing about whether randomization worked."
                            + (f" Fisher's exact test for this 2x2 table: {_p_eq_html(assoc['fisher_p'])}." if assoc.get("fisher_p") is not None else "")
                            + "</em>"
                        )
                    elif p >= 0.05:
                        html_parts.append("<br><em>No significant association between condition and gender (randomization appears successful)</em>")
                    html_parts.append("</div>")

                # Contingency table
                html_parts.append("<table><tr><th>Condition</th>")
                for col in contingency.columns:
                    html_parts.append(f"<th>{col}</th>")
                html_parts.append("</tr>")
                for idx, row in contingency.iterrows():
                    html_parts.append(f"<tr><td>{idx}</td>")
                    for val in row:
                        html_parts.append(f"<td>{val}</td>")
                    html_parts.append("</tr>")
                html_parts.append("</table>")
            except Exception as chi_err:
                logger.warning("Condition x Gender association test skipped: %s", chi_err)

        html_parts.append("<a href='#top' class='back-to-top'>Back to top</a>")

    def _html_exec_summary(self, ctx: Any) -> None:
        """Section 2: build the executive summary and put it into its reserved place."""
        html_parts, all_scale_results, prereg_text = ctx.out, ctx.all_scale_results, ctx.prereg_text
        n_total, conditions, exec_summary_index = ctx.n_total, ctx.conditions, ctx.exec_summary_index
        _cond_label = ctx.cond_label  # v1.2.9.1: one collision-free label per condition

        # v1.3.4: Insert executive summary into its placeholder position (before statistical analysis)
        if all_scale_results:
            exec_summary = self._generate_executive_summary(
                all_scale_results,
                prereg_text,
                n_total,
                [_cond_label(c) for c in conditions]
            )
            html_parts[exec_summary_index] = exec_summary
        else:
            html_parts[exec_summary_index] = ""  # Remove placeholder if no results

    def _html_effect_verification(self, ctx: Any) -> None:
        """Section 6: configured versus observed effect sizes."""
        html_parts, metadata = ctx.out, ctx.metadata

        # ── Observed vs Configured Effect Sizes ─────────────────────────
        obs_effects = metadata.get("effect_sizes_observed", [])
        cfg_effects = metadata.get("effect_sizes_configured", [])
        if obs_effects or cfg_effects:
            html_parts.append("<a id='effect-verification'></a>")
            html_parts.append("<h2>6. Effect Size Verification</h2>")
            if cfg_effects and any(e.get("cohens_d", 0) > 0 for e in cfg_effects):
                html_parts.append("<h3>Configured Effects</h3>")
                html_parts.append("<table><tr><th>DV</th><th>Target d</th><th>Direction</th><th>Comparison</th></tr>")
                for eff in cfg_effects:
                    d_val = eff.get("cohens_d", 0)
                    if d_val > 0:
                        html_parts.append(
                            f"<tr><td>{eff.get('variable', '')}</td>"
                            f"<td>{d_val:.2f}</td>"
                            f"<td>{eff.get('direction', '')}</td>"
                            f"<td>{eff.get('level_high', '')} vs {eff.get('level_low', '')}</td></tr>"
                        )
                html_parts.append("</table>")
            if obs_effects:
                html_parts.append("<h3>Observed Effects in Generated Data</h3>")
                html_parts.append("<table><tr><th>DV</th><th>Observed d</th><th>Comparison</th><th>Interpretation</th></tr>")
                for eff in obs_effects:
                    d_val = abs(_safe_float(eff.get("cohens_d", eff.get("d", 0))))
                    if d_val < 0.2:
                        interp = "Negligible"
                    elif d_val < 0.5:
                        interp = "Small"
                    elif d_val < 0.8:
                        interp = "Medium"
                    else:
                        interp = "Large"
                    var_name = eff.get("variable", eff.get("scale", ""))
                    comp = f"{eff.get('condition_high', '')} vs {eff.get('condition_low', '')}"
                    html_parts.append(
                        f"<tr><td>{var_name}</td><td>{d_val:.3f}</td><td>{comp}</td><td>{interp}</td></tr>"
                    )
                html_parts.append("</table>")

    def _html_exclusions(self, ctx: Any) -> None:
        """Section 7: exclusion flags."""
        html_parts, metadata = ctx.out, ctx.metadata

        # ── Exclusion Summary ─────────────────────────────────────────
        excl = metadata.get("exclusion_summary", {})
        if excl:
            html_parts.append("<a id='data-quality'></a>")
            html_parts.append("<h2>7. Data Quality &amp; Exclusions</h2>")
            html_parts.append("<div class='metric-grid'>")
            html_parts.append(f"<div class='metric-card' style='background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);'>"
                              f"<div class='metric-value'>{excl.get('flagged_speed', 0)}</div>"
                              f"<div class='metric-label'>Speed Flags</div></div>")
            html_parts.append(f"<div class='metric-card' style='background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);'>"
                              f"<div class='metric-value'>{excl.get('flagged_attention', 0)}</div>"
                              f"<div class='metric-label'>Attention Flags</div></div>")
            html_parts.append(f"<div class='metric-card' style='background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);'>"
                              f"<div class='metric-value'>{excl.get('flagged_straightline', 0)}</div>"
                              f"<div class='metric-label'>Straight-line Flags</div></div>")
            html_parts.append(f"<div class='metric-card' style='background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);'>"
                              f"<div class='metric-value'>{excl.get('total_excluded', 0)}</div>"
                              f"<div class='metric-label'>Total Excluded</div></div>")
            html_parts.append("</div>")

    def _html_generation_warnings(self, ctx: Any) -> None:
        """Warnings the engine recorded while generating the data."""
        html_parts, metadata = ctx.out, ctx.metadata

        # ── Generation Warnings ────────────────────────────────────────
        gen_warnings = metadata.get("generation_warnings", [])
        if gen_warnings:
            html_parts.append("<div class='warning-box' style='border-left:4px solid #f59e0b;background:#fffbeb;padding:15px;margin:20px 0;'>")
            html_parts.append("<strong>Generation Warnings:</strong><ul>")
            for gw in gen_warnings:
                html_parts.append(f"<li>{gw}</li>")
            html_parts.append("</ul></div>")

    def _html_methodology(self, ctx: Any) -> None:
        """Section 8: notes for instructors and methodology."""
        html_parts = ctx.out

        # Footer - Notes for Instructors
        html_parts.append("<a id='methodology'></a>")
        html_parts.append("<h2>8. Instructor Notes &amp; Methodology</h2>")
        html_parts.append("<div class='warning-box'>")
        html_parts.append("<strong>Important: This is simulated data.</strong> Results demonstrate what the analysis pipeline will produce. "
                          "Students should practice these analyses independently and may get similar (but not identical) results due to random variation.")
        html_parts.append("</div>")
        html_parts.append("<div class='stat-box'>")
        html_parts.append("<h3>About This Simulation</h3>")
        html_parts.append("<ul>")
        html_parts.append("<li><strong>Persona-based generation:</strong> Each participant is assigned a response style persona "
                          "(engaged, satisficer, extreme, acquiescent, careless, etc.) based on survey methodology research.</li>")
        html_parts.append("<li><strong>Domain-specific knowledge:</strong> Responses draw on 225+ research domains across 33 categories, "
                          "ensuring contextually appropriate language and attitudes.</li>")
        html_parts.append("<li><strong>Effect size calibration:</strong> Treatment effects are calibrated to target Cohen's d values, "
                          "applied at the individual response level with validation checks.</li>")
        html_parts.append("<li><strong>Scale reliability:</strong> Multi-item scales use a factor model (Response = λ·Factor + √(1-λ²)·Error) "
                          "producing realistic Cronbach's alpha values (typically 0.75-0.90).</li>")
        html_parts.append("<li><strong>Reproducibility:</strong> Simulations are seeded for exact reproducibility. "
                          "The same seed + parameters produce identical datasets.</li>")
        html_parts.append("</ul>")
        html_parts.append("<h3>Scientific References</h3>")
        html_parts.append("<ol style='font-size:0.9em;'>")
        html_parts.append("<li>Krosnick, J. A. (1991). Response strategies for coping with the cognitive demands of attitude measures. <em>Applied Cognitive Psychology, 5</em>, 213-236.</li>")
        html_parts.append("<li>Greenleaf, E. A. (1992). Measuring extreme response style. <em>Public Opinion Quarterly, 56</em>, 328-351.</li>")
        html_parts.append("<li>Billiet, J. B., &amp; McClendon, M. J. (2000). Modeling acquiescence in measurement models. <em>Structural Equation Modeling, 7</em>, 608-628.</li>")
        html_parts.append("<li>Meade, A. W., &amp; Craig, S. B. (2012). Identifying careless responses in survey data. <em>Psychological Methods, 17</em>, 437-455.</li>")
        html_parts.append("<li>Cohen, J. (1988). <em>Statistical power analysis for the behavioral sciences</em>. Lawrence Erlbaum.</li>")
        html_parts.append("<li>Richard, F. D., Bond, C. F., &amp; Stokes-Zoota, J. J. (2003). One hundred years of social psychology quantitatively described. <em>Review of General Psychology, 7</em>, 331-363.</li>")
        html_parts.append("</ol>")
        html_parts.append("</div>")

        html_parts.append("<a href='#top' class='back-to-top'>Back to top</a>")

    def _html_data_dictionary(self, ctx: Any) -> None:
        """Section 9: description of every output column."""
        html_parts, metadata = ctx.out, ctx.metadata

        # === DATA DICTIONARY (placed at the very bottom for reference) ===
        _col_descs_html = metadata.get("column_descriptions", {})
        if _col_descs_html and isinstance(_col_descs_html, dict):
            html_parts.append("<a id='data-dictionary'></a>")
            html_parts.append("<h2>9. Data Dictionary</h2>")
            html_parts.append("<p>Complete reference for every column in the output CSV:</p>")
            html_parts.append("<table><tr><th>Column</th><th>Description</th></tr>")
            for _cd_col, _cd_desc in _col_descs_html.items():
                _cd_col_safe = str(_cd_col).replace("<", "&lt;").replace(">", "&gt;")
                _cd_desc_safe = str(_cd_desc).replace("<", "&lt;").replace(">", "&gt;")
                html_parts.append(f"<tr><td><code>{_cd_col_safe}</code></td><td>{_cd_desc_safe}</td></tr>")
            html_parts.append("</table>")
            html_parts.append("<a href='#top' class='back-to-top'>Back to top</a>")

    def _html_footer(self, ctx: Any) -> None:
        """Closing line and the end of the report container."""
        html_parts = ctx.out

        html_parts.append(f"<p style='color:#999;font-size:0.9em;margin-top:30px;text-align:center;'>"
                          f"Generated by Behavioral Experiment Simulation Tool v{__version__} "
                          f"&middot; Software by Dr. Eugen Dimant &middot; PolyForm Noncommercial 1.0.0</p>")
        html_parts.append("</div></div></body></html>")  # close report-container + page-wrapper
