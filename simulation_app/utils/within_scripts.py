"""Data-preparation scripts for repeated-measures runs (R, Python, Julia, SPSS, Stata) and the codebook note.

Like the between-subjects scripts these only PREPARE the data: they read ``Simulated_Data.csv`` (one row
per participant, one column per measure and condition), recode reverse-keyed items, compute one composite
per measure and condition, label the factors and apply the optional exclusion step. They do not run the
statistical analysis; the closing comments only point to the long-format file and the usual tests.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

__all__ = ["script_for", "wide_spec", "codebook_addendum", "methods_sentence"]


def _txt(x: Any) -> str:
    return re.sub(r"[\r\n]+", " ", str(x if x is not None else "")).strip()


def wide_spec(engine: Any, df: pd.DataFrame) -> Dict[str, Any]:
    """What the scripts need: cells, groups and, per measure and cell, the item columns that exist in ``df``."""
    md = (getattr(engine, "_within_result", None) or {}).get("metadata") or {}
    design = md.get("design") or {}
    cells = design.get("cells") or []
    scales = {}
    for s in md.get("scales") or []:
        if isinstance(s, dict):
            from .instructor_report import _report_clean_column_name
            scales[_report_clean_column_name(str(s.get("variable_name") or s.get("name") or ""))] = s
    blocks: List[Dict[str, Any]] = []
    for dv, per in (design.get("wide_columns") or {}).items():
        sc = scales.get(dv, {})
        rev_items = sorted(int(x) for x in (sc.get("reverse_items") or []) if str(x).lstrip("-").isdigit())
        lo = float(sc.get("scale_min", 1))
        hi = float(sc.get("scale_max", sc.get("scale_points", 7)))
        flip = lo + hi
        for cell in cells:
            info = per.get(cell["label"]) or {}
            items = [c for c in (info.get("items") or []) if c in df.columns]
            if not items:
                continue
            reverse_cols = [items[i - 1] for i in rev_items if 1 <= i <= len(items)] if len(items) > 1 else []
            blocks.append({"dv": dv, "cell": cell["label"], "slug": cell["slug"], "items": items,
                           "reverse": reverse_cols, "flip": int(flip) if float(flip).is_integer() else flip,
                           "composite": f"{dv}_{cell['slug']}_composite", "raw": f"{dv} ({cell['label']})"})
    return {"design": design, "cells": cells, "blocks": blocks, "mixed": design.get("type") == "mixed",
            "groups": design.get("group_labels") or [], "has_gender": "Gender" in df.columns}


def _layout_notes(spec: Dict[str, Any], comment: str) -> List[str]:
    d = spec["design"]
    out = [
        f"{comment} DESIGN: {d.get('type', 'within')}-subjects. Every row is ONE participant; each measure appears once per condition",
        f"{comment}   as <Measure>_<Condition>_<item>. Simulated_Data_Long.csv holds the same data with one row per participant and condition.",
        f"{comment}   Order = presentation order the participant saw; Position_<Condition> = position of that condition (1 = first).",
        f"{comment}   Conditions: " + ", ".join(c["label"] for c in spec["cells"]),
        f"{comment}   Participants who dropped out have missing values for the conditions presented after their last finished one.",
        f"{comment} This script prepares the data only; it does not run any analysis.",
    ]
    return out


def _header(engine: Any, title: str, comment: str) -> List[str]:
    return [
        f"{comment} ============================================================",
        f"{comment} {title} - {_txt(engine.study_title)}",
        f"{comment} Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{comment} Run ID: {_txt(engine.run_id)}",
        f"{comment} ============================================================",
        "",
    ]


def _r(engine: Any, df: pd.DataFrame, spec: Dict[str, Any]) -> str:
    q = lambda x: '"' + _txt(x).replace("\\", "\\\\").replace('"', '\\"') + '"'  # noqa: E731
    lines = _header(engine, "R Data Preparation Script", "#") + _layout_notes(spec, "#") + [
        "", "suppressPackageStartupMessages({", "  library(readr)", "  library(dplyr)", "})", "",
        "# Load the wide file (Qualtrics-style export; one header row)",
        'data <- read_csv("Simulated_Data.csv", show_col_types = FALSE)', "",
        "# Presentation order as a factor",
        "if (\"Order\" %in% names(data)) data$Order <- factor(data$Order)", ""]
    if spec["mixed"] and spec["groups"]:
        lines += ["# Between-subjects group", "data$CONDITION <- factor(data$CONDITION, levels = c(" + ", ".join(q(g) for g in spec["groups"]) + "))", ""]
    if spec["has_gender"]:
        lines += ["data$Gender <- factor(data$Gender)", ""]
    for b in spec["blocks"]:
        if b["reverse"]:
            lines.append(f"# {b['raw']} - reverse code {', '.join(b['reverse'])}")
            for c in b["reverse"]:
                lines.append(f"data${c}_R <- {b['flip']} - data${c}")
        cols = [f"data${c}_R" if c in b["reverse"] else f"data${c}" for c in b["items"]]
        lines.append(f"data${b['composite']} <- rowMeans(cbind({', '.join(cols)}), na.rm = TRUE)")
        lines.append("")
    lines += ["# Optional exclusion step", f"# {engine._EXCLUSION_NOTE}",
              'if (file.exists("Simulation_Diagnostics.csv")) {',
              '  diagnostics <- read_csv("Simulation_Diagnostics.csv", show_col_types = FALSE)',
              '  data <- left_join(data, diagnostics[, c("ResponseId", "Exclude_Recommended")], by = "ResponseId")',
              "  data_clean <- data[!is.na(data$Exclude_Recommended) & data$Exclude_Recommended == 0, ]",
              "} else {", '  message("Simulation_Diagnostics.csv not found; no exclusions applied.")', "  data_clean <- data", "}", "",
              'cat("Total N:", nrow(data), "\\n")', 'cat("Clean N:", nrow(data_clean), "\\n")', "",
              "# Ready for analysis. The long file (one row per participant and condition) is Simulated_Data_Long.csv:",
              '# long <- read_csv("Simulated_Data_Long.csv", show_col_types = FALSE)']
    return "\n".join(lines)


def _py(engine: Any, df: pd.DataFrame, spec: Dict[str, Any]) -> str:
    q = lambda x: "'" + _txt(x).replace("\\", "\\\\").replace("'", "\\'") + "'"  # noqa: E731
    lines = _header(engine, "Python Data Preparation Script", "#") + _layout_notes(spec, "#") + [
        "", "import os", "import pandas as pd", "import numpy as np", "",
        "# Load the wide file (Qualtrics-style export; one header row)", "data = pd.read_csv('Simulated_Data.csv')", "",
        "# Presentation order as a categorical", "if 'Order' in data.columns:", "    data['Order'] = pd.Categorical(data['Order'])", ""]
    if spec["mixed"] and spec["groups"]:
        lines += ["# Between-subjects group", "data['CONDITION'] = pd.Categorical(data['CONDITION'], categories=[" + ", ".join(q(g) for g in spec["groups"]) + "], ordered=True)", ""]
    if spec["has_gender"]:
        lines += ["data['Gender'] = pd.Categorical(data['Gender'])", ""]
    for b in spec["blocks"]:
        if b["reverse"]:
            lines.append(f"# {b['raw']} - reverse code {', '.join(b['reverse'])}")
            for c in b["reverse"]:
                lines.append(f"data['{c}_R'] = {b['flip']} - data['{c}']")
        cols = [f"'{c}_R'" if c in b["reverse"] else f"'{c}'" for c in b["items"]]
        lines.append(f"data['{b['composite']}'] = data[[{', '.join(cols)}]].mean(axis=1)")
        lines.append("")
    lines += ["# Optional exclusion step", f"# {engine._EXCLUSION_NOTE}", "if os.path.exists('Simulation_Diagnostics.csv'):",
              "    diagnostics = pd.read_csv('Simulation_Diagnostics.csv')",
              "    data = data.merge(diagnostics[['ResponseId', 'Exclude_Recommended']], on='ResponseId', how='left')",
              "    data_clean = data[data['Exclude_Recommended'] == 0].copy()", "else:",
              "    print('Simulation_Diagnostics.csv not found; no exclusions applied.')", "    data_clean = data.copy()", "",
              "print(f'Total N: {len(data)}')", "print(f'Clean N: {len(data_clean)}')", "",
              "# Ready for analysis. The long file (one row per participant and condition) is Simulated_Data_Long.csv:",
              "# long = pd.read_csv('Simulated_Data_Long.csv')"]
    return "\n".join(lines)


def _jl(engine: Any, df: pd.DataFrame, spec: Dict[str, Any]) -> str:
    q = lambda x: '"' + _txt(x).replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'  # noqa: E731
    lines = _header(engine, "Julia Data Preparation Script", "#") + _layout_notes(spec, "#") + [
        "", "using CSV", "using DataFrames", "using CategoricalArrays", "using Statistics", "",
        "# Load the wide file (Qualtrics-style export; one header row)", 'data = CSV.read("Simulated_Data.csv", DataFrame)', ""]
    if spec["mixed"] and spec["groups"]:
        lines += ["# Between-subjects group", "data.CONDITION = categorical(data.CONDITION, levels=[" + ", ".join(q(g) for g in spec["groups"]) + "], ordered=true)", ""]
    if spec["has_gender"]:
        lines += ["data.Gender = categorical(data.Gender)", ""]
    for b in spec["blocks"]:
        if b["reverse"]:
            lines.append(f"# {b['raw']} - reverse code {', '.join(b['reverse'])}")
            for c in b["reverse"]:
                lines.append(f'data.{c}_R = {b["flip"]} .- data.{c}')
        syms = ", ".join(f":{c}_R" if c in b["reverse"] else f":{c}" for c in b["items"])
        lines.append(f"data.{b['composite']} = [isempty(collect(skipmissing(collect(r)))) ? missing : "
                     f"mean(skipmissing(collect(r))) for r in eachrow(data[:, [{syms}]])]")
        lines.append("")
    lines += ["# Optional exclusion step", f"# {engine._EXCLUSION_NOTE}", 'if isfile("Simulation_Diagnostics.csv")',
              '    diagnostics = CSV.read("Simulation_Diagnostics.csv", DataFrame)',
              "    data = leftjoin(data, select(diagnostics, [:ResponseId, :Exclude_Recommended]), on = :ResponseId)",
              "    data_clean = filter(row -> coalesce(row.Exclude_Recommended, 1) == 0, data)", "else",
              '    println("Simulation_Diagnostics.csv not found; no exclusions applied.")', "    data_clean = copy(data)", "end", "",
              'println("Total N: ", nrow(data))', 'println("Clean N: ", nrow(data_clean))', "",
              "# Ready for analysis. The long file (one row per participant and condition) is Simulated_Data_Long.csv:",
              '# long = CSV.read("Simulated_Data_Long.csv", DataFrame)']
    return "\n".join(lines)


def _spss(engine: Any, df: pd.DataFrame, spec: Dict[str, Any]) -> str:
    lines = _header(engine, "SPSS Data Preparation Syntax", "*")
    lines = [ln + "." if ln.startswith("*") else ln for ln in lines]
    lines += [ln + "." for ln in _layout_notes(spec, "*")] + [
        "", "* Load the data first using File > Import Data > CSV Data... (Simulated_Data.csv, one header row).", "",
        "DATASET NAME data WINDOW=FRONT.", ""]
    if spec["mixed"]:
        lines += ["* The between-subjects group is a string column; create a numeric version with value labels.",
                  "AUTORECODE VARIABLES=CONDITION /INTO CONDITION_num /PRINT.", ""]
    for b in spec["blocks"]:
        if b["reverse"]:
            lines.append(f"* {b['raw']} - reverse code {', '.join(b['reverse'])}.")
            for c in b["reverse"]:
                lines.append(f"COMPUTE {c}_R = {b['flip']} - {c}.")
            lines.append("EXECUTE.")
        cols = " ".join(f"{c}_R" if c in b["reverse"] else c for c in b["items"])
        lines += [f"COMPUTE {b['composite']} = MEAN({cols}).", "EXECUTE.", ""]
    lines += ["* Optional exclusion step.", f"* {engine._EXCLUSION_NOTE}",
              "* Import 'Simulation_Diagnostics.csv' the same way, then run:", "DATASET NAME diag WINDOW=FRONT.",
              "DATASET ACTIVATE diag.", "SORT CASES BY ResponseId (A).", "DATASET ACTIVATE data.", "SORT CASES BY ResponseId (A).",
              "MATCH FILES /FILE=* /TABLE=diag /BY ResponseId.", "EXECUTE.", "", "USE ALL.",
              "COMPUTE filter_$=(Exclude_Recommended = 0).", "VARIABLE LABELS filter_$ 'Exclude_Recommended = 0 (FILTER)'.",
              "VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'.", "FORMATS filter_$ (f1.0).", "FILTER BY filter_$.", "EXECUTE.", "",
              "* Descriptive statistics.", "DESCRIPTIVES VARIABLES=ALL /STATISTICS=MEAN STDDEV MIN MAX.", "",
              "* Repeated-measures analyses (GLM ... /WSFACTOR) use the composite columns above; the long file",
              "* Simulated_Data_Long.csv is the layout for MIXED models (one row per participant and condition).", ""]
    return "\n".join(lines)


def _stata(engine: Any, df: pd.DataFrame, spec: Dict[str, Any]) -> str:
    q = lambda x: '"' + _txt(x).replace('"', "'").replace("`", "'").replace("$", "") + '"'  # noqa: E731
    lines = _header(engine, "Stata Data Preparation Do-File", "//") + _layout_notes(spec, "//") + [
        "", "// Load the wide file (Qualtrics-style export; one header row)",
        "// Note: import delimited lower-cases variable names (CONDITION -> condition, Wellbeing_Pre_1 -> wellbeing_pre_1).",
        'import delimited "Simulated_Data.csv", clear varnames(1)', ""]
    if spec["mixed"] and spec["groups"]:
        lines.append("// Label the between-subjects group")
        for i, g in enumerate(spec["groups"]):
            lines.append(f"label define condition_lbl {i + 1} {q(g)}, add")
        lines += ["encode condition, gen(condition_num) label(condition_lbl)", ""]
    for b in spec["blocks"]:
        items = [c.lower() for c in b["items"]]
        rev = [c.lower() for c in b["reverse"]]
        if rev:
            lines.append(f"// {b['raw']} - reverse code {', '.join(rev)}")
            for c in rev:
                lines.append(f"gen {c}_r = {b['flip']} - {c}")
        cols = " ".join(f"{c}_r" if c in rev else c for c in items)
        lines += [f"egen {b['composite'].lower()} = rowmean({cols})", ""]
    lines += ["// Optional exclusion step", f"// {engine._EXCLUSION_NOTE}", 'capture confirm file "Simulation_Diagnostics.csv"',
              "if _rc == 0 {", "    preserve", '    import delimited "Simulation_Diagnostics.csv", clear varnames(1)',
              "    keep responseid exclude_recommended", "    tempfile diag", "    save `diag'", "    restore",
              "    merge 1:1 responseid using `diag', keep(master match) nogenerate", "} else {",
              '    display "Simulation_Diagnostics.csv not found; no exclusions applied."', "    gen exclude_recommended = 0", "}", "",
              'display "Total N: " _N', "preserve", "keep if exclude_recommended == 0", 'display "Clean N: " _N', "",
              "summarize", "", "// Ready for analysis. For xtmixed / mixed models use Simulated_Data_Long.csv", "// (one row per participant and condition).",
              "restore"]
    return "\n".join(lines)


_BUILDERS = {"r": _r, "python": _py, "julia": _jl, "spss": _spss, "stata": _stata}


def script_for(language: str, engine: Any, df: pd.DataFrame) -> str:
    """The data-preparation script of ``language`` (r, python, julia, spss, stata) for a repeated-measures run."""
    return _BUILDERS[language](engine, df, wide_spec(engine, df))


def methods_sentence(engine: Any) -> str:
    """One sentence for the methods write-up that replaces 'randomly assigned to conditions'."""
    d = ((getattr(engine, "_within_result", None) or {}).get("metadata") or {}).get("design") or {}
    k = len(d.get("cells") or [])
    order = d.get("order_effective") or d.get("order") or "random"
    if d.get("type") == "mixed":
        return (f"N = {engine.sample_size} synthetic participants were randomly assigned to between-subjects groups and "
                f"each completed the measures in {k} within-subject conditions (order: {order}).")
    return (f"N = {engine.sample_size} synthetic participants each completed the measures in all {k} conditions "
            f"(within-subjects; order: {order}).")


def codebook_addendum(metadata: Dict[str, Any]) -> str:
    """Section appended to the codebook (Data_Codebook_Handbook.txt) for a repeated-measures run."""
    d = (metadata or {}).get("design") or {}
    if d.get("type") not in ("within", "mixed"):
        return ""
    cells = d.get("cells") or []
    lines = [
        "", "-" * 70, "REPEATED-MEASURES LAYOUT", "-" * 70, "",
        f"Design: {d.get('type')} ({'every participant sees every condition' if d.get('type') == 'within' else 'between-subjects groups x within-subject conditions'}).",
        "Simulated_Data.csv has ONE ROW PER PARTICIPANT. Each measure appears once per condition:",
        "    <Measure>_<Condition>_<item number>      item answers",
        "    <Measure>_<Condition>_mean (diagnostics)  composite of the items (reverse-keyed items recoded)",
        "Simulated_Data_Long.csv has ONE ROW PER PARTICIPANT AND CONDITION with the columns",
        "    Condition, the within factor(s), Position, Order, <Measure>_<item number>, <Measure>_mean.",
        "", "Conditions (column-name label -> condition):",
    ]
    for c in cells:
        lines.append(f"    {c.get('slug')}  ->  {c.get('label')}")
    lines += [
        "", "Design columns:",
        "    Order                    the order in which this participant saw the conditions (first > last)",
        "    Position_<Condition>     the position (1 = first) of that condition for this participant",
        "    Conditions_Completed     how many conditions the participant finished (attrition: dropouts miss later conditions)",
        "", f"Order scheme: {d.get('order_effective') or d.get('order')}.",
        f"Requested within-person correlation of the same measure across conditions: {d.get('within_correlation')} ({d.get('correlation_structure')}).",
        "Effect sizes on a within factor are d_av (mean difference / average SD of the two conditions); the paired d_z is larger when the",
        "conditions are positively correlated: d_z = d_av / sqrt(2 (1 - r)).",
    ]
    oe = d.get("order_effect") or {}
    if oe.get("enabled"):
        lines.append(f"A small order / fatigue drift of {oe.get('d_per_position')} SD per later position is included, so Order can be used as a covariate.")
    lines.append("")
    return "\n".join(lines)
