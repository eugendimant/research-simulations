"""User-facing download package: the data, a plain codebook, rudimentary data-preparation
scripts — and nothing about how the data were produced.

Version: 1.3.1.2

The ZIP the user downloads and the package emailed to the owner used to be the SAME bytes
(``app.py`` handed one ``files`` dict to both the download button and ``_notify_instructor``).
That package carries the simulator's own working notes: the random seed, the persona roster and
its per-condition breakdown, the trait averages, the configured and observed Cohen's d per
contrast (the answer key), the calibration/registry provenance with the names of engine
internals, and the engine's own identifiers on demographic columns. Anyone who downloaded a run
could read how the generator works and reconstruct its settings.

This module derives the *user* package from the full one. The owner's email and the server-side
run archive keep the full package unchanged.

Three rules shape what survives:

1. **Allowlist, never denylist, for structured files.** ``Metadata.json`` and the diagnostics
   sidecar are rebuilt from a fixed set of keys/columns, so a field added to the engine later is
   absent from the user package until someone adds it here on purpose. A new internal cannot
   leak by being forgotten.
2. **Section filters for prose**, plus :func:`find_simulation_internals` as a backstop. The
   codebook and the study summary are human documents; they are filtered by section and then
   scanned. A section that still trips the scanner is dropped rather than shipped.
3. **The data themselves are untouched.** ``Simulated_Data.csv`` and the Qualtrics-raw variant
   are a survey export and contain no simulator columns; they pass through byte for byte.

Nothing here hides that the data are simulated — the tool says so throughout, and so does the
user package. What it hides is the mechanism.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__version__ = "1.3.1.2"

__all__ = [
    "build_user_package",
    "find_simulation_internals",
    "sanitize_metadata",
    "sanitize_diagnostics_csv",
    "sanitize_codebook",
    "sanitize_study_summary",
    "strip_internal_columns",
    "public_run_id",
    "USER_FILE_NAMES",
    "OWNER_ONLY_FILE_NAMES",
]


# ──────────────────────────────────────────────────────────────────────────────
# What the user's ZIP may contain
# ──────────────────────────────────────────────────────────────────────────────

#: Entries kept in the user ZIP. ``Source_Files/`` is kept too (the user's own uploads).
USER_FILE_NAMES: Tuple[str, ...] = (
    "Simulated_Data.csv",
    "Simulated_Data_Qualtrics_Raw.csv",
    "Simulation_Diagnostics.csv",
    "Simulated_Data_Long.csv",
    "Study_Metadata.json",
    "Metadata.json",
    "Data_Codebook_Handbook.txt",
    "Schema_Validation.json",
    "User_Study_Summary.md",
    "User_Study_Summary.html",
    "R_Prepare_Data.R",
    "Python_Prepare_Data.py",
    "Julia_Prepare_Data.jl",
    "SPSS_Prepare_Data.sps",
    "Stata_Prepare_Data.do",
)

#: Entries that must never reach the user ZIP even if someone adds them to ``files``.
OWNER_ONLY_FILE_NAMES: Tuple[str, ...] = (
    "INSTRUCTOR_Statistical_Report.html",
    "INSTRUCTOR_Detailed_Analysis.md",
    "Instructor_Report.md",
    "Engine_Log.txt",
    "Validation_Results.json",
)


# ──────────────────────────────────────────────────────────────────────────────
# The scanner (also the regression test's oracle)
# ──────────────────────────────────────────────────────────────────────────────

# Each entry is (label, compiled pattern). The patterns name simulator mechanics only; the
# vocabulary a real codebook needs ("simulated data", "Simulated_Data.csv", "Max_Straight_Line",
# "Cohen's d" in a textbook effect-size table) is deliberately NOT matched, because the user
# package legitimately contains it.
_INTERNALS_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        ("persona", r"persona"),
        # Underscore is a word character, so \b does not separate "Careless_Responder".
        # Letter-boundaries are used throughout instead.
        ("response-style taxonomy",
         r"(?<![A-Za-z])(satisfic\w*|acquiescen\w*|response[ _]style|responder)(?![A-Za-z])"),
        ("careless-responder machinery", r"(?<![A-Za-z])careless(?![A-Za-z])"),
        ("trait model",
         r"(?<![A-Za-z])trait[s_ ]|trait_average|social[_ ]desirability|"
         r"(?<![A-Za-z])extremity(?![A-Za-z])|ideological_consistency"),
        ("random seed",
         r"(?<![A-Za-z])seed(?![A-Za-z])|simulation[_ ]seed|random[_ ]seed"),
        ("configured/observed effect ledger",
         r"cohens?_d|effect_sizes?_(configured|observed|applied)|intended d\b|specified effect|"
         r"inferred (from condition names|effect|difference)|not calibrated"),
        ("calibration/registry provenance",
         r"calibrat\w*|registry|shrinkage|evidence[_ ]tier|unsourced|empirical[_ ]correlation|"
         r"marginal_source|effect_source|provenance"),
        ("engine identity",
         r"\babe[_ ]?3\b|\babe3_|abe_v\d|adaptive behavioral engine|enhanced_simulation_engine|"
         r"simulation engine|response_library|persona_library|generation[_ ]method"),
        ("domain detection", r"detected[_ ]domains?|study[_ ]domain|primary research domain|domain interpretation"),
        ("text-generation backend", r"\bllm\b|language model|template fallback|open[_ ]ended generation"),
        ("methods literature behind the generator",
         r"krosnick|meade & craig|meade and craig|yan & tourangeau|yan and tourangeau|"
         r"\bANES\b|census[- ]weighted|engel 2011|iyengar|balliet"),
        ("simulation-mechanics narration",
         r"simulation intelligence|how the simulation approached|condition effects strategy|"
         r"simulation settings|the simulation (assigned|analyz|applied|infer)"),
        ("cross-DV / correlation machinery", r"cross[_ ]dv|cross_dv_correlation|latent factor|kronecker"),
    )
)


def find_simulation_internals(text: str) -> List[str]:
    """Labels of simulation internals found in ``text`` (empty when the text is clean).

    This is the single definition of "reveals simulation mechanics" used by both the
    sanitizers' backstop and ``tests/test_user_download_secrecy.py``.
    """
    if not text:
        return []
    hits: List[str] = []
    for label, pattern in _INTERNALS_PATTERNS:
        if pattern.search(text):
            hits.append(label)
    return hits


def _decode(data: Any) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", "replace")
    return str(data or "")


# ──────────────────────────────────────────────────────────────────────────────
# Run id
# ──────────────────────────────────────────────────────────────────────────────

def public_run_id(run_id: str) -> str:
    """An opaque run id for the user package.

    The internal id embeds the random seed (``PILOT_S0000000007`` is seed 7), which is enough to
    reproduce a run. The public id keeps the mode prefix and replaces the rest with a digest, so
    a user can still quote "which run" without carrying the seed.
    """
    raw = str(run_id or "").strip()
    if not raw:
        return ""
    prefix = raw.split("_", 1)[0].upper() if "_" in raw else "RUN"
    if prefix not in ("PILOT", "FINAL", "RUN"):
        prefix = "RUN"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}_{digest}"


# ──────────────────────────────────────────────────────────────────────────────
# Metadata.json
# ──────────────────────────────────────────────────────────────────────────────

#: Top-level metadata keys the user may see. Everything else is dropped.
_METADATA_KEYS: Tuple[str, ...] = (
    "run_id",
    "generation_timestamp",
    "study_title",
    "study_description",
    "sample_size",
    "conditions",
    "factors",
    "design_type",
    "column_descriptions",
    "open_ended_questions",
    "preregistration_summary",
    "design_review",
)

#: Per-scale keys the user may see: the shape of the instrument they configured, not how it was filled.
_SCALE_KEYS: Tuple[str, ...] = (
    "name", "variable_name", "num_items", "scale_points", "scale_min", "scale_max",
    "reverse_items", "item_names", "anchors", "scale_anchors", "description", "dv_type",
)

#: Keys of the repeated-measures ``design`` block the user needs to read their own layout.
_DESIGN_KEYS: Tuple[str, ...] = (
    "type", "within_factors", "between_factors", "group_labels", "conditions",
    "order", "long_format_file", "n_conditions",
    # Column mapping: the user cannot read the wide/long layout without these.
    "cells", "wide_columns", "open_ended_columns", "open_ended_once",
)


def _clean_scale(scale: Any) -> Dict[str, Any]:
    if not isinstance(scale, dict):
        return {}
    return {k: scale[k] for k in _SCALE_KEYS if k in scale}


def sanitize_metadata(metadata: Any) -> Dict[str, Any]:
    """The user's ``Study_Metadata.json``: the study as the user described it, nothing more.

    Rebuilt from :data:`_METADATA_KEYS` rather than filtered, so engine fields added later are
    absent by default. The seed, the persona roster, the trait averages, the configured and
    observed effect ledgers, the calibration provenance, the engine block and the domain
    detection are all gone.
    """
    if not isinstance(metadata, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in _METADATA_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        if key == "run_id":
            value = public_run_id(value)
        out[key] = value

    scales = metadata.get("scales")
    if isinstance(scales, list):
        cleaned = [s for s in (_clean_scale(s) for s in scales) if s]
        if cleaned:
            out["scales"] = cleaned

    # ``column_descriptions`` is the codebook's source. It describes the dropped simulator
    # columns and carries the methods citations behind the response-time model, so it is
    # filtered the same way the diagnostics sidecar is.
    descriptions = metadata.get("column_descriptions")
    if isinstance(descriptions, dict):
        cleaned_desc: Dict[str, Any] = {}
        for column, text in descriptions.items():
            name = str(column)
            if name in _DIAGNOSTIC_DROP:
                continue
            label = _DIAGNOSTIC_RENAME.get(name, name)
            body = re.sub(r"Census-weighted (\w)", lambda m: m.group(1).upper(),
                          _CITATION_PARENS.sub("", str(text)))
            body = re.sub(r"Response style.*$", "Response pattern descriptor", body).strip()
            if find_simulation_internals(f"{label}: {body}"):
                continue
            cleaned_desc[label] = body
        out["column_descriptions"] = cleaned_desc

    design = metadata.get("design")
    if isinstance(design, dict):
        cleaned_design = {k: design[k] for k in _DESIGN_KEYS if k in design}
        if cleaned_design:
            out["design"] = cleaned_design

    out["note"] = (
        "This file describes the study that was simulated (design, conditions, instruments). "
        "It does not describe how the responses were produced."
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Simulation_Diagnostics.csv
# ──────────────────────────────────────────────────────────────────────────────

#: Diagnostics columns the user keeps: the identifiers needed to join, and the data-quality
#: flags the prep scripts use for the optional exclusion step.
_DIAGNOSTIC_KEEP: Tuple[str, ...] = (
    "ResponseId",
    "PARTICIPANT_ID",
    "Completion_Time_Seconds",
    "Attention_Pass_Rate",
    "Max_Straight_Line",
    "Flag_Speed",
    "Flag_Attention",
    "Flag_StraightLine",
    "Exclude_Recommended",
    "Mean_Item_RT_ms",
    "Total_Scale_RT_ms",
)

#: Engine-prefixed respondent background columns. The values are ordinary survey variables; only
#: the prefix names the engine, so they are renamed rather than dropped.
_DIAGNOSTIC_RENAME: Dict[str, str] = {
    "ABE3_Education": "Education",
    "ABE3_Income": "Income",
    "ABE3_PartyID": "PartyID",
    "ABE3_Ideology": "Ideology",
    "ABE3_State": "State",
    "ABE3_Region": "Region",
}

#: Columns dropped outright: the seed, the internal run id, the mode, the response-style label
#: and the per-cell record of which generator wrote the text.
_DIAGNOSTIC_DROP: Tuple[str, ...] = (
    "RUN_ID", "SIMULATION_MODE", "SIMULATION_SEED", "ABE3_ResponseStyle", "_Generation_Source",
)


def _diagnostic_column_kept(name: str) -> bool:
    if name in _DIAGNOSTIC_DROP:
        return False
    if name in _DIAGNOSTIC_KEEP or name in _DIAGNOSTIC_RENAME:
        return True
    # Researcher composites (``<Scale>_mean``) are the user's own outcome, not an internal.
    return name.endswith("_mean")


def sanitize_diagnostics_csv(csv_bytes: Any) -> bytes:
    """Drop the simulator's bookkeeping columns from the diagnostics sidecar.

    Kept: ``ResponseId``/``PARTICIPANT_ID`` for the join, the data-quality flags the prep
    scripts use, the composites, and the respondent background columns (renamed off the engine
    prefix). Dropped: the seed, the internal run id, the mode, the response-style label and the
    generator-source column. On any parse failure the sidecar is left out entirely rather than
    shipped unfiltered.
    """
    text = _decode(csv_bytes)
    if not text.strip():
        return b""
    try:
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
    except Exception:
        return b""
    if not rows:
        return b""
    header = rows[0]
    keep_idx = [i for i, name in enumerate(header) if _diagnostic_column_kept(name.strip())]
    if not keep_idx:
        return b""
    out_header = [_DIAGNOSTIC_RENAME.get(header[i].strip(), header[i].strip()) for i in keep_idx]
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(out_header)
    for row in rows[1:]:
        writer.writerow([row[i] if i < len(row) else "" for i in keep_idx])
    return buf.getvalue().encode("utf-8")


def _is_internal_column(name: str) -> bool:
    """True for a column that names the simulator rather than the survey.

    Deliberately narrow — an exact name from :data:`_DIAGNOSTIC_DROP`, an engine prefix, or a
    leading underscore — so a user's own scale (``Traits_1``, ``Response_Style_1``) is never
    mistaken for an internal and dropped from their data.
    """
    stripped = name.strip()
    return (
        stripped in _DIAGNOSTIC_DROP
        or stripped.startswith("ABE3_")
        or stripped.startswith("_")
    )


def strip_internal_columns(csv_bytes: Any) -> bytes:
    """Drop simulator-named columns from a delivered CSV, byte-identical when there are none.

    Works on the three-row Qualtrics header too, because the columns are dropped by index in
    every row. The data file is never dropped as a whole: it is the point of the download.
    """
    if isinstance(csv_bytes, bytes) and not csv_bytes:
        return b""
    text = _decode(csv_bytes)
    first = text.split("\n", 1)[0]
    if not any(_is_internal_column(cell) for cell in next(csv.reader(io.StringIO(first)), [])):
        return csv_bytes if isinstance(csv_bytes, bytes) else text.encode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return b""
    keep = [i for i, cell in enumerate(rows[0]) if not _is_internal_column(cell)]
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    for row in rows:
        writer.writerow([row[i] if i < len(row) else "" for i in keep])
    return buf.getvalue().encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Data_Codebook_Handbook.txt
# ──────────────────────────────────────────────────────────────────────────────

#: Codebook sections dropped whole. The headings are the literals the engine's explainer writes.
_CODEBOOK_DROP_SECTIONS: Tuple[str, ...] = (
    "EXPECTED EFFECT SIZES",
    "EXCLUSION CRITERIA",
    "PERSONA",
    "SIMULATION SETTINGS",
    "CALIBRATION",
)

_SECTION_RULE = re.compile(r"^-{10,}$")


def _codebook_blocks(text: str) -> Iterable[Tuple[Optional[str], List[str]]]:
    """Yield ``(heading, lines)``; ``heading`` is None for the preamble.

    The explainer writes a section as a rule line, the heading, a rule line, then the body.
    """
    lines = text.split("\n")
    heading: Optional[str] = None
    body: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_heading = (
            _SECTION_RULE.match(line.strip())
            and i + 2 < len(lines)
            and _SECTION_RULE.match(lines[i + 2].strip())
        )
        if is_heading:
            yield heading, body
            heading = lines[i + 1].strip()
            body = []
            i += 3
            continue
        body.append(line)
        i += 1
    yield heading, body


def _strip_column_entries(lines: List[str], names: Sequence[str]) -> List[str]:
    """Remove ``name`` + its indented description from a codebook column list."""
    drop = set(names)
    out: List[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() in drop and not lines[i].startswith(" "):
            i += 1
            while i < len(lines) and (lines[i].startswith(" ") or not lines[i].strip()):
                if not lines[i].strip():  # consume the single blank separator, then stop
                    i += 1
                    break
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


_CITATION_PARENS = re.compile(
    r"\s*\((?:[^()]*(?:Krosnick|Tourangeau|ANES|Meade|Engel|Iyengar|Balliet)[^()]*)\)", re.IGNORECASE
)


def sanitize_codebook(text: Any, *, run_id: str = "", public_id: str = "") -> bytes:
    """A plain data dictionary: what each column is, and nothing about how it was filled.

    Drops the expected-effect-size table (the answer key), the exclusion-threshold table, the
    detected-domain line and the seed; renames the engine-prefixed background columns and strips
    the methods citations from their descriptions.
    """
    body = _decode(text)
    if not body.strip():
        return b""

    kept: List[str] = []
    for heading, lines in _codebook_blocks(body):
        if heading is not None:
            upper = heading.upper()
            if any(token in upper for token in _CODEBOOK_DROP_SECTIONS):
                continue
        lines = _strip_column_entries(lines, _DIAGNOSTIC_DROP)
        text_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Detected Domains:") or stripped.startswith("Random Seed:"):
                continue
            if stripped.startswith("Mode:"):
                continue
            for old, new in _DIAGNOSTIC_RENAME.items():
                line = line.replace(old, new)
            line = _CITATION_PARENS.sub("", line)
            # The Qualtrics-metadata section explains that the timestamps are drawn rather than
            # real. Keep the fact, drop the mention of the seed (which the backstop below would
            # otherwise take the whole section for).
            line = line.replace("are generated from the random seed; they are not the time you ran the tool.",
                                "are drawn for the dataset; they are not the time you ran the tool.")
            line = re.sub(r"Census-weighted (\w)", lambda m: m.group(1).upper(), line)
            line = re.sub(r"7-point party identification", "Party identification (7-point)", line)
            line = re.sub(r"Response style.*$", "Response pattern descriptor", line)
            text_lines.append(line)
        block = "\n".join(text_lines)
        if heading is not None:
            rule = "-" * 70
            block = f"{rule}\n{heading}\n{rule}\n{block}"
        if find_simulation_internals(block):
            continue  # backstop: a section that still names internals is dropped, not shipped
        kept.append(block)

    out = "\n".join(kept)
    if run_id and public_id:
        out = out.replace(run_id, public_id)
    out = re.sub(r"\n{4,}", "\n\n\n", out)
    return out.encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# User_Study_Summary.md / .html
# ──────────────────────────────────────────────────────────────────────────────

#: ``##`` sections of the study summary dropped whole, matched case-insensitively on a prefix.
_SUMMARY_DROP_SECTIONS: Tuple[str, ...] = (
    "simulation intelligence report",
    "simulation settings",
    "persona distribution",
    "response style",
)

#: Rows of the ``Generation Information`` / design tables that name the generator.
_SUMMARY_DROP_ROW = re.compile(
    r"^\|\s*\*{0,2}(random seed|seed|generation method|mode|research domain|detected domains|"
    r"tool version|generation engine)\*{0,2}\s*\|", re.IGNORECASE
)


def sanitize_study_summary(markdown: Any, *, run_id: str = "", public_id: str = "") -> str:
    """The study summary with every simulation-mechanics section removed.

    What survives is a description of the user's own study and ordinary analysis guidance: the
    design, condition counts, attention checks, exclusion counts, the data dictionary, the
    statistical-test recommendations, the power table and the starter analysis code. What goes is
    the "Simulation Intelligence Report" (domain detection, the effect strategy, the configured
    and observed d per contrast, the persona rationale), the "Simulation Settings" transparency
    block and the persona distribution with its trait profiles.

    Any remaining ``##`` section that still trips :func:`find_simulation_internals` is dropped
    too, so a new section in the report generator cannot leak silently.
    """
    text = _decode(markdown)
    if not text.strip():
        return ""

    chunks: List[Tuple[Optional[str], List[str]]] = []
    heading: Optional[str] = None
    body: List[str] = []
    for line in text.split("\n"):
        if line.startswith("## ") and not line.startswith("###"):
            chunks.append((heading, body))
            heading = line
            body = []
            continue
        body.append(line)
    chunks.append((heading, body))

    kept: List[str] = []
    for head, lines in chunks:
        if head is not None:
            title = head[3:].strip().lower()
            if any(title.startswith(token) for token in _SUMMARY_DROP_SECTIONS):
                continue
        filtered: List[str] = []
        in_code = False
        for line in lines:
            if line.lstrip().startswith("```"):
                in_code = not in_code
            if _SUMMARY_DROP_ROW.match(line.strip()):
                continue
            for old, new in _DIAGNOSTIC_RENAME.items():
                line = line.replace(f"`{old}`", f"`{new}`").replace(f"| {old} ", f"| {new} ")
            line = _CITATION_PARENS.sub("", line)
            # A single offending sentence (a bullet about careless responders, a note naming the
            # engine) should cost that sentence, not the whole section. Table rows and code lines
            # cannot be dropped in isolation without breaking the block, so those fall through to
            # the section-level backstop below.
            if not in_code and _is_prose_line(line) and find_simulation_internals(line):
                continue
            filtered.append(line)
        filtered = _drop_summary_table_rows(filtered)
        block = "\n".join(([head] if head else []) + filtered)
        if find_simulation_internals(block):
            continue  # backstop
        kept.append(block)

    out = "\n".join(kept)
    if run_id and public_id:
        out = out.replace(run_id, public_id)
    out = re.sub(r"\n{4,}", "\n\n\n", out).strip() + "\n"
    return out


def _is_prose_line(line: str) -> bool:
    """True for a sentence or bullet that can be dropped on its own without breaking a block."""
    stripped = line.strip()
    if not stripped or stripped.startswith("|") or stripped.startswith("#") or stripped.startswith("```"):
        return False
    return True


def _drop_summary_table_rows(lines: List[str]) -> List[str]:
    """Drop data-dictionary rows for columns the user package no longer ships."""
    dropped = {f"`{name}`" for name in _DIAGNOSTIC_DROP}
    out: List[str] = []
    for line in lines:
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 2 and cells[1] in dropped:
            continue
        out.append(line)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# The package
# ──────────────────────────────────────────────────────────────────────────────

_PACKAGE_README = """\
WHAT IS IN THIS DOWNLOAD
========================

Simulated_Data.csv
    The dataset, laid out like a Qualtrics export (one header row).
Simulated_Data_Qualtrics_Raw.csv
    The same rows with the three-row Qualtrics header, for practising the usual
    "delete rows 2-3" step.
Simulation_Diagnostics.csv
    Per-response data-quality flags, keyed by ResponseId. Join it on ResponseId
    and filter Exclude_Recommended == 0 to work with clean responses.
Data_Codebook_Handbook.txt
    What every column means.
Study_Metadata.json
    The study you described: design, conditions, instruments.
Schema_Validation.json
    Whether the delivered columns match the design you specified.
User_Study_Summary.md / .html
    A summary of the study, the data-quality counts and suggested analyses.
R_/Python_/Julia_/SPSS_/Stata_Prepare_Data.*
    Data-preparation starters: load the file, label the condition factor,
    reverse-key and average the scale items, join the quality flags, count rows.
    They prepare the data for your own analysis; they are not how the data were
    produced.
Source_Files/
    The files you uploaded for this run, unchanged.

The responses in this dataset are simulated, not collected from people. Treat
them as a stand-in for real data while you build and check an analysis
pipeline, and never report them as empirical findings.
"""


def build_user_package(
    files: Dict[str, Any],
    metadata: Any,
    *,
    markdown_to_html: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, bytes]:
    """Derive the user-facing ZIP contents from the full (owner) package.

    ``files`` is the full dict ``app.py`` builds; it is not modified. The returned dict holds
    only the entries in :data:`USER_FILE_NAMES` (plus the user's own ``Source_Files/`` uploads
    and a ``README_Download_Contents.txt``), with ``Metadata.json`` replaced by the reduced
    ``Study_Metadata.json``, and the diagnostics sidecar, the codebook and the study summary
    sanitized.

    ``markdown_to_html`` builds the user HTML summary from the sanitized Markdown, so the HTML
    can never carry a section the Markdown dropped. Without it the HTML summary is left out.
    """
    source: Dict[str, Any] = dict(files or {})
    meta = metadata if isinstance(metadata, dict) else {}
    raw_run_id = str(meta.get("run_id") or "")
    pub_run_id = public_run_id(raw_run_id)

    out: Dict[str, bytes] = {}

    def _put(name: str, data: Any) -> None:
        if isinstance(data, bytes) and data:
            out[name] = data
        elif isinstance(data, str) and data:
            out[name] = data.encode("utf-8")

    # 1. Data: passed through unchanged (the delivered export is a survey export and carries no
    #    simulator columns), but the header is checked so a column added to the export later
    #    cannot ride along.
    for name in ("Simulated_Data.csv", "Simulated_Data_Qualtrics_Raw.csv"):
        if name in source:
            _put(name, strip_internal_columns(source[name]))
    for name in source:
        if name.endswith("_Long.csv"):
            _put(name, strip_internal_columns(source[name]))

    # 2. Diagnostics sidecar: column allowlist.
    if "Simulation_Diagnostics.csv" in source:
        _put("Simulation_Diagnostics.csv", sanitize_diagnostics_csv(source["Simulation_Diagnostics.csv"]))

    # 3. Metadata: key allowlist, under a name that says what it is.
    _put("Study_Metadata.json", json.dumps(sanitize_metadata(meta), indent=2, default=str))

    # 4. Codebook.
    if "Data_Codebook_Handbook.txt" in source:
        _put(
            "Data_Codebook_Handbook.txt",
            sanitize_codebook(source["Data_Codebook_Handbook.txt"], run_id=raw_run_id, public_id=pub_run_id),
        )

    # 5. Schema validation: structural checks on the delivered columns; no mechanics.
    if "Schema_Validation.json" in source:
        schema_text = _decode(source["Schema_Validation.json"])
        if raw_run_id and pub_run_id:
            schema_text = schema_text.replace(raw_run_id, pub_run_id)
        if not find_simulation_internals(schema_text):
            _put("Schema_Validation.json", schema_text)

    # 6. Study summary: section filter, then the HTML built from the sanitized Markdown.
    if "User_Study_Summary.md" in source:
        summary_md = sanitize_study_summary(
            source["User_Study_Summary.md"], run_id=raw_run_id, public_id=pub_run_id
        )
        if summary_md.strip():
            _put("User_Study_Summary.md", summary_md)
            if markdown_to_html is not None:
                try:
                    title = f"Study Summary: {meta.get('study_title', 'Study')}"
                    html = markdown_to_html(summary_md, title)
                    if html and not find_simulation_internals(html):
                        _put("User_Study_Summary.html", html)
                except Exception:
                    pass  # the Markdown summary is enough; an HTML failure is not a blocker

    # 7. Data-preparation scripts: rudimentary load/clean/summarise starters.
    for name in ("R_Prepare_Data.R", "Python_Prepare_Data.py", "Julia_Prepare_Data.jl",
                 "SPSS_Prepare_Data.sps", "Stata_Prepare_Data.do"):
        if name not in source:
            continue
        script = _decode(source[name])
        if raw_run_id and pub_run_id:
            script = script.replace(raw_run_id, pub_run_id)
        for old, new in _DIAGNOSTIC_RENAME.items():
            script = script.replace(old, new)
        hits = find_simulation_internals(script)
        if hits:
            script = _redact_script(script)
            if find_simulation_internals(script):
                continue  # a script that still names internals is left out
        _put(name, script)

    # 8. The user's own uploads, unchanged.
    for name, data in source.items():
        if name.startswith("Source_Files/"):
            _put(name, data)

    _put("README_Download_Contents.txt", _PACKAGE_README)
    return out


_SCRIPT_COMMENT = re.compile(r"^\s*(#|\*|//|/\*|\*/)")


def _redact_script(script: str) -> str:
    """Drop comment lines that name simulator internals; code lines are left alone.

    The prep scripts are generated code that loads, labels, averages and counts. Only their
    commentary has ever mentioned mechanics, so dropping the offending comment lines keeps the
    script runnable. A script whose *code* trips the scanner is left out of the package by the
    caller instead.
    """
    out: List[str] = []
    for line in script.split("\n"):
        if _SCRIPT_COMMENT.match(line) and find_simulation_internals(line):
            continue
        out.append(line)
    return "\n".join(out)
