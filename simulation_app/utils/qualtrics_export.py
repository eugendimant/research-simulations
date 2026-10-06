"""
Qualtrics-style export of simulated data.

The simulation engine returns a dataframe that mixes participant-facing
responses with simulator-internal bookkeeping (run identifiers, researcher
composites, quality flags, response-time summaries, ...). A real Qualtrics
export contains none of the internal columns and always contains a block of
survey-metadata columns (StartDate, EndDate, Progress, Duration, ...).

This module turns the engine dataframe into the file participants of a class
would actually receive:

* ``build_qualtrics_export`` -> (export_df, diagnostics_df)
    export_df       what a Qualtrics "Download Data Table" CSV looks like
                    (single header row), response values untouched.
    diagnostics_df  the internal columns removed from export_df, keyed by
                    ResponseId, row order identical to export_df.
* ``to_qualtrics_raw_csv`` -> bytes with the three-row Qualtrics header
    (column names, question text, ImportId JSON).

Everything is pure and deterministic: no wall clock, no global RNG. All
randomness derives from the engine seed via hashlib / np.random.RandomState.
The engine dataframe itself is never modified.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "QUALTRICS_METADATA_COLUMNS",
    "QUALTRICS_COLUMN_DESCRIPTIONS",
    "DIAGNOSTICS_KEY_COLUMNS",
    "is_internal_column",
    "split_columns",
    "build_qualtrics_export",
    "export_to_csv_bytes",
    "to_qualtrics_raw_csv",
]

# Fixed reference date. Study dates are placed before it so no wall clock is used.
_REFERENCE_DATE = datetime(2026, 1, 1)

# Qualtrics metadata columns, in export order.
QUALTRICS_METADATA_COLUMNS: List[str] = [
    "StartDate",
    "EndDate",
    "Status",
    "IPAddress",
    "Progress",
    "Duration (in seconds)",
    "Finished",
    "RecordedDate",
    "ResponseId",
    "RecipientLastName",
    "RecipientFirstName",
    "RecipientEmail",
    "ExternalReference",
    "LocationLatitude",
    "LocationLongitude",
    "DistributionChannel",
    "UserLanguage",
]

# Human-readable meaning (used in the codebook).
QUALTRICS_COLUMN_DESCRIPTIONS: Dict[str, str] = {
    "StartDate": "Date and time the respondent started the survey (YYYY-MM-DD HH:MM:SS)",
    "EndDate": "Date and time of the respondent's last activity; StartDate + Duration",
    "Status": "Response type; 0 = response collected from an IP address (standard response)",
    "IPAddress": "Respondent IP address; left blank, as in anonymized responses",
    "Progress": "Percent of the survey answered (0-100)",
    "Duration (in seconds)": "Seconds between StartDate and EndDate",
    "Finished": "1 = respondent reached the end of the survey, 0 = did not",
    "RecordedDate": "Date and time the response was recorded (shortly after EndDate)",
    "ResponseId": "Unique response identifier (R_ followed by 15 characters)",
    "RecipientLastName": "Contact-list field; blank for anonymous links",
    "RecipientFirstName": "Contact-list field; blank for anonymous links",
    "RecipientEmail": "Contact-list field; blank for anonymous links",
    "ExternalReference": "Contact-list reference; blank for anonymous links",
    "LocationLatitude": "Respondent latitude; blank (location not recorded)",
    "LocationLongitude": "Respondent longitude; blank (location not recorded)",
    "DistributionChannel": "How the survey was distributed ('anonymous' = anonymous link)",
    "UserLanguage": "Survey language code (EN)",
}

# (question text shown in row 2, ImportId shown in row 3) for the raw 3-row CSV.
_RAW_META_HEADER: Dict[str, Tuple[str, str]] = {
    "StartDate": ("Start Date", "startDate"),
    "EndDate": ("End Date", "endDate"),
    "Status": ("Response Type", "status"),
    "IPAddress": ("IP Address", "ipAddress"),
    "Progress": ("Progress", "progress"),
    "Duration (in seconds)": ("Duration (in seconds)", "duration"),
    "Finished": ("Finished", "finished"),
    "RecordedDate": ("Recorded Date", "recordedDate"),
    "ResponseId": ("Response ID", "_recordId"),
    "RecipientLastName": ("Recipient Last Name", "recipientLastName"),
    "RecipientFirstName": ("Recipient First Name", "recipientFirstName"),
    "RecipientEmail": ("Recipient Email", "recipientEmail"),
    "ExternalReference": ("External Data Reference", "externalDataReference"),
    "LocationLatitude": ("Location Latitude", "locationLatitude"),
    "LocationLongitude": ("Location Longitude", "locationLongitude"),
    "DistributionChannel": ("Distribution Channel", "distributionChannel"),
    "UserLanguage": ("User Language", "userLanguage"),
}

# Leading columns of the diagnostics sidecar.
DIAGNOSTICS_KEY_COLUMNS: List[str] = [
    "ResponseId",
    "PARTICIPANT_ID",
    "RUN_ID",
    "SIMULATION_MODE",
    "SIMULATION_SEED",
]

# Simulator-internal columns that never exist in a real Qualtrics export.
_INTERNAL_EXACT: Set[str] = {
    "PARTICIPANT_ID",
    "RUN_ID",
    "SIMULATION_MODE",
    "SIMULATION_SEED",
    "_Generation_Source",
    "Completion_Time_Seconds",
    "Attention_Pass_Rate",
    "Max_Straight_Line",
    "Flag_Speed",
    "Flag_Attention",
    "Flag_StraightLine",
    "Exclude_Recommended",
    "Mean_Item_RT_ms",
    "Total_Scale_RT_ms",
}
_INTERNAL_PREFIXES: Tuple[str, ...] = ("ABE3_", "Flag_")

_ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Relative likelihood of starting a survey in each hour of the day (daytime/evening heavy).
_HOUR_WEIGHTS = np.array(
    [0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 4.0,
     4.0, 4.0, 4.0, 4.0, 3.5, 3.0, 3.0, 3.5, 3.5, 3.0, 2.0, 1.0],
    dtype=float,
)
_HOUR_PROBS = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

def _composite_columns(columns: Sequence[str]) -> Set[str]:
    """Return ``<Scale>_mean`` columns that are researcher composites of item columns."""
    cols = list(columns)
    out: Set[str] = set()
    for c in cols:
        if not isinstance(c, str) or not c.endswith("_mean") or len(c) <= 5:
            continue
        prefix = c[:-5] + "_"
        if any(o != c and isinstance(o, str) and o.startswith(prefix) for o in cols):
            out.add(c)
    return out


def is_internal_column(name: str, all_columns: Optional[Sequence[str]] = None) -> bool:
    """True if ``name`` is simulator-internal (absent from a real Qualtrics export).

    Args:
        name: Engine dataframe column name.
        all_columns: All column names (needed to recognise ``<Scale>_mean`` composites).
    """
    if name in _INTERNAL_EXACT or name.startswith(_INTERNAL_PREFIXES) or name.startswith("_"):
        return True
    if all_columns is not None and name in _composite_columns(all_columns):
        return True
    return False


def split_columns(columns: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split engine column names into (participant_facing, internal), order preserved."""
    cols = list(columns)
    composites = _composite_columns(cols)
    facing: List[str] = []
    internal: List[str] = []
    for c in cols:
        if c in _INTERNAL_EXACT or c.startswith(_INTERNAL_PREFIXES) or c.startswith("_") or c in composites:
            internal.append(c)
        else:
            facing.append(c)
    return facing, internal


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def _stable_int(*parts: Any, bits: int = 32) -> int:
    """Deterministic integer from arbitrary parts (hashlib, independent of PYTHONHASHSEED)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2 ** bits)


def _resolve_seed(df: pd.DataFrame, metadata: Optional[Mapping[str, Any]]) -> int:
    """Seed from metadata, else SIMULATION_SEED column, else 0."""
    meta = metadata or {}
    for cand in (meta.get("seed"), meta.get("simulation_seed")):
        try:
            if cand is not None and not (isinstance(cand, float) and np.isnan(cand)):
                return int(cand)
        except (TypeError, ValueError):
            continue
    if "SIMULATION_SEED" in df.columns and len(df):
        try:
            return int(pd.to_numeric(df["SIMULATION_SEED"], errors="coerce").dropna().iloc[0])
        except (IndexError, ValueError):
            pass
    return 0


def _make_response_ids(seed: int, keys: Sequence[Any]) -> List[str]:
    """'R_' + 15 alphanumerics per key; deterministic in (seed, key) and unique."""
    ids: List[str] = []
    seen: Set[str] = set()
    for key in keys:
        attempt = 0
        while True:
            digest = hashlib.sha256(f"RID|{seed}|{key}|{attempt}".encode("utf-8")).digest()
            rid = "R_" + "".join(_ALNUM[b % len(_ALNUM)] for b in digest[:15])
            if rid not in seen:
                break
            attempt += 1
        seen.add(rid)
        ids.append(rid)
    return ids


def _fmt(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Dropout / progress detection
# ---------------------------------------------------------------------------

def _is_blank(series: pd.Series) -> pd.Series:
    """True where a value is missing or an empty string."""
    blank = series.isna()
    if series.dtype == object:
        blank = blank | series.astype(str).str.strip().eq("")
    return blank


def _detect_progress(
    df: pd.DataFrame,
    question_cols: List[str],
    item_cols: List[str],
    rt_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (progress, is_dropout) arrays.

    A dropout is a participant whose numeric item columns end in a block of missing
    values that reaches the last item column, and (when the engine's response-time
    summaries exist) whose response-time summaries are also missing. Isolated
    item-level missingness in the middle of the survey is not a dropout.
    Progress for a dropout is the share of question columns up to the last
    answered one; completers get 100.
    """
    n = len(df)
    progress = np.full(n, 100, dtype=int)
    is_dropout = np.zeros(n, dtype=bool)
    if n == 0 or not item_cols:
        return progress, is_dropout

    item_missing = np.column_stack([df[c].isna().to_numpy() for c in item_cols])
    q_blank = np.column_stack([_is_blank(df[c]).to_numpy() for c in question_cols])
    rt_all_missing = np.ones(n, dtype=bool)
    if rt_cols:
        for c in rt_cols:
            rt_all_missing &= df[c].isna().to_numpy()
    n_items = len(item_cols)
    for i in range(n):
        row = item_missing[i]
        run = 0
        for j in range(n_items - 1, -1, -1):
            if row[j]:
                run += 1
            else:
                break
        if run == 0:
            continue
        if rt_cols:
            dropout = bool(rt_all_missing[i])
        else:
            dropout = run >= 2
        if not dropout:
            continue
        is_dropout[i] = True
        answered = np.flatnonzero(~q_blank[i])
        last = int(answered[-1]) + 1 if len(answered) else 0
        # Anything after the dropout point counts as unanswered (text after the break is blanked).
        first_missing_item = n_items - run
        first_missing_col = question_cols.index(item_cols[first_missing_item])
        last = min(last, first_missing_col)
        pct = int(100 * last / max(len(question_cols), 1))
        progress[i] = int(min(max(pct, 1), 99))
    return progress, is_dropout


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_qualtrics_export(
    df: pd.DataFrame,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the Qualtrics-style delivered dataset and the diagnostics sidecar.

    Args:
        df: Engine dataframe (not modified).
        metadata: Engine metadata dict. Used for the seed, run id, mode and the
            list of open-ended question columns; every key is optional.

    Returns:
        (export_df, diagnostics_df). Both have one row per participant in the same
        order (sorted by StartDate) and share ``ResponseId``.
    """
    meta: Mapping[str, Any] = metadata or {}
    work = df.reset_index(drop=True)
    n = len(work)
    seed = _resolve_seed(work, meta)

    facing, internal = split_columns(list(work.columns))
    item_like = [c for c in facing if c != "CONDITION"]

    # Participant keys: PARTICIPANT_ID when present, else row position.
    if "PARTICIPANT_ID" in work.columns:
        keys = list(work["PARTICIPANT_ID"].tolist())
    else:
        keys = list(range(1, n + 1))

    # Open-ended text columns (only these may be blanked after a dropout).
    oe_cols: Set[str] = set()
    for q in meta.get("open_ended_questions", []) or []:
        if isinstance(q, Mapping):
            for k in ("variable_name", "name"):
                v = q.get(k)
                if v and v in work.columns:
                    oe_cols.add(str(v))
    descs = meta.get("column_descriptions", {}) or {}
    for c in item_like:
        if str(descs.get(c, "")).startswith("Open-ended"):
            oe_cols.add(c)

    # Numeric item columns = participant-facing numeric columns that can be missing
    # (attention-check and demographic columns are protected in the engine and thus complete).
    item_cols = [
        c for c in item_like
        if c not in oe_cols and pd.api.types.is_numeric_dtype(work[c])
        and c not in ("Age",) and "attention_check" not in c.lower()
    ]
    rt_cols = [c for c in ("Mean_Item_RT_ms", "Total_Scale_RT_ms") if c in work.columns]

    progress, is_dropout = _detect_progress(work, item_like, item_cols, rt_cols)

    # ---- Duration ----------------------------------------------------------
    rs_dur = np.random.RandomState(_stable_int("duration", seed))
    if "Completion_Time_Seconds" in work.columns:
        base_dur = pd.to_numeric(work["Completion_Time_Seconds"], errors="coerce").to_numpy(dtype=float)
    else:
        base_dur = np.full(n, np.nan)
    fallback_dur = (
        45.0 + 9.0 * max(len(item_like), 1) * np.exp(rs_dur.normal(0.0, 0.35, size=n))
    )
    base_dur = np.where(np.isfinite(base_dur) & (base_dur > 0), base_dur, fallback_dur)
    duration = np.maximum(np.rint(base_dur).astype(int), 1)
    duration = np.where(
        is_dropout,
        np.maximum((duration * progress / 100.0).astype(int), 5),
        duration,
    ).astype(int)

    # ---- Timestamps --------------------------------------------------------
    rs_t = np.random.RandomState(_stable_int("timestamps", seed))
    offset_days = 30 + _stable_int("base-date", seed) % 700
    span_days = 1 + _stable_int("span", seed) % 4
    base_day = _REFERENCE_DATE - timedelta(days=int(offset_days))
    day_idx = rs_t.randint(0, span_days, size=n)
    hours = rs_t.choice(24, size=n, p=_HOUR_PROBS)
    secs = rs_t.randint(0, 3600, size=n)
    start_off = day_idx.astype(np.int64) * 86400 + hours.astype(np.int64) * 3600 + secs
    rec_lag = rs_t.randint(0, 6, size=n)

    order = sorted(range(n), key=lambda i: (int(start_off[i]), str(keys[i])))
    order_arr = np.array(order, dtype=int)

    starts = [base_day + timedelta(seconds=int(start_off[i])) for i in range(n)]
    ends = [starts[i] + timedelta(seconds=int(duration[i])) for i in range(n)]
    recorded = [ends[i] + timedelta(seconds=int(rec_lag[i])) for i in range(n)]

    response_ids = _make_response_ids(seed, keys)

    meta_frame = pd.DataFrame({
        "StartDate": [_fmt(t) for t in starts],
        "EndDate": [_fmt(t) for t in ends],
        "Status": np.zeros(n, dtype=int),
        "IPAddress": [""] * n,
        "Progress": progress,
        "Duration (in seconds)": duration,
        "Finished": np.where(is_dropout, 0, 1).astype(int),
        "RecordedDate": [_fmt(t) for t in recorded],
        "ResponseId": response_ids,
        "RecipientLastName": [""] * n,
        "RecipientFirstName": [""] * n,
        "RecipientEmail": [""] * n,
        "ExternalReference": [""] * n,
        "LocationLatitude": [""] * n,
        "LocationLongitude": [""] * n,
        "DistributionChannel": ["anonymous"] * n,
        "UserLanguage": ["EN"] * n,
    })[QUALTRICS_METADATA_COLUMNS]

    body = work[facing].copy()
    # A partial response cannot contain text written after the point of dropout.
    for c in oe_cols:
        if c in body.columns and is_dropout.any():
            body[c] = body[c].astype(object)
            body.loc[is_dropout, c] = ""
    # Whole-number values print as 6, not 6.0 (Qualtrics writes integers); values unchanged.
    for c in body.columns:
        s = body[c]
        if pd.api.types.is_float_dtype(s):
            nn = s.dropna()
            if len(nn) and np.all(np.isfinite(nn.to_numpy())) and np.all(nn.to_numpy() == np.rint(nn.to_numpy())):
                body[c] = s.round().astype("Int64")

    export_df = pd.concat([meta_frame, body], axis=1)
    export_df = export_df.iloc[order_arr].reset_index(drop=True)

    # ---- Diagnostics sidecar ----------------------------------------------
    diag = pd.DataFrame({"ResponseId": response_ids})
    diag["PARTICIPANT_ID"] = work["PARTICIPANT_ID"].to_numpy() if "PARTICIPANT_ID" in work.columns else keys
    diag["RUN_ID"] = work["RUN_ID"].to_numpy() if "RUN_ID" in work.columns else str(meta.get("run_id", ""))
    diag["SIMULATION_MODE"] = (
        work["SIMULATION_MODE"].to_numpy() if "SIMULATION_MODE" in work.columns
        else str(meta.get("simulation_mode", "")).upper()
    )
    diag["SIMULATION_SEED"] = (
        work["SIMULATION_SEED"].to_numpy() if "SIMULATION_SEED" in work.columns else seed
    )
    for c in internal:
        if c not in diag.columns:
            diag[c] = work[c].to_numpy()
    diag = diag.iloc[order_arr].reset_index(drop=True)

    assert list(diag["ResponseId"]) == list(export_df["ResponseId"])
    return export_df, diag


def export_to_csv_bytes(export_df: pd.DataFrame) -> bytes:
    """Single-header-row CSV (the file analysis scripts read), UTF-8."""
    return export_df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Raw three-row-header CSV
# ---------------------------------------------------------------------------

_ITEM_RE = re.compile(r"^(?P<stem>.+)_(?P<num>\d+)$")


def _import_ids(columns: Sequence[str]) -> Dict[str, str]:
    """ImportId per column: fixed ids for metadata, QID-style for questions.

    Multi-item scales (Trust_1..Trust_4) share one QID (QID7_1..QID7_4), like matrix
    questions; CONDITION is embedded data and keeps its field name.
    """
    ids: Dict[str, str] = {}
    qcols = [c for c in columns if c not in _RAW_META_HEADER and c != "CONDITION"]
    stems: Dict[str, int] = {}
    for c in qcols:
        m = _ITEM_RE.match(c)
        if m:
            stems[m.group("stem")] = stems.get(m.group("stem"), 0) + 1
    qnum = 0
    stem_qid: Dict[str, int] = {}
    for c in columns:
        if c in _RAW_META_HEADER:
            ids[c] = _RAW_META_HEADER[c][1]
        elif c == "CONDITION":
            ids[c] = "CONDITION"
        else:
            m = _ITEM_RE.match(c)
            if m and stems.get(m.group("stem"), 0) >= 2:
                stem = m.group("stem")
                if stem not in stem_qid:
                    qnum += 1
                    stem_qid[stem] = qnum
                ids[c] = f"QID{stem_qid[stem]}_{m.group('num')}"
            else:
                qnum += 1
                ids[c] = f"QID{qnum}"
    return ids


def to_qualtrics_raw_csv(
    export_df: pd.DataFrame,
    question_labels: Optional[Mapping[str, str]] = None,
) -> bytes:
    """CSV with the three Qualtrics header rows.

    Row 1: column names. Row 2: question text (``question_labels[col]`` when known,
    otherwise the column name). Row 3: ``{"ImportId":"<id>"}`` per column.

    Args:
        export_df: Output of ``build_qualtrics_export``.
        question_labels: Optional mapping column -> question text/description.
    """
    labels = question_labels or {}
    cols = list(export_df.columns)
    ids = _import_ids(cols)
    row2 = [
        _RAW_META_HEADER[c][0] if c in _RAW_META_HEADER
        else (c if c == "CONDITION" else str(labels.get(c) or c))  # embedded data: field name
        for c in cols
    ]
    row3 = [json.dumps({"ImportId": ids[c]}, separators=(",", ":")) for c in cols]

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(cols)
    writer.writerow(row2)
    writer.writerow(row3)
    body = export_df.to_csv(index=False, header=False, lineterminator="\n")
    return (buf.getvalue() + body).encode("utf-8")
