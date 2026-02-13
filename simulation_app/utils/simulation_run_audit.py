"""Simulation run archival and continuous improvement audits.

Each simulation run is persisted to a dedicated folder and then audited.
The audit detects quality problems and appends actionable findings to
an aggregate log for iterative code improvements.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

STANDARD_TEXT_EXCLUSIONS = {
    "CONDITION",
    "PARTICIPANT_ID",
    "RUN_ID",
    "SIMULATION_MODE",
    "SIMULATION_SEED",
    "Gender",
    "_PERSONA",
    "EXCLUSION_REASON",
}

GENERIC_PATTERNS = (
    "i think",
    "it depends",
    "not sure",
    "hard to say",
    "good question",
    "n/a",
    "na",
    "idk",
)


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _extract_open_ended_columns(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if df[c].dtype == object and c not in STANDARD_TEXT_EXCLUSIONS
    ]


def _compute_issue_severity(issues: List[Dict[str, Any]]) -> str:
    if any(i["level"] == "error" for i in issues):
        return "error"
    if any(i["level"] == "warning" for i in issues):
        return "warning"
    return "ok"


def _compute_quality_score(issues: List[Dict[str, Any]]) -> float:
    """Return 0-100 quality score weighted by issue severity."""
    score = 100.0
    for issue in issues:
        lvl = str(issue.get("level", "")).lower()
        if lvl == "error":
            score -= 20.0
        elif lvl == "warning":
            score -= 7.5
    return max(0.0, round(score, 1))


def _sha256_for_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_manifest(run_dir: Path) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": [],
    }
    for file_path in sorted([p for p in run_dir.iterdir() if p.is_file()]):
        manifest["files"].append({
            "name": file_path.name,
            "size_bytes": file_path.stat().st_size,
            "sha256": _sha256_for_file(file_path),
        })
    (run_dir / "Run_Manifest.json").write_text(_safe_json(manifest), encoding="utf-8")


def persist_simulation_run(
    *,
    output_root: Path,
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    instructor_report_md: str,
    engine_log: Optional[List[str]] = None,
    validation_results: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist a run's instructor copy, data output, and logs to its own folder."""
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(metadata.get("run_id") or metadata.get("RUN_ID") or "").strip() or timestamp
    folder_name = f"{timestamp}__{re.sub(r'[^A-Za-z0-9_.-]+', '_', run_id)[:80]}"
    run_dir = output_root / folder_name
    counter = 1
    while run_dir.exists():
        counter += 1
        run_dir = output_root / f"{folder_name}_{counter}"
    run_dir.mkdir(parents=True, exist_ok=False)

    df.to_csv(run_dir / "Simulated_Data.csv", index=False)
    (run_dir / "Metadata.json").write_text(_safe_json(metadata), encoding="utf-8")
    (run_dir / "Instructor_Copy.md").write_text(instructor_report_md or "", encoding="utf-8")

    if validation_results is not None:
        (run_dir / "Validation_Results.json").write_text(_safe_json(validation_results), encoding="utf-8")

    if engine_log:
        log_body = "\n".join(str(x) for x in engine_log)
        (run_dir / "Engine_Log.txt").write_text(log_body, encoding="utf-8")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(df)) if df is not None else 0,
        "columns": int(len(df.columns)) if df is not None else 0,
        "open_ended_columns": _extract_open_ended_columns(df) if df is not None else [],
    }
    (run_dir / "Run_Summary.json").write_text(_safe_json(summary), encoding="utf-8")
    _write_manifest(run_dir)
    return run_dir


def audit_run_directory(run_dir: Path) -> Dict[str, Any]:
    """Analyze a stored run folder and detect quality issues/recommendations."""
    issues: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    data_path = run_dir / "Simulated_Data.csv"
    metadata_path = run_dir / "Metadata.json"

    if not data_path.exists():
        issues.append({"level": "error", "code": "missing_data_file", "message": "Simulated_Data.csv missing."})
        return {"run_dir": str(run_dir), "issues": issues, "recommendations": recommendations, "severity": "error"}

    df = pd.read_csv(data_path)
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            issues.append({"level": "warning", "code": "metadata_parse_failed", "message": "Metadata.json could not be parsed."})

    manifest_path = run_dir / "Run_Manifest.json"
    if not manifest_path.exists():
        issues.append({"level": "warning", "code": "manifest_missing", "message": "Run_Manifest.json missing for run folder."})
    else:
        try:
            _ = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            issues.append({"level": "warning", "code": "manifest_parse_failed", "message": "Run_Manifest.json could not be parsed."})

    if df.empty:
        issues.append({"level": "error", "code": "empty_dataset", "message": "Generated dataset has 0 rows."})

    oe_cols = _extract_open_ended_columns(df)
    for col in oe_cols:
        s = df[col].fillna("").astype(str).str.strip()
        blank_pct = float((s == "").mean() * 100)
        if blank_pct > 0:
            lvl = "error" if blank_pct > 10 else "warning"
            issues.append({
                "level": lvl,
                "code": "oe_blank_values",
                "message": f"{col}: {blank_pct:.1f}% blank open-ended responses.",
            })

        non_blank = [x for x in s.tolist() if x]
        if non_blank:
            unique_pct = len(set(non_blank)) / len(non_blank) * 100
            if unique_pct < 75:
                issues.append({
                    "level": "warning",
                    "code": "oe_low_uniqueness",
                    "message": f"{col}: low uniqueness ({unique_pct:.1f}%).",
                })

            short_count = sum(1 for response in non_blank if len(response.split()) < 5)
            short_pct = short_count / len(non_blank) * 100
            if short_pct > 20:
                issues.append({
                    "level": "warning",
                    "code": "oe_too_short",
                    "message": f"{col}: {short_pct:.1f}% responses are very short (<5 words).",
                })

            generic_hits = 0
            for response in non_blank:
                norm = response.lower()
                if any(p in norm for p in GENERIC_PATTERNS):
                    generic_hits += 1
            generic_pct = generic_hits / len(non_blank) * 100
            if generic_pct > 25:
                issues.append({
                    "level": "warning",
                    "code": "oe_generic_content",
                    "message": f"{col}: {generic_pct:.1f}% responses appear generic.",
                })

    llm_stats = metadata.get("llm_stats") or metadata.get("llm_response_stats") or {}
    llm_calls = int(llm_stats.get("llm_calls", 0) or 0)
    llm_attempts = int(llm_stats.get("llm_attempts", 0) or 0)
    fallback_uses = int(llm_stats.get("fallback_uses", 0) or 0)
    if oe_cols and llm_calls <= 0 and llm_attempts <= 0:
        issues.append({
            "level": "error",
            "code": "oe_no_llm_activity",
            "message": "Open-ended columns exist but no LLM calls/attempts were recorded.",
        })
    if oe_cols and fallback_uses > 0:
        issues.append({
            "level": "warning",
            "code": "oe_fallback_used",
            "message": f"Fallback used for {fallback_uses} responses.",
        })

    if any(i["code"] == "oe_generic_content" for i in issues):
        recommendations.append("Tighten OE prompt with topic anchors and minimum detail requirements.")
    if any(i["code"] == "oe_blank_values" for i in issues):
        recommendations.append("Enforce non-empty OE post-processing guard before final CSV export.")
    if any(i["code"] == "oe_no_llm_activity" for i in issues):
        recommendations.append("Inspect provider auth/order and fail-run when llm_calls==0 for OE runs.")
    if any(i["code"] == "oe_too_short" for i in issues):
        recommendations.append("Raise minimum OE response length and require context-specific details.")
    if any(i["code"] in {"manifest_missing", "manifest_parse_failed"} for i in issues):
        recommendations.append("Rebuild run artifacts to restore manifest integrity and checksum traceability.")

    audit = {
        "run_dir": str(run_dir),
        "audited_at": datetime.now().isoformat(timespec="seconds"),
        "issues": issues,
        "recommendations": recommendations,
        "severity": _compute_issue_severity(issues),
        "issue_count": len(issues),
        "quality_score": _compute_quality_score(issues),
    }
    (run_dir / "Quality_Audit.json").write_text(_safe_json(audit), encoding="utf-8")
    return audit



def _collect_issue_trends(output_root: Path) -> Dict[str, Any]:
    """Aggregate issue frequency across all audited run folders."""
    issue_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {"error": 0, "warning": 0, "ok": 0}
    quality_scores: List[float] = []
    audited_runs = 0

    for run_dir in sorted([p for p in output_root.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        qa_path = run_dir / "Quality_Audit.json"
        if not qa_path.exists():
            continue
        try:
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        audited_runs += 1
        sev = str(qa.get("severity", "")).lower()
        if sev in severity_counts:
            severity_counts[sev] += 1
        quality_scores.append(float(qa.get("quality_score", _compute_quality_score(qa.get("issues", [])))))
        for issue in qa.get("issues", []):
            code = str(issue.get("code", "unknown")).strip() or "unknown"
            issue_counts[code] = issue_counts.get(code, 0) + 1

    top_issues = sorted(issue_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    avg_quality = round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else 100.0
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "audited_run_count": audited_runs,
        "severity_counts": severity_counts,
        "issue_counts": issue_counts,
        "top_issues": [{"code": c, "count": n} for c, n in top_issues],
        "avg_quality_score": avg_quality,
    }

def audit_new_runs(
    *,
    output_root: Path,
    state_file: Path,
    running_log_file: Path,
) -> Dict[str, Any]:
    """Audit only newly created run folders and append findings to running log."""
    output_root.mkdir(parents=True, exist_ok=True)
    state = {"seen_runs": []}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            state = {"seen_runs": []}

    seen = set(state.get("seen_runs", []))
    run_dirs = sorted([
        p for p in output_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ])
    new_dirs = [p for p in run_dirs if p.name not in seen]

    audits = [audit_run_directory(run_dir) for run_dir in new_dirs]

    log_lines: List[str] = []
    now = datetime.now().isoformat(timespec="seconds")
    errors = warnings = 0
    for audit in audits:
        run_name = Path(audit["run_dir"]).name
        sev = audit["severity"].upper()
        log_lines.append(f"[{now}] run={run_name} severity={sev} issues={audit['issue_count']}")
        for issue in audit["issues"]:
            if issue["level"] == "error":
                errors += 1
            elif issue["level"] == "warning":
                warnings += 1
            log_lines.append(f"  - ({issue['level'].upper()}) {issue['code']}: {issue['message']}")
        for rec in audit["recommendations"]:
            log_lines.append(f"  -> recommendation: {rec}")
        if not audit["issues"]:
            log_lines.append("  - No issues detected.")

    if log_lines:
        running_log_file.parent.mkdir(parents=True, exist_ok=True)
        with running_log_file.open("a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")

    seen.update(p.name for p in new_dirs)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(_safe_json({"seen_runs": sorted(seen)}), encoding="utf-8")

    trends = _collect_issue_trends(output_root)
    (output_root / "issue_trends.json").write_text(_safe_json(trends), encoding="utf-8")

    summary = {
        "last_audit_at": now,
        "new_run_count": len(new_dirs),
        "error_count": errors,
        "warning_count": warnings,
        "running_log_file": str(running_log_file),
        "top_issue_codes": [i["code"] for i in trends.get("top_issues", [])],
        "avg_quality_score": trends.get("avg_quality_score", 100.0),
    }
    (output_root / "audit_summary.json").write_text(_safe_json(summary), encoding="utf-8")

    return {
        "new_run_count": len(new_dirs),
        "audits": audits,
        "running_log_file": str(running_log_file),
        "error_count": errors,
        "warning_count": warnings,
        "top_issue_codes": summary["top_issue_codes"],
        "avg_quality_score": summary["avg_quality_score"],
    }
