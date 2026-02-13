import json
from pathlib import Path

import pandas as pd

from simulation_app.utils.simulation_run_audit import persist_simulation_run, audit_new_runs


def test_persist_simulation_run_creates_required_files(tmp_path: Path):
    df = pd.DataFrame({
        "PARTICIPANT_ID": [1, 2],
        "CONDITION": ["A", "B"],
        "Q_open": ["Detailed answer one is here", "Detailed answer two is here"],
    })
    metadata = {"run_id": "run-123", "llm_stats": {"llm_calls": 2, "llm_attempts": 2}}

    run_dir = persist_simulation_run(
        output_root=tmp_path,
        df=df,
        metadata=metadata,
        instructor_report_md="# Instructor copy",
        engine_log=["line1", "line2"],
        validation_results={"passed": True},
    )

    assert run_dir.exists()
    expected = {
        "Simulated_Data.csv",
        "Metadata.json",
        "Instructor_Copy.md",
        "Validation_Results.json",
        "Engine_Log.txt",
        "Run_Summary.json",
        "Run_Manifest.json",
    }
    assert expected.issubset({p.name for p in run_dir.iterdir()})


def test_audit_new_runs_only_processes_new_directories(tmp_path: Path):
    state_file = tmp_path / ".state.json"
    log_file = tmp_path / "continuous_improvement_log.txt"

    df = pd.DataFrame({
        "PARTICIPANT_ID": [1, 2, 3],
        "CONDITION": ["A", "A", "B"],
        "Q_open": ["", "Not sure", "It depends"],
    })

    persist_simulation_run(
        output_root=tmp_path,
        df=df,
        metadata={"run_id": "problem-run", "llm_stats": {"llm_calls": 0, "llm_attempts": 0}},
        instructor_report_md="report",
    )

    # Hidden folders are ignored by scanner
    (tmp_path / ".hidden_dir").mkdir(exist_ok=True)

    first = audit_new_runs(output_root=tmp_path, state_file=state_file, running_log_file=log_file)
    assert first["new_run_count"] == 1
    assert first["audits"][0]["issue_count"] >= 1
    issue_codes = {i["code"] for i in first["audits"][0]["issues"]}
    assert "oe_no_llm_activity" in issue_codes
    assert "oe_too_short" in issue_codes
    assert first["error_count"] >= 1

    second = audit_new_runs(output_root=tmp_path, state_file=state_file, running_log_file=log_file)
    assert second["new_run_count"] == 0

    state = json.loads(state_file.read_text())
    assert len(state["seen_runs"]) == 1
    assert "severity=ERROR" in log_file.read_text(encoding="utf-8")
    assert (tmp_path / "audit_summary.json").exists()
    assert (tmp_path / "issue_trends.json").exists()
    assert first.get("top_issue_codes")


def test_audit_flags_missing_manifest(tmp_path: Path):
    state_file = tmp_path / ".state.json"
    log_file = tmp_path / "continuous_improvement_log.txt"

    df = pd.DataFrame({
        "PARTICIPANT_ID": [1],
        "CONDITION": ["A"],
        "Q_open": ["This is a detailed contextual response"],
    })

    run_dir = persist_simulation_run(
        output_root=tmp_path,
        df=df,
        metadata={"run_id": "manifest-missing", "llm_stats": {"llm_calls": 1, "llm_attempts": 1}},
        instructor_report_md="report",
    )

    (run_dir / "Run_Manifest.json").unlink()

    result = audit_new_runs(output_root=tmp_path, state_file=state_file, running_log_file=log_file)
    issue_codes = {i["code"] for i in result["audits"][0]["issues"]}
    assert "manifest_missing" in issue_codes


def test_issue_trends_aggregate_multiple_runs(tmp_path: Path):
    state_file = tmp_path / ".state.json"
    log_file = tmp_path / "continuous_improvement_log.txt"

    df1 = pd.DataFrame({
        "PARTICIPANT_ID": [1, 2],
        "CONDITION": ["A", "B"],
        "Q_open": ["", "Not sure"],
    })
    df2 = pd.DataFrame({
        "PARTICIPANT_ID": [3],
        "CONDITION": ["A"],
        "Q_open": ["This is a detailed contextual response"],
    })

    persist_simulation_run(
        output_root=tmp_path,
        df=df1,
        metadata={"run_id": "r1", "llm_stats": {"llm_calls": 0, "llm_attempts": 0}},
        instructor_report_md="report",
    )
    persist_simulation_run(
        output_root=tmp_path,
        df=df2,
        metadata={"run_id": "r2", "llm_stats": {"llm_calls": 1, "llm_attempts": 1}},
        instructor_report_md="report",
    )

    result = audit_new_runs(output_root=tmp_path, state_file=state_file, running_log_file=log_file)
    trends = json.loads((tmp_path / "issue_trends.json").read_text(encoding="utf-8"))

    assert result["new_run_count"] == 2
    assert trends["audited_run_count"] >= 2
    assert isinstance(trends["top_issues"], list)
