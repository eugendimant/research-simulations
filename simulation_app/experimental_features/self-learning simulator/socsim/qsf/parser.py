from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from datetime import datetime, timezone
import hashlib

@dataclass
class QSFParseResult:
    survey_name: str
    n_blocks: int
    n_questions: int
    blocks: List[Dict[str, Any]]
    questions: List[Dict[str, Any]]

def parse_qsf(path: Path) -> QSFParseResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    survey_name = str(data.get("SurveyEntry", {}).get("SurveyName", ""))
    elems = data.get("SurveyElements", []) or []
    blocks: List[Dict[str, Any]] = []
    questions: List[Dict[str, Any]] = []

    for el in elems:
        etype = el.get("Element")
        if etype == "BL":
            payload = el.get("Payload", {}) or {}
            block_order = payload.get("BlockOrder", []) or []
            blocks.append({"block_order": block_order, "raw": payload})
        if etype == "SQ":
            payload = el.get("Payload", {}) or {}
            qid = str(payload.get("QuestionID", ""))
            qtext = payload.get("QuestionText", "")
            qtype = payload.get("QuestionType", "")
            selector = payload.get("Selector", "")
            choices = payload.get("Choices", {}) or {}
            questions.append({
                "qid": qid,
                "question_type": qtype,
                "selector": selector,
                "text": qtext,
                "n_choices": len(choices),
            })

    return QSFParseResult(
        survey_name=survey_name,
        n_blocks=len(blocks),
        n_questions=len(questions),
        blocks=blocks,
        questions=questions,
    )

def qsf_to_atomic_units(res: QSFParseResult, source: Dict[str, Any], tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tags = tags or []
    base = {
        "kind": "instrument",
        "title": f"Qualtrics instrument: {res.survey_name}".strip(),
        "source": source,
        "tags": tags + ["qsf", "instrument"],
        "provenance": {"added_by":"socsim.qsf", "added_at_utc": retrieved, "extraction_method":"metadata_only", "notes":"QSF parsed without response text inference."},
    }
    out: List[Dict[str, Any]] = []
    out.append({**base, "id": "qsf_" + hashlib.sha256((res.survey_name or '').encode('utf-8')).hexdigest()[:16], "payload": {"n_blocks":res.n_blocks,"n_questions":res.n_questions}})
    return out
