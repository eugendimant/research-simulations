"""Run the behavioral engine through its real public interfaces.

The design enters as a Qualtrics QSF (built with ``survey_builder.generate_qsf_from_design`` and parsed with
``QSFPreviewParser`` exactly as the app does), then ``app._preview_to_engine_inputs`` and
``EnhancedSimulationEngine`` are called as in REPLICATION_README.md. Nothing outcome-related is passed in.
"""
from __future__ import annotations

import copy
import os
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "simulation_app"))


def _imports():
    from utils.qsf_preview import QSFPreviewParser  # noqa: E402
    from utils.survey_builder import ParsedCondition, ParsedDesign, ParsedScale, generate_qsf_from_design  # noqa: E402
    import app  # noqa: E402  (needs streamlit installed; no server is started)
    from utils.enhanced_simulation_engine import EffectSizeSpec, EnhancedSimulationEngine  # noqa: E402
    return QSFPreviewParser, ParsedCondition, ParsedDesign, ParsedScale, generate_qsf_from_design, app, EffectSizeSpec, EnhancedSimulationEngine


def build_inputs(design: Dict) -> Dict:
    """design.json -> engine inputs, via QSF generation and parsing. Returns inputs plus a record of what the parser found."""
    (QSFPreviewParser, ParsedCondition, ParsedDesign, ParsedScale, gen_qsf, app, _E, _Eng) = _imports()
    pd = ParsedDesign(
        conditions=[ParsedCondition(c) for c in design["conditions"]],
        scales=[ParsedScale(s["label"], s["name"], 1, s["lo"], s["hi"], s["type"], s["description"]) for s in design["dvs"]],
        factors=design["factors"],
        sample_size=design["n"],
        study_title=design["title"],
        study_description=design["description"],
    )
    qsf = gen_qsf(pd)
    prev = QSFPreviewParser().parse(qsf)
    inp = app._preview_to_engine_inputs(prev)
    found = {s["variable_name"]: (s["scale_min"], s["scale_max"]) for s in inp["scales"]}
    scales = copy.deepcopy(inp["scales"])
    notes: List[str] = []
    for dv in design["dvs"]:
        if dv["name"] not in found:
            # The parser did not detect this DV from the generated QSF. Add it by hand, as a user can on the Design page.
            template = copy.deepcopy(scales[0])
            template.update(name=dv["name"], variable_name=dv["name"], scale_min=dv["lo"], scale_max=dv["hi"],
                            scale_points=dv["hi"] - dv["lo"] + 1, question_text=dv["description"], dv_description=dv["description"])
            scales.append(template)
            notes.append(f"DV '{dv['name']}' was not detected from the generated QSF and was added manually")
    if [f["name"] for f in inp["factors"]] != [f["name"] for f in design["factors"]]:
        notes.append("the parser reported one factor with four levels, not the 2x2 structure; "
                     f"the engine received the parser's factors unchanged (found: {[f['name'] for f in inp['factors']]})")
    return {"conditions": inp["conditions"], "factors": inp["factors"], "scales": scales, "notes": notes, "qsf_bytes": len(qsf)}


def run_engine(design: Dict, inputs: Dict, seed: int, auto_effects: bool, prior_d: Optional[float] = None,
               dv_polarity: Optional[Dict[str, str]] = None):
    (_Q, _PC, _PD, _PS, _g, _app, EffectSizeSpec, Engine) = _imports()
    specs = None
    if prior_d is not None:
        focal = design["focal_cell"]
        others = [c for c in design["conditions"] if c != focal]
        # d is the focal cell against each of the other cells, so that focal-vs-rest is about d (checked at N = 1,000: 0.31 for d = 0.3)
        specs = [EffectSizeSpec(variable=dv["name"], factor=inputs["factors"][0]["name"], level_high=focal, level_low=o,
                                cohens_d=abs(prior_d), direction="positive" if prior_d >= 0 else "negative")
                 for dv in design["dvs"] for o in others]
    eng = Engine(study_title=design["title"], study_description=design["description"], sample_size=design["n"],
                 conditions=inputs["conditions"], factors=inputs["factors"], scales=inputs["scales"], additional_vars=[],
                 demographics={"gender_quota": 50, "age_mean": 45, "age_sd": 15}, effect_sizes=specs,
                 seed=seed, auto_effects=auto_effects, dv_polarity=dv_polarity)
    eng.llm_generator.disable_permanently("offline")
    df, meta = eng.generate()
    return df, meta
