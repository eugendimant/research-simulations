"""
SocSim Adapter — Bridge between EnhancedSimulationEngine and SocSim experimental engine.

Detects economic game DVs in the study design, maps them to socsim ExperimentSpecs,
runs socsim simulation, and enriches the main DataFrame with evidence-traceable
behavioral data grounded in Fehr-Schmidt utility models.

v1.0.8.3 — OE narrative fix.
"""
from __future__ import annotations

__version__ = "1.0.9.0"

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup — locate socsim package
# ---------------------------------------------------------------------------
_SOCSIM_BASE = Path(__file__).resolve().parent.parent / "experimental_features" / "self-learning simulator"
_SOCSIM_DATA = _SOCSIM_BASE / "socsim" / "data"

# Game keyword detection patterns — maps variable/scale names to socsim game names
_GAME_KEYWORD_MAP: Dict[str, List[str]] = {
    "dictator": [
        r"\bdictator\b", r"\bgiving\b", r"\ballocation\b", r"\bdg\b",
        r"\bendowment\b.*\bgiv", r"\bsplit\b", r"\bdonat",
    ],
    "trust": [
        r"\btrust\s*game\b", r"\btrust\s*invest", r"\btrustee\b",
        r"\btrustor\b", r"\bsend\b.*\breturn\b", r"\btrust_game\b",
    ],
    "ultimatum": [
        r"\bultimatum\b", r"\bpropos", r"\brespond.*\baccept",
        r"\breject\b.*\boffer\b", r"\bultimatum_game\b",
    ],
    "public_goods": [
        r"\bpublic\s*good", r"\bcontribut", r"\bfree\s*rid",
        r"\bcooperat.*\bgame\b", r"\bpgg\b", r"\bpublic_goods\b",
    ],
    "pd": [
        r"\bprisoner", r"\bdilemma\b", r"\bcooperate\b.*\bdefect\b",
        r"\bpd_game\b", r"\bprisoners_dilemma\b",
    ],
    "die_roll": [
        r"\bdie\s*roll", r"\bhonesty\b.*\bgame\b", r"\breport.*\bdie\b",
        r"\bdishonesty\b.*\bparadigm\b",
    ],
    "gift_exchange": [
        r"\bgift\s*exchange\b", r"\bwage\b.*\beffort\b",
        r"\breciprocity\b.*\bgame\b",
    ],
    "stag_hunt": [
        r"\bstag\s*hunt\b", r"\bcoordinat.*\bgame\b",
    ],
    "risk_holt_laury": [
        r"\bholt\s*laury\b", r"\brisk\s*elicitation\b", r"\brisk\s*prefer\b",
        r"\blottery\s*choice\b",
    ],
    "beauty_contest": [
        r"\bbeauty\s*contest\b", r"\bguessing\s*game\b", r"\bp-beauty\b",
    ],
    "bribery_game": [
        r"\bbribery\b", r"\bcorrupt", r"\bbribe\b",
    ],
    "common_pool_resource": [
        r"\bcommon\s*pool\b", r"\bresource\s*dilemma\b", r"\bharvest\b.*\bgame\b",
    ],
}

# Condition keyword → socsim topic tags
_CONDITION_TOPIC_TAGS: Dict[str, List[str]] = {
    "ingroup": ["ingroup"],
    "outgroup": ["outgroup"],
    "anonymous": ["anonymity"],
    "identified": ["identified"],
    "punishment": ["punishment"],
    "reward": ["reward"],
    "norm": ["norm_salience"],
    "high_stake": ["high_stakes"],
    "low_stake": ["low_stakes"],
    "political": ["political"],
    "racial": ["racial"],
    "gender": ["gender"],
    "partisan": ["political", "partisan"],
}

# Condition keyword → socsim context features
_CONDITION_CONTEXT_MAP: Dict[str, Dict[str, Any]] = {
    "ingroup": {"ingroup_partner": 1, "group_salience": 1},
    "outgroup": {"ingroup_partner": 0, "group_salience": 1},
    "anonymous": {"anonymity": True},
    "identified": {"anonymity": False, "social_image": 1},
    "punishment": {"punishment_available": True},
    "norm": {"norm_salience": 1},
    "control": {},
}


def detect_game_dvs(
    scales: List[Dict[str, Any]],
    study_title: str = "",
    study_description: str = "",
    conditions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Detect which scales/DVs correspond to economic games that socsim can simulate.

    Returns a list of dicts with:
      - scale: the original scale dict
      - game_name: socsim game name (e.g., "dictator", "trust")
      - confidence: float 0-1 indicating detection confidence
    """
    detected: List[Dict[str, Any]] = []
    _context_text = f"{study_title} {study_description} {' '.join(conditions or [])}".lower()

    for scale in scales:
        _var = str(scale.get("variable_name", scale.get("name", ""))).lower()
        _name = str(scale.get("name", "")).lower()
        _search_text = f"{_var} {_name} {_context_text}"

        best_game = None
        best_conf = 0.0

        for game_name, patterns in _GAME_KEYWORD_MAP.items():
            for pattern in patterns:
                if re.search(pattern, _search_text, re.IGNORECASE):
                    # Higher confidence if match is in the variable name itself
                    conf = 0.85 if re.search(pattern, f"{_var} {_name}", re.IGNORECASE) else 0.55
                    if conf > best_conf:
                        best_conf = conf
                        best_game = game_name

        if best_game and best_conf >= 0.5:
            detected.append({
                "scale": scale,
                "game_name": best_game,
                "confidence": best_conf,
            })

    return detected


def _build_experiment_spec(
    game_name: str,
    condition: str,
    study_title: str = "",
    study_description: str = "",
    scale: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a socsim ExperimentSpec dict from the study context.
    Returns a plain dict (not the dataclass) for flexibility.
    """
    # Extract topic from study context
    _text = f"{study_title} {study_description} {condition}".lower()
    topic_name = "general"
    topic_tags: List[str] = []
    context: Dict[str, Any] = {"anonymity": True}

    # Detect topic tags from condition text
    _cond_lower = condition.lower()
    for keyword, tags in _CONDITION_TOPIC_TAGS.items():
        if keyword in _cond_lower:
            topic_tags.extend(tags)

    # Detect context features from condition text
    for keyword, ctx_features in _CONDITION_CONTEXT_MAP.items():
        if keyword in _cond_lower:
            context.update(ctx_features)

    # Detect topic from study domain
    if any(w in _text for w in ("identity", "partisan", "political")):
        topic_name = "identity"
    elif any(w in _text for w in ("racial", "ethnic", "race")):
        topic_name = "intergroup_racial"
    elif any(w in _text for w in ("gender", "sex")):
        topic_name = "gender"
    elif any(w in _text for w in ("trust", "cooperation")):
        topic_name = "trust_cooperation"
    elif any(w in _text for w in ("fairness", "justice")):
        topic_name = "fairness"
    elif any(w in _text for w in ("norm", "social norm")):
        topic_name = "norms"

    # Game-specific default parameters
    game_params: Dict[str, Any] = {}
    if game_name == "dictator":
        # Try to detect endowment from scale
        _max_val = 10
        if scale:
            _max_val = int(scale.get("max_value", scale.get("scale_max", 10)))
            if _max_val < 2:
                _max_val = 10
        game_params = {"endowment": _max_val, "step": 1, "norm_target_share": 0.5}
    elif game_name == "trust":
        _max_val = 10
        if scale:
            _max_val = int(scale.get("max_value", scale.get("scale_max", 10)))
            if _max_val < 2:
                _max_val = 10
        game_params = {"endowment": _max_val, "multiplier": 3}
    elif game_name == "ultimatum":
        game_params = {"endowment": 10, "step": 1}
    elif game_name == "public_goods":
        game_params = {"endowment": 10, "mpcr": 0.5, "group_size": 4}
    elif game_name == "pd":
        game_params = {"R": 3, "T": 5, "S": 0, "P": 1}
    elif game_name == "die_roll":
        game_params = {"sides": 6, "pay_per_pip": 0.5}
    elif game_name == "risk_holt_laury":
        game_params = {}
    elif game_name == "stag_hunt":
        game_params = {"stag_payoff": 4, "hare_payoff": 3, "fail_payoff": 0}
    elif game_name == "gift_exchange":
        game_params = {"max_wage": 10, "min_wage": 1, "effort_levels": 5}
    elif game_name == "beauty_contest":
        game_params = {"p": 0.67}
    elif game_name == "bribery_game":
        game_params = {"endowment": 10}
    elif game_name == "common_pool_resource":
        game_params = {"resource_pool": 100, "group_size": 4}

    return {
        "game": {"name": game_name, "params": game_params},
        "topic": {"name": topic_name, "tags": list(set(topic_tags))},
        "context": context,
        "population": {"label": "adult_online"},
    }


def run_socsim_enrichment(
    df: "pd.DataFrame",
    game_dvs: List[Dict[str, Any]],
    conditions: List[str],
    study_title: str = "",
    study_description: str = "",
    sample_size: int = 200,
    seed: int = 42,
    progress_callback: Optional[callable] = None,
) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    """
    Run socsim simulation for detected game DVs and enrich the DataFrame.

    Args:
        df: Main simulation DataFrame from EnhancedSimulationEngine
        game_dvs: List of detected game DVs from detect_game_dvs()
        conditions: List of condition names
        study_title: Study title for context
        study_description: Study description for context
        sample_size: Total sample size
        seed: Random seed
        progress_callback: Optional callback(phase, current, total)

    Returns:
        Tuple of (enriched_df, socsim_metadata)
    """
    import pandas as pd

    socsim_metadata: Dict[str, Any] = {
        "socsim_used": True,
        "socsim_version": "0.16.0",
        "enriched_dvs": [],
        "games_simulated": [],
        "errors": [],
        "traces": {},
    }

    if not game_dvs:
        socsim_metadata["socsim_used"] = False
        return df, socsim_metadata

    # Ensure socsim is importable
    socsim_pkg_path = str(_SOCSIM_BASE)
    if socsim_pkg_path not in sys.path:
        sys.path.insert(0, socsim_pkg_path)

    try:
        from socsim.simulator import simulate, SimulationResult
        from socsim.specs import ExperimentSpec, GameSpec, TopicSpec, PopulationSpec
    except ImportError as e:
        logger.warning(f"SocSim import failed: {e}")
        socsim_metadata["errors"].append(f"Import failed: {e}")
        socsim_metadata["socsim_used"] = False
        return df, socsim_metadata

    # Check data files exist
    priors_path = _SOCSIM_DATA / "priors.json"
    latent_classes_path = _SOCSIM_DATA / "latent_classes.json"
    evidence_store_path = _SOCSIM_DATA / "evidence_store.json"
    schema_path = _SOCSIM_BASE / "socsim" / "schema" / "evidence_unit_schema.json"

    if not priors_path.exists():
        socsim_metadata["errors"].append(f"Priors file not found: {priors_path}")
        socsim_metadata["socsim_used"] = False
        return df, socsim_metadata

    _total_tasks = len(game_dvs) * len(conditions)
    _current_task = 0

    def _report(current: int, total: int) -> None:
        if progress_callback:
            try:
                progress_callback("socsim_enrichment", current, total)
            except Exception:
                pass

    _report(0, _total_tasks)

    for game_dv in game_dvs:
        scale = game_dv["scale"]
        game_name = game_dv["game_name"]
        var_name = scale.get("variable_name", scale.get("name", ""))

        for cond_idx, condition in enumerate(conditions):
            _current_task += 1
            _report(_current_task, _total_tasks)

            try:
                # Build experiment spec for this game × condition
                spec_dict = _build_experiment_spec(
                    game_name=game_name,
                    condition=condition,
                    study_title=study_title,
                    study_description=study_description,
                    scale=scale,
                )

                spec = ExperimentSpec(
                    game=GameSpec(**spec_dict["game"]),
                    topic=TopicSpec(**spec_dict["topic"]),
                    context=spec_dict["context"],
                    population=PopulationSpec(**spec_dict["population"]),
                )

                # How many participants in this condition?
                cond_mask = df["CONDITION"] == condition
                n_cond = int(cond_mask.sum())
                if n_cond <= 0:
                    continue

                # Run socsim simulation
                result: SimulationResult = simulate(
                    spec=spec,
                    n=n_cond,
                    seed=seed + hash(condition) % 10000,
                    priors_path=priors_path,
                    latent_classes_path=latent_classes_path,
                    evidence_store_path=evidence_store_path,
                    schema_path=schema_path if schema_path.exists() else None,
                    return_traces=True,
                    safe_mode=True,
                )

                if not result.rows:
                    continue

                # Map socsim output to the main DataFrame
                # Primary action variable — the main DV
                _primary_action_key = _get_primary_action(game_name, result.rows[0])
                if _primary_action_key:
                    # Get socsim values for this condition's participants
                    socsim_values = [
                        float(row.get(_primary_action_key, 0.0))
                        for row in result.rows[:n_cond]
                    ]

                    # Scale socsim values to match the DV's scale range
                    scale_min = float(scale.get("scale_min", scale.get("min_value", 1)))
                    scale_max = float(scale.get("scale_max", scale.get("max_value", 7)))
                    game_max = float(spec_dict["game"]["params"].get("endowment", 10))

                    # v1.0.8.6: Detect bipolar scale (e.g., -100 to +100 for taking games)
                    # SocSim produces values in [0, game_max] but for taking games
                    # the target range includes negative values
                    _src_min = 0.0
                    if scale_min < 0:
                        # Bipolar target: socsim may need to produce negative values
                        # If socsim output has values near 0, they map to taking behavior
                        _src_min = -game_max if any(v < 0 for v in socsim_values) else 0.0

                    # Normalize socsim output to [scale_min, scale_max]
                    scaled_values = _scale_to_range(
                        socsim_values, _src_min, game_max, scale_min, scale_max
                    )

                    # Find the actual column(s) in df that correspond to this scale
                    matching_cols = _find_scale_columns(df, var_name, scale)

                    for col in matching_cols:
                        if col in df.columns:
                            # Update only participants in this condition
                            cond_indices = df.index[cond_mask].tolist()
                            for i, idx in enumerate(cond_indices):
                                if i < len(scaled_values):
                                    df.at[idx, col] = scaled_values[i]

                    socsim_metadata["enriched_dvs"].append({
                        "variable": var_name,
                        "game": game_name,
                        "condition": condition,
                        "n_enriched": min(n_cond, len(socsim_values)),
                        "action_key": _primary_action_key,
                        "action_mean": float(np.mean(socsim_values)),
                    })

                # Store traces
                socsim_metadata["traces"][f"{game_name}_{condition}"] = {
                    "summary": result.summary,
                    "sample_trace": json.loads(result.rows[0].get("trace_json", "{}")) if result.rows else {},
                }

                # Store latent class info
                socsim_metadata["games_simulated"].append({
                    "game": game_name,
                    "condition": condition,
                    "n": n_cond,
                    "latent_class_counts": result.summary.get("latent_class_counts", {}),
                    "action_means": result.summary.get("action_means", {}),
                })

            except Exception as e:
                logger.warning(f"SocSim enrichment failed for {game_name}/{condition}: {e}")
                socsim_metadata["errors"].append(f"{game_name}/{condition}: {e}")

    _report(_total_tasks, _total_tasks)
    return df, socsim_metadata


def _get_primary_action(game_name: str, sample_row: Dict[str, Any]) -> Optional[str]:
    """Get the primary action key from a socsim result row."""
    # Priority mapping for each game
    _priority = {
        "dictator": ["act::give", "act::share"],
        "trust": ["act::invest", "act::send"],
        "ultimatum": ["act::offer", "act::propose"],
        "public_goods": ["act::contribute"],
        "pd": ["act::cooperate", "act::choice"],
        "die_roll": ["act::report", "act::claim"],
        "gift_exchange": ["act::wage", "act::effort"],
        "stag_hunt": ["act::choice", "act::cooperate"],
        "risk_holt_laury": ["act::switch_row", "act::n_safe"],
        "beauty_contest": ["act::guess"],
        "bribery_game": ["act::bribe", "act::transfer"],
        "common_pool_resource": ["act::harvest", "act::extract"],
    }

    priorities = _priority.get(game_name, [])
    for key in priorities:
        if key in sample_row:
            return key

    # Fallback: first act:: key
    for key in sorted(sample_row.keys()):
        if key.startswith("act::"):
            return key

    return None


def _scale_to_range(
    values: List[float],
    src_min: float,
    src_max: float,
    dst_min: float,
    dst_max: float,
) -> List[float]:
    """Scale values from [src_min, src_max] to [dst_min, dst_max]."""
    if src_max == src_min:
        return [float((dst_min + dst_max) / 2)] * len(values)

    result = []
    for v in values:
        # Clamp to source range
        v_clamped = max(src_min, min(src_max, float(v)))
        # Normalize to [0, 1]
        normalized = (v_clamped - src_min) / (src_max - src_min)
        # Scale to destination range
        scaled = dst_min + normalized * (dst_max - dst_min)
        # Round to nearest integer if destination is integer-like
        if dst_min == int(dst_min) and dst_max == int(dst_max):
            scaled = round(scaled)
        result.append(scaled)

    return result


def _find_scale_columns(
    df: "pd.DataFrame",
    var_name: str,
    scale: Dict[str, Any],
) -> List[str]:
    """Find DataFrame columns that belong to a given scale."""
    matching = []
    n_items = int(scale.get("num_items", scale.get("n_items", 1)))

    if n_items <= 1:
        # Single-item DV — look for exact match or close match
        for col in df.columns:
            if col == var_name or col.lower() == var_name.lower():
                matching.append(col)
                break
        if not matching:
            # Try partial match
            for col in df.columns:
                if var_name.lower() in col.lower():
                    matching.append(col)
                    break
    else:
        # Multi-item scale — look for var_name_1, var_name_2, etc.
        for item_idx in range(1, n_items + 1):
            candidates = [
                f"{var_name}_{item_idx}",
                f"{var_name}_{item_idx:02d}",
                f"{var_name}_Item_{item_idx}",
            ]
            for cand in candidates:
                if cand in df.columns:
                    matching.append(cand)
                    break

    return matching


def get_available_games() -> List[str]:
    """Return list of available socsim game names."""
    return sorted(_GAME_KEYWORD_MAP.keys())


def get_game_description(game_name: str) -> str:
    """Return a human-readable description of a socsim game."""
    _descriptions = {
        "dictator": "Dictator Game — one player divides an endowment with a passive recipient. Based on Engel (2011) meta-analysis: mean giving ~28%.",
        "trust": "Trust Game — trustor sends money (multiplied), trustee decides return. Based on Berg et al. (1995): ~50% sent, ~33% returned.",
        "ultimatum": "Ultimatum Game — proposer offers a split; responder accepts or rejects. Based on Güth et al. (1982): modal offer ~40-50%.",
        "public_goods": "Public Goods Game — players contribute to a shared pool (multiplied by MPCR). Based on Zelmer (2003) meta-analysis: ~40-60% contributed.",
        "pd": "Prisoner's Dilemma — simultaneous cooperate/defect choice. Based on Sally (1995) meta-analysis: ~47% cooperation rate.",
        "die_roll": "Die-Rolling Task — self-reported outcome for payment (honesty paradigm). Based on Abeler et al. (2019) meta-analysis.",
        "gift_exchange": "Gift Exchange Game — employer sets wage, worker chooses effort. Based on Fehr et al. (1993): reciprocity in labor markets.",
        "stag_hunt": "Stag Hunt — coordination game with payoff-dominant vs. risk-dominant equilibria.",
        "risk_holt_laury": "Holt-Laury Risk Elicitation — lottery choice task measuring risk preferences. Based on Holt & Laury (2002).",
        "beauty_contest": "Beauty Contest / Guessing Game — guess 2/3 of average. Measures strategic depth (Nagel, 1995).",
        "bribery_game": "Bribery Game — models corruption decisions with transfers and penalties.",
        "common_pool_resource": "Common Pool Resource — harvest from a shared resource with depletion risk.",
    }
    return _descriptions.get(game_name, f"Economic game: {game_name}")
