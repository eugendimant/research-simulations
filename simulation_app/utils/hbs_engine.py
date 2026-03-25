"""
Human Behavior Simulator (HBS) Engine — Orchestrator
=====================================================

Wraps the existing EnhancedSimulationEngine with DANEEL-grade enhancements:
- Census-weighted demographic persona coherence
- Calibrated human error distributions (education-stratified)
- Stylometric voice fingerprinting for all open-ended responses
- Adversarial self-validation pipeline with auto-correction
- Question classification for tool-dispatched responses

Architecture: HBSEngine delegates numeric generation entirely to ABE 2.0
(EnhancedSimulationEngine) and applies HBS-specific post-processing to
the generated DataFrame — enriching demographics, applying stylometric
fingerprints to OE columns, calibrating error patterns, and validating
the output against DANEEL benchmark axes.

References:
    - DANEEL (2025 PNAS): Cross-page persona memory, demographic coherence
    - DANEEL+ (2026 Nature): Programmatic tool-use, 81/81 bot-trap pass rate
    - Frederick (2005): CRT calibrated error rates
    - Pennebaker & King (1999): LIWC stylometric features
    - Meade & Craig (2012): Careless responding detection
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["HBSEngine"]


class HBSEngine:
    """
    Human Behavior Simulator — wraps EnhancedSimulationEngine with
    DANEEL-grade persona coherence, calibrated errors, stylometric
    fingerprinting, and adversarial self-validation.

    Usage:
        engine = HBSEngine(**same_params_as_EnhancedSimulationEngine)
        df, metadata = engine.generate()
    """

    def __init__(
        self,
        # All parameters are forwarded to EnhancedSimulationEngine
        study_title: str = "",
        study_description: str = "",
        sample_size: int = 30,
        conditions: Optional[List[str]] = None,
        factors: Optional[List[Dict[str, Any]]] = None,
        scales: Optional[List[Dict[str, Any]]] = None,
        additional_vars: Optional[List[Dict[str, Any]]] = None,
        demographics: Optional[Dict[str, Any]] = None,
        attention_rate: float = 0.95,
        random_responder_rate: float = 0.05,
        effect_sizes: Optional[list] = None,
        exclusion_criteria: Optional[Any] = None,
        custom_persona_weights: Optional[Dict[str, float]] = None,
        open_ended_questions: Optional[List[Dict[str, Any]]] = None,
        study_context: Optional[Dict[str, Any]] = None,
        stimulus_evaluations: Optional[List[Dict[str, Any]]] = None,
        condition_allocation: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        mode: str = "pilot",
        precomputed_visibility: Optional[Dict[str, Dict[str, bool]]] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        missing_data_rate: float = 0.0,
        dropout_rate: float = 0.0,
        missing_data_mechanism: str = "realistic",
        allow_template_fallback: bool = True,
        progress_callback: Optional[Callable] = None,
        use_socsim_experimental: bool = True,
        use_abe_v2: bool = True,
        free_llm_oe_cap: int = 0,
    ):
        self._init_params = {
            "study_title": study_title,
            "study_description": study_description,
            "sample_size": sample_size,
            "conditions": conditions or [],
            "factors": factors or [],
            "scales": scales or [],
            "additional_vars": additional_vars or [],
            "demographics": demographics or {},
            "attention_rate": attention_rate,
            "random_responder_rate": random_responder_rate,
            "effect_sizes": effect_sizes,
            "exclusion_criteria": exclusion_criteria,
            "custom_persona_weights": custom_persona_weights,
            "open_ended_questions": open_ended_questions,
            "study_context": study_context,
            "stimulus_evaluations": stimulus_evaluations,
            "condition_allocation": condition_allocation,
            "seed": seed,
            "mode": mode,
            "precomputed_visibility": precomputed_visibility,
            "correlation_matrix": correlation_matrix,
            "missing_data_rate": missing_data_rate,
            "dropout_rate": dropout_rate,
            "missing_data_mechanism": missing_data_mechanism,
            "allow_template_fallback": allow_template_fallback,
            "progress_callback": progress_callback,
            "use_socsim_experimental": True,  # HBS always uses ABE 2.0 for numeric
            "use_abe_v2": True,  # HBS uses ABE v2 narrative for OE
            "free_llm_oe_cap": free_llm_oe_cap,
        }

        self.sample_size = sample_size
        self.progress_callback = progress_callback
        self._seed = seed or int(time.time()) % (2**31)
        self._rng = random.Random(self._seed)

        # --- Initialize HBS sub-modules (lazy, fault-tolerant) ---
        self._participant_states: list = []
        self._stylometric_engine = None
        self._error_calibrator = None
        self._validator = None
        self._question_classifier = None

        try:
            from utils.hbs_participant_state import HBSParticipantFactory
            self._participant_factory = HBSParticipantFactory()
        except Exception as e:
            logger.warning("HBS ParticipantFactory unavailable: %s", e)
            self._participant_factory = None

        try:
            from utils.hbs_stylometric_engine import HBSStylometricEngine
            self._stylometric_engine = HBSStylometricEngine()
        except Exception as e:
            logger.warning("HBS StylometricEngine unavailable: %s", e)

        try:
            from utils.hbs_error_calibrator import HBSErrorCalibrator
            self._error_calibrator = HBSErrorCalibrator()
        except Exception as e:
            logger.warning("HBS ErrorCalibrator unavailable: %s", e)

        try:
            from utils.hbs_validator import HBSValidator
            self._validator = HBSValidator()
        except Exception as e:
            logger.warning("HBS Validator unavailable: %s", e)

        try:
            from utils.hbs_question_classifier import HBSQuestionClassifier
            self._question_classifier = HBSQuestionClassifier()
        except Exception as e:
            logger.warning("HBS QuestionClassifier unavailable: %s", e)

        # --- Initialize the base engine ---
        from utils.enhanced_simulation_engine import EnhancedSimulationEngine
        self._base_engine = EnhancedSimulationEngine(**self._init_params)

        # Proxy key attributes that app.py reads
        self.llm_generator = getattr(self._base_engine, "llm_generator", None)
        self.validation_log = getattr(self._base_engine, "validation_log", [])
        self.run_id = getattr(self._base_engine, "run_id", "")
        self.detected_domains = getattr(self._base_engine, "detected_domains", [])
        self.column_info = getattr(self._base_engine, "column_info", [])
        self.conditions = getattr(self._base_engine, "conditions", [])
        self.open_ended_questions = getattr(self._base_engine, "open_ended_questions", [])

        logger.info(
            "HBSEngine initialized: N=%d, %d conditions, %d HBS modules active",
            sample_size,
            len(self.conditions),
            sum(1 for m in [
                self._participant_factory,
                self._stylometric_engine,
                self._error_calibrator,
                self._validator,
                self._question_classifier,
            ] if m is not None),
        )

    # ------------------------------------------------------------------
    # Proxy properties that app.py accesses on the engine
    # ------------------------------------------------------------------

    @property
    def _oe_budget_exceeded(self) -> bool:
        return getattr(self._base_engine, "_oe_budget_exceeded", False)

    @_oe_budget_exceeded.setter
    def _oe_budget_exceeded(self, value: bool) -> None:
        self._base_engine._oe_budget_exceeded = value

    @property
    def _oe_budget_switched_count(self) -> int:
        return getattr(self._base_engine, "_oe_budget_switched_count", 0)

    @property
    def _generation_source_map(self) -> dict:
        return getattr(self._base_engine, "_generation_source_map", {})

    @property
    def _last_oe_source(self) -> str:
        return getattr(self._base_engine, "_last_oe_source", "Template")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate simulated dataset with HBS enhancements.

        Pipeline:
        1. Create HBS participant states (census-weighted demographics)
        2. Delegate to EnhancedSimulationEngine for core generation
        3. Enrich demographics with HBS census-weighted profiles
        4. Apply stylometric fingerprinting to OE columns
        5. Run adversarial self-validation + auto-correction
        6. Return enhanced DataFrame + metadata
        """
        _hbs_start = time.time()

        # Step 1: Create HBS participant states
        self._create_participant_states()

        # Step 2: Delegate to ABE 2.0 for core generation
        df, metadata = self._base_engine.generate()

        # Re-sync proxied attributes after generation
        self.validation_log = getattr(self._base_engine, "validation_log", [])
        self.column_info = getattr(self._base_engine, "column_info", [])

        # Step 3: Enrich demographics
        df = self._enrich_demographics(df)

        # Step 4: Apply stylometric fingerprints to OE columns
        df = self._apply_stylometric_fingerprints(df)

        # Step 5: Run self-validation + auto-correction
        df, validation_report = self._run_validation(df)

        # Step 6: Add HBS metadata
        _hbs_elapsed = time.time() - _hbs_start
        metadata["hbs_engine"] = {
            "enabled": True,
            "version": "1.0.0",
            "participant_states_created": len(self._participant_states),
            "stylometric_engine_active": self._stylometric_engine is not None,
            "error_calibrator_active": self._error_calibrator is not None,
            "validator_active": self._validator is not None,
            "validation_report": validation_report,
            "hbs_processing_time_seconds": round(_hbs_elapsed, 2),
        }

        logger.info(
            "HBS generation complete: %d rows, %.1fs total HBS processing",
            len(df), _hbs_elapsed,
        )

        return df, metadata

    # ------------------------------------------------------------------
    # Step 1: Create participant states
    # ------------------------------------------------------------------

    def _create_participant_states(self) -> None:
        """Create census-weighted HBS participant states for each participant."""
        if self._participant_factory is None:
            logger.info("HBS ParticipantFactory not available — skipping participant state creation")
            return

        try:
            _domain = ""
            if self.detected_domains:
                _domain = self.detected_domains[0] if isinstance(self.detected_domains, list) else str(self.detected_domains)

            self._participant_states = self._participant_factory.create_batch(
                n=self.sample_size,
                conditions=self.conditions,
                domain=_domain,
                seed=self._seed,
            )
            logger.info("Created %d HBS participant states", len(self._participant_states))
        except Exception as e:
            logger.warning("HBS participant state creation failed: %s", e)
            self._participant_states = []

    # ------------------------------------------------------------------
    # Step 3: Enrich demographics
    # ------------------------------------------------------------------

    def _enrich_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add HBS census-weighted demographic columns to the DataFrame.

        Adds columns like HBS_Education, HBS_Income, HBS_PartyID, HBS_Ideology,
        HBS_State, HBS_Region — only if participant states were created.
        These provide richer demographic profiles than the base engine's
        simple Age/Gender columns.
        """
        if not self._participant_states:
            return df

        n = len(df)
        n_states = len(self._participant_states)

        if n_states < n:
            logger.warning(
                "HBS has %d participant states but DataFrame has %d rows — "
                "padding with recycled states", n_states, n,
            )

        # Map participant states to rows (cycle if needed)
        _get_state = lambda i: self._participant_states[i % n_states]

        # Add HBS demographic columns
        _hbs_cols = {
            "HBS_Education": [_get_state(i).education_level for i in range(n)],
            "HBS_Income": [_get_state(i).income_bracket for i in range(n)],
            "HBS_PartyID": [_get_state(i).party_id for i in range(n)],
            "HBS_Ideology": [round(_get_state(i).ideology, 2) for i in range(n)],
            "HBS_State": [_get_state(i).state for i in range(n)],
            "HBS_Region": [_get_state(i).region for i in range(n)],
            "HBS_ResponseStyle": [_get_state(i).response_style for i in range(n)],
        }

        for col_name, col_data in _hbs_cols.items():
            df[col_name] = col_data

        # Add column info
        self.column_info.extend([
            ("HBS_Education", "HBS census-weighted education level"),
            ("HBS_Income", "HBS census-weighted income bracket"),
            ("HBS_PartyID", "HBS 7-point party identification"),
            ("HBS_Ideology", "HBS ideology score (-3.0 liberal to +3.0 conservative)"),
            ("HBS_State", "HBS U.S. state (2-letter code)"),
            ("HBS_Region", "HBS U.S. region (Northeast/South/Midwest/West)"),
            ("HBS_ResponseStyle", "HBS response style (Krosnick taxonomy)"),
        ])

        logger.info("Enriched DataFrame with %d HBS demographic columns", len(_hbs_cols))
        return df

    # ------------------------------------------------------------------
    # Step 4: Apply stylometric fingerprints
    # ------------------------------------------------------------------

    def _apply_stylometric_fingerprints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stylometric voice fingerprinting to all open-ended text columns.

        Each participant gets a consistent writing fingerprint (sentence length,
        vocab richness, typo patterns, filler words) that is applied across
        all their OE responses. This ensures the same participant sounds like
        the same person across all questions.
        """
        if self._stylometric_engine is None:
            logger.info("HBS StylometricEngine not available — skipping fingerprinting")
            return df

        if not self._participant_states:
            logger.info("No HBS participant states — skipping fingerprinting")
            return df

        # Identify OE columns (string dtype, substantial text)
        oe_cols = []
        for col in df.columns:
            if df[col].dtype == object:
                # Check if it looks like OE text (not IDs, conditions, etc.)
                _sample = df[col].dropna().head(10)
                if len(_sample) > 0:
                    _avg_len = _sample.str.len().mean()
                    if _avg_len > 15:  # Likely OE text, not short labels
                        oe_cols.append(col)

        if not oe_cols:
            logger.info("No OE columns detected — skipping fingerprinting")
            return df

        n = len(df)
        n_states = len(self._participant_states)
        _fingerprints: Dict[int, Any] = {}
        _applied_count = 0

        for i in range(n):
            state = self._participant_states[i % n_states]

            # Build or retrieve fingerprint for this participant
            if i not in _fingerprints:
                try:
                    fp = self._stylometric_engine.build_fingerprint(
                        state, rng=random.Random(self._seed + i),
                    )
                    _fingerprints[i] = fp
                except Exception as e:
                    logger.debug("Fingerprint build failed for participant %d: %s", i, e)
                    continue

            fp = _fingerprints[i]

            # Apply fingerprint to each OE column for this participant
            for col in oe_cols:
                text = df.at[i, col]
                if not isinstance(text, str) or len(text.strip()) < 5:
                    continue

                try:
                    modified = self._stylometric_engine.apply_fingerprint(
                        text, fp, rng=random.Random(self._seed + i + hash(col)),
                    )
                    if modified and modified.strip():
                        df.at[i, col] = modified
                        _applied_count += 1
                except Exception as e:
                    logger.debug(
                        "Fingerprint application failed for P%d col=%s: %s",
                        i, col, e,
                    )

        if _applied_count > 0:
            logger.info(
                "Applied stylometric fingerprints: %d OE responses across %d columns",
                _applied_count, len(oe_cols),
            )

        return df

    # ------------------------------------------------------------------
    # Step 5: Self-validation
    # ------------------------------------------------------------------

    def _run_validation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Run adversarial self-validation on the generated dataset.

        Checks DANEEL benchmark axes (completion time, attention rates,
        OE uniqueness, straight-lining, rating-text coherence) and
        auto-corrects any out-of-range metrics.
        """
        if self._validator is None:
            logger.info("HBS Validator not available — skipping validation")
            return df, {"skipped": True, "reason": "validator_unavailable"}

        try:
            df, report = self._validator.validate_and_correct(
                df, self._participant_states,
            )
            _passed = report.get("passed", False)
            _n_checks = len(report.get("checks", {}))
            _n_corrections = len(report.get("auto_corrections", []))
            logger.info(
                "HBS validation: %s (%d checks, %d auto-corrections)",
                "PASSED" if _passed else "CORRECTED",
                _n_checks, _n_corrections,
            )
            return df, report
        except Exception as e:
            logger.warning("HBS validation failed: %s", e)
            return df, {"skipped": True, "reason": str(e)}

    # ------------------------------------------------------------------
    # Compatibility: attributes app.py expects on the engine
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the base engine for compatibility."""
        if name.startswith("_") and name not in (
            "_oe_budget_exceeded",
            "_oe_budget_switched_count",
            "_generation_source_map",
            "_last_oe_source",
            "_llm_disabled_by_oe_cap",
        ):
            raise AttributeError(name)
        return getattr(self._base_engine, name)
