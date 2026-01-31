# simulation_app/utils/simulation_engine.py
from __future__ import annotations
"""
Legacy-compatible SimulationEngine wrapper.

Some older parts of the app import `SimulationEngine` from `utils.simulation_engine`.
To keep the codebase stable, we provide a thin wrapper around `EnhancedSimulationEngine`.
"""

# Version identifier to help track deployed code
__version__ = "2.1.1"  # Synced with app.py

from .enhanced_simulation_engine import EnhancedSimulationEngine, EffectSizeSpec, ExclusionCriteria


class SimulationEngine(EnhancedSimulationEngine):
    """Alias/wrapper for backward compatibility."""
    pass


__all__ = ["SimulationEngine", "EffectSizeSpec", "ExclusionCriteria"]
