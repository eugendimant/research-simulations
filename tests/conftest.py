"""
Shared test configuration and path setup.

All test files in this directory import from simulation_app/utils/.
This conftest.py adds simulation_app/ to sys.path once, so individual
test files don't need their own path manipulation.
"""

import os
import sys

# Add simulation_app/ to the Python path so `from utils.X import Y` works
_SIMULATION_APP_DIR = os.path.join(os.path.dirname(__file__), "..", "simulation_app")
if _SIMULATION_APP_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_SIMULATION_APP_DIR))

# Also expose the project root for any tests that need it
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIMULATION_APP_DIR = os.path.abspath(_SIMULATION_APP_DIR)
EXAMPLE_FILES_DIR = os.path.join(SIMULATION_APP_DIR, "example_files")
