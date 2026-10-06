# Tests

All tests are plain pytest (config in `pyproject.toml`). Every check can genuinely fail.

```bash
python3 -m pytest tests/ -q -m "not slow"   # default / CI suite (a few minutes)
python3 -m pytest tests/ -q -m slow         # slow suite: all 302 example QSFs, AI-path generation (long)
python3 -m pytest tests/ -q                 # everything
python3 scripts/check_version_sync.py       # version locations from CLAUDE.md must match
ruff check simulation_app tests scripts --exclude simulation_app/experimental_features
```

Notes
- Several files started life as print-and-exit scripts. They are now pytest-native: the old
  script body lives in `_run_all()` / `main()` and a `test_*` function runs it once and asserts
  there are no failures. `python3 tests/<file>.py` still works as a script.
- `slow` marker: file-wide QSF sweeps (`test_simulation_stress`, `test_qsf_simulation_match`) and the
  AI-provider paths (`test_all_methods`, `test_progress_callbacks`) which wait on unreachable LLM
  providers when there is no network/API key. The fast variants use the smallest example QSFs.
- Non-`test_` helper scripts (`effect_fuzz.py`, `qsf_e2e_sim.py`, `qsf_robustness.py`,
  `random_qsf_n200.py`, `smoke_sim.py`, `student_qsf_inspect.py`) are run manually:
  `python3 tests/smoke_sim.py`. They are not collected by pytest.
