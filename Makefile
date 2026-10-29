# Reviewer entry points. Python 3.12 or 3.13; `make install` or `pip install -r simulation_app/requirements.txt pytest scipy` first.
PYTHON ?= python3
SAMPLE_QSF ?= simulation_app/example_files/Coffee_Shop_Loyalty_Programs.qsf
OUTDIR ?= reproducibility_output
SEED ?= 7
SEEDS ?= 40

.PHONY: install reproduce audit-version smoke benchmark test-fast clean-reproduce

# 0. Install the pinned versions the checks were run under (Python 3.12).
install:
	$(PYTHON) -m pip install -r requirements-reproduce.txt

# 1. The determinism and provenance demonstration (about 15 seconds, offline).
reproduce:
	$(PYTHON) scripts/reproduce_reference_run.py --qsf $(SAMPLE_QSF) --seed $(SEED) --outdir $(OUTDIR)

# 2. Version strings agree across all enforced locations.
audit-version:
	$(PYTHON) scripts/check_version_sync.py

# 3. The fast release gates a reviewer can run in under two minutes.
smoke:
	$(PYTHON) -m pytest tests/test_app_import_safety.py tests/test_qualtrics_export.py tests/test_reproducibility_v1288.py tests/test_version_sync.py tests/test_declare_first_v1308.py tests/test_p1_fixes_v1309.py tests/test_user_download_secrecy.py tests/test_no_secrets_in_repo.py -q -p no:cacheprovider

# 4. The retrospective benchmark against public human data (needs network once; needs scipy).
benchmark:
	$(PYTHON) -m benchmarks.run_benchmark brader_valentino_suhay_2008 --seeds $(SEEDS) --out benchmarks/results
	$(PYTHON) -m benchmarks.report benchmarks/results/brader_valentino_suhay_2008.json

# 5. The full fast suite (the PR job in .github/workflows runs the same selection).
test-fast:
	$(PYTHON) -m pytest tests -m "not slow" -q -p no:cacheprovider

clean-reproduce:
	rm -rf $(OUTDIR)
