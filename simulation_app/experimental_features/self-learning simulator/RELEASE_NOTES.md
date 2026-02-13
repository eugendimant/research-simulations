# socsim 0.16.0

Focus: richer behavioral coverage (games + surveys) and more expressive benchmark targets.

New:
- Moments DSL: by_* grouped moments, gini, share ops, corr, slope
- Pack runner returns targets + provenance dicts (still writes files)
- Survey primitives: Likert, conjoint choice, list experiment, randomized response, endorsement
- Games: beauty contest, stag hunt, common pool resource, Tullock contest, bribery
- Docs: SURVEYS.md, FIELDS.md

Verify:
```bash
python scripts/verify.py
python -m pytest -q
```
