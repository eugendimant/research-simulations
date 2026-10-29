# Acceptable use

This tool generates synthetic rehearsal data. Its outputs are not human observations.

## Allowed uses

- Testing survey exports and codebooks.
- Rehearsing data-cleaning and exclusion rules.
- Testing analysis scripts before human data collection.
- Teaching research methods with realistic but synthetic data.
- Benchmarking synthetic rehearsal against human data with transparent reporting.

## Disallowed uses

- Presenting synthetic outputs as collected human data.
- Using synthetic outputs as evidence for claims about real people.
- Removing simulation labels or provenance metadata.
- Creating fake support for a preferred result.
- Misrepresenting benchmark failures as successes.

## Required labeling

Any table, figure, or report based on synthetic outputs should say that the data are simulated.

## What the tool does

The user's download names the data file `Simulated_Data.csv` and carries a README that states the responses are simulated, not collected from people (`utils/user_package.py`). Every run is also archived on the owner's side with its run ID, seed, version, generation mode and a per-outcome calibration-provenance table, so any delivered dataset can be traced to the run that produced it.
