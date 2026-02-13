#!/usr/bin/env bash
set -euo pipefail

python -m socsim autocorpus --n 100 --store socsim/data/evidence_store.json

python -m socsim simulate --spec examples/spec_trust.json --n 500 --seed 7 --out outputs/trust --traces
python -m socsim simulate --spec examples/spec_public_goods.json --n 500 --seed 7 --out outputs/pgg --traces
python -m socsim simulate --spec examples/spec_ultimatum.json --n 500 --seed 7 --out outputs/ug --traces
python -m socsim simulate --spec examples/spec_die_roll.json --n 500 --seed 7 --out outputs/die --traces

python -m socsim simulate --spec examples/spec_holt_laury.json --n 300 --seed 7 --out outputs/risk --traces
python -m socsim simulate --spec examples/spec_mpl_time.json --n 300 --seed 7 --out outputs/time --traces
python -m socsim simulate --spec examples/spec_repeated_pd.json --n 200 --seed 7 --out outputs/rpd --traces
