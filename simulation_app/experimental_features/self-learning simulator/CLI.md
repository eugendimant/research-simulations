# socsim CLI cheatsheet

```bash
python -m socsim version
python -m socsim doctor
python -m socsim list_games

python -m socsim spec_template --game ultimatum > my_spec.json
python -m socsim spec_lint --spec my_spec.json

python -m socsim bench_cv --out bench_cv
python -m socsim bench_scoreboard --run-dir bench_runs
python -m socsim bench_pack --bench-dir benchmarks/<id> --out benchmarks/<id>/bench_pack.json

python -m socsim insight_add --item path/to/insight.json
python -m socsim insight_search --game ultimatum
```

python -m socsim bench_run_pack --bench-dir benchmarks/<id> --out-dir bench_targets/<id>
