# Table 5 — additive random valuation

Archive includes 4,000 generated instances, corrected labels and portable manifests,
seed-1000 model/training artifacts, 120 sweep rows, and final TeX rows.
FCP/BSP fixed-solver InS/OOS averages reproduce the paper's 24 values.
**CPBSD-A and runtime cells are reused paper values, not independently replayed
in this release.** See [acceptance scope](../../docs/ACCEPTANCE.md).

```bash
python3 scripts/extract_bundle.py table5
# Reaggregate the published reference CSV into TeX (no optimizer):
uv run python src/random_valuation/make_table5_rows.py
# New full FCP/BSP sweep, without overwriting the reference CSV:
uv run python src/random_valuation/run_experiment_zfix.py \
  --model-path artifacts/random_valuation/models/best_model_edge_cpbsd_mb_x_2layer_seed1000.pt \
  --out-csv results/table5/experiment_zfix.csv
```

`make_table5_rows.py` reads the **archived** CSV in
`artifacts/random_valuation/results/`, not the new CSV in `results/table5/`.
Compare a new sweep's method/variant/scale/cost/seed keys and InS/OOS averages
against that archive; do not mistake reprinting archived TeX for a new replay.
Generation, labeling and training are described [here](../../docs/TRAINING_AND_DATA.md).
