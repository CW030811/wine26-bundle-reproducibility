# Table 2 — main comparison

Accepted: three sizes × FCP/PCP/FCPLS/BSP. Each neural method has ten seeds ×
100 samples per size; BSP has 100 samples. Small rounding-boundary exceptions
are recorded in [provenance](../../provenance/TABLE2_EXACT_REPRO.md).

```bash
python3 scripts/extract_bundle.py table2
uv run python scripts/verify_main_results.py --experiment table2
# Full replay (requires full Gurobi license; can take hours):
uv run python scripts/run_table2_exact.py
```

Source: `scripts/run_table2_exact.py` and the FCP/PCP/FCPLS/BSP evaluators under
`src/deterministic/`. Inputs: three `data/deterministic/test_m*n10_correct_1e_3`
directories. Models: `models/main_base_4layer_correct_lr_3`. Reference CSVs:
`results/main_exact_rerun/table2/`. Use the Table 6 bundle for the shared training data.
