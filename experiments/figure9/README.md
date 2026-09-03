# Figure 9 — cutoff sensitivity

Three datasets, nine cutoffs, independent FCP=30 / PCP=10 samples per dataset,
seed-1 model, 60-second per-solve limit; 1,080 rows.
Archive includes reference CSV/PDF and the independently replayed CSV.
The driver was reconstructed from final configuration and archived evidence.

```bash
python3 scripts/extract_bundle.py figure9
uv run python src/appendix/rerun_seed1.py --mode cutoff
# Strict row-level comparison, separate from package acceptance:
uv run python scripts/compare_appendix_cd_rerun.py --figure figure9
```

The reference CSV, replay CSV and comparison tools are provided together.
Package acceptance uses the implemented verification policy; strict row-level
comparison is a separate check. See [acceptance scope](../../docs/ACCEPTANCE.md).

Sources: `src/appendix/rerun_seed1.py` and `src/appendix/legacy_runtime/`.
Inputs: the three `data/deterministic/test_m*n10_1e_3/` datasets.
Model: `models/appendix_seed1/model_edge_4layer_seed1.pt`.
