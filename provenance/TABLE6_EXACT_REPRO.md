# Table 6 exact-reproduction record

Table 6 was regenerated from the recovered final `(m,n)=(10,10)` data and the ten
`correct_lr_3` checkpoints. Runtime fields are recorded in the CSV files but are
excluded from the exactness decision because they depend on the machine.

## Evidence

- Training data: `data/deterministic/train_m10n10_correct_1e_3/` (3,000 files).
- Test data: `data/deterministic/test_m10n10_correct_1e_3/` (100 files).
- Models and loss histories: `models/main_base_4layer_correct_lr_3/`.
- Per-sample ten-seed results: `results/main_exact_rerun/table6/test_result_FCP_4layer_test_m10n10_correct_1e_3_seed_sample.csv` (1,000 rows).
- Per-seed summary: `results/main_exact_rerun/table6/test_result_FCP_4layer_test_m10n10_correct_1e_3_seed_avg.csv`.
- Verification command: `scripts/verify_main_results.py --experiment table6`.

The verifier checked all ten seeds. For every seed, final training loss and best
validation loss match the paper to four decimals, while FCP profit mean and
population standard deviation match to three decimals. This is 40/40 matching
non-runtime values. No sample or seed failed.

## Commands

```bash
uv run python src/deterministic/test_FCP_multi_model_avg.py \
  --data_dir . \
  --test_subdirs data/deterministic/test_m10n10_correct_1e_3 \
  --model_dir models/main_base_4layer_correct_lr_3 \
  --layers 4 \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --result_dir results/main_exact_rerun/table6

uv run python scripts/verify_main_results.py --experiment table6
```
