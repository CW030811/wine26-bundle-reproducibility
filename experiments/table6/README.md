# Table 6 — ten-seed robustness

Archive includes 3,000 final labeled training instances, 100 test instances,
ten models with loss histories, and 1,000 seed×sample FCP replay rows.
Forty reported loss/profit values match; training-loss checks use the archived
records, not a fresh training run. [Detailed evidence](../../provenance/TABLE6_EXACT_REPRO.md).

```bash
python3 scripts/extract_bundle.py table6
uv run python scripts/verify_main_results.py --experiment table6
uv run python src/deterministic/test_FCP_multi_model_avg.py \
  --data_dir . --test_subdirs data/deterministic/test_m10n10_correct_1e_3 \
  --model_dir models/main_base_4layer_correct_lr_3 --layers 4 \
  --seeds 1,2,3,4,5,6,7,8,9,10 --result_dir results/main_exact_rerun/table6
uv run python scripts/verify_main_results.py --experiment table6
```

See [training instructions](../../docs/TRAINING_AND_DATA.md) for an independent
retraining entry point. Archived checkpoints are the paper replay reference.
