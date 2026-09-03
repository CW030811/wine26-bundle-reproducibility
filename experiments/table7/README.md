# Table 7 — OOD robustness

Four final OOD datasets, 100 instances each, ten base models, and all 4,000
seed×sample replay rows. The small Beta(.5,.5) standard-deviation convention
difference is [documented](../../provenance/TABLE7_EXACT_REPRO.md).

```bash
python3 scripts/extract_bundle.py table7
uv run python scripts/verify_main_results.py --experiment table7
uv run python src/deterministic/test_FCP_multi_model_avg.py \
  --data_dir . \
  --test_subdirs 'data/ood/test_m10n10_beta_5_5_correct_1e_3;data/ood/test_m10n10_beta_half_half_correct_1e_3' \
  --model_dir models/main_base_4layer_correct_lr_3 --result_dir results/main_exact_rerun/table7/beta
uv run python src/deterministic/test_FCP_multi_model_avg_log.py \
  --data_dir . --test_subdirs data/ood/test_m10n10_log_correct_1e_3 \
  --model_dir models/main_base_4layer_correct_lr_3 --result_dir results/main_exact_rerun/table7/log
uv run python src/deterministic/test_FCP_multi_model_avg_f33.py \
  --data_dir . --test_subdirs data/ood/test_m10n10_f0.33_correct_1e_3 \
  --model_dir models/main_base_4layer_correct_lr_3 --result_dir results/main_exact_rerun/table7/cube_root
uv run python scripts/verify_main_results.py --experiment table7
```

The default evaluator seeds are 1–10. The Table 6 archive supplies shared training data.
