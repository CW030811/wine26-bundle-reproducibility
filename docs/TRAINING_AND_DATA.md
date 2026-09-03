# Data generation, labeling and training

All distributed instances are synthetic experiment data. Use the archived inputs
for paper comparisons; running a generator is not a guarantee of recovering the
same random draw. Model checkpoints and loss histories accompany the relevant bundles.

## Deterministic experiments

The Table 6 bundle supplies the final 3,000 labeled training instances and ten
base checkpoints. Training source is `src/deterministic/Training_multi_layer.py`.
It trains seeds 1–10 with four layers; recorded settings include batch size 256,
learning rate 0.001, dropout 0.2 and weight decay 0.0001.

```bash
python3 scripts/extract_bundle.py table6
uv run python src/deterministic/Training_multi_layer.py \
  --data_dir . \
  --train_subdir data/deterministic/train_m10n10_correct_1e_3 \
  --model_subdir results/new_base_training/models \
  --charts_subdir results/new_base_training/charts \
  --log_subdir results/new_base_training/logs \
  --no_cleanup
```

This is a training entry point, **not a newly validated exact-retraining result**.
The source's historical default points to a missing placeholder, so the explicit
`--train_subdir` above is required. Never train into the archived model directory.

Relevant generation/labeling sources:

- `src/deterministic/generate_data_bundle.py`: synthetic bundle instances / labels.
- `src/deterministic/generate_data_BSP.py`: BSP-labeled instances.
- `src/deterministic/generate_data_PCP_cp.py`: PCP labels for self-improvement.
- `src/deterministic/generate_data_bundle_log.py`, `generate_data_bundle_f0.33.py`,
  `generate_data_bundle_beta.py`: OOD generation variants.

Some generation scripts expose configuration in their main block instead of a
uniform CLI. Inspect these settings before generating a new dataset. Existing
archived inputs are the authoritative reference, not a claim that every default
generator setting recreates every archived draw.

## Random valuation (Table 5)

The Table 5 bundle supplies 4,000 generated N=5 instances, corrected MB labels,
portable train/eval/test manifests, the seed-1000 checkpoint and training records.
Use the corrected Z lower bound (`-GRB.INFINITY`) for labeling and evaluation.

```bash
python3 scripts/extract_bundle.py table5
uv run python src/random_valuation/Training_multi_layer_cpbsd_mb_x.py \
  --device cpu --model_dir results/new_table5_training/models --no_cleanup
```

Generation: `generate_data_CPBSD.py`; labeling:
`label_cpbsd_mb_from_manifest_zfix.py`, `relabel_all_zfix.py`; optimization:
`solve_mb_bsp_on_cpbsd_v2_zfix.py` and `table5_solvers_zfix.py`.
The final evaluator regenerates test draws using seeds 20260413–20260417,
N in {10,30}, K=50 and 5,000 out-of-sample draws. Its reference checkpoint is
`artifacts/random_valuation/models/best_model_edge_cpbsd_mb_x_2layer_seed1000.pt`.

## Pending self-improvement experiments

Final self-improved materials exist locally but are deliberately not in this
accepted release until Tables 3–4 / Figures 6–8 have passed full replay checks.
See `experiments/pending/README.md`. Base models for accepted experiments are
already included; absence of a pending bundle is a publication gate, not proof
that its underlying data is lost.
