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
uv run python scripts/reproduce.py prepare table6
uv run python scripts/data_pipeline.py train base --output-dir results/training/base
uv run python scripts/reproduce.py prepare table4
uv run python scripts/data_pipeline.py train self_improved --output-dir results/training/self_improved
```

This is a training entry point, **not a newly validated exact-retraining result**.
All new generation/training outputs must be below the repository `results/`
directory. Generation refuses an existing destination, and training refuses a
nonempty destination. Use `--dry-run` to inspect commands.

Relevant generation/labeling sources:

- `src/deterministic/generate_data_bundle.py`: synthetic bundle instances / labels.
- `src/deterministic/generate_data_BSP.py`: BSP-labeled instances.
- `src/deterministic/generate_data_PCP_cp.py`: PCP labels for self-improvement.
- `src/deterministic/generate_data_bundle_log.py`, `generate_data_bundle_f0.33.py`,
  `generate_data_bundle_beta.py`: OOD generation variants.

The shared generation interface calls these functions with explicit settings:

```bash
# Base mixed-bundling data and optimal labels, default 3,000 instances:
uv run python scripts/data_pipeline.py generate base --output-dir results/generated/base
# BSP test data, example m=10,n=20,count=100:
uv run python scripts/data_pipeline.py generate bsp --products 20 --segments 10 --count 100 --output-dir results/generated/bsp_m10n20
# PCP labels: n=50,m=10,seed=9,threshold=0.5,count=3000,base seed-9 model:
uv run python scripts/data_pipeline.py generate self_improved --output-dir results/generated/self_improved
```

`--products`, `--segments`, `--count`, `--seed` and `--model` configure new
generation. Deterministic generation includes the MILP labeling step. OOD recipes
are `log`, `cube_root`, `beta_5_5`, `beta_half_half`, with n=m=10 and count=100.
The original random state of every base/BSP/OOD draw was not archived; use
released inputs for paper comparisons. A fresh draw is not an identical dataset.

## Random valuation (Table 5)

The Table 5 bundle supplies 4,000 generated N=5 instances, corrected MB labels,
portable train/eval/test manifests, the seed-1000 checkpoint and training records.
Use the corrected Z lower bound (`-GRB.INFINITY`) for labeling and evaluation.

```bash
uv run python scripts/reproduce.py prepare table5
uv run python scripts/data_pipeline.py train random --output-dir results/training/random
# Or generate, label and train a new dataset through the common manifests:
uv run python scripts/data_pipeline.py generate random --output-dir results/generated/random
uv run python scripts/data_pipeline.py label random --input-dir results/generated/random --output-dir results/labels/random
uv run python scripts/data_pipeline.py train random --input-dir results/labels/random --output-dir results/training/new_random
```

Generation: `generate_data_CPBSD.py`; labeling:
`label_cpbsd_mb_from_manifest_zfix.py`, `relabel_all_zfix.py`; optimization:
`solve_mb_bsp_on_cpbsd_v2_zfix.py` and `table5_solvers_zfix.py`.
The final evaluator regenerates test draws using seeds 20260413–20260417,
N in {10,30}, K=50 and 5,000 out-of-sample draws. Its reference checkpoint is
`artifacts/random_valuation/models/best_model_edge_cpbsd_mb_x_2layer_seed1000.pt`.

Random generation defaults to N=5,K=50,normal,rho=0,full heterogeneity,
`random_ind` costs and seeds 1000–4999. It writes train/eval/test manifests
with 3000/600/400 instances. Labeling writes corrected labels and portable
`manifest_train.csv`, `manifest_eval.csv`, `manifest_test.csv`, which the
training command consumes directly through `--input-dir`.

Random training uses two layers, seed 1000, batch size 32 and learning rate
0.001. The deterministic trainers use four layers and seeds 1–10. `--epochs`
can shorten a smoke test; the paper recipes default to 200.

## Fast verification of recovered generation

```bash
uv run python scripts/reproduce.py prepare table5
uv run python scripts/reproduce.py prepare table4
uv run python scripts/verify_generation.py
```

This rebuilds all decoded fields of the 4,000 random-valuation inputs and the
four random input arrays of all 3,000 PCP training instances. It does not rerun
optimization labels or retrain models. Results are recorded in
`provenance/GENERATION_AUDIT.json`. Preparing Table 4 or Figure 8 restores all
three shared self-improvement training shards automatically.

Random-valuation generation uses normal CDF/inverse-CDF routines whose final
float64 bits can vary between platform math libraries. The audit records the
maximum absolute difference and permits only `atol=rtol=1e-12`; array shapes,
integer identities and metadata remain strict. Archive byte hashes are unchanged.
