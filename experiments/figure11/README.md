# Figure 11 — LP-to-MILP move translation

Sixty exact selected inputs (ten from each of six datasets), seed-1 model,
50 maximum iterations, tolerance `1e-6`. Reference and replay translation vectors
and paths match; 489 of 522 accepted LP moves translate successfully.

```bash
python3 scripts/extract_bundle.py figure11
uv run python src/appendix_rerun/run_guarded_sensitivity.py \
  --experiment lp_milp --output-dir results/appendix_e_replay \
  --layer 4 --seed 1 --lp-milp-samples 10 \
  --max-iterations 50 --tolerance 1e-6
uv run python src/report/replot_verification_paths.py
```

Inputs: `Dataset/`; model: `models_multi_layer_edge_update/best_model_edge_4layer_seed1.pt`.
Sources: guarded `src/test/test_FCPLS_score_cached_lp*.py` and
`src/appendix_rerun/run_guarded_sensitivity.py`. The two identical compatibility
source copies are deliberately retained because the runner validates both hashes.
The public sources replace the historical personal CLI fallback with the current
directory. Guard hashes were updated for this path-only change; use the top-level
runner above, which resolves all actual inputs relative to this repository.

Reference artifacts: `artifacts/appendix_e_final/`; archived fresh replay JSONs:
`results/appendix_e_replay/lp_milp/`. Replotting alone uses archived data and is
not an independent optimizer replay.
