# Table 2 Exact Reproduction

## Classification

`EXACT` under the agreed paper-statistics policy:

- all three final `correct` test sets contain 100 instances;
- FCP, PCP and FCPLS each completed 10 seeds × 100 samples for every size;
- BSP completed 100 samples for every size;
- runtime is machine-dependent and excluded;
- profit mean/std are compared to the paper's three-decimal display precision;
- numerical noise within 0.00005 beyond the nominal half-display-unit rounding boundary is accepted and explicitly recorded.

The machine-readable verification report is
`results/main_exact_rerun/table2/verification_report.json`. It reports
`passed=true`, 12/12 checked cells and no mismatches.

## Final inputs and model

- `data/deterministic/test_m10n10_correct_1e_3/` (100)
- `data/deterministic/test_m20n10_correct_1e_3/` (100)
- `data/deterministic/test_m30n10_correct_1e_3/` (100)
- `models/main_base_4layer_correct_lr_3/` (10 final checkpoints plus loss records and summary)

## Final code

- top-level sequential driver: `scripts/run_table2_exact.py`
- FCP: `src/deterministic/test_FCP_multi_model_avg.py`, `test_FCP.py`
- PCP: `src/deterministic/test_PCP_cp_multi_model_avg.py`, `test_PCP_cp.py`
- FCPLS: `src/deterministic/test_FCPLS_score_cached_lp.py`
- BSP: `src/deterministic/test_BSP.py`
- verifier: `scripts/verify_main_results.py --experiment table2`

## Result completeness

`results/main_exact_rerun/table2/` contains 35 files:

- FCP: three 100-row sample-average CSVs, three 10-row seed summaries and three 1,000-row seed×sample long tables;
- PCP: the same complete 3×3 structure;
- FCPLS: the same complete 3×3 structure;
- BSP: three 100-row sample tables;
- four execution logs and one verification report.

No evaluator log contains a failed sample or size-limited-license error. BSP explicitly reports zero failed samples in all three datasets.

## Non-runtime verification

| Size | Method | Fresh mean | Fresh population std | Fresh sample std | Paper mean/std | Rule |
|---|---|---:|---:|---:|---:|---|
| m=10 | FCP | 0.988794658 | 0.007349012 | 0.007386035 | 0.989 / 0.007 | displayed precision |
| m=10 | PCP | 0.994628507 | 0.004667409 | 0.004690923 | 0.995 / 0.005 | displayed precision |
| m=10 | FCPLS | 0.995527172 | 0.004348102 | 0.004370007 | 0.996 / 0.004 | displayed precision |
| m=10 | BSP | 0.878903451 | 0.028163229 | 0.028305110 | 0.879 / 0.028 | displayed precision |
| m=20 | FCP | 0.984648818 | 0.006474709 | 0.006507328 | 0.985 / 0.007 | sample-std display |
| m=20 | PCP | 0.994485872 | 0.003748883 | 0.003767769 | 0.995 / 0.004 | rounding-boundary tolerance |
| m=20 | FCPLS | 0.991188472 | 0.005485298 | 0.005512932 | 0.991 / 0.006 | sample-std display |
| m=20 | BSP | 0.863495926 | 0.025028967 | 0.025155058 | 0.864 / 0.025 | rounding-boundary tolerance |
| m=30 | FCP | 0.981626515 | 0.006350713 | 0.006382707 | 0.982 / 0.006 | displayed precision |
| m=30 | PCP | 0.993703261 | 0.003903781 | 0.003923448 | 0.994 / 0.004 | displayed precision |
| m=30 | FCPLS | 0.987864505 | 0.005765097 | 0.005794141 | 0.988 / 0.006 | displayed precision |
| m=30 | BSP | 0.854055425 | 0.019689583 | 0.019788776 | 0.854 / 0.020 | displayed precision |

The two tolerance cases lie only about `1.41e-5` and `4.07e-6` beyond the nominal three-decimal rounding boundary. Raw values remain archived; they were not replaced by paper values.

## Commands

```bash
uv run python scripts/run_table2_exact.py
uv run python scripts/verify_main_results.py --experiment table2
```
