# Figure 10 — candidate-budget K sensitivity

Seven K rules, three datasets, 30 samples per dataset: 630 rows.
Archive includes reference CSV/PDF and independent replay CSV; every non-runtime
result field matches. The paper reference uses the seed-1 checkpoint.

```bash
python3 scripts/extract_bundle.py figure10
uv run python scripts/compare_appendix_cd_rerun.py --figure figure10
uv run python src/appendix/rerun_seed1.py --mode k
uv run python scripts/compare_appendix_cd_rerun.py --figure figure10
```

Source/input/model mapping is the same as [Figure 9](../figure9/README.md).
The final driver defaults to 60 seconds. The historical validation's larger limit
never triggered, as documented in [acceptance scope](../../docs/ACCEPTANCE.md).
