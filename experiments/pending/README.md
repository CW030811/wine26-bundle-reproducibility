# Pending experiments — source entry points only

These are **not accepted reproduction results** in this release. Final inputs
and models have been located locally, but the remaining full-run checks must
finish before their experiment bundles are published.

| Experiment | Purpose | Source / driver |
|---|---|---|
| Table 3 | Base FCP/PCP/FCPLS on larger instances | `scripts/run_remaining_exact.py --batch tables` |
| Table 4 | Self-improved FCP-I/PCP-I/FCPLS-I | Same tables batch, self-improved models |
| Figure 6 | Customer-count scalability (m) | `scripts/run_remaining_exact.py --batch figures` |
| Figure 7 | Product-count scalability (n) | Same figures batch, base models |
| Figure 8 | Self-improvement comparison | Same figures batch, self-improved models |

Inspect the task plan without running missing inputs:

```bash
uv run python scripts/run_remaining_exact.py --batch figures --dry-run
uv run python scripts/run_remaining_exact.py --batch tables --dry-run
```

Validation entry point, **only after future input/model/result bundles are added**:

```bash
uv run python scripts/verify_remaining_results.py --experiment all
```

Individual evaluators and final plot scripts are in `src/deterministic/`.
The plot scripts contain paper target arrays. Replotting them is not evidence
that new results reproduce the paper.

Figure 6's m=80 case has 30 available inputs but the paper reports 29 samples.
Any inferred excluded sample must be explicitly documented; matching a subset
to a published statistic alone does not establish original sample provenance.

Future accepted updates must add final data, checkpoints, full per-sample output,
verification evidence, a versioned bundle and a status update together.
