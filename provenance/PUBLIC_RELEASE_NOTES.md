# Public export notes

- This is a clean export of all twelve accepted numerical experiment entries, not
  a copy of the entire work directory or its Git history.
- Numerical data and model tensors are unchanged. Machine-specific path metadata
  in text artifacts is normalized; transformed archive members are listed in
  `bundles/manifest.json`. Redundant nonportable Table 5 manifests and machine logs
  are omitted. Portable manifests resolve to the bundled instances and labels.
- Source algorithms are preserved. Figure 11 privacy changes and guard hashes
  remain synchronized. Deterministic evaluator/training defaults now point to
  the published datasets/checkpoints; the Appendix C/D driver accepts an output
  root. `SOURCE_EXPORT.json` records source/export hashes. Shared training-summary
  checkpoint paths are normalized to the published model directories.
- Public orchestration now covers prepare, replay, verify, plot, generate, label
  and train. Per-experiment outputs are isolated, random-data manifests connect
  all three data/training stages, and plotting uses a headless backend.
- Tables 3–4 and Figures 6–8 include final inputs, best checkpoints and all
  per-sample results. Three shared shards contain all 3,000 self-improvement
  training instances. Figure 8 verification includes all five curves.
- Figure 6 uses its registered fixed 29-sample m=80 reference. The full input
  and replay inventory is retained alongside the frozen selection metadata.
- The environment adds TensorBoard, required by the deterministic training
  module but missing from the previous dependency list. Other direct experiment
  dependency pins are retained; `uv.lock` fixes the complete resolution.
- The old local README/inventory predates recovery of the final data. It is not
  published as current guidance. This release's README and acceptance document
  supersede those stale statements.
- Published target values retain the Figure 7/8 standard-deviation arrays.
  The unified verifier checks every experiment; Table 6/7 checks now require
  complete input/seed identities in addition to matching summary statistics.
- Generation verification reproduces 4,000 random-valuation inputs and all
  3,000 self-improvement input arrays. A small new-data → labels → training run
  verifies the public interfaces without claiming a full exact retraining.
- Reference and replay CSVs for Figures 9–10 are both bundled. Figure 11 includes
  its archived independent replay JSONs. No new full solver replay was required
  merely to export these previously accepted results.
- No paper PDF, account configuration, solver license, authentication material,
  virtual environment or machine logs are distributed.
