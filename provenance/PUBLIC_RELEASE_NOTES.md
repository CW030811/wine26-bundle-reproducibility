# Public export notes

- This is a clean export of the seven locally accepted experiment entries, not
  a copy of the entire work directory or its Git history.
- Numerical data and model tensors are unchanged. Machine-specific path metadata
  in text artifacts is normalized; transformed archive members are listed in
  `bundles/manifest.json`. Redundant nonportable Table 5 manifests and machine logs
  are omitted. Portable manifests resolve to the bundled instances and labels.
- Source algorithms are preserved. Three Python files receive path-only privacy
  changes: the two Figure 11 CLI defaults now use the current directory, and the
  runner's plotting cache uses a repository-relative location. Guard hashes are
  updated consistently. `SOURCE_EXPORT.json` records pre/post-export hashes;
  `PRIVACY_AUDIT.json` records AST-level checks of the permitted changes.
- Public-only additions are the extractor, its tests, the combined reference
  evidence auditor, experiment documentation and manifests.
- The environment adds TensorBoard, required by the deterministic training
  module but missing from the previous dependency list. Other direct experiment
  dependency pins are retained; `uv.lock` fixes the complete resolution.
- The old local README/inventory predates recovery of the final data. It is not
  published as current guidance. This release's README and acceptance document
  supersede those stale statements.
- Published target values include the Figure 7/8 standard-deviation arrays added
  during pending-run preparation. Those additions changed the local index's
  shared-file checksum; this release records a new export hash and separately
  rechecks the seven accepted experiments' reference statistics.
- Reference and replay CSVs for Figures 9–10 are both bundled. Figure 11 includes
  its archived independent replay JSONs. No new full solver replay was required
  merely to export these previously accepted results.
- No paper PDF, account configuration, solver license, authentication material,
  virtual environment or unverified remaining-run results is distributed.
