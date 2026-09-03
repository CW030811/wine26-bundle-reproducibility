# Contributing

Open an issue or pull request with the experiment number, exact command,
Python/solver versions, solver status, input/model hashes, and observed vs expected
non-runtime values. Do not include license files, tokens, personal paths or raw
machine logs containing credentials.

To promote a pending experiment:

1. Finish the complete planned sample/seed grid using final inputs and checkpoints.
2. Check completeness, failures, solver limits, means/std and per-row evidence.
3. Document every accepted exception; do not select a favorable subset just to match a table.
4. Add the final archive, per-member SHA-256 inventory, experiment guide and verifier.
5. Update `EXPERIMENTS.json`, the README status table and release checksums.
6. Run archive tests, existing unit tests, and the numerical reference audit in a fresh checkout.

Keep reference archives unchanged when rerunning experiments. Submit any new
reference as a versioned change with its evidence and scope. Unverified results
must not be described as exact. The owner can grant GitHub collaborator access
through repository settings; public visibility does not grant write access.
