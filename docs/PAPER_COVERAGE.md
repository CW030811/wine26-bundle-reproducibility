# Paper coverage

| Content | Coverage |
|---|---|
| Tables 2–7 | Six guides, source mappings, data/model/results archives and numerical checks |
| Figures 6–11 | Six guides, inputs, replay outputs, checks and data-driven plots |
| Table 1 | Analytical complexity comparison; no numerical experiment |
| Figure 1 | Motivating commercial example; no experimental dataset |
| Figures 2–5 | Graph representation, GNN, pruning and self-improvement diagrams; implemented by the algorithm sources |
| Appendices C–E | Figures 9–11 sensitivity and LP/MILP translation |
| Appendices I–J | Tables 6–7 seed and OOD robustness |
| Appendices A–B, F–H, K | Formulations, algorithms, training descriptions and proofs; numerical components covered by the listed sources |

The package retains the final deterministic and random-valuation source
closures, reconstructed appendix drivers, and guarded LP/MILP solver sources.
Historical versions, transient machine logs and duplicate intermediate
checkpoints are excluded. `EXPERIMENTS.json` maps numerical artifacts to
sources and archives; `bundles/manifest.json` lists every archived file.

Figures are rendered from released/replayed numerical data. Registered values,
sample identities and computational results are the reproduction targets;
pixel-identical rendering and identical runtimes across machines are not required.
