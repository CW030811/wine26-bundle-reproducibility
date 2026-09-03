# Reproduction scope and acceptance policy

The local project previously called these seven entries `EXACT`. This public
release uses `ACCEPTED_WITH_DOCUMENTED_SCOPE` to avoid implying bitwise,
end-to-end retraining equivalence where that was not established.

1. Runtime, runtime ratios, and machine-dependent timing columns are not acceptance targets.
2. Table 2 checks each final ten-seed/sample aggregation (BSP is seed-independent).
   Two means cross the nominal three-decimal rounding boundary by only about
   `1.41e-5` and `4.07e-6`. The documented tolerance is a half display unit plus
   at most `0.00005`; raw values are retained, not replaced by paper values.
3. Table 6 checks ten archived training-loss records and a fresh ten-seed FCP replay
   (40 reported non-runtime values). It is **not** evidence of newly retraining all ten models.
4. Table 7's Beta(.5,.5) population std is `0.007491434479918724`, while sample std
   is `0.007529174942855045`. The paper's `0.008` is accepted using the documented
   sample-std convention. Other mean/std values match their displayed precision.
5. Figure 9 supplies a seed-1 cutoff-sensitivity reference/replay package with
   1,080 experiment keys, nine cutoffs and a 60-second per-solve limit.
   Package acceptance follows the implemented verification policy; the strict
   row-level comparator is a separate check, not a claim of bitwise equivalence.
6. Figure 10 has 630 identical non-runtime replay rows/fields. Its archived audit
   replay initially used 600 seconds, but all total row runtimes were below 60
   seconds; the packaged final driver defaults to 60 seconds.
7. Figure 11's 60 per-instance translation vectors and paths reproduce `489/522`.
   Its exact solver sources and checkpoint are SHA-256 guarded.
8. Table 5 contains 120 FCP/BSP sweep rows: two scales, three cost regimes, five
   seeds, two methods and fixed/buggy Z variants. The fixed-method InS/OOS
   averages reproduce 24 paper statistics. The independently checked sweep
   scope is FCP/BSP InS/OOS.

`scripts/verify_reference_results.py` audits the released files. Passing this
command does not mean it just retrained a model or solved a MILP. Actual replay
commands are listed in each experiment guide. Training can vary across hardware
and library builds; archived checkpoints are the reference for numerical replay.

Pending experiments have no accepted result bundles. Plotting hard-coded paper
arrays is not accepted as an independent experiment reproduction.
