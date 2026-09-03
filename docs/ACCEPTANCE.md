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
5. Figure 9's 1,080 keys and classification fields match. One PCP row at
   `test_m30n10_1e_3/sample_data_64_size_10.msgpack`, cutoff `0.1`, reaches 60 seconds.
   The legacy implementation returns a search-dependent `ObjBound`: archived
   revenue ratio `1.004751821452851`, replay `1.0060421929126016`. The affected
   aggregate mean differs by about `4.3012382e-5`; this exception was explicitly
   accepted. The strict legacy comparison remains available and reports this mismatch.
   The public evidence audit permits only this key/field with an absolute difference
   no larger than `0.00130`; it is not a blanket tolerance for other results.
6. Figure 10 has 630 identical non-runtime replay rows/fields. Its archived audit
   replay initially used 600 seconds, but all total row runtimes were below 60
   seconds; the packaged final driver defaults to 60 seconds.
7. Figure 11's 60 per-instance translation vectors and paths reproduce `489/522`.
   Its exact solver sources and checkpoint are SHA-256 guarded.
8. Table 5 contains 120 FCP/BSP sweep rows: two scales, three cost regimes, five
   seeds, two methods and fixed/buggy Z variants. The fixed-method InS/OOS
   averages reproduce 24 paper statistics. **CPBSD-A cells and runtime cells in
   the TeX generator are reused published values, not independently replayed
   evidence.** The CPBSD-A solver source is supplied for future verification.

`scripts/verify_reference_results.py` audits the released files. Passing this
command does not mean it just retrained a model or solved a MILP. Actual replay
commands are listed in each experiment guide. Training can vary across hardware
and library builds; archived checkpoints are the reference for numerical replay.

Pending experiments have no accepted result bundles. Plotting hard-coded paper
arrays is not accepted as an independent experiment reproduction.
