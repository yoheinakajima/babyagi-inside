# Goal-refresh experiment

This experimental variant added an input encoding that requested an initial
model-side write of goal.txt, then compiled a goal read before every creation
phase. It used 21 microstates. All 1,707 circuit checks and the four-position
native-logit smoke comparison passed.

The live run was deliberately stopped after the first creation/prioritization
failure: it copied the objective and phase instructions into task entries,
then copied multiple entries instead of selecting one. This is a partial,
failed behavioral experiment, not a successful three-cycle trace. The raw log
and compiled ROM are retained. The experimental source is recorded at local commit `1508fc4`. The primary
implementation retains the smaller 17-state controller. Apply
`source-variant.patch` to the primary source to rebuild this variant.
