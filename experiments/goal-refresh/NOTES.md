# Goal-refresh experiment

This experimental variant added an input encoding that requested an initial
model-side write of goal.txt, then compiled a goal read before every creation
phase. It used 21 microstates. All 1,707 circuit checks and the four-position
native-logit smoke comparison passed.

The live run was deliberately stopped after the first creation/prioritization
failure: it copied the objective and phase instructions into task entries,
then copied multiple entries instead of selecting one. This is a partial,
failed behavioral experiment, not a successful three-cycle trace. The raw log
and compiled ROM are retained. The source at this Git commit implements the
experimental variant; the following commit restores the smaller 17-state core.
