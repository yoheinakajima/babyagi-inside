# BabyAGI agent diagnosis before protocol v2

This diagnosis supersedes any interpretation of the typed-v17 pine trace as an
agent-capability result. That trace remains valid systems evidence for compiled
control flow and qwenFS. It does not demonstrate semantic prioritization.

All baselines here use the frozen `babyagi-heldout-v1` suite, greedy decoding,
Qwen3.5-2B native weights, and checksum
`22891c9fe8b277dfc111c3c4feb766e62fc29853ed339eeb1b769ae4391c2f49`.

## Predeclared working criterion

Before protocol v2 is evaluated, “working” means:

1. at least 50% of the plain-chat automated ceiling on full held-out items;
2. manual language criteria reported separately, never converted to passes;
3. the creator-designated next task appears in slot 2 in at least one third of
   loop cycles;
4. semantic choice accuracy is above chance, with Task-2 recall reported; and
5. greedy decoding, no host semantic fallback, and a fixed token budget.

The measured thinking-off automated ceiling is 4/12, so the current minimum is
2/12 automated full-item passes. Manual review makes the strict ceiling 3/12:
the Aster output meets automated constraints but omits the rover relationship.

## Ceiling and host-loop result

| Mode | Automated full items | Parsed full runs | Main finding |
| --- | ---: | ---: | --- |
| Plain chat, thinking off | 4/12 | 9/12 | All three arithmetic items pass; Aster is automated-only and fails one manual criterion. |
| Plain chat, thinking on | 0/12 | 1/12 | All 12 hit the 512-token cap; only one exposed a parseable result before truncation. |
| Python BabyAGI loop, thinking off | 1/12 | 12/12 | Structural protocol works, but only Aster meets automated constraints and it fails manual review. |
| Python BabyAGI loop, thinking on | 0/12 | 0/12 | Every item stops in cycle 1; every attempted call hits the 512-token cap. |

A 2,048-token thinking-on pilot on the first item ran for more than eight
minutes without returning a final envelope and was stopped. The 512-token suite
therefore measures bounded usability, not an unlimited-chain ceiling.

On the automated metric, the thinking-off host loop retains 1/4 = 25% of the
plain-chat ceiling, below the required 50%. After manual review, it retains
0/3 = 0% of the strict ceiling.

## Per-item and per-phase table

`choose` is correctness against the creator call's hidden `next_task` label
after deterministic host counterbalancing. `execute` is the number of exact
result prefixes through that cycle. Thinking-on stop is the first failed phase.

| Item | Plain off | Plain on parsed | Loop off | Choose | Execute | Loop-on stop |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| literal-cinder-5 | fail | no | fail | 3/5 | 1/5 | execute |
| literal-moss-3 | fail | no | fail | 2/3 | 0/3 | execute |
| literal-small-moon-4 | fail | no | fail | 1/4 | 0/4 | create |
| arithmetic-seven-3 | pass | no | fail | 1/3 | 0/3 | create |
| arithmetic-ten-4 | pass | no | fail | 3/4 | 0/4 | execute |
| arithmetic-three-5 | pass | no | fail | 2/5 | 0/5 | create |
| structured-numbers-3 | fail | no | fail | 2/3 | 1/3 | execute |
| structured-words-3 | fail | no | fail | 1/3 | 0/3 | execute |
| structured-records-3 | fail | yes, truncated | fail | 1/3 | 0/3 | create |
| language-day-acrostic-3 | fail | no | fail | 1/3 | 0/3 | create |
| language-aster-summary-3 | automated only | no | automated only | 2/3 | 3/3 | execute |
| language-tidetally-announcement-5 | fail | no | fail | 3/5 | 0/5 | create |

Thinking-off host-loop totals:

- create parsing: 44/44;
- distinct candidate pairs: 42/44;
- choice parsing: 44/44;
- semantic choices: 22/44, exactly chance;
- Task 2 expected: 23/44; Task 2 selected: 9/44;
- execution parsing: 44/44;
- exact result prefixes: 5/44; and
- wrong result prefixes: 39/44.

This separates three failures that the original 0/12 number conflated:

1. plain Qwen can solve some tasks;
2. the host loop loses most of that capability during task decomposition and
   stateful execution; and
3. the side-by-side selector has a severe position bias.

## Fifty-probe selector diagnosis

Twenty-five work-versus-distractor scenarios were evaluated in both orders,
plus ten neutral calibration prompts.

| Metric | Result |
| --- | ---: |
| Neutral mean logit(1) − logit(2) | +6.3375 |
| Raw accuracy | 25/50 = 50% |
| Work-first accuracy | 25/25 = 100% |
| Work-second accuracy | 0/25 = 0% |
| Raw Task-2 selection | 0/50 = 0% |
| Accuracy after subtracting neutral constant | 25/50 = 50% |
| Calibrated Task-2 selection | 50/50 = 100% |

Scenario deltas overlap completely: all raw deltas favor token 1, and the
neutral offset is larger than every scenario delta, so a constant correction
merely changes “always 1” into “always 2.” Calibration cannot recover semantic
signal that is not present in the side-by-side decision.

Protocol v2 should therefore score each task independently (“is this the next
undone task?”) and compare the two judgments. A binary yes/no version is the
first native prototype; a small ordinal score with an in-weight comparator is
the stronger compiled design.

## Bounded-think feasibility

The current five-cycle checkpoint uses 24 of 32 reserved microstates and
scratch dimensions `[3072, 3379)`, leaving 717 dimensions before 4096.

A fixed model-owned span can live inside a stage: the ROM opens `<think>`, Qwen
owns exactly N ordinary tokens, and the ROM forces `</think>` plus the tool-call
prefix. This needs roughly N additional offset dimensions and one token-mask
dimension, but no additional microstate.

| Fixed think budget | Estimated scratch end | Dimensions remaining |
| ---: | ---: | ---: |
| 64 | 3444 | 652 |
| 128 | 3508 | 588 |
| 256 | 3636 | 460 |
| 512 | 3892 | 204 |

These are layout estimates; MLP-unit capacity and checkpoint logits still need
an actual compile-and-oracle test. An execute-only design using a separate
microstate would add five states (24 → 29) and also fit. Splitting both create
calls and execute across extra states would require 39 states and exceed the
current 32-state reservation.

Native thinking-on demonstrates why the controller must force the transition:
Qwen does not reliably close its reasoning and return structured output within
512 tokens. The first v2 experiment should therefore use a fixed 128-token
execute span, not unbounded native thinking. Create can remain thinking-off
until execute is measured.

## Decision

Do not move to 4B. Qwen3.5-2B has a measurable plain-chat ceiling, and the
subsequent ledger/JEV/bounded-execute protocol reaches the predeclared 50%
retention threshold. See `docs/babyagi-ledger-evidence.md` for the selected
protocol, rejected ablations, and disjoint confirmation result. The next
ordered work is:

1. compile the fixed 128-token execute span;
2. implement an in-weight representation and comparison of two independent
   Y/N margins;
3. run circuit, parity, accounting, and live merged-checkpoint tests; and
4. evaluate the compiled protocol on the already-frozen v2 confirmation suite
   without changing prompts or thresholds.
