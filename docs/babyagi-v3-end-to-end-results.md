# BabyAGI v3 paired end-to-end results

## Result

The frozen 256-token JSON-ledger loop improved exact whole-chain success over
one-turn thinking-enabled Qwen3.5-2B on the same 32 deterministic objectives:

| arm | exact chains | chain accuracy | exact transitions | transition accuracy |
| --- | ---: | ---: | ---: | ---: |
| plain chat | 5/32 | 15.6% | 23/144 | 16.0% |
| host JSON-ledger loop | 9/32 | 28.1% | 52/144 | 36.1% |
| paired change | +4 | +12.5 points | +29 | +20.1 points |

The protocol retained all five plain-chat successes and added four successes:

- `v3-arithmetic-multiply-2`
- `v3-arithmetic-multiply-8`
- `v3-structured-even-8`
- `v3-structured-odd-8`

There were zero paired regressions. The exact McNemar test over the four
discordant items is `p = 0.125`, so this small suite does not establish a
conventional significance claim. The preregistered item bootstrap interval for
the paired delta is `[+3.1, +25.0]` percentage points. The loop achieved 180%
of the plain-chat exact-chain ceiling (`28.1 / 15.6`).

These are greedy bf16/MPS results on the frozen suite and suite hash in
`babyagi-v3-end-to-end-preregistration.md`. No prompt or scorer was changed
after observing the results.

## Breakdown

### By category

| category | plain chains | host chains | plain transitions | host transitions |
| --- | ---: | ---: | ---: | ---: |
| ordered literal | 0/8 | 0/8 | 0/36 | 1/36 |
| sequential arithmetic | 5/8 | 7/8 | 23/36 | 34/36 |
| structured operation | 0/8 | 2/8 | 0/36 | 16/36 |
| checkable language | 0/8 | 0/8 | 0/36 | 1/36 |

The gain is therefore real but narrow: the loop materially helps arithmetic
and recovers two eight-step structured chains, while literal and language
chains remain unsolved.

### By source-chain length

| cycles | plain chains | host chains | plain transitions | host transitions |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1/8 | 2/8 | 2/16 | 5/16 |
| 3 | 1/8 | 1/8 | 3/24 | 4/24 |
| 5 | 2/8 | 2/8 | 10/40 | 10/40 |
| 8 | 1/8 | 4/8 | 8/64 | 33/64 |

The largest difference appears at eight cycles, where decomposition and
explicit state carry raise exact-chain success from 1/8 to 4/8.

## Failure and plan diagnostics

The plain arm hit its registered maximum-new-token ceiling on all 32 items and
failed result-array parsing on 24. A ceiling hit does not imply failure by
itself: five capped generations still contained a correct parseable chain.
The host arm had zero trace parse failures and zero external limit hits.

All 32 host ledgers parsed, and all 32 extracted `initial` values matched. The
frozen exact task-string metric was nevertheless 0/32. This is a lexical
fidelity failure, not a JSON failure: for example, the model wrote
`multiply by 2` where the provenance string is
`multiply the result by 2`. Some ledgers that fail this exact-string metric
still execute every transition correctly. The result therefore supports
reliable structured ledger extraction, but not verbatim reproduction of gold
task clauses. A semantic plan scorer would be post hoc and is not substituted
for the frozen metric here.

## Live compiled traces

The preregistered selector chose the lexicographically first host success at
each length, which was the add-first arithmetic item at 2, 3, 5, and 8 cycles.
Each checkpoint passed the independent circuit oracle before live decoding.
The live decoder received one objective prefill; every subsequent input was a
sampled model token. qwenFS responses, phase transitions, the cycle pointer,
and HALT were produced by the compiled model.

| cycles | exact results | phases | controller / FS / HALT | emitted | model-generated | host answers exact |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 2 | `[9, 18]` | 6/6 | pass | 1,475 | 583 | 3/3 |
| 3 | `[11, 22, 19]` | 9/9 | pass | 2,094 | 861 | 4/4 |
| 5 | `[15, 30, 27, 32, 96]` | 15/15 | pass | 3,376 | 1,416 | 6/6 |
| 8 | `[21, 42, 39, 44, 132, 124, 248, 257]` | 24/24 | pass | 5,396 | 2,250 | 9/9 |

Token ownership accounted exactly in every trace. The 8-cycle run contains
2,048 model-owned reasoning tokens and 112 model-owned payload tokens; 3,146
of 5,396 emitted tokens are controller or filesystem forced. All model-owned
ledger/result answer spans match the corresponding host trace token-for-token.
Private bounded reasoning does not match token-for-token because the compiled
and host executions have different low-level contexts; answer equivalence is
the relevant semantic comparison.

These four selected traces are systems-equivalence evidence, not a full
compiled-suite capability estimate. They also do not cover the host-only wins,
because the frozen lexicographic selector chose the add-first family.

## Reproduction artifacts

The machine-readable paired results are under the ignored directory
`results/end-to-end-v3-b256/`:

- `summary.json`: suite-level and paired statistics;
- `plain/` and `host/`: all 64 item records;
- `compiled/`: live runs, controller/filesystem evaluations, token accounts,
  and host-answer equivalence reports for 2, 3, 5, and 8 cycles.

The compiled checkpoints are:

- `build/babyagi-2b-ledger-pointer-b256-m2`
- `build/babyagi-2b-ledger-pointer-b256-m3`
- `build/babyagi-2b-ledger-pointer-b256-m5`
- `build/babyagi-2b-ledger-pointer-b256-m8`

Each checkpoint is approximately 8.8 GB. The exact paired-run command remains
in the preregistration document.

## Interpretation

This experiment supports the narrow claim that a pretrained 2B Qwen model can
run a compiled BabyAGI-style create-ledger / fixed-priority / execute / persist
loop for at least eight cycles, and that the tested JSON-ledger protocol can
improve complete-chain execution over a one-turn baseline. It does not yet
support broad autonomous planning: priority is a compiled cycle pointer, exact
plan strings fail, the suite is deterministic, and two of four task categories
remain at zero exact chains.
