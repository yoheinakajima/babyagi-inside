# BabyAGI v3 end-to-end results and parity correction

## Result

The compiled BabyAGI-style loop is a working systems result and a narrow
arithmetic capability result. Across ten live compiled traces, all ten executed
the expected controller phases, filesystem reads and writes, cycle pointer, and
HALT with exact token accounting. Eight arithmetic traces solved their complete
chains at lengths 2, 3, 5, or 8. Two structured-operation traces ran all eight
cycles correctly at the systems level but produced the wrong task ledger and
therefore failed the benchmark.

This is stronger direct evidence than the host semantic simulator, but it is
not a compiled-suite accuracy estimate: the ten traces were selected for
diagnosis rather than sampled as a complete or representative set.

## Post-run parity correction

The first paired run reported host-loop success of 9/32 versus 5/32 for plain
chat. A subsequent live-compile audit found that the stock-weight host harness
omitted one semantic timestep present before every compiled stage. The compiled
control symbol has a nonzero view in Qwen's pretrained dimensions:

```text
control embedding in native dimensions = 0.25 × newline embedding
```

The original host run therefore did not simulate the model context actually
used by the compiled checkpoint. In particular, its two structured-operation
successes were false positives. The 9/32 result is retained as an audit artifact
but is superseded and must not be used as the paper headline.

The corrected harness flushes pending visible tokens and feeds the control
symbol's pretrained-stack embedding before every stage. It leaves the frozen
benchmark, prompts, decoding, token budgets, and scorer unchanged. Reusing the
validated plain records, the corrected paired diagnostic is:

| arm | exact chains | chain accuracy | exact transitions | transition accuracy |
| --- | ---: | ---: | ---: | ---: |
| plain chat | 5/32 | 15.6% | 23/144 | 16.0% |
| control-aware host proxy | 7/32 | 21.9% | 35/144 | 24.3% |
| paired change | +2 | +6.25 points | +12 | +8.3 points |

There were three host-proxy improvements and one host-proxy regression. The
exact McNemar result is `p = 0.625`; the preregistered item bootstrap interval
is `[-6.25, +18.75]` percentage points. This does not establish a statistical
capability improvement.

All seven control-aware host successes are sequential arithmetic. The other
categories remain at zero exact chains:

| category | plain chains | host-proxy chains | plain transitions | host-proxy transitions |
| --- | ---: | ---: | ---: | ---: |
| ordered literal | 0/8 | 0/8 | 0/36 | 0/36 |
| sequential arithmetic | 5/8 | 7/8 | 23/36 | 34/36 |
| structured operation | 0/8 | 0/8 | 0/36 | 0/36 |
| checkable language | 0/8 | 0/8 | 0/36 | 1/36 |

The host proxy is still an approximation because qwenFS file responses use
extended embeddings that stock Qwen cannot consume directly. The harness feeds
ordinary-token semantic surrogates for those responses. That remaining gap is
material: the proxy stopped during the five-cycle multiply-first item, whereas
the live compiled checkpoint completed it exactly. Host results are therefore
prompt-screening and diagnostic evidence; live compiled traces are authoritative
for compiled behavior.

## Live compiled evidence

The four add-first traces were selected by the preregistered rule. The remaining
six were diagnostic traces chosen after the first paired run: its four apparent
host-only gains, the corrected host's additional three-cycle gain, and its
apparent five-cycle regression.

| item | cycles | benchmark result | controller / FS / HALT | exact result sequence |
| --- | ---: | --- | --- | --- |
| `v3-arithmetic-add-2` | 2 | pass | pass | `[9, 18]` |
| `v3-arithmetic-add-3` | 3 | pass | pass | `[11, 22, 19]` |
| `v3-arithmetic-add-5` | 5 | pass | pass | `[15, 30, 27, 32, 96]` |
| `v3-arithmetic-add-8` | 8 | pass | pass | `[21, 42, 39, 44, 132, 124, 248, 257]` |
| `v3-arithmetic-multiply-2` | 2 | pass | pass | `[21, 25]` |
| `v3-arithmetic-multiply-3` | 3 | pass | pass | `[24, 29, 25]` |
| `v3-arithmetic-multiply-5` | 5 | pass | pass | `[30, 37, 33, 66, 73]` |
| `v3-arithmetic-multiply-8` | 8 | pass | pass | `[39, 49, 45, 90, 97, 194, 185, 196]` |
| `v3-structured-even-8` | 8 | fail | pass | wrong ledger; 0/8 transitions |
| `v3-structured-odd-8` | 8 | fail | pass | wrong ledger; 0/8 transitions |

Every positive trace includes a model-written JSON ledger, one model-written
result per cycle, state persisted through qwenFS, and bounded model-owned
reasoning at each execute stage. Priority is a deterministic compiled ledger
pointer, not a model-owned comparison. The multiply-first five-cycle trace is
especially useful: the live checkpoint produced `[30, 37, 33, 66, 73]` even
though the corrected host proxy stopped while closing its third write.

The negative structured traces isolate the semantic failure from the machinery.
Both checkpoints completed 24/24 phases and halted. The model instead created a
ledger whose first task added the combined amount and whose remaining tasks each
added one. The controller then faithfully executed that bad plan for eight
cycles. This is a useful failure trace, not an agent-capability success.

All ten traces pass exact token-ownership accounting. For example, the
five-cycle multiply-first trace emitted 3,448 tokens: 1,428 model-generated
tokens, including 1,280 bounded-reasoning tokens, and 2,020 ROM, filesystem, or
controller-forced tokens. The eight-cycle multiply-first trace emitted 5,610
tokens and completed all 24 phases.

## Plan diagnostics

All 32 control-aware host ledgers parsed, and 31/32 extracted `initial` values
matched. The frozen exact task-string metric was 0/32 because semantically close
phrases such as `multiply by 2` do not equal the provenance string
`multiply the result by 2`. No post-hoc semantic scorer is substituted for that
metric.

The structured negative traces reveal a more important failure than lexical
variation: the generated operations themselves are wrong. A future benchmark
should report lexical fidelity and semantic task fidelity separately.

## What the experiment supports

The evidence supports this claim:

> A pretrained Qwen3.5-2B checkpoint can execute an in-weight compiled
> BabyAGI-style create-ledger / fixed-priority / execute / persist loop for at
> least eight cycles, with model-owned JSON creation and bounded reasoning. On
> the tested traces it reliably executes several sequential-arithmetic chains,
> while structured task creation remains unsolved.

It does not support broad autonomous planning, a model-owned prioritization
claim, a statistically established improvement over plain chat, or a 32-item
compiled accuracy claim. Establishing the latter requires live decoding all 32
items or a new preregistered representative sample.

## Reproduction artifacts

The ignored result roots are:

- `results/end-to-end-v3-b256/`: original paired run, four preregistered live
  traces, and six diagnostic live traces;
- `results/end-to-end-v3-b256-control-surrogate/`: corrected control-aware host
  diagnostic and the validated reused plain records;
- `results/end-to-end-v3-b256/control-surrogate-diagnostic/`: direct structured
  parity reproduction.

The compiled checkpoints are:

- `build/babyagi-2b-ledger-pointer-b256-m2`
- `build/babyagi-2b-ledger-pointer-b256-m3`
- `build/babyagi-2b-ledger-pointer-b256-m5`
- `build/babyagi-2b-ledger-pointer-b256-m8`

The corrected host diagnostic is reproduced with:

```sh
uv run python -m qfs.end_to_end models/Qwen3.5-2B \
  benchmarks/babyagi-v3.json \
  results/end-to-end-v3-b256-control-surrogate \
  --think-tokens 256 --plain-answer-tokens 256 \
  --control-surrogate-scale 0.25 \
  --plain-from results/end-to-end-v3-b256 --device mps
```
