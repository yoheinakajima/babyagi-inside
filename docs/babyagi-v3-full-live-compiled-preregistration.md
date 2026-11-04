# BabyAGI v3 full live-compiled sweep preregistration

Frozen before the first item in the 32-item compiled sweep on 2026-11-04.

## Question

What is the exact end-to-end capability of the live compiled checkpoint on all
32 frozen v3 objectives, and how does it compare with the already frozen plain
Qwen baseline?

The complete sweep replaces selected-trace and host-proxy capability estimates.
It does not change the compiled protocol, prompts, reasoning budget, benchmark,
or scorer.

## Frozen inputs

- Suite: `benchmarks/babyagi-v3.json`
- Suite ID: `babyagi-execution-gradient-v3`
- Suite SHA-256:
  `e880e043f7ac412aaab938036cccd5f79305aed4cd0c13a7204e09fe07c1dff4`
- Items: 32, with two objectives per category at each length 2, 3, 5, and 8
- Model family: Qwen3.5-2B, bf16, MPS
- Protocol: `babyagi_ledger_pointer`
- Bounded model reasoning: 256 tokens per execute stage
- Decoding: greedy (`temperature = 0`), seed 0
- External safety bound: 7,000 emitted tokens per item; no retry on a bound hit
- Run order: ascending cycle count, then frozen suite order
- Frozen plain comparison: the 32 checksum-bound records under
  `results/end-to-end-v3-b256/plain/`, with 5/32 exact chains

The cycle count is fixed at checkpoint construction. This experiment therefore
uses four separate checkpoints rather than a runtime-configurable program
counter:

| cycles | checkpoint tree SHA-256 | bytes |
| ---: | --- | ---: |
| 2 | `a09216d695e10cadf89f6ebf1bdf64d12fe2c30e30ac568f729eb386af8aebf0` | 9,440,394,906 |
| 3 | `18faa8cb598775fd8a0159359e8eafffe767b94caba844d857f2d94e9a423c11` | 9,440,398,327 |
| 5 | `42a221cbd17404705577d751f5b881af2d4328f276adb0e366517efbfa222778` | 9,440,405,169 |
| 8 | `88a02c618c073cdb7abaef0b5e157bc9ab635608f176571b9e5d0bf0793c8da6` | 9,440,415,432 |

The generated pre-run manifest has experiment SHA-256
`03ca577087e9a3d235ac5a6893137d994882056697cf7e31a8a39e8b260f611f`.
It records every checkpoint file hash, all 32 plain-record hashes, and hashes of
the runner, decoder, evaluator, token accountant, benchmark scorer, paired
statistics, and exploratory plan scorer. A changed source, suite, checkpoint,
baseline, or setting is a different experiment and cannot resume this one.

## Execution contract

Each item begins with a fresh qwenFS cache. The host supplies one objective chat
prefill plus checkpoint START. It then greedily feeds each emitted model token
back to the checkpoint until checkpoint HALT or the 7,000-token external bound.
The host does not parse phases, inject later prompts, execute tools, repair a
trace, or retry a failure.

Weights are loaded once per cycle count only to avoid redundant checkpoint
loading. No attention cache, filesystem state, emitted token, or decoded value
carries between objectives.

The runner writes `run.json` before post-processing and then independently
writes controller/filesystem evaluation, exact token accounting, benchmark
score, and plan score. Each completed item record binds those artifacts by
SHA-256. An interrupted run may resume only a matching partial or complete
record; it never silently overwrites a mismatch.

## Metrics

Primary systems metrics:

- exact controller sequence and checkpoint HALT;
- token-exact qwenFS reads of the latest prior write;
- all expected cycles and phases complete;
- exact disjoint accounting of every emitted token.

Primary capability metrics:

- exact complete-chain success across all 32 items;
- exact transition accuracy;
- paired compiled-minus-plain chain-success difference;
- exact McNemar test and item bootstrap 95% interval for that paired difference.

Secondary breakdowns:

- exact chains and transitions by category and chain length;
- Wilson 95% interval for compiled chain accuracy;
- failure taxonomy separating controller, filesystem, accounting, plan, and
  execution failures;
- elapsed decode time and emitted-token count per item.

The frozen primary plan metric is exact JSON-ledger task-string equality, plus
exact initial-state extraction. It remains primary even if its score is zero.

An additional deterministic operation-signature score is reported as
**exploratory and post hoc** because it was designed after observing host
ledgers. It canonicalizes only operation families in the frozen v3 generator
(for example, `multiply 7 by 3` and `multiply the result by 3`). Unknown task
language receives no semantic credit. This metric may describe failure modes
but cannot rescue or replace the frozen exact-string result.

## Stopping and reporting

Run every item exactly once under greedy decoding. Preserve semantic and system
failures. Do not tune prompts, budgets, checkpoints, parsing, task signatures,
or scoring against the sweep. Report the result even if it is 0/32, equal to
plain chat, statistically inconclusive, or worse than the host proxy.

The full sweep is complete only when all 32 checksum-bound records exist. The
paper will distinguish:

1. exact compiled machinery;
2. exact compiled task performance;
3. exploratory semantic plan fidelity;
4. the already frozen plain-chat comparison.

CUDA circuit-oracle validation and performance timing beyond MPS wall time
remain separate open work and are not conditions for this sweep's result.

## Reproduction

```sh
uv run python -m qfs.compiled_sweep \
  benchmarks/babyagi-v3.json \
  results/babyagi-v3-full-live-compiled \
  --checkpoint-pattern 'build/babyagi-2b-ledger-pointer-b256-m{cycles}' \
  --plain-root results/end-to-end-v3-b256 \
  --device mps --seed 0 --temperature 0 --limit 7000
```
