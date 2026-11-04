# BabyAGI v3 paired end-to-end preregistration

## Question

Does the frozen 256-token JSON-ledger loop preserve or improve complete-chain
success relative to plain Qwen on the same deterministic objectives?

This experiment is frozen after the complete single-step budget curve and
before any v3 end-to-end model run.

## Fixed inputs

- Suite: `benchmarks/babyagi-v3.json`
- Suite SHA-256:
  `e880e043f7ac412aaab938036cccd5f79305aed4cd0c13a7204e09fe07c1dff4`
- Model: `models/Qwen3.5-2B`, bf16, MPS
- Decoding: greedy
- Agent protocol: `babyagi_ledger_pointer`
- Per-cycle agent reasoning: exactly 256 ordinary-vocabulary tokens
- Result ceiling: 256 tokens per host write

## Arms

### Plain Qwen

One thinking-enabled chat turn solves the entire objective and returns the
complete result after every operation. Its maximum new-token budget is:

```text
cycles × 256 + 256
```

The final 256 tokens allow the result array after the reasoning allowance. A
hit at this external ceiling is recorded rather than silently retried.

### Host JSON-ledger loop

The stock model runs the exact compiled-agent program. Python advances the
stages and simulates the Qwen-visible filesystem responses. Qwen writes one
ledger, receives the fixed cycle pointer, reasons for exactly 256 tokens per
execute phase, and writes one JSON result per cycle.

This arm measures the semantic protocol. It is not evidence that the compiled
controller or qwenFS ran in weights.

## Metrics

Primary:

- exact complete-chain success per objective;
- paired host-minus-plain chain-success delta;
- exact McNemar test over objectives;
- objective bootstrap 95% interval for the paired delta.

Secondary:

- exact transition accuracy;
- chain and transition accuracy by category and source-chain length;
- prefix survival through each cycle;
- parse failures and plain-chat ceiling hits;
- exact JSON-ledger extraction rate.

No prompt or scorer changes are permitted after results are observed. V3 tests
deterministic task extraction and execution, not open-ended planning.

## Live compiled traces

Compiled checkpoints are fixed by cycle count. After host results are complete,
select one trace at each length 2, 3, 5, and 8 using this rule:

1. among exact host-chain passes at that length, take the lexicographically
   smallest item ID;
2. if no host item passes, take the lexicographically smallest item and label
   it a negative trace.

Compile missing 256-token checkpoints for those lengths. Each selected live
trace must pass the circuit oracle, controller/filesystem/HALT evaluation,
benchmark scoring, token accounting, and host/compiled answer comparison.

Four selected compiled traces are systems-equivalence evidence, not an estimate
of compiled-suite capability. A full compiled-suite estimate would require all
32 live decodes.

## Reproduction

```sh
uv run python -m qfs.end_to_end models/Qwen3.5-2B \
  benchmarks/babyagi-v3.json results/end-to-end-v3-b256 \
  --think-tokens 256 --plain-answer-tokens 256 --device mps
```
