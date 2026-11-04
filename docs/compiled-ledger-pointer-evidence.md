# Compiled JSON-ledger pointer with bounded Qwen reasoning

## Status

This experiment compiles the repaired BabyAGI-style protocol into the widened
Qwen3.5-2B checkpoint. It is a positive systems result and a modest semantic
improvement, not evidence of a generally capable agent.

The winning architecture is `babyagi_ledger_pointer`:

1. Qwen writes one persistent `ledger.json` containing `initial` and an ordered
   `tasks` array. The first instruction retains the BabyAGI homage: “INITIAL
   TASK — Create the first task, then the complete ordered task ledger.”
2. The compiled controller holds the cycle index. It reads `ledger.json` on
   every cycle and instructs Qwen to execute exactly array entry `i`. There is
   no semantic selector, host task parser, or host scheduler.
3. qwenFS carries `result.json` between cycles.
4. Each execute phase gives Qwen exactly 128 ordinary-vocabulary reasoning
   tokens. The compiled ROM then forces `</think>` and opens the result write;
   Qwen owns the JSON result value and the normal tool-call close.

For five cycles this uses 15 microstates and controller scratch dimensions
3072–3401, leaving 694 of 4096 residual dimensions unused.

## Compiler result

Checkpoint:

```text
build/babyagi-2b-ledger-pointer-b128-m5
```

The scripted checkpoint oracle passes:

| Check | Result |
|---|---:|
| Forced controller and qwenFS tokens | 1,716 / 1,716 exact |
| Bounded free-span positions | 640 / 640 legal ordinary-vocabulary outputs |
| Minimum legal-over-control gate margin | 1,500.0 logits |
| Five-cycle live controller | PASS |
| Exact live qwenFS reads | 9 / 9 |
| Compiled HALT | PASS |
| Agent-inactive top-1 parity | 4 / 4 |
| Maximum native-to-compiled KL on parity smoke | 0.001896 nats |

The KL result passes the repository's existing 0.03 smoke threshold but does
not meet the aspirational 0.001 paper threshold proposed in the external Opus
review. That stronger non-interference study remains open.

### Important gate correction

The first free-span gate gave every legal token the same large positive logit
offset. In bf16, this erased nearby native-logit differences and produced
punctuation ties. The retained construction leaves legal-token logits
untouched and applies a large negative offset only to illegal control tokens.
This restored normal Qwen reasoning while preserving a large legality margin.

## Positive compiled traces

### Five-cycle synthetic arithmetic trace

Objective:

> Begin with 2. First add 3. Next double the result. Then subtract 4. Next
> square the result. Finally add 1.

The live checkpoint wrote this persistent ledger:

```json
{
  "initial": 2,
  "tasks": [
    "add 3",
    "double the result",
    "subtract 4",
    "square the result",
    "add 1"
  ]
}
```

Its exact state chain was:

```text
5 -> 10 -> 6 -> 36 -> 37
```

Evidence:

- `results/ledger-pointer-compiled/live-arithmetic-5.json`
- `results/ledger-pointer-compiled/live-arithmetic-5.accounting.json`
- `results/ledger-pointer-compiled/live-arithmetic-5.equivalence.json`

### Frozen v2 five-cycle trace

Item `arithmetic-two-5` from `benchmarks/babyagi-v2.json` ran through the live
compiled checkpoint and passed every exact-state check:

```text
10 -> 16 -> 48 -> 40 -> 41
```

The trace has five complete cycles, nine token-exact filesystem reads, six
writes, 640 model-owned reasoning tokens, and compiled HALT.

Evidence:

- `results/ledger-pointer-compiled/v2-arithmetic-two-5.json`
- `results/ledger-pointer-compiled/v2-arithmetic-two-5.evaluation.json`
- `results/ledger-pointer-compiled/v2-arithmetic-two-5.accounting.json`
- `results/ledger-pointer-compiled/v2-arithmetic-two-5.equivalence.json`

## Provenance and host equivalence

For the compiled v2 trace:

| Token owner | Count | Share |
|---|---:|---:|
| Compiled ROM + qwenFS + controller | 1,960 | 71.637% |
| Model generated | 776 | 28.363% |
| — bounded private reasoning | 640 | 23.392% of all emitted tokens |
| — ledger/result payload | 76 | 2.778% |
| — ordinary tool-call close | 60 | 2.193% |

The host/compiled comparison must be stated carefully:

- The model-written ledger and every model-written answer/tool suffix are
  token-identical: 6/6 answer stages exact.
- The private 128-token reasoning spans are not token-identical: 0/5 exact.
  Small numerical and context differences compound autoregressively even when
  the final answer remains identical.
- Therefore this artifact demonstrates controller, filesystem, result-payload,
  and semantic-state equivalence on the positive traces. It does **not** yet
  demonstrate full token-for-token equivalence of hidden reasoning.

## Semantic suites

All suite numbers below use the stock-weight host simulator with the exact
compiled program and greedy decoding. They screen semantic quality; only live
traces prove in-weight control.

| Suite | Automated pass | Notes |
|---|---:|---|
| Mutable prompt-adversarial development set | 0/5 | Correctly exposes literal-format and example-value sensitivity. |
| Frozen v1 (used during prior development) | 5/12 | Four exact-state passes; Aster passes automation but awaits/fails a stricter manual reading. |
| Value-disjoint v2 replicate | 3/12 | Two exact arithmetic passes; Lumen passes automation but awaits/fails a stricter manual reading. |

Exact v1 passes:

- `literal-cinder-5`: 5/5 states;
- `arithmetic-seven-3`: 3/3 states;
- `arithmetic-ten-4`: 4/4 states;
- `arithmetic-three-5`: 5/5 states.

Exact v2 passes:

- `arithmetic-eight-3`: 3/3 states;
- `arithmetic-two-5`: 5/5 states, also confirmed in the live compiled trace.

These numbers improve on the earlier host ledger protocol (2/12 exact on v1 and
1/12 exact on v2), but the suites are small and v1 is a development set. They
support further measurement, not a capability headline.

## What holds and what does not

What now holds:

- a persistent model-written JSON task ledger lives inside qwenFS;
- a deterministic cycle pointer and all file operations execute in weights;
- five exact 128-token model-owned reasoning spans execute in weights;
- two independent five-cycle live traces reach HALT with exact state chains;
- result payloads match the host protocol token for token on those traces.

What does not yet hold:

- full host/compiled private-thought token identity;
- the proposed KL <= 0.001 non-interference threshold;
- broad structured-data or language competence;
- evidence that semantic prioritization is required or compiled;
- a preregistered gradient suite with confidence intervals.

The defensible current phrase is:

> A bounded BabyAGI-style JSON-ledger controller, deterministic task pointer,
> persistent filesystem, and fixed model-owned reasoning spans were compiled
> into a widened Qwen3.5-2B checkpoint. On positive five-cycle traces the live
> checkpoint preserves the host protocol's ledger and result payloads exactly;
> overall semantic success remains limited by the base model.

Do not yet say “a semantic agent is baked into Qwen” or claim full
token-for-token equivalence.

## Reproduction

```sh
uv run python -m qfs.agent models/Qwen3.5-2B \
  build/babyagi-2b-ledger-pointer-b128-m5 \
  --cycles 5 --protocol babyagi_ledger_pointer --device mps

uv run python -m tests.test_agent_circuit \
  build/babyagi-2b-ledger-pointer-b128-m5 --device mps

uv run python -m qfs.agent_decode \
  build/babyagi-2b-ledger-pointer-b128-m5 \
  "Start with 2. First multiply by 5; next add 6; then multiply by 3; then subtract 8; finally add 1. Perform one operation per task in order and return the current number." \
  --output results/ledger-pointer-compiled/v2-arithmetic-two-5.json \
  --temperature 0 --device mps
```
