# BabyAGI v3 execution-budget pilot and B256 compiled trace

## Result

The preregistered 32-step pilot selected 256 and 128 reasoning tokens for
compilation. The 256-token checkpoint then completed an unseen five-cycle v3
arithmetic chain exactly inside the widened Qwen3.5-2B checkpoint.

These are pilot and positive-trace results. They do not replace the complete
144-step curve or establish broad agent capability.

## Frozen inputs

- Suite: `benchmarks/babyagi-v3.json`
- Benchmark ID: `babyagi-execution-gradient-v3`
- Suite SHA-256:
  `e880e043f7ac412aaab938036cccd5f79305aed4cd0c13a7204e09fe07c1dff4`
- Preregistration: `docs/babyagi-v3-preregistration.md`
- Model: `models/Qwen3.5-2B`, bf16, greedy decoding on MPS
- Pilot: eight gold task/state transitions per category, covering cycle
  positions 1–8; 32 transitions total
- Selection rule: highest exact step accuracy, ties favor the smaller budget

The first invocation used a faulty sampler that selected only cycle-1 records.
It was stopped before any cross-budget comparison. Its 16-token outputs are
preserved under `results/execution-budget-v3-pilot-aborted-cycle1-selection`
and excluded. The correction and disclosure were committed before restarting.

## Pilot budget curve

| Fixed think tokens | Exact steps | Accuracy | Wilson 95% interval | Parse failures |
|---:|---:|---:|---:|---:|
| 16 | 19/32 | 59.4% | 42.3–74.5% | 0 |
| 32 | 19/32 | 59.4% | 42.3–74.5% | 0 |
| 64 | 21/32 | 65.6% | 48.3–79.6% | 0 |
| 128 | 25/32 | 78.1% | 61.2–89.0% | 0 |
| 256 | 28/32 | 87.5% | 71.9–95.0% | 0 |

Category accuracy:

| Budget | Literal | Arithmetic | Structured | Language editing |
|---:|---:|---:|---:|---:|
| 16 | 6/8 | 4/8 | 4/8 | 5/8 |
| 32 | 6/8 | 4/8 | 4/8 | 5/8 |
| 64 | 6/8 | 4/8 | 5/8 | 6/8 |
| 128 | 7/8 | 6/8 | 6/8 | 6/8 |
| 256 | 7/8 | 7/8 | 7/8 | 7/8 |

The pilot supports budget starvation as one important cause of the earlier
failures. It does not explain every failure: each category still has one miss
at 256 tokens, and some operations fail at both 128 and 256. The complete
144-step run is required before estimating stable category accuracy.

Machine-readable pilot evidence:

- `results/execution-budget-v3-pilot/summary.json`
- one resumable record per budget, item, and cycle below that directory

## Selected checkpoints

The precommitted rule selected:

1. 256 tokens: `build/babyagi-2b-ledger-pointer-b256-m5`
2. 128 tokens: `build/babyagi-2b-ledger-pointer-b128-m5`

The 128-token checkpoint was already built and verified. The new 256-token
checkpoint uses 15 microstates and scratch dimensions 3072–3529, leaving 566
of 4096 residual dimensions unused.

### B256 compiler oracle

| Check | Result |
|---|---:|
| Forced controller and qwenFS tokens | 1,716 / 1,716 exact |
| Bounded free-span positions | 1,280 / 1,280 legal |
| Minimum legal-over-control margin | 1,500.0 logits |
| Inactive-agent top-1 agreement | 4 / 4 |
| Maximum native-to-compiled KL | 0.001896 nats |

The KL result passes the repository smoke threshold of 0.03 but remains above
the aspirational paper threshold of 0.001.

## Live B256 v3 trace

Item: `v3-arithmetic-add-5`

Objective:

> Start with the number 9. First add 6; next multiply the result by 2; then
> subtract 3; then add 5; finally multiply the result by 3.

The live compiled checkpoint wrote the correct ledger and exact state chain:

```text
15 -> 30 -> 27 -> 32 -> 96
```

Controller evidence:

- five complete cycles;
- nine exact qwenFS reads;
- six expected writes;
- controller and filesystem checks pass;
- compiled HALT reached;
- all five exact-state checks pass.

Token accounting:

| Owner | Tokens | Share |
|---|---:|---:|
| Compiled ROM + qwenFS + controller | 1,960 | 58.057% |
| Model generated | 1,416 | 41.943% |
| — bounded private reasoning | 1,280 | 37.915% of all emitted tokens |
| — ledger/result payload | 76 | 2.251% |
| — tool-call closure | 60 | 1.777% |

Host/compiled comparison:

- model-written ledger and result/tool suffixes: 6/6 stages token-identical;
- private 256-token reasoning: 0/5 stages token-identical;
- all controller-only stages exact;
- full model-token identity is therefore false.

This reproduces the 128-token boundary: compiled control and semantic payloads
hold on the positive trace, while private autoregressive thoughts diverge.

Evidence:

- `results/ledger-pointer-b256-compiled/v3-arithmetic-add-5.json`
- `results/ledger-pointer-b256-compiled/v3-arithmetic-add-5.evaluation.json`
- `results/ledger-pointer-b256-compiled/v3-arithmetic-add-5.accounting.json`
- `results/ledger-pointer-b256-compiled/v3-arithmetic-add-5.equivalence.json`
- `results/ledger-pointer-b256-host/v3-arithmetic-add-5.json`

## Next gate

Run the complete 144-step curve without changing the frozen suite, prompts,
budgets, scorer, or selection rule. Then compare plain chat, the host ledger
loop, and compiled checkpoints on the supported chain lengths. Do not use the
pilot accuracy as the final paper estimate.
