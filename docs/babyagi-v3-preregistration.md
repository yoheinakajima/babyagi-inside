# BabyAGI execution-budget and gradient-suite preregistration

## Frozen artifacts

- Generator: `qfs/generate_benchmark_v3.py`
- Suite: `benchmarks/babyagi-v3.json`
- Benchmark ID: `babyagi-execution-gradient-v3`
- Frozen: 2026-11-03, before any model evaluation
- SHA-256: `e880e043f7ac412aaab938036cccd5f79305aed4cd0c13a7204e09fe07c1dff4`

The deterministic generator emits 32 objectives: two unseen template families
in each of four categories at chain lengths 2, 3, 5, and 8. The suite contains
144 gold single-step records. Every record fixes the objective, cycle, incoming
state, task, expected state, and comparison normalization.

The four categories are ordered literal transformations, sequential arithmetic,
structured JSON operations, and constrained language editing. All v3 items use
exact state-chain scoring. V3 does not test open-ended planning; that will be a
separate suite rather than an ambiguous fifth category.

## Experiment A: bounded execution pilot

Question: how many fixed model-owned reasoning tokens does Qwen3.5-2B need to
execute a gold task against a gold incoming state?

- Model: local `models/Qwen3.5-2B`, bf16, MPS.
- Decoding: greedy.
- Budgets: 16, 32, 64, 128, and 256 reasoning tokens.
- Answer ceiling: 256 tokens after the forced result prefix.
- Pilot: eight deterministically selected steps per category, 32 total.
- Primary metric: exact single-step accuracy.
- Secondary metrics: parse failures, accuracy by category, Wilson 95% intervals,
  and answer-token count.
- Selection rule: compile the two budgets with highest exact accuracy; resolve
  ties in favor of the smaller budget.

The pilot selects checkpoint budgets only. It is not a final capability result.
After selection, the complete 144-step curve is run without changing the
prompt, scorer, budgets, or selection rule.

The controller forces an exact reasoning span and suppresses control tokens
inside it, so natural `</think>` termination is unobservable in this experiment.
Do not claim a natural-termination curve from these data.

## Experiment B: compiled checkpoints

Compile the two pilot-selected budgets with the unchanged
`babyagi_ledger_pointer` protocol. Because compiled checkpoints have a fixed
cycle count, first compile five-cycle variants for direct comparison with the
existing 128-token evidence. Compile an eight-cycle variant only after its
scratch layout and circuit oracle pass.

For each checkpoint require:

1. every forced ROM/controller/qwenFS token exact in the circuit oracle;
2. every free-span position prefers native vocabulary over control tokens;
3. controller, filesystem reads/writes, and HALT pass on a live trace;
4. ledger/result payload comparison against the host run;
5. separate reporting of answer equality and private-reasoning equality; and
6. inactive-agent native parity, including maximum KL.

## Experiment C: complete gradient evaluation

Run the frozen suite through three arms using the chosen budget:

1. plain Qwen chat with a matched total token budget;
2. host-orchestrated JSON ledger plus deterministic pointer; and
3. the compiled checkpoint for supported chain lengths.

Report exact chain success, per-step accuracy, and failure hazard by length.
Host/compiled equivalence is the systems endpoint. Any claim that the loop
improves capability requires a paired advantage over plain chat; compilation
alone does not add semantic capability.

## Reproduction

```sh
uv run python -m qfs.generate_benchmark_v3 \
  --output /tmp/babyagi-v3.json

uv run python -m qfs.benchmark validate benchmarks/babyagi-v3.json

uv run python -m qfs.execute_budget models/Qwen3.5-2B \
  benchmarks/babyagi-v3.json results/execution-budget-v3-pilot \
  --budgets 16 32 64 128 256 --max-steps-per-category 8 --device mps
```

The budget runner writes one JSON result per step and resumes only when the
suite hash, runner-source hash, budget, item, and cycle all match.
