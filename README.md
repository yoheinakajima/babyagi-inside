# BabyAGI Inside

BabyAGI Inside compiles a bounded BabyAGI-style agent loop into a widened Qwen3.5-2B checkpoint: the host supplies one objective and then only samples until HALT, while compiled controller state advances a fixed task pointer, qwenFS persists the ledger and results, and Qwen writes the plan, reasoning, and payloads. The result is a systems demonstration—not a claim of general agency: the ten archived live traces verify exact controller, filesystem, HALT, and token-accounting behavior, while their semantic results show reliable sequential arithmetic and an unresolved structured-planning bottleneck.

Repository: <https://github.com/yoheinakajima/babyagi-inside>

## The host loop

```python
ids = prefill(objective) + [START]
while ids[-1] != HALT:
    ids.append(argmax(model(ids)))
```

The host does not parse phases, execute tools, inject follow-up prompts, repair outputs, or manage a task list after the initial objective.

## Checkpoint layout

The widened residual stream is partitioned as follows:

| Dimensions | Owner | Purpose |
| ---: | --- | --- |
| 0–2047 | Qwen | Native Qwen3.5-2B model state |
| 2048–3071 | qwenFS | Compiled filesystem state and operations |
| 3072–3208 | Agent | Controller and bounded-agent scratch state |

Cycle count is fixed at checkpoint construction. The verified set uses separate checkpoints for 2, 3, 5, and 8 cycles.

## Rebuild checkpoints locally

Checkpoint weights are rebuilt locally and are not shipped in this Git repository. After placing Qwen3.5-2B under `models/Qwen3.5-2B`, run:

```sh
scripts/build_checkpoints.sh
```

The script invokes the existing `qfs.agent` build command for `babyagi_ledger_pointer`, 256 reasoning tokens, and cycle counts 2, 3, 5, and 8. Each resulting directory is approximately 8.8 GB (`du -sh` on the reference build).

- Expected build time per checkpoint: `TODO(verify)`—no build-duration measurement is preserved in the tracked artifacts.
- Expected peak memory: `TODO(verify)`—no peak-memory measurement is preserved in the tracked artifacts. Build one checkpoint at a time until this is measured.
- Qwen3.5-2B source revision: `TODO(verify)`—the exact 2B snapshot hash is not recorded in tracked files or the local Hugging Face cache.
- qwenFS-derived compiler source: `48f4f45b6269bf8a0f27ea954a05b2caced929a6`; the repository documentation records upstream qwenFS baseline `fe3f5c0dbebe5c827781e98ad839aecebcce32fa`.

## Reproduce the ten archived live traces

Requirements: Apple Silicon with MPS, the local `models/Qwen3.5-2B` base model, `uv`, and roughly 10 GB free disk per compiled checkpoint. These are the command forms implemented by the repository's existing CLI modules. Historical records confirm greedy decoding, seed 0, a 256-token reasoning span, and the checkpoint settings shown below. The exact historical base-model revision remains `TODO(verify)` before claiming byte-for-byte checkpoint reproduction.

Build the four cycle-specific checkpoints:

```sh
for cycles in 2 3 5 8; do
  uv run python -m qfs.agent models/Qwen3.5-2B \
    "build/babyagi-2b-ledger-pointer-b256-m${cycles}" \
    --cycles "${cycles}" --protocol babyagi_ledger_pointer \
    --think-tokens 256 --device mps
done
```

Then decode, inspect, account for, and score the ten objectives. Each invocation starts with a fresh qwenFS cache. The two structured items are expected to fail the exact semantic scorer; the command preserves those failures.

```sh
trace_ids='v3-arithmetic-add-2
v3-arithmetic-add-3
v3-arithmetic-add-5
v3-arithmetic-add-8
v3-arithmetic-multiply-2
v3-arithmetic-multiply-3
v3-arithmetic-multiply-5
v3-arithmetic-multiply-8
v3-structured-even-8
v3-structured-odd-8'

mkdir -p results/babyagi-inside-live
printf '%s\n' "$trace_ids" | while IFS= read -r trace_id; do
  cycles="$(jq -r --arg id "$trace_id" '.items[] | select(.id == $id) | .cycles' benchmarks/babyagi-v3.json)"
  objective="$(jq -r --arg id "$trace_id" '.items[] | select(.id == $id) | .objective' benchmarks/babyagi-v3.json)"
  prefix="results/babyagi-inside-live/${trace_id}"

  uv run python -m qfs.agent_decode \
    "build/babyagi-2b-ledger-pointer-b256-m${cycles}" "$objective" \
    --output "${prefix}-run.json" --device mps \
    --temperature 0 --seed 0 --limit 6000
  uv run python -m qfs.agent_evaluate \
    "${prefix}-run.json" --output "${prefix}-evaluation.json"
  uv run python -m qfs.agent_account \
    "${prefix}-run.json" --output "${prefix}-account.json"
  uv run python -m qfs.benchmark score-evaluation \
    benchmarks/babyagi-v3.json "$trace_id" "${prefix}-evaluation.json" || true
done
```

The `|| true` keeps the batch running after the scorer returns status 1 for either preserved structured-task failure; it does not change an artifact or score. The clean, compact exports of the historical runs are in [`traces/`](traces/README.md). Regenerating those JSON exports is CPU-only and does not load a model.

## What is verified

- Ten completed live compiled-checkpoint diagnostics reach HALT with the expected number of cycles and exact controller, qwenFS read/write, and disjoint token-accounting checks.
- All eight sequential arithmetic chains match their frozen expected results, at 2, 3, 5, and 8 cycles.
- Both eight-cycle structured-operation traces complete mechanically but fail exact task semantics: the controller executes the ledger Qwen wrote, and the ledger is wrong.
- The host provides nothing after the objective; the checkpoints run greedily until their compiled HALT token.

These ten traces were selected for diagnosis, not sampled to estimate accuracy. `TODO(sweep)`: report the frozen 32-item live-compiled sweep, its paired plain-Qwen comparison, confidence intervals, and any system exceptions after that run is complete and validated.

## What remains exploratory

- Whether the compiled protocol improves capability over plain Qwen. The earlier corrected host proxy was statistically inconclusive and is not a substitute for the live sweep.
- Post-hoc operation-signature plan fidelity. Frozen exact task strings remain the primary plan metric.
- Runtime comparisons and the compiled circuit-oracle check on CUDA.
- Runtime-variable cycle count; the current controller uses one checkpoint per fixed cycle count.

## Credits and licenses

- [qwenFS](https://github.com/vixhal-baraiya/qwenFS) by Vixhal Baraiya supplies the compiled filesystem foundation and is MIT-licensed.
- [Qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal is credited for compiler lineage.
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) is provided by the Qwen team under Apache License 2.0.
- BabyAGI was created by Yohei Nakajima.

Repository code is MIT-licensed; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Qwen-derived checkpoint weights are separately governed by Qwen's Apache-2.0 terms and are not relicensed by this repository's MIT license.
