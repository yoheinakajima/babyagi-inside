# BabyAGI Inside

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23966217.svg)](https://doi.org/10.5281/zenodo.23966217)

BabyAGI Inside compiles a bounded BabyAGI-style agent loop into widened Qwen3.5-2B and Qwen3.5-4B checkpoints: the host supplies one objective and then only samples until HALT, while compiled controller state advances a fixed task pointer, qwenFS persists the ledger and results, and Qwen writes the plan, reasoning, and payloads. The result is a systems demonstration—not a claim of general agency. On 2B, the ten archived live traces verify exact controller, filesystem, HALT, and token-accounting behavior, 30 of the full 32-item sweep's runs pass every system check, and the semantic results show reliable sequential arithmetic and an unresolved structured-planning bottleneck. On 4B, the same compiled loop runs every phase of all 32 frozen objectives exactly and completes 28 of them, against 15 for plain Qwen3.5-4B.

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
| 3072–3547 | Agent | Controller and bounded-agent scratch state (3072–3511 at 2 cycles, 3072–3547 at 8) |

The Qwen3.5-4B checkpoints use the same 4,096-dimension residual and keep qwenFS where it is. Qwen3.5-4B's native residual is 2,560 dimensions wide, so it continues above qwenFS, and the agent moves up:

| Dimensions | Owner | Purpose |
| ---: | --- | --- |
| 0–2047 | Qwen | Native Qwen3.5-4B model state, first 2,048 dimensions |
| 2048–3071 | qwenFS | Compiled filesystem state and operations, unchanged |
| 3072–3583 | Qwen | Native Qwen3.5-4B model state, remaining 512 dimensions |
| 3584–4059 | Agent | Controller and bounded-agent scratch state (3584–4023 at 2 cycles, 3584–4059 at 8) |

Qwen3.5-4B's DeltaNet layers have 32 value heads over 16 key heads, so compiled qwenFS head h writes value head 2h and the other value heads stay zero. Qwen3.5-2B has 16 of each, and its checkpoints are byte-identical to the release builds.

Cycle count is fixed at checkpoint construction. The verified sets use separate checkpoints for 2, 3, 5, and 8 cycles.

## Rebuild checkpoints locally

Checkpoint weights are rebuilt locally and are not shipped in this Git repository. After placing Qwen3.5-2B under `models/Qwen3.5-2B`, or Qwen3.5-4B under `models/Qwen3.5-4B`, run:

```sh
scripts/build_checkpoints.sh      # Qwen3.5-2B
scripts/build_checkpoints.sh 4b   # Qwen3.5-4B
```

The script invokes the existing `qfs.agent` build command for `babyagi_ledger_pointer`, 256 reasoning tokens, and cycle counts 2, 3, 5, and 8. Each resulting directory is approximately 8.8 GB for 2B and 15 GB for 4B (`du -sh`).

- Build time per checkpoint on an Apple M5 (32 GB): 31–36 s for 2B and 40–58 s for 4B.
- Peak memory: macOS reports a peak memory footprint of 18.9 GB for a 2B build (one 5-cycle build measured) and 25.3–26.4 GB for 4B; maximum resident set size is lower, 13.8–16.1 GB for 2B and 12.6–15.0 GB for 4B. Builds allocate the widened model directly in bf16. Before that change a 2B build peaked at a 23–24 GB footprint (max RSS 11–12 GB), and the fp32 allocation of the widened 4B model alone would be about 33 GB (an estimate; it was never run).
- Builds are deterministic: rebuilding from the same inputs reproduces each checkpoint's tree SHA-256. The digests are in the [2B](docs/babyagi-v3-full-live-compiled-preregistration.md) and [4B](docs/qwen35-4b-full-live-compiled-preregistration.md) sweep preregistrations.
- Qwen3.5-2B content pin: [`provenance/qwen3.5-2b.sha256`](provenance/qwen3.5-2b.sha256) records SHA-256 for the exact weights, config, tokenizer, and tokenizer config used locally. The upstream revision ID was not recorded.
- Qwen3.5-4B content pin: [`provenance/qwen3.5-4b.sha256`](provenance/qwen3.5-4b.sha256) records SHA-256 for the weights, weight index, config, tokenizer, tokenizer config, and chat template, from Hugging Face revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- qwenFS-derived compiler source: `48f4f45b6269bf8a0f27ea954a05b2caced929a6`; the repository documentation records upstream qwenFS baseline `fe3f5c0dbebe5c827781e98ad839aecebcce32fa`.

## Reproduce the ten archived live traces

Requirements: Apple Silicon with MPS, the local `models/Qwen3.5-2B` base model matching [`provenance/qwen3.5-2b.sha256`](provenance/qwen3.5-2b.sha256), `uv`, and roughly 10 GB free disk per compiled checkpoint. These are the command forms implemented by the repository's existing CLI modules. Historical records confirm greedy decoding, seed 0, a 256-token reasoning span, and the checkpoint settings shown below.

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

## Reproduce the Qwen3.5-4B sweep

With Qwen3.5-4B under `models/Qwen3.5-4B` matching [`provenance/qwen3.5-4b.sha256`](provenance/qwen3.5-4b.sha256), build the four checkpoints with `scripts/build_checkpoints.sh 4b` (16.4 GB each, 65.6 GB for all four) and run `qfs.compiled_sweep` as shown in the [4B results](docs/qwen35-4b-full-live-compiled-results.md#reproduction). On an Apple M5 the sweep takes about 4.6 hours at about 6 tokens per second.

## What is verified

Two frozen 32-item live-compiled sweeps on the v3 suite (four task families, two objectives per family at 2, 3, 5, and 8 cycles), one per base model, with the same runner, decoder, evaluator, token accountant, scorer, and settings: greedy decoding, seed 0, a 256-token reasoning span, and a 7,000-token external bound. The plain rows are one thinking-enabled turn of the same base model with a length-matched token budget, from two versions of the plain-arm runner (2B `aa833883…`; 4B `8c1e55f6…`, which reproduces the 2B plain records 32/32).

| | Qwen3.5-2B | Qwen3.5-4B |
| --- | ---: | ---: |
| Forced-token oracle at 2, 3, 5, and 8 cycles | 840, 1,132, 1,716, 2,592: all exact | the same totals, all exact |
| Runs passing every system check (HALT, controller, qwenFS reads, cycles, accounting) | 30/32 | 32/32 |
| Exact chains, compiled checkpoint | 8/32 (Wilson 95% [13.3, 42.1]) | 28/32 (Wilson 95% [71.9, 95.0]) |
| Exact chains, plain Qwen | 5/32 | 15/32 |
| Paired compiled minus plain (item bootstrap 95% CI; exact McNemar) | +9.4 points [0.0, +21.9]; p = 0.25 | +40.6 points [+25.0, +59.4]; p = 0.00024 |

- **Qwen3.5-2B.** All eight sequential arithmetic chains pass and the other 24 fail; outside arithmetic, no 2B ledger matches its task even under the exploratory operation-signature score. In two runs, tool-call markup inside Qwen's reasoning switched qwenFS into write mode, a parser behavior inherited from qwenFS, so those runs' file reads were not exact. The paired result against plain chat is inconclusive.
- **Qwen3.5-4B.** 8/8 arithmetic, 8/8 structured-operation, 6/8 checkable-language, and 6/8 ordered-literal chains. All four failures are Qwen's exact-string edits (casing and punctuation placement) at five and eight cycles, inside runs that pass every system check. The improvement over plain Qwen3.5-4B was preregistered and holds with no item regressed. Every plain 4B run reached its token budget, and 10 of the 13 items the compiled loop gains are ones where plain Qwen gave no parseable answer, so much of the gain is plain Qwen running out of budget before it answers. Agent-inactive parity against stock Qwen3.5-4B: maximum KL 3.8e-5 nats over four prompts, top-1 4/4. Details: [preregistration](docs/qwen35-4b-full-live-compiled-preregistration.md) and [results](docs/qwen35-4b-full-live-compiled-results.md).
- The host provides nothing after the objective; the checkpoints run greedily until their compiled HALT token.
- Traces: the ten archived 2B diagnostics are in [`traces/`](traces/README.md), selected for diagnosis before the 2B sweep and not an accuracy sample. All 32 4B sweep runs are in [`traces/qwen3.5-4b/`](traces/qwen3.5-4b/).

## What remains exploratory

- Capability beyond this suite. The v3 suite is 32 short chains with deterministic, checkable answers; it does not test open-ended planning or tool use.
- Post-hoc operation-signature plan fidelity. Frozen exact task strings remain the primary plan metric.
- Runtime comparisons and the compiled circuit-oracle check on CUDA.
- Runtime-variable cycle count; the current controller uses one checkpoint per fixed cycle count.
- Broad agent-inactive parity. The current check is a four-prompt smoke test.
- A filesystem-only Qwen3.5-4B model (`python -m qfs.merge`) and mounting saved files into a 4,096-wide agent checkpoint (`qfs.disk.mount`) are not supported.

## Credits and licenses

- [qwenFS](https://github.com/vixhal-baraiya/qwenFS) by Vixhal Baraiya supplies the compiled filesystem foundation and is MIT-licensed.
- The original filesystem demo video remains available in the upstream [qwenFS repository](https://github.com/vixhal-baraiya/qwenFS); it is not duplicated here.
- [Qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal is credited for compiler lineage.
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) and [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) are provided by the Qwen team under Apache License 2.0.
- BabyAGI was created by Yohei Nakajima.

Repository code is MIT-licensed; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Qwen-derived checkpoint weights are separately governed by Qwen's Apache-2.0 terms and are not relicensed by this repository's MIT license.
