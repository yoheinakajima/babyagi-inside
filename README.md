# BabyAGI Inside

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23966217.svg)](https://doi.org/10.5281/zenodo.23966217)

BabyAGI Inside compiles a bounded BabyAGI-style agent loop into the weights of a widened Qwen3.5-4B checkpoint, with no training: the host supplies one objective and then only samples until HALT, while compiled controller state advances a fixed task pointer, qwenFS persists the ledger and results, and Qwen writes the plan, reasoning, and payloads. The result is a systems demonstration—not a claim of general agency.

On the frozen 32-task v3 suite, all 32 live Qwen3.5-4B runs pass every system check (controller sequence, qwenFS reads, HALT, and token accounting), and the compiled loop completes 28 of the 32 objectives. Plain Qwen3.5-4B, given the same token budget and made to answer, completes 29, with less than half the tokens and about a quarter of the time: on this suite the loop adds exact control, not measured capability ([budget-forced results](docs/plain-forced-v3-results.md)). On 48 longer held-out tasks (4 to 32 steps), a preregistered screen of an adaptive version of the loop, run as host code, completes 20, or 25 when stopped at the true length, against plain Qwen3.5-4B's 38 at the same budget ([longer-task screen](docs/long-tasks-results.md)).

The first release compiled the same loop into Qwen3.5-2B, and the archived preprint at the DOI above reports it. The current code still builds its checkpoints byte-identically, and the commands to rebuild them and replay its ten archived live traces are [below](#reproduce-the-qwen35-2b-release). Its plain comparison is superseded: the one-turn plain arm stopped every run before Qwen closed its thinking, and at the loop's token budget plain Qwen3.5-2B completes 12 chains to the loop's 8.

Repository: <https://github.com/yoheinakajima/babyagi-inside>

## Results

Two frozen 32-item live-compiled sweeps on the v3 suite (four task families, two objectives per family at 2, 3, 5, and 8 cycles), one per base model, with the same runner, decoder, evaluator, token accountant, scorer, and settings: greedy decoding, seed 0, a 256-token reasoning span, and a 7,000-token external bound. The primary plain rows are budget-forced: plain Qwen's thinking is closed at a cap and it then answers under its own cap, at the compiled loop's model-token allowance (F2, preregistered; [results](docs/plain-forced-v3-results.md)). The one-turn rows are a single thinking-enabled turn with a length-matched ceiling, from two versions of the plain-arm runner (4B `8c1e55f6…`, which reproduces the 2B plain records 32/32; 2B `aa833883…`); every one-turn run reached its ceiling before Qwen closed its thinking, so those rows measure truncation.

| | Qwen3.5-4B | Qwen3.5-2B |
| --- | ---: | ---: |
| Forced-token oracle at 2, 3, 5, and 8 cycles | 840, 1,132, 1,716, 2,592: all exact | the same totals, all exact |
| Runs passing every system check (HALT, controller, qwenFS reads, cycles, accounting) | 32/32 | 30/32 |
| Exact chains, compiled checkpoint | 28/32 (Wilson 95% [71.9, 95.0]) | 8/32 (Wilson 95% [13.3, 42.1]) |
| Exact chains, plain Qwen at the loop's token budget (F2) | 29/32 | 12/32 |
| Paired compiled minus F2 plain (item bootstrap 95% CI; exact McNemar) | −3.1 points [−9.4, 0.0]; p = 1.0 | −12.5 points [−25.0, −3.1]; p = 0.125 |
| Tokens generated over the suite, compiled / F2 plain | 100,545 / 46,800 | 104,586 / 52,256 |
| Wall time on one Apple M5, compiled / F2 plain | 4.6 h / 1.2 h | about 3.5 h / 0.6 h |
| Exact chains, plain Qwen in one turn (truncated before answering) | 15/32 | 5/32 |
| Paired compiled minus one-turn plain | +40.6 points [+25.0, +59.4]; p = 0.00024 | +9.4 points [0.0, +21.9]; p = 0.25 |

- **Qwen3.5-4B.** 8/8 arithmetic, 8/8 structured-operation, 6/8 checkable-language, and 6/8 ordered-literal chains. All four failures are Qwen's exact-string edits (casing and punctuation placement) at five and eight cycles, inside runs that pass every system check. Budget-forced plain Qwen3.5-4B completes 29 under each of three preregistered budgets, reaching the compiled loop's verdict on 31 items; the loop wins no item plain Qwen loses. The earlier +40.6-point gain over the one-turn arm was preregistered, but every one-turn run stopped before Qwen closed its thinking, so it measured truncation, not capability. Of the compiled loop's tokens, 58,972 are forced by the circuit and 41,573 are written by Qwen; the widened checkpoint decodes at about 6 tokens per second. Agent-inactive parity against stock Qwen3.5-4B: maximum KL 3.8e-5 nats over four prompts, top-1 4/4. Details: [preregistration](docs/qwen35-4b-full-live-compiled-preregistration.md) and [results](docs/qwen35-4b-full-live-compiled-results.md).
- **Longer tasks (host-run screen, Qwen3.5-4B).** On 48 held-out tasks of 4 to 32 steps (long listed chains, rules applied until a condition holds, and worklists), a host-run adaptive loop that writes its own ledger and decides after each step whether to stop completes 20/48; stopped by the host at the true length it completes 25/48; budget-forced plain Qwen at the loop's token allowance completes 38/48. Loop minus plain is −37.5 points [−52.1, −22.9], and with the stop given −27.1 [−39.6, −14.6]; the loop wins no task plain Qwen loses, and the preregistered decision is NO-GO. Most of the gap is execution, not stopping: on worklists plain completes 16/16 and the loop 5/16, usually by dropping a queued job. This is a host simulation, not a compiled checkpoint. Details: [preregistration](docs/long-tasks-preregistration.md) and [results](docs/long-tasks-results.md).
- **Qwen3.5-2B.** All eight sequential arithmetic chains pass and the other 24 fail; outside arithmetic, no 2B ledger matches its task even under the exploratory operation-signature score. In two runs, tool-call markup inside Qwen's reasoning switched qwenFS into write mode, a parser behavior inherited from qwenFS, so those runs' file reads were not exact. At the loop's token budget plain Qwen3.5-2B completes 12 chains to the loop's 8, winning four items and losing none: plain is ahead by the preregistered interval reading, though the exact test is not significant.
- The host provides nothing after the objective; the checkpoints run greedily until their compiled HALT token.
- Traces: all 32 4B sweep runs are in [`traces/qwen3.5-4b/`](traces/qwen3.5-4b/). The ten archived 2B diagnostics are in [`traces/`](traces/README.md), selected for diagnosis before the 2B sweep and not an accuracy sample.

## The host loop

```python
ids = prefill(objective) + [START]
while ids[-1] != HALT:
    ids.append(argmax(model(ids)))
```

The host does not parse phases, execute tools, inject follow-up prompts, repair outputs, or manage a task list after the initial objective.

## Checkpoint layout

The widened residual stream is 4,096 dimensions. Qwen3.5-4B's native residual is 2,560 dimensions wide; qwenFS keeps the slot it has in the 2B release, so Qwen's state continues above it, and the agent sits above that:

| Dimensions | Owner | Purpose |
| ---: | --- | --- |
| 0–2047 | Qwen | Native Qwen3.5-4B model state, first 2,048 dimensions |
| 2048–3071 | qwenFS | Compiled filesystem state and operations |
| 3072–3583 | Qwen | Native Qwen3.5-4B model state, remaining 512 dimensions |
| 3584–4059 | Agent | Controller and bounded-agent scratch state (3584–4023 at 2 cycles, 3584–4059 at 8) |

Qwen3.5-4B's DeltaNet layers have 32 value heads over 16 key heads, so compiled qwenFS head h writes value head 2h and the other value heads stay zero.

The Qwen3.5-2B checkpoints use the same 4,096-dimension residual, with Qwen's 2,048 native dimensions below qwenFS:

| Dimensions | Owner | Purpose |
| ---: | --- | --- |
| 0–2047 | Qwen | Native Qwen3.5-2B model state |
| 2048–3071 | qwenFS | Compiled filesystem state and operations |
| 3072–3547 | Agent | Controller and bounded-agent scratch state (3072–3511 at 2 cycles, 3072–3547 at 8) |

Qwen3.5-2B has 16 value heads and 16 key heads, and its checkpoints are byte-identical to the release builds.

Cycle count is fixed at checkpoint construction. The verified sets use separate checkpoints for 2, 3, 5, and 8 cycles.

## Rebuild checkpoints locally

Checkpoint weights are rebuilt locally and are not shipped in this Git repository. After placing Qwen3.5-4B under `models/Qwen3.5-4B`, or Qwen3.5-2B under `models/Qwen3.5-2B`, run:

```sh
scripts/build_checkpoints.sh 4b   # Qwen3.5-4B
scripts/build_checkpoints.sh      # Qwen3.5-2B
```

The script invokes the existing `qfs.agent` build command for `babyagi_ledger_pointer`, 256 reasoning tokens, and cycle counts 2, 3, 5, and 8. Each resulting directory is approximately 15 GB for 4B and 8.8 GB for 2B (`du -sh`).

- Build time per checkpoint on an Apple M5 (32 GB): 40–58 s for 4B and 31–36 s for 2B.
- Peak memory: macOS reports a peak memory footprint of 25.3–26.4 GB for a 4B build and 18.9 GB for 2B (one 5-cycle build measured); maximum resident set size is lower, 12.6–15.0 GB for 4B and 13.8–16.1 GB for 2B. Builds allocate the widened model directly in bf16. Before that change a 2B build peaked at a 23–24 GB footprint (max RSS 11–12 GB), and the fp32 allocation of the widened 4B model alone would be about 33 GB (an estimate; it was never run).
- Builds are deterministic: rebuilding from the same inputs reproduces each checkpoint's tree SHA-256. The digests are in the [4B](docs/qwen35-4b-full-live-compiled-preregistration.md) and [2B](docs/babyagi-v3-full-live-compiled-preregistration.md) sweep preregistrations.
- Qwen3.5-4B content pin: [`provenance/qwen3.5-4b.sha256`](provenance/qwen3.5-4b.sha256) records SHA-256 for the weights, weight index, config, tokenizer, tokenizer config, and chat template, from Hugging Face revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Qwen3.5-2B content pin: [`provenance/qwen3.5-2b.sha256`](provenance/qwen3.5-2b.sha256) records SHA-256 for the exact weights, config, tokenizer, and tokenizer config used locally. The upstream revision ID was not recorded.
- qwenFS-derived compiler source: `48f4f45b6269bf8a0f27ea954a05b2caced929a6`; the repository documentation records upstream qwenFS baseline `fe3f5c0dbebe5c827781e98ad839aecebcce32fa`.

## Reproduce the Qwen3.5-4B results

Requirements: Apple Silicon with MPS, Qwen3.5-4B under `models/Qwen3.5-4B` matching [`provenance/qwen3.5-4b.sha256`](provenance/qwen3.5-4b.sha256), `uv`, and 65.6 GB free disk for the four checkpoints (16.4 GB each).

The compiled sweep, about 4.6 hours on an Apple M5 at about 6 tokens per second. `--plain-root` names the one-turn plain records the sweep pairs against; see the [4B results](docs/qwen35-4b-full-live-compiled-results.md#reproduction).

```sh
scripts/build_checkpoints.sh 4b
uv run python -m qfs.compiled_sweep benchmarks/babyagi-v3.json \
  results/qwen35-4b-full-live-compiled \
  --checkpoint-pattern 'build/babyagi-4b-ledger-pointer-b256-m{cycles}' \
  --plain-root PLAIN_4B_ROOT --device mps --seed 0 --temperature 0 --limit 7000
```

The budget-forced plain arm, on stock weights, and the paired comparison:

```sh
uv run python -m qfs.plain_forced run models/Qwen3.5-4B benchmarks/babyagi-v3.json \
  results/plain-forced-v3/compiled.json 4b F2 /tmp/plain-forced-4b-F2
uv run python -m qfs.plain_forced compare results/plain-forced-v3/compiled.json \
  results/plain-forced-v3
```

Re-checking the committed records needs no model. These re-score every budget-forced and longer-task record against its frozen task and recompute each comparison above on CPU:

```sh
uv run python -m tests.test_plain_forced
uv run python -m tests.test_long_tasks
```

The longer-task screen's runner, as run, is in [`results/long-tasks-4b/runner/`](results/long-tasks-4b/runner/); it imports the research lab's own packages, so it does not run from this repository.

## Reproduce the Qwen3.5-2B release

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

Then decode, inspect, account for, and score the ten archived objectives. Each invocation starts with a fresh qwenFS cache. The two structured items are expected to fail the exact semantic scorer; the command preserves those failures.

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

## What remains exploratory

- Capability beyond these suites. The v3 suite is 32 short chains with deterministic, checkable answers, and the longer-task screen is 48 generated checkable tasks run once each through a host-simulated adaptive loop; neither tests open-ended planning or tool use, and no compiled checkpoint has run at those lengths.
- Post-hoc operation-signature plan fidelity. Frozen exact task strings remain the primary plan metric.
- Runtime comparisons and the compiled circuit-oracle check on CUDA.
- Runtime-variable cycle count; the current controller uses one checkpoint per fixed cycle count.
- Broad agent-inactive parity. The current check is a four-prompt smoke test.
- A filesystem-only Qwen3.5-4B model (`python -m qfs.merge`) and mounting saved files into a 4,096-wide agent checkpoint (`qfs.disk.mount`) are not supported.

## Credits and licenses

- [qwenFS](https://github.com/vixhal-baraiya/qwenFS) by Vixhal Baraiya supplies the compiled filesystem foundation and is MIT-licensed.
- The original filesystem demo video remains available in the upstream [qwenFS repository](https://github.com/vixhal-baraiya/qwenFS); it is not duplicated here.
- [Qwenputer](https://github.com/divagr18/qwenputer) by Divyansh Agrawal is credited for compiler lineage.
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) are provided by the Qwen team under Apache License 2.0.
- BabyAGI was created by Yohei Nakajima.

Repository code is MIT-licensed; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Qwen-derived checkpoint weights are separately governed by Qwen's Apache-2.0 terms and are not relicensed by this repository's MIT license.
