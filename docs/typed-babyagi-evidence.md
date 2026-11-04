# Compiling a bounded BabyAGI loop into Qwen weights

## Claim boundary

The defensible claim is narrow but real:

> A five-cycle create–prioritize–execute controller, persistent files, and a
> constrained file-selection operation were merged into a Qwen3.5-2B
> checkpoint. After one objective prefill and a BOS-like `START` token, the
> inference host only samples the checkpoint until its `HALT` token.

This is not evidence that Qwen3.5-2B is a generally capable BabyAGI. The frozen
held-out benchmark is a clear negative result. The contribution is a systems
artifact: bounded agent control flow and filesystem operations execute inside
the merged model, while the pretrained model supplies the semantic content and
one-token choices.

## Architecture

The `babyagi_typed` protocol uses 24 compiled microstates for five cycles.

- Cycle 1: create `task1.txt`; create `task2.txt`; choose-and-read a task;
  execute it into `result.txt`.
- Cycles 2–5: read `result.txt`, then repeat those four stages.
- Task 1 is executable work. Task 2 is a verification candidate. This avoids a
  2B failure observed during development: when Task 2 described future work,
  later cycles often treated the merely generated candidate as completed.
- The first instruction is the original BabyAGI homage: “Create the first
  task.”

Prioritization is not an ordinary free-text tool call. The ROM opens
`read_file(task`, Qwen owns exactly one token selected from `1` or `2`, and the
ROM closes `.txt)` and the XML tool wrapper. qwenFS therefore returns the
selected file immediately. The host has no logits processor, task parser,
scheduler, tool executor, or recovery callback.

The typed gate uses one added residual coordinate. Tokens `1` and `2` have a
positive value in that coordinate; every other vocabulary item has zero. At
the decision offset, the compiled layer-7 ROM activates the coordinate and the
final norm amplifies it. This preserves Qwen's relative logits between the two
legal choices while making every other token lose the top-logit competition.

## Evidence snapshot

All results below use greedy decoding unless stated otherwise.

| Test | Result |
| --- | --- |
| Live compiled five-cycle development trace | PASS: `pine-red`, `pine-red-blue`, `pine-red-blue-gold`, `pine-red-blue-gold-green`, `pine-red-blue-gold-green-silver` |
| Controller and qwenFS inspection | PASS: 5/5 complete cycles, 15/15 expected writes, token-exact reads, compiled HALT |
| Scripted checkpoint circuit oracle | PASS: 4,788/4,788 forced tokens exact |
| Typed-choice circuit margin | PASS at five positions; minimum margin 1,893.125 logits over the best illegal token |
| Agent-inactive native parity | PASS: 4/4 top-token agreement; max native-to-compiled KL 0.001896 |
| Exact provenance partition | PASS: all 5,010 emitted tokens accounted for |
| Provenance | 4,769 forced tokens (95.19%); 241 model tokens (4.81%) |
| Semantic ownership | 15 free-text model writes and five typed model choices |
| Frozen held-out v1, native semantic harness | FAIL: 0/12 full items; 1/33 exact cycle checks |
| Counterbalanced native selector probe | FAIL: 1/2; Qwen chose Task 1 in both orderings |

The live trace took 481.1 seconds on the test Mac and emitted 5,010 tokens. The
native harness is used only for cheap semantic screening. It explicitly does
not prove that the compiled controller or qwenFS ran; those claims come from
the live compiled trace and its independent inspector.

## What the positive trace establishes

The live trace establishes more than prompt replay:

1. The objective is supplied only in the initial chat prefill.
2. The checkpoint emits its own phase prompts, tool calls, filesystem
   responses, and phase-control tokens.
3. Qwen generates both candidate contents and execution results.
4. At each priority point Qwen supplies a legal choice token, which becomes
   part of the filename of the read call.
5. qwenFS returns the chosen task content and carries each result into the next
   cycle.
6. The checkpoint ends with its compiled `HALT` token after cycle 5.

## What it does not establish

- The selector circuit guarantees a valid choice, not a correct one. The
  counterbalanced probe exposes strong Task-1 positional preference.
- The five-cycle append trace is a development demonstration, not a held-out
  generalization result.
- Qwen3.5-2B did not reliably handle sequential arithmetic, structured data,
  or constrained language generation in the frozen suite.
- The prompt contains a few format-only examples. Development showed that
  concrete examples can improve first-task formation but can also leak into
  later cycles. The verification-candidate design mitigated one failure mode;
  it did not solve general example contamination.
- A vector store is absent. The bounded experiment needs only the objective,
  three persistent files, and five cycles.

These limitations should remain in any public release. “BabyAGI is baked into
Qwen” is supportable only when immediately qualified as a bounded compiled
controller, not as a claim of general autonomous competence.

## Publication framing

A suitable working title is:

> **Compiling a Bounded BabyAGI Loop into Language-Model Weights**

A concise abstract claim is:

> We compile a bounded create–prioritize–execute loop and a persistent
> filesystem into a Qwen3.5-2B checkpoint. A one-objective inference host only
> samples tokens until a compiled halt symbol; it contains no scheduler, parser,
> tool executor, or logits processor. A five-cycle compiled trace passes
> token-exact controller and filesystem checks. We also introduce a typed
> in-weight selector that restricts a model-owned decision to two filenames.
> Frozen held-out evaluation and counterbalanced probes show that structural
> validity does not imply semantic competence: the 2B model fails all 12
> held-out items and exhibits positional selection bias. The artifact therefore
> separates compiled orchestration from pretrained-model capability.

This framing turns the negative results into part of the contribution: it
shows exactly which behavior is implemented by the circuit and which remains a
property of the pretrained model.

## Reproduce

Build and verify the checkpoint:

```bash
uv run python -m qfs.agent models/Qwen3.5-2B \
  build/babyagi-2b-typed-v17-m5 --cycles 5 --protocol babyagi_typed
uv run python -m tests.test_agent_circuit build/babyagi-2b-typed-v17-m5
uv run python -m tests.test_agent_parity \
  build/babyagi-2b-typed-v17-m5 models/Qwen3.5-2B
```

Run the live compiled trace:

```bash
uv run python -m qfs.agent_decode build/babyagi-2b-typed-v17-m5 \
  'Begin with pine; append -red, -blue, -gold, -green, and -silver, one operation per task.' \
  --output results/m5-2b-typed-v17/live-pine5-greedy.json \
  --temperature 0 --seed 0 --limit 7000
uv run python -m qfs.agent_evaluate \
  results/m5-2b-typed-v17/live-pine5-greedy.json \
  --output results/m5-2b-typed-v17/live-pine5-greedy.evaluation.json
uv run python -m qfs.agent_account \
  results/m5-2b-typed-v17/live-pine5-greedy.json \
  --output results/m5-2b-typed-v17/live-pine5-greedy.accounting.json
uv run python -m qfs.benchmark score-evaluation \
  benchmarks/babyagi-v1-dev.json dev-pine-label-5 \
  results/m5-2b-typed-v17/live-pine5-greedy.evaluation.json
```

Run the frozen native screen and counterbalanced selector probe:

```bash
uv run python -m qfs.agent_suite models/Qwen3.5-2B \
  benchmarks/babyagi-v1.json results/native-heldout/typed-v17-greedy \
  --protocol babyagi_typed --temperature 0 --seed 0
uv run python -m qfs.selector_probe models/Qwen3.5-2B \
  --output results/native-dev/typed-v17-selector-counterbalance.json
```

The frozen benchmark checksum is
`22891c9fe8b277dfc111c3c4feb766e62fc29853ed339eeb1b769ae4391c2f49`.
