# Compiled BabyAGI proof of concept

**The model-side control loop passed. The semantic agent test failed.**

> M5 follow-up: Qwen3.5-2B with the new objective-agnostic `literal` protocol
> passed the same three-step semantic objective in one seed-0 run. A separate
> `generative` protocol produced cumulative story text but still failed exact
> `Input` state reporting. See [m5-babyagi-followup.md](m5-babyagi-followup.md).
> The remainder of this document records the original 0.8B experiment.

A Qwen3.5-0.8B checkpoint was augmented using qwenFS's existing constructed
layers. From one initial objective it completed three create/prioritize/execute
cycles and emitted its own HALT. The host only supplied the initial chat prefill,
sampled tokens, updated ordinary inference caches, and saved the trace.

The tested objective was to start with `oak`, append `-red`, then `-blue`, then
`-gold`, one operation per task. The prototype instead generated unrelated
product-research tasks, copied multiple tasks during prioritization, and claimed
results without performing the requested string operations. None of the three
expected execution results passed the objective-specific checker. This is a
control-placement proof, not a working general autonomous agent.

## Recorded results

| Check | Result |
| --- | --- |
| Single-objective continuous decoding | 3 cycles; model-generated HALT |
| Logical phases | 9: create, prioritize, execute, repeated three times |
| Compiled microstates / file operations | 17 |
| Semantic file writes | 9 |
| Exact read responses | 8/8, token-for-token against stored values |
| Compiled ROM, response and transition test | 1,476/1,476 forced tokens correct |
| Generated tokens / inference time | 2,287 / 469.6 seconds on MPS |
| Objective correctness | FAIL, 0/3 expected append results |
| Native-model numerical smoke comparison | Matching top prediction at 4/4 positions; maximum KL 0.00536 |

The numerical comparison is small and does not establish full behavioral
parity. One full autonomous run is not a reliability benchmark. An exploratory
greedy run without the tool schema also drifted; it was stopped after the
behavioral failure. Sampling and prompt changes were not isolated causally.

Raw tokens, phase contents, checks, environment versions, and the compiled ROM
are retained under `experiments/minimal-agent/`.

## What lives in the weights

`qfs/agent.py` fills unused capacity in qwenFS's existing eight prefix layers:

- A Gated DeltaNet register holds the microstate; another head holds a decaying
  instruction pointer.
- MLP weights encode the static phase instructions, complete file-read calls,
  and the opening of each write call.
- Pretrained Qwen generates task descriptions, a priority choice, execution
  text, and XML closing tags. Those semantic decisions were not learned anew.
- The existing qwenFS circuit performs lookups, writes, exact copies, and reply
  framing. Completing a filesystem response triggers the next microstate.
- The last response triggers HALT. The host never counts phases or restarts a
  request. Temporary parser flags reset at boundaries; file contents persist.

The task objective and generated content are not baked into the program. The
three files are `tasks.txt`, `current.txt`, and `result.txt`. Each cycle
regenerates two candidates; a complete persistent backlog is outside this POC.

The controller uses the 0.8B model's unused 1024..2047 residual band. qwenFS's
filesystem occupies 2048..3071. Scratch is cleared before Qwen's learned layers.
The merge scales norms for the actual source width. The original 2B checkpoint
was not tested; it needs a different scratch layout. No training or additional
transformer layers beyond qwenFS's existing prefix were used.

## Host boundary

`qfs/agent_decode.py` formats one initial objective with the normal Qwen tool
schema and checkpoint-defined BOS. Every subsequent model input is exactly the
last sampled token. It has no phase parser, scheduler, subsequent prompt
composer, tool executor, logits override, or recovery callback. The initial tool
schema describes internal filesystem operations; the host never executes them.

The host recognizes checkpoint-defined EOS/HALT and an external token budget.
qwenFS's cache-mask hook supports its attention memory. No disk paging/mounting
is used. The decoder never reads the ROM in `agent.json`. Post-hoc evaluation
modules never participate in generation.

## Reproduce

Tested on a 16 GB Apple-silicon iMac, Python 3.11.15, torch 2.15.0,
transformers 5.22.1, MPS/bf16. Approximately 6 GB is needed for compiled weights.
The upstream starting commit is `fe3f5c0dbebe5c827781e98ad839aecebcce32fa`.

```sh
uv sync
uv run hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir models/Qwen3.5-0.8B
uv run python -m qfs.agent models/Qwen3.5-0.8B build/babyagi --cycles 3
uv run python -m tests.test_agent_circuit build/babyagi
uv run python -m qfs.agent_decode build/babyagi \
  'Create a growing label. Begin with oak. First append -red; next append -blue; finally append -gold. Carry out exactly one append operation per task in that order, using the previous result as the next input. Each result must show Input and Output. Do not repeat a finished operation.' \
  --output results/run.json --seed 0 --temperature 0.7 --limit 4000
uv run python -m qfs.agent_evaluate results/run.json --output results/evaluation.json
uv run python -m tests.test_agent results/run.json
# This last command is expected to report the semantic failure for the recorded run:
uv run python -m tests.test_agent_append_goal results/run.json
```

For an archived trace on another path, pass `--checkpoint build/babyagi` to the
evaluator or structural checker. CPU/CUDA require separate validation.

The tested local checkpoint is
`/Users/yoheinakajima/Developer/Untapped/qwenfs-babyagi-poc/build/babyagi-0.8b-io`.

## Goal-refresh experiment

An additional 21-state variant stored the objective as an initial model-side
file write and explicitly read it before task creation. Its 1,707 forced-token
checks passed, but the live model copied the objective and instructions into
its task list. That failed behavioral run was stopped early and is not counted
as a completed autonomous test. Its log, ROM, numerical check, and a source patch
are in `experiments/goal-refresh/`. The local source snapshot is commit
`1508fc4`; the primary implementation retains the smaller verified controller.

## Limits and conclusion

The experiment establishes that qwenFS-style weights can sequence a bounded
BabyAGI workflow and perform its internal file operations while the host only
decodes. Useful task creation, valid prioritization, truthful execution, dynamic
completion detection, and long-run stability remain unproven or failed here.

The fixed three-cycle program inherits qwenFS's file, version, and text bounds.
It does not provide indefinite operation, external actions, durable memory,
semantic validation, or evidence of an advantage over a Python scaffold. A
stronger model or protocol training is a next experiment, not a demonstrated fix.
