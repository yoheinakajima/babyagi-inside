# Compiled BabyAGI: bounded proof of concept

One initial objective starts three cycles of task creation, prioritization, and
execution. Qwen3.5-0.8B supplies the semantic content. A constructed transformer
circuit supplies the instructions, file operations, and phase transitions. The
host runs ordinary autoregressive inference and samples tokens. No training is
performed.

## What is compiled

`qfs/agent.py` installs a small program in qwenFS's existing transformer layers:

- A Gated DeltaNet register holds one of 21 microstates for three cycles.
- Another head holds qwenFS's decaying instruction-pointer counter.
- An MLP ROM contains static instructions, chat framing, complete read calls,
  and the opening of each write call. Qwen generates the write's text value and
  its normal XML closing tags.
- Existing qwenFS layers compute file operations and response tokens.
- The completed filesystem response triggers the next microstate. The final
  execution response triggers HALT. There is no extra model decision about
  which operation to call next, nor a host decision about when to change phase.
- Control tokens reset transient filesystem parser flags. Files, versions,
  lookup rows, and copy memory persist across phase boundaries.

Each logical phase writes a value after its compiled reads:

| Phase | Compiled operations | Qwen's contribution |
| --- | --- | --- |
| Create | Read previous result after cycle 1, read goal.txt, write tasks.txt | Two candidate task descriptions |
| Prioritize | Read tasks.txt; write current.txt | Selected next task |
| Execute | Read current.txt; write result.txt | Actual task result |

The objective, generated tasks, selection, and results are NOT in the ROM.
The objective is stored in goal.txt and explicitly read before task creation.
`tasks.txt` holds two candidates; each cycle regenerates that bounded set. This
is not a full persistent backlog or a semantic deduplication system.

The circuit uses Qwen3.5-0.8B's unused residual band, dimensions 1024..2047.
The original qwenFS circuit occupies 2048..3071. Scratch features are cleared
before the pretrained language layers. Norm scaling in `qfs/merge.py` is
generalized to the actual source width (1024 for the 0.8B source model).

## Host boundary

`qfs/agent_decode.py` encodes the objective once as an initial goal-file write, after
the checkpoint-defined BOS. The model computes the write acknowledgment and
state update; the host does not execute that command. Every subsequent model input is exactly the last sampled token. It has no phase
parser, scheduler, dynamic prompt composer, tool executor, logits override, or
recovery loop. No tool schema is needed because file calls are compiled. The host recognizes
the checkpoint-defined EOS/HALT and enforces
an external experimental token budget.

qwenFS's attention-cache mask hook is part of the inference runtime; no disk
mounting/paging or external file operations execute the agent's tool calls.
Console rendering and trace persistence are host operations. The decoder never
reads `agent.json`; the copy of the ROM in that file is for independent,
post-hoc verification. Evaluation modules never participate in decoding.

## Reproduce on Apple silicon

Python 3.11+, uv, and room for approximately 6 GB of compiled weights:

```sh
uv sync
uv run hf download Qwen/Qwen3.5-0.8B --revision 2fc06364715b967f1860aea9cf38778875588b17 --local-dir models/Qwen3.5-0.8B
uv run python -m qfs.agent models/Qwen3.5-0.8B build/babyagi-0.8b --cycles 3
uv run python -m tests.test_agent_circuit build/babyagi-0.8b
uv run python -m qfs.agent_decode build/babyagi-0.8b \
  'Begin with oak. Append -red, then -blue, then -gold, in that order. Perform one append per task. Each result must show Input and Output. Do not repeat completed operations.' \
  --output results/append.json
uv run python -m qfs.agent_evaluate results/append.json --output results/append-evaluation.json
uv run python -m tests.test_agent results/append.json
uv run python -m tests.test_agent_append_goal results/append.json
```

The compiler supports bounded programs of 1..3 cycles, subject to the available
MLP ROM capacity. MPS/bf16 is the tested backend. CUDA/CPU can be selected with
`--device` but need separate validation. An unchanged 2B qwenFS checkpoint is not
compatible with this minimal spare-dimension layout.

## What the checks establish

The circuit test includes the real tool-schema prefill and independently
scripted text values. It checks every ROM token, file response, transition, and
HALT against reference expectations. That test isolates circuit correctness;
its scripted values are not counted as autonomous work.

The separate end-to-end run uses native Qwen generation throughout. Acceptance requires
all 21 microstates in order, exact ROM output, all nine semantic writes plus the initial goal acknowledgment, eleven
exact reads (including three goal reads), and bounded HALT. Read responses are
compared token-for-token with the preceding stored values. Task correctness is assessed separately against
the objective and shown in the experiment report.

This is a fixed three-cycle program with pretrained semantic judgment. It does
not establish indefinite operation, learned autonomy, reliable self-evaluation,
external execution, permanent memory, or improvement over a Python scaffold.
It inherits qwenFS's finite file/version/text limits. The initial 0.8B approach
that asked Qwen to choose its own tool calls failed; exploratory logs are kept
with the results. The final design compiles those calls directly.
