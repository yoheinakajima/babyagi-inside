# M5 BabyAGI prompt and 2B follow-up

## Result

Qwen3.5-2B is sufficient for the first semantic BabyAGI success in this
repository. A static, objective-agnostic `literal` protocol completed the
three-step label objective:

```text
Input: oak
Output: oak-red

Input: oak-red
Output: oak-red-blue

Input: oak-red-blue
Output: oak-red-blue-gold
```

The run used seed 0, temperature 0.7, top-p 0.9, and the same objective as the
failed baseline. It halted after 2,487 tokens and 247.2 seconds on the M5. The
post-hoc checks passed all three expected results, all three controller cycles,
all 8 exact filesystem reads, all 9 semantic writes, every compiled ROM segment,
and model-generated HALT. This is one successful run, not a reliability rate.
Sampling was deliberately held at the baseline decoder settings to isolate the
prompt change; Qwen's current model card recommends a different non-thinking
sampling recipe, which should be evaluated as a separate experiment.

The baseline and literal ROMs remain separate selectable protocols. The
baseline ROM is byte-identical to the pre-follow-up checkpoint. No runtime
objective values or expected answers occur in the literal prompt.

## Original BabyAGI homage

The optional `babyagi` protocol restores the original bootstrap idea: before
ordinary task generation, the initial meta-task is **Create the first task**.
The ROM opens `tasks.txt` with the static prefix `FIRST TASK:` and Qwen completes
one concrete executable task from the objective. Later creation phases mark the
bootstrap task complete so it cannot be selected again.

This variant passed the same label objective at seed 0:

- cycle 1 bootstrap: `FIRST TASK: 1. Append -red to 'oak'`;
- semantic objective: 3/3 exact `Input` and `Output` pairs;
- controller: PASS, three cycles, 8 exact reads, 9 writes, and model HALT;
- circuit oracle: 2,263/2,263 forced tokens correct; and
- live decode: 2,578 tokens in 256.0 seconds.

The homage needed the compiled `FIRST TASK:` prefix. Without it, 2B expanded
the bootstrap into the entire task list and then repeated the completed first
operation. The successful version also explicitly retires `FIRST TASK` after
cycle 1. Task-list discipline is still imperfect: prioritization sometimes
copied both candidates, and an unused second candidate in cycle 2 had the wrong
intermediate state. Execution nevertheless selected the correct first operation
in all three cycles. This remains a one-seed proof, not a reliability claim.

The classic BabyAGI examples used an initial task such as `Develop a task list`,
which was executed before the regular task-creation loop. The compiled homage
keeps that spirit while using the more direct label `Create the first task`.

## Why the prompt worked

The failed baseline asked for two generic “next tasks.” The 2B model interpreted
`oak` as a brand and invented product work. Direct native-model probing showed
that the same 2B weights could return the exact three-step label chain when the
output contract was explicit. The successful compiled prompt therefore:

- says the first user message must be followed literally;
- treats strings, symbols, numbers, order, and requested formats as requirements;
- forbids reinterpretation as a business, brand, or fictional project;
- makes Task 1 the earliest unfinished explicit operation;
- requires prioritization to copy only one task; and
- gives execution an exact two-line `Input` / `Output` contract.

This follows Qwen's own recommendation to standardize output formats and its
warning that complex tool protocols which rely on the model to stay on track
are less reliable than concise, controlled templates:

- <https://huggingface.co/Qwen/Qwen3.5-2B>
- <https://qwen.readthedocs.io/en/latest/framework/function_call.html>

The live run did not perfectly obey task-list formatting: cycle 1 listed three
candidates and cycle 3 listed one. Prioritization still isolated the correct
single operation, and execution was exact. Task cardinality should be measured
separately from objective completion.

## Generative text experiment

A separate `generative` protocol adds a `result.txt` reread immediately before
execution in cycles 2 and 3. This keeps the accumulated text recent enough for
2B to extend it instead of replacing it.

On a three-sentence fox story objective, the compiled model generated:

```text
A fox named Mira sits beside a frozen lake.
A silver key is found in Mira's pocket.
The ice sang.
```

The final stored `Output` contained all three sentences in order. The controller
also passed: three cycles, 10 exact reads, 9 writes, 3,129 tokens, model-generated
HALT, and 312.5 seconds of inference. This is only a partial semantic result.
The model incorrectly reused the first task description in every later `Input`
field even though it correctly retained and extended the previous `Output`.

Longer prompt variants fixed one failure while regressing another. In
particular, carrying state in every task caused the literal benchmark to repeat
the completed `-red` operation. The source therefore keeps `literal` as the
proven transformation protocol and labels `generative` experimental instead of
claiming one universal prompt.

## DSPy assessment

DSPy is applicable as an offline prompt-search harness, but it should not be in
the compiled runtime yet. DSPy optimizers require a program, a metric, and a
small training set; instruction optimizers such as MIPROv2 and reflective
optimizers such as GEPA then evaluate multiple prompt candidates. DSPy also
supports local models through providers such as Ollama.

The current target is unusual: phase prompts are compiled into a ROM, a full
candidate build takes roughly half a minute, and a single three-cycle M5 rollout
takes roughly four to five minutes. Optimizing directly through the compiled
checkpoint would be slow and would overfit badly to one objective. A useful DSPy
experiment first needs:

1. a fast native-model simulator for the create/prioritize/execute transcript;
2. a held-out benchmark of at least literal transformations, ordered arithmetic,
   structured extraction, and cumulative text generation;
3. decomposed metrics for task selection, state continuity, output truthfulness,
   and format compliance; and
4. final validation of winning prompts in the compiled loop across multiple
   seeds.

The official optimizer guidance recommends starting with few-shot optimization
at around ten examples and using instruction-only MIPROv2 for zero-shot prompt
optimization. GEPA can use textual failure feedback, which fits the distinct
failure categories observed here:

- <https://dspy.ai/learn/optimization/optimizers/>
- <https://arxiv.org/abs/2406.11695>
- <https://arxiv.org/abs/2507.19457>

No DSPy dependency was added in this follow-up.

## Why not move to 4B yet

The official Qwen3.5-4B configuration has a 2,560-wide text residual. qwenFS
currently owns dimensions 2,048 through 3,071. A 4B merge would therefore
overlap 512 native dimensions with the filesystem circuit. Supporting 4B
requires relocating or parameterizing qwenFS's circuit dimensions and then
revalidating every exact register, norm scale, cache mask, native-logit parity,
and forced token. It is not a model-path substitution.

The 2B prompt success means there is no current evidence that capacity is the
main blocker. Stay on 2B while building a multi-objective prompt benchmark. Move
to 4B only if held-out failures persist after state routing and prompt search,
or treat 4B relocation as a separate compiler project.

Official 4B configuration: <https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json>

## Reproduce the passing literal run

```sh
uv run python -m qfs.agent models/Qwen3.5-2B build/babyagi-2b-literal \
  --cycles 3 --protocol literal
uv run python -m tests.test_agent_circuit build/babyagi-2b-literal
uv run python -m qfs.agent_decode build/babyagi-2b-literal \
  'Create a growing label. Begin with oak. First append -red; next append -blue; finally append -gold. Carry out exactly one append operation per task in that order, using the previous result as the next input. Each result must show Input and Output. Do not repeat a finished operation.' \
  --output results/literal-oak.json --seed 0 --temperature 0.7 --limit 4000
uv run python -m tests.test_agent results/literal-oak.json
uv run python -m tests.test_agent_append_goal results/literal-oak.json
```

Use `--protocol generative` only for the experimental state-refresh variant.
Use `--protocol babyagi` for the successful initial-task homage.
