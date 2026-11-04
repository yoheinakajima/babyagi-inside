# BabyAGI held-out benchmark v1

`benchmarks/babyagi-v1.json` is the frozen publication suite. Its SHA-256 is
stored in `benchmarks/babyagi-v1.sha256`; the validator refuses a frozen suite
whose bytes no longer match that sidecar. Any substantive change creates a new
benchmark ID and file instead of updating v1 in place.

The suite contains 12 objectives, three in each category:

- ordered literal transformations;
- sequential arithmetic;
- sorting, filtering, and structured operations; and
- language tasks with automated constraints plus explicit manual criteria.

Previously used oak, pine, prompt-exemplar, and Luma objectives live in the
separate mutable `babyagi-v1-dev.json`. They are never held-out results. Frozen
sentinel phrases are tested against `qfs/agent.py` so an accidental prompt leak
fails cheaply. The benchmark file is evaluation data and is not imported by the
compiler or decoder.

## Scoring contract

Exact tasks provide one expected result per cycle. Text comparison strips only
outer whitespace, numeric comparison uses decimal value, and JSON comparison
parses the structure rather than comparing formatting.

Language tasks report automated and manual results separately. A run with
unscored manual criteria has `pass: null`, never an implicit pass. Publication
tables should report both columns and preserve the manual criterion text.

Primary evaluation should use deterministic greedy decoding. Sampled seeds are
a separate robustness experiment. Report at least:

- task creation correctness;
- typed choice correctness;
- completed-task repetition;
- state continuity;
- execution versus task-text copying;
- final objective correctness;
- controller/read/write/HALT integrity; and
- forced ROM, forced filesystem, typed-choice, and free semantic tokens.

## Commands

```sh
uv run python -m qfs.benchmark validate benchmarks/babyagi-v1.json
uv run python -m tests.test_benchmark
```

To score one item, save a JSON array with one result per cycle:

```sh
uv run python -m qfs.benchmark score benchmarks/babyagi-v1.json \
  arithmetic-seven-3 results.json
```

An inspected compiled trace can be scored directly; the scorer removes only a
leading `Output:` result-file envelope:

```sh
uv run python -m qfs.benchmark score-evaluation \
  benchmarks/babyagi-v1-dev.json dev-pine-label-5 \
  results/m5-2b-typed-v17/live-pine5-greedy.evaluation.json
```

The stock-weight semantic screen loads Qwen once, saves every full native
trace, and writes a checksum-bound summary. It is a prompt-development and
held-out semantic experiment, not proof that the compiled controller ran:

```sh
uv run python -m qfs.agent_suite models/Qwen3.5-2B \
  benchmarks/babyagi-v1.json results/native-heldout/typed-v17-greedy \
  --protocol babyagi_typed --temperature 0 --seed 0
```

Constraint-based language items exit nonzero until every manual criterion has
an explicit boolean score supplied with `--manual`.
