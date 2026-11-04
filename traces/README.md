# Public trace schema

These ten files are CPU-only exports of completed live compiled-checkpoint runs.
They are the diagnostic traces reported before the full 32-item sweep: eight
exact arithmetic chains and two structured-operation failures. They are not a
representative accuracy sample.

Each file has this schema:

```json
{
  "trace_id": "benchmark item id",
  "cycles": 5,
  "objective": "the checkpoint's only host-supplied objective",
  "ledger": {},
  "results": [],
  "expected": [],
  "pass": true,
  "tokens": {
    "total": 0,
    "forced": 0,
    "qwenfs": 0,
    "qwen_reasoning": 0,
    "qwen_payload": 0
  },
  "stages": [
    {"cycle": 1, "phase": "create", "emitter": "mixed:controller+qwen+qwenFS",
     "start": null, "end": null}
  ]
}
```

Token fields use the frozen accountant's disjoint ownership totals:

- `forced` is all ROM, qwenFS, and controller-control tokens;
- `qwenfs` is the qwenFS subset of `forced`, not an additional category;
- `qwen_reasoning` is the fixed 256-token reasoning span per execute phase;
- `qwen_payload` is every other Qwen-written token, including tool-call closure
  syntax, so `forced + qwen_reasoning + qwen_payload = total`.

The archived accountant records stage order, cycle, phase, and ownership counts,
but it does **not** record flat-token `start` and `end` offsets. Those fields are
therefore `null`. They were not reconstructed. `emitter` describes the emitters
present in the phase; a mixed phase contains compiled scaffolding, Qwen content,
and the qwenFS response.

Regenerate the exports without loading a model:

```sh
python3 scripts/export_finished_traces.py \
  /path/to/results/end-to-end-v3-b256 \
  benchmarks/babyagi-v3.json traces
```
