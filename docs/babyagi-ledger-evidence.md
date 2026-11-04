# BabyAGI semantic protocol: diagnosis, repair, and confirmation

## Result

The original compiled `babyagi_typed` trace remains valid systems evidence but
is not agent-capability evidence. Its priority choice was degenerate, its
semantic held-out score was 0/12, and 150 of its 241 model tokens were closing
tool-call tags.

A host-side protocol repair now clears the predeclared minimum without changing
the model:

- create one persistent JSON ledger with separate `initial` and ordered `tasks`;
- counterbalance the next ledger task against a completed or future task;
- score each candidate independently with the model's `Y`-versus-`N` logit
  margin and choose the larger margin; and
- give execute exactly 128 model-owned reasoning tokens, then force the result
  envelope and carry the result into the next cycle.

On v1 validation, this protocol passes 2/12 full items, exactly 50% of the 4/12
plain-chat automated ceiling. The passes are exact multi-cycle state chains,
not language-rubric edge cases: `arithmetic-seven-3` and
`structured-records-3`. On the fresh v2 confirmation suite, frozen only after
the protocol was selected, it passes `arithmetic-eight-3` exactly (3/3 states).
Plain chat passes 1/12 on v2, so the protocol retains 100% of that suite's
measured ceiling while succeeding on a different item.

This is not yet a new compiled-checkpoint result. The host currently parses the
ledger and compares two continuous logit margins. Those operations must move
into the checkpoint before the semantic result can support the phrase “baked
into Qwen.” The previously compiled controller/filesystem claim is unchanged.

## Diagnosis first

All v1 runs use Qwen3.5-2B native weights, greedy decoding, benchmark checksum
`22891c9fe8b277dfc111c3c4feb766e62fc29853ed339eeb1b769ae4391c2f49`.

| Baseline | Full automated items | Structural parse | Finding |
| --- | ---: | ---: | --- |
| Plain chat, thinking off | 4/12 | 9/12 | Three arithmetic items pass; Aster is automated-only and fails manual review. |
| Plain chat, thinking on, 512-token bound | 0/12 | 1/12 | All 12 hit the cap. |
| Original Python create/choose/execute loop, thinking off | 1/12 | 12/12 | Aster is automated-only and fails manual review. |
| Original Python loop, thinking on | 0/12 | 0/12 | Every item stops in cycle 1 after hitting the cap. |

The original loop parsed every thinking-off phase, but its selector was only
22/44 and it produced five correct result prefixes in 44 cycles. The failure
was semantic, not compilation-induced: these measurements used native weights.

The 50-probe counterbalanced side-by-side selector confirmed the flaw:

| Metric | Result |
| --- | ---: |
| Raw accuracy | 25/50 |
| Work first | 25/25 |
| Work second | 0/25 |
| Task-2 selection | 0/50 |
| Neutral `logit(1)-logit(2)` | +6.3375 |
| Constant-offset calibration | 25/50; flips always-1 into always-2 |

There is no recoverable semantic signal in that 1/2 comparison. Calibration is
therefore rejected rather than built into the gate.

The neutral probe is motivated by Zhao et al.'s contextual calibration, which
estimates answer bias from content-free inputs. This experiment is narrower: it
tests only a constant two-token logit offset and is not a reproduction of their
full calibration method.

## Phase isolation

The execution oracle is explicitly diagnostic-only: it supplies the correct
human-authored task list, removes create and choose, rolls Qwen's results
forward, and keeps the same 128-token execute span. It passes 5/12 automated
items and 20/41 attempted result prefixes. Four passes are exact-state items:

- `literal-cinder-5` (5/5 states);
- `arithmetic-seven-3` (3/3);
- `arithmetic-three-5` (5/5); and
- `structured-records-3` (3/3).

The fifth, Aster, meets automated constraints but still requires manual review.
This shows that 2B execution can sustain non-trivial sequential work when the
task representation is correct. A move to 4B is not justified yet.

## Why the JSON ledger helps

One-shot planning initially failed because Qwen treated “Begin with X” or
“Start with X” as an initialization task, consumed one task slot, and dropped
the last operation. The ledger gives that impulse an explicit field:

```json
{
  "initial": 7,
  "tasks": [
    "Add 5 to the current number",
    "Multiply the result by 3",
    "Subtract 8 from the result"
  ]
}
```

On v1, the ledger parses on 12/12 items. On the disjoint v2 suite it parses on
11/12; one structured-word plan repeats until the 384-token bound. The first
instruction retains the original BabyAGI homage: “Create the first task,” then
the model creates the remaining queue in the same call.

Two format demonstrations were tested and rejected. The 2B model copied
unrelated example content such as `-a` and “Add 2 to 4” into held-out tasks.
Examples are therefore absent from the winning prompt.

## Winning host-protocol metrics

| Suite | Plain automated ceiling | Agent full items | Plan parse | Choice accuracy | Task 2 expected / selected | Correct prefixes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 validation | 4/12 | 2/12 | 12/12 | 28/44 | 23 / 21 | 13/44 |
| v2 confirmation | 1/12 | 1/12 | 11/12 | 28/41 | 21 / 22 | 7/41 |
| Combined | 5/24 | 3/24 | 23/24 | 56/85 | 44 / 43 | 20/85 |

The Task-2 rate is non-trivial: the right task occupies slot 2 in 44/85 cycles,
and the model selects slot 2 in 43/85. This is not the original always-Task-1
trace. Choice correctness is measured against the due entry in the
model-generated ordered ledger; ledger quality is reported separately through
plan parsing and end-to-end scores.

The fresh v2 suite checksum is
`f740d07a275313c3f1c72ba38d055493f83c3ea9fac93d77d89bf3b38a666816`.
It was created after the v1 protocol result and evaluated once. No prompt or
threshold was changed in response to v2.

## Rejected selector variants

All variants below replay the same 44 v1 ledger decisions.

| Selector | Correct | Task-2 selections | Decision |
| --- | ---: | ---: | --- |
| Independent raw Y/N margin | 28/44 | 21 | Keep for host protocol. |
| Independent Y/N with 32-token think span | 26/44 | 25 | Reject; added variance. |
| Ordinal 0–3 emitted digit | 22/44 | 3 | Reject; severe low-position bias. |
| Ordinal expected value from four logits | 25/44 | 22 | Reject. |
| Copy chosen candidate text | 25/44 | 19 | Reject as primary selector. |

The raw continuous-margin comparison is the only tested selector that clears
chance while remaining position-balanced. Its compiled implementation is the
main open systems problem: either quantize each independent margin inside the
checkpoint and compare the codes, or add a post-native scoring circuit that
stores both margins before routing the selected task.

## Bounded-reasoning feasibility

The current 2B compiled checkpoint uses 24 of 32 reserved microstates and ends
controller scratch at dimension 3379, leaving 717 dimensions. A fixed span can
share the existing microstate and extends the offset decoder approximately one
dimension per token:

| Span | Estimated scratch end | Remaining dimensions |
| ---: | ---: | ---: |
| 64 | 3444 | 652 |
| 128 | 3508 | 588 |
| 256 | 3636 | 460 |
| 512 | 3892 | 204 |

The native prototype proves the exact-128 pattern: the controller opens
`<think>`, Qwen owns 128 ordinary tokens, the controller forces `</think>` and
the result prefix, and Qwen emits the result. Actual compilation, circuit
oracle coverage, and live merged-checkpoint traces are still required.

This bounded transition is necessary because Qwen3.5's official model card
describes thinking as the default mode, while the native bounded suite shows
that the 2B model does not reliably close an unconstrained reasoning span.

## Publication boundary

Defensible now:

> The systems artifact compiles a five-cycle create–prioritize–execute
> controller and persistent filesystem into Qwen3.5-2B. A separate native-weight
> protocol study diagnoses its 0/12 semantic result, replaces a degenerate
> positional selector with independent judgments, introduces a persistent JSON
> task ledger and bounded execution, and recovers exact multi-cycle successes on
> both validation and disjoint confirmation suites.

Not defensible yet:

> The semantically successful ledger/JEV protocol is baked into Qwen.

That stronger sentence requires an in-weight margin comparator, compiled
bounded spans, a live merged trace, and rerunning the frozen confirmation suite
without a host parser or logits comparator.

## Reproduce

```bash
# Four-way v1 diagnosis
uv run python -m qfs.ceiling models/Qwen3.5-2B \
  benchmarks/babyagi-v1.json results/ceiling-2b-v1

# Fifty counterbalanced side-by-side probes
uv run python -m qfs.selector_probe models/Qwen3.5-2B \
  --output results/diagnosis-2b-v1/selector-50.json

# Diagnostic-only execute ceiling
uv run python -m qfs.oracle_execute models/Qwen3.5-2B \
  benchmarks/babyagi-v1.json results/diagnosis-2b-v1/oracle-execute-bounded128

# Winning v1 validation protocol
uv run python -m qfs.ledger_protocol models/Qwen3.5-2B \
  benchmarks/babyagi-v1.json results/ledger-protocol-2b-v1

# Frozen confirmation ceiling and protocol
uv run python -m qfs.ceiling models/Qwen3.5-2B \
  benchmarks/babyagi-v2.json results/confirmation-v2-ceiling \
  --modes plain_off --plain-max 512
uv run python -m qfs.ledger_protocol models/Qwen3.5-2B \
  benchmarks/babyagi-v2.json results/confirmation-v2-ledger
```

## References

- Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh,
  [“Calibrate Before Use: Improving Few-shot Performance of Language
  Models”](https://proceedings.mlr.press/v139/zhao21c.html), ICML 2021.
- Qwen team, [Qwen3.5-2B model
  card](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/README.md).
