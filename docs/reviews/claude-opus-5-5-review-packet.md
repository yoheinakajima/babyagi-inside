# External review packet: semantic BabyAGI inside Qwen3.5-2B

## Review request

Act as a skeptical senior ML systems researcher and prospective paper
coauthor. Review the result below before we continue implementation. Do not
polish the story prematurely. Identify confounds, invalid metrics, benchmark
leakage, overclaims, and engineering dead ends. Then recommend the highest
information-gain next step.

Please return a decision memo with:

1. **Bottom line:** what is real, what is not, and whether this is potentially
   publishable.
2. **Fatal or serious flaws:** ranked by severity.
3. **Protocol assessment:** whether the diagnosis and repaired protocol are
   technically sound.
4. **Compilation recommendation:** the best way to move the independent
   selector and bounded reasoning into weights, or a reason not to.
5. **Model decision:** stay at Qwen3.5-2B, move to 4B, or test both; include a
   stop condition.
6. **Next three experiments:** exact design, metrics, and success/failure
   thresholds.
7. **Paper claim:** one defensible title and a two-sentence claim, plus wording
   we must not use yet.
8. **Reviewer attack:** the strongest likely rejection argument and how to
   answer it honestly.

Be concrete and blunt. Distinguish evidence from inference. If a proposed
metric is weak, replace it.

## Repository state

- Project: qwenFS BabyAGI follow-up
- Branch: `codex/compiled-babyagi-poc`
- Commit under review: `654c16556257d4d36403bf6e1223a1e4acadda2e`
- Model: stock Qwen3.5-2B native weights for semantic experiments
- Decoding: greedy
- Existing compiled artifact: qwenFS plus a bounded five-cycle
  create-prioritize-execute controller merged into a widened Qwen3.5-2B
  checkpoint

The existing compiled artifact is a real systems result: after one objective
prefill and a START token, the inference host samples until a compiled HALT
token. The checkpoint emits phase prompts and tool calls; qwenFS persists and
returns files; a circuit oracle verifies 4,788/4,788 forced tokens; a live
five-cycle trace and accounting exist.

However, the earlier positive pine trace was not valid agent-capability
evidence. Task 2 was always a verification candidate, so Task 1 was always the
correct choice. The model selected Task 1 in all five cycles. The frozen
semantic suite scored 0/12.

## Original diagnosis

### Four native-weight baselines on v1

All use benchmark checksum
`22891c9fe8b277dfc111c3c4feb766e62fc29853ed339eeb1b769ae4391c2f49`.

| Mode | Full automated items | Parsed runs | Main result |
| --- | ---: | ---: | --- |
| Plain chat, thinking off | 4/12 | 9/12 | Three arithmetic items pass; Aster meets automated constraints but fails manual rover criterion. |
| Plain chat, thinking on, 512-token cap | 0/12 | 1/12 | Every item hits the cap. |
| Host Python BabyAGI loop, thinking off | 1/12 | 12/12 | Only Aster meets automated constraints and it fails manual review. |
| Same loop, thinking on | 0/12 | 0/12 | Every item stops in cycle 1; every attempted call hits the cap. |

The thinking-off loop parsed 44/44 create calls, 44/44 choices, and 44/44
execute calls. But choice correctness was 22/44 and only 5/44 result prefixes
were correct. This is native-weight evidence, so compilation was not the cause
of the semantic failure.

### Positional selector failure

We evaluated 25 work-versus-distractor scenarios in both orders plus 10 neutral
prompts.

| Metric | Result |
| --- | ---: |
| Raw accuracy | 25/50 |
| Work-first accuracy | 25/25 |
| Work-second accuracy | 0/25 |
| Task-2 selection | 0/50 |
| Neutral mean `logit(1)-logit(2)` | +6.3375 |
| After subtracting neutral constant | 25/50, now always Task 2 |

The scenario deltas all favored token 1 and overlapped completely. A constant
calibration turns always-1 into always-2; it does not reveal semantic signal.

### Token accounting

The old compiled five-cycle trace attributed 241 tokens to the model, but 150
were closing tool-call tags. Only 91 were actual content across the run.
Execute phases received 2–8 content tokens and every phase was prefixed with an
empty `<think>\n\n</think>` span. This is an unreasonable semantic protocol for
a small hybrid reasoning model.

## Repaired host protocol

The selected protocol is `ledger_jev_bounded128`:

1. The model creates one JSON ledger:

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

   Separating `initial` from `tasks` fixes a recurring error in which Qwen used
   “Begin with X” as a task, consumed one queue slot, and dropped the last
   operation. The first instruction retains the BabyAGI homage: “Create the
   first task,” then asks for the remaining queue in the same call.

2. Each cycle counterbalances the due ledger task against a future or completed
   task. The due task is in slot 2 for roughly half the cycles.

3. Each candidate is evaluated in a separate prompt: “Is this the single next
   undone task?” The host reads `logit(Y)-logit(N)` for each prompt and selects
   the larger margin. Candidate slot is absent from the prompt.

4. Execute receives exactly 128 model-owned reasoning tokens. The controller
   then forces `</think>` and `RESULT_JSON: {"result":`; the model generates
   the payload. The result is carried to the next cycle.

This is currently a **host-side semantic protocol**. The host parses the JSON
ledger and compares two continuous logit margins. Therefore it is not evidence
that the repaired semantic agent is baked into the checkpoint.

## Predeclared success criterion

Before evaluating protocol v2, “working” was defined as:

- at least 50% of the plain-chat automated ceiling on full items;
- greedy decoding and fixed budgets;
- manual language criteria reported separately;
- the due task in slot 2 for at least one third of cycles; and
- semantic choice accuracy above chance, with Task-2 behavior reported.

For v1, plain chat's automated ceiling was 4/12, so the minimum was 2/12.

## Results

### v1 validation

| Metric | Result |
| --- | ---: |
| Full automated passes | 2/12 |
| Passing items | `arithmetic-seven-3`, `structured-records-3` |
| Plain-chat automated ceiling | 4/12 |
| Ceiling retention | 50% |
| Ledger parse | 12/12 |
| Choice correctness | 28/44 |
| Task 2 expected / selected | 23 / 21 |
| Correct result prefixes | 13/44 |

Both passes are exact multi-cycle state chains, not language-rubric edge cases.

### v2 disjoint confirmation

After selecting and freezing the protocol on v1, we created a second 12-item
suite with the same four categories and new values, names, and operations. Its
checksum is
`f740d07a275313c3f1c72ba38d055493f83c3ea9fac93d77d89bf3b38a666816`.
We evaluated the frozen protocol once and did not change prompts or thresholds
afterward.

| Metric | Result |
| --- | ---: |
| Full automated passes | 1/12 |
| Passing item | `arithmetic-eight-3` |
| Plain-chat automated ceiling | 1/12 |
| Ceiling retention | 100% |
| Ledger parse | 11/12 |
| Choice correctness | 28/41 |
| Task 2 expected / selected | 21 / 22 |
| Correct result prefixes | 7/41 |

Plain chat's one v2 pass is a different item, `language-lumen-summary-3`; its
output also passes manual review. The agent's pass is an exact three-cycle
arithmetic chain. Combined across v1 and v2, plain chat passes 5/24 and the
agent passes 3/24; choice accuracy is 56/85; the due task is in slot 2 for 44/85
cycles and the model selects slot 2 in 43/85.

Concern for review: v2 is value-disjoint but structurally templated after v1.
Please assess whether calling it a confirmation suite is justified and what a
stronger confirmation design would require.

## Representative successful traces

### v1 arithmetic validation

Objective: Start with 7; add 5; multiply by 3; subtract 8.

Plan:

```json
{"initial":7,"tasks":["Add 5 to the current number","Multiply the result by 3","Subtract 8 from the result"]}
```

| Cycle | State | Candidate 1 / score | Candidate 2 / score | Selected | Result |
| ---: | ---: | --- | --- | ---: | ---: |
| 1 | 7 | multiply by 3 / 1.125 | add 5 / 1.375 | 2 | 12 |
| 2 | 12 | multiply by 3 / 0.625 | subtract 8 / 0.000 | 1 | 36 |
| 3 | 36 | add 5 / -0.875 | subtract 8 / 0.375 | 2 | 28 |

All three expected states match.

### v2 arithmetic confirmation

Objective: Start with 8; add 4; multiply by 2; subtract 5.

Plan:

```json
{"initial":8,"tasks":["Add 4 to the current number","Multiply the result by 2","Subtract 5 from the result"]}
```

| Cycle | State | Candidate 1 / score | Candidate 2 / score | Selected | Result |
| ---: | ---: | --- | --- | ---: | ---: |
| 1 | 8 | multiply by 2 / 1.000 | add 4 / 1.375 | 2 | 12 |
| 2 | 12 | multiply by 2 / 0.750 | subtract 5 / 0.000 | 1 | 24 |
| 3 | 24 | add 4 / -0.625 | subtract 5 / 0.125 | 2 | 19 |

All three expected states match.

## Phase-isolation result

A diagnostic-only oracle removes planning and selection, supplies the correct
human-authored task list, and retains the same 128-token execute span and state
carry. It passes 5/12 automated v1 items and 20/41 attempted result prefixes.
Four passes are exact-state items:

- literal cinder chain, 5/5 states;
- arithmetic 7, 3/3;
- arithmetic 3, 5/5; and
- structured records, 3/3.

The fifth, Aster, passes automated constraints but needs manual review. This is
why we did not move to 4B: 2B execution has a measurable higher ceiling when
planning and selection are correct.

## Rejected ablations

### Planning

Two format demonstrations caused direct contamination: unrelated values and
operations such as `-a` and “Add 2 to 4” appeared in held-out task plans. The
winning ledger prompt is example-free.

### Selection, replayed on the same 44 v1 ledger decisions

| Selector | Correct | Task-2 selections | Decision |
| --- | ---: | ---: | --- |
| Independent raw Y/N margins | 28/44 | 21 | Selected host protocol |
| Independent Y/N plus 32-token think span | 26/44 | 25 | Reject |
| Ordinal 0–3 emitted digit | 22/44 | 3 | Reject |
| Ordinal expected value from four logits | 25/44 | 22 | Reject |
| Copy chosen candidate text | 25/44 | 19 | Reject |

Only continuous independent-margin comparison clears chance while remaining
position-balanced in these tests.

## Compiling the repair: current feasibility and problem

The existing five-cycle checkpoint uses 24 of 32 reserved microstates. Its
controller scratch ends at dimension 3379 of 4096, leaving 717 dimensions.
A fixed free span can reuse a stage and needs about one offset dimension per
token plus one token-mask coordinate:

| Think span | Estimated scratch end | Dimensions remaining |
| ---: | ---: | ---: |
| 64 | 3444 | 652 |
| 128 | 3508 | 588 |
| 256 | 3636 | 460 |
| 512 | 3892 | 204 |

The bounded execute span is therefore plausible. The hard issue is selection:
the winning host reads two continuous final-model logit margins, whereas the
current front-loaded compiled circuit observes emitted tokens, not final native
logits.

Options under consideration:

1. **Quantized score tokens.** Add score tokens whose output-head rows encode
   bins of `logit(Y)-logit(N)`, emit one score per candidate, store both in the
   compiled controller, compare, and route the winning task. Because input and
   output embeddings are tied, this may require untying `lm_head` or a carefully
   gated score-token construction.
2. **Post-native scoring layer.** Append a small constructed layer after Qwen's
   native layers, read the Y/N direction from the normalized hidden state,
   store or quantize both candidate scores recurrently, and emit the selected
   route. This is architecturally cleaner but expands the claim and proof
   surface.
3. **Change the protocol.** Use a token-observable selector. Tested binary,
   ordinal, and copy variants lost too much accuracy, but a better protocol may
   exist.
4. **Move to 4B.** This may make an emitted-token selector viable, but the 4B
   residual layout overlaps qwenFS's current band and requires a new port.

Please recommend among these, including whether the quantized-margin approach
is scientifically clean or looks like benchmark-specific circuit engineering.

## Current claim boundary

Supported:

> A five-cycle agent controller and persistent filesystem are compiled into a
> Qwen3.5-2B checkpoint. A separate native-weight protocol study diagnoses the
> original 0/12 semantic failure and recovers exact multi-cycle successes on
> validation and a disjoint confirmation suite using a JSON ledger,
> position-independent model judgments, and bounded execution.

Not supported yet:

> The semantically successful ledger/JEV/bounded-execute protocol is baked
> directly into Qwen.

## Questions that should determine the next implementation

1. Is protocol success divided by plain-chat success a meaningful primary
   metric with ceilings this small, or should we replace it with cycle-level
   regret, task-conditional success, or a larger difficulty-gradient suite?
2. Is the current selector a legitimate model-owned decision, or does the host
   comparison of continuous margins make the agent claim too weak?
3. Is 28/44 and 28/41 enough evidence of semantic selection, or should the
   protocol be rejected until confidence intervals and a larger counterbalanced
   decision set exist?
4. Does the oracle result justify staying at 2B, or merely show that a hand-held
   2B can execute a narrow subset?
5. What is the smallest compiled experiment that would materially change the
   paper's credibility?
