# Claude Opus 5.5 High decision memo

- Date: 2026-11-03
- Reviewer: Claude Opus 5.5, High reasoning
- Source packet: `docs/reviews/claude-opus-5-5-review-packet.md`
- Reviewed commit: `654c165`
- Claude conversation: <https://claude.ai/chat/d5459a26-c1ba-4438-b5cf-686ace7c99b8>
- Private Claude Doc: <https://claude.ai/artifact/SeadfMpTXmQniTKLg8RnwL>

This is a formatting-normalized capture of the external decision memo. Numerical
claims and recommendations below are the reviewer's, not additional results from
this repository.

## 1. Bottom line

> **Verdict:** the compilation result is real; the semantic agent result is not
> yet evidence of anything beyond “not broken.” Do not compile the selector
> next. Fix the measurement, demote selection to a deterministic pointer, and
> make the paper about semantics-preserving compilation.

### Real, according to the evidence

- A fixed five-cycle controller plus qwenFS runs inside the checkpoint from
  START to HALT, with 4,788/4,788 forced tokens verified by the circuit oracle.
- The original 0/12 is a native-weight failure, not a compilation failure: the
  host loop with thinking off was 1/12, and the original selector made 0/50
  Task-2 selections.
- The repaired protocol removes positional collapse. Slot is absent from the
  prompt, and Task-2 selections track the number of Task-2 expected cases
  (43/85 versus 44/85).

### Not real yet

- There is no end-to-end capability gain. Pooled, the agent passes 3/24
  (Wilson 95% CI 4–31%) and plain chat passes 5/24 (9–41%). The point estimate
  favors plain chat; v2 is 1/12 versus 1/12.
- “Ceiling retention” is noise. v1's 50% has a bootstrap 95% interval of
  0–200%; v2's “100%” comes from two different items each passing once.
- Selection is above chance but weak and possibly trivial. Held-out v2 is
  28/41 = 68% (Wilson 53–80%, one-sided p = 0.014 before clustering). v1's
  28/44 is a tuning-set result: the best of five chance selectors reaches at
  least 28/44 about 15–22% of the time.
- No semantic behavior is in the weights yet.

### Publishability

Not as an agent-capability paper. Potentially yes as a systems paper whose claim
is equivalence: a host agent protocol compiled into weights produces
decision- and token-identical traces, while semantic quality remains bounded by
the base model. A workshop paper on the diagnosis alone—positional collapse,
budget starvation, and tag-token inflation—is defensible today.

## 2. Fatal and serious flaws, ranked

1. **Fatal for a capability claim: the selection task is degenerate.** With an
   ordered ledger, the correct choice is the earliest task not yet done. The
   host already knows this from the JSON it parsed, so the model is re-deriving
   a pointer the controller holds. Make selection deterministic. Test semantic
   selection only when list order does not reveal the answer.

2. **Distractor type is confounded with cycle index.** In a three-task ledger,
   cycle 1 can only offer a future distractor and the final cycle can only offer
   a completed one. Slot counterbalancing is vacuous under independent scoring
   because slot never enters the prompt. Report distractor type by cycle and
   sample the two independently.

3. **Margins sit at bf16 resolution.** Every reported margin is a multiple of
   0.125, the bf16 spacing for logits in the relevant range. Many choices ride
   on one or two ULPs, while tie handling is unreported. Upcast final norm and
   `lm_head` to fp32, then report gap distribution, tie rate, tie rule, and the
   bf16-to-fp32 flip rate.

4. **v1 is a development set, not validation.** At least five selectors plus
   prompt and demo variants were chosen on the same 44 decisions. Rename v1
   “dev” and report only held-out numbers as results.

5. **“Planning” is transcription on enumerated arithmetic objectives.** Add
   objectives whose steps are not listed, or call this task extraction.

6. **Execution is the binding constraint.** With an oracle ledger and oracle
   selection, correct result prefixes are 20/41 (49%, CI 34–64%). The forced
   `</think>` at 128 tokens likely truncates reasoning mid-step. Measure this
   rather than assuming it.

7. **v2 is a parallel form, not confirmation.** It uses the same templates with
   new values and has only 12 items. Plain chat falls from 4/12 to 1/12, so the
   suites are not difficulty-matched. Call it a value-disjoint replicate.

8. **Metrics are under-defined.** Define every phase metric as a conditional
   probability: model or gold ledger, model or gold task, and whether incoming
   state is correct.

9. **Non-interference is unmeasured.** Scratch dimensions can change the final
   RMSNorm denominator and therefore native logits. Report KL and top-1
   agreement against stock Qwen at every model-owned position.

10. **Language items use a single manual rater.** Add a blind second rater and
    report agreement.

## 3. Protocol assessment

The diagnosis is sound. The repaired protocol is a sound diagnostic but a weak
agent design. The positional-collapse analysis is the strongest result:
subtracting the neutral constant flips always-1 to always-2, cleanly showing
that calibration moved the bias without uncovering signal. The token accounting
finding—150 of 241 model tokens were closing tags—is also worth publishing as a
methods finding.

What holds up:

- Separating `initial` from `tasks` fixes a real failure mode without embedding
  benchmark content.
- The example-free ledger prompt is the right response to demo contamination.
- Independent Y/N log-likelihood scoring is a legitimate model-owned score.

What does not:

- The selector is a legitimate score but a redundant decision. The host argmax
  is not the problem; the question has a symbolic answer the host already has.
- “Task 2 expected 23 / selected 21” is not evidence of position balance.
  Independent scoring guarantees this apart from ties.
- The 128-token budget was chosen, not measured. Earlier thinking-on runs with a
  512-token cap hit the cap on every item, so truncation is the likeliest
  execution failure.

Answers to the review packet's key questions:

- Retention is not a valid primary metric.
- Host margin comparison is model-owned as a score, but not evidence of agency.
- 28/44 and 28/41 are insufficient. Only 28/41 is held out. Pooled 56/85 has a
  roughly 53–78% clustered 95% interval: above chance, but compatible with a
  shallow heuristic.
- The oracle supports staying at 2B on engineering grounds, not as proof of a
  capability ceiling.
- The smallest compiled experiment that matters is bounded execution plus a
  deterministic pointer, followed by trace equivalence against the host.

### Replacement metrics

| Metric | Definition | Purpose |
|---|---|---|
| Step execution accuracy, p_e | P(correct output \| gold task, gold incoming state), one step | Isolates execution |
| Selection accuracy, p_s, plus AUROC | P(due task outscores distractor \| gold ledger, gold progress), fp32 margins | Measures selector without a threshold artifact |
| Plan exactness, p_p | P(ledger equals gold up to paraphrase) | Separates parsing from planning |
| Per-step hazard, h | Slope of log chain success versus lengths 2, 3, 5, 8 | Tests whether a loop degrades more slowly |
| Paired end-to-end delta | Agent minus plain chat on the same items, McNemar plus paired bootstrap | Replaces ratios of tiny counts |
| Decomposition check | Observed chain success versus p_p · (p_s p_e)^k | Detects correlated errors and state-carry bugs |

With p_s = 0.66 and p_e = 0.49, a three-step chain is expected to succeed about
3.4% of the time, consistent with 3/24. Raising p_e alone to 0.90 gives about
21%, so execution is the main lever.

## 4. Compilation recommendation

Compile the bounded execute span now. Do not compile the independent selector.
Replace it with a compiled ledger pointer, which is also closer to BabyAGI: the
original execution agent takes the first task in the prioritized list.

### Target architecture: ledger pointer

- Model-owned: ledger JSON during create, plus the execute payload after a
  fixed think span.
- Compiled controller: a pointer `i` into the ledger, incremented each cycle.
  Each phase emits `TASK_ID i`; qwenFS returns task text and carried state.
- Compiled budget: a fixed think span B selected empirically, then forced
  `</think>` and `RESULT_JSON: {"result":`. At B = 128, roughly 588 dimensions
  remain; at B = 256, roughly 460 remain. Both fit.
- Optional semantic step outside the critical path: re-prioritization where the
  model emits a permutation of task IDs as tokens. Add this only when there is a
  suite where order is not already given.

If a margin selector is eventually compiled, use a post-native readout layer.
It should compute the same Y/N margin as the host, then retrieve the two stored
margins and compare them with a frozen tie rule. Quantized score tokens are
legitimate but worse: preserving current gaps requires bin width at most 0.125,
roughly 50 tokens over ±3, an untied `lm_head`, and additional ties.

Freeze the scoring template, threshold, tie rule, and any bin width before
looking at benchmark outcomes. Claim only equivalence to the host.

## 5. Model decision

Compile at 2B. Run 4B host-side only as a diagnostic arm in the selector and
execution studies. Port qwenFS to 4B only if all three conditions hold:

1. 2B's best step execution accuracy at any budget up to 256 is below 0.85 on
   arithmetic and records.
2. 4B reaches at least 0.95 on the same steps and budget.
3. 4B's per-step hazard is at most half of 2B's.

Stay at 2B if it reaches at least 0.90 step accuracy at a budget up to 256.
Stop the 4B arm if its step accuracy is within five points of 2B.

## 6. Next three experiments

Experiments 1 and 2 are host-side and can run in parallel. Experiment 3 is
gated on their outputs. Compute margins in fp32 and bootstrap confidence
intervals over items, not individual correlated decisions.

### Experiment 1: selector validity audit

Question: does the Y/N margin read task progress, or task-text priors and list
position?

- Data: about 500 decisions from about 60 new gold ledgers of 3–8 tasks.
  Distractor type and distance are sampled independently of cycle.
- Models: 2B and 4B, host-side.
- Conditions:
  - Full current prompt.
  - Candidate text only, no ledger or progress.
  - Ledger with progress, completed list, and state ablated.
  - Display-shuffled ledger where the true order must be recovered from the
    objective.
  - Symbolic earliest-undone baseline, 100% by construction.
- Metrics: accuracy, margin AUROC, accuracy by distractor type and cycle, tie
  rate, and bf16-to-fp32 flip rate.
- Success for keeping semantic selection as a research track: Full at least
  0.80 with CI lower bound at least 0.70; every distractor type at least 0.70;
  candidate-only at most 0.60; progress-ablated completed distractors at most
  0.60; shuffled display at least 0.70; flip rate at most 2%.
- Stop: Full CI lower bound below 0.60, or candidate-only within five points of
  Full. Then drop semantic-selection claims and ship the pointer.

### Experiment 2: execution budget curve

Question: is execution failing from capability limits or forced truncation?

- Data: at least 300 single steps with gold task and gold incoming state,
  stratified across four categories. Use two blind raters for language items.
- Arms: thinking off with a 64-token visible scratch line; thinking budgets 64,
  128, 256, and 512; uncapped with a 2,048-token ceiling.
- Metrics: step accuracy, natural-termination rate, accuracy on naturally
  terminated versus truncated steps, and tokens used.
- Success: some B* up to 256 gives at least 0.90 on arithmetic and records, and
  the natural-versus-truncated gap explains most 128-token failures. Freeze B*.
- Stop: no budget up to 512 reaches 0.80 on 2B. Try one alternate result format.
  If that fails, apply the 4B rule above. Do not tune by category.

### Experiment 3: preregistered gradient suite

Question: is compilation semantics-preserving, and does the loop degrade more
slowly with chain length than plain chat?

- Suite: 160 items = 4 categories × lengths 2, 3, 5, and 8 × 10. Commit and
  hash generator, templates, gold answers, and scorer before generation. Hold
  out at least two phrasing families per category from development. Seal half
  the suite for a single run. Report a separate non-enumerated-step category as
  the only planning test.
- Arms:
  - Plain chat with thinking on and total token budget matched to k × B*.
  - Host ledger + pointer + B* protocol.
  - Compiled checkpoint running the same protocol in weights.
- Primary equivalence metric: identity of every model-owned token between host
  and compiled arms. Non-interference: mean KL at most 1e-3 nats and top-1
  agreement at least 99.9% against stock Qwen at model-owned positions.
- Capability metric: per-step hazard plus paired McNemar delta at lengths 5 and
  8.
- Success: token identity in at least 99% of runs, with every divergence
  explained by numerical nondeterminism. This alone supports the systems paper.
  Capability language additionally requires the host loop to beat plain chat at
  length at least 5 with a paired CI excluding zero.
- Stop: identity below 99% without a numerical explanation is a compilation
  bug. If the host never beats plain chat, use no capability language.

## 7. Paper claim

### Defensible title after Experiment 3 passes

**Compiling an Agent Loop into a Language Model's Weights: A
Semantics-Preserving Controller and Filesystem for Qwen3.5-2B**

### Claim

> We compile a bounded BabyAGI-style controller and a persistent filesystem
> into a widened Qwen3.5-2B checkpoint. On a preregistered suite, the compiled
> checkpoint reproduces the host-orchestrated protocol token for token without
> measurably perturbing native outputs. Task success is therefore bounded by the
> base model's per-step execution accuracy, which we measure and decompose.

Defensible today at workshop scope: the compiled controller result plus the
diagnosis of positional collapse, tag-token inflation, and budget starvation.

Wording to avoid for now:

- “semantic agent baked into the weights,” “self-contained agent,” or “the
  model runs BabyAGI itself”
- “recovers” or “matches plain-chat performance,” “100% ceiling retention,” or
  “50% of the ceiling”
- “confirmation suite” for v2; use “value-disjoint replicate”
- “validation” for v1; use “development set”
- “planning” for enumerated objectives; use “task extraction”
- “semantic selection” or “position-independent reasoning” until Experiment 1
  passes
- “agent capability,” “emergent,” or comparisons to frontier agents

## 8. Strongest reviewer attack and honest answer

Attack: this is a hand-built finite-state machine wired into spare residual
dimensions, in the lineage of RASP, Tracr, and looped-transformer
constructions. It does nothing a short Python loop cannot. The authors' own
data show the agent is no better than plain chat, and the selector solves a
problem the host already solved. There is engineering here but no ML insight.

Answer:

1. Concede that compilation adds no capability. State that task success is
   bounded by the base model's measured per-step accuracy.
2. Claim the verified result: a multi-phase controller with tool I/O compiled
   into a pretrained production model, token-equivalent to the host protocol
   and non-interfering with native outputs. Run a literature review before any
   “first” claim.
3. Explain why equivalence matters: the host shrinks to a sampler plus file
   store, while the controller becomes a fixed, verifiable artifact shipped
   with the weights.
4. Treat positional collapse, 62% tag-token overhead, and budget starvation as
   reusable evaluation findings.
5. If motivation remains unconvincing, target a systems or workshop venue
   rather than inflating the semantic claim.

The second-strongest attack is that selection is solvable by `min(undone)`.
The pointer architecture and selector-audit symbolic baseline address that
directly.
