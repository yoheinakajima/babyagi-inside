# Qwen3.5-4B full live-compiled sweep results

## Result

> **Update (2026-11-08).** The plain comparison below is superseded. Every one-turn plain
> run stopped at its token ceiling before Qwen closed its thinking, so the +40.6 points
> measured truncation. In a preregistered follow-up at token budgets matched to the
> compiled loop's, with thinking closed at a cap and an answer budget of its own, plain
> Qwen3.5-4B completes **29/32**: compiled minus plain is −3.1 points (95% CI [−9.4, 0.0];
> 0 improved, 1 regressed; p = 1.0). See the [budget-forced results](plain-forced-v3-results.md).
> The systems results in this document are unaffected.

The bounded BabyAGI-style loop compiles into Qwen3.5-4B with the same exactness as the
2B release, and the compiled 4B checkpoints complete **28 of the 32** frozen v3
objectives. Plain Qwen3.5-4B completes 15. Every one of the 32 live runs executed every
compiled phase in order, read back every qwenFS file exactly, halted on the compiled HALT
token, and accounted for every emitted token. The paired improvement over plain 4B is
**+40.6 points** (item-bootstrap 95% CI [+25.0, +59.4]; 13 items improved, 0 regressed;
exact McNemar p = 0.00024). The preregistered reading is an improvement, since the
interval's lower bound is above zero.

The sweep was preregistered before its first item
([preregistration](qwen35-4b-full-live-compiled-preregistration.md)) and ran every item
once, greedily, from a fresh qwenFS cache. The runner, decoder, evaluator, token
accountant, scorer, and settings are byte-identical to the [2B sweep's](babyagi-v3-full-live-compiled-preregistration.md).
The scorer is the frozen exact-string scorer; a casing slip fails a chain.

## Port checks before the sweep

Four checkpoints, one per cycle count, built from Qwen3.5-4B (revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) on an Apple M5 with 32 GB, torch 2.15.0,
transformers 5.22.1:

| Cycles | Build | Rebuild identical | Forced-token oracle | Parity max KL (nats) | Top-1 |
| ---: | --- | --- | ---: | ---: | ---: |
| 2 | 40.7 s, max RSS 12.6 GB | yes, `bbc5aa8b…` | 840/840 | 3.8e-5 | 4/4 |
| 3 | 39.7 s, 14.8 GB | yes, `edbceb5f…` | 1,132/1,132 | 3.8e-5 | 4/4 |
| 5 | 42.3 s, 14.6 GB | yes, `8ab06101…` | 1,716/1,716 | 3.8e-5 | 4/4 |
| 8 | 41.1 s, 14.0 GB | yes, `913f236d…` | 2,592/2,592 | 3.8e-5 | 4/4 |

- **Rebuild identical:** a second build from the same inputs had the same tree SHA-256.
  Full digests are in the preregistration. The table gives the first build; the rebuilds
  took 41.6–57.7 s with max RSS 13.9–15.0 GB. macOS reports a peak memory footprint of
  25.3–26.4 GB across the eight builds, above the RSS figures and below 32 GB.
- **Oracle:** `tests.test_agent_circuit`. The totals equal the 2B release's, and there
  were no typed-choice or free-span failures.
- **Parity:** `tests.test_agent_parity` (four prompts, agent inactive) against stock
  Qwen3.5-4B, with the reference model's rotary frequencies initialized (the
  `merge.empty_model` fix). It is identical across cycle counts because the agent's
  cycle-dependent weights are never reached when it is inactive.

The 2B checkpoints built from the same code are byte-identical to the 2B release's
preregistered digests.

## Systems

| Check | Compiled 4B | Compiled 2B (release sweep) |
| --- | ---: | ---: |
| Runs complete and checksum-bound | 32/32 | 32/32 |
| Checkpoint HALT | 32/32 | 32/32 |
| Controller sequence exact | 32/32 | 32/32 |
| All cycles and phases complete | 32/32 (432/432 phases) | 32/32 |
| qwenFS reads token-exact | 32/32 (256/256 reads) | 30/32 |
| Disjoint token accounting exact | 32/32 | 32/32 |
| **All system checks** | **32/32** | 30/32 |

The two 2B filesystem exceptions came from tool-call markup inside Qwen's reasoning that
switched qwenFS into write mode, a parser behavior inherited from qwenFS. It did not
occur in the 4B sweep. That is not evidence the behavior is gone: nothing in qwenFS's
parser changed.

Token accounting over all 32 runs: 100,545 emitted tokens. 58,972 were forced (39,616
ROM, 18,924 qwenFS, 432 controller control), and 41,573 were written by Qwen (36,864
bounded reasoning, 2,949 payload, 1,760 tool-call closure). The longest run emitted
5,675 tokens against the 7,000-token bound, and no run reached the bound. Decoding ran at
6.0 to 6.1 tokens per second at every length, 16,577 s in total (2B: 8.2 tokens per
second).

## Capability

| | Compiled 4B | Plain 4B | Compiled 2B | Plain 2B |
| --- | ---: | ---: | ---: | ---: |
| Exact chains | **28/32 = 87.5%** | 15/32 = 46.9% | 8/32 = 25.0% | 5/32 = 15.6% |
| Wilson 95% CI | [71.9, 95.0] | | [13.3, 42.1] | |
| Exact steps | 134/144 = 93.1% | | 36/144 = 25.0% | |

By category and length (compiled 4B, exact chains; exact steps in parentheses):

| Category | Chains | | Length | Chains |
| --- | ---: | --- | ---: | ---: |
| Sequential arithmetic | 8/8 (36/36) | | 2 | 8/8 |
| Structured operation | 8/8 (36/36) | | 3 | 8/8 |
| Checkable language | 6/8 (30/36) | | 5 | 7/8 |
| Ordered literal | 6/8 (32/36) | | 8 | 5/8 |

Paired comparisons over the same 32 items (compiled 4B minus each; item-bootstrap 95% CI,
10,000 resamples, seed 0; exact McNemar):

| Comparison | Difference | 95% CI | Improved / regressed | McNemar p |
| --- | ---: | --- | ---: | ---: |
| Plain 4B, one turn (preregistered primary; truncated) | +40.6 points | [+25.0, +59.4] | 13 / 0 | 0.00024 |
| Plain 4B, budget-forced F2 (later preregistration) | **−3.1 points** | [−9.4, 0.0] | 0 / 1 | 1.0 |
| Compiled 2B (descriptive) | +62.5 points | [+46.9, +78.1] | 20 / 0 | 1.9 × 10⁻⁶ |
| 4B host simulation (descriptive) | +3.1 points | [0.0, +9.4] | 1 / 0 | 1.0 |

The 2B sweep's paired result against plain 2B was +9.4 points [0.0, +21.9] (p = 0.25),
which was inconclusive.

The plain arm is one thinking-enabled turn with a token budget matched to the item's
length (256 per cycle plus 256 for the answer). Every plain 4B run reached that budget,
and 13 of 32 gave no parseable answer. Ten of the 13 items the compiled loop gains are
among those 13. Much of the measured gain is therefore plain Qwen running out of budget
before it answers, where the compiled loop spends the same reasoning budget one step at
a time and writes each result to qwenFS.

### Against the host simulation

Before the port, a host simulation of the same protocol with stock Qwen3.5-4B completed
27/32 chains. The compiled checkpoints agree with it on 31 of 32 chain verdicts and
produce identical results on 30 of 32 items. The two differences:

- `v3-literal-route-3`: the compiled run is exact; the simulation wrote `willow3` for
  `willow-n3`.
- `v3-literal-route-8`: the compiled run is exact for seven steps and fails the eighth;
  the simulation wrote `n8` for `N8` from step 2. Both fail the chain.

The simulation approximates qwenFS responses and control symbols in stock Qwen, so small
greedy divergences are expected. Both differences are single-token divergences in a
literal string, and on neither item is the compiled result worse.

### The four failures

Every failure is a Qwen content error inside a correct loop: all four runs pass every
system check, and in each the controller executed the ledger Qwen wrote.

| Item | Steps | First failed step | What happened |
| --- | ---: | ---: | --- |
| `v3-literal-route-5` | 2/5 | 3 | Qwen's ledger listed precomputed states instead of operations, and its third state already ended in `?` (`R5:COBALT-N5?` for `R5:COBALT-N5`); step 4 then added a stray `-`. Identical in the host simulation |
| `v3-literal-route-8` | 7/8 | 8 | Put the final `!` inside the bracket (`[r8:juniper_n8?!]` for `[r8:juniper_n8?]!`) |
| `v3-language-lantern-8` | 5/8 | 6 | "uppercase only the word lanterns" produced `Lanterns`, not `LANTERNS`. Identical in the host simulation |
| `v3-language-river-8` | 5/8 | 6 | The same uppercase-one-word step failed. Identical in the host simulation |

### Plans

Qwen wrote a parseable JSON ledger on 32/32 items. The frozen primary plan metric, exact
task strings plus exact initial state, is 6/32. The exploratory operation-signature score,
designed after observing 2B ledgers and reported as post hoc, is 21/32. 4B usually
paraphrases a step (for example "Sort the array in ascending numeric order." for the
frozen "sort it in ascending numeric order") and then executes it correctly. That is why
it completes all 8 structured-operation chains with 0 exact structured ledgers.

## Interpretation

- The compiled machinery carries over to a larger base unchanged in kind: the same
  forced-token totals, exact HALT, exact reads, and exact accounting. qwenFS's layout and
  circuit constants did not move. Only the native residual's placement, the controller
  block's position (from 3072 to 3584), and the DeltaNet head map changed.
- On this suite, compiling the loop into Qwen3.5-4B improves over plain Qwen3.5-4B, with
  no item regressed, though much of that gain is plain Qwen exhausting its single-turn
  budget (see Capability). On 2B, the same compiled loop's paired result against plain
  chat was inconclusive. The ledgers point to planning as the difference between the
  bases: 4B's ledgers name executable steps on 21 of 32 items, across all four
  categories and often paraphrased, while 2B's were executable only for arithmetic (8/32
  by the exploratory signature score). Some 4B ledgers list states rather than operations
  (`v3-literal-route-5`).
- The residual errors are exact-string edits (casing and punctuation placement) at five
  and eight cycles, not loop or filesystem errors.

## Limits

- One frozen suite of 32 short, checkable chains with deterministic answers. It tests
  bounded sequential execution and task extraction, not open-ended planning or tool use.
- One run per item, greedy, on one machine (Apple M5, MPS, bf16). There is no CUDA
  oracle or timing yet.
- The plain baseline is one thinking-enabled turn with a length-matched token budget.
  Every plain 4B run reached that budget, and 10 of the 13 compiled gains are items where
  plain Qwen gave no parseable answer. A plain baseline with a larger budget was not run.
- Cycle count is fixed per checkpoint, as in the 2B release.
- Agent-inactive parity is a four-prompt smoke check, not a broad equivalence test.

## Reproduction

```sh
scripts/build_checkpoints.sh 4b
uv run python -m qfs.compiled_sweep benchmarks/babyagi-v3.json \
  results/qwen35-4b-full-live-compiled \
  --checkpoint-pattern 'build/babyagi-4b-ledger-pointer-b256-m{cycles}' \
  --plain-root PLAIN_4B_ROOT --device mps --seed 0 --temperature 0 --limit 7000
```

The sweep manifest's experiment SHA-256 is
`4a96e7aab95cc0db341a35281fe4a5562eef92ead69ecd16c39fd9ebf6491a29`, and its summary
SHA-256 is `8b0ee42deb182cd591c5b45c79cfa1f1f0a0d14a27bdbd24f60656fb3a33904e`. All 32
runs are exported in the public trace schema in [`traces/qwen3.5-4b/`](../traces/qwen3.5-4b/).
