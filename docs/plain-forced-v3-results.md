# Budget-forced plain baseline results

## Result

Given an answer budget at a token budget matched to the compiled loop's, plain Qwen does
at least as well as the compiled checkpoints on the frozen v3 suite, on both base models.
On Qwen3.5-4B, plain Qwen completes **29/32** chains under all three budgets against the
compiled checkpoints' 28/32. The primary F2 comparison, compiled minus plain, is **−3.1
points** (item-bootstrap 95% CI [−9.4, 0.0]; 0 items improved, 1 regressed; exact McNemar
p = 1.0): inconclusive by the preregistered reading, so H-4B is not supported. On
Qwen3.5-2B, plain Qwen completes 9, 12 and 15 under F1, F2 and F3 against the compiled
loop's 8. Under no arm, on either model, does the compiled loop pass an item plain Qwen
fails.

The compiled loop's earlier advantage over plain Qwen (+40.6 points on 4B, +9.4 on 2B) came
from the one-turn plain arm, which stopped every run before Qwen closed its thinking. With
an answer guaranteed at that same one-turn ceiling (F1), plain 4B already completes 29.

The arms, caps, and analysis were preregistered
([preregistration](plain-forced-v3-preregistration.md)) and every item ran once per arm.
The records are in [`results/plain-forced-v3/`](../results/plain-forced-v3/).

## Results

Every plain run in all six steps was force-closed at its thinking cap; none closed its own
thinking.

| Model | Arm | Plain chains | Steps | Parsed | Answer cap hit | Model tokens | Seconds | Compiled minus plain | 95% CI | Improved / regressed | McNemar p | Reading |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| 4B | F1 | 29 | 130/144 | 31 | 1 | 38,771 | 3,479 | −3.1 | [−9.4, 0.0] | 0 / 1 | 1.0 | inconclusive |
| 4B | **F2** | **29** | 130/144 | 32 | 0 | 46,800 | 4,228 | **−3.1** | **[−9.4, 0.0]** | 0 / 1 | 1.0 | **inconclusive** |
| 4B | F3 | 29 | 130/144 | 31 | 0 | 65,425 | 5,885 | −3.1 | [−9.4, 0.0] | 0 / 1 | 1.0 | inconclusive |
| 2B | F1 | 9 | 49/144 | 22 | 6 | 40,383 | 1,679 | −3.1 | [−9.4, 0.0] | 0 / 1 | 1.0 | inconclusive |
| 2B | **F2** | **12** | 61/144 | 23 | 3 | 52,256 | 2,199 | **−12.5** | **[−25.0, −3.1]** | 0 / 4 | 0.125 | **plain ahead** |
| 2B | F3 | 15 | 74/144 | 27 | 3 | 73,305 | 3,081 | −21.9 | [−37.5, −9.4] | 0 / 7 | 0.016 | plain ahead |

- **Qwen3.5-4B.** The three arms reach the same verdict on every item. Plain Qwen fails
  the three eight-cycle exact-string items the compiled loop also fails (lantern-8,
  river-8, route-8) and passes route-5, the compiled loop's fourth failure. More budget
  than the one-turn ceiling changes nothing.
- **Qwen3.5-2B.** Plain Qwen gains with budget, all of it outside sequential arithmetic,
  which it passes 8/8 under every arm. The F2 interval lies below zero (plain ahead by the
  preregistered reading), though with four disagreements the exact test is not
  significant; under F3 both agree.
- **Cost.** Under F2, plain 4B used 46,800 model tokens and 1.2 hours. The compiled 4B
  sweep emitted 100,545 tokens, 41,573 of them chosen by Qwen, in 4.6 hours.
- **Predictions that could fail held:** every arm beats the one-turn arm on its model
  (2B 9, 12, 15 against 5; 4B 29 against 15), and every run was force-closed.

## What this means

On this suite, the compiled loop provides control, not measurable capability: an exact
controller, filesystem and HALT, and a trace that assigns every token to its owner, at
about four times plain Qwen's time. One greedy run per item on 32 short checkable chains
cannot detect small effects, and it says nothing about tasks a single turn cannot hold.

## Reproduction

The records were produced by the lab runner (record identity `runner_sha256`
`8e191ee0…`). [`qfs/plain_forced.py`](../qfs/plain_forced.py) carries the same prompt,
caps, forcing, parsing, and scoring. Run on the same Apple M5 (torch 2.15.0,
transformers 5.22.1), it regenerated four records token for token: 2B F1 on
`v3-arithmetic-add-2` and `v3-language-lantern-2`, and 4B F2 on `v3-arithmetic-add-2` and
`v3-literal-route-2`. The other 188 were not regenerated.

```bash
uv run python -m qfs.plain_forced run models/Qwen3.5-4B benchmarks/babyagi-v3.json \
  results/plain-forced-v3/compiled.json 4b F2 /tmp/plain-forced-4b-F2
uv run python -m qfs.plain_forced compare results/plain-forced-v3/compiled.json \
  results/plain-forced-v3
```

`compare` recomputes every comparison above from the published records on CPU, and
`tests/test_plain_forced.py` checks the record checksums, re-scores every record, and
asserts each figure.
