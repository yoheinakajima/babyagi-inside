# Budget-forced plain baseline preregistration

Frozen on 2026-11-08 (01:11:56 UTC) before the first run, which started at 01:12:11 UTC.

The authoritative preregistration is experiment E14 in the private BabyAGI Inside research
lab (prereg SHA-256 `c2ce8f5b3f386ce9abaabc468154201c03f9842a6b197eb9b991a29f7cf4df43`,
config SHA-256 `d6b8211c612cb22289e1c2071fe295c9759ab0117b14d0d4ffd52846206a7974`, lock
commit `834dcc91c5f7734b1b06289209bdef61f5c3f08d`). The lock time is self-reported: the lab
repository is private, and no third party timestamped it. This document records its
frozen inputs and rules and adds none.

## Question

If plain Qwen is made to answer, with its thinking closed at a cap and an answer budget of
its own, and its token budget matches the compiled loop's, does the compiled loop still
complete more of the 32 frozen v3 chains than plain Qwen? Asked for Qwen3.5-2B and
Qwen3.5-4B.

## Why

The plain comparison in the 2B release (5/32) and the 4B sweep (15/32) is one
thinking-enabled turn with a ceiling of N x 256 + 256 new tokens. On both models every
plain run reached that ceiling and no run closed its thinking, so every plain score came
from JSON the parser found inside unfinished reasoning; 24/32 (2B) and 13/32 (4B) runs gave
no parseable result. That comparison measures plain Qwen running out of budget mid-thought
as much as it measures planning or execution.

## Hypotheses

- H-4B: under F2, compiled 4B completes more chains than budget-forced plain 4B: the paired
  difference's item-bootstrap 95% interval has its lower bound above zero.
- H-2B: no directional prediction.
- Predictions that can fail: under every arm, plain Qwen completes more chains than the
  one-turn arm on the same model (5/32, 15/32); and most runs are force-closed.

## Arms

Stock Qwen3.5-2B and Qwen3.5-4B (contents pinned in [`provenance/`](../provenance/)), bf16
on MPS, greedy, the one-turn arm's prompt (`qfs.ceiling.run_plain`'s, verbatim), thinking
enabled. Generation runs until Qwen emits `</think>` or reaches the thinking cap. In the
second case the host appends `"\n</think>\n\n"` (tokens 198, 248069, 271). Qwen then writes
its answer until end of turn or the answer cap. Parsing is `qfs.ceiling.parse_plain` over
the whole generation and scoring is `qfs.benchmark.score_item`. N is the item's cycle
count; T is the compiled run's total emitted tokens on that item and model, forced tokens
included.

| Arm | Thinking cap | Answer cap | Total cap | Matches |
| --- | --- | --- | --- | --- |
| F1 | N x 256 | 256 | N x 256 + 256 | the one-turn ceiling |
| **F2 (primary)** | (N + 1) x 256 | N x 256 | (2N + 1) x 256 | the loop's model-token allowance: N reasoning spans and N + 1 writes |
| F3 | T - N x 256 | N x 256 | T | the compiled run's own total |

The comparison arm is the validated live compiled sweeps: 2B (summary SHA-256
`47eaf336…`, 8/32) and 4B (`8b0ee42d…`, 28/32), per item chain verdict (benchmark pass
and HALT) and token total, frozen in
[`results/plain-forced-v3/compiled.json`](../results/plain-forced-v3/compiled.json).

## Items, metrics, and analysis

All 32 items of `benchmarks/babyagi-v3.json` (SHA-256 `e880e043…`), in suite order, once
per arm per model. Nothing is added, dropped, or tuned.

Primary, per model: the paired chain difference, compiled minus F2 plain, with the
item-bootstrap 95% interval (10,000 resamples, seed 0) and exact McNemar p from
`qfs.end_to_end._paired_summary`. It is read SUPPORTED if the lower bound is above zero,
CONTRADICTED if the upper bound is below zero, and INCONCLUSIVE otherwise. H-4B is decided
by the 4B F2 comparison alone; the 2B F2 comparison is reported as it falls. F1 and F3 are
secondary and descriptive. No multiplicity correction.

Reporting commitment: whatever the result, F2 becomes the headline plain comparison
wherever the compiled loop is compared with plain Qwen (the README and docs, the 4B
pull request, the preprint, the site, and the journal revision). The one-turn arm stays
reported, described as truncated before any answer.

## Stop conditions and budget

Every item runs once per arm; an interrupted step resumes from records whose identity
(model, arm, item, caps, suite and runner digests) matches. A record is never edited or
dropped. Apple M5, 32 GB; 3 hours per 2B step, 6 per 4B step.

## What happened before this lock

The frozen one-turn and compiled records were read to count runs that hit the ceiling,
closed their thinking, or failed to parse. No arm setting was chosen from plain or
compiled outputs. The runner was checked once on 2B with one objective that is not a
suite item. No suite item had been decoded under budget forcing.

Results: [plain-forced-v3-results.md](plain-forced-v3-results.md).
