# Qwen3.5-4B full live-compiled sweep preregistration

Frozen before the first item in the 32-item compiled Qwen3.5-4B sweep on 2026-11-07.

This repeats the [2B sweep preregistration](babyagi-v3-full-live-compiled-preregistration.md)
on Qwen3.5-4B checkpoints. The suite, protocol, reasoning budget, decoding, bound, run
order, runner, decoder, evaluator, token accountant, scorer, and plan scorer are the 2B
sweep's, unchanged. Only the base model and the frozen plain comparison differ. The
authoritative preregistration is experiment E12 in the private BabyAGI Inside research
lab, locked before any E12 run; this document records its frozen inputs and adds no rule.

## Question

What is the exact end-to-end capability of the live compiled Qwen3.5-4B checkpoint on all
32 frozen v3 objectives, and how does it compare with plain Qwen3.5-4B?

## Motivation and what came before

In a preregistered host-simulator screen, Qwen3.5-4B completed 27/32 chains against
Qwen3.5-2B's 7/32 under the same `babyagi_ledger_pointer` protocol. A host simulation
says nothing about whether a compiled checkpoint executes correctly, so the compiled
sweep is the test. The 4B port was written after that screen, which motivated it, and no
suite item was used to develop or tune it. Before this freeze, each 4B checkpoint was
built, passed the forced-token oracle, and decoded one fresh objective that is not a suite
item. No suite item had been decoded on a compiled 4B checkpoint. The compiler code was
frozen at `78961097d7e6cfc30db6f13bcec68c5d29019b36` before this sweep was preregistered.

## Frozen inputs

- Suite: `benchmarks/babyagi-v3.json` (`babyagi-execution-gradient-v3`, SHA-256
  `e880e043f7ac412aaab938036cccd5f79305aed4cd0c13a7204e09fe07c1dff4`), 32 items, two
  per category at each length 2, 3, 5, and 8
- Model family: Qwen3.5-4B (Hugging Face revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, contents pinned in
  [`provenance/qwen3.5-4b.sha256`](../provenance/qwen3.5-4b.sha256)), bf16, MPS
- Compiler: commit `78961097d7e6cfc30db6f13bcec68c5d29019b36`
- Protocol: `babyagi_ledger_pointer`, 256 reasoning tokens per execute stage
- Decoding: greedy (`temperature = 0`), seed 0
- External safety bound: 7,000 emitted tokens per item; no retry on a bound hit
- Run order: ascending cycle count, then frozen suite order
- Frozen plain comparison: 32 plain Qwen3.5-4B records, 15/32 exact chains (records
  digest `93cdc89604278cb71b5d12321e7f0f2c08108c3a7801d576adce8495057751f9`). They come
  from the lab's model-scale screen and were produced by the same plain-arm runner
  (`qfs.end_to_end`, SHA-256 `8c1e55f6…`) and settings as the 2B sweep's plain records
  (5/32), which that screen reproduced 32/32 on 2B. Like the 2B plain records, they are
  not in this repository.
- Sweep sources: `qfs.compiled_sweep` hashes its runner, decoder, evaluator, accountant,
  scorer, statistics, and plan scorer into the sweep manifest. At this commit they hash
  identically to the 2B sweep manifest's `source_sha256`.

Checkpoint tree SHA-256 (`qfs.compiled_sweep._checkpoint_manifest`). Each checkpoint was
built twice from the same inputs, and both builds had this digest:

| cycles | checkpoint tree SHA-256 | bytes |
| ---: | --- | ---: |
| 2 | `bbc5aa8b074c4dfacdd801eb4088bcb1f074f632d601f8d6c70028413539e3d8` | 16,396,544,275 |
| 3 | `edbceb5f3fdccb49d7ec631a57d2ad92185d8d02edb4c075db307cabb688236b` | 16,396,547,696 |
| 5 | `8ab06101cc2c2bb3e55229a6e8225a3f361a3d0a96cf6f206f11c0a2c40f7105` | 16,396,554,538 |
| 8 | `913f236d2ea9649ac7f395534a4da3583ba35451b7adf8731818fd0e993757b3` | 16,396,564,801 |

## Execution contract, metrics, and reporting

As in the [2B preregistration](babyagi-v3-full-live-compiled-preregistration.md): each
item starts with a fresh qwenFS cache; the host supplies one objective prefill plus START
and then only feeds back greedy tokens until HALT or the bound; every completed item
record binds its run, evaluation, accounting, score, and plan artifacts by SHA-256.

Primary systems metrics: exact controller sequence and HALT, token-exact qwenFS reads,
all cycles and phases complete, exact disjoint accounting.

Primary capability metrics: exact complete-chain success over all 32 items, exact
transition accuracy, and the paired compiled-minus-plain chain difference with an exact
McNemar test and an item-bootstrap 95% interval. The paired result is read as an
improvement only if the interval's lower bound is above zero; an interval that includes
zero is reported as inconclusive.

Secondary: chains and transitions by category and length, a Wilson 95% interval for
compiled chain accuracy, the failure taxonomy, elapsed time and tokens per item, and two
descriptive paired comparisons over the same 32 items: against the compiled 2B sweep and
against the 4B host simulation. The exact JSON-ledger plan metric stays primary for plans
even at zero; the operation-signature plan score stays exploratory and post hoc.

Run every item once. Preserve every failure. Report the result whatever it is: 0/32,
equal to plain, inconclusive, or below the host simulation.

## Reproduction

```sh
scripts/build_checkpoints.sh 4b
uv run python -m qfs.compiled_sweep \
  benchmarks/babyagi-v3.json \
  results/qwen35-4b-full-live-compiled \
  --checkpoint-pattern 'build/babyagi-4b-ledger-pointer-b256-m{cycles}' \
  --plain-root PLAIN_4B_ROOT \
  --device mps --seed 0 --temperature 0 --limit 7000
```

`PLAIN_4B_ROOT` is a directory whose `plain/` holds the 32 frozen plain Qwen3.5-4B
records. A different plain root, checkpoint, source, or setting is a different
experiment and cannot resume this one.
