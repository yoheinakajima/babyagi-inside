# BabyAGI Inside release checklist

## Done in `release-prep`

- [x] Work isolated in the `babyagi-release` worktree on branch `release-prep`.
- [x] Supplied single-file site copied byte-for-byte to `site/index.html`; SHA-256 `acc3254643aa73e77735e649a0f61e7f7bbb183d8e1fe03d2ba57038db662583`.
- [x] Public README documents the thesis, three-line host loop, checkpoint layout, exact CLI command forms, evidence boundary, credits, and licensing split.
- [x] Repository MIT license retains the upstream qwenFS/TheVixhal notice.
- [x] NOTICE separates MIT repository code from Apache-2.0 Qwen-derived weights.
- [x] Citation metadata added with `TODO(zenodo)`.
- [x] Ten completed live traces exported into the public compact schema without rerunning a model or scorer.
- [x] Structured failures preserved and clearly labeled; the trace set is not presented as an accuracy sample.
- [x] No push or publication performed.

## Waits on the frozen sweep

- [ ] `TODO(sweep)`: validate all 32 checksum-bound live records and final systems/accounting totals.
- [ ] `TODO(sweep)`: add exact compiled accuracy, transition accuracy, category/length breakdowns, paired plain-Qwen result, effect size, and confidence intervals.
- [ ] `TODO(sweep)`: disclose any controller, filesystem, accounting, bound, or process exception.
- [ ] `TODO(sweep)`: update the site and README headline only from the frozen final summary; do not tune prompts, checkpoints, settings, or scorers.
- [ ] `TODO(sweep)`: choose any additional public representative traces using a documented rule, rather than cherry-picking.

## Decisions for Yohei

- [ ] **Repository name:** recommend `babyagi-inside`; the current Git remotes do not identify a BabyAGI Inside public repository, so both `[REPO URL]` placeholders in the supplied site remain unchanged.
- [ ] **License:** recommend MIT for repository code, preserving the qwenFS notice, with Qwen-derived checkpoints explicitly remaining Apache-2.0. Have counsel confirm before distributing weights if commercial or sublicensing considerations matter.
- [ ] **Checkpoint hosting:** recommend Hugging Face Hub for immutable, versioned large checkpoints plus manifests/checksums; keep rebuild scripts and small metadata in GitHub. A scripts-to-rebuild-only release is smaller and avoids hosting roughly 9 GB per checkpoint, but is less reproducible because compilation is costly and the exact base revision/compiler commit still needs verification.
- [ ] **Release scope:** decide whether to publish all four checkpoints (2/3/5/8 cycles), one showcase checkpoint plus rebuild scripts, or scripts/manifests only.
- [ ] **Provenance:** pin the exact historical Qwen3.5-2B revision and compiler source commit. Current trace metadata verifies settings but not those two immutable identifiers (`TODO(verify)`).
- [ ] **DOI and canonical URLs:** create the Zenodo record, choose the public repository URL, and decide the canonical site URL before replacing placeholders.
- [ ] **Site wording:** the supplied HTML says stage order is reconstructed; the public trace export intentionally does not reconstruct missing flat-token `start`/`end` offsets and uses `null`. Keep the requested byte-identical site for now, but reconcile that wording before publication if it is meant to describe these JSON files.
- [ ] **Final validation:** perform a link check and browser review after URLs and sweep-dependent numbers are filled.

## Open technical evidence

- [ ] CUDA circuit-oracle check.
- [ ] Reproducible timing measurements.
- [ ] Larger frozen capability suite if a capability claim beyond the systems result is desired.
