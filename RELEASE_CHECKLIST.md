# BabyAGI Inside release checklist

## Done in `release-prep`

- [x] Work isolated in the `babyagi-release` worktree on branch `release-prep`.
- [x] Supplied single-file site retained at `site/index.html`; only its two repository URL placeholders were filled, and `site/CNAME` was added separately.
- [x] Public README documents the thesis, three-line host loop, checkpoint layout, exact CLI command forms, evidence boundary, credits, and licensing split.
- [x] Repository MIT license retains the upstream qwenFS/TheVixhal notice.
- [x] NOTICE separates MIT repository code from Apache-2.0 Qwen-derived weights.
- [x] Citation metadata added with `TODO(zenodo)`.
- [x] Ten completed live traces exported into the public compact schema without rerunning a model or scorer.
- [x] Structured failures preserved and clearly labeled; the trace set is not presented as an accuracy sample.
- [x] Private repository created at `https://github.com/yoheinakajima/babyagi-inside`.
- [x] Checkpoint weights excluded; local rebuild script covers 2, 3, 5, and 8 cycles.
- [x] GitHub Pages workflow and custom-domain file prepared without enabling Pages.

## Waits on the frozen sweep

- [ ] `TODO(sweep)`: validate all 32 checksum-bound live records and final systems/accounting totals.
- [ ] `TODO(sweep)`: add exact compiled accuracy, transition accuracy, category/length breakdowns, paired plain-Qwen result, effect size, and confidence intervals.
- [ ] `TODO(sweep)`: disclose any controller, filesystem, accounting, bound, or process exception.
- [ ] `TODO(sweep)`: update the site and README headline only from the frozen final summary; do not tune prompts, checkpoints, settings, or scorers.
- [ ] `TODO(sweep)`: choose any additional public representative traces using a documented rule, rather than cherry-picking.

## After sweep

- [ ] `/usr/bin/time -l scripts/build_checkpoints.sh m5` — record wall time + max RSS in README.

## Decisions for Yohei

- [ ] **Public launch:** make repo public → Settings › Pages › Source: GitHub Actions → DNS CNAME babyagi → yoheinakajima.github.io.
- [ ] **License:** recommend MIT for repository code, preserving the qwenFS notice, with Qwen-derived checkpoints explicitly remaining Apache-2.0. Have counsel confirm before distributing weights if commercial or sublicensing considerations matter.
- [ ] **Checkpoint hosting:** recommend Hugging Face Hub for immutable, versioned large checkpoints plus manifests/checksums; keep rebuild scripts and small metadata in GitHub. A scripts-to-rebuild-only release is smaller and avoids hosting roughly 9 GB per checkpoint, but is less reproducible because compilation is costly and the exact base revision/compiler commit still needs verification.
- [ ] **Release scope:** decide whether to publish all four checkpoints (2/3/5/8 cycles), one showcase checkpoint plus rebuild scripts, or scripts/manifests only.
- [x] **Model provenance:** exact local Qwen3.5-2B contents are pinned in `provenance/qwen3.5-2b.sha256`; the upstream revision ID was not recorded. The qwenFS-derived compiler source is `48f4f45b6269bf8a0f27ea954a05b2caced929a6`.
- [ ] **Build resources:** after the sweep, run the command above and record checkpoint build time and peak memory.
- [ ] **DOI:** create the Zenodo record and replace the DOI placeholder.
- [ ] **Site wording:** the supplied HTML says stage order is reconstructed; the public trace export intentionally does not reconstruct missing flat-token `start`/`end` offsets and uses `null`. Keep the requested byte-identical site for now, but reconcile that wording before publication if it is meant to describe these JSON files.
- [ ] **Final validation:** perform a link check and browser review after URLs and sweep-dependent numbers are filled.

## Open technical evidence

- [ ] CUDA circuit-oracle check.
- [ ] Reproducible timing measurements.
- [ ] Larger frozen capability suite if a capability claim beyond the systems result is desired.
