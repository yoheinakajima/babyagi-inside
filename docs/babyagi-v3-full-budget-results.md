# BabyAGI v3 complete execution-budget curve

## Outcome

The complete frozen v3 single-step experiment confirms 256 model-owned
reasoning tokens as the best tested budget for Qwen3.5-2B.

The run covers 144 gold task/state transitions at each of five budgets, for 720
conditions total. All prompts, expected values, normalization, budgets, and the
selection rule were frozen before the pilot. Results are greedy bf16 inference
on MPS.

Suite SHA-256:
`e880e043f7ac412aaab938036cccd5f79305aed4cd0c13a7204e09fe07c1dff4`

Runner SHA-256:
`d28482826e1dabf08e3e6f90b7699e51a5dfc0d2a5645426ce4ac68d0ea54ae4`

## Full budget curve

| Fixed think tokens | Exact steps | Accuracy | Wilson 95% interval | Parse failures |
|---:|---:|---:|---:|---:|
| 16 | 72/144 | 50.0% | 41.9–58.1% | 1 |
| 32 | 61/144 | 42.4% | 34.6–50.5% | 1 |
| 64 | 79/144 | 54.9% | 46.7–62.8% | 0 |
| 128 | 115/144 | 79.9% | 72.6–85.6% | 0 |
| 256 | 126/144 | 87.5% | 81.1–91.9% | 0 |

The curve is not monotonic at small budgets: 32 tokens is significantly worse
than 16 on these paired transitions. Fixed-span reasoning is an intervention on
the model's autoregressive trajectory, not simply “more compute,” so extra
tokens can move the answer into a worse state. The large gains begin between 64
and 128 tokens.

## Category results

| Budget | Literal | Arithmetic | Structured | Language editing |
|---:|---:|---:|---:|---:|
| 16 | 26/36 (72.2%) | 15/36 (41.7%) | 17/36 (47.2%) | 14/36 (38.9%) |
| 32 | 25/36 (69.4%) | 11/36 (30.6%) | 12/36 (33.3%) | 13/36 (36.1%) |
| 64 | 25/36 (69.4%) | 17/36 (47.2%) | 19/36 (52.8%) | 18/36 (50.0%) |
| 128 | 28/36 (77.8%) | 33/36 (91.7%) | 29/36 (80.6%) | 25/36 (69.4%) |
| 256 | 33/36 (91.7%) | 34/36 (94.4%) | 31/36 (86.1%) | 28/36 (77.8%) |

At 256 tokens, arithmetic and literal transformations exceed 90%. Structured
operations reach 86.1%, and constrained language editing reaches 77.8%. The
strict broader target of at least 90% on arithmetic and structured execution is
therefore not fully met.

## Results by source-chain length

Each record is still an isolated gold step. “Length” means the length of the
source objective whose context was shown to the model; it does not measure
end-to-end state carry.

| Budget | Length 2 | Length 3 | Length 5 | Length 8 |
|---:|---:|---:|---:|---:|
| 16 | 14/16 (87.5%) | 14/24 (58.3%) | 16/40 (40.0%) | 28/64 (43.8%) |
| 32 | 10/16 (62.5%) | 14/24 (58.3%) | 16/40 (40.0%) | 21/64 (32.8%) |
| 64 | 13/16 (81.2%) | 13/24 (54.2%) | 21/40 (52.5%) | 32/64 (50.0%) |
| 128 | 14/16 (87.5%) | 23/24 (95.8%) | 28/40 (70.0%) | 50/64 (78.1%) |
| 256 | 14/16 (87.5%) | 22/24 (91.7%) | 34/40 (85.0%) | 56/64 (87.5%) |

The non-monotonic length-two result is another warning against interpreting
token budget as a simple scale variable. The 256-token arm is substantially
more stable on length-five and length-eight contexts.

## Paired comparisons

| Comparison | Accuracy delta | Improvements | Regressions | Exact McNemar p | Item-clustered bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| 16 → 32 | −7.6 points | 6 | 17 | 0.0347 | −13.5 to −2.0 points |
| 32 → 64 | +12.5 points | 28 | 10 | 0.0051 | +4.7 to +20.0 points |
| 64 → 128 | +25.0 points | 45 | 9 | <0.000001 | +14.3 to +35.5 points |
| 128 → 256 | +7.6 points | 17 | 6 | 0.0347 | 0.0 to +16.1 points |

McNemar uses paired transitions. The bootstrap resamples the 32 objectives as
clusters and retains all steps within each sampled objective. The 128-to-256
gain is positive by the exact paired test, but its clustered interval touches
zero; report it as modest evidence rather than a settled universal effect.

## Decision

Freeze 256 tokens as the end-to-end protocol budget. Keep 128 as the compact
ablation checkpoint.

The data support these claims:

- budget starvation was a major cause of the original execution failures;
- 128 tokens is enough for strong arithmetic but not broad execution;
- 256 tokens produces the best tested aggregate and category results;
- remaining failures are semantic and template-sensitive, not parser failures;
- 2B is adequate for the compilation/equivalence paper, but not a broadly
  reliable agent executor.

Do not tune the prompt against v3. The next experiment uses the frozen 256-token
protocol to compare plain chat, the host ledger loop, and compiled execution.

## Evidence

- `results/execution-budget-v3-full/summary.json`
- `results/execution-budget-v3-full/analysis.json`
- `results/execution-budget-v3-full/budget-0016/`
- `results/execution-budget-v3-full/budget-0032/`
- `results/execution-budget-v3-full/budget-0064/`
- `results/execution-budget-v3-full/budget-0128/`
- `results/execution-budget-v3-full/budget-0256/`

The directory contains one checksum-bound JSON record per transition and is
safe to audit or resume.
