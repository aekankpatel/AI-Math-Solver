# MathForge benchmark — mode `oracle`

Cases: 196 run (181 answerable, 15 should be refused) + 34 tampered solutions.

| Metric | Value |
|---|---|
| Final-answer accuracy | 100.0% |
| Refusal accuracy (edge / ambiguous / unsupported) | 100.0% |
| Overall accuracy | 100.0% |
| Tool-selection accuracy | n/a |
| — development set (150) accuracy / tool selection | 100.0% / n/a |
| — held-out paraphrases (31) accuracy / tool selection | 100.0% / n/a |
| Verified precision (verified ⇒ correct) | 100.0% |
| Verified recall (correct ⇒ verified) | 87.9% |
| False verification rate (incorrect marked verified) | 0.0% |
| Error detection rate (incorrect marked failed) | 100.0% |
| Tampered solutions detected | 34/34 |
| Latency mean / p50 / p95 / max (ms) | 22.5 / 5.8 / 54.8 / 744.4 |
| Crash rate | 0.0% |
| Unexpected refusal rate | 0.0% |
| Wrong-answer rate | 0.0% |
| Answered when it should refuse | 0.0% |

Verification status on correct outputs: `{'verified': 189, 'approximate': 26}`  
Verification status on incorrect outputs: `{'failed': 34}`

## Accuracy by category

| Category | Correct |
|---|---|
| algebra | 51/51 |
| ambiguous | 2/2 |
| calculus | 28/28 |
| edge | 13/13 |
| integrals | 21/21 |
| linear_algebra | 17/17 |
| paraphrase | 31/31 |
| probability | 24/24 |
| statistics | 9/9 |

## Accuracy by tool (answerable cases)

| Tool | Correct |
|---|---|
| combinatorics | 11/11 |
| descriptive_statistics | 11/11 |
| differentiate | 20/20 |
| expand_expression | 6/6 |
| factor_expression | 7/7 |
| integrate | 25/25 |
| limit | 14/14 |
| matrix_operation | 22/22 |
| probability_distribution | 20/20 |
| simplify_expression | 8/8 |
| solve_equation | 28/28 |
| solve_system | 9/9 |

## Failures

None.

## Tampered solutions not marked `failed`

None.
