# Mathora benchmark — mode `oracle`

**378 benchmark items** = 308 problems (264 answerable + 44 that should be refused or clarified) + 34 tampered solutions + 36 tampered interpretations.
Problems run in this mode: 279 (29 skipped: oracle mode cannot run natural-language-only cases (no canonical tool arguments)).

Every rate is shown with its numerator/denominator.

## Accuracy by split

| Split | Answer accuracy | Refusal / clarification accuracy | Tool selection | Wrong answers |
|---|---|---|---|---|
| development set | 100.0% (150/150) | 100.0% (15/15) | n/a | 0.0% (0/150) |
| held-out paraphrases (Phase 2) | 100.0% (31/31) | n/a | n/a | 0.0% (0/31) |
| unseen real-world set (Phase 3, frozen) | 100.0% (45/45) | n/a | n/a | 0.0% (0/45) |
| NL tuning set (Phase 3) | 100.0% (38/38) | n/a | n/a | 0.0% (0/38) |
| **all problems** | 100.0% (264/264) | 100.0% (15/15) | n/a | 0.0% (0/264) |

## Mathematical verification

Scope: answers produced on benchmark problems (tamper controls excluded).

| Metric | Value |
|---|---|
| Verified precision (math `verified` ⇒ answer correct) | 100.0% (234/234) |
| Verified recall (correct ⇒ math `verified`) | 88.6% (234/264) |
| Incorrect answers marked math-`verified` | n/a |
| Tampered solutions detected as `failed` | 100.0% (34/34) |
| Tampered solutions marked `verified` | 0.0% (0/34) |
| Untampered controls `verified` (rest are `approximate`) | 85.3% (29/34) |

Status on correct answers: `{'verified': 234, 'approximate': 30}`  
Status on incorrect answers: `{}`

## Intent verification and end-to-end trust

Scope: answers produced on benchmark problems; 'fully verified' = math verified AND intent confirmed.

| Metric | Value |
|---|---|
| Fully-verified precision | 100.0% (229/229) |
| Fully-verified recall | 86.7% (229/264) |
| Incorrect answers presented as fully verified | n/a |
| Incorrect answers whose math was `verified` (misinterpretations) | n/a |
| Clarification requests | 0.0% (0/279) |

Intent status on correct answers: `{'confirmed': 256, 'assumed': 8}`  
Intent status on incorrect answers: `{}`

## Tampered interpretations (wrong tool / arguments for the original question)

| Metric | Value |
|---|---|
| Flagged (mismatch or ambiguous) | 100.0% (36/36) |
| Accepted as confirmed | 0.0% (0/36) |
| Controls handled correctly (accepted, or clarified when the question lacks information) | 100.0% (36/36) |
| Valid controls wrongly flagged (false alarms) | 0.0% (0/35) |

## Latency and robustness

Latency mean / p50 / p95 / max: 33.1 / 12.4 / 90.3 / 1359.0 ms. Crashes: 0.0% (0/279).

## Accuracy by category

| Category | Correct |
|---|---|
| algebra | 51/51 |
| ambiguous | 2/2 |
| calculus | 28/28 |
| dev_nl | 38/38 |
| edge | 13/13 |
| integrals | 21/21 |
| linear_algebra | 17/17 |
| paraphrase | 31/31 |
| probability | 24/24 |
| statistics | 9/9 |
| unseen | 45/45 |

## Accuracy by tool (answerable problems)

| Tool | Correct |
|---|---|
| combinatorics | 17/17 |
| descriptive_statistics | 16/16 |
| differentiate | 30/30 |
| expand_expression | 8/8 |
| factor_expression | 9/9 |
| integrate | 33/33 |
| limit | 20/20 |
| matrix_operation | 31/31 |
| probability_distribution | 39/39 |
| simplify_expression | 10/10 |
| solve_equation | 40/40 |
| solve_system | 11/11 |

## Failures

None.

## Tampered solutions not marked `failed`

None.

## Tampered interpretations not flagged

None.
