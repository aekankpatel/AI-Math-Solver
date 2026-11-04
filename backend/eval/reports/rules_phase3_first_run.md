# Mathora benchmark — mode `rules`

**378 benchmark items** = 308 problems (264 answerable + 44 that should be refused or clarified) + 34 tampered solutions + 36 tampered interpretations.
Problems run in this mode: 308.

Every rate is shown with its numerator/denominator.

## Accuracy by split

| Split | Answer accuracy | Refusal / clarification accuracy | Tool selection | Wrong answers |
|---|---|---|---|---|
| development set | 100.0% (150/150) | 100.0% (29/29) | 100.0% (150/150) | 0.0% (0/150) |
| held-out paraphrases (Phase 2) | 96.8% (30/31) | n/a | 96.8% (30/31) | 0.0% (0/31) |
| unseen real-world set (Phase 3, frozen) | 93.3% (42/45) | 90.9% (10/11) | 97.8% (44/45) | 4.4% (2/45) |
| NL tuning set (Phase 3) | 100.0% (38/38) | 100.0% (4/4) | 100.0% (38/38) | 0.0% (0/38) |
| **all problems** | 98.5% (260/264) | 97.7% (43/44) | 99.2% (262/264) | 0.8% (2/264) |

## Mathematical verification

Scope: answers produced on benchmark problems (tamper controls excluded).

| Metric | Value |
|---|---|
| Verified precision (math `verified` ⇒ answer correct) | 99.1% (231/233) |
| Verified recall (correct ⇒ math `verified`) | 88.8% (231/260) |
| Incorrect answers marked math-`verified` | 66.7% (2/3) |
| Tampered solutions detected as `failed` | 100.0% (34/34) |
| Tampered solutions marked `verified` | 0.0% (0/34) |
| Untampered controls `verified` (rest are `approximate`) | 85.3% (29/34) |

Status on correct answers: `{'verified': 231, 'approximate': 29}`  
Status on incorrect answers: `{'unverified': 1, 'verified': 2}`

## Intent verification and end-to-end trust

Scope: answers produced on benchmark problems; 'fully verified' = math verified AND intent confirmed.

| Metric | Value |
|---|---|
| Fully-verified precision | 100.0% (176/176) |
| Fully-verified recall | 67.7% (176/260) |
| Incorrect answers presented as fully verified | 0.0% (0/3) |
| Incorrect answers whose math was `verified` (misinterpretations) | 66.7% (2/3) |
| Clarification requests | 6.5% (20/308) |

Intent status on correct answers: `{'confirmed': 197, 'assumed': 63}`  
Intent status on incorrect answers: `{'assumed': 3}`

## Tampered interpretations (wrong tool / arguments for the original question)

| Metric | Value |
|---|---|
| Flagged (mismatch or ambiguous) | 100.0% (36/36) |
| Accepted as confirmed | 0.0% (0/36) |
| Controls handled correctly (accepted, or clarified when the question lacks information) | 100.0% (36/36) |
| Valid controls wrongly flagged (false alarms) | 0.0% (0/35) |

## Latency and robustness

Latency mean / p50 / p95 / max: 36.9 / 18.3 / 92.6 / 1073.8 ms. Crashes: 0.0% (0/308).

## Accuracy by category

| Category | Correct |
|---|---|
| algebra | 51/51 |
| ambiguous | 8/8 |
| calculus | 28/28 |
| dev_nl | 42/42 |
| edge | 13/13 |
| integrals | 21/21 |
| linear_algebra | 17/17 |
| paraphrase | 30/31 |
| probability | 24/24 |
| statistics | 9/9 |
| unseen | 52/56 |
| unsupported | 8/8 |

## Accuracy by tool (answerable problems)

| Tool | Correct |
|---|---|
| combinatorics | 15/17 |
| descriptive_statistics | 16/16 |
| differentiate | 29/30 |
| expand_expression | 8/8 |
| factor_expression | 9/9 |
| integrate | 33/33 |
| limit | 20/20 |
| matrix_operation | 31/31 |
| probability_distribution | 39/39 |
| simplify_expression | 10/10 |
| solve_equation | 39/40 |
| solve_system | 11/11 |

## Failures

| Case | Problem | Outcome | Tool (planned/expected) | Detail |
|---|---|---|---|---|
| para-01 | `What values of x satisfy x squared minus 9 equals 0?` | rejected | None / solve_equation | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| unseen-11 | `Find f''(x) for f(x) = x^3 + 2x^2` | answered | differentiate / differentiate | got 0 |
| unseen-40 | `How many different 5-card hands can be dealt from a 52-card deck?` | rejected | None / combinatorics | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| unseen-41 | `In how many ways can 6 runners finish first, second and third?` | answered | combinatorics / combinatorics | got 720 |
| unseen-56 | `How many ways can 3 people be chosen from a group?` | answered | combinatorics / None | answered a problem that should have been refused |

## Tampered solutions not marked `failed`

None.

## Tampered interpretations not flagged

None.
