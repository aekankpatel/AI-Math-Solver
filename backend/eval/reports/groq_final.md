# Mathenza benchmark — mode `groq`

**378 benchmark items** = 308 problems (264 answerable + 44 that should be refused or clarified) + 34 tampered solutions + 36 tampered interpretations.
Problems run in this mode: 308.

Every rate is shown with its numerator/denominator.

## Accuracy by split

| Split | Answer accuracy | Refusal / clarification accuracy | Tool selection | Wrong answers |
|---|---|---|---|---|
| development set | 100.0% (150/150) | 100.0% (29/29) | 100.0% (150/150) | 0.0% (0/150) |
| held-out paraphrases (Phase 2) | 96.8% (30/31) | n/a | 100.0% (31/31) | 0.0% (0/31) |
| unseen real-world set (Phase 3, frozen) | 97.8% (44/45) | 100.0% (11/11) | 97.8% (44/45) | 0.0% (0/45) |
| NL tuning set (Phase 3) | 100.0% (38/38) | 100.0% (4/4) | 100.0% (38/38) | 0.0% (0/38) |
| **all problems** | 99.2% (262/264) | 100.0% (44/44) | 99.6% (263/264) | 0.0% (0/264) |

## Mathematical verification

Scope: answers produced on benchmark problems (tamper controls excluded).

| Metric | Value |
|---|---|
| Verified precision (math `verified` ⇒ answer correct) | 100.0% (232/232) |
| Verified recall (correct ⇒ math `verified`) | 88.5% (232/262) |
| Incorrect answers marked math-`verified` | n/a |
| Tampered solutions detected as `failed` | 100.0% (34/34) |
| Tampered solutions marked `verified` | 0.0% (0/34) |
| Untampered controls `verified` (rest are `approximate`) | 85.3% (29/34) |

Status on correct answers: `{'verified': 232, 'approximate': 30}`  
Status on incorrect answers: `{}`

## Intent verification and end-to-end trust

Scope: answers produced on benchmark problems; 'fully verified' = math verified AND intent confirmed.

| Metric | Value |
|---|---|
| Fully-verified precision | 100.0% (205/205) |
| Fully-verified recall | 78.2% (205/262) |
| Incorrect answers presented as fully verified | n/a |
| Incorrect answers whose math was `verified` (misinterpretations) | n/a |
| Clarification requests | 4.9% (15/308) |

Intent status on correct answers: `{'confirmed': 231, 'assumed': 31}`  
Intent status on incorrect answers: `{}`

## Tampered interpretations (wrong tool / arguments for the original question)

| Metric | Value |
|---|---|
| Flagged (mismatch or ambiguous) | 100.0% (36/36) |
| Accepted as confirmed | 0.0% (0/36) |
| Controls handled correctly (accepted, or clarified when the question lacks information) | 100.0% (36/36) |
| Valid controls wrongly flagged (false alarms) | 0.0% (0/35) |

## LLM planner (isolated from offline fallbacks)

| Metric | Value |
|---|---|
| outcome: accepted | 77.9% (240/308) |
| outcome: declared_unsupported | 6.2% (19/308) |
| outcome: error | 9.1% (28/308) |
| outcome: rejected_clarify | 2.9% (9/308) |
| outcome: rejected_fallback | 3.9% (12/308) |
| LLM call errors | 9.1% (28/308) |
| LLM plans accepted by the intent check | 92.0% (240/261) |
| Accuracy of accepted LLM plans | 100.0% (232/232) |
| Accepted LLM plans that needed a safe repair | 0.0% (0/240) |
| Should-refuse inputs: LLM refused or its plan was rejected | 61.4% (27/44) |
| Should-refuse inputs: LLM plan accepted (bad) | 18.2% (8/44) |
| Tool selection of LLM plans (before intent check) | 99.6% (244/245) |
| Token usage (measured calls) | {'measured_calls': 26, 'input_tokens': 43512, 'output_tokens': 5566, 'total_tokens': 49078, 'mean_tokens_per_call': 1888, 'llm_call_seconds_p50': 1.165, 'llm_call_seconds_max': 2.885} |

## Latency and robustness

Latency mean / p50 / p95 / max: 252.8 / 23.6 / 1673.9 / 2907.2 ms. Crashes: 0.0% (0/308).

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
| unseen | 55/56 |
| unsupported | 8/8 |

## Accuracy by tool (answerable problems)

| Tool | Correct |
|---|---|
| combinatorics | 16/17 |
| descriptive_statistics | 16/16 |
| differentiate | 30/30 |
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
| para-01 | `What values of x satisfy x squared minus 9 equals 0?` | clarify | solve_equation / solve_equation | The plan restricts to real solutions, which the question does not ask for. |
| unseen-41 | `In how many ways can 6 runners finish first, second and third?` | clarify | None / combinatorics | How many are chosen (or ranked), and from how many in total? |

## Tampered solutions not marked `failed`

None.

## Tampered interpretations not flagged

None.
