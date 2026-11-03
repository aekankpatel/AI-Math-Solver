# MathForge benchmark — mode `groq`

**348 benchmark items** = 278 problems (238 answerable + 40 that should be refused or clarified) + 34 tampered solutions + 36 tampered interpretations.
Problems run in this mode: 278.

Every rate is shown with its numerator/denominator.

## Accuracy by split

| Split | Answer accuracy | Refusal / clarification accuracy | Tool selection | Wrong answers |
|---|---|---|---|---|
| development set | 100.0% (150/150) | 100.0% (29/29) | 100.0% (150/150) | 0.0% (0/150) |
| held-out paraphrases (Phase 2) | 96.8% (30/31) | n/a | 100.0% (31/31) | 0.0% (0/31) |
| unseen real-world set (Phase 3, frozen) | 97.8% (44/45) | 100.0% (11/11) | 97.8% (44/45) | 0.0% (0/45) |
| NL tuning set (Phase 3) | 100.0% (12/12) | n/a | 100.0% (12/12) | 0.0% (0/12) |
| **all problems** | 99.2% (236/238) | 100.0% (40/40) | 99.6% (237/238) | 0.0% (0/238) |

## Mathematical verification

Scope: answers produced on benchmark problems (tamper controls excluded).

| Metric | Value |
|---|---|
| Verified precision (math `verified` ⇒ answer correct) | 100.0% (210/210) |
| Verified recall (correct ⇒ math `verified`) | 89.0% (210/236) |
| Incorrect answers marked math-`verified` | n/a |
| Tampered solutions detected as `failed` | 100.0% (34/34) |
| Tampered solutions marked `verified` | 0.0% (0/34) |
| Untampered controls `verified` (rest are `approximate`) | 85.3% (29/34) |

Status on correct answers: `{'verified': 210, 'approximate': 26}`  
Status on incorrect answers: `{}`

## Intent verification and end-to-end trust

Scope: answers produced on benchmark problems; 'fully verified' = math verified AND intent confirmed.

| Metric | Value |
|---|---|
| Fully-verified precision | 100.0% (187/187) |
| Fully-verified recall | 79.2% (187/236) |
| Incorrect answers presented as fully verified | n/a |
| Incorrect answers whose math was `verified` (misinterpretations) | n/a |
| Clarification requests | 4.7% (13/278) |

Intent status on correct answers: `{'confirmed': 209, 'assumed': 27}`  
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
| outcome: accepted | 78.4% (218/278) |
| outcome: declared_unsupported | 6.5% (18/278) |
| outcome: error | 8.6% (24/278) |
| outcome: rejected_clarify | 2.5% (7/278) |
| outcome: rejected_fallback | 4.0% (11/278) |
| LLM call errors | 8.6% (24/278) |
| LLM plans accepted by the intent check | 92.4% (218/236) |
| Accuracy of accepted LLM plans | 100.0% (210/210) |
| Accepted LLM plans that needed a safe repair | 0.0% (0/218) |
| Should-refuse inputs: LLM refused or its plan was rejected | 60.0% (24/40) |
| Should-refuse inputs: LLM plan accepted (bad) | 20.0% (8/40) |
| Tool selection of LLM plans (before intent check) | 99.6% (221/222) |
| Token usage (measured calls) | {'measured_calls': 16, 'input_tokens': 26720, 'output_tokens': 2824, 'total_tokens': 29544, 'mean_tokens_per_call': 1846, 'llm_call_seconds_p50': 1.192, 'llm_call_seconds_max': 1.978} |

**INCOMPLETE RUN:** daily Groq quota reached (resets in 10m40.656s.); progress is cached - re-run the same command after the reset to resume (278/308 problems done)

## Latency and robustness

Latency mean / p50 / p95 / max: 211.4 / 28.6 / 1515.4 / 2685.2 ms. Crashes: 0.0% (0/278).

## Accuracy by category

| Category | Correct |
|---|---|
| algebra | 51/51 |
| ambiguous | 8/8 |
| calculus | 28/28 |
| dev_nl | 12/12 |
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
| combinatorics | 13/14 |
| descriptive_statistics | 14/14 |
| differentiate | 28/28 |
| expand_expression | 8/8 |
| factor_expression | 9/9 |
| integrate | 29/29 |
| limit | 17/17 |
| matrix_operation | 28/28 |
| probability_distribution | 30/30 |
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
