# MathForge benchmark — mode `groq`

**316 benchmark items** = 246 problems (217 answerable + 29 that should be refused or clarified) + 34 tampered solutions + 36 tampered interpretations.
Problems run in this mode: 246.

Every rate is shown with its numerator/denominator.

## Accuracy by split

| Split | Answer accuracy | Refusal / clarification accuracy | Tool selection | Wrong answers |
|---|---|---|---|---|
| development set | 100.0% (150/150) | 96.5% (28/29) | 100.0% (150/150) | 0.0% (0/150) |
| held-out paraphrases (Phase 2) | 96.8% (30/31) | n/a | 100.0% (31/31) | 0.0% (0/31) |
| unseen real-world set (Phase 3, frozen) | 100.0% (36/36) | n/a | 100.0% (36/36) | 0.0% (0/36) |
| **all problems** | 99.5% (216/217) | 96.5% (28/29) | 100.0% (217/217) | 0.0% (0/217) |

## Mathematical verification

Scope: answers produced on benchmark problems (tamper controls excluded).

| Metric | Value |
|---|---|
| Verified precision (math `verified` ⇒ answer correct) | 99.5% (190/191) |
| Verified recall (correct ⇒ math `verified`) | 88.0% (190/216) |
| Incorrect answers marked math-`verified` | 100.0% (1/1) |
| Tampered solutions detected as `failed` | 100.0% (34/34) |
| Tampered solutions marked `verified` | 0.0% (0/34) |
| Untampered controls `verified` (rest are `approximate`) | 85.3% (29/34) |

Status on correct answers: `{'verified': 190, 'approximate': 26}`  
Status on incorrect answers: `{'verified': 1}`

## Intent verification and end-to-end trust

Scope: answers produced on benchmark problems; 'fully verified' = math verified AND intent confirmed.

| Metric | Value |
|---|---|
| Fully-verified precision | 100.0% (176/176) |
| Fully-verified recall | 81.5% (176/216) |
| Incorrect answers presented as fully verified | 0.0% (0/1) |
| Incorrect answers whose math was `verified` (misinterpretations) | 100.0% (1/1) |
| Clarification requests | 1.6% (4/246) |

Intent status on correct answers: `{'confirmed': 198, 'assumed': 18}`  
Intent status on incorrect answers: `{'assumed': 1}`

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
| outcome: accepted | 84.2% (207/246) |
| outcome: declared_unsupported | 5.7% (14/246) |
| outcome: error | 4.5% (11/246) |
| outcome: rejected_clarify | 1.2% (3/246) |
| outcome: rejected_fallback | 4.5% (11/246) |
| LLM call errors | 4.5% (11/246) |
| LLM plans accepted by the intent check | 93.7% (207/221) |
| Accuracy of accepted LLM plans | 100.0% (198/198) |
| Accepted LLM plans that needed a safe repair | 0.0% (0/207) |
| Should-refuse inputs: LLM refused or its plan was rejected | 55.2% (16/29) |
| Should-refuse inputs: LLM plan accepted (bad) | 31.0% (9/29) |
| Tool selection of LLM plans (before intent check) | 99.5% (209/210) |
| Token usage (measured calls) | {'measured_calls': 221, 'input_tokens': 368886, 'output_tokens': 40596, 'total_tokens': 409482, 'mean_tokens_per_call': 1853, 'llm_call_seconds_p50': 1.151, 'llm_call_seconds_max': 3.272} |

**INCOMPLETE RUN:** daily Groq quota reached (see the Groq console); progress is cached - re-run the same command after the reset to resume (246/308 problems done)

## Latency and robustness

Latency mean / p50 / p95 / max: 1198.3 / 1190.0 / 2015.2 / 3316.1 ms. Crashes: 0.0% (0/246).

## Accuracy by category

| Category | Correct |
|---|---|
| algebra | 51/51 |
| ambiguous | 8/8 |
| calculus | 28/28 |
| edge | 13/13 |
| integrals | 21/21 |
| linear_algebra | 17/17 |
| paraphrase | 30/31 |
| probability | 24/24 |
| statistics | 9/9 |
| unseen | 36/36 |
| unsupported | 7/8 |

## Accuracy by tool (answerable problems)

| Tool | Correct |
|---|---|
| combinatorics | 11/11 |
| descriptive_statistics | 14/14 |
| differentiate | 25/25 |
| expand_expression | 6/6 |
| factor_expression | 7/7 |
| integrate | 29/29 |
| limit | 17/17 |
| matrix_operation | 28/28 |
| probability_distribution | 27/27 |
| simplify_expression | 8/8 |
| solve_equation | 34/35 |
| solve_system | 10/10 |

## Failures

| Case | Problem | Outcome | Tool (planned/expected) | Detail |
|---|---|---|---|---|
| unsup-08 | `find the area of a triangle with sides 3, 4 and 5` | answered | simplify_expression / None | answered a problem that should have been refused |
| para-01 | `What values of x satisfy x squared minus 9 equals 0?` | clarify | solve_equation / solve_equation | The plan restricts to real solutions, which the question does not ask for. |

## Tampered solutions not marked `failed`

None.

## Tampered interpretations not flagged

None.
