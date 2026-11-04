# Mathenza benchmark — mode `groq`

**95 benchmark items** = 25 problems (19 answerable + 6 that should be refused or clarified) + 34 tampered solutions + 36 tampered interpretations.
Problems run in this mode: 25.

Every rate is shown with its numerator/denominator. **Subset run: smoke (25 problems; not the full benchmark).**

## Accuracy by split

| Split | Answer accuracy | Refusal / clarification accuracy | Tool selection | Wrong answers |
|---|---|---|---|---|
| development set | 100.0% (3/3) | 100.0% (4/4) | 100.0% (3/3) | 0.0% (0/3) |
| held-out paraphrases (Phase 2) | 100.0% (7/7) | n/a | 100.0% (7/7) | 0.0% (0/7) |
| unseen real-world set (Phase 3, frozen) | 100.0% (8/8) | 100.0% (2/2) | 100.0% (8/8) | 0.0% (0/8) |
| NL tuning set (Phase 3) | 100.0% (1/1) | n/a | 100.0% (1/1) | 0.0% (0/1) |
| **all problems** | 100.0% (19/19) | 100.0% (6/6) | 100.0% (19/19) | 0.0% (0/19) |

## Mathematical verification

Scope: answers produced on benchmark problems (tamper controls excluded).

| Metric | Value |
|---|---|
| Verified precision (math `verified` ⇒ answer correct) | 100.0% (17/17) |
| Verified recall (correct ⇒ math `verified`) | 89.5% (17/19) |
| Incorrect answers marked math-`verified` | n/a |
| Tampered solutions detected as `failed` | 100.0% (34/34) |
| Tampered solutions marked `verified` | 0.0% (0/34) |
| Untampered controls `verified` (rest are `approximate`) | 85.3% (29/34) |

Status on correct answers: `{'verified': 17, 'approximate': 2}`  
Status on incorrect answers: `{}`

## Intent verification and end-to-end trust

Scope: answers produced on benchmark problems; 'fully verified' = math verified AND intent confirmed.

| Metric | Value |
|---|---|
| Fully-verified precision | 100.0% (15/15) |
| Fully-verified recall | 79.0% (15/19) |
| Incorrect answers presented as fully verified | n/a |
| Incorrect answers whose math was `verified` (misinterpretations) | n/a |
| Clarification requests | 4.0% (1/25) |

Intent status on correct answers: `{'confirmed': 17, 'assumed': 2}`  
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
| outcome: accepted | 72.0% (18/25) |
| outcome: declared_unsupported | 16.0% (4/25) |
| outcome: error | 8.0% (2/25) |
| outcome: rejected_clarify | 4.0% (1/25) |
| LLM call errors | 8.0% (2/25) |
| LLM plans accepted by the intent check | 94.7% (18/19) |
| Accuracy of accepted LLM plans | 100.0% (18/18) |
| Accepted LLM plans that needed a safe repair | 0.0% (0/18) |
| Should-refuse inputs: LLM refused or its plan was rejected | 83.3% (5/6) |
| Should-refuse inputs: LLM plan accepted (bad) | 0.0% (0/6) |
| Tool selection of LLM plans (before intent check) | 100.0% (18/18) |
| Token usage (measured calls) | {'measured_calls': 24, 'input_tokens': 40844, 'output_tokens': 4198, 'total_tokens': 45042, 'mean_tokens_per_call': 1877, 'llm_call_seconds_p50': 1.164, 'llm_call_seconds_max': 1.81} |

## Latency and robustness

Latency mean / p50 / p95 / max: 1265.0 / 1210.1 / 1833.8 / 1844.5 ms. Crashes: 0.0% (0/25).

## Accuracy by category

| Category | Correct |
|---|---|
| algebra | 1/1 |
| ambiguous | 1/1 |
| calculus | 1/1 |
| dev_nl | 1/1 |
| edge | 1/1 |
| integrals | 1/1 |
| paraphrase | 7/7 |
| unseen | 10/10 |
| unsupported | 2/2 |

## Accuracy by tool (answerable problems)

| Tool | Correct |
|---|---|
| combinatorics | 1/1 |
| descriptive_statistics | 1/1 |
| differentiate | 3/3 |
| expand_expression | 1/1 |
| factor_expression | 1/1 |
| integrate | 2/2 |
| limit | 1/1 |
| matrix_operation | 2/2 |
| probability_distribution | 3/3 |
| simplify_expression | 1/1 |
| solve_equation | 2/2 |
| solve_system | 1/1 |

## Failures

None.

## Tampered solutions not marked `failed`

None.

## Tampered interpretations not flagged

None.
