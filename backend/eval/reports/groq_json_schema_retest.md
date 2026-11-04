# Mathora benchmark — mode `groq`

**77 benchmark items** = 7 problems (3 answerable + 4 that should be refused or clarified) + 34 tampered solutions + 36 tampered interpretations.
Problems run in this mode: 7.

Every rate is shown with its numerator/denominator.

## Accuracy by split

| Split | Answer accuracy | Refusal / clarification accuracy | Tool selection | Wrong answers |
|---|---|---|---|---|
| development set | 100.0% (3/3) | 100.0% (4/4) | 100.0% (3/3) | 0.0% (0/3) |
| **all problems** | 100.0% (3/3) | 100.0% (4/4) | 100.0% (3/3) | 0.0% (0/3) |

## Mathematical verification

Scope: answers produced on benchmark problems (tamper controls excluded).

| Metric | Value |
|---|---|
| Verified precision (math `verified` ⇒ answer correct) | 100.0% (3/3) |
| Verified recall (correct ⇒ math `verified`) | 100.0% (3/3) |
| Incorrect answers marked math-`verified` | n/a |
| Tampered solutions detected as `failed` | 100.0% (34/34) |
| Tampered solutions marked `verified` | 0.0% (0/34) |
| Untampered controls `verified` (rest are `approximate`) | 85.3% (29/34) |

Status on correct answers: `{'verified': 3}`  
Status on incorrect answers: `{}`

## Intent verification and end-to-end trust

Scope: answers produced on benchmark problems; 'fully verified' = math verified AND intent confirmed.

| Metric | Value |
|---|---|
| Fully-verified precision | 100.0% (3/3) |
| Fully-verified recall | 100.0% (3/3) |
| Incorrect answers presented as fully verified | n/a |
| Incorrect answers whose math was `verified` (misinterpretations) | n/a |
| Clarification requests | 14.3% (1/7) |

Intent status on correct answers: `{'confirmed': 3}`  
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
| outcome: accepted | 71.4% (5/7) |
| outcome: declared_unsupported | 14.3% (1/7) |
| outcome: error | 14.3% (1/7) |
| LLM call errors | 14.3% (1/7) |
| LLM plans accepted by the intent check | 100.0% (5/5) |
| Accuracy of accepted LLM plans | 100.0% (3/3) |
| Accepted LLM plans that needed a safe repair | 0.0% (0/5) |
| Should-refuse inputs: LLM refused or its plan was rejected | 25.0% (1/4) |
| Should-refuse inputs: LLM plan accepted (bad) | 50.0% (2/4) |
| Tool selection of LLM plans (before intent check) | 100.0% (3/3) |
| Token usage (measured calls) | {'measured_calls': 1, 'input_tokens': 2046, 'output_tokens': 299, 'total_tokens': 2345, 'mean_tokens_per_call': 2345, 'llm_call_seconds_p50': 1.395, 'llm_call_seconds_max': 1.395} |

**INCOMPLETE RUN:** daily Groq quota reached (resets in 13m59.808s.); progress is cached - re-run the same command after the reset to resume (7/28 problems done)

## Latency and robustness

Latency mean / p50 / p95 / max: 318.6 / 22.3 / 1404.4 / 1404.4 ms. Crashes: 0.0% (0/7).

## Accuracy by category

| Category | Correct |
|---|---|
| algebra | 1/1 |
| edge | 2/2 |
| linear_algebra | 1/1 |
| probability | 1/1 |
| unsupported | 2/2 |

## Accuracy by tool (answerable problems)

| Tool | Correct |
|---|---|
| combinatorics | 1/1 |
| matrix_operation | 1/1 |
| solve_equation | 1/1 |

## Failures

None.

## Tampered solutions not marked `failed`

None.

## Tampered interpretations not flagged

None.
