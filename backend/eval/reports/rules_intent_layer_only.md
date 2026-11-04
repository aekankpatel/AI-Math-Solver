# Mathora benchmark — mode `rules`

**336 benchmark items** = 266 problems (226 answerable + 40 that should be refused or clarified) + 34 tampered solutions + 36 tampered interpretations.
Problems run in this mode: 266.

Every rate is shown with its numerator/denominator.

## Accuracy by split

| Split | Answer accuracy | Refusal / clarification accuracy | Tool selection | Wrong answers |
|---|---|---|---|---|
| development set | 100.0% (150/150) | 100.0% (29/29) | 100.0% (150/150) | 0.0% (0/150) |
| held-out paraphrases (Phase 2) | 51.6% (16/31) | n/a | 67.7% (21/31) | 0.0% (0/31) |
| unseen real-world set (Phase 3, frozen) | 48.9% (22/45) | 100.0% (11/11) | 71.1% (32/45) | 0.0% (0/45) |
| **all problems** | 83.2% (188/226) | 100.0% (40/40) | 89.8% (203/226) | 0.0% (0/226) |

## Mathematical verification

Scope: answers produced on benchmark problems (tamper controls excluded).

| Metric | Value |
|---|---|
| Verified precision (math `verified` ⇒ answer correct) | 100.0% (165/165) |
| Verified recall (correct ⇒ math `verified`) | 87.8% (165/188) |
| Incorrect answers marked math-`verified` | n/a |
| Tampered solutions detected as `failed` | 100.0% (34/34) |
| Tampered solutions marked `verified` | 0.0% (0/34) |
| Untampered controls `verified` (rest are `approximate`) | 85.3% (29/34) |

Status on correct answers: `{'verified': 165, 'approximate': 23}`  
Status on incorrect answers: `{}`

## Intent verification and end-to-end trust

Scope: answers produced on benchmark problems; 'fully verified' = math verified AND intent confirmed.

| Metric | Value |
|---|---|
| Fully-verified precision | 100.0% (165/165) |
| Fully-verified recall | 87.8% (165/188) |
| Incorrect answers presented as fully verified | n/a |
| Incorrect answers whose math was `verified` (misinterpretations) | n/a |
| Clarification requests | 8.6% (23/266) |

Intent status on correct answers: `{'confirmed': 188}`  
Intent status on incorrect answers: `{}`

## Tampered interpretations (wrong tool / arguments for the original question)

| Metric | Value |
|---|---|
| Flagged (mismatch or ambiguous) | 100.0% (36/36) |
| Accepted as confirmed | 0.0% (0/36) |
| Controls handled correctly (accepted, or clarified when the question lacks information) | 100.0% (36/36) |
| Valid controls wrongly flagged (false alarms) | 0.0% (0/35) |

## Latency and robustness

Latency mean / p50 / p95 / max: 37.7 / 11.6 / 105.1 / 1361.0 ms. Crashes: 0.0% (0/266).

## Accuracy by category

| Category | Correct |
|---|---|
| algebra | 51/51 |
| ambiguous | 8/8 |
| calculus | 28/28 |
| edge | 13/13 |
| integrals | 21/21 |
| linear_algebra | 17/17 |
| paraphrase | 16/31 |
| probability | 24/24 |
| statistics | 9/9 |
| unseen | 33/56 |
| unsupported | 8/8 |

## Accuracy by tool (answerable problems)

| Tool | Correct |
|---|---|
| combinatorics | 9/14 |
| descriptive_statistics | 14/14 |
| differentiate | 19/25 |
| expand_expression | 6/7 |
| factor_expression | 6/8 |
| integrate | 25/29 |
| limit | 15/17 |
| matrix_operation | 28/28 |
| probability_distribution | 21/30 |
| simplify_expression | 9/9 |
| solve_equation | 27/35 |
| solve_system | 9/10 |

## Failures

| Case | Problem | Outcome | Tool (planned/expected) | Detail |
|---|---|---|---|---|
| para-01 | `What values of x satisfy x squared minus 9 equals 0?` | rejected | None / solve_equation | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| para-02 | `Find x if 4x - 7 = 9` | clarify | solve_equation / solve_equation | 'x if 4x - 7 = 9' does not match any equation in the question. |
| para-03 | `Find all real x such that x^2 = 16` | clarify | solve_equation / solve_equation | 'all real x such that x^2 = 16' does not match any equation in the question. |
| para-06 | `Factorise x^2 - 7x + 12` | rejected | None / factor_expression | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| para-07 | `Multiply out (x + 3)(x - 3)` | rejected | None / expand_expression | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| para-08 | `What is the slope of y = x^3 at x = 2?` | clarify | solve_equation / differentiate | The question asks for 'derivative' but the plan computes 'solve'. |
| para-09 | `How fast is sin(x) changing at x = 0?` | clarify | solve_equation / differentiate | The question asks for 'derivative' but the plan computes 'solve'. |
| para-11 | `What's the derivative of cos(3x)?` | rejected | differentiate / differentiate | Expression contains disallowed characters: "'" |
| para-12 | `Compute the area under y = x^2 between x = 0 and x = 2` | rejected | integrate / integrate | Unknown identifier 'between'. Use standard math functions. |
| para-14 | `What's the antiderivative of 1/(x+1)?` | rejected | integrate / integrate | Expression contains disallowed characters: "'" |
| para-16 | `What does (x^3 - 8)/(x - 2) approach as x approaches 2?` | rejected | None / limit | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| para-26 | `Heights are normally distributed with mean 170 and standard deviation 10. What fraction are taller than 190?` | rejected | None / probability_distribution | What probability should be computed? e.g. P(X <= 3) or P(1 < X < 2). |
| para-28 | `Find P(Z < 1.5) for a standard normal variable Z` | rejected | None / probability_distribution | What probability should be computed? e.g. P(X <= 3) or P(1 < X < 2). |
| para-29 | `How many ways can 4 books be arranged on a shelf?` | rejected | None / combinatorics | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| para-30 | `In how many ways can a committee of 3 be chosen from 10 people?` | rejected | None / combinatorics | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| unseen-01 | `A rectangle satisfies x^2 + 3x = 40. What positive value of x works?` | clarify | solve_equation / solve_equation | 'A rectangle satisfies x^2 + 3x = 40. What positive value of x works' does not match any equation in the question. |
| unseen-02 | `What are the real roots of x^3 - x^2 + x - 1 = 0?` | clarify | solve_equation / solve_equation | 'What are the real roots of x^3 - x^2 + x - 1 = 0' does not match any equation in the question. |
| unseen-03 | `Find the positive solution of t^2 - 2t - 3 = 0` | clarify | solve_equation / solve_equation | 'positive solution of t^2 - 2t - 3 = 0' does not match any equation in the question. |
| unseen-04 | `If 3(x - 2) = 2x + 5, what is x?` | clarify | solve_equation / solve_equation | 'If 3(x - 2) = 2x + 5, what is x' does not match any equation in the question. |
| unseen-06 | `Two numbers x and y satisfy x + y = 12 and x - y = 4. Find them.` | clarify | solve_system / solve_system | 'y satisfy x + y = 12' does not match any equation in the question. |
| unseen-07 | `For a 2023 physics project, find all t with -5t^2 + 20t = 0` | clarify | solve_equation / solve_equation | '2023 physics project, find all t with -5t^2 + 20t = 0' does not match any equation in the question. |
| unseen-09 | `What's the rate of change of f(t) = t^2 + 3t when t = 1?` | clarify | solve_equation / differentiate | The question asks for 'derivative' but the plan computes 'solve'. |
| unseen-10 | `Find the instantaneous rate of change of x^3 at x = -2` | clarify | solve_equation / differentiate | The question asks for 'derivative' but the plan computes 'solve'. |
| unseen-11 | `Find f''(x) for f(x) = x^3 + 2x^2` | rejected | differentiate / differentiate | Expression contains disallowed characters: "'" |
| unseen-14 | `Compute the definite integral of 2x + 1 over [0, 4]` | rejected | integrate / integrate | Expression contains disallowed characters: '[]' |
| unseen-15 | `Find the area under the curve y = sin(x) from x = 0 to x = pi/2` | rejected | integrate / integrate | Unknown identifier 'curve'. Use standard math functions. |
| unseen-18 | `What value does sin(3x)/x approach as x tends to 0?` | rejected | None / limit | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| unseen-30 | `A basketball player makes 70% of free throws. If she takes 10 shots, what is the probability she makes exactly 7?` | rejected | None / probability_distribution | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| unseen-31 | `Each of 12 light bulbs fails independently with probability 0.1. What is the probability that at most 1 fails?` | rejected | None / probability_distribution | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| unseen-33 | `Emails arrive at an average rate of 4 per minute (Poisson). What is the probability of fewer than 2 emails in a minute?` | rejected | None / probability_distribution | What probability should be computed? e.g. P(X <= 3) or P(1 < X < 2). |
| unseen-35 | `Test scores are normally distributed with mean 75 and standard deviation 8. What proportion scored below 91?` | rejected | None / probability_distribution | What probability should be computed? e.g. P(X <= 3) or P(1 < X < 2). |
| unseen-37 | `For a standard normal Z, find P(Z > -1)` | rejected | None / probability_distribution | What probability should be computed? e.g. P(X <= 3) or P(1 < X < 2). |
| unseen-38 | `A bus arrives uniformly at random between 0 and 20 minutes. What's the probability you wait more than 15 minutes?` | rejected | None / probability_distribution | What probability should be computed? e.g. P(X <= 3) or P(1 < X < 2). |
| unseen-39 | `A component's lifetime is exponential with rate 0.2 per year. What is the probability it lasts more than 5 years?` | rejected | None / probability_distribution | What probability should be computed? e.g. P(X <= 3) or P(1 < X < 2). |
| unseen-40 | `How many different 5-card hands can be dealt from a 52-card deck?` | rejected | None / combinatorics | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| unseen-41 | `In how many ways can 6 runners finish first, second and third?` | rejected | None / combinatorics | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| unseen-42 | `How many ways can 7 people line up?` | rejected | None / combinatorics | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| unseen-44 | `Factor completely: 3x^2 - 12` | rejected | factor_expression / factor_expression | Expression contains disallowed characters: ':' |

## Tampered solutions not marked `failed`

None.

## Tampered interpretations not flagged

None.
