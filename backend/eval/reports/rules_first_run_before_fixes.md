# Mathenza benchmark — mode `rules`

Cases: 210 run (181 answerable, 29 should be refused) + 34 tampered solutions.

| Metric | Value |
|---|---|
| Final-answer accuracy | 91.2% |
| Refusal accuracy (edge / ambiguous / unsupported) | 100.0% |
| Overall accuracy | 92.4% |
| Tool-selection accuracy | 94.5% |
| — development set (150) accuracy / tool selection | 100.0% / 100.0% |
| — held-out paraphrases (31) accuracy / tool selection | 48.4% / 67.7% |
| Verified precision (verified ⇒ correct) | 99.4% |
| Verified recall (correct ⇒ verified) | 87.4% |
| False verification rate (incorrect marked verified) | 2.8% |
| Error detection rate (incorrect marked failed) | 94.4% |
| Tampered solutions detected | 34/34 |
| Latency mean / p50 / p95 / max (ms) | 26.5 / 9.8 / 63.4 / 685.6 |
| Crash rate | 0.0% |
| Unexpected refusal rate | 7.7% |
| Wrong-answer rate | 1.1% |
| Answered when it should refuse | 0.0% |

Verification status on correct outputs: `{'verified': 174, 'approximate': 25}`  
Verification status on incorrect outputs: `{'unverified': 1, 'verified': 1, 'failed': 34}`

## Accuracy by category

| Category | Correct |
|---|---|
| algebra | 51/51 |
| ambiguous | 8/8 |
| calculus | 28/28 |
| edge | 13/13 |
| integrals | 21/21 |
| linear_algebra | 17/17 |
| paraphrase | 15/31 |
| probability | 24/24 |
| statistics | 9/9 |
| unsupported | 8/8 |

## Accuracy by tool (answerable cases)

| Tool | Correct |
|---|---|
| combinatorics | 9/11 |
| descriptive_statistics | 11/11 |
| differentiate | 17/20 |
| expand_expression | 5/6 |
| factor_expression | 6/7 |
| integrate | 23/25 |
| limit | 13/14 |
| matrix_operation | 21/22 |
| probability_distribution | 18/20 |
| simplify_expression | 8/8 |
| solve_equation | 25/28 |
| solve_system | 9/9 |

## Failures

| Case | Problem | Outcome | Tool (planned/expected) | Detail |
|---|---|---|---|---|
| para-01 | `What values of x satisfy x squared minus 9 equals 0?` | rejected | None / solve_equation | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| para-02 | `Find x if 4x - 7 = 9` | answered | solve_equation / solve_equation | got [-2*sqrt(-I/f), 2*sqrt(-I/f)] |
| para-03 | `Find all real x such that x^2 = 16` | rejected | solve_equation / solve_equation | Unknown identifier 'real'. Use standard math functions. |
| para-06 | `Factorise x^2 - 7x + 12` | rejected | None / factor_expression | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| para-07 | `Multiply out (x + 3)(x - 3)` | rejected | None / expand_expression | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| para-08 | `What is the slope of y = x^3 at x = 2?` | rejected | solve_equation / differentiate | Unknown identifier 'slope'. Use standard math functions. |
| para-09 | `How fast is sin(x) changing at x = 0?` | rejected | solve_equation / differentiate | Unknown identifier 'fast'. Use standard math functions. |
| para-11 | `What's the derivative of cos(3x)?` | rejected | differentiate / differentiate | Expression contains disallowed characters: "'" |
| para-12 | `Compute the area under y = x^2 between x = 0 and x = 2` | rejected | integrate / integrate | Unknown identifier 'between'. Use standard math functions. |
| para-14 | `What's the antiderivative of 1/(x+1)?` | rejected | integrate / integrate | Expression contains disallowed characters: "'" |
| para-16 | `What does (x^3 - 8)/(x - 2) approach as x approaches 2?` | rejected | None / limit | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| para-21 | `Determine the rank of the matrix [[1, 2, 3], [2, 4, 6], [1, 0, 1]]` | answered | matrix_operation / matrix_operation | got 0 |
| para-26 | `Heights are normally distributed with mean 170 and standard deviation 10. What fraction are taller than 190?` | rejected | None / probability_distribution | What probability should be computed? e.g. P(X <= 3) or P(1 < X < 2). |
| para-28 | `Find P(Z < 1.5) for a standard normal variable Z` | rejected | None / probability_distribution | What probability should be computed? e.g. P(X <= 3) or P(1 < X < 2). |
| para-29 | `How many ways can 4 books be arranged on a shelf?` | rejected | None / combinatorics | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |
| para-30 | `In how many ways can a committee of 3 be chosen from 10 people?` | rejected | None / combinatorics | I couldn't recognise this as a supported problem (algebra, calculus, integrals, linear algebra, probability or statistics). Try rephrasing,  |

## Tampered solutions not marked `failed`

None.
