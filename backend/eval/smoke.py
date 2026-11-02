"""Stratified 25-question smoke-test subset for real-LLM (Groq) runs.

Chosen by hand, in advance, from the existing benchmark by case id. The
questions themselves are not copied or modified, so smoke-test results are
always computed on the same text and ground truth as the full benchmark.
It is a quick end-to-end health check before spending free-tier quota on all
308 problems, NOT a substitute for the full benchmark: never report smoke-test
accuracy as benchmark accuracy.

Strata (test-enforced in tests/test_groq_eval.py):
* all 12 tools, at least once among the answerable questions
* every split: development, held-out paraphrases, unseen, NL tuning
* natural wording / word problems (held-out, unseen, NL tuning)
* constraints and misreading traps ("positive", "Determine the rank", f'', slope)
* 6 should-refuse inputs: missing information, ambiguity, unsupported, unsafe, unsolvable
"""

SMOKE_IDS: tuple[str, ...] = (
    # --- answerable (19): tool coverage, natural wording, traps
    "unseen-01",  # solve_equation, "positive" constraint
    "para-02",  # solve_equation, "Find x if 4x - 7 = 9"
    "unseen-06",  # solve_system, word problem
    "simp-02",  # simplify_expression
    "para-06",  # factor_expression, "Factorise"
    "nl-08",  # expand_expression, "Please multiply out"
    "diff-13",  # differentiate at a point
    "unseen-11",  # differentiate, f''(x) notation
    "para-08",  # differentiate, "slope of y = x^3 at x = 2"
    "unseen-15",  # integrate, "area under the curve"
    "dint-05",  # integrate, improper integral
    "para-16",  # limit, "approach as x approaches 2"
    "para-21",  # matrix_operation, "Determine the rank" (determinant trap)
    "unseen-25",  # matrix_operation, solve Ax = b
    "unseen-27",  # descriptive_statistics, word problem
    "unseen-33",  # probability (Poisson), "fewer than 2" strictness
    "para-26",  # probability (normal), "taller than 190"
    "unseen-30",  # probability (binomial), "70% ... exactly 7"
    "para-30",  # combinatorics, committee
    # --- should be refused or clarified (6)
    "unseen-46",  # "Solve x + y = 10": several unknowns, one equation
    "unseen-51",  # normal distribution without a standard deviation
    "amb-02",  # "determinant" with no matrix
    "unsup-02",  # differential equation (unsupported)
    "unsup-05",  # code-injection attempt
    "edge-05",  # inverse of a singular matrix
)
