# Mathenza

**A maths solver that shows its working, then checks it.** You type a question in plain text or LaTeX. Mathenza works
out what is being asked, computes the answer exactly with SymPy, NumPy and SciPy, and confirms it a second, independent
way before showing it. A language model (Groq) may help *read* the question, but it never does the arithmetic.

![Mathenza](docs/screenshot.png)

## What it can answer

| Area | Examples |
|---|---|
| Algebra | `Solve x^2 - 5x + 6 = 0`, `x + 2y = 7, 3x - y = 7`, `factor x^3 - 8`, `\|2x - 3\| = 5`, `evaluate x^3 - 3x at x = 2` |
| Calculus | `derivative of x^3 sin(x) at x = 1`, `limit of sin(x)/x as x -> 0`, `critical points of x^3 - 3x` |
| Integrals | `integrate x^2 from 0 to 3`, `integral of x e^x dx`, `area between y = x^2 and y = x from 0 to 1` |
| Sums and sequences | `sum of the first 100 positive integers`, `sum of (1/2)^k from k = 1 to infinity`, `10th term of 3, 7, 11, ...` |
| Linear algebra | determinant, inverse, eigenvalues, rank, RREF, products, `solve [[2,1],[1,3]] x = [3,5]` |
| Matrices described by facts | `A is 3×3 with eigenvalues 1, -2, 3: find det(A^3 - 2A) and tr((A + I)^-1)` |
| Probability and statistics | mean/median/std, binomial, Poisson, normal, uniform, exponential, `at least one six in 4 rolls of a fair die`, drawing without replacement, counting |
| Percentages | `15% of 240`, `12 is what percent of 48?`, `percentage change from 80 to 120` |

Questions with several parts (`(a) … (b) …`, `solve … and find …`) are split and every part is answered on its own.
Follow-up questions keep the context: `now integrate it`, `explain step 2`, `how was this checked?`.

Each answer comes with exact maths, step-by-step working, an interactive graph where useful, a plain-language
explanation, and a mark that says how far it can be trusted:

| Mark | Meaning |
|---|---|
| **Fully verified** | The maths was confirmed independently and the reading matches every part of the question |
| **Verified with assumptions** | The maths is confirmed, but the reading relies on assumptions that are listed |
| **Approximate** | Confirmed numerically within tolerance, not symbolically |
| **Needs clarification** | Information is missing or the question can be read several ways, so nothing is computed |

## How a question is answered

```
browser ──► API (FastAPI) ──► rate limit + Groq budget ──► planner ──► sandbox worker ──► answer
                                                          (Groq or       │
                                                           offline)      ├─ intent check
                                                                         ├─ math tool (SymPy / NumPy / SciPy)
                                                                         ├─ independent verification
                                                                         └─ steps, graph, explanation
```

1. **Clean-up.** LaTeX (`\(…\)`, `\frac`, `\text{Tr}`) is turned into plain maths text. A question with several parts is
   split into self-contained parts, each of which goes through the steps below.
2. **Limits.** Per-visitor rate limits, a request-size limit and strict input validation are applied first.
3. **Planning.** The question becomes a *plan*: one whitelisted tool plus its arguments (for example
   `integrate(expression="x^2", lower=0, upper=3)`).
   - If Groq is configured and within its daily budget, it proposes a plan (JSON-schema output, `openai/gpt-oss-120b`).
   - Offline planners (rules and a language analyser) always run too, and take over whenever Groq is unavailable, out
     of budget or wrong. The page says which planner was used and why.
4. **Intent check.** Before anything is computed, the plan is compared with the question using rules of its own: the
   task, every number and condition, bounds, points, the event in a probability question, every part of a multi-part
   question. A plan that misreads the question is repaired from the question itself, replaced by another planner's
   plan, or turned into a clarification request. Plans in which the model did arithmetic itself are rejected.
5. **Computation.** The tool runs in an isolated worker process: no network, no secrets, a time limit (the worker is
   killed on timeout) and a memory limit. Inputs that would explode (`9^9^9^9`, `(x+1)^5000`) are refused before parsing.
6. **Verification.** A different method confirms the result (table below). If the check disagrees, the answer is
   marked as failed rather than shown as correct.
7. **Presentation.** Steps and graphs are built from the computed objects, never from model text. The explanation
   comes from a template (or optionally the model, which may not change any number).

| Tool | Independent check |
|---|---|
| Equations and systems | Substitute every solution back; a numeric root scan confirms no real root is missing |
| Simplify / factor / expand | Symbolic difference plus numeric equality at random points |
| Derivatives | Finite differences at sample points, and integrating the result back |
| Integrals | Differentiate the antiderivative; numeric quadrature for definite integrals (absolute values split at sign changes) |
| Limits | Numeric evaluation approaching from each side |
| Matrices | NumPy recomputation, `A·A⁻¹ = I`, characteristic polynomial, `A·x = b` residual |
| Matrices described by facts | Build concrete matrices with each possible spectrum and recompute every condition and answer |
| Sums and sequences | Add the terms directly / generate the sequence term by term; `S(n) − S(n−1)` for formulas in n |
| Critical points | `f′(c) = 0` exactly, the sign of `f′` on each side, and a numeric scan so none is missed |
| Statistics, probability, counting, percentages | NumPy / `scipy.stats` / exact integer and fraction arithmetic |

## Safety and limits

- The model never runs code and never computes: it can only choose from a fixed list of tools, and every expression
  goes through an allow-list parser (no builtins, attributes or unknown functions) before SymPy sees it.
- Every computation runs in a sandboxed worker with a time limit and a memory limit; a runaway job is killed and the
  worker replaced.
- Groq usage is capped by a rolling 24-hour token budget (below the free tier), a per-visitor share and an automatic
  pause after rate limits or errors; the offline planners answer meanwhile.
- The API key exists only in the backend process: never in workers, responses, logs or the frontend bundle.
- Per-visitor rate limits (default 10 a minute, 300 a day), 16 KB request limit, strict CORS, security headers.

## Tech stack

FastAPI, LangGraph, SymPy, NumPy, SciPy, mpmath, Pydantic · React 19, TypeScript, Vite, KaTeX, Plotly · Groq
(optional) · pytest, Vitest, ruff, oxlint · Docker, GitHub Actions · Vercel (frontend) and Render (backend).

```
backend/app/
  main.py, service.py        API, rate limits, Groq budget, multi-part questions, sessions
  agents/                    planners (llm, rules, nl, spectral, multipart), intent check, LangGraph workflow, tutor
  math_engine/               safe parser, tools, verifier, graphs, spectral tool, LaTeX clean-up
  sandbox.py, jobs.py        isolated worker processes
  budget.py, ratelimit.py, security.py
backend/eval/                308-question benchmark and reports
frontend/src/                React app
deploy/render/, render.yaml, frontend/vercel.json   deployment
```

## Run it locally

With Docker (UI on http://localhost:8080):

```bash
docker compose up --build
```

Without Docker, start the backend:

```bash
cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
```

```bash
uvicorn app.main:app --reload --port 8000
```

Then the frontend in a second terminal (http://localhost:5173):

```bash
cd frontend && npm install && npm run dev
```

Groq is optional. Without a key everything works with the offline planners. To use it, put a free key from
https://console.groq.com/keys in `backend/.env` as `GROQ_API_KEY=…` (see `backend/.env.example` for all settings,
including rate limits, the token budget and sandbox limits). Never commit `.env`.

## Tests

```bash
cd backend && pytest
```

```bash
cd frontend && npm test
```

475 backend tests and 20 frontend tests, all offline (no Groq calls). Besides correctness for every tool, they
deliberately corrupt results and plans to prove the verifier and the intent check catch them, and they cover the
sandbox, rate limits, the Groq budget, secret redaction and the UI. CI runs linting, type checks, both test suites,
the benchmark and a Docker smoke test on every push.

## Evaluation

### Dataset

`backend/eval/` holds a hand-checked benchmark. Ground truth is written from closed forms and definitions, and
`python -m eval.ground_truth` confirms every answer independently with mpmath, NumPy and SciPy, without importing
any application code.

| Split | Problems | Purpose |
|---|---|---|
| Development | 150 answerable + 29 to refuse | Every tool, edge cases, ambiguous, unsupported and unsafe input |
| Held-out paraphrases | 31 | Natural wording, written after the planner was frozen |
| Unseen real-world | 45 answerable + 11 to clarify | Word problems, distractor numbers, missing information; written before any tuning on it |
| NL tuning | 38 answerable + 4 to clarify | Used to develop the language handling |
| Tampered solutions | 34 | Correct results corrupted (wrong root, missing root, wrong derivative, …): the verifier must fail them |
| Tampered readings | 36 | Correct plans with one misreading (wrong operation, swapped bounds, invented parameter, …): the intent check must catch them |

### Results

| Split | Offline planners | With Groq (`gpt-oss-120b`) |
|---|---|---|
| Development: answers / refusals | 150/150 · 29/29 | 150/150 · 29/29 |
| Held-out paraphrases | 30/31 | 30/31 |
| Unseen real-world: answers / clarifications | 43/45 · 11/11 | 44/45 · 11/11 |
| NL tuning: answers / clarifications | 38/38 · 4/4 | 38/38 · 4/4 |
| **All 308 problems** | **261/264 answered correctly, 44/44 refusals** | **262/264 answered correctly, 44/44 refusals** |
| **Wrong answers** | **0** | **0** |

The few unanswered questions get a clarification request, never a guess.

| Trust and robustness | Result |
|---|---|
| Marked *fully verified* ⇒ correct | 177/177 offline · 205/205 with Groq |
| Maths marked *verified* ⇒ correct | 232/232 |
| Tampered solutions marked failed | 34/34 (none marked verified) |
| Tampered readings flagged / valid plans wrongly flagged | 36/36 / 0/35 |
| Correct plans wrongly rejected by the intent check (oracle mode) | 0/264 |
| Crashes | 0/308 |

**Groq planner on its own:** 240 plans passed the intent check and all 232 that produced an answer were correct; it
chose the right tool 244/245 times. 21 plans were rejected by the intent check, which is the check working: 10 added
an unrequested "real numbers only", 7 filled in information the question did not give, 4 were misreadings. 28 calls
(9.1%) returned malformed output with tool calling, which led to the switch to JSON-schema output; the offline
planners answered those questions meanwhile.

**Unseen stress tests** (questions written after development, in varied wording):

| Test | Result |
|---|---|
| 30 mixed questions (word problems, absolute values, areas, series, eigen-facts, …) | 28 answered correctly offline, the other 2 need Groq to read the wording; 0 wrong |
| 20 fresh questions on sums, sequences, extrema, percentages, probability | 19 answered correctly, 1 unusual phrasing asked for clarification; 0 wrong |

Each stress test found real bugs (two answers to the wrong question, a missing root in a verifier, a dropped
question part). They were fixed and turned into regression tests.

**Speed:** offline, a typical problem takes 21 ms (p50) and 94 ms (p95). A Groq planning call adds about 1.2 s (p50) and
1.9 s (p95). On the free 512 MB / 0.1 CPU Render instance it is 0.3 to 3.6 s per problem.

Reproduce (the Groq column used a real API key; the committed reports are in `backend/eval/reports/`):

```bash
cd backend && python -m eval.ground_truth
```

```bash
python -m eval.run --mode rules
```

```bash
python -m eval.run --mode oracle
```

The Groq results come from the full 308-question run, before the newest question types (sums, sequences, critical
points, percentages, word-problem probability) were added. Those are covered by the stress tests and regression tests.

## Deployment

Free tiers only: the frontend on **Vercel**, the backend as a Docker web service on **Render** (512 MB, sleeps after
15 minutes idle), Groq's free API for planning. Step-by-step instructions, environment variables and live checks are in
[DEPLOYMENT.md](DEPLOYMENT.md).

## Limitations

- Not supported yet: inequalities, differential equations, implicit derivatives, multivariable calculus, proofs.
  These get a clear "not supported" or a clarification request rather than a guess.
- The offline planners understand common phrasings; unusual wording needs Groq or a rephrase.
- The intent check is rule-based. A misreading that matches every number and cue can still pass as "verified with
  assumptions", never as fully verified.
- Infinite series and some limits are confirmed numerically, so they are marked approximate.
- Sessions, rate limits and the token budget live in memory in one process and reset on restart.
