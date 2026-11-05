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

474 backend tests and 20 frontend tests, all offline (no Groq calls). Besides correctness for every tool, they
deliberately corrupt results and plans to prove the verifier and the intent check catch them, and they cover the
sandbox, rate limits, the Groq budget, secret redaction and the UI. CI runs linting, type checks, both test suites,
the benchmark and a Docker smoke test on every push.

## Benchmark

`backend/eval/` holds 308 hand-checked problems (264 answerable, 44 that should be refused or clarified), including
held-out paraphrases and an unseen set written before tuning, plus 34 tampered solutions and 36 tampered readings.
Ground truth is confirmed independently with mpmath/SciPy (`python -m eval.ground_truth`). Reports are in
`backend/eval/reports/`.

| | Offline planners | With Groq (`gpt-oss-120b`) |
|---|---|---|
| Correct answers | 261 / 264 | 262 / 264 |
| Correct refusals / clarifications | 44 / 44 | 44 / 44 |
| Wrong answers | **0** | **0** |
| Fully verified ⇒ correct | 177 / 177 | 205 / 205 |
| Tampered solutions caught | 34 / 34 | 34 / 34 |
| Tampered readings caught (false alarms) | 36 / 36 (0 / 35) | 36 / 36 (0 / 35) |

```bash
cd backend && python -m eval.run --mode rules
```

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

Built by Aekank Patel.
