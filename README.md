# MathForge AI

**Solve · Verify · Explain.** MathForge AI is a math reasoning app built around a
**Solver–Verifier–Tutor** workflow. An LLM (Groq) can read the problem and pick a tool. The math itself
is always done by **SymPy, NumPy and SciPy**, and a separate verifier checks every result with a
**different method** before it reaches the user.

![MathForge AI screenshot](docs/screenshot.png)

## What it does

| Category | Examples |
|---|---|
| Algebra | `Solve x^2 - 5x + 6 = 0`, `x + 2y = 7, 3x - y = 7`, `factor x^3 - 8`, `cos(x) = x` (numeric) |
| Calculus | `derivative of x^3 sin(x)`, `second derivative of exp(x^2)`, `derivative of x^2 at x = 3`, `limit of sin(x)/x as x -> 0` |
| Integrals | `Integrate x^2 from 0 to 3`, `integral of x e^x dx`, `integrate exp(-x^2) from -oo to oo` |
| Linear algebra | `determinant / inverse / eigenvalues / rank / rref / transpose of [[…]]`, `solve [[2,1],[1,3]] x = [3,5]`, matrix products |
| Probability & statistics | `mean and standard deviation of 2, 4, 4, 5`, `Binomial n=10 p=0.5, P(X <= 3)`, `normal mean 100 std 15, P(X > 130)`, Poisson / uniform / exponential, `10 choose 3`, `P(5,2)`, `6!` |

Every solution includes:

- an **exact answer** (LaTeX, rendered with KaTeX)
- a **verification badge**: `verified`, `approximate`, `unverified` or `failed`, plus each check that was run
- **step-by-step working**, built from the actual computation
- an **interactive Plotly chart**: roots on the curve, f with f′ and a tangent line, shaded area under a curve, limit behaviour, eigenvalues in the complex plane, histograms, and shaded probability regions
- an **explanation**: written by the LLM when a key is set, from a template otherwise
- **follow-up questions** in the same session (`now integrate it`, `explain the steps`, `how was this verified?`)

## Architecture

```
                ┌──────────── LangGraph StateGraph ─────────────┐
 problem ──►  plan ──► solve ──► verify ──► explain ──► visualize ──► response
               │         │
               │         └─ tool error ──────────────────────────► error response
               └─ unsupported / ambiguous ───────────────────────► "not supported" response
```

| Layer | Path | Responsibility |
|---|---|---|
| Safe parser | `backend/app/math_engine/parser.py` | Lexical allowlist (characters, identifiers, length) runs **before** SymPy sees anything. No builtins, attribute access, dunders or unknown functions. |
| Tools | `backend/app/math_engine/tools.py` | 12 whitelisted tools (`solve_equation`, `integrate`, `matrix_operation`, `probability_distribution`, …) with timeouts, input validation and generated steps |
| Verifier | `backend/app/math_engine/verifier.py` | Checks each result with a *different* method (see below) |
| Visualizer | `backend/app/math_engine/visualize.py` | Builds Plotly JSON from the computed objects |
| Planner (LLM) | `backend/app/agents/llm_planner.py` | Groq via `langchain-groq` structured output (tool calling). Returns only a tool name and string arguments, validated by Pydantic. |
| Planner (rules) | `backend/app/agents/rule_planner.py` | Deterministic offline planner. Used when no key is set, or as a fallback when the LLM's plan fails. |
| Tutor | `backend/app/agents/tutor.py` | Explains the verified result without adding numbers, and answers follow-ups |
| Graph | `backend/app/agents/graph.py` | LangGraph workflow with conditional edges |
| API | `backend/app/main.py`, `service.py` | FastAPI: `/api/solve`, `/api/followup`, `/api/health`, `/api/examples`; in-memory LRU sessions |
| UI | `frontend/` | React 19 + TypeScript + Vite, KaTeX, Plotly (loaded lazily) |

### How each result is verified

| Tool | Independent check(s) |
|---|---|
| Equations | Substitute each solution back (symbolic, then numeric residual). Separately, a SciPy `brentq` sign-change search on [-50, 50] confirms no real root was missed. |
| Systems | Substitute each solution into every equation |
| Simplify / factor / expand | Symbolic difference, plus numeric equality at random points |
| Derivatives | Central finite differences at random points, and integrating the result back |
| Integrals | Differentiate the antiderivative back to the integrand; definite integrals also checked with `scipy.integrate.quad` |
| Limits | Evaluate numerically while approaching the point from each side |
| Matrices | NumPy recomputation, `A·A⁻¹ = I`, eigenvalues as roots of the characteristic polynomial, `A·x = b` residual, Berkowitz determinant |
| Statistics / probability | NumPy / `scipy.stats` recomputation; Python integer arithmetic for combinatorics |

**Status meanings**

- `verified`: an exact check passed and every numeric check agrees.
- `approximate`: only numeric agreement was possible, for example a root found with Newton's method or a limit.
- `unverified`: no independent check applied.
- `failed`: a check disagreed. The UI shows this in red and opens the details.

The tests corrupt correct solver outputs on purpose (a wrong root, a missing root, a wrong
derivative, a wrong eigenvalue, and so on) and assert that the verifier reports `failed`.

### Safety model

- The LLM **never runs code**. It can only pick from a fixed list of tools (a `Literal` type).
  Arguments that a tool doesn't declare are rejected.
- Every expression passes the allowlist in `parser.py` before SymPy parses it. SymPy runs with an
  empty `__builtins__` and an explicit symbol table. Input like `__import__("os")`, `x.__class__`, `exec(…)`,
  quotes, `;` or `[]` inside an expression is rejected (`tests/test_parser.py`).
- Matrix and list inputs are read with `ast.literal_eval`, which accepts literals only.
- Limits on input size: expressions up to 500 characters, matrices up to 10×10, data sets up to 10,000 values.
  Each symbolic computation times out after 10 seconds.
- If the problem is ambiguous (several variables and none specified), unsupported (for example a differential
  equation), or has no solution (`x = x + 1`, a singular matrix), the app returns a clear message instead of a guess.

## Running it

### Option 1: Docker (one command)

```bash
cp backend/.env.example .env        # optional: add a free GROQ_API_KEY
```

```bash
docker compose up --build
```

Open http://localhost:8080 for the UI. The API is at http://localhost:8000/docs.

### Option 2: Local development

Backend (Python 3.12+):

```bash
cd backend && python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements-dev.txt
```

```bash
cp .env.example .env   # optional: set GROQ_API_KEY
```

```bash
uvicorn app.main:app --reload --port 8000
```

Frontend (Node 22+), in a second terminal. The Vite dev server proxies `/api` to port 8000:

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173.

### LLM configuration (optional, free)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Free key from https://console.groq.com/keys. Without one, the deterministic planner and template explanations are used. |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Any Groq chat model that supports tool calling |
| `LLM_ENABLED` | `true` | Set to `false` to force offline mode |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:8080` | Allowed browser origins |

Never commit `.env`. It is listed in `.gitignore`.

## Testing

```bash
cd backend && pytest
```

```bash
cd frontend && npm test
```

The backend suite (142 tests) covers parser security, the correctness of every tool against known answers,
verifier acceptance *and* failure detection, the rule planner, the LangGraph workflow with a fake LLM
(including fallback when the LLM's plan is invalid or the provider is down), and the HTTP API.
It runs fully offline. The frontend suite uses Vitest and Testing Library.

CI (`.github/workflows/ci.yml`) runs ruff, pytest, oxlint, vitest and the production build. It then
builds both Docker images and smoke-tests the running stack.

## Known limitations / roadmap

- No inequalities, ODEs, series or multivariable calculus yet. Requests for these are reported as unsupported.
- Sessions are kept in memory and are lost on restart. A shared store such as Redis would be needed for more than one replica.
- The offline rule planner understands common phrasings only. Free-form word problems need the LLM planner.
- LLM-backed planning and explanations are covered by mocked tests only. CI does not call Groq.
- Approximate numeric roots come from multi-start Newton search on [-10, 10]. Roots outside that range can be missed.
  The verifier's [-50, 50] sweep then reports `failed`.
- An automated evaluation set (a benchmark of word problems with an accuracy report) is planned next.
