# Deploying Mathenza ($0)

| Part | Service (free tier) | What runs there |
|---|---|---|
| Frontend | Vercel Hobby | The static React build (`frontend/`) |
| Backend | Render free web service (Docker, 512 MB RAM, 0.1 CPU) | FastAPI + the SymPy worker sandbox (`backend/app`) |
| LLM | Groq free tier, `openai/gpt-oss-120b` | Planning only; optional |

No paid plan or paid API is used. Render may ask for a card to verify a new account (a temporary hold that is released;
nothing is charged on the free instance type). Without a Groq key the app still works, using its offline planners.

Mathenza's own budget (150k tokens per rolling 24 h) stays below Groq's free limit (200k). When it is used up, or Groq returns
a rate-limit error, the offline planner answers and the page says so.

Hugging Face Spaces was the first plan, but new Docker Spaces now need a paid plan, so the backend moved to Render.

## Sizing for the free instance (measured)

The Render image was run locally with the free instance's limits (512 MB RAM, 0.1 CPU):

- **Memory:** about 250–300 MB in use.
- **Speed:** typical problems take 0.3–3.6 s; a cold start takes about 70 s.
- **Correctness:** a third of the benchmark gave exactly the same answers as an unrestricted run.
- **Runaway work:** a runaway computation is stopped at the time limit without the instance running out of memory.

The image therefore uses:

| Setting | Value | Why |
|---|---|---|
| `SANDBOX_WORKERS` | `1` | Only one worker fits next to the API process in 512 MB |
| `SANDBOX_MEMORY_MB` | `640` | Each worker's address-space limit, about 215 MB above its idle size, so a memory-hungry job fails inside the worker first |
| `SANDBOX_TIMEOUT_SECONDS` | `20` | The instance has a tenth of a CPU |

## What is in the repository

| File | Purpose |
|---|---|
| `render.yaml` | Render Blueprint: one free Docker web service `mathenza-api`, health check `/api/health`, auto-deploy after CI passes, rebuilt only for backend changes |
| `deploy/render/Dockerfile` | The Render image (build context `backend/`): non-root user, listens on `$PORT`, one API process, the sandbox settings above |
| `frontend/vercel.json` | Vercel build settings, SPA routing and security headers (the browser may only call `self` and `*.onrender.com`) |
| `frontend/.env.example` | The only frontend setting: `VITE_API_URL` |

Local development is unchanged: `docker compose up --build` (nginx + backend) or `npm run dev` + `uvicorn`.

## Settings and secrets

**Render** (service → Environment):

| Name | Kind | Value |
|---|---|---|
| `GROQ_API_KEY` | Secret | Your key from https://console.groq.com/keys (optional; without it the offline planners are used) |
| `CORS_ORIGINS` | Value | Your Vercel production URL, exactly, e.g. `https://mathenza.vercel.app` (https, no trailing slash; comma-separate several). `*` is ignored. |

Already set in the image (override in Render only if needed): `TRUSTED_PROXY_HOPS=1`, `SANDBOX_WORKERS=1`,
`SANDBOX_MEMORY_MB=640`, `SANDBOX_TIMEOUT_SECONDS=20`, `ENABLE_DOCS=false`, `LLM_EXPLANATIONS=false`.
Application defaults: `LLM_MODEL=openai/gpt-oss-120b`, `LLM_STRUCTURED_METHOD=json_schema`, `LLM_DAILY_TOKEN_BUDGET=150000`,
`LLM_DAILY_REQUEST_BUDGET=800`, `LLM_CALLS_PER_CLIENT_PER_DAY=40`, `RATE_LIMIT_PER_MINUTE=10`, `RATE_LIMIT_PER_DAY=300`.
See the configuration table in `README.md` for all of them.

**Vercel** (project → Settings → Environment Variables):

| Name | Value |
|---|---|
| `VITE_API_URL` | The Render URL, e.g. `https://mathenza-api.onrender.com` (no trailing slash) |

`VITE_API_URL` is compiled into the public JavaScript; it is a URL, not a secret. Never add an API key to Vercel: the Groq key only
exists as a Render secret, inside the backend process. It never reaches the sandbox workers, API responses or logs.

## Step by step

### 1. Deploy the backend on Render

1. Sign up at https://render.com with **GitHub** (free plan), and allow Render to access the repository.
2. **New → Blueprint →** select the repository (branch `main`). Render reads `render.yaml` and shows one service,
   `mathenza-api`, on the **Free** instance type.
3. When asked for the secret values, enter:
   - `GROQ_API_KEY` = your Groq key;
   - `CORS_ORIGINS` = `https://placeholder.invalid` for now (you will set the real Vercel URL in step 3).
4. **Apply.** The first build takes about 5–10 minutes. The service URL is shown at the top, e.g. `https://mathenza-api.onrender.com`;
   if the name is taken, Render adds a suffix.
5. Open `<render-url>/api/health`. Expect `"status":"ok"` and `"llm_configured":true` (or `false` without a key).

**If Render rejects the Blueprint**, create the service by hand instead: **New → Web Service →** the repository, with:

| Field | Value |
|---|---|
| Language / runtime | Docker |
| Branch | `main` |
| Dockerfile path | `deploy/render/Dockerfile` |
| Docker build context directory | `backend` |
| Instance type | Free |
| Health check path | `/api/health` |
| Environment | `GROQ_API_KEY` (secret), `CORS_ORIGINS` |
| Auto-deploy | On commit, or after CI checks pass |

Either way, the service is now connected to GitHub: every push to `main` that changes the backend redeploys it once CI has passed.

### 2. Deploy the frontend on Vercel

1. Sign in at https://vercel.com with GitHub (Hobby plan, free).
2. **Add New → Project →** import the repository.
3. Set **Root Directory** to `frontend`. Vercel picks up `vercel.json`: framework Vite, install `npm ci`, build
   `npm run build`, output `dist`, with SPA routing and security headers.
4. Under **Environment Variables** add `VITE_API_URL` = your Render URL.
5. **Deploy.** Note the production URL, e.g. `https://mathenza.vercel.app`. The project name decides the URL; use `mathenza` if it is free.

### 3. Connect the two

1. Render → `mathenza-api` → **Environment** → set `CORS_ORIGINS` to the exact Vercel production URL. Save; the service redeploys.
2. If you later change `VITE_API_URL` in Vercel, **redeploy** the frontend: Vite reads it at build time.

Only the production URL is allowed. Vercel preview deployments use other URLs, so the browser blocks their API calls. That is
intended; add a preview URL to `CORS_ORIGINS` temporarily if you need to test one.

### 4. Test the live application

1. Open the Vercel URL. The header should show "Groq planner · openai/gpt-oss-120b", or "Offline planner" without a key.
   A free Render service sleeps after 15 minutes without traffic. The first visit after that shows "Connecting to the server…"
   for about 1–3 minutes while it wakes.
2. Solve `Integrate x^2 from 0 to 3`: the answer is 9, with a "Fully verified" badge, steps and a plot.
3. Try `A fair coin is tossed 6 times. What is the probability of exactly 2 heads?`. It should be "Planned by Groq LLM" when the key is set.
4. Try `simplify 9^9^9^9`: it is refused immediately. `expand (x+y+z+w+1)^200` stops at the 20-second limit with "Time limit reached".
5. Ask a follow-up such as "how was this verified?".
6. From a terminal, check that other origins are refused (no `access-control-allow-origin` header in the reply):

   ```bash
   curl -s -D - -o /dev/null -X OPTIONS https://mathenza-api.onrender.com/api/solve -H 'Origin: https://evil.example' -H 'Access-Control-Request-Method: POST'
   ```

7. Check that per-visitor rate limits use real client addresses. Add the Render variable `LOG_LEVEL=DEBUG`, solve a problem,
   and open **Logs**: the `client key` lines should show your public IP. If they show a private address such as `10.x`,
   raise `TRUSTED_PROXY_HOPS`; if they show an address that isn't yours, lower it. Then remove `LOG_LEVEL`.

## Operating notes

- **Sleep:** the free service sleeps after 15 minutes idle and wakes on the next request (about 1–3 minutes). The free plan includes 750 instance hours a month, which is enough for one service running all month.
- **Restarts:** the disk is not persistent, so sessions, rate-limit counters and the budget ledger start fresh after a
  restart or sleep. The budget is conservative, and Groq's own 429s still pause the planner.
- **Single instance:** run one instance only, because limits and the budget are kept in memory.
- **Quota:** watch your usage on the Groq console. Nothing in this setup can incur charges: the services stay on free plans and the Groq free tier has no billing attached.
- **Roll back:** in Render, open **Events** and choose an earlier deploy → *Rollback*. In Vercel, open **Deployments** and *Promote* an earlier one.
