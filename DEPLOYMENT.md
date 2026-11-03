# Deploying Mathora ($0)

| Part | Service (free tier) | What runs there |
|---|---|---|
| Frontend | Vercel Hobby | The static React build (`frontend/`) |
| Backend | Hugging Face Spaces, Docker SDK, "CPU basic" (2 vCPU, 16 GB RAM) | FastAPI + the SymPy worker sandbox (`backend/app`) |
| LLM | Groq free tier, `openai/gpt-oss-120b` | Planning only; optional |

No credit card, paid plan or paid API is needed. Without a Groq key the app still works, using its offline planners.

Mathora's own budget (150k tokens per rolling 24 h) stays below Groq's free limit (200k). When it is used up, or Groq returns
a rate-limit error, the offline planner answers and the page says so.

## What is in the repository

| File | Purpose |
|---|---|
| `deploy/huggingface/Dockerfile` | Space image: Python 3.12, runs as UID 1000, listens on port 7860, one API process, 2 sandbox workers |
| `deploy/huggingface/README.md` | Space configuration header (`sdk: docker`, `app_port: 7860`) and the Space page text |
| `scripts/build_space_bundle.sh` | Assembles the Space content: that README and Dockerfile, plus `backend/app` and `backend/requirements.txt`. Tests, benchmark data, the frontend and local files are not included. |
| `.github/workflows/deploy-space.yml` | Manual GitHub Action that pushes the bundle to the Space (needs `HF_TOKEN` and `HF_SPACE`) |
| `frontend/vercel.json` | Vercel build settings, SPA routing and security headers |
| `frontend/.env.example` | The only frontend setting: `VITE_API_URL` |

Local development is unchanged: `docker compose up --build` (nginx + backend) or `npm run dev` + `uvicorn`.

## Settings and secrets

**Hugging Face Space** (Space → Settings → Variables and secrets):

| Name | Kind | Value |
|---|---|---|
| `GROQ_API_KEY` | **Secret** | Your key from https://console.groq.com/keys (optional; without it the offline planners are used) |
| `CORS_ORIGINS` | Variable | Your Vercel production URL, exactly, e.g. `https://mathora.vercel.app` (https, no trailing slash; comma-separate several). `*` is ignored. |

Already set in the Space image (override only if needed): `TRUSTED_PROXY_HOPS=1`, `SANDBOX_WORKERS=2`, `SANDBOX_MEMORY_MB=1536`,
`ENABLE_DOCS=false`, `LLM_EXPLANATIONS=false`. Application defaults: `LLM_MODEL=openai/gpt-oss-120b`,
`LLM_STRUCTURED_METHOD=json_schema`, `LLM_DAILY_TOKEN_BUDGET=150000`, `LLM_DAILY_REQUEST_BUDGET=800`,
`LLM_CALLS_PER_CLIENT_PER_DAY=40`, `RATE_LIMIT_PER_MINUTE=10`, `RATE_LIMIT_PER_DAY=300`, `SANDBOX_TIMEOUT_SECONDS=10`.
See the configuration table in `README.md` for all of them.

**GitHub repository** (Settings → Secrets and variables → Actions), used only by the deploy workflow:

| Name | Kind | Value |
|---|---|---|
| `HF_TOKEN` | Secret | A Hugging Face access token with **write** access (ideally a fine-grained token limited to this Space) |
| `HF_SPACE` | Variable | The Space id, `<hf-username>/<space-name>` |

**Vercel project** (Settings → Environment Variables):

| Name | Value |
|---|---|
| `VITE_API_URL` | The Space's app URL, `https://<hf-username>-<space-name>.hf.space` (no trailing slash) |

`VITE_API_URL` is compiled into the public JavaScript; it is a URL, not a secret. Never add an API key to Vercel: the Groq key only
exists as a Space secret, inside the backend process. It never reaches the sandbox workers, API responses or logs.

## Step by step

### 1. Create the Hugging Face Space

1. Sign in at https://huggingface.co (free account).
2. Go to **New Space**. Choose a name (e.g. `mathora-api`), pick the **Docker** SDK with the **Blank** template, set the hardware to
   **CPU basic · free**, and make it **Public** (Vercel must reach it without a token).
3. Create it. It stays empty until the first deploy.

### 2. Connect the Space to the GitHub repository

1. On Hugging Face go to **Settings → Access Tokens → Create new token**. Use a fine-grained token with write access to
   this Space only (or a "Write" token). Copy it once; do not paste it anywhere else.
2. On GitHub, in the repository, go to **Settings → Secrets and variables → Actions**:
   - **Secrets** tab → *New repository secret* → name `HF_TOKEN`, value: the token.
   - **Variables** tab → *New repository variable* → name `HF_SPACE`, value `<hf-username>/mathora-api`.
3. Go to **Actions → Deploy backend to Hugging Face Space → Run workflow** (branch `main`). It pushes the bundle to the
   Space, and the Space then builds the image, which takes about 3–5 minutes. Run the workflow again whenever you want to
   deploy new backend code. It never runs on its own.

### 3. Configure the Groq key and CORS on the Space

1. Space → **Settings → Variables and secrets**:
   - *New secret* → `GROQ_API_KEY` = your Groq key.
   - *New variable* → `CORS_ORIGINS` = `https://placeholder.invalid` for now (you will set the real Vercel URL in step 5).
2. The Space restarts. Open `https://<hf-username>-mathora-api.hf.space/api/health`. Expect `"status":"ok"` and
   `"llm_configured":true` (or `false` if you skipped the key).

### 4. Deploy the frontend to Vercel

1. Sign in at https://vercel.com with GitHub (Hobby plan, free).
2. **Add New → Project →** import the repository.
3. Set **Root Directory** to `frontend`. Vercel picks up `vercel.json`: framework Vite, install `npm ci`, build
   `npm run build`, output `dist`, with SPA routing and security headers.
4. Under **Environment Variables** add `VITE_API_URL` = `https://<hf-username>-mathora-api.hf.space`.
5. **Deploy.** Note the production URL, e.g. `https://mathora.vercel.app` (you can choose the project name, and with it the URL).

### 5. Connect the two

1. Space → Settings → change the `CORS_ORIGINS` variable to the exact Vercel production URL (e.g. `https://mathora.vercel.app`).
   The Space restarts.
2. If you later change `VITE_API_URL` in Vercel, **redeploy** the frontend: Vite reads it at build time.

Only the production URL is allowed. Vercel preview deployments use other URLs, so the browser blocks their API calls. That is
intended; add a preview URL to `CORS_ORIGINS` temporarily if you need to test one.

### 6. Test the live application

1. Open the Vercel URL. The header should show "Groq planner · openai/gpt-oss-120b", or "Offline planner" without a key.
   After a long idle period the Space sleeps; the header shows "Connecting to the server…" for up to a minute or two while it wakes.
2. Solve `Integrate x^2 from 0 to 3`: the answer is 9, with a "Fully verified" badge, steps and a plot. The planner badge says
   which planner produced it.
3. Try a word problem such as `A fair coin is tossed 6 times. What is the probability of exactly 2 heads?`. It should be
   "Planned by Groq LLM" when the key is set.
4. Try `simplify 9^9^9^9`: it is refused immediately. `expand (x+y+z+w+1)^200` stops at the 10-second limit with "Time limit reached".
5. Ask a follow-up such as "how was this verified?".
6. From a terminal, check that other origins are refused (no `access-control-allow-origin` header in the reply):

   ```bash
   curl -s -D - -o /dev/null -X OPTIONS https://<hf-username>-mathora-api.hf.space/api/solve -H 'Origin: https://evil.example' -H 'Access-Control-Request-Method: POST'
   ```

7. Check that per-visitor rate limits use real client addresses. Set the Space variable `LOG_LEVEL=DEBUG`, solve a problem,
   and read the Space **Logs**: the `client key` lines should show your public IP. If they show a private address such as
   `10.x` or `172.x`, raise `TRUSTED_PROXY_HOPS`; if they show a private address followed by your IP, lower it. Then remove
   `LOG_LEVEL`.

## Operating notes

- **Sleep:** a free Space sleeps after a long idle period (about 48 hours) and wakes on the next request.
- **Restarts:** the Space's disk is not persistent, so sessions, rate-limit counters and the budget ledger start fresh after a
  restart or rebuild. The budget is conservative, and Groq's own 429s still pause the planner.
- **Single instance:** run one replica only, because limits and the budget are kept in memory.
- **Quota:** watch your usage on the Groq console. Nothing in this setup can incur charges: the Groq free tier has no billing
  attached, and no paid plan is used anywhere.
- **Roll back:** re-run the deploy workflow from an older commit (Run workflow → pick the ref), or revert on `main` and re-run it.
