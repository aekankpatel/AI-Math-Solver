---
title: Mathora API
emoji: 🧮
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Solver–Verifier–Tutor math API (FastAPI, SymPy, Groq)
---

# Mathora API

Backend of [Mathora](https://github.com/aekankpatel/AI-Math-Solver): a FastAPI service that plans math problems
(Groq LLM or offline planners), solves them with SymPy/NumPy/SciPy in sandboxed worker processes and verifies every result
independently.

This Space is generated from the GitHub repository by `.github/workflows/deploy-space.yml`; edit the code there, not here.

- Health: `/api/health`
- Solve: `POST /api/solve` with `{"problem": "Integrate x^2 from 0 to 3"}`

The web interface is deployed separately; this API only accepts browser requests from the origins in `CORS_ORIGINS`.
