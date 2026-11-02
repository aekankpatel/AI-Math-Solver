"""One-request Groq authentication check (no completion tokens are used).

    python -m eval.groq_check                       # explains what it would do; sends nothing
    GROQ_AUTH=proxy python -m eval.groq_check --allow-network
    GROQ_API_KEY=... python -m eval.groq_check --allow-network

Sends a single GET https://api.groq.com/openai/v1/models (free: it lists models
and consumes no tokens) and reports only: whether authentication succeeded,
the HTTP status class, and whether the configured model is available. It never
prints, logs or stores the key or any request header.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-network", action="store_true", help="actually send the single GET /models request")
    args = ap.parse_args(argv)

    from app.agents.llm import api_key_for_sdk
    from app.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    mode = s.groq_auth
    has_local_key = bool(s.groq_api_key)
    print(f"auth mode: {mode} | local GROQ_API_KEY present: {has_local_key} | model: {s.llm_model}")
    if mode == "api_key" and not has_local_key:
        print("No credentials: set GROQ_API_KEY, or GROQ_AUTH=proxy when a network secret injects the header.")
        return 2
    if not args.allow_network:
        print(
            "Would send ONE request: GET https://api.groq.com/openai/v1/models (no tokens). Re-run with "
            "--allow-network to send it. Nothing was sent."
        )
        return 0

    import groq

    client = groq.Groq(api_key=api_key_for_sdk(), max_retries=0, timeout=20)
    try:
        models = client.models.list()
    except groq.AuthenticationError:
        print(
            "AUTH FAILED (401). "
            + (
                "The proxy did not inject a valid credential for api.groq.com: check the "
                "network secret (host api.groq.com, path /openai/v1/, Authorization: Bearer) and start a NEW session "
                "so it is picked up."
                if mode == "proxy"
                else "GROQ_API_KEY was rejected."
            )
        )
        return 1
    except groq.PermissionDeniedError:
        print("FORBIDDEN (403): the key is valid but not allowed, or the egress policy blocked api.groq.com.")
        return 1
    except groq.APIConnectionError as exc:
        print(f"CONNECTION FAILED ({type(exc).__name__}): check network access to api.groq.com.")
        return 1
    except groq.APIStatusError as exc:
        print(f"HTTP {exc.status_code} from Groq.")
        return 1
    ids = {m.id for m in models.data}
    print(f"AUTH OK: {len(ids)} models visible. Configured model '{s.llm_model}' available: {s.llm_model in ids}")
    if s.llm_model not in ids:
        print("Set LLM_MODEL to an available tool-calling model before running the benchmark.")
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("LLM_ENABLED", "true")
    sys.exit(main())
