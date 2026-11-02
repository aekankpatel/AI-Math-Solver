"""Run the MathForge benchmark.

    python -m eval.run --mode oracle            # math engine only (canonical tool args)
    python -m eval.run --mode rules             # full LangGraph pipeline, deterministic planner
    python -m eval.run --mode groq --allow-network   # full pipeline with the Groq planner

``groq`` mode makes real API calls and therefore refuses to run unless
GROQ_API_KEY is set AND --allow-network is passed explicitly. Reports are
written to eval/reports/<mode>.json and eval/reports/<mode>.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sympy as sp

from eval.dataset import CASES, INTENT_TAMPER, TAMPER
from eval.scoring import score

REPORT_DIR = Path(__file__).parent / "reports"


def _configure(mode: str) -> None:
    os.environ["LLM_ENABLED"] = "true" if mode == "groq" else "false"
    if mode != "groq":
        os.environ["GROQ_API_KEY"] = ""
    from app.config import get_settings

    get_settings.cache_clear()


# ----------------------------------------------------------------------------- tamper


def _apply_mutation(data: dict[str, Any], mutate: dict[str, Any]) -> None:
    symbols = {}
    for key in ("var",):
        if isinstance(data.get(key), sp.Symbol):
            symbols[data[key].name] = data[key]
    for s in data.get("vars", []) or []:
        symbols[s.name] = s

    def parse(text: Any) -> sp.Basic:
        return sp.sympify(text, locals=symbols)

    for key, value in mutate.items():
        if key == "solutions":
            data["solutions"] = [parse(v) for v in value]
            data["real_solutions"] = [v for v in data["solutions"] if v.is_real]
        elif key == "system":
            data["solutions"] = [{symbols.get(k, sp.Symbol(k)): parse(v) for k, v in sol.items()} for sol in value]
        elif key == "result_matrix":
            data["result"] = sp.Matrix([[parse(v) for v in row] for row in value])
        elif key == "eigenvalues":
            ev = {parse(k): m for k, m in value.items()}
            data["result"] = data["eigenvalues"] = ev
        elif key.startswith("stats."):
            data["stats"][key.split(".", 1)[1]] = parse(value)
        elif key == "numeric":
            data["numeric"] = float(value)
        else:
            data[key] = parse(value)


def run_intent_tamper() -> list[dict[str, Any]]:
    """Wrong readings of a question must be flagged; correct ones accepted."""
    from app.agents.intent import check_intent
    from app.schemas import ProblemCategory, ToolCall

    rows = []
    for case in INTENT_TAMPER:
        good = ToolCall(tool=case["tool"], category=ProblemCategory.ALGEBRA, args=dict(case["args"]))
        bad_args = {k: v for k, v in {**case["args"], **case["bad"]}.items() if v is not None}
        if case["bad_tool"] == "solve_equation" and "equations" in bad_args:
            bad_args.pop("equations")
        bad = ToolCall(tool=case["bad_tool"] or case["tool"], category=ProblemCategory.ALGEBRA, args=bad_args)
        control, _ = check_intent(case["problem"], good)
        tampered, _ = check_intent(case["problem"], bad)
        rows.append(
            {
                "id": case["id"],
                "note": case["note"],
                "control_status": control.status.value,
                "control_expect": case.get("control_expect", "accept"),
                "tampered_status": tampered.status.value,
                "flag_reason": tampered.clarification,
            }
        )
    return rows


def run_tamper() -> list[dict[str, Any]]:
    from app.math_engine.tools import run_tool
    from app.math_engine.verifier import verify

    rows = []
    for case in TAMPER:
        t0 = time.perf_counter()
        out = run_tool(case["tool"], dict(case["args"]))
        control = verify(case["tool"], out).status.value
        _apply_mutation(out.data, case["mutate"])
        status = verify(case["tool"], out).status.value
        rows.append(
            {
                "id": case["id"],
                "tool": case["tool"],
                "note": case["note"],
                "control_status": control,
                "tampered_status": status,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        )
    return rows


# ----------------------------------------------------------------------------- main cases


def _run_case(case: dict[str, Any], mode: str) -> dict[str, Any]:
    from app.math_engine.parser import ExpressionError
    from app.math_engine.tools import ToolError, run_tool
    from app.math_engine.verifier import verify

    rec: dict[str, Any] = {
        "id": case["id"],
        "category": case["category"],
        "problem": case["problem"],
        "expected_tool": case["tool"],
        "expect": "reject" if case["truth"]["type"] == "reject" else "answer",
        "split": case.get("split") or ("heldout" if case.get("heldout") else "dev"),
    }
    t0 = time.perf_counter()
    try:
        if mode == "oracle":
            if not case["args"]:
                rec["outcome"] = "skipped"
                return rec
            rec["planned_tool"] = case["tool"]
            try:
                out = run_tool(case["tool"], dict(case["args"]))
            except (ToolError, ExpressionError) as exc:
                out, rec["outcome"], rec["error"] = None, "rejected", str(exc)
            verification = verify(case["tool"], out) if out else None
            tool = case["tool"]
            from app.agents.graph import overall_status
            from app.agents.intent import check_intent
            from app.schemas import ProblemCategory, ToolCall

            canon = ToolCall(tool=tool, category=ProblemCategory.ALGEBRA, args=dict(case["args"]))
            intent, _ = check_intent(case["problem"], canon)
            rec["intent"] = intent.status.value
            if out is not None:
                rec["overall"] = overall_status(verification, intent).value
        else:
            from app.agents.graph import run_solver

            state = run_solver(case["problem"])
            call = state.get("tool_call")
            rec["planned_tool"] = call.tool if call else None
            rec["planner"] = state.get("planner")
            out, verification, tool = state.get("tool_output"), state.get("verification"), rec["planned_tool"]
            if state.get("intent") is not None:
                rec["intent"] = state["intent"].status.value
                rec["interpretation"] = state["intent"].interpretation
            if state.get("overall") is not None:
                rec["overall"] = state["overall"].value
            if mode == "groq":
                rec.update(_llm_accounting(state))
            if state.get("status") == "needs_clarification":
                rec["outcome"], rec["error"] = "clarify", state.get("error")
            elif state.get("status") != "solved":
                rec["outcome"], rec["error"] = "rejected", state.get("error")
    except Exception as exc:  # anything escaping the pipeline is a crash
        rec["outcome"], rec["error"] = "crash", f"{type(exc).__name__}: {exc}"
        out = verification = None
        tool = None
    rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if out is not None:
        rec["outcome"] = "answered"
        rec["answer"] = out.answer_text[:200]
        rec["verification"] = verification.status.value if verification else None
        if rec["expect"] == "answer":
            rec["correct"], rec["detail"] = score(tool, out.data, case["truth"])
        else:
            rec["correct"], rec["detail"] = False, "answered a problem that should have been refused"
    return rec


def _quota_check(rec: dict[str, Any], seen: dict[str, str], tpm: float) -> tuple[str | None, float]:
    """Inspect Groq's rate-limit headers after a call. Returns (stop reason or None, allowed TPM).

    Groq headers: x-ratelimit-limit/remaining-requests = requests per DAY,
    x-ratelimit-limit/remaining-tokens = tokens per MINUTE. The daily token cap is not in the headers;
    it only appears as a 429 whose message mentions a per-day limit.
    """
    from app.agents.llm import rate_limits

    seen.update(rate_limits())
    allowed = tpm
    try:
        allowed = min(tpm, float(seen["x-ratelimit-limit-tokens"]))
    except (KeyError, ValueError):
        pass
    err = (rec.get("llm_error") or "").lower()
    if "429" in err or "rate limit" in err or "rate_limit" in err:
        if "per day" in err or "tpd" in err or "rpd" in err:
            hint = re.search(r"try again in ([\w.]+)", err)
            return (
                f"daily Groq quota reached ({'resets in ' + hint.group(1) if hint else 'see the Groq console'}); "
                "progress is cached - re-run the same command after the reset to resume"
            ), allowed
    try:
        if int(seen["x-ratelimit-remaining-requests"]) <= 0:
            return (
                f"daily request quota exhausted (resets in {seen.get('x-ratelimit-reset-requests', '?')}); "
                "re-run the same command after the reset to resume"
            ), allowed
    except (KeyError, ValueError):
        pass
    return None, allowed


def _llm_accounting(state: dict[str, Any]) -> dict[str, Any]:
    """What happened to the LLM's plan for this problem (groq mode only).

    accepted              the LLM plan passed the intent check and was executed (possibly after a safe repair)
    rejected_fallback     the intent check rejected the LLM plan; an offline plan that passes was used instead
    rejected_clarify      the LLM plan was rejected and no faithful plan existed: the user is asked
    declared_unsupported  the LLM itself refused (supported = false)
    error                 the LLM call failed (rate limit, timeout, malformed output): offline planners ran
    """
    plan, err, final = state.get("llm_plan"), state.get("llm_error"), state.get("tool_call")
    out: dict[str, Any] = {}
    if plan is not None:
        out["llm_plan"] = {"tool": plan.tool, "args": plan.args, "ambiguities": plan.ambiguities}
    if err:
        # account identifiers are not credentials, but reports are committed: redact them
        out["llm"], out["llm_error"] = "error", re.sub(r"org_[A-Za-z0-9]+", "org_<redacted>", err)
    elif plan is None and state.get("planner") == "llm" and state.get("status") == "unsupported":
        out["llm"] = "declared_unsupported"
    elif plan is None:
        out["llm"] = "no_plan"
    elif state.get("status") == "needs_clarification":
        out["llm"] = "rejected_clarify"
    elif final is not None and final.source == "llm":
        out["llm"] = "accepted"
        out["llm_repaired"] = bool(state.get("intent") and state["intent"].repaired)
    else:
        out["llm"] = "rejected_fallback"
    return out


def _llm_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ran = [r for r in records if r.get("outcome") != "skipped"]
    count = Counter(r.get("llm", "no_plan") for r in ran)
    accepted = [r for r in ran if r.get("llm") == "accepted"]
    acc_ans = [r for r in accepted if r["expect"] == "answer"]
    refuse = [r for r in ran if r["expect"] == "reject"]
    return {
        "scope": "every problem run; end-to-end metrics above include offline fallbacks, these isolate the LLM",
        "outcomes": {k: _m(v, len(ran)) for k, v in sorted(count.items())},
        "llm_errors": _m(count.get("error", 0), len(ran)),
        "llm_plans_accepted_by_intent_check": _m(len(accepted), sum(1 for r in ran if "llm_plan" in r)),
        "accuracy_of_accepted_llm_plans": _m(sum(1 for r in acc_ans if r.get("correct")), len(acc_ans)),
        "llm_plans_repaired": _m(sum(1 for r in accepted if r.get("llm_repaired")), len(accepted)),
        "llm_refused_or_was_rejected_on_should_refuse_inputs": _m(
            sum(1 for r in refuse if r.get("llm") in ("declared_unsupported", "rejected_clarify", "rejected_fallback")),
            len(refuse),
        ),
        "llm_plan_answered_should_refuse_input": _m(sum(1 for r in refuse if r.get("llm") == "accepted"), len(refuse)),
        "tool_selection_of_llm_plans": _m(
            sum(
                1
                for r in ran
                if r["expect"] == "answer" and "llm_plan" in r and r["llm_plan"].get("tool") == r.get("expected_tool")
            ),
            sum(1 for r in ran if r["expect"] == "answer" and "llm_plan" in r),
        ),
        "token_usage": _usage_totals(ran),
    }


def _usage_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    used = [r["llm_usage"] for r in rows if r.get("llm_usage")]
    tot = lambda k: sum(u.get(k) or 0 for u in used)  # noqa: E731
    secs = sorted(u["seconds"] for u in used if u.get("seconds") is not None)
    return {
        "measured_calls": len(used),
        "input_tokens": tot("input_tokens"),
        "output_tokens": tot("output_tokens"),
        "total_tokens": tot("total_tokens"),
        "mean_tokens_per_call": round(tot("total_tokens") / len(used)) if used else None,
        "llm_call_seconds_p50": secs[len(secs) // 2] if secs else None,
        "llm_call_seconds_max": secs[-1] if secs else None,
    }


SPLITS = ("dev", "heldout", "unseen", "dev_nl")
SPLIT_LABELS = {
    "dev": "development set",
    "heldout": "held-out paraphrases (Phase 2)",
    "unseen": "unseen real-world set (Phase 3, frozen)",
    "dev_nl": "NL tuning set (Phase 3)",
}


def _m(n: int, d: int) -> dict[str, Any]:
    """A metric with an explicit numerator and denominator."""
    return {"n": n, "d": d, "rate": round(n / d, 4) if d else None}


def _refused(r: dict[str, Any]) -> bool:
    return r.get("outcome") in ("rejected", "clarify")


def _split_metrics(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    ans = [r for r in rows if r["expect"] == "answer"]
    ref = [r for r in rows if r["expect"] == "reject"]
    return {
        "answerable": len(ans),
        "should_refuse": len(ref),
        "answer_accuracy": _m(sum(1 for r in ans if r.get("correct")), len(ans)),
        "refusal_accuracy": _m(sum(1 for r in ref if _refused(r)), len(ref)),
        "overall_accuracy": _m(sum(1 for r in ans if r.get("correct")) + sum(1 for r in ref if _refused(r)), len(rows)),
        "tool_selection_accuracy": None
        if mode == "oracle"
        else _m(sum(1 for r in ans if r.get("planned_tool") == r["expected_tool"]), len(ans)),
        "wrong_answers": _m(sum(1 for r in ans if r.get("outcome") == "answered" and not r.get("correct")), len(ans)),
        "answered_when_should_refuse": _m(sum(1 for r in ref if r.get("outcome") == "answered"), len(ref)),
    }


def summarize(
    records: list[dict[str, Any]],
    tamper: list[dict[str, Any]],
    mode: str,
    intent_tamper: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    intent_tamper = intent_tamper or []
    ran = [r for r in records if r.get("outcome") != "skipped"]
    answered = [r for r in ran if r.get("outcome") == "answered"]
    # every answered output is labelled correct/incorrect against ground truth
    # (an answer to a problem that should have been refused is incorrect)
    good = [r for r in answered if r.get("correct")]
    bad = [r for r in answered if not r.get("correct")]
    math_verified = [r for r in answered if r.get("verification") == "verified"]
    overall_verified = [r for r in answered if r.get("overall") == "verified"]

    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_tool: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in ran:
        ok = bool(r.get("correct")) if r["expect"] == "answer" else _refused(r)
        by_cat[r["category"]][0] += ok
        by_cat[r["category"]][1] += 1
        if r["expect"] == "answer":
            by_tool[r["expected_tool"]][0] += ok
            by_tool[r["expected_tool"]][1] += 1

    lat = [r["latency_ms"] for r in ran if "latency_ms" in r]
    q = statistics.quantiles(lat, n=20) if len(lat) >= 20 else [max(lat)] * 19
    splits = {sp_: [r for r in ran if r["split"] == sp_] for sp_ in SPLITS}

    summary = {
        "mode": mode,
        "counts": {
            "benchmark_items": len(records) + len(tamper) + len(intent_tamper),
            "problems": len(records),
            "answerable": sum(1 for r in records if r["expect"] == "answer"),
            "should_refuse": sum(1 for r in records if r["expect"] == "reject"),
            "tampered_solutions": len(tamper),
            "tampered_interpretations": len(intent_tamper),
            "problems_run": len(ran),
            "problems_skipped": len(records) - len(ran),
            "skipped_reason": "oracle mode cannot run natural-language-only cases (no canonical tool arguments)"
            if len(records) != len(ran)
            else None,
        },
        "all_problems": _split_metrics(ran, mode),
        "splits": {sp_: _split_metrics(rows, mode) for sp_, rows in splits.items() if rows},
        "math_verification": {
            "scope": "answers produced on benchmark problems (tamper controls excluded)",
            "verified_precision": _m(sum(1 for r in math_verified if r.get("correct")), len(math_verified)),
            "verified_recall": _m(sum(1 for r in good if r.get("verification") == "verified"), len(good)),
            "incorrect_answers_marked_verified": _m(
                sum(1 for r in bad if r.get("verification") == "verified"), len(bad)
            ),
            "status_on_correct": dict(Counter(r.get("verification") for r in good)),
            "status_on_incorrect": dict(Counter(r.get("verification") for r in bad)),
        },
        "tampered_solutions": {
            "detected_as_failed": _m(sum(1 for t in tamper if t["tampered_status"] == "failed"), len(tamper)),
            "marked_verified": _m(sum(1 for t in tamper if t["tampered_status"] == "verified"), len(tamper)),
            "untampered_controls_verified": _m(
                sum(1 for t in tamper if t["control_status"] == "verified"), len(tamper)
            ),
            "untampered_controls_failed": _m(sum(1 for t in tamper if t["control_status"] == "failed"), len(tamper)),
        },
        "latency_ms": {
            "mean": round(statistics.mean(lat), 1),
            "p50": round(statistics.median(lat), 1),
            "p95": round(q[18], 1),
            "max": round(max(lat), 1),
        },
        "crashes": _m(sum(1 for r in ran if r.get("outcome") == "crash"), len(ran)),
        "accuracy_by_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())},
        "accuracy_by_tool": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_tool.items())},
    }
    if any("overall" in r for r in answered):
        summary["end_to_end_trust"] = {
            "scope": "answers produced on benchmark problems; 'fully verified' = math verified AND intent confirmed",
            "fully_verified_precision": _m(sum(1 for r in overall_verified if r.get("correct")), len(overall_verified)),
            "fully_verified_recall": _m(sum(1 for r in good if r.get("overall") == "verified"), len(good)),
            "incorrect_answers_fully_verified": _m(sum(1 for r in bad if r.get("overall") == "verified"), len(bad)),
            "math_verified_but_wrong": _m(sum(1 for r in bad if r.get("verification") == "verified"), len(bad)),
            "intent_status_on_correct": dict(Counter(r.get("intent") for r in good)),
            "intent_status_on_incorrect": dict(Counter(r.get("intent") for r in bad)),
            "clarification_requests": _m(sum(1 for r in ran if r.get("outcome") == "clarify"), len(ran)),
        }
    if intent_tamper:
        summary["tampered_interpretations"] = {
            "flagged": _m(
                sum(1 for t in intent_tamper if t["tampered_status"] in ("mismatch", "ambiguous")), len(intent_tamper)
            ),
            "accepted_as_confirmed": _m(
                sum(1 for t in intent_tamper if t["tampered_status"] == "confirmed"), len(intent_tamper)
            ),
            "controls_handled_correctly": _m(sum(1 for t in intent_tamper if _control_ok(t)), len(intent_tamper)),
            "valid_controls_wrongly_flagged": _m(
                sum(1 for t in intent_tamper if t["control_expect"] == "accept" and not _control_ok(t)),
                sum(1 for t in intent_tamper if t["control_expect"] == "accept"),
            ),
        }
    return summary


def _control_ok(t: dict[str, Any]) -> bool:
    flagged = t["control_status"] in ("mismatch", "ambiguous")
    return flagged if t["control_expect"] == "clarify" else not flagged


def _fmt(metric: dict[str, Any] | None) -> str:
    if not metric or metric["rate"] is None:
        return "n/a"
    return f"{100 * metric['rate']:.1f}% ({metric['n']}/{metric['d']})"


def to_markdown(
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    tamper: list[dict[str, Any]],
    intent_tamper: list[dict[str, Any]] | None = None,
) -> str:
    c, a = summary["counts"], summary["all_problems"]
    mv, ts = summary["math_verification"], summary["tampered_solutions"]
    lines = [
        f"# MathForge benchmark — mode `{summary['mode']}`",
        "",
        f"**{c['benchmark_items']} benchmark items** = {c['problems']} problems ({c['answerable']} answerable + "
        f"{c['should_refuse']} that should be refused or clarified) + {c['tampered_solutions']} tampered solutions"
        + (f" + {c['tampered_interpretations']} tampered interpretations" if c["tampered_interpretations"] else "")
        + ".",
        f"Problems run in this mode: {c['problems_run']}"
        + (f" ({c['problems_skipped']} skipped: {c['skipped_reason']})." if c["problems_skipped"] else "."),
        "",
        "Every rate is shown with its numerator/denominator."
        + (f" **Subset run: {summary['subset']}.**" if summary.get("subset") else ""),
        "",
        "## Accuracy by split",
        "",
        "| Split | Answer accuracy | Refusal / clarification accuracy | Tool selection | Wrong answers |",
        "|---|---|---|---|---|",
    ]
    for sp_, m in summary["splits"].items():
        lines.append(
            f"| {SPLIT_LABELS[sp_]} | {_fmt(m['answer_accuracy'])} | {_fmt(m['refusal_accuracy'])} | "
            f"{_fmt(m['tool_selection_accuracy'])} | {_fmt(m['wrong_answers'])} |"
        )
    lines.append(
        f"| **all problems** | {_fmt(a['answer_accuracy'])} | {_fmt(a['refusal_accuracy'])} | "
        f"{_fmt(a['tool_selection_accuracy'])} | {_fmt(a['wrong_answers'])} |"
    )
    lines += [
        "",
        "## Mathematical verification",
        "",
        f"Scope: {mv['scope']}.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Verified precision (math `verified` ⇒ answer correct) | {_fmt(mv['verified_precision'])} |",
        f"| Verified recall (correct ⇒ math `verified`) | {_fmt(mv['verified_recall'])} |",
        f"| Incorrect answers marked math-`verified` | {_fmt(mv['incorrect_answers_marked_verified'])} |",
        f"| Tampered solutions detected as `failed` | {_fmt(ts['detected_as_failed'])} |",
        f"| Tampered solutions marked `verified` | {_fmt(ts['marked_verified'])} |",
        f"| Untampered controls `verified` (rest are `approximate`) | {_fmt(ts['untampered_controls_verified'])} |",
        "",
        f"Status on correct answers: `{mv['status_on_correct']}`  ",
        f"Status on incorrect answers: `{mv['status_on_incorrect']}`",
    ]
    if "end_to_end_trust" in summary:
        e = summary["end_to_end_trust"]
        lines += [
            "",
            "## Intent verification and end-to-end trust",
            "",
            f"Scope: {e['scope']}.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Fully-verified precision | {_fmt(e['fully_verified_precision'])} |",
            f"| Fully-verified recall | {_fmt(e['fully_verified_recall'])} |",
            f"| Incorrect answers presented as fully verified | {_fmt(e['incorrect_answers_fully_verified'])} |",
            f"| Incorrect answers whose math was `verified` (misinterpretations) | {_fmt(e['math_verified_but_wrong'])} |",
            f"| Clarification requests | {_fmt(e['clarification_requests'])} |",
            "",
            f"Intent status on correct answers: `{e['intent_status_on_correct']}`  ",
            f"Intent status on incorrect answers: `{e['intent_status_on_incorrect']}`",
        ]
    if "tampered_interpretations" in summary:
        t = summary["tampered_interpretations"]
        lines += [
            "",
            "## Tampered interpretations (wrong tool / arguments for the original question)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Flagged (mismatch or ambiguous) | {_fmt(t['flagged'])} |",
            f"| Accepted as confirmed | {_fmt(t['accepted_as_confirmed'])} |",
            f"| Controls handled correctly (accepted, or clarified when the question lacks information) | "
            f"{_fmt(t['controls_handled_correctly'])} |",
            f"| Valid controls wrongly flagged (false alarms) | {_fmt(t['valid_controls_wrongly_flagged'])} |",
        ]
    if "llm_planner" in summary:
        g = summary["llm_planner"]
        lines += ["", "## LLM planner (isolated from offline fallbacks)", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| outcome: {k} | {_fmt(v)} |" for k, v in g["outcomes"].items()]
        lines += [
            f"| LLM call errors | {_fmt(g['llm_errors'])} |",
            f"| LLM plans accepted by the intent check | {_fmt(g['llm_plans_accepted_by_intent_check'])} |",
            f"| Accuracy of accepted LLM plans | {_fmt(g['accuracy_of_accepted_llm_plans'])} |",
            f"| Accepted LLM plans that needed a safe repair | {_fmt(g['llm_plans_repaired'])} |",
            f"| Should-refuse inputs: LLM refused or its plan was rejected | "
            f"{_fmt(g['llm_refused_or_was_rejected_on_should_refuse_inputs'])} |",
            f"| Should-refuse inputs: LLM plan accepted (bad) | {_fmt(g['llm_plan_answered_should_refuse_input'])} |",
            f"| Tool selection of LLM plans (before intent check) | {_fmt(g['tool_selection_of_llm_plans'])} |",
            f"| Token usage (measured calls) | {g['token_usage']} |",
        ]
    if summary.get("incomplete"):
        lines += ["", f"**INCOMPLETE RUN:** {summary['incomplete']}"]
    lines += [
        "",
        "## Latency and robustness",
        "",
        f"Latency mean / p50 / p95 / max: {summary['latency_ms']['mean']} / {summary['latency_ms']['p50']} / "
        f"{summary['latency_ms']['p95']} / {summary['latency_ms']['max']} ms. Crashes: {_fmt(summary['crashes'])}.",
        "",
        "## Accuracy by category",
        "",
        "| Category | Correct |",
        "|---|---|",
        *[f"| {k} | {val} |" for k, val in summary["accuracy_by_category"].items()],
        "",
        "## Accuracy by tool (answerable problems)",
        "",
        "| Tool | Correct |",
        "|---|---|",
        *[f"| {k} | {val} |" for k, val in summary["accuracy_by_tool"].items()],
        "",
        "## Failures",
        "",
    ]
    failures = [
        r
        for r in records
        if r.get("outcome") != "skipped"
        and ((r["expect"] == "answer" and not r.get("correct")) or (r["expect"] == "reject" and not _refused(r)))
    ]
    if not failures:
        lines.append("None.")
    else:
        lines += ["| Case | Problem | Outcome | Tool (planned/expected) | Detail |", "|---|---|---|---|---|"]
        for r in failures:
            detail = (r.get("detail") or r.get("error") or "").replace("|", "\\|")[:140]
            lines.append(
                f"| {r['id']} | `{r['problem']}` | {r.get('outcome')} | {r.get('planned_tool')} / "
                f"{r['expected_tool']} | {detail} |"
            )
    missed = [t for t in tamper if t["tampered_status"] != "failed"]
    lines += ["", "## Tampered solutions not marked `failed`", ""]
    lines += ["None."] if not missed else [f"- {t['id']} ({t['note']}): {t['tampered_status']}" for t in missed]
    if intent_tamper:
        missed_i = [t for t in intent_tamper if t["tampered_status"] not in ("mismatch", "ambiguous")]
        lines += ["", "## Tampered interpretations not flagged", ""]
        lines += ["None."] if not missed_i else [f"- {t['id']} ({t['note']}): {t['tampered_status']}" for t in missed_i]
    return "\n".join(lines) + "\n"


class _ReplayModel:
    """Stand-in chat model for --replay: every planner response must come from the
    cache; a cache miss raises instead of calling the network."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def with_structured_output(self, schema: Any) -> Any:
        return self

    def invoke(self, messages: Any) -> Any:
        raise RuntimeError("replay mode: no cached planner response for this problem")


def _select_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = CASES
    if args.subset == "smoke":
        from eval.smoke import SMOKE_IDS

        by_id = {c["id"]: c for c in CASES}
        cases = [by_id[i] for i in SMOKE_IDS]
    if args.split:
        cases = [c for c in cases if (c.get("split") or ("heldout" if c.get("heldout") else "dev")) == args.split]
    if args.only:
        prefixes = tuple(args.only.split(","))
        cases = [c for c in cases if c["id"].startswith(prefixes)]
    if getattr(args, "cases_from", None):  # exactly the problems of an earlier report (e.g. to re-score it)
        done = {r["id"] for r in json.loads(Path(args.cases_from).read_text())["cases"]}
        cases = [c for c in cases if c["id"] in done]
    if args.limit:
        cases = cases[: args.limit]
    return cases


def _tokens_per_call(problem: str = "") -> int:
    """Rough prompt + completion size of one planner call (characters / 4)."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from app.agents.llm_planner import PLANNER_SYSTEM_PROMPT, PlannerOutput

    chars = len(PLANNER_SYSTEM_PROMPT) + len(json.dumps(convert_to_openai_tool(PlannerOutput))) + len(problem)
    return chars // 4 + 250


def _write_reports(
    name: str,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    tamper: list[dict[str, Any]],
    intent_tamper: list[dict[str, Any]],
) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / f"{name}.json").write_text(
        json.dumps(
            {"summary": summary, "cases": records, "tamper": tamper, "intent_tamper": intent_tamper},
            indent=2,
            default=str,
        )
        + "\n"
    )
    (REPORT_DIR / f"{name}.md").write_text(to_markdown(summary, records, tamper, intent_tamper))
    print(f"reports written to {REPORT_DIR}/{name}.{{json,md}}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["oracle", "rules", "groq"], default="rules")
    ap.add_argument(
        "--allow-network", action="store_true", help="required for --mode groq (real, rate-limited API calls)"
    )
    ap.add_argument("--dry-run", action="store_true", help="groq mode: show what would be called, call nothing")
    ap.add_argument(
        "--replay", action="store_true", help="groq mode: use only cached planner responses (no network, no key)"
    )
    ap.add_argument("--cache", default=str(REPORT_DIR / "groq_cache.jsonl"), help="planner response cache (JSONL)")
    ap.add_argument("--rpm", type=float, default=20.0, help="max uncached requests per minute (groq)")
    ap.add_argument("--tpm", type=float, default=6000.0, help="token budget per minute used for pacing (groq)")
    ap.add_argument("--retries", type=int, default=5, help="SDK retries per call on rate limits / transient errors")
    ap.add_argument("--max-consecutive-errors", type=int, default=5, help="stop the run after N failed LLM calls")
    ap.add_argument("--llm-explanations", action="store_true", help="groq mode: also generate LLM explanations")
    ap.add_argument("--subset", choices=["smoke"], help="run the stratified 25-question smoke subset (eval/smoke.py)")
    ap.add_argument("--only", help="comma-separated case-id prefixes to run")
    ap.add_argument("--split", choices=SPLITS, help="run only one split")
    ap.add_argument("--limit", type=int, help="run only the first N selected problems")
    ap.add_argument("--cases-from", help="run only the problems present in this earlier report (JSON)")
    ap.add_argument("--out-name", help="report file name (default: the mode, or <mode>_smoke for the smoke subset)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    cases = _select_cases(args)
    name = args.out_name or (f"{args.mode}_smoke" if args.subset == "smoke" else args.mode)

    model_name = ""
    if args.mode == "groq":
        from app.config import get_settings

        model_name = os.environ.get("LLM_MODEL") or get_settings().llm_model
        per_call = _tokens_per_call()
        if args.dry_run:
            from app.agents.llm_planner import _cache_get, _cache_key

            previous = os.environ.get("MATHFORGE_LLM_CACHE")
            os.environ["MATHFORGE_LLM_CACHE"] = args.cache  # read-only lookup, restored right after
            try:
                cached_n = sum(1 for c in cases if _cache_get(_cache_key(model_name, c["problem"], None)) is not None)
            finally:
                if previous is None:
                    os.environ.pop("MATHFORGE_LLM_CACHE", None)
                else:
                    os.environ["MATHFORGE_LLM_CACHE"] = previous
            measured = None
            smoke = REPORT_DIR / "groq_smoke.json"
            if smoke.exists():
                smoke_summary = json.loads(smoke.read_text())["summary"]
                usage = smoke_summary.get("llm_planner", {}).get("token_usage", {})
                if smoke_summary.get("environment", {}).get("model") == model_name:  # same model only
                    measured = usage.get("mean_tokens_per_call")
            per = measured or per_call
            new_calls = (len(cases) - cached_n) * (2 if args.llm_explanations else 1)
            wait = max(60.0 / args.rpm, per * 60.0 / args.tpm)
            print(
                f"groq dry run: {len(cases)} problems, {cached_n} already cached -> {new_calls} new chat completions "
                f"with {model_name}. Tokens per call: {per} ({'measured in the smoke run' if measured else 'estimate'}), "
                f"~{new_calls * per:,} tokens in total. Pacing: one call every {wait:.1f}s (rpm={args.rpm:g}, "
                f"tpm={args.tpm:g}) => ~{new_calls * wait / 60:.0f} min plus call latency. Nothing was sent."
            )
            return 0
        if not args.replay:
            proxy_auth = (os.environ.get("GROQ_AUTH") or get_settings().groq_auth) == "proxy"
            if not os.environ.get("GROQ_API_KEY") and not proxy_auth:
                print(
                    "groq mode needs credentials: GROQ_API_KEY (local) or GROQ_AUTH=proxy (cloud network secret); "
                    "refusing to run.",
                    file=sys.stderr,
                )
                return 2
            if not args.allow_network:
                print("groq mode makes real API calls; re-run with --allow-network to confirm.", file=sys.stderr)
                return 2
        os.environ["MATHFORGE_LLM_CACHE"] = args.cache
        os.environ["LLM_MAX_RETRIES"] = str(args.retries)
    _configure(args.mode)
    if args.mode == "groq" and args.replay:
        import app.agents.llm_planner as llm_planner

        llm_planner.get_chat_model = lambda: _ReplayModel(model_name)
    if args.mode == "groq" and not args.llm_explanations:
        import app.agents.graph as graph
        from app.agents.tutor import template_explanation

        graph.explain = lambda problem, tool, answer, steps, verification, intent=None: (
            template_explanation(tool, answer, verification, intent),
            False,
        )

    pace = args.mode == "groq" and not args.replay
    records: list[dict[str, Any]] = []
    incomplete: str | None = None
    progress = REPORT_DIR / f"{name}.progress.jsonl"
    if not args.no_write:
        REPORT_DIR.mkdir(exist_ok=True)
        progress.write_text("")
    consecutive_errors = 0
    account_limits: dict[str, str] = {}
    effective_tpm = args.tpm
    try:
        for idx, case in enumerate(cases, 1):
            cached = False
            if args.mode == "groq":
                from app.agents.llm_planner import _cache_get, _cache_key

                cached = _cache_get(_cache_key(model_name, case["problem"], None)) is not None
            t_call = time.perf_counter()
            rec = _run_case(case, args.mode)
            if args.mode == "groq":
                from app.agents.llm_planner import pop_call_stats

                rec["llm_cached"] = cached
                stats = pop_call_stats(case["problem"])
                if stats:
                    rec["llm_usage"] = stats
                print(
                    f"[{idx}/{len(cases)}] {case['id']}: llm={rec.get('llm')} outcome={rec.get('outcome')} "
                    f"correct={rec.get('correct')}" + (" (cached)" if cached else ""),
                    flush=True,
                )
            records.append(rec)
            if not args.no_write:
                with progress.open("a") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
            consecutive_errors = consecutive_errors + 1 if rec.get("llm") == "error" else 0
            if args.mode == "groq" and not args.replay:
                stop, tpm_now = _quota_check(rec, account_limits, args.tpm)
                if tpm_now < effective_tpm:
                    print(f"account token limit is {tpm_now:g}/min: lowering pacing from {effective_tpm:g}", flush=True)
                    effective_tpm = tpm_now
                if stop:
                    incomplete = stop
                    print(incomplete, file=sys.stderr)
                    break
            if args.mode == "groq" and consecutive_errors >= args.max_consecutive_errors:
                incomplete = (
                    f"stopped after {consecutive_errors} consecutive LLM errors (last: {rec.get('llm_error')}); "
                    "re-run the same command to resume from the cache"
                )
                print(incomplete, file=sys.stderr)
                break
            if pace and not cached:
                used = (rec.get("llm_usage") or {}).get("total_tokens") or per_call
                wait = max(60.0 / args.rpm, used * 60.0 / effective_tpm)  # pace on the tokens actually used
                time.sleep(max(0.0, wait - (time.perf_counter() - t_call)))
    except KeyboardInterrupt:
        incomplete = "interrupted; re-run the same command to resume (cached planner responses are reused)"
        print(incomplete, file=sys.stderr)

    tamper = run_tamper()
    intent_tamper = run_intent_tamper()
    summary = summarize(records, tamper, args.mode, intent_tamper)
    summary["environment"] = {"python": platform.python_version(), "sympy": sp.__version__}
    if args.subset:
        summary["subset"] = f"{args.subset} ({len(cases)} problems; not the full benchmark)"
    if incomplete:
        summary["incomplete"] = f"{incomplete} ({len(records)}/{len(cases)} problems done)"
    if args.mode == "groq":
        from app.agents.llm_planner import PROMPT_VERSION

        summary["environment"].update(
            model=model_name,
            prompt_version=PROMPT_VERSION,
            replay=args.replay,
            pacing={"rpm": args.rpm, "tpm_requested": args.tpm, "tpm_used": effective_tpm},
            account_rate_limits=account_limits or None,
        )
        summary["llm_planner"] = _llm_summary(records)

    print(to_markdown(summary, records, tamper, intent_tamper).split("## Latency")[0])
    if not args.no_write:
        _write_reports(name, summary, records, tamper, intent_tamper)
        progress.unlink(missing_ok=True)
    return 3 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
