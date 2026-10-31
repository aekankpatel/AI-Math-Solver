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
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sympy as sp

from eval.dataset import CASES, TAMPER
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
        "heldout": bool(case.get("heldout")),
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
        else:
            from app.agents.graph import run_solver

            state = run_solver(case["problem"])
            call = state.get("tool_call")
            rec["planned_tool"] = call.tool if call else None
            rec["planner"] = state.get("planner")
            out, verification, tool = state.get("tool_output"), state.get("verification"), rec["planned_tool"]
            if state.get("status") != "solved":
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


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def summarize(records: list[dict[str, Any]], tamper: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    ran = [r for r in records if r.get("outcome") != "skipped"]
    answerable = [r for r in ran if r["expect"] == "answer"]
    refusals = [r for r in ran if r["expect"] == "reject"]
    correct = [r for r in answerable if r.get("correct")]
    refused_ok = [r for r in refusals if r.get("outcome") == "rejected"]

    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_tool: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in ran:
        ok = bool(r.get("correct")) if r["expect"] == "answer" else r.get("outcome") == "rejected"
        by_cat[r["category"]][0] += ok
        by_cat[r["category"]][1] += 1
        if r["expect"] == "answer":
            by_tool[r["expected_tool"]][0] += ok
            by_tool[r["expected_tool"]][1] += 1

    tool_sel = [r for r in answerable if r.get("planned_tool") is not None or r.get("outcome") != "crash"]
    tool_ok = [r for r in tool_sel if r.get("planned_tool") == r["expected_tool"]]

    # Verification labels: an answer is "correct" only if it matches ground truth.
    labelled = [(r.get("verification"), bool(r.get("correct"))) for r in ran if r.get("outcome") == "answered"]
    labelled += [(t["tampered_status"], False) for t in tamper]
    labelled += [(t["control_status"], True) for t in tamper]
    good = [s for s, c in labelled if c]
    bad = [s for s, c in labelled if not c]
    verified = [c for s, c in labelled if s == "verified"]

    def subset_acc(rows: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [r for r in rows if r["expect"] == "answer"]
        return {
            "cases": len(rows),
            "final_answer_accuracy": _rate(sum(1 for r in rows if r.get("correct")), len(rows)),
            "tool_selection_accuracy": None
            if mode == "oracle"
            else _rate(sum(1 for r in rows if r.get("planned_tool") == r["expected_tool"]), len(rows)),
        }

    lat = [r["latency_ms"] for r in ran if "latency_ms" in r]
    q = statistics.quantiles(lat, n=20) if len(lat) >= 20 else [max(lat)] * 19

    return {
        "mode": mode,
        "cases": {
            "total": len(records),
            "run": len(ran),
            "answerable": len(answerable),
            "should_refuse": len(refusals),
            "tamper": len(tamper),
        },
        "final_answer_accuracy": _rate(len(correct), len(answerable)),
        "refusal_accuracy": _rate(len(refused_ok), len(refusals)),
        "overall_accuracy": _rate(len(correct) + len(refused_ok), len(ran)),
        "tool_selection_accuracy": None if mode == "oracle" else _rate(len(tool_ok), len(answerable)),
        "development_set": subset_acc([r for r in ran if not r["heldout"]]),
        "heldout_paraphrases": subset_acc([r for r in ran if r["heldout"]]),
        "verification": {
            "labelled_outputs": len(labelled),
            "correct_outputs": len(good),
            "incorrect_outputs": len(bad),
            "verified_precision": _rate(sum(verified), len(verified)),
            "verified_recall": _rate(sum(1 for s in good if s == "verified"), len(good)),
            "false_verification_rate": _rate(sum(1 for s in bad if s == "verified"), len(bad)),
            "error_detection_rate": _rate(sum(1 for s in bad if s == "failed"), len(bad)),
            "incorrect_not_trusted_rate": _rate(sum(1 for s in bad if s in ("failed", "unverified")), len(bad)),
            "status_on_correct": dict(Counter(good)),
            "status_on_incorrect": dict(Counter(bad)),
            "tamper_detected": f"{sum(1 for t in tamper if t['tampered_status'] == 'failed')}/{len(tamper)}",
            "tamper_controls_verified": f"{sum(1 for t in tamper if t['control_status'] == 'verified')}/{len(tamper)}",
        },
        "latency_ms": {
            "mean": round(statistics.mean(lat), 1),
            "p50": round(statistics.median(lat), 1),
            "p95": round(q[18], 1),
            "max": round(max(lat), 1),
        },
        "failure_rates": {
            "crash_rate": _rate(sum(1 for r in ran if r.get("outcome") == "crash"), len(ran)),
            "unexpected_refusal_rate": _rate(
                sum(1 for r in answerable if r.get("outcome") == "rejected"), len(answerable)
            ),
            "wrong_answer_rate": _rate(
                sum(1 for r in answerable if r.get("outcome") == "answered" and not r.get("correct")), len(answerable)
            ),
            "answered_when_should_refuse_rate": _rate(
                sum(1 for r in refusals if r.get("outcome") == "answered"), len(refusals)
            ),
        },
        "accuracy_by_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())},
        "accuracy_by_tool": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_tool.items())},
    }


def to_markdown(summary: dict[str, Any], records: list[dict[str, Any]], tamper: list[dict[str, Any]]) -> str:
    v = summary["verification"]
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    lines = [
        f"# MathForge benchmark — mode `{summary['mode']}`",
        "",
        f"Cases: {summary['cases']['run']} run ({summary['cases']['answerable']} answerable, "
        f"{summary['cases']['should_refuse']} should be refused) + {summary['cases']['tamper']} tampered solutions.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Final-answer accuracy | {pct(summary['final_answer_accuracy'])} |",
        f"| Refusal accuracy (edge / ambiguous / unsupported) | {pct(summary['refusal_accuracy'])} |",
        f"| Overall accuracy | {pct(summary['overall_accuracy'])} |",
        f"| Tool-selection accuracy | {pct(summary['tool_selection_accuracy'])} |",
        f"| — development set ({summary['development_set']['cases']}) accuracy / tool selection | "
        f"{pct(summary['development_set']['final_answer_accuracy'])} / "
        f"{pct(summary['development_set']['tool_selection_accuracy'])} |",
        f"| — held-out paraphrases ({summary['heldout_paraphrases']['cases']}) accuracy / tool selection | "
        f"{pct(summary['heldout_paraphrases']['final_answer_accuracy'])} / "
        f"{pct(summary['heldout_paraphrases']['tool_selection_accuracy'])} |",
        f"| Verified precision (verified ⇒ correct) | {pct(v['verified_precision'])} |",
        f"| Verified recall (correct ⇒ verified) | {pct(v['verified_recall'])} |",
        f"| False verification rate (incorrect marked verified) | {pct(v['false_verification_rate'])} |",
        f"| Error detection rate (incorrect marked failed) | {pct(v['error_detection_rate'])} |",
        f"| Tampered solutions detected | {v['tamper_detected']} |",
        f"| Latency mean / p50 / p95 / max (ms) | {summary['latency_ms']['mean']} / {summary['latency_ms']['p50']} / "
        f"{summary['latency_ms']['p95']} / {summary['latency_ms']['max']} |",
        f"| Crash rate | {pct(summary['failure_rates']['crash_rate'])} |",
        f"| Unexpected refusal rate | {pct(summary['failure_rates']['unexpected_refusal_rate'])} |",
        f"| Wrong-answer rate | {pct(summary['failure_rates']['wrong_answer_rate'])} |",
        f"| Answered when it should refuse | {pct(summary['failure_rates']['answered_when_should_refuse_rate'])} |",
        "",
        f"Verification status on correct outputs: `{v['status_on_correct']}`  ",
        f"Verification status on incorrect outputs: `{v['status_on_incorrect']}`",
        "",
        "## Accuracy by category",
        "",
        "| Category | Correct |",
        "|---|---|",
        *[f"| {k} | {val} |" for k, val in summary["accuracy_by_category"].items()],
        "",
        "## Accuracy by tool (answerable cases)",
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
        and (
            (r["expect"] == "answer" and not r.get("correct"))
            or (r["expect"] == "reject" and r.get("outcome") != "rejected")
        )
    ]
    if not failures:
        lines.append("None.")
    else:
        lines += ["| Case | Problem | Outcome | Tool (planned/expected) | Detail |", "|---|---|---|---|---|"]
        for r in failures:
            detail = (r.get("detail") or r.get("error") or "").replace("|", "\\|")[:140]
            lines.append(
                f"| {r['id']} | `{r['problem']}` | {r.get('outcome')} | {r.get('planned_tool')} / {r['expected_tool']} | {detail} |"
            )
    missed = [t for t in tamper if t["tampered_status"] != "failed"]
    lines += ["", "## Tampered solutions not marked `failed`", ""]
    lines += ["None."] if not missed else [f"- {t['id']} ({t['note']}): {t['tampered_status']}" for t in missed]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["oracle", "rules", "groq"], default="rules")
    ap.add_argument(
        "--allow-network", action="store_true", help="required for --mode groq (real, rate-limited API calls)"
    )
    ap.add_argument("--rpm", type=float, default=20.0, help="max requests per minute in groq mode")
    ap.add_argument("--llm-explanations", action="store_true", help="groq mode: also generate LLM explanations")
    ap.add_argument("--only", help="comma-separated case-id prefixes to run")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    if args.mode == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            print("groq mode needs GROQ_API_KEY in the environment; refusing to run.", file=sys.stderr)
            return 2
        if not args.allow_network:
            print("groq mode makes real API calls; re-run with --allow-network to confirm.", file=sys.stderr)
            return 2
    _configure(args.mode)
    if args.mode == "groq" and not args.llm_explanations:
        import app.agents.graph as graph
        from app.agents.tutor import template_explanation

        graph.explain = lambda problem, tool, answer, steps, verification: (
            template_explanation(tool, answer, verification),
            False,
        )

    cases = CASES
    if args.only:
        prefixes = tuple(args.only.split(","))
        cases = [c for c in CASES if c["id"].startswith(prefixes)]
    delay = 60.0 / args.rpm if args.mode == "groq" else 0.0
    records = []
    for case in cases:
        records.append(_run_case(case, args.mode))
        if delay:
            time.sleep(delay)
    tamper = run_tamper()
    summary = summarize(records, tamper, args.mode)
    summary["environment"] = {"python": platform.python_version(), "sympy": sp.__version__}
    if args.mode == "groq":
        from app.config import get_settings

        summary["environment"]["model"] = get_settings().llm_model

    print(json.dumps({k: v for k, v in summary.items() if k not in ("accuracy_by_tool",)}, indent=2))
    if not args.no_write:
        REPORT_DIR.mkdir(exist_ok=True)
        (REPORT_DIR / f"{args.mode}.json").write_text(
            json.dumps({"summary": summary, "cases": records, "tamper": tamper}, indent=2, default=str) + "\n"
        )
        (REPORT_DIR / f"{args.mode}.md").write_text(to_markdown(summary, records, tamper))
        print(f"reports written to {REPORT_DIR}/{args.mode}.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
