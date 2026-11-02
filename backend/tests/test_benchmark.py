"""Benchmark integrity and regression floors (offline, deterministic)."""

import copy

import pytest

from eval import ground_truth
from eval.dataset import ALL_TOOLS, CASES, INTENT_TAMPER, TAMPER
from eval.run import _run_case, run_intent_tamper, run_tamper, summarize
from eval.run import main as run_main


def test_dataset_size_and_coverage():
    assert len(CASES) + len(TAMPER) >= 150
    answerable = [c for c in CASES if c["truth"]["type"] != "reject"]
    assert len(answerable) >= 150
    assert {c["tool"] for c in answerable} == set(ALL_TOOLS)
    assert {"edge", "ambiguous", "unsupported", "paraphrase", "unseen", "dev_nl"} <= {c["category"] for c in CASES}
    assert len(INTENT_TAMPER) >= 30
    assert len({c["id"] for c in CASES + TAMPER}) == len(CASES) + len(TAMPER)


def test_ground_truth_is_independently_confirmed():
    assert ground_truth.main() == 0


@pytest.mark.parametrize(
    "case_id, bad_truth",
    [
        ("eq-01", {"type": "roots", "values": ["2", "4"]}),
        ("eq-05", {"type": "roots", "values": ["I"]}),
        ("diff-01", {"type": "expr", "value": "3*x**2*cos(x)", "form": None}),
        ("dint-01", {"type": "number", "value": "10"}),
        ("lin-04", {"type": "matrix", "value": [["3/5", "-1/5"], ["-7/10", "2/5"]]}),
        ("prob-01", {"type": "number", "value": "1/8"}),
        ("lim-09", {"type": "number", "value": "oo"}),
    ],
)
def test_ground_truth_validator_rejects_wrong_truth(case_id, bad_truth):
    case = copy.deepcopy(next(c for c in CASES if c["id"] == case_id))
    case["truth"] = bad_truth
    assert ground_truth.validate(case) is not None


def _rate(metric):
    return metric["rate"]


def test_oracle_mode_regression_floor():
    records = [_run_case(c, "oracle") for c in CASES]
    s = summarize(records, run_tamper(), "oracle", run_intent_tamper())
    assert _rate(s["all_problems"]["answer_accuracy"]) == 1.0
    assert _rate(s["all_problems"]["refusal_accuracy"]) == 1.0
    assert s["math_verification"]["incorrect_answers_marked_verified"]["n"] == 0
    assert s["tampered_solutions"]["detected_as_failed"]["n"] == len(TAMPER)
    assert s["crashes"]["n"] == 0
    # correct canonical plans must not be flagged as misreadings
    flagged = [r["id"] for r in records if r["expect"] == "answer" and r.get("intent") in ("mismatch", "ambiguous")]
    assert flagged == []
    t = s["tampered_interpretations"]
    assert t["flagged"]["n"] == t["flagged"]["d"] == len(INTENT_TAMPER)
    assert t["valid_controls_wrongly_flagged"]["n"] == 0


def test_rules_mode_regression_floor():
    records = [_run_case(c, "rules") for c in CASES]
    s = summarize(records, [], "rules")
    dev = s["splits"]["dev"]
    assert _rate(dev["answer_accuracy"]) == 1.0
    assert _rate(dev["tool_selection_accuracy"]) == 1.0
    assert _rate(dev["refusal_accuracy"]) == 1.0
    assert s["crashes"]["n"] == 0
    # Every split: the planner may refuse or ask, but must never silently answer wrongly ...
    for split, m in s["splits"].items():
        assert m["wrong_answers"]["n"] == 0, split
        assert m["answered_when_should_refuse"]["n"] == 0, split
    # ... and nothing incorrect is ever presented as fully verified.
    assert s["end_to_end_trust"]["incorrect_answers_fully_verified"]["n"] == 0
    # Floors from the Phase 3 run (see eval/reports/rules.md); raise them, never lower them.
    assert _rate(s["splits"]["unseen"]["answer_accuracy"]) >= 0.95
    assert _rate(s["splits"]["heldout"]["answer_accuracy"]) >= 0.96


def test_counts_are_consistent():
    records = [_run_case(c, "rules") for c in CASES[:5]]
    s = summarize(records, run_tamper(), "rules")
    c = s["counts"]
    assert c["benchmark_items"] == c["problems"] + c["tampered_solutions"] + c["tampered_interpretations"]


def test_groq_mode_requires_key_and_explicit_permission(monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert run_main(["--mode", "groq", "--no-write"]) == 2
    monkeypatch.setenv("GROQ_API_KEY", "dummy-not-used")
    assert run_main(["--mode", "groq", "--no-write"]) == 2  # no --allow-network
    assert "allow-network" in capsys.readouterr().err


def test_groq_replay_mode_runs_offline_from_cache(tmp_path, monkeypatch):
    """--replay drives the full pipeline with cached planner outputs: no key, no network."""
    import json

    import app.agents.graph as graph
    from app.agents import llm_planner
    from app.config import get_settings

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # main() patches these for groq mode; register them so they are restored afterwards
    monkeypatch.setattr(llm_planner, "get_chat_model", llm_planner.get_chat_model)
    monkeypatch.setattr(graph, "explain", graph.explain)
    for var in ("LLM_ENABLED", "MATHFORGE_LLM_CACHE", "LLM_MAX_RETRIES"):
        monkeypatch.setenv(var, "0")  # register so main()'s changes are undone ...
        monkeypatch.delenv(var)  # ... starting from "unset"
    model = get_settings().llm_model
    cache = tmp_path / "cache.jsonl"
    good = {
        "supported": True,
        "category": "algebra",
        "tool": "solve_equation",
        "equation": "x^2 - 5x + 6 = 0",
        "interpretation": "Solve x^2 - 5x + 6 = 0 for x.",
    }
    misread = {
        "supported": True,
        "category": "linear_algebra",
        "tool": "matrix_operation",
        "operation": "determinant",
        "matrix": [[1, 2], [2, 4]],
    }
    rows = [("Solve x^2 - 5x + 6 = 0", good), ("Determine the rank of the matrix [[1, 2], [2, 4]]", misread)]
    cache.write_text(
        "".join(
            json.dumps({"key": llm_planner._cache_key(model, p, None), "problem": p, "output": o}) + "\n"
            for p, o in rows
        )
    )
    assert run_main(["--mode", "groq", "--replay", "--cache", str(cache), "--only", "eq-01", "--no-write"]) == 0
    # the cached LLM plan is used and verified
    from app.agents.graph import run_solver

    state = run_solver("Solve x^2 - 5x + 6 = 0")
    assert state["planner"] == "llm" and state["status"] == "solved"
    # a cached misreading is caught by the intent check and replaced by a faithful plan
    state = run_solver("Determine the rank of the matrix [[1, 2], [2, 4]]")
    assert state["tool_call"].args["operation"] == "rank"
    # an uncached problem never reaches the network in replay mode
    state = run_solver("Solve x^2 - 9 = 0")
    assert state["planner"] != "llm"
