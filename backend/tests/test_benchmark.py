"""Benchmark integrity and regression floors (offline, deterministic)."""

import copy

import pytest

from eval import ground_truth
from eval.dataset import ALL_TOOLS, CASES, TAMPER
from eval.run import _run_case, run_tamper, summarize
from eval.run import main as run_main


def test_dataset_size_and_coverage():
    assert len(CASES) + len(TAMPER) >= 150
    answerable = [c for c in CASES if c["truth"]["type"] != "reject"]
    assert len(answerable) >= 150
    assert {c["tool"] for c in answerable} == set(ALL_TOOLS)
    assert {"edge", "ambiguous", "unsupported", "paraphrase"} <= {c["category"] for c in CASES}
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
    s = summarize(records, run_tamper(), "oracle")
    assert _rate(s["all_problems"]["answer_accuracy"]) == 1.0
    assert _rate(s["all_problems"]["refusal_accuracy"]) == 1.0
    assert s["math_verification"]["incorrect_answers_marked_verified"]["n"] == 0
    assert s["tampered_solutions"]["detected_as_failed"]["n"] == len(TAMPER)
    assert s["crashes"]["n"] == 0


def test_rules_mode_regression_floor():
    records = [_run_case(c, "rules") for c in CASES]
    s = summarize(records, [], "rules")
    dev = s["splits"]["dev"]
    assert _rate(dev["answer_accuracy"]) == 1.0
    assert _rate(dev["tool_selection_accuracy"]) == 1.0
    assert _rate(dev["refusal_accuracy"]) == 1.0
    assert s["crashes"]["n"] == 0
    # Held-out and unseen problems: the planner may refuse, but must not silently answer wrongly.
    for split in ("heldout", "unseen"):
        assert s["splits"][split]["wrong_answers"]["n"] == 0, split


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
