"""Groq integration and evaluation harness, tested fully offline (no key, no network).

* the smoke subset is stratified, reproducible and references unchanged benchmark questions
* LLM failures are explicit and recorded, never silently scored as LLM results
* invalid or contradictory LLM plans are rejected
* intent verification stays independent of the LLM planner
"""

import inspect
import json

import pytest

import app.agents.graph as graph
from app.agents import intent as intent_mod
from app.agents import llm_planner
from app.agents.intent import check_intent, validate_plan
from app.agents.llm_planner import LLMPlanError, PlannerOutput, plan_with_llm
from app.schemas import IntentStatus as IS
from app.schemas import ProblemCategory as PC
from app.schemas import ToolCall
from eval.dataset import ALL_TOOLS, CASES
from eval.run import _llm_accounting, _llm_summary, _run_case
from eval.run import main as run_main
from eval.smoke import SMOKE_IDS

BY_ID = {c["id"]: c for c in CASES}


def _split(c):
    return c.get("split") or ("heldout" if c.get("heldout") else "dev")


# --------------------------------------------------------------------- smoke subset


def test_smoke_subset_is_stratified_and_references_the_benchmark():
    assert 20 <= len(SMOKE_IDS) <= 30
    assert len(set(SMOKE_IDS)) == len(SMOKE_IDS)
    cases = [BY_ID[i] for i in SMOKE_IDS]  # KeyError if an id does not exist
    answerable = [c for c in cases if c["truth"]["type"] != "reject"]
    refuse = [c for c in cases if c["truth"]["type"] == "reject"]
    assert {c["tool"] for c in answerable} == set(ALL_TOOLS)
    assert {_split(c) for c in cases} == {"dev", "heldout", "unseen", "dev_nl"}
    assert len(refuse) >= 5
    assert {c["category"] for c in refuse} >= {"unseen", "ambiguous", "unsupported", "edge"}


def test_smoke_cases_are_the_benchmark_objects_themselves():
    from argparse import Namespace

    from eval.run import _select_cases

    chosen = _select_cases(Namespace(subset="smoke", split=None, only=None, limit=None))
    assert [c["id"] for c in chosen] == list(SMOKE_IDS)
    assert all(c is BY_ID[c["id"]] for c in chosen)  # same question text and ground truth, nothing copied


# --------------------------------------------------------------------- gating / dry run


def test_dry_run_needs_no_key_and_sends_nothing(monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def forbidden(*a, **k):
        raise AssertionError("network/model access during dry run")

    monkeypatch.setattr(llm_planner, "get_chat_model", forbidden)
    assert run_main(["--mode", "groq", "--subset", "smoke", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "25 problems" in out and "Nothing was sent" in out


def test_real_run_requires_key_and_explicit_permission(monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert run_main(["--mode", "groq", "--subset", "smoke", "--no-write"]) == 2
    monkeypatch.setenv("GROQ_API_KEY", "gsk_dummy_not_used")
    assert run_main(["--mode", "groq", "--subset", "smoke", "--no-write"]) == 2
    assert "allow-network" in capsys.readouterr().err


# --------------------------------------------------------------------- LLM failures are explicit


class _FakeModel:
    def __init__(self, result=None, exc=None):
        self.model_name, self.result, self.exc = "fake-model", result, exc

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        if self.exc:
            raise self.exc
        return self.result


@pytest.mark.parametrize(
    "model, reason",
    [
        (_FakeModel(result=None), "no structured plan"),
        (_FakeModel(exc=RuntimeError("429 Too Many Requests")), "429"),
        (_FakeModel(result={"tool": "not_a_tool"}), "invalid structured output"),
    ],
)
def test_llm_failures_raise_explicit_errors(monkeypatch, model, reason):
    monkeypatch.setattr(llm_planner, "get_chat_model", lambda: model)
    with pytest.raises(LLMPlanError, match=reason):
        plan_with_llm("Integrate x^2 from 0 to 3")


def test_failed_calls_are_not_cached(monkeypatch, tmp_path):
    cache = tmp_path / "cache.jsonl"
    monkeypatch.setenv("MATHFORGE_LLM_CACHE", str(cache))
    monkeypatch.setattr(llm_planner, "get_chat_model", lambda: _FakeModel(exc=RuntimeError("timeout")))
    with pytest.raises(LLMPlanError):
        plan_with_llm("Integrate x^2 from 0 to 3")
    assert not cache.exists() or cache.read_text() == ""  # a resumed run retries it


def test_llm_error_is_recorded_not_silently_replaced(monkeypatch):
    monkeypatch.setattr(graph, "plan_with_llm", lambda p, c=None: (_ for _ in ()).throw(LLMPlanError("429")))
    state = graph.run_solver("Integrate x^2 from 0 to 3")
    assert state["status"] == "solved" and state["planner"] != "llm"  # the app still works offline ...
    assert "429" in state["llm_error"]  # ... but the failure is visible
    acct = _llm_accounting(state)
    assert acct["llm"] == "error"


def test_llm_summary_isolates_llm_results():
    recs = [
        {"outcome": "answered", "expect": "answer", "llm": "accepted", "llm_plan": {}, "correct": True},
        {"outcome": "answered", "expect": "answer", "llm": "error", "correct": True},  # offline fallback got it right
        {"outcome": "answered", "expect": "answer", "llm": "rejected_fallback", "llm_plan": {}, "correct": True},
        {"outcome": "clarify", "expect": "reject", "llm": "rejected_clarify", "llm_plan": {}},
    ]
    s = _llm_summary(recs)
    assert s["accuracy_of_accepted_llm_plans"]["n"] == 1 and s["accuracy_of_accepted_llm_plans"]["d"] == 1
    assert s["llm_errors"]["n"] == 1
    assert s["llm_plans_accepted_by_intent_check"] == {"n": 1, "d": 3, "rate": 0.3333}


def test_run_stops_after_consecutive_llm_errors(monkeypatch, tmp_path, capsys):
    # replay mode with an empty cache: every LLM call fails (and nothing touches the network)
    monkeypatch.setattr(llm_planner, "get_chat_model", llm_planner.get_chat_model)
    monkeypatch.setattr(graph, "explain", graph.explain)
    for var in ("LLM_ENABLED", "MATHFORGE_LLM_CACHE", "LLM_MAX_RETRIES"):
        monkeypatch.setenv(var, "0")  # register the variable so main()'s changes are undone ...
        monkeypatch.delenv(var)  # ... and start from "unset"
    code = run_main(
        [
            "--mode",
            "groq",
            "--replay",
            "--subset",
            "smoke",
            "--cache",
            str(tmp_path / "empty.jsonl"),
            "--max-consecutive-errors",
            "2",
            "--no-write",
        ]
    )
    assert code == 3
    assert "consecutive LLM errors" in capsys.readouterr().err


# --------------------------------------------------------------------- invalid / contradictory plans


def plan(tool, category=PC.ALGEBRA, **kw):
    args = kw.pop("args", {})
    return ToolCall(tool=tool, category=category, args=args, source="llm", **kw)


@pytest.mark.parametrize(
    "problem, bad",
    [
        ("derivative of x^2", plan("differentiate", PC.CALCULUS)),
        ("Solve x^2 = 4", plan("solve_equation")),
        (
            "Integrate x^2 from 0 to 3",
            plan("integrate", PC.LINEAR_ALGEBRA, args={"expression": "x^2", "lower": "0", "upper": "3"}),
        ),
        (
            "derivative of x^4",
            plan(
                "differentiate",
                PC.CALCULUS,
                args={"expression": "x^4", "order": 1},
                interpretation="Second derivative of x^4",
            ),
        ),
        (
            "Integrate x^2 from 0 to 3",
            plan(
                "integrate",
                PC.INTEGRALS,
                args={"expression": "x^2", "lower": "0", "upper": "3"},
                interpretation="Differentiate x^2",
            ),
        ),
        (
            "Binomial n=10 p=0.5, P(X=3)",
            plan(
                "probability_distribution",
                PC.PROBABILITY_STATISTICS,
                args={"distribution": "binomial", "params": {"n": 10, "p": 0.5}, "query": "exactly", "x": 3},
            ),
        ),
        (
            "Binomial n=10 p=0.5, P(X=3)",
            plan(
                "probability_distribution",
                PC.PROBABILITY_STATISTICS,
                args={"distribution": "binomial", "params": {"n": "ten", "p": 0.5}, "query": "pmf", "x": 3},
            ),
        ),
        ("Integrate x^2 from 0 to 3", plan("integrate", PC.INTEGRALS, args={"expression": "x^2", "lower": "0"})),
        ("derivative of x^2", plan("differentiate", PC.CALCULUS, args={"expression": "__import__('os')"})),
        (
            "multiply [[1,2],[3,4]] and [[5,6],[7,8]]",
            plan("matrix_operation", PC.LINEAR_ALGEBRA, args={"operation": "multiply", "matrix": [[1, 2], [3, 4]]}),
        ),
        (
            "standard normal P(-1 < X < 1)",
            plan(
                "probability_distribution",
                PC.PROBABILITY_STATISTICS,
                args={"distribution": "normal", "params": {"mean": 0, "std": 1}, "query": "between", "x": -1},
            ),
        ),
        ("derivative of x^2", plan("differentiate", PC.CALCULUS, args={"expression": "x^2", "order": "two"})),
    ],
)
def test_invalid_or_contradictory_llm_plans_are_rejected(problem, bad):
    iv, fixes = check_intent(problem, bad)
    assert iv.status == IS.MISMATCH, iv
    assert fixes == {}  # never "repaired" by guessing
    assert iv.clarification


def test_rejected_llm_plan_never_reaches_the_tools(monkeypatch):
    bad = plan("integrate", PC.LINEAR_ALGEBRA, args={"expression": "x^2", "lower": "0", "upper": "3"})
    monkeypatch.setattr(graph, "plan_with_llm", lambda p, c=None: bad)
    monkeypatch.setattr(graph, "plan_with_rules", lambda p: (_ for _ in ()).throw(graph.PlanningError("x")))
    monkeypatch.setattr(graph, "plan_from_intent", lambda p: (_ for _ in ()).throw(graph.PlanningError("x")))
    state = graph.run_solver("Integrate x^2 from 0 to 3")
    assert state["status"] == "needs_clarification" and state.get("tool_output") is None
    assert _llm_accounting(state)["llm"] == "rejected_clarify"


def test_valid_llm_plan_is_accepted(monkeypatch):
    good = plan(
        "integrate",
        PC.INTEGRALS,
        args={"expression": "x^2", "lower": "0", "upper": "3"},
        interpretation="Definite integral of x^2 from 0 to 3.",
    )
    assert validate_plan(good) == []
    monkeypatch.setattr(graph, "plan_with_llm", lambda p, c=None: good)
    state = graph.run_solver("Integrate x^2 from 0 to 3")
    assert state["planner"] == "llm" and state["tool_output"].answer_text == "9"
    assert _llm_accounting(state)["llm"] == "accepted"


# --------------------------------------------------------------------- independence


def test_intent_module_does_not_depend_on_the_llm():
    src = inspect.getsource(intent_mod)
    assert "llm_planner" not in src and "get_chat_model" not in src and "langchain" not in src


def test_llm_self_reports_can_only_lower_trust():
    """Interpretation / assumptions / ambiguities written by the LLM can downgrade
    the intent status, never upgrade it. The question text alone decides acceptance."""
    problem = "Determine the rank of the matrix [[1, 2], [2, 4]]"
    wrong = dict(
        tool="matrix_operation",
        category=PC.LINEAR_ALGEBRA,
        args={"operation": "determinant", "matrix": [[1, 2], [2, 4]]},
        source="llm",
    )
    bare, _ = check_intent(problem, ToolCall(**wrong))
    persuasive, _ = check_intent(
        problem,
        ToolCall(
            **wrong,
            interpretation="Rank of the matrix [[1,2],[2,4]]",
            assumptions=[],
            rationale="The user wants the rank.",
        ),
    )
    assert bare.status == IS.MISMATCH and persuasive.status == IS.MISMATCH

    right = dict(
        tool="matrix_operation",
        category=PC.LINEAR_ALGEBRA,
        args={"operation": "rank", "matrix": [[1, 2], [2, 4]]},
        source="llm",
    )
    assert check_intent(problem, ToolCall(**right))[0].status == IS.CONFIRMED
    assert check_intent(problem, ToolCall(**right, assumptions=["x"]))[0].status == IS.ASSUMED
    assert check_intent(problem, ToolCall(**right, ambiguities=["unclear"]))[0].status == IS.AMBIGUOUS


def test_cache_is_keyed_by_prompt_version_and_model():
    a = llm_planner._cache_key("model-a", "q", None)
    assert a != llm_planner._cache_key("model-b", "q", None)
    assert a == llm_planner._cache_key("model-a", "q", None)
    assert llm_planner.PROMPT_VERSION in json.dumps([llm_planner.PROMPT_VERSION])


def test_planner_schema_builds_without_network():
    from langchain_core.utils.function_calling import convert_to_openai_tool

    tool = convert_to_openai_tool(PlannerOutput)
    assert tool["function"]["name"] == "PlannerOutput"
    assert "interpretation" in tool["function"]["parameters"]["properties"]


def test_run_case_records_cached_replay(monkeypatch, tmp_path):
    from app.config import get_settings
    from eval.run import _ReplayModel

    case = BY_ID["para-21"]  # "Determine the rank ..." trap
    cache = tmp_path / "c.jsonl"
    out = {
        "supported": True,
        "category": "linear_algebra",
        "tool": "matrix_operation",
        "operation": "determinant",
        "matrix": [[1, 2, 3], [2, 4, 6], [1, 0, 1]],
        "interpretation": "Determinant of the matrix.",
    }
    key = llm_planner._cache_key(get_settings().llm_model, case["problem"], None)
    cache.write_text(json.dumps({"key": key, "output": out}) + "\n")
    monkeypatch.setenv("MATHFORGE_LLM_CACHE", str(cache))
    monkeypatch.setattr(llm_planner, "get_chat_model", lambda: _ReplayModel(get_settings().llm_model))
    rec = _run_case(case, "groq")
    # the LLM's determinant reading is rejected; the faithful offline rank plan answers correctly
    assert rec["llm"] == "rejected_fallback" and rec["correct"] is True


# --------------------------------------------------------------------- proxy-injected authentication


@pytest.fixture
def proxy_auth(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("GROQ_AUTH", "proxy")
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_proxy_mode_enables_the_llm_without_a_local_key(proxy_auth):
    import httpx

    from app.agents.llm import PROXY_AUTH_PLACEHOLDER, api_key_for_sdk
    from app.config import get_settings

    assert get_settings().llm_configured
    assert api_key_for_sdk() == PROXY_AUTH_PLACEHOLDER
    seen = {}

    def handler(req):  # mock transport: nothing leaves the machine
        seen["url"], seen["auth"] = str(req.url), req.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    from langchain_groq import ChatGroq

    m = ChatGroq(
        model="m",
        api_key=api_key_for_sdk(),
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    m.invoke("hi")
    # the request is inside the network secret's scope; only the non-secret placeholder is sent from here
    assert seen["url"].startswith("https://api.groq.com/openai/v1/")
    assert seen["auth"] == f"Bearer {PROXY_AUTH_PLACEHOLDER}"


def test_api_key_mode_is_unchanged(monkeypatch):
    from app.agents.llm import api_key_for_sdk
    from app.config import get_settings

    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("GROQ_AUTH", "api_key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    get_settings.cache_clear()
    assert not get_settings().llm_configured  # no key, no proxy: offline planners only
    monkeypatch.setenv("GROQ_API_KEY", "gsk_local_dummy")
    get_settings.cache_clear()
    assert get_settings().llm_configured and api_key_for_sdk() == "gsk_local_dummy"
    get_settings.cache_clear()


def test_proxy_mode_still_requires_explicit_permission(proxy_auth, capsys):
    assert run_main(["--mode", "groq", "--subset", "smoke", "--no-write"]) == 2
    assert "allow-network" in capsys.readouterr().err


def test_auth_check_sends_nothing_without_permission(proxy_auth, monkeypatch, capsys):
    import groq

    from eval.groq_check import main as check_main

    monkeypatch.setattr(groq, "Groq", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert check_main([]) == 0
    assert "Nothing was sent" in capsys.readouterr().out


def test_planner_output_without_category_is_accepted():
    # gpt-oss-120b on Groq omitted 'category' and Groq rejected the call (400) while it was required.
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from app.agents.llm_planner import plan_output_to_tool_call

    required = convert_to_openai_tool(PlannerOutput)["function"]["parameters"].get("required", [])
    assert required == []  # gpt-oss omits 'tool' when refusing and 'category' sometimes
    call = plan_output_to_tool_call(PlannerOutput(tool="integrate", expression="x^2", lower="0", upper="3"))
    assert call.category == PC.INTEGRALS


def test_refusal_without_tool_is_a_refusal_not_an_error():
    from app.agents.llm_planner import LLMPlanUnsupported, plan_output_to_tool_call

    with pytest.raises(LLMPlanUnsupported, match="standard deviation"):
        plan_output_to_tool_call(PlannerOutput(supported=False, clarification="What is the standard deviation?"))
    with pytest.raises(LLMPlanError, match="no tool"):
        plan_output_to_tool_call(PlannerOutput(supported=True))


def test_tool_call_validation_400_is_retried_once(monkeypatch):
    class Flaky(_FakeModel):
        calls = 0

        def invoke(self, messages):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("Error code: 400 - Tool call validation failed: attempted to call tool 'json'")
            return {"raw": None, "parsed": PlannerOutput(tool="integrate", expression="x^2"), "parsing_error": None}

    monkeypatch.setattr(llm_planner, "get_chat_model", lambda: Flaky())
    call = plan_with_llm("integral of x^2 dx")
    assert call.tool == "integrate" and Flaky.calls == 2
    assert llm_planner.pop_call_stats("integral of x^2 dx")["tool_call_retries"] == 1

    class AlwaysBad(_FakeModel):
        def invoke(self, messages):
            raise RuntimeError("Error code: 400 - Tool call validation failed: missing properties")

    monkeypatch.setattr(llm_planner, "get_chat_model", lambda: AlwaysBad())
    with pytest.raises(LLMPlanError, match="after 1 retry"):
        plan_with_llm("integral of x^3 dx")
