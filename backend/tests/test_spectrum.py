"""Matrix questions answered from spectral facts (matrix_spectrum), LaTeX input and multi-part answers."""

import copy

import pytest
import sympy as sp

from app.agents.graph import run_solver
from app.agents.llm_planner import PlannerOutput, plan_output_to_tool_call
from app.agents.spectral_planner import plan_spectrum
from app.agents.tutor import normalize_math_delimiters
from app.math_engine.latex import normalize_latex
from app.math_engine.tools import ToolError, run_tool
from app.math_engine.verifier import verify
from app.schemas import ToolCall

Q_ANNIHILATING = r"""Let \(A\) be a \(4 \times 4\) real matrix with a trace of \(2\) (\(\text{Tr}(A) = 2\)) and a determinant of \(1\) (\(\det(A) = 1\)).
Suppose the characteristic polynomial of \(A\) can be written in the form:
\(p(\lambda )=\lambda ^{4}-2\lambda ^{3}+c_{2}\lambda ^{2}+c_{1}\lambda +1\)
If \(A^4 - 2A^3 + 3A^2 - 2A + I_4 = 0\) (where \(I_{4}\) is the \(4 \times 4\) identity matrix), find all possible complex eigenvalues of \(A\) and calculate the exact value of \(\text{Tr}(A^5)\)."""

Q_EIGEN = r"""Let \(A\) be a \(3 \times 3\) matrix with eigenvalues \(\lambda_1 = 1\), \(\lambda_2 = -2\), and \(\lambda_3 = 3\). 1. Find the determinant of the matrix \(B = A^3 - 2A\). 2. Find the trace of the matrix \(C = (A + I_3)^{-1}\), where \(I_{3}\) is the \(3 \times 3\) identity matrix."""

OMEGA = sp.Rational(1, 2) + sp.sqrt(3) * sp.I / 2


def _solve(problem):
    state = run_solver(problem)
    return state, state.get("tool_output")


# --------------------------------------------------------------------------- end to end (offline planner)


def test_annihilating_polynomial_question_fully_verified():
    state, out = _solve(Q_ANNIHILATING)
    assert state["status"] == "solved" and state["tool_call"].tool == "matrix_spectrum"
    assert state["overall"].value == "verified" and state["intent"].status.value == "confirmed"
    spectra = out.data["spectra"]
    assert len(spectra) == 1
    assert sorted(spectra[0], key=str) == sorted([OMEGA, OMEGA, sp.conjugate(OMEGA), sp.conjugate(OMEGA)], key=str)
    trace = next(r for r in out.data["results"] if r["kind"] == "trace")
    assert trace["distinct"] == [2]
    assert "c1 = -2" in out.answer_text and "c2 = 3" in out.answer_text


def test_two_part_eigenvalue_question_fully_verified():
    state, out = _solve(Q_EIGEN)
    assert state["overall"].value == "verified"
    assert out.answer_text == "1. det(A^3 - 2A) = 84; 2. tr((A + I)^-1) = -1/4"


@pytest.mark.parametrize(
    "problem, expected",
    [
        ("A 3x3 matrix A has eigenvalues 2, -1 and 4. Find det(A^2 + I) and the trace of A^3.", ["170", "71"]),
        ("Let A be a 2 by 2 matrix with A^2 = A and trace 1. What is det(A)?", ["det(A) = 0"]),
        (
            "Suppose A is a 3 × 3 real matrix satisfying A^3 = I with tr(A) = 0. Find the eigenvalues of A and det(A + 2I).",
            ["det(A + 2I) = 9"],
        ),
        (
            "The eigenvalues of a 3x3 matrix A are 1, 2 and 5. Compute the determinant of A^-1 and the trace of A^2.",
            ["1/10", "30"],
        ),
        ("Let A be a 2x2 matrix with eigenvalues 3 and -3. Find the eigenvalues of B = A^2 - I.", ["8 (×2)"]),
    ],
)
def test_rephrased_spectral_questions(problem, expected):
    state, out = _solve(problem)
    assert state["status"] == "solved" and state["overall"].value == "verified", state.get("error")
    for e in expected:
        assert e in out.answer_text


def test_several_possible_spectra_list_every_value():
    out = run_tool("matrix_spectrum", {"size": 2, "annihilating": "A^2 - 3A + 2I = 0", "queries": ["det(A)"]})
    assert "possible values" in out.answer_text
    assert sorted(out.data["results"][0]["distinct"]) == [1, 2, 4]
    assert verify("matrix_spectrum", out).status.value == "verified"


def test_inverse_of_singular_matrix_is_refused():
    with pytest.raises(ToolError, match="not invertible"):
        run_tool("matrix_spectrum", {"size": 2, "eigenvalues": ["-1", "2"], "queries": ["trace((A + I)^-1)"]})


def test_inconsistent_conditions_are_reported():
    with pytest.raises(ToolError, match="No matrix satisfies"):
        run_tool("matrix_spectrum", {"size": 2, "annihilating": "A^2 = I", "trace": "5", "queries": ["det(A)"]})


def test_missing_information_is_reported():
    with pytest.raises(ToolError, match="Not enough information"):
        run_tool("matrix_spectrum", {"size": 3, "trace": "2", "queries": ["det(A)"]})


def test_only_a_and_i_are_allowed_in_expressions():
    for bad in ("det(A + B)", "trace(__import__('os'))", "det(A^T)"):
        with pytest.raises(ToolError):
            run_tool("matrix_spectrum", {"size": 2, "eigenvalues": ["1", "2"], "queries": [bad]})


# --------------------------------------------------------------------------- verification catches wrong results


def _tampered(out, mutate):
    bad = copy.copy(out)
    bad.data = copy.deepcopy(out.data)
    mutate(bad.data)
    return verify("matrix_spectrum", bad)


def test_verifier_detects_a_wrong_trace_and_a_wrong_spectrum():
    out = run_tool(
        "matrix_spectrum",
        {"size": 3, "eigenvalues": ["1", "-2", "3"], "queries": ["det(A^3 - 2A)", "trace((A + I)^-1)"]},
    )
    assert verify("matrix_spectrum", out).status.value == "verified"

    def wrong_det(d):
        d["results"][0]["values"][0] = sp.Integer(85)

    assert _tampered(out, wrong_det).status.value == "failed"

    q_out = run_tool(
        "matrix_spectrum",
        {
            "size": 4,
            "real": True,
            "trace": "2",
            "determinant": "1",
            "annihilating": "A^4 - 2A^3 + 3A^2 - 2A + I = 0",
            "queries": ["eigenvalues"],
        },
    )

    def missing_case(d):  # claim a spectrum that violates the trace condition
        d["spectra"][0] = [sp.Integer(1)] * 4
        d["results"][0]["values"][0] = [sp.Integer(1)] * 4

    assert _tampered(q_out, missing_case).status.value == "failed"


# --------------------------------------------------------------------------- intent checks on LLM plans


def _with_llm_plan(problem, args):
    call = ToolCall(tool="matrix_spectrum", category="linear_algebra", args=args)
    return run_solver(problem, llm_result={"call": call}, explain_with_llm=False)


GOOD = {"size": 3, "eigenvalues": ["1", "-2", "3"], "queries": ["det(A^3 - 2A)", "trace((A + I)^-1)"]}


def test_faithful_llm_plan_is_confirmed():
    state = _with_llm_plan(Q_EIGEN, GOOD)
    assert state["planner"] == "llm" and state["intent"].status.value == "confirmed" and not state["intent"].repaired


@pytest.mark.parametrize(
    "change",
    [
        {"eigenvalues": ["1", "-2", "4"]},  # misread eigenvalue
        {"queries": ["det(A^3 - 2A)"]},  # dropped the second part
        {"real": True},  # unstated assumption
    ],
)
def test_unfaithful_llm_plans_are_repaired_from_the_question(change):
    state = _with_llm_plan(Q_EIGEN, {**GOOD, **change})
    assert state["intent"].repaired
    assert state["tool_output"].answer_text == "1. det(A^3 - 2A) = 84; 2. tr((A + I)^-1) = -1/4"


def test_llm_plan_with_an_invented_expression_is_rejected():
    state = _with_llm_plan(Q_EIGEN, {**GOOD, "queries": ["det(A^3 - 5A)", "trace((A + I)^-1)"]})
    assert state["planner"] != "llm"  # the faithful offline plan answered instead
    assert "84" in state["tool_output"].answer_text


def test_planner_output_schema_maps_to_the_tool():
    out = PlannerOutput(
        tool="matrix_spectrum",
        category="linear_algebra",
        size=3,
        eigenvalues=["1", "-2", "3"],
        queries=["det(A^3 - 2A)"],
        expression="ignored for this tool",
    )
    call = plan_output_to_tool_call(out)
    assert call.args == {"size": 3, "eigenvalues": ["1", "-2", "3"], "queries": ["det(A^3 - 2A)"]}


def test_offline_planner_reads_the_latex_question():
    call = plan_spectrum(Q_ANNIHILATING)
    assert call.args["size"] == 4 and call.args["real"] is True
    assert call.args["trace"] == "2" and call.args["determinant"] == "1"
    assert call.args["annihilating"].startswith("A^4 - 2A^3 + 3A^2 - 2A + I")
    assert call.args["queries"] == ["eigenvalues", "trace(A^5)"]


def test_explicit_matrices_still_use_matrix_operation():
    state, out = _solve("eigenvalues of [[2,1],[1,2]]")
    assert state["tool_call"].tool == "matrix_operation" and out.answer_text == "3, 1"


# --------------------------------------------------------------------------- text handling


def test_latex_normalisation():
    assert normalize_latex(r"Find \(\text{Tr}(A^{5})\) for \(I_{4}\)") == "Find Tr(A^5) for I_4"
    assert normalize_latex(r"\(\frac{1}{2} + \sqrt{x}\)") == "((1)/(2)) + sqrt(x)"
    assert normalize_latex("Solve x^2 = 4") == "Solve x^2 = 4"  # plain text unchanged


def test_tutor_output_uses_renderable_delimiters():
    text = r"It equals \[ \frac{12}{4}\times 4 = 12, \] so \(x = 12\)."
    assert normalize_math_delimiters(text) == r"It equals $$\frac{12}{4}\times 4 = 12,$$ so $x = 12$."
