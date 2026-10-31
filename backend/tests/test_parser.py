import pytest
import sympy as sp

from app.math_engine.parser import (
    ExpressionError,
    parse_equation,
    parse_expression,
    parse_matrix,
    validate_expression_text,
)

x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2x^2 + 3x - 5", 2 * x**2 + 3 * x - 5),
        ("sin(x)^2", sp.sin(x) ** 2),
        ("sinx", sp.sin(x)),
        ("e^x", sp.exp(x)),
        ("ln(x)", sp.log(x)),
        ("xy + 1", x * y + 1),
        ("sqrt(16)", 4),
        ("π", sp.pi),
        ("x²", x**2),
    ],
)
def test_parses_common_notation(text, expected):
    assert sp.simplify(parse_expression(text) - expected) == 0


def test_parse_equation():
    eq = parse_equation("x^2 = 4")
    assert isinstance(eq, sp.Eq)
    assert eq.lhs == x**2 and eq.rhs == 4


@pytest.mark.parametrize(
    "malicious",
    [
        '__import__("os").system("ls")',
        "os.system(1)",
        "x.__class__",
        "lambda: 1",
        "exec(1)",
        "eval(x)",
        "open(x)",
        "x; y",
        "x[0]",
        "getattr(x, y)",
        "a" * 600,
        "x @ y",
        "{x}",
        "`x`",
    ],
)
def test_rejects_unsafe_input(malicious):
    with pytest.raises(ExpressionError):
        parse_expression(malicious)


@pytest.mark.parametrize("bad", ["", "   ", "(x + 1", "x = = 2", "= 3", "hello(x)"])
def test_rejects_invalid_expressions(bad):
    with pytest.raises(ExpressionError):
        parse_equation(bad) if "=" in bad else parse_expression(bad)


def test_expression_rejects_equation():
    with pytest.raises(ExpressionError):
        parse_expression("x = 2")


def test_matrix_validation():
    assert parse_matrix([[1, 2], [3, 4]]).det() == -2
    with pytest.raises(ExpressionError):
        parse_matrix([[1, 2], [3]])
    with pytest.raises(ExpressionError):
        parse_matrix([])
    with pytest.raises(ExpressionError):
        parse_matrix([[1] * 11] * 11)


def test_validate_normalises_unicode():
    assert validate_expression_text("2×3 − 1") == "2*3 - 1"
