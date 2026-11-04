"""Offline planner for matrix questions stated through spectral facts (no explicit matrix).

Recognises, in plain or LaTeX-normalised text:

* facts: the size ("4 × 4"), "real", given eigenvalues ("eigenvalues 1, -2 and 3",
  "λ1 = 1, λ2 = -2"), the trace / determinant ("Tr(A) = 2", "a determinant of 1"),
  a matrix polynomial equation ("A^4 - 2A^3 + 3A^2 - 2A + I = 0", "A^2 = A") and a
  characteristic polynomial ("p(λ) = λ^4 - 2λ^3 + c2 λ^2 + c1 λ + 1");
* queries, in the order asked: eigenvalues (of A or of a named matrix), trace(...),
  det(...), the characteristic polynomial; named matrices ("B = A^3 - 2A") are resolved.

The extraction is also used by the intent check to see that every part of a multi-part
question is answered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.math_engine.latex import normalize_latex
from app.schemas import ProblemCategory, ToolCall

_NUM = r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?"
_MATRIX_CHARS = r"[0-9AI_^+\-*() ]"
_SPECTRAL_WORDS = re.compile(
    r"eigen|\btrace\b|\btr\s*\(|\bdet\b|\bdet\s*\(|determinant|characteristic polynomial|spectrum", re.I
)


@dataclass
class SpectralFacts:
    size: int | None = None
    real: bool = False
    eigenvalues: list[str] = field(default_factory=list)
    trace: str | None = None
    determinant: str | None = None
    annihilating: str | None = None
    charpoly: str | None = None
    queries: list[str] = field(default_factory=list)


def is_spectral_question(text: str) -> bool:
    t = normalize_latex(text)
    if re.search(r"\[\s*\[", t):  # an explicit matrix: handled by matrix_operation
        return False
    return bool(re.search(r"\bmatri(?:x|ces)\b", t, re.I) and _SPECTRAL_WORDS.search(t) and re.search(r"\bA\b", t))


def _compact(s: str) -> str:
    s = re.sub(r"\bI_?\(?(?:\d+|n)\)?\b", "I", s)
    return re.sub(r"[\s*]", "", s).replace("**", "^")


def _balanced(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "(":
        return None
    depth = 0
    for i in range(start, len(text)):
        depth += text[i] == "("
        depth -= text[i] == ")"
        if depth == 0:
            return text[start + 1 : i], i + 1
    return None


def _trim(expr: str) -> str:
    """Drop trailing words/brackets picked up after an expression: '... = 0 (' -> '... = 0'."""
    e = expr.strip()
    while True:
        before = e
        e = e.rstrip(" .,;")
        if e.count("(") > e.count(")") and e.endswith("("):
            e = e[:-1]
        if e.count(")") > e.count("(") and e.endswith(")"):
            e = e[:-1]
        if e.count(")") > e.count("(") and e.startswith(")"):
            e = e[1:]
        e = e.strip()
        if e == before:
            return e


def _definitions(text: str) -> dict[str, str]:
    """Named matrices defined from A: 'B = A^3 - 2A', 'C = (A + I_3)^-1'."""
    defs: dict[str, str] = {}
    for m in re.finditer(rf"\b([B-HJ-Z])\s*=\s*({_MATRIX_CHARS}+)", text):
        rhs = _trim(m.group(2))
        if "A" in rhs:
            defs[m.group(1)] = re.sub(r"\s+", " ", rhs)
    return defs


def matrix_equations(text: str) -> list[str]:
    """Polynomial equations in A, e.g. 'A^4 - 2A^3 + 3A^2 - 2A + I_4 = 0' or 'A^2 = A'."""
    out = []
    for m in re.finditer(rf"({_MATRIX_CHARS}+)=({_MATRIX_CHARS}+)", text):
        lhs, rhs = _trim(m.group(1)), _trim(m.group(2))
        if "A" not in lhs or not lhs or not rhs:
            continue
        if lhs.strip("() ") == "A" and "A" not in rhs:
            continue  # "Tr(A) = 2" / "det(A) = 1": a fact about A, not a matrix equation
        lhs_power = re.search(r"A\s*\^\s*\d", lhs) or re.search(r"A\s*\^\s*\d", rhs)
        if not (lhs_power or "I" in lhs + rhs):
            continue
        prev = text[max(0, m.start() - 6) : m.start()]
        if re.search(r"[B-HJ-Z]\s*$", prev) or re.search(r"\b(?:tr|det|Tr|trace)\s*$", prev):
            continue
        out.append(f"{lhs} = {rhs}")
    return out


def extract(problem: str) -> SpectralFacts:
    text = normalize_latex(problem)
    low = text.lower()
    f = SpectralFacts()
    m = re.search(r"\b(\d+)\s*(?:×|x|\*|by)\s*(\d+)\b", text)
    if m and m.group(1) == m.group(2):
        f.size = int(m.group(1))
    f.real = bool(re.search(r"\breal(?:[- ]valued)?\s+(?:\w+\s+){0,2}matri|\breal entries\b", low))

    # ---- eigenvalues given
    lam = re.findall(rf"\b(?:lambda|λ)_?\(?\d\)?\s*=\s*({_NUM})", text)
    if lam:
        f.eigenvalues = [v.replace(" ", "") for v in lam]
    else:
        m = re.search(
            rf"(?:with|has|having|whose)\s+(?:the\s+)?eigenvalues?\s*(?:are|is|of|:)?\s*((?:{_NUM}\s*(?:,|and|,\s*and)?\s*)+)",
            text,
            re.I,
        ) or re.search(
            rf"\beigenvalues\s+(?:of\s+(?:the\s+|a\s+|an\s+)?(?:[\w×]+\s+){{0,4}}?A\s+)?(?:are|is)\s+"
            rf"((?:{_NUM}\s*(?:,|and|,\s*and)?\s*)+)",
            text,
            re.I,
        )
        if m:
            f.eigenvalues = [v.replace(" ", "") for v in re.findall(_NUM, m.group(1))]

    # ---- trace / determinant facts about A
    m = re.search(
        rf"\b(?:trace|tr)\s*(?:\(\s*A\s*\)|of\s+A)?\s*(?:is|=|equal to|equals|of)?\s*({_NUM})(?![\w(^])", text, re.I
    )
    if m:
        f.trace = m.group(1).replace(" ", "")
    m = re.search(
        rf"\b(?:determinant|det)\s*(?:\(\s*A\s*\)|of\s+A|A)?\s*(?:is|=|equal to|equals|of)?\s*({_NUM})(?![\w(^])",
        text,
        re.I,
    )
    if m:
        f.determinant = m.group(1).replace(" ", "")

    eqs = matrix_equations(text)
    if eqs:
        f.annihilating = eqs[0]
    m = re.search(
        r"\b[pPfq]\s*\(\s*(?:lambda|λ|x|t)\s*\)\s*=\s*([^\n=]+?)(?=\s*(?:\n|$|\.\s|\bif\b|\bwhere\b|\bIf\b))", text
    )
    if m and re.search(r"lambda|λ", m.group(1)) and "characteristic" in low:
        f.charpoly = m.group(1).strip().rstrip(".,;")

    # ---- queries, in the order they appear
    defs = _definitions(text)
    found: list[tuple[int, str]] = []

    def target_at(pos: int) -> str | None:
        """The matrix a quantity refers to, starting at text[pos:]."""
        rest = text[pos:]
        b = _balanced(rest, len(rest) - len(rest.lstrip()))
        if b:
            return _trim(b[0])
        mm = re.match(r"\s*(?:of\s+)?(?:the\s+)?(?:matrix\s+)?([A-Z])(\s*\^\s*-?\d+)?\b", rest)
        if mm:
            name = mm.group(1)
            if name == "A":
                return "A" + (mm.group(2) or "").replace(" ", "")
            return defs.get(name)
        return None

    for m in re.finditer(r"\b(trace|tr|determinant|det)\b", text, re.I):
        tgt = target_at(m.end())
        if tgt is None:
            continue
        after = text[m.end() :]
        if _compact(tgt) == "A" and re.match(
            rf"\s*(?:\(\s*A\s*\)|of\s+A|A)?\s*(?:is|=|of|equals|equal to)?\s*{_NUM}", after, re.I
        ):
            continue  # a given fact
        kind = "trace" if m.group(1).lower() in ("trace", "tr") else "det"
        found.append((m.start(), f"{kind}({tgt})"))
    for m in re.finditer(r"\beigenvalues?\b", text, re.I):
        before = text[max(0, m.start() - 60) : m.start()].lower()
        if not re.search(r"\b(find|determine|compute|calculate|what|list|give|all possible|possible)\b[^.]*$", before):
            continue
        rest = text[m.end() :]
        mm = re.match(r"\s*of\s+(?:the\s+)?(?:matrix\s+)?([B-HJ-Z])\b", rest)
        if mm and mm.group(1) in defs:
            found.append((m.start(), f"eigenvalues({defs[mm.group(1)]})"))
        else:
            found.append((m.start(), "eigenvalues"))
    if re.search(r"\b(find|determine|compute|what is|give)\b[^.]*characteristic polynomial", low) and not f.charpoly:
        mm = re.search(r"characteristic polynomial", low)
        found.append((mm.start() if mm else 0, "characteristic polynomial"))
    seen: set[str] = set()
    for _, q in sorted(found):
        key = _compact(q.lower())
        if key not in seen:
            seen.add(key)
            f.queries.append(q)
    return f


def plan_spectrum(problem: str) -> ToolCall | None:
    if not is_spectral_question(problem):
        return None
    f = extract(problem)
    if not f.queries or not (f.eigenvalues or f.annihilating or f.charpoly):
        return None
    args: dict[str, Any] = {"queries": f.queries}
    if f.size is not None:
        args["size"] = f.size
    for key in ("eigenvalues", "annihilating", "charpoly", "trace", "determinant"):
        value = getattr(f, key)
        if value:
            args[key] = value
    if f.real:
        args["real"] = True
    return ToolCall(
        tool="matrix_spectrum",
        category=ProblemCategory.LINEAR_ALGEBRA,
        args=args,
        rationale="Matrix described by spectral facts: use the spectral mapping theorem.",
        source="rules",
    )
