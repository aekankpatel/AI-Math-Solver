"""Matrix questions answered from spectral facts, without the matrix itself.

Typical questions: "A is a 3×3 matrix with eigenvalues 1, -2, 3; find det(A³ − 2A) and
tr((A + I)⁻¹)", or "a 4×4 real matrix with trace 2 and det 1 satisfies
A⁴ − 2A³ + 3A² − 2A + I = 0; find its possible eigenvalues and tr(A⁵)".

Method (exact, SymPy):

1. Candidate spectra. Known eigenvalues give one spectrum. With a polynomial that annihilates
   A, every eigenvalue is a root of it (the minimal polynomial divides it), and conversely any
   multiset of its roots is the spectrum of a matrix satisfying it (a diagonal matrix, or a real
   block-diagonal one for conjugate pairs). So the candidates are all multisets of n roots.
2. Constraints remove candidates: trace = sum, determinant = product, a real matrix has
   non-real eigenvalues in conjugate pairs, and known coefficients of the characteristic
   polynomial must match.
3. Spectral mapping: for a polynomial or rational function f, the eigenvalues of f(A) are
   f(λ) with the same algebraic multiplicities. Hence det f(A) = ∏ f(λᵢ), tr f(A) = Σ f(λᵢ).
   f(A) with an inverse needs f's denominator to be non-zero at every eigenvalue.

Several quantities can be asked for in one call ("queries"). Every result is checked by
``verify_spectrum`` with a different method: concrete matrices with each candidate spectrum
are built and all conditions and answers are recomputed by direct matrix arithmetic, and the
candidate list is re-enumerated numerically with NumPy.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

import numpy as np
import sympy as sp

from app.math_engine.parser import ExpressionError, parse_expression
from app.schemas import Step, Verification, VerificationCheck, VerificationStatus

X = sp.Symbol("x")
MAX_SIZE = 8
MAX_CANDIDATES = 5000
MAX_VERIFIED_SPECTRA = 12


class SpectrumError(ValueError):
    pass


# --------------------------------------------------------------------------- parsing


def _clean(text: str) -> str:
    s = str(text).strip().replace("**", "^").replace("·", "*").replace("−", "-")
    s = re.sub(r"\bI_?\(?(?:\d+|n)\)?\b", "I", s)  # I_4, I_n, I(3) -> I
    s = re.sub(r"\b(?:Id|E)\b", "I", s)
    return s


def matrix_function(text: str) -> sp.Expr:
    """'A^3 - 2A' -> x**3 - 2*x ; '(A + I)^-1' -> 1/(x + 1). Only A, I and numbers are allowed."""
    s = _clean(text)
    s = re.sub(r"\binv(?:erse)?\s*\(", "(1)/(", s)
    if re.search(r"(?<![A-Za-z])A\s*\^\s*\(?\s*T\b|transpose", s):
        raise SpectrumError("Transposes are not supported in matrix expressions.")
    s = re.sub(r"(?<![A-Za-z])A(?![A-Za-z])", "(x)", s)
    s = re.sub(r"(?<![A-Za-z])I(?![A-Za-z])", "(1)", s)
    if re.search(r"[A-Za-z]", s.replace("x", "").replace("sqrt", "").replace("pi", "")):
        raise SpectrumError(f"Unsupported symbol in the matrix expression {text!r}; only A, I and numbers are allowed.")
    try:
        f = parse_expression(s)
    except ExpressionError as exc:
        raise SpectrumError(f"Could not read the matrix expression {text!r}.") from exc
    if not f.free_symbols <= {X}:
        raise SpectrumError(f"The matrix expression {text!r} may only involve A and I.")
    num, den = sp.fraction(sp.together(f))
    if not (num.is_polynomial(X) and den.is_polynomial(X)):
        raise SpectrumError(f"{text!r} is not a polynomial or rational expression in A.")
    return sp.together(f)


def annihilating_polynomial(text: str) -> sp.Expr:
    s = _clean(text)
    if "=" in s:
        lhs, rhs = s.split("=", 1)
        rhs = rhs.strip()
        s = f"({lhs}) - ({rhs})" if rhs not in ("0", "O", "") else lhs
    q = sp.expand(matrix_function(s))
    if not q.is_polynomial(X) or sp.degree(q, X) < 1:
        raise SpectrumError("The matrix equation must be a polynomial in A of degree at least 1.")
    return q


def characteristic_polynomial(text: str) -> tuple[sp.Expr, list[sp.Symbol]]:
    """'lambda^4 - 2lambda^3 + c_2 lambda^2 + c_1 lambda + 1' -> (poly in x, unknown coefficient symbols)."""
    s = _clean(text)
    if "=" in s:
        lhs, rhs = s.split("=", 1)
        s = (
            rhs
            if re.fullmatch(r"\s*[pPfqc]?\s*(?:_?A)?\s*\(\s*\w+\s*\)\s*", lhs) or not lhs.strip()
            else f"({lhs})-({rhs})"
        )
    s = re.sub(r"lambda|λ", "x", s)
    s = re.sub(r"\bt\b", "x", s)
    s = re.sub(r"(?<=[0-9)])\s*x", "*x", s)
    s = re.sub(r"\b([a-wyzA-Z])_?(\d)\b", r"\1\2", s)  # c_2 -> c2
    try:
        p = sp.expand(parse_expression(s))
    except ExpressionError as exc:
        raise SpectrumError("Could not read the characteristic polynomial.") from exc
    unknowns = sorted(p.free_symbols - {X}, key=str)
    if not p.is_polynomial(X) or sp.degree(p, X) < 1:
        raise SpectrumError("The characteristic polynomial must be a polynomial in λ.")
    return p, unknowns


def _scalar(text: Any, what: str) -> sp.Expr:
    try:
        v = parse_expression(str(text).replace("i", "I") if re.fullmatch(r"[\d\s.+\-*/()]*i", str(text)) else str(text))
    except ExpressionError as exc:
        raise SpectrumError(f"Could not read the {what} {text!r}.") from exc
    if v.free_symbols:
        raise SpectrumError(f"The {what} must be a number, got {text!r}.")
    return v


_QUERY = re.compile(
    r"^\s*(?P<kind>eigenvalues?|spectrum|trace|tr|determinant|det|charpoly|characteristic[ _]polynomial)"
    r"\s*(?:\(\s*(?P<arg>.*)\s*\))?\s*$",
    re.I,
)


def parse_query(text: str) -> tuple[str, str, sp.Expr]:
    m = _QUERY.match(str(text))
    if not m:
        raise SpectrumError(
            f"Unsupported query {text!r}; use eigenvalues, trace(...), det(...) or the characteristic polynomial."
        )
    kind = m.group("kind").lower()
    kind = {"eigenvalue": "eigenvalues", "spectrum": "eigenvalues", "tr": "trace", "determinant": "det"}.get(kind, kind)
    if kind.startswith("char"):
        kind = "charpoly"
    arg = (m.group("arg") or "A").strip() or "A"
    if kind == "charpoly" and arg != "A":
        raise SpectrumError("Only the characteristic polynomial of A itself is supported.")
    return kind, arg, matrix_function(arg)


# --------------------------------------------------------------------------- exact helpers


def _canon(e: sp.Expr) -> sp.Expr:
    # expand() multiplies out powers of algebraic numbers (and keeps symbols like λ untouched)
    return sp.expand(e)


def is_zero(e: sp.Expr) -> bool | None:
    """True/False when decided exactly; None when only a numeric check could decide."""
    e = sp.nsimplify(e) if e.is_Float else e
    c = _canon(e)
    if c == 0:
        return True
    s = sp.radsimp(c)
    if s != 0 and s.is_number and abs(complex(sp.N(s, 40))) > 1e-25:
        return False
    s = sp.simplify(s) if s != 0 else s
    if s == 0:
        return True
    if s.is_number:
        val = complex(sp.N(s, 40))
        if abs(val) > 1e-25:
            return False
        return None
    return None


def same(a: sp.Expr, b: sp.Expr) -> bool:
    z = is_zero(a - b)
    if z is None:  # numeric only
        return abs(complex(sp.N(a - b, 40))) < 1e-25
    return z


def tidy(e: sp.Expr) -> sp.Expr:
    t = sp.simplify(sp.expand_complex(e))
    return sp.nsimplify(t) if t.is_Float else t


def _conjugate_index(values: list[sp.Expr]) -> list[int | None]:
    out: list[int | None] = []
    for v in values:
        c = sp.conjugate(v)
        out.append(next((j for j, w in enumerate(values) if same(c, w)), None))
    return out


def arg_latex(arg: str) -> str:
    """LaTeX for a matrix expression as the user wrote it ('(A + I)^-1' -> '(A + I)^{-1}')."""
    s = _clean(arg).replace("*", " ")
    s = re.sub(r"\^\s*\(?\s*(-?\d+)\s*\)?", r"^{\1}", s)
    return s


def _lam_latex(v: sp.Expr) -> str:
    return sp.latex(v)


def txt(v: Any) -> str:
    """Readable plain text for a value: 1/2 - √3·i/2 rather than 1/2 - sqrt(3)*I/2."""
    s = sp.sstr(v)
    s = re.sub(r"sqrt\((\d+)\)", r"√\1", s)
    s = s.replace("sqrt(", "√(").replace("**", "^")
    s = re.sub(r"\*I\b", "·i", s)
    s = re.sub(r"\bI\*", "i·", s)
    s = re.sub(r"\bI\b", "i", s)
    s = s.replace("lambda", "λ").replace("*", "·")
    return s


def _spectrum_text(spec: list[sp.Expr]) -> str:
    counts: list[tuple[sp.Expr, int]] = []
    for v in spec:
        for i, (w, c) in enumerate(counts):
            if same(v, w):
                counts[i] = (w, c + 1)
                break
        else:
            counts.append((v, 1))
    return ", ".join(txt(v) + (f" (×{c})" if c > 1 else "") for v, c in counts)


def _spectrum_latex(spec: list[sp.Expr]) -> str:
    counts: list[tuple[sp.Expr, int]] = []
    for v in spec:
        for i, (w, c) in enumerate(counts):
            if same(v, w):
                counts[i] = (w, c + 1)
                break
        else:
            counts.append((v, 1))
    return r",\ ".join(sp.latex(v) + (rf"\ (\times{c})" if c > 1 else "") for v, c in counts)


# --------------------------------------------------------------------------- the tool


def matrix_spectrum(
    queries: list[str],
    size: int | None = None,
    eigenvalues: list[Any] | None = None,
    annihilating: str | None = None,
    charpoly: str | None = None,
    trace: Any = None,
    determinant: Any = None,
    real: bool = False,
) -> dict[str, Any]:
    """Compute the requested quantities. Returns a dict consumed by ``tools.matrix_spectrum``."""
    if not queries:
        raise SpectrumError("Nothing to compute: give at least one query (e.g. 'eigenvalues', 'trace(A^2)').")
    if len(queries) > 6:
        raise SpectrumError("At most 6 quantities can be asked for at once.")
    parsed = [parse_query(q) for q in queries]
    n = int(size) if size is not None else (len(eigenvalues) if eigenvalues else None)
    if n is None:
        raise SpectrumError("The size of the matrix (n × n) is needed.")
    if not 1 <= n <= MAX_SIZE:
        raise SpectrumError(f"Matrices up to {MAX_SIZE}×{MAX_SIZE} are supported.")

    steps: list[Step] = []
    facts = [f"{n}×{n}" + (" real" if real else "") + " matrix"]
    T = _scalar(trace, "trace") if trace is not None else None
    D = _scalar(determinant, "determinant") if determinant is not None else None
    cp, unknowns = characteristic_polynomial(charpoly) if charpoly else (None, [])
    q = annihilating_polynomial(annihilating) if annihilating else None

    # ---- 1. candidate spectra
    if eigenvalues:
        ev = [_scalar(v, "eigenvalue") for v in eigenvalues]
        if len(ev) != n:
            raise SpectrumError(
                f"{len(ev)} eigenvalue(s) were given for a {n}×{n} matrix; all {n} are needed (with multiplicity)."
            )
        roots: list[sp.Expr] = []
        for v in ev:
            if not any(same(v, r) for r in roots):
                roots.append(v)
        candidates = [[next(i for i, r in enumerate(roots) if same(v, r)) for v in ev]]
        facts.append("eigenvalues " + ", ".join(sp.sstr(v) for v in ev))
        steps.append(Step(title="Given eigenvalues", latex=r"\lambda \in \{" + _spectrum_latex(ev) + r"\}"))
    elif q is not None:
        found = sp.roots(sp.Poly(q, X))
        if sum(found.values()) != sp.degree(q, X):
            raise SpectrumError("The roots of the matrix polynomial cannot be found exactly.")
        roots = [tidy(r) for r in found]
        facts.append(f"p(A) = 0 with p(λ) = {sp.sstr(sp.factor(q.subs(X, sp.Symbol('lambda'))))}")
        steps.append(
            Step(
                title="Eigenvalues must be roots of the annihilating polynomial",
                detail="If p(A) = 0 then p(λ) = 0 for every eigenvalue λ (the minimal polynomial divides p).",
                latex=rf"p(\lambda) = {sp.latex(sp.factor(q.subs(X, sp.Symbol('lambda'))))} = 0"
                + r"\ \Rightarrow\ \lambda \in \{"
                + r",\ ".join(_lam_latex(r) for r in roots)
                + r"\}",
            )
        )
        total = sp.binomial(len(roots) + n - 1, n)
        if total > MAX_CANDIDATES:
            raise SpectrumError("Too many possible eigenvalue combinations to examine.")
        candidates = [list(c) for c in itertools.combinations_with_replacement(range(len(roots)), n)]
    elif cp is not None and not unknowns:
        found = sp.roots(sp.Poly(cp, X))
        if sum(found.values()) != n:
            raise SpectrumError("The roots of the characteristic polynomial cannot be found exactly.")
        roots = [tidy(r) for r in found]
        candidates = [[i for i, (r, m) in enumerate(found.items()) for _ in range(m)]]
        steps.append(
            Step(title="Eigenvalues are the roots of the characteristic polynomial", latex=sp.latex(sp.factor(cp)))
        )
    elif T is not None or D is not None:
        return _monomial_answers(queries, parsed, n, T, D, real, steps, facts)
    else:
        raise SpectrumError(
            "Not enough information to determine the eigenvalues: give them, a polynomial equation the matrix "
            "satisfies, or its full characteristic polynomial."
        )

    # ---- 2. constraints
    conj = _conjugate_index(roots)
    if cp is not None:
        lead = sp.Poly(cp, X).LC()
        cp = sp.expand(cp / lead)
        if sp.degree(cp, X) != n:
            raise SpectrumError(f"The characteristic polynomial of a {n}×{n} matrix must have degree {n}.")
    feasible: list[list[int]] = []
    solved_unknowns: list[dict[sp.Symbol, sp.Expr]] = []
    reasons = {"trace": 0, "det": 0, "real": 0, "charpoly": 0}
    for cand in candidates:
        spec = [roots[i] for i in cand]
        if real:
            ok = all(conj[i] is not None and cand.count(i) == cand.count(conj[i]) for i in set(cand))
            if not ok:
                reasons["real"] += 1
                continue
        if T is not None and not same(sp.Add(*spec), T):
            reasons["trace"] += 1
            continue
        if D is not None and not same(sp.Mul(*spec), D):
            reasons["det"] += 1
            continue
        sol: dict[sp.Symbol, sp.Expr] = {}
        if cp is not None:
            actual = sp.Poly(sp.expand(sp.Mul(*[X - v for v in spec])), X).all_coeffs()
            given = sp.Poly(cp, X).all_coeffs()
            eqs = [g - a for g, a in zip(given, actual, strict=True)]
            numeric = [e for e in eqs if not (e.free_symbols & set(unknowns))]
            symbolic = [e for e in eqs if e.free_symbols & set(unknowns)]
            if not all(same(e, 0) for e in numeric):
                reasons["charpoly"] += 1
                continue
            if symbolic:
                res = sp.solve(symbolic, unknowns, dict=True)
                if not res:
                    reasons["charpoly"] += 1
                    continue
                sol = {k: tidy(v) for k, v in res[0].items()}
        feasible.append(cand)
        solved_unknowns.append(sol)

    used = [f"trace = {sp.sstr(T)}"] * (T is not None) + [f"det = {sp.sstr(D)}"] * (D is not None)
    if real:
        used.append("real entries (non-real eigenvalues come in conjugate pairs)")
    if cp is not None:
        used.append("the given characteristic polynomial coefficients")
    if used and len(candidates) > 1:
        removed = ", ".join(f"{k}: {v}" for k, v in reasons.items() if v)
        steps.append(
            Step(
                title="Apply the conditions",
                detail=f"Checked all {len(candidates)} combinations of {n} eigenvalues against "
                + "; ".join(used)
                + f". {len(feasible)} remain"
                + (f" (removed by {removed})." if removed else "."),
            )
        )
    if not feasible:
        raise SpectrumError("No matrix satisfies all the stated conditions; please check the problem statement.")

    spectra = [[roots[i] for i in c] for c in feasible]
    for s in spectra[:6]:
        steps.append(
            Step(
                title="Possible spectrum" if len(spectra) > 1 else "Spectrum",
                latex=r"\{"
                + _spectrum_latex(s)
                + r"\}"
                + rf",\quad \det(\lambda I - A) = {sp.latex(sp.factor(_canon(sp.Mul(*[sp.Symbol('lambda') - v for v in s]))))}",
            )
        )

    # ---- 3. answers by spectral mapping
    results: list[dict[str, Any]] = []
    lines, latex_lines = [], []
    for raw, (kind, arg, f) in zip(queries, parsed, strict=True):
        arg = _clean(arg)
        label = {"trace": f"tr({arg})", "det": f"det({arg})", "eigenvalues": "eigenvalues", "charpoly": "char. poly"}[
            kind
        ]
        if kind == "eigenvalues" and arg != "A":
            label = f"eigenvalues of {arg}"
        num, den = sp.fraction(f)
        values: list[Any] = []
        for s in spectra:
            if any(same(den.subs(X, v), 0) for v in s):
                values.append(None)
                continue
            mapped = [tidy(f.subs(X, v)) for v in s]
            if kind == "trace":
                values.append(tidy(sp.Add(*mapped)))
            elif kind == "det":
                values.append(tidy(sp.Mul(*mapped)))
            elif kind == "eigenvalues":
                values.append(mapped)
            else:
                values.append(sp.factor(_canon(sp.Mul(*[sp.Symbol("lambda") - v for v in s]))))
        if all(v is None for v in values):
            raise SpectrumError(f"{arg} is not invertible: its denominator vanishes at an eigenvalue of A.")
        defined = [v for v in values if v is not None]
        if kind == "eigenvalues":
            union: list[sp.Expr] = []
            for vs in defined:
                for v in vs:
                    if not any(same(v, u) for u in union):
                        union.append(v)
            if len(defined) == 1:
                text = f"{label}: {_spectrum_text(defined[0])}"
                ltx = (r"\lambda" if arg == "A" else rf"\text{{eig}}({sp.latex(f.subs(X, sp.Symbol('A')))})") + (
                    r" \in \{" + _spectrum_latex(defined[0]) + r"\}"
                )
            else:
                text = f"possible {label}: " + ", ".join(txt(u) for u in union)
                ltx = r"\lambda \in \{" + r",\ ".join(sp.latex(u) for u in union) + r"\}"
            results.append({"query": raw, "kind": kind, "arg": arg, "values": values, "union": union})
        else:
            distinct: list[sp.Expr] = []
            for v in defined:
                if not any(same(v, d) for d in distinct):
                    distinct.append(v)
            name = {"trace": r"\operatorname{tr}", "det": r"\det"}.get(kind, "p")
            arg_ltx = arg_latex(arg)
            if kind == "charpoly":
                ltx_head, txt_head = r"\det(\lambda I - A)", "det(λI − A)"
            else:
                ltx_head, txt_head = rf"{name}\left({arg_ltx}\right)", label
            if len(distinct) == 1:
                text = f"{txt_head} = {txt(distinct[0])}"
                ltx = rf"{ltx_head} = {sp.latex(distinct[0])}"
            else:
                text = f"{txt_head} depends on the case; possible values: " + ", ".join(txt(d) for d in distinct)
                ltx = rf"{ltx_head} \in \{{" + r",\ ".join(sp.latex(d) for d in distinct) + r"\}"
            results.append({"query": raw, "kind": kind, "arg": arg, "values": values, "distinct": distinct})
            if kind in ("trace", "det") and len(spectra) == 1:
                mapped = [tidy(f.subs(X, v)) for v in spectra[0]]
                op = "+" if kind == "trace" else r"\cdot"
                steps.append(
                    Step(
                        title=f"Spectral mapping for {label}",
                        detail=f"The eigenvalues of {arg} are f(λ) for each eigenvalue λ of A"
                        + (f", with f(λ) = {sp.sstr(f).replace('**', '^').replace('x', 'λ')}." if arg != "A" else "."),
                        latex=rf"{ltx_head} = "
                        + f" {op} ".join(rf"\left({sp.latex(m)}\right)" for m in mapped)
                        + rf" = {sp.latex(distinct[0])}",
                    )
                )
        if any(v is None for v in values):
            text += f" (only where {arg} is invertible)"
        lines.append(text)
        latex_lines.append(ltx)

    if unknowns and solved_unknowns and all(solved_unknowns):
        keys = sorted(solved_unknowns[0], key=str)
        for k in keys:
            vals: list[sp.Expr] = []
            for sol in solved_unknowns:
                if k in sol and not any(same(sol[k], v) for v in vals):
                    vals.append(sol[k])
            if len(vals) == 1:
                lines.append(f"{k} = {txt(vals[0])}")
                latex_lines.append(rf"{sp.latex(k)} = {sp.latex(vals[0])}")
    return {
        "lines": lines,
        "latex_lines": latex_lines,
        "steps": steps,
        "spectra": spectra,
        "roots": roots,
        "results": results,
        "n": n,
        "real": real,
        "trace": T,
        "det": D,
        "annihilating": q,
        "charpoly": cp,
        "unknowns": unknowns,
        "eigenvalues_given": bool(eigenvalues),
        "facts": facts,
    }


def _monomial(f: sp.Expr) -> tuple[sp.Expr, int] | None:
    """c·x^k -> (c, k) for an integer k (k may be negative)."""
    c, k = sp.Mul(f).as_coeff_exponent(X)
    if c.free_symbols or not (k.is_Integer):
        return None
    return sp.nsimplify(c), int(k)


def _monomial_answers(queries, parsed, n, T, D, real, steps, facts) -> dict[str, Any]:  # noqa: ANN001
    """Without the eigenvalues, det(c·A^k) = c^n·det(A)^k and tr(c·A) = c·tr(A) still follow from the given facts."""
    lines, latex_lines, results = [], [], []
    for raw, (kind, arg, f) in zip(queries, parsed, strict=True):
        mono = _monomial(f)
        if kind == "det" and mono and D is not None:
            c, k = mono
            if k < 0 and same(D, 0):
                raise SpectrumError(f"{arg} does not exist: det(A) = 0, so A is not invertible.")
            value = tidy(c**n * D**k)
            steps.append(
                Step(
                    title=f"det({arg}) from det(A)",
                    detail="det(cM) = cⁿ·det(M) for an n×n matrix, and det(Aᵏ) = det(A)ᵏ.",
                    latex=rf"\det({arg_latex(arg)}) = {sp.latex(c)}^{{{n}}}\cdot ({sp.latex(D)})^{{{k}}} = {sp.latex(value)}",
                )
            )
            label = f"det({_clean(arg)})"
            lines.append(f"{label} = {txt(value)}")
            latex_lines.append(rf"\det\left({arg_latex(arg)}\right) = {sp.latex(value)}")
        elif kind == "trace" and mono and mono[1] in (0, 1) and (mono[1] == 0 or T is not None):
            c, k = mono
            value = tidy(c * (T if k == 1 else n))
            steps.append(
                Step(
                    title=f"tr({arg})",
                    detail="The trace is linear: tr(cA) = c·tr(A), and tr(I) = n.",
                    latex=rf"\operatorname{{tr}}({arg_latex(arg)}) = {sp.latex(value)}",
                )
            )
            lines.append(f"tr({_clean(arg)}) = {txt(value)}")
            latex_lines.append(rf"\operatorname{{tr}}\left({arg_latex(arg)}\right) = {sp.latex(value)}")
        else:
            raise SpectrumError(
                f"{raw} cannot be determined from the trace and determinant alone; the eigenvalues "
                "(or a matrix equation) are needed."
            )
        results.append({"query": raw, "kind": kind, "arg": arg, "values": [value], "distinct": [value]})
    return {
        "lines": lines,
        "latex_lines": latex_lines,
        "steps": steps,
        "spectra": [],
        "roots": [],
        "results": results,
        "n": n,
        "real": real,
        "trace": T,
        "det": D,
        "annihilating": None,
        "charpoly": None,
        "unknowns": [],
        "eigenvalues_given": False,
        "facts": facts,
        "monomial": True,
    }


def _example_spectra(n: int, T: sp.Expr | None, D: sp.Expr | None) -> list[list[sp.Expr]]:
    """Two different spectra with the given trace and determinant (for checking identities on concrete matrices)."""
    out = []
    for fill in (sp.Integer(1), sp.Integer(2)):
        if n == 1:
            spec = [D if D is not None else (T if T is not None else fill)]
        else:
            rest = [fill] * (n - 2)
            s = (T - sum(rest)) if T is not None else fill + 3
            p = (D / sp.Mul(*rest)) if D is not None else fill * 3
            r1 = (s + sp.sqrt(s**2 - 4 * p)) / 2
            spec = [tidy(r1), tidy(s - r1), *rest]
        out.append(spec)
    return out


# --------------------------------------------------------------------------- independent verification


def _realisation(spec: list[sp.Expr], real: bool) -> sp.Matrix:
    """A concrete, non-diagonal matrix with exactly this spectrum (real if requested)."""
    n = len(spec)
    blocks: list[sp.Matrix] = []
    remaining = list(spec)
    while remaining:
        v = remaining.pop(0)
        if real and not same(sp.im(v), 0):
            c = sp.conjugate(v)
            j = next(k for k, w in enumerate(remaining) if same(w, c))
            remaining.pop(j)
            a, b = tidy(sp.re(v)), tidy(sp.im(v))
            blocks.append(sp.Matrix([[a, -b], [b, a]]))
        else:
            blocks.append(sp.Matrix([[v]]))
    B = sp.diag(*blocks)
    L = sp.Matrix(n, n, lambda i, j: 1 if i >= j else 0)
    U = sp.Matrix(n, n, lambda i, j: 1 if j >= i else 0)
    P = L * U  # integer, determinant 1
    return (P * B * P.inv()).applyfunc(_canon)


def _poly_at(p: sp.Expr, M: sp.Matrix) -> sp.Matrix:
    coeffs = sp.Poly(p, X).all_coeffs()
    out = sp.zeros(M.rows)
    for c in coeffs:
        out = (out * M + c * sp.eye(M.rows)).applyfunc(_canon)
    return out


def _fn_at(f: sp.Expr, M: sp.Matrix) -> sp.Matrix:
    num, den = sp.fraction(sp.together(f))
    N = _poly_at(sp.expand(num), M)
    if den == 1:
        return N
    Dm = _poly_at(sp.expand(den), M)
    return (N * Dm.inv(method="LU")).applyfunc(lambda e: sp.radsimp(_canon(e)))


def _charpoly(M: sp.Matrix, var: sp.Symbol) -> sp.Expr:
    return _canon(M.charpoly(var).as_expr())


def _numeric_candidates(data: dict[str, Any]) -> list[list[complex]] | None:
    """Re-enumerate the feasible spectra with floating point (NumPy roots), independently of SymPy."""
    n, real = data["n"], data["real"]
    if data["eigenvalues_given"] or data["annihilating"] is None:
        return None
    coeffs = [complex(sp.N(c)) for c in sp.Poly(data["annihilating"], X).all_coeffs()]
    raw = np.roots(coeffs)
    distinct: list[complex] = []
    for r in raw:
        if not any(abs(r - d) < 1e-6 for d in distinct):
            distinct.append(complex(r))
    T = complex(sp.N(data["trace"])) if data["trace"] is not None else None
    D = complex(sp.N(data["det"])) if data["det"] is not None else None
    cp = data["charpoly"]
    known = None
    if cp is not None:
        given = sp.Poly(cp, X).all_coeffs()
        known = [(i, complex(sp.N(g))) for i, g in enumerate(given) if not (g.free_symbols - {X})]
    out = []
    for cand in itertools.combinations_with_replacement(distinct, n):
        if real and any(
            abs(v.imag) > 1e-9
            and sum(abs(w - v) < 1e-6 for w in cand) != sum(abs(w - v.conjugate()) < 1e-6 for w in cand)
            for v in cand
        ):
            continue
        if T is not None and abs(sum(cand) - T) > 1e-6:
            continue
        if D is not None and abs(complex(np.prod(cand)) - D) > 1e-6:
            continue
        if known is not None:
            pc = np.poly(cand)
            if any(abs(pc[i] - g) > 1e-6 for i, g in known):
                continue
        out.append(list(cand))
    return out


def _verify_monomial(data: dict[str, Any]) -> Verification:
    checks: list[VerificationCheck] = []
    exact = True
    for idx, spec in enumerate(_example_spectra(data["n"], data["trace"], data["det"])):
        M = _realisation(spec, False)
        if data["det"] is not None:
            ok = is_zero(M.det(method="berkowitz") - data["det"])
            checks.append(
                VerificationCheck(
                    method=f"example matrix {idx + 1}: det(M) = given determinant", passed=ok is not False
                )
            )
        if data["trace"] is not None:
            ok = is_zero(M.trace() - data["trace"])
            checks.append(
                VerificationCheck(method=f"example matrix {idx + 1}: tr(M) = given trace", passed=ok is not False)
            )
        for res in data["results"]:
            F = _fn_at(matrix_function(res["arg"]), M)
            got = F.det(method="berkowitz") if res["kind"] == "det" else F.trace()
            z = is_zero(got - res["values"][0])
            if z is None:
                exact = False
                z = abs(complex(sp.N(got - res["values"][0], 40))) < 1e-20
            checks.append(
                VerificationCheck(
                    method=f"{res['kind']}({res['arg']}) recomputed on example matrix {idx + 1}", passed=bool(z)
                )
            )
    if any(not c.passed for c in checks):
        return Verification(
            status=VerificationStatus.FAILED,
            checks=checks,
            summary="Independent check disagreed: " + ", ".join(c.method for c in checks if not c.passed),
        )
    status = VerificationStatus.VERIFIED if exact else VerificationStatus.APPROXIMATE
    return Verification(
        status=status,
        checks=checks,
        summary="The identity holds for every matrix with these facts; recomputed on two different concrete matrices.",
    )


def verify_spectrum(data: dict[str, Any]) -> Verification:
    if data.get("monomial"):
        return _verify_monomial(data)
    checks: list[VerificationCheck] = []
    exact = True
    spectra = data["spectra"]
    tested = spectra[:MAX_VERIFIED_SPECTRA]
    for idx, spec in enumerate(tested):
        M = _realisation(spec, data["real"])
        tag = f" (case {idx + 1})" if len(tested) > 1 else ""
        conds: list[tuple[str, sp.Expr]] = []
        if data["annihilating"] is not None:
            R = _poly_at(data["annihilating"], M)
            conds.append(("p(M) = 0", sp.Add(*[e * sp.conjugate(e) for e in R])))
        if data["trace"] is not None:
            conds.append(("tr(M) = given trace", M.trace() - data["trace"]))
        if data["det"] is not None:
            conds.append(("det(M) = given determinant", M.det(method="berkowitz") - data["det"]))
        if data["charpoly"] is not None:
            actual = sp.Poly(_charpoly(M, X), X).all_coeffs()
            given = sp.Poly(data["charpoly"], X).all_coeffs()
            diffs = [g - a for g, a in zip(given, actual, strict=True) if not (g.free_symbols - {X})]
            conds.append(("char. polynomial coefficients match", sp.Add(*[d * sp.conjugate(d) for d in diffs])))
        if data["real"]:
            conds.append(("M has real entries", sp.Add(*[sp.im(e) ** 2 for e in M])))
        for name, residual in conds:
            z = is_zero(residual)
            if z is None:
                exact = False
                z = abs(complex(sp.N(residual, 40))) < 1e-20
            checks.append(VerificationCheck(method=f"concrete matrix M: {name}{tag}", passed=bool(z)))
        for res in data["results"]:
            claimed = res["values"][idx]
            if claimed is None:
                continue
            f = matrix_function(res["arg"])
            F = _fn_at(f, M)
            if res["kind"] == "trace":
                diff = F.trace() - claimed
            elif res["kind"] == "det":
                diff = F.det(method="berkowitz") - claimed
            elif res["kind"] == "charpoly":
                lam = sp.Symbol("lambda")
                diff = _charpoly(M, lam) - _canon(claimed)
            else:  # eigenvalues: the characteristic polynomial of f(M) must have exactly the claimed roots
                diff = _charpoly(F, X) - _canon(sp.Mul(*[X - v for v in claimed]))
                coeffs = sp.Poly(diff, X).all_coeffs() if diff != 0 else [0]
                diff = sp.Add(*[c * sp.conjugate(c) for c in coeffs])
                z = is_zero(diff)
                if z is None:
                    exact = False
                    z = abs(complex(sp.N(diff, 40))) < 1e-20
                checks.append(
                    VerificationCheck(
                        method=f"eigenvalues of {res['arg']}: characteristic polynomial of the concrete matrix{tag}",
                        passed=bool(z),
                    )
                )
                continue
            z = is_zero(diff)
            if z is None:
                exact = False
                z = abs(complex(sp.N(diff, 40))) < 1e-20
            label = {"trace": "tr", "det": "det", "charpoly": "char. poly"}[res["kind"]]
            checks.append(
                VerificationCheck(
                    method=f"{label}({res['arg']}) recomputed by matrix arithmetic on M{tag}",
                    passed=bool(z),
                    detail="M = P·B·P⁻¹ with the candidate spectrum, P an integer matrix with det 1",
                )
            )
    numeric = _numeric_candidates(data)
    if numeric is not None:
        claimed = [[complex(sp.N(v)) for v in s] for s in spectra]

        def matches(a: list[complex], b: list[complex]) -> bool:
            rest = list(b)
            for v in a:
                j = next((k for k, w in enumerate(rest) if abs(v - w) < 1e-6), None)
                if j is None:
                    return False
                rest.pop(j)
            return True

        same_sets = len(numeric) == len(claimed) and all(any(matches(c, m) for m in numeric) for c in claimed)
        checks.append(
            VerificationCheck(
                method="all possible spectra re-enumerated with numpy.roots",
                passed=same_sets,
                detail=f"{len(numeric)} feasible case(s)",
            )
        )
    if not checks:
        return Verification(status=VerificationStatus.UNVERIFIED, summary="No independent check was possible.")
    if any(not c.passed for c in checks):
        return Verification(
            status=VerificationStatus.FAILED,
            checks=checks,
            summary="Independent check disagreed with the solver: "
            + ", ".join(c.method for c in checks if not c.passed)
            + ".",
        )
    extra = "" if len(tested) == len(spectra) else f" (first {len(tested)} of {len(spectra)} cases checked)"
    if exact:
        return Verification(
            status=VerificationStatus.VERIFIED,
            checks=checks,
            summary="Confirmed exactly on concrete matrices with the computed eigenvalues" + extra + ".",
        )
    return Verification(
        status=VerificationStatus.APPROXIMATE,
        checks=checks,
        summary="Confirmed numerically on concrete matrices; no exact symbolic proof" + extra + ".",
    )
