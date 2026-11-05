"""Offline plans for probability word problems that do not name a distribution.

* repeated trials with a fair die or coin -> binomial:
  "probability of at least one six in 4 rolls of a fair die" -> Binomial(n=4, p=1/6), P(X >= 1)
* drawing without replacement -> hypergeometric:
  "a bag has 5 red and 3 blue balls; two are drawn without replacement; probability both are red"
  -> 5 of the asked-for colour, 3 others, 2 drawn, P(X = 2)

Every parameter is copied from the question (or is the face probability of a fair die/coin);
the tool does the arithmetic, and the intent check confirms the numbers and the event.
"""

from __future__ import annotations

import re

from app.schemas import ProblemCategory, ToolCall

_WORDS = {
    "no": 0,
    "none": 0,
    "zero": 0,
    "one": 1,
    "a": 1,
    "an": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "both": None,
    "all": None,
}
_NUMWORD = r"\d+|no|none|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
_FACES = r"(?:sixes|six|fives|five|fours|four|threes|three|twos|two|ones|one|\d+s?)"


def _num(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _WORDS.get(token)


def _event(low: str, noun: str) -> tuple[str, int] | None:
    """(query, x) for 'exactly k', 'at least k', 'at most k', 'more than k', 'fewer than k', 'no' + noun."""
    pats = [
        (rf"\bexactly\s+({_NUMWORD})\b", "pmf", 0),
        (rf"\bat\s+least\s+({_NUMWORD}|a|an)\b", "greater", 0),
        (rf"\bat\s+most\s+({_NUMWORD})\b", "cdf", 0),
        (rf"\b(?:more|greater)\s+than\s+({_NUMWORD})\b", "greater", 1),
        (rf"\b(?:fewer|less)\s+than\s+({_NUMWORD})\b", "cdf", -1),
        (rf"\b(no|none|zero)\s+{noun}", "pmf", 0),
    ]
    for pat, q, shift in pats:
        m = re.search(pat, low)
        if m:
            k = _num(m.group(1))
            if k is not None:
                return q, k + shift
    return None


def plan_trials(text: str) -> ToolCall | None:
    low = text.lower()
    die = re.search(r"\b(?:fair\s+)?(?:die|dice)\b", low)
    coin = re.search(r"\b(?:fair\s+)?coins?\b", low)
    if not (die or coin) or "probabilit" not in low and "chance" not in low:
        return None
    m = re.search(rf"\b({_NUMWORD})\s+(?:rolls|throws|tosses|flips|times)\b", low) or re.search(
        rf"\b(?:rolled|thrown|tossed|flipped)\s+({_NUMWORD})\s+times\b", low
    )
    if not m or _num(m.group(1)) in (None, 0):
        return None
    n = _num(m.group(1))
    if die:
        face = re.search(
            rf"\b(?:a|an|the)?\s*({_FACES})\b(?!\s+(?:rolls|times|throws))", low[m.end() :] + " " + low[: m.start()]
        )
        noun = r"(?:sixes|six|fives|five|fours|four|threes|three|twos|two|ones|one|\d+s?)"
        p = "1/6"
        if not face:
            return None
    else:
        if not re.search(r"\bheads?\b|\btails?\b", low):
            return None
        noun = r"(?:heads?|tails?)"
        p = "1/2"
    ev = _event(low, noun)
    if ev is None:
        return None
    query, x = ev
    if x < 0 or x > n:
        return None
    return ToolCall(
        tool="probability_distribution",
        category=ProblemCategory.PROBABILITY_STATISTICS,
        args={"distribution": "binomial", "params": {"n": n, "p": p}, "query": query, "x": x},
        rationale=f"{n} independent {'rolls of a fair die' if die else 'tosses of a fair coin'}: a binomial count.",
    )


_COLOURS = r"red|blue|green|white|black|yellow|orange|purple|pink|brown|grey|gray|silver|gold"


def plan_draws(text: str) -> ToolCall | None:
    low = text.lower()
    items = re.findall(rf"\b({_NUMWORD})\s+({_COLOURS})\b", low)
    if len(items) != 2 or not re.search(r"\b(?:drawn|picked|chosen|selected|taken|removed)\b", low):
        return None
    if re.search(r"\bwith\s+replacement\b", low):
        return None  # independent draws: a different model
    m = re.search(
        rf"\b({_NUMWORD})\s+(?:\w+\s+)?(?:are|is|balls?\s+are|marbles?\s+are)?\s*(?:drawn|picked|chosen|selected|taken|removed)\b",
        low,
    )
    if not m:
        return None
    draws = _num(m.group(1))
    if draws is None or draws < 1:
        return None
    counts = {colour: _num(c) for c, colour in items}
    if any(v is None for v in counts.values()):
        return None
    (c1, n1), (c2, n2) = counts.items()
    after = low[m.end() :]
    target = re.search(rf"\b({_COLOURS})\b", after)
    if not target or target.group(1) not in counts:
        return None
    good_colour = target.group(1)
    good, bad = (n1, n2) if good_colour == c1 else (n2, n1)
    if re.search(rf"\b(?:both|all)\b[^.?]*\b{good_colour}\b", after):
        query, x = "pmf", draws
    else:
        ev = _event(after, good_colour)
        if ev is None:
            return None
        query, x = ev
    return ToolCall(
        tool="probability_distribution",
        category=ProblemCategory.PROBABILITY_STATISTICS,
        args={
            "distribution": "hypergeometric",
            "params": {"success": good, "failure": bad, "draws": draws},
            "query": query,
            "x": x,
        },
        rationale="Drawing without replacement: a hypergeometric count.",
    )
