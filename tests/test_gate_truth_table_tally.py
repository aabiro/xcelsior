"""The gate truth table's tally must agree with its own clause rows.

`docs/gate-truth-table.md` is the project's answer to "which gate clauses are
actually met". It carries a per-gate summary table at the bottom, maintained by
hand, and that table had drifted: it reported **21 PASS when the clause rows
held 20**. The overstatement was invisible because both numbers live in the
same document and nothing compared them — the summary is exactly the kind of
hand-kept duplicate that has been removed from this codebase repeatedly.

A tally that overstates is worse than no tally. It is the number quoted in
status updates, and it moved a gate from "one clause short" to "met" without
any clause changing.

## What is derived from what

Nothing here is a hardcoded expectation. The clause rows are the source: each
is a table row whose first cell is a clause number and whose third cell holds
a bolded verdict. Those are counted per section heading and compared against
the Tally table parsed out of the same file. Adding a clause, changing a
verdict, or adding a whole gate needs no edit here — and a summary edited
without the rows beneath it fails immediately.
"""

from __future__ import annotations

import collections
import pathlib
import re

import pytest

TABLE = pathlib.Path(__file__).resolve().parent.parent / "docs" / "gate-truth-table.md"

#: A clause row: `| 3 | the clause text | **PASS** | the evidence |`.
CLAUSE_ROW = re.compile(r"^\|\s*\d+\s*\|")
#: The bolded verdict, wherever it sits in the verdict cell — rows carry
#: history like `**PASS** *(was PARTIAL)*`, and the first bold token is the
#: current one.
#: Hyphen included so `ACCEPTED-UNFIXABLE` is one token. Without it the regex
#: matched `ACCEPTED` and the verdict would have been silently unrecognised —
#: counted in no column, which is how a clause disappears from a derived tally.
VERDICT = re.compile(r"\*\*([A-Z][A-Z-]*)\*\*")
#: A row of the summary table: `| P3 | 1 | 1 | 1 | — |`.
TALLY_ROW = re.compile(r"^\|\s*(\*\*)?(§1 universal|P\d|Total)(\*\*)?\s*\|")

#: Column order in the Tally table. `ACCEPTED-UNFIXABLE` is last and separate
#: on purpose: it is neither proven nor outstanding, and giving it its own
#: column is what stops it inflating either number.
#: Number words the prose may use, generated rather than typed.
#: The hand-written version stopped at twenty-nine and went red the moment
#: the denominator passed it — a guard failing because the project grew is a
#: guard that trains its reader to edit it rather than read it.
_NUMBER_WORDS = {
    "Eighteen": 18,
    "Nineteen": 19,
    "Twenty": 20,
    "Twenty-one": 21,
    "Twenty-two": 22,
    "Twenty-three": 23,
    "Twenty-four": 24,
    "Twenty-five": 25,
    "Twenty-six": 26,
    "Twenty-seven": 27,
    "Twenty-eight": 28,
    "Twenty-nine": 29,
    "Thirty": 30,
    "Thirty-one": 31,
    "Thirty-two": 32,
    "Thirty-three": 33,
    "Thirty-four": 34,
    "Thirty-five": 35,
    "Thirty-six": 36,
    "Thirty-seven": 37,
    "Thirty-eight": 38,
    "Thirty-nine": 39,
    "Forty": 40,
    "Forty-one": 41,
    "Forty-two": 42,
    "Forty-three": 43,
    "Forty-four": 44,
    "Forty-five": 45,
    "Forty-six": 46,
    "Forty-seven": 47,
    "Forty-eight": 48,
    "Forty-nine": 49,
    "Fifty": 50,
    "Fifty-one": 51,
    "Fifty-two": 52,
    "Fifty-three": 53,
    "Fifty-four": 54,
    "Fifty-five": 55,
    "Fifty-six": 56,
    "Fifty-seven": 57,
    "Fifty-eight": 58,
    "Fifty-nine": 59,
}


VERDICTS = ("PASS", "PARTIAL", "FAIL", "ACCEPTED-UNFIXABLE")


def _sections() -> dict[str, collections.Counter]:
    """Clause verdicts counted per gate, read from the clause rows."""
    counts: dict[str, collections.Counter] = {}
    section: str | None = None
    for line in TABLE.read_text().splitlines():
        heading = re.match(r"^##\s+(.*)", line)
        if heading:
            title = heading.group(1).strip()
            if title.startswith("§1"):
                section = "§1 universal"
            elif title.startswith("Gate P"):
                section = title.split()[1]
            else:
                # Prose sections, the Tally itself, the rulings. Their tables
                # are not clause rows and must not be counted as any gate's.
                section = None
            if section:
                counts.setdefault(section, collections.Counter())
            continue
        if section and CLAUSE_ROW.match(line):
            cells = [c.strip() for c in line.split("|")]
            found = VERDICT.search(cells[3]) if len(cells) > 3 else None
            if found:
                counts[section][found.group(1)] += 1
    return counts


def _tally() -> dict[str, list[int]]:
    """The summary table as written, `{row label: [pass, partial, fail]}`."""
    rows: dict[str, list[int]] = {}
    in_tally = False
    for line in TABLE.read_text().splitlines():
        if re.match(r"^##\s+Tally", line):
            in_tally = True
            continue
        if in_tally and re.match(r"^##\s+", line):
            break
        if not in_tally or not TALLY_ROW.match(line):
            continue
        cells = [c.strip().replace("*", "") for c in line.split("|")]
        label = cells[1]
        numbers = [0 if c in ("—", "") else int(c) for c in cells[2 : 2 + len(VERDICTS)]]
        rows[label] = numbers
    return rows


def test_the_truth_table_is_present_and_parseable():
    """A guard over an empty parse passes; this is what stops that reading green."""
    assert TABLE.exists(), f"{TABLE} is missing"
    sections, tally = _sections(), _tally()
    assert sections, "no clause rows were found; the parser has lost the table"
    assert "Total" in tally, "no Total row was found in the Tally section"
    assert len(tally) >= 3, f"only {len(tally)} tally rows parsed; expected one per gate"


@pytest.mark.parametrize("verdict_index,verdict", list(enumerate(VERDICTS)))
def test_each_gates_tally_matches_its_clause_rows(verdict_index: int, verdict: str):
    sections, tally = _sections(), _tally()
    wrong = []
    for label, counted in sections.items():
        claimed = tally.get(label)
        if claimed is None:
            wrong.append(f"{label}: has clause rows but no tally row")
            continue
        if claimed[verdict_index] != counted[verdict]:
            wrong.append(
                f"{label}: tally says {claimed[verdict_index]} {verdict}, "
                f"clause rows hold {counted[verdict]}"
            )
    assert not wrong, "the tally disagrees with the clauses it summarises: " + "; ".join(wrong)


@pytest.mark.parametrize("verdict_index,verdict", list(enumerate(VERDICTS)))
def test_the_total_row_is_the_sum_of_the_clause_rows(verdict_index: int, verdict: str):
    """The row that had drifted. It is a sum, so it is never independently true."""
    sections, tally = _sections(), _tally()
    actual = sum(c[verdict] for c in sections.values())
    claimed = tally["Total"][verdict_index]
    assert claimed == actual, (
        f"the Total row claims {claimed} {verdict}; the clause rows hold {actual}. "
        "This is the number quoted in status updates — correct the total, not the rows."
    )


def test_the_prose_count_matches_the_table():
    """ "Twenty-one of twenty-nine" is a claim, and it is checkable."""
    words = _NUMBER_WORDS
    text = TABLE.read_text()
    # Case-insensitive: the sentence capitalises the first number and not the
    # second, and which one is capitalised is a fact about the sentence rather
    # than about the count.
    stated = re.search(
        r"\b(" + "|".join(words) + r")\b of \b(" + "|".join(words) + r")\b clauses are fully met",
        text,
        re.IGNORECASE,
    )
    assert stated, "the prose no longer states a count; it is the sentence people quote"

    sections = _sections()
    met = sum(c["PASS"] for c in sections.values())
    total = sum(sum(c[v] for v in VERDICTS) for c in sections.values())
    assert words[stated.group(1).capitalize()] == met, (
        f"the prose says {stated.group(1)} clauses are met; {met} are"
    )
    assert words[stated.group(2).capitalize()] == total, (
        f"the prose counts {stated.group(2)} clauses; there are {total}"
    )


def test_no_clause_carries_a_verdict_the_tally_has_no_column_for():
    """A verdict with no column is a clause that vanishes from a derived tally.

    The tally is computed from the rows, so an unrecognised verdict is not a
    loud failure — it is a row counted in nothing, and a total that silently
    drops by one. Introducing `ACCEPTED-UNFIXABLE` was exactly that risk: the
    original verdict regex was `[A-Z]+`, which matches `ACCEPTED` and stops at
    the hyphen.
    """
    known = set(VERDICTS) | {"BLOCKED", "SUPERSEDED"}
    seen: set[str] = set()
    for counts in _sections().values():
        seen.update(counts)
    unknown = sorted(seen - known)
    assert not unknown, (
        f"clause rows carry verdicts the tally has no column for: {unknown}. "
        "Add a column, or the rows using them are counted nowhere."
    )


def test_accepted_unfixable_is_not_counted_as_met():
    """The whole reason it has its own column.

    A clause accepted as unobtainable is not proven. If it ever merges into the
    PASS column the headline number moves without any clause changing — the
    same drift this tally already had once, when the Total row overstated by
    one for as long as nobody compared it to the rows beneath it.
    """
    sections = _sections()
    accepted = sum(c["ACCEPTED-UNFIXABLE"] for c in sections.values())
    if not accepted:
        pytest.skip("no clause is currently ACCEPTED-UNFIXABLE")

    met = sum(c["PASS"] for c in sections.values())
    tally = _tally()
    assert tally["Total"][VERDICTS.index("PASS")] == met, (
        "the PASS total no longer equals the PASS rows; an ACCEPTED-UNFIXABLE "
        "clause may have been folded in"
    )
    assert tally["Total"][VERDICTS.index("ACCEPTED-UNFIXABLE")] == accepted

    # And the sentence people quote counts only what is met.
    import re as _re

    text = TABLE.read_text()
    stated = _re.search(r"\b([A-Za-z-]+)\b of \b[A-Za-z-]+\b clauses are fully met", text)
    assert stated, "the prose no longer states how many clauses are met"
    words = _NUMBER_WORDS
    assert words.get(stated.group(1).capitalize()) == met, (
        f"the prose says {stated.group(1)} clauses are fully met; {met} are. "
        "An accepted-unobtainable clause is not a met one."
    )
