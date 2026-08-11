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
VERDICT = re.compile(r"\*\*([A-Z]+)\*\*")
#: A row of the summary table: `| P3 | 1 | 1 | 1 | — |`.
TALLY_ROW = re.compile(r"^\|\s*(\*\*)?(§1 universal|P\d|Total)(\*\*)?\s*\|")

VERDICTS = ("PASS", "PARTIAL", "FAIL")


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
        numbers = [0 if c in ("—", "") else int(c) for c in cells[2:5]]
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
    words = {
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
    }
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
