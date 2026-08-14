"""The gate truth table's evidence must point at things that exist.

`tests/test_gate_truth_table_tally.py` derives the summary from the clause rows,
so the *numbers* cannot drift. Nothing checks that what the rows **cite** is
real, and two kinds of citation carry weight a reader acts on:

1. **The evidence column** names test files. That is what someone auditing a
   PASS opens. A file renamed out from under the row leaves the verdict looking
   supported by something nobody can find.
2. **The blocker table** names which clauses are waiting on hardware, on a
   cutover, or on code. That is the row the person with the hardware reads to
   decide what to do tonight. A clause that has moved on but stayed listed
   sends them to do work that is already done; a clause that fails with no
   blocker named means nobody knows they are needed.

## Found red, not written green

The first assertion here failed the moment it was written, against two real
citations. `dba3141` renamed `test_gate_p5_placement_refuses_end_to_end.py` to
`test_placement_refuses_end_to_end.py` and `tests/live/…_preference_refuses_live.py`
to `tests/live/test_placement_refuses_live.py` — the commit's own subject is
*"names describe things not phases"* — and the truth table kept the old names.
Gate P5 clause 2 is a **PASS** whose evidence had pointed at nothing since.

That is the whole argument for this file. The rename was correct, the tests
still exist and still pass, and the tally was right the entire time. Only the
citation rotted, and the tally guard cannot see citations.

## Why API paths are deliberately not checked here

The obvious third assertion — every route path the table names must resolve —
is wrong in this document specifically. The table names `/api/billing/topup`
and `/api/v1/promotions` **in order to say they never existed**; §1.3's row is
the account of discovering that two live gates asserted against phantom routes.
A guard that resolved every path it found would go red on the sentence
explaining the bug. That is the match-a-mention defect this suite has caught
repeatedly, and the correct response is not an exclusion list that has to be
maintained — it is to leave paths to `tests/test_live_gate_paths_resolve.py`,
which reads the gate scripts, where a named path is always a claim of existence.
"""

from __future__ import annotations

import collections
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLE = ROOT / "docs" / "gate-truth-table.md"

#: A clause row: `| 3 | the clause text | **PASS** | the evidence |`.
CLAUSE_ROW = re.compile(r"^\|\s*(\d+)\s*\|")
VERDICT = re.compile(r"\*\*([A-Z][A-Z-]*)\*\*")
#: `tests/foo.py`, including the `tests/live/` subdirectory.
CITED_TEST = re.compile(r"tests/[A-Za-z0-9_/]+\.py")
#: A clause reference in the blocker table: `P5.1`, `P7.1`.
CLAUSE_REF = re.compile(r"\bP(\d+)\.(\d+)\b")


def _clause_verdicts() -> dict[str, str]:
    """`{"P5.1": "PASS", …}` — every clause row, keyed by gate and number.

    Derived from the section headings the same way the tally guard derives its
    counts, so a new gate needs no edit here.
    """
    verdicts: dict[str, str] = {}
    gate: str | None = None
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^##\s+(.*)", line)
        if heading:
            title = heading.group(1).strip()
            if title.startswith("§1"):
                gate = "§1"
            elif title.startswith("Gate P"):
                gate = title.split()[1]
            else:
                gate = None
            continue
        row = CLAUSE_ROW.match(line)
        if not gate or not row:
            continue
        cells = [c.strip() for c in line.split("|")]
        found = VERDICT.search(cells[3]) if len(cells) > 3 else None
        if found:
            verdicts[f"{gate}.{row.group(1)}"] = found.group(1)
    return verdicts


def _blocker_rows() -> dict[str, set[str]]:
    """`{"hardware": {"P5.1", "P3.3"}, …}` from the blocker table.

    The table is `| **hardware** | P5.1 (…), P3.3 (…) |`. Only rows whose first
    cell is a bolded single word are taken, so the surrounding prose tables are
    not harvested.
    """
    rows: dict[str, set[str]] = {}
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\|\s*\*\*(\w+)\*\*\s*\|(.*)\|", line)
        if not match:
            continue
        label = match.group(1)
        if label not in ("hardware", "cutover", "code"):
            continue
        rows[label] = {f"P{m.group(1)}.{m.group(2)}" for m in CLAUSE_REF.finditer(match.group(2))}
    return rows


# ── Calibration ───────────────────────────────────────────────────────


def test_both_sources_parse():
    """Two empty sets agree perfectly. This is what stops that reading green."""
    verdicts = _clause_verdicts()
    assert len(verdicts) >= 30, f"only {len(verdicts)} clause rows parsed; the table has more"
    assert any(v == "FAIL" for v in verdicts.values()), "no FAIL clause parsed"
    blockers = _blocker_rows()
    assert blockers, "the blocker table did not parse; its shape has changed"
    assert sum(len(v) for v in blockers.values()) >= 3, "the blocker table parsed nearly empty"
    assert CITED_TEST.findall(TABLE.read_text(encoding="utf-8")), "no test citations parsed"


# ── Citations ─────────────────────────────────────────────────────────


def test_every_test_file_the_table_cites_exists():
    """A renamed test leaves a verdict supported by a path nobody can open.

    Found red: `dba3141` renamed two placement tests and the table kept the old
    names, so Gate P5 clause 2 read as PASS on evidence that could not be found.
    """
    cited = sorted(set(CITED_TEST.findall(TABLE.read_text(encoding="utf-8"))))
    missing = [path for path in cited if not (ROOT / path).exists()]
    assert not missing, (
        "the truth table cites test files that do not exist: "
        + ", ".join(missing)
        + ". A verdict is only as good as evidence someone can open — check "
        "whether the file was renamed and update the citation."
    )


# ── The row that tells someone what to do ─────────────────────────────


def test_the_blocker_table_names_every_failing_clause():
    """A FAIL with no blocker is a clause nobody knows is waiting on them."""
    failing = {clause for clause, verdict in _clause_verdicts().items() if verdict == "FAIL"}
    listed = set().union(*_blocker_rows().values()) if _blocker_rows() else set()
    unlisted = sorted(failing - listed)
    assert not unlisted, (
        f"these clauses are FAIL but appear under no blocker: {unlisted}. The "
        "blocker table is what someone reads to know what is waiting on them."
    )


def test_the_blocker_table_lists_no_clause_that_has_moved_on():
    """The direction that wastes someone's evening.

    A clause listed as blocked after it has passed sends the person with the
    hardware to do work that is already done — and quietly overstates the
    backlog, which is the same drift the tally guard exists to stop.
    """
    verdicts = _clause_verdicts()
    stale = sorted(
        f"{clause} (now {verdicts.get(clause, 'absent from the table')})"
        for clause in set().union(*_blocker_rows().values())
        if verdicts.get(clause) != "FAIL"
    )
    assert not stale, (
        "the blocker table lists clauses that are no longer FAIL: "
        + ", ".join(stale)
        + ". Remove them in the commit that moved the verdict."
    )


def test_no_clause_is_blocked_on_two_different_things():
    """One clause, one thing in the way. Two rows means one of them is stale."""
    seen = collections.Counter(
        clause for clauses in _blocker_rows().values() for clause in clauses
    )
    doubled = sorted(clause for clause, count in seen.items() if count > 1)
    assert not doubled, f"listed under more than one blocker: {doubled}"
