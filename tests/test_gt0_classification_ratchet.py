"""GT0's audit, made countable so it can only move forward.

The gate is *every operation tagged `covered` / `gap` / `internal` /
`redundant`, with a reason. Zero unclassified.* 516 operations, 158 classified,
358 outstanding.

A number in a commit message is not a ratchet. Without a check, the count drifts
upward silently — a new route module arrives unclassified, and the gate looks
exactly as far away as it did before while actually being further. This makes
the remaining audit a measurement.

**Why the number is not zero, and is not being closed cheaply.** `covered` was
derived from what the MCP server calls, which is evidence. `internal` and `gap`
were applied to *homogeneous* modules — every route in `routes/agent.py` is a
worker callback; `routes/volumes.py` is entirely what P3 needs and no tool
exposes — where the judgement is true of the whole module and a reviewer can
check it in one pass.

The rest are mixed. `routes/serverless.py` holds user-facing job submission next
to worker claim/heartbeat callbacks, so a module-level label would mislabel one
or the other. Splitting them needs product intent, which is what GT0 *is*.
Pattern-matching them into a green gate would be the failure this test exists to
prevent.

To make progress: classify rows, lower `MAX_UNCLASSIFIED` in the same commit.
"""

from __future__ import annotations

import collections
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "docs" / "generated" / "endpoint-inventory.md"

#: Lower this as rows are classified. It may never rise: a new endpoint arrives
#: classified, or the commit that adds it also classifies it.
MAX_UNCLASSIFIED = 358

#: The only labels GT0 accepts.
VALID_CLASSES = {"covered", "gap", "internal", "redundant"}

_ROW = re.compile(r"^\|\s*(GET|POST|PUT|PATCH|DELETE|HEAD)\s*\|")


def _rows() -> list[list[str]]:
    return [
        [cell.strip() for cell in line.split("|")]
        for line in INVENTORY.read_text(encoding="utf-8").splitlines()
        if _ROW.match(line)
    ]


def classification_counts() -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    for cells in _rows():
        counts[cells[5] or ""] += 1
    return counts


def test_the_inventory_still_parses():
    """Prove the reach: a format change would make every count below zero.

    If the table shape moved, `_rows()` would return nothing and the ratchet
    would report perfect compliance — the shape of a guard that passes because
    it is looking at the wrong thing.
    """
    rows = _rows()
    assert len(rows) > 500, (
        f"parsed only {len(rows)} inventory rows; the table format changed and "
        "every assertion below would pass vacuously"
    )


def test_unclassified_count_does_not_grow():
    """The ratchet."""
    unclassified = classification_counts()[""]
    assert unclassified <= MAX_UNCLASSIFIED, (
        f"{unclassified} operations are unclassified, up from "
        f"{MAX_UNCLASSIFIED}. A new endpoint is classified in the commit that "
        "adds it; this number never rises."
    )


def test_every_label_used_is_one_gt0_accepts():
    """A typo creates a class nobody audits and inflates apparent progress."""
    used = {label for label in classification_counts() if label}
    invalid = sorted(used - VALID_CLASSES)
    assert not invalid, (
        f"inventory rows carry labels GT0 does not define: {invalid}; "
        f"valid are {sorted(VALID_CLASSES)}"
    )


def test_every_classified_row_carries_a_reason():
    """A label without a reason is an assertion, not an audit.

    This is the check that stops the remaining 358 being closed by filling the
    class column and leaving notes blank.
    """
    unreasoned = [
        f"{cells[1]} {cells[2]}"
        for cells in _rows()
        if cells[5] and len(cells[6]) < 8
    ]
    assert not unreasoned, (
        "classified without a reason — the reason is the audit, the label is "
        f"just its summary: {unreasoned[:10]}"
    )


def test_gt0_is_not_reported_as_closed_while_rows_remain():
    """The gate's own claim, asserted rather than assumed.

    GT0 closes at zero. Until then, anything describing it as complete is
    wrong, and this test is where that would be caught.
    """
    counts = classification_counts()
    if counts[""] == 0:
        return
    assert MAX_UNCLASSIFIED > 0, (
        "MAX_UNCLASSIFIED is 0, which asserts GT0 is closed, but "
        f"{counts['']} rows are still unclassified"
    )
