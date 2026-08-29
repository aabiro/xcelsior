"""An alias must not be counted as a second capability.

`routes/agent_v2_aliases.py` mounts 22 existing v1 handlers under `/agent/v2/`.
They are the *same function objects* — same auth, same behaviour, one capability
reachable at two paths.

The endpoint classification answers "is this capability exposed as a tool?", so
counting an alias again distorts the answer. Inheriting each v1 row's label was
the first thing tried here and it moved `gap` from 98 to 101 — three missing
tools that do not exist, on a number whose entire purpose is to measure how much
of GT0 is left. `redundant` is the label already used for compatibility shims
and superseded routes, and it is what an alias is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routes.agent_v2_aliases import ALIASES  # noqa: E402

CLASSIFICATION = ROOT / "docs" / "endpoint-classification.json"


def _classification() -> dict[str, dict[str, str]]:
    return json.loads(CLASSIFICATION.read_text(encoding="utf-8"))


def test_every_alias_is_classified() -> None:
    """The ratchet holds unclassified at zero; aliases are not exempt."""
    table = _classification()
    missing = [
        f"{method.upper()} {v2}" for _, method, v2 in ALIASES
        if f"{method.upper()} {v2}" not in table
    ]
    assert not missing, f"aliases absent from the classification: {missing}"


def test_no_alias_is_counted_as_a_gap_or_as_covered() -> None:
    """Only the v1 row may carry the capability's label."""
    table = _classification()
    wrong = {
        f"{method.upper()} {v2}": table[f"{method.upper()} {v2}"]["class"]
        for _, method, v2 in ALIASES
        if table.get(f"{method.upper()} {v2}", {}).get("class") != "redundant"
    }
    assert not wrong, (
        "these aliases are classified as something other than 'redundant', which "
        f"counts one capability twice in the GT0 audit: {wrong}"
    )


def test_each_alias_names_the_v1_route_it_duplicates() -> None:
    """A bare 'redundant' is unreviewable — the note must say redundant *with what*."""
    table = _classification()
    unexplained = [
        f"{method.upper()} {v2}"
        for v1, method, v2 in ALIASES
        if v1 not in table[f"{method.upper()} {v2}"]["notes"]
    ]
    assert not unexplained, (
        f"these alias notes do not name the v1 route they duplicate: {unexplained}"
    )
