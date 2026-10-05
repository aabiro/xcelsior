"""A task a migration seeds must be a task something runs.

Migration 057 seeded `telemetry_partition_maintenance` into `scheduled_tasks`
and wrote that it "owns partition lifecycle" from then on. Nothing registered a
function under that name. The dispatcher claimed the row every day, logged
"Scheduled task 'telemetry_partition_maintenance' not found in registry", and
recorded `failed`. In production on 2026-10-05: `last_status=failed`,
`last_run_at=NULL` — it had never once run.

So `telemetry_samples` had partitions for exactly the three months 057 created,
July–September 2026, and from 1 October would have written into the DEFAULT
partition, which once populated *blocks* creating the month it overlaps.

The only test asserted that the row existed. A row is a promise; this checks
that the promise has someone keeping it.

Both sides are derived: seeded names from the migrations, registered names from
`bg_worker.main()` read statically, so the worker never starts.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "migrations" / "versions"
BG_WORKER = ROOT / "bg_worker.py"

_SEED = re.compile(r"INSERT\s+INTO\s+scheduled_tasks\b(.*?)(?:ON\s+CONFLICT|;|\"\"\")", re.I | re.S)


def _seeded_task_names() -> dict[str, str]:
    seeded: dict[str, str] = {}
    for path in sorted(MIGRATIONS.glob("*.py")):
        for statement in _SEED.findall(path.read_text(encoding="utf-8")):
            values = statement.split("VALUES", 1)[-1] if "VALUES" in statement.upper() else ""
            match = re.search(r"'([a-z][a-z0-9_]*)'", values)
            if match:
                seeded[match.group(1)] = path.name
    return seeded


def _registered_task_names() -> set[str]:
    tree = ast.parse(BG_WORKER.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "register_task"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return names


def test_the_scan_finds_the_seeded_tasks() -> None:
    """A regex that matches nothing would pass the test below vacuously."""
    seeded = _seeded_task_names()
    assert {"telemetry_partition_maintenance", "host_agent_token_expiry"} <= set(seeded), seeded
    assert len(_registered_task_names()) > 20


def test_every_seeded_task_is_registered_with_the_worker() -> None:
    registered = _registered_task_names()
    missing = {name: mig for name, mig in _seeded_task_names().items() if name not in registered}
    assert not missing, (
        "scheduled_tasks rows seeded with no handler — the dispatcher claims them, "
        "logs 'not found in registry' and records `failed`, forever: "
        + ", ".join(f"{n} (seeded by {m})" for n, m in sorted(missing.items()))
    )
