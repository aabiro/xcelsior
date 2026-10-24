"""No runtime SQL writer may delegate lifecycle fields to projection triggers.

This is an inventory guard, complemented by real trigger-free transition tests.
Metadata-only updates do not change native lifecycle state. SQLite's small
development schema and historical Alembic migrations are outside the contract.
"""

import ast
import re

from tests.test_job_writer_inventory import _iter_py_files


def _statements():
    for path, rel in _iter_py_files():
        if rel.startswith("api/migrations/"):
            continue
        tree = ast.parse(path.read_text(), filename=rel)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"execute", "executemany"}
                and node.args
            ):
                continue
            query = node.args[0]
            if isinstance(query, ast.Constant) and isinstance(query.value, str):
                value = query.value
            elif isinstance(query, ast.JoinedStr):
                value = " ".join(
                    n.value
                    for n in query.values
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                )
            else:
                continue
            if "?" in value:  # SQLite uses ? placeholders; native SQL uses %s.
                continue
            yield rel, node.lineno, value


def test_every_job_status_writer_sets_native_phase():
    checked = set()
    for rel, line, query in _statements():
        update = re.search(
            r"UPDATE\s+(?:public\.)?jobs(?:\s+\w+)?\s+SET\s+(.*?)(?:\bWHERE\b|$)",
            query,
            re.I | re.S,
        )
        if update and re.search(r"(?:^|,)\s*status\s*=", update[1]):
            assert re.search(
                r"\bphase\s*=", update[1]
            ), f"{rel}:{line} delegates phase to a trigger"
            checked.add(rel)
        if re.search(r"INSERT\s+INTO\s+(?:public\.)?jobs\s*\(", query, re.I):
            columns = query.split("(", 1)[1].split(")", 1)[0]
            for column in ["phase", "desired_state", "queued_at"]:
                assert re.search(rf"\b{column}\b", columns), f"{rel}:{line} omits {column}"
    assert {
        "control_plane/attempts.py",
        "control_plane/leases.py",
        "control_plane/lifecycle.py",
        "control_plane/serverless_workers.py",
        "privacy/privacy_sinks.py",
    } <= checked


def test_every_host_admission_or_status_writer_sets_native_states():
    checked = set()
    for rel, line, query in _statements():
        update = re.search(
            r"UPDATE\s+(?:public\.)?hosts(?:\s+\w+)?\s+SET\s+(.*?)(?:\bWHERE\b|$)",
            query,
            re.I | re.S,
        )
        if update and re.search(r"(?:^|,)\s*(?:admission_state|status)\s*=", update[1]):
            for column in ["administrative_state", "availability_state"]:
                assert re.search(rf"\b{column}\s*=", update[1]), f"{rel}:{line} omits {column}"
            checked.add(rel)
        if re.search(r"INSERT\s+INTO\s+(?:public\.)?hosts\s*\(", query, re.I):
            columns = query.split("(", 1)[1].split(")", 1)[0]
            for column in ["administrative_state", "availability_state"]:
                assert re.search(rf"\b{column}\b", columns), f"{rel}:{line} omits {column}"
    assert {"hosts/host_admission.py", "privacy/privacy_sinks.py"} <= checked
