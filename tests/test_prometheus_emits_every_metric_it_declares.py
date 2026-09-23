"""Every metric block in the Prometheus exposition must actually emit a series.

`/metrics/prometheus` builds its body from ~30 `try:` blocks, each of which
queries the database and then appends its HELP/TYPE/value lines *inside* the
same `try`. That shape means a query fault does not produce a wrong number — it
produces no series at all, and the `except` logs at `debug`.

`xcelsior_inference_tokens_per_second` was in that state. Its query read
`result->>'tokens_generated' FROM jobs`, and `jobs` has no `result` column; it
holds `payload`, `spec` and `reason_details`. So the metric had never once been
exported, and any dashboard panel or alert rule on it had always shown no data
— which looks identical to an idle fleet.

Rather than pin that one metric, this asserts the invariant for the ones where
it holds: an *unlabelled* metric that `routes/health.py` declares with a
`# TYPE` literal must appear as a sample, because its value line is appended in
the same `try` that ran the query. No sample means the query raised.

Three kinds are exempt, and each exemption is derived from the source rather
than listed, so a new metric is covered without editing this file:

- A labelled metric emitted in a loop (`xcelsior_projection_deliveries` per
  sink) legitimately has zero series when there is nothing to report.
- A histogram emits `_bucket`/`_sum`/`_count`, never a bare name.
- `generate_latest(REGISTRY)` is appended too, and a `prometheus_client`
  Counter with labels correctly declares TYPE with no sample until its first
  observation. Those are not declared in `routes/health.py`, so they are out of
  scope here by construction.

The sample values are all zero in a test database. That is fine and is the
point — this checks the series exists, not what it holds.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app

client = TestClient(app)

TYPE_LINE = re.compile(r"^#\s*TYPE\s+(\S+)\s+(\S+)", re.MULTILINE)
# `<name>{labels} value` or `<name> value`, but not a comment.
SAMPLE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)(?:\{[^}]*\})?\s+\S+", re.MULTILINE)


@pytest.fixture(scope="module")
def exposition() -> str:
    r = client.get("/metrics/prometheus")
    assert r.status_code == 200, r.text[:300]
    return r.text


def test_the_inference_throughput_metric_is_exported(exposition):
    """The specific one that was missing, named so the regression is legible."""
    assert "xcelsior_inference_tokens_per_second" in exposition, (
        "the metric is absent — its query failed and the whole block, "
        "HELP/TYPE/value included, was skipped by the surrounding except"
    )


def test_every_unlabelled_hand_rolled_metric_has_a_sample(exposition):
    health_src = (Path(__file__).resolve().parents[1] / "routes" / "health.py").read_text(
        encoding="utf-8"
    )
    declared = set(re.findall(r"#\s*TYPE\s+([a-zA-Z_][a-zA-Z0-9_]*)", health_src))
    assert declared, "no `# TYPE` literals in routes/health.py — did the builder move?"

    # Emission sites are read through the AST, not line by line: the labelled
    # ones split the name and its label set across implicitly-concatenated
    # literals —
    #
    #     "xcelsior_projection_deliveries"
    #     f'{{sink={sink_label},status="{status}"}} {sink[status]}'
    #
    # — so the line holding the name has no `{` on it. Python merges the two
    # into one JoinedStr, so reading the tree sees the whole series.
    emitted: list[str] = []
    for node in ast.walk(ast.parse(health_src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            emitted.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            emitted.append(
                "".join(
                    v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "\x00"
                    for v in node.values
                )
            )

    required = set()
    for name in declared:
        sites = [t for t in emitted if name in t and not t.lstrip().startswith("#")]
        if any("{" in t for t in sites):
            continue  # labelled — zero series is a legitimate state
        if any(f"{name}_{sfx}" in health_src for sfx in ("bucket", "sum", "count")):
            continue  # histogram — emits suffixed series, never a bare name
        required.add(name)

    sampled = set(SAMPLE.findall(exposition))
    missing = sorted(required - sampled)
    assert not missing, (
        "declared by a `# TYPE` line in routes/health.py, emitted unlabelled, but "
        f"absent from the response — the query behind each of these raised: {missing}"
    )
