"""The compliance page's SLA table was permanently empty, and said so politely.

`GET /api/sla/targets` answers `{"ok": true, "targets": {...}}`, keyed by tier
name, each value an `sla.SLATarget`: `availability_pct`, `latency_ttft_ms`,
`throughput_floor_pct`, `response_time_hours`, `heartbeat_grace_sec`,
`max_thermal_c`.

The dashboard read `d.tiers` — a key the route has never sent — and its `|| {}`
turned `undefined` into an empty object, an empty tier list, and the message
**"No SLA tier data available"**. Which reads as *"the platform has no SLA
tiers"* rather than *"this screen is broken"*, so it could sit there forever.

The inner field names were wrong too, independently: the page expected
`uptime_pct` and `credit_pct_10 / _25 / _100`. None of those exist on any
endpoint. `penalty_rate` was `credit_pct_100 / 100`, so the "Penalty rate" row
was rendering a number derived from nothing — it is gone rather than guessed at,
because SLA credit is computed from **measured** uptime by
`sla.compute_credit_pct`, not carried as a per-tier constant. A per-tier penalty
figure would need the credit ladder exposed, which is a decision, not a repair.

## Why the fields are asserted against the dataclass

`SLATarget` is the definition; the route serialises it. Asserting the wire
against a list written here would be a second copy that drifts. Renaming a field
on the dataclass should fail this test, because the browser reads it by name.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import fields

os.environ.setdefault("XCELSIOR_ENV", "test")

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402
from sla import SLATarget  # noqa: E402
from tests._source_tree import strip_ts_comments  # noqa: E402

client = TestClient(app)
ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGE = ROOT / "frontend/src/app/(dashboard)/dashboard/compliance/page.tsx"


def _targets():
    response = client.get("/api/sla/targets")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "targets" in body, (
        f"the response key changed to {sorted(k for k in body if k != 'ok')}. "
        "The dashboard reads `targets`; update both together."
    )
    return body["targets"]


def test_the_key_is_targets_and_not_tiers():
    body = client.get("/api/sla/targets").json()
    assert "tiers" not in body, (
        "the route now also sends `tiers`. Two keys for one thing is how the "
        "frontend came to read the wrong one; pick one."
    )
    assert isinstance(_targets(), dict)
    assert len(_targets()) >= 2


def test_every_target_carries_every_field_the_dataclass_declares():
    declared = {f.name for f in fields(SLATarget)}
    for name, target in _targets().items():
        missing = declared - set(target)
        assert not missing, f"tier {name!r} is missing {sorted(missing)}"


def test_the_fields_the_page_used_to_read_do_not_exist():
    """`uptime_pct` / `credit_pct_*` were read for real and never sent."""
    for name, target in _targets().items():
        for invented in ("uptime_pct", "credit_pct_10", "credit_pct_25", "credit_pct_100"):
            assert invented not in target, (
                f"tier {name!r} now sends {invented!r}. If that is deliberate, the "
                "page can show a per-tier penalty again — but decide it rather "
                "than letting two names for one number coexist."
            )


def test_availability_is_a_percentage_and_ordered_by_tier_strength():
    values = [t["availability_pct"] for t in _targets().values()]
    for v in values:
        assert 0 < float(v) <= 100, f"availability_pct {v} is not a percentage"


def test_the_page_reads_the_key_the_route_sends():
    # Comments stripped: this file's own explanation quotes `d.tiers`, and a
    # guard that reports a file for documenting the defect it prevents is one
    # people learn to ignore. Same lesson as the host-key parity guard.
    page = strip_ts_comments(PAGE.read_text(encoding="utf-8"))
    assert "d.targets" in page, "the compliance page no longer reads `targets`"
    assert "d.tiers" not in page, (
        "the compliance page reads `d.tiers` again — the route does not send it, "
        "and `|| {}` turns that into a permanent 'No SLA tier data available'"
    )


def test_the_page_does_not_render_a_penalty_it_cannot_source():
    page = strip_ts_comments(PAGE.read_text(encoding="utf-8"))
    assert "penalty_rate" not in page, (
        "the page renders a penalty rate again. SLA credit is computed from "
        "measured uptime by `sla.compute_credit_pct`; there is no per-tier "
        "constant to show, so any figure here is invented."
    )
