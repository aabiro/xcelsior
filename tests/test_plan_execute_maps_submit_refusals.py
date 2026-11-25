"""An approved plan that the scheduler refuses must say why, not answer 500.

``/instance`` maps the plain exceptions ``submit_job`` raises to 402, 503 and
409. Plan execution, which every MCP launch goes through, let them escape as
"Internal Server Error". The live rehearsal found it on 2026-11-24 by launching
with the scheduler paused.
"""

from __future__ import annotations

import pytest

from control_plane.launch import service
from control_plane.spot_admission import SpotAdmissionRejected
from service_tier_pricing import (
    TierFundingRequired,
    TierPricingUnavailable,
    TierSchedulingUnavailable,
)


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (TierFundingRequired("fund the tier"), 402, "tier_funding_required"),
        (TierSchedulingUnavailable("scheduler paused"), 503, "scheduling_unavailable"),
        (TierPricingUnavailable("no price"), 409, "launch_refused"),
        (SpotAdmissionRejected("no spot capacity"), 409, "launch_refused"),
    ],
)
def test_each_refusal_keeps_the_status_instance_gives_it(exc, status, code):
    refusal = service._submit_refusal(exc)
    assert refusal is not None
    assert (refusal.status, refusal.code, refusal.detail) == (status, code, str(exc))


def test_an_unknown_failure_stays_a_fault():
    assert service._submit_refusal(RuntimeError("bug")) is None


def test_execute_raises_the_refusal_rather_than_the_raw_exception():
    # The mapping is worthless if execute() still re-raises the original.
    import inspect

    body = inspect.getsource(service.execute)
    assert "_submit_refusal(exc)" in body
    assert "raise refusal from exc" in body
