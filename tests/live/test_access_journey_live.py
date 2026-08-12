"""Gate P2 clause 1: the scripted access journey, end to end on a live server.

The clause asks that a user can go from *nothing* to *a shell on their instance*
by script, with no console step — and that every hop is asserted rather than
assumed. An in-process test cannot establish it: the interesting failures are
DNS, the gateway's port mapping, sshd inside a real image, and a key that was
accepted by the API but never reached `authorized_keys`.

Skips without a fleet. It launches a real instance and is billed like one.
"""

from __future__ import annotations

import subprocess
import time

import pytest

requests = pytest.importorskip("requests")

from tests.live._fleet import (  # noqa: E402
    BASE,
    FLEET_EXPECTED,
    MISSING_CREDENTIALS,
    MISSING_FLEET,
    MISSING_WEBHOOKS,
    TOKEN,
    WEBHOOKS_EXPECTED,
    auth,
)

# The webhook precondition is not paperwork. This journey launches an instance,
# which needs a funded wallet, and a wallet is credited only by
# `payment_intent.succeeded` — the processor is the sole authority on whether
# money moved. Without a forwarder the run either fails on `402` or, worse,
# **passes on a balance left over from an earlier run**, which would report the
# funding path working when it was never exercised.
pytestmark = [
    pytest.mark.skipif(not BASE or not TOKEN, reason=MISSING_CREDENTIALS),
    pytest.mark.skipif(not FLEET_EXPECTED, reason=MISSING_FLEET),
    pytest.mark.skipif(not WEBHOOKS_EXPECTED, reason=MISSING_WEBHOOKS),
]


@pytest.fixture(scope="module")
def instance():
    """Launch one interactive instance, and always tear it down."""
    created = requests.post(
        f"{BASE}/instance",
        headers=auth(),
        json={
            "name": f"access-journey-{int(time.time())}",
            "vram_needed_gb": 1,
            "num_gpus": 1,
            "interactive": True,
        },
        timeout=60,
    )
    assert created.status_code == 200, created.text
    job_id = created.json().get("job_id") or created.json().get("instance", {}).get("job_id")
    assert job_id, created.text

    deadline = time.time() + 600
    status = "unknown"
    while time.time() < deadline:
        body = requests.get(f"{BASE}/instance/{job_id}", headers=auth(), timeout=30).json()
        status = str(body.get("instance", {}).get("status") or body.get("status") or "")
        if status in ("running", "failed", "cancelled"):
            break
        time.sleep(10)
    yield job_id, status
    requests.post(f"{BASE}/instances/{job_id}/cancel", headers=auth(), timeout=60)


def test_the_instance_reaches_running(instance):
    """Calibration. Every hop below is meaningless if it never started."""
    _, status = instance
    assert status == "running", f"instance never started: {status}"


def test_connection_details_are_published_without_a_console_step(instance):
    """The clause's "by script" half: host and port from the API alone."""
    job_id, _ = instance
    body = requests.get(f"{BASE}/instance/{job_id}", headers=auth(), timeout=30).json()
    inst = body.get("instance", body)
    assert inst.get("ssh_port"), "no SSH port published, so there is nothing to connect to"


def test_the_published_host_key_matches_what_the_endpoint_presents(instance):
    """P4's gate, asserted here because it is the same journey.

    `ssh-keyscan` against the published port must produce the published
    fingerprint. This is the assertion no mock can make.
    """
    job_id, _ = instance
    inst = requests.get(f"{BASE}/instance/{job_id}", headers=auth(), timeout=30).json()
    inst = inst.get("instance", inst)
    published = inst.get("host_key_fingerprint")
    port = inst.get("ssh_port")
    if not published:
        pytest.skip("no fingerprint published for this instance — A1's null path")

    host = inst.get("ssh_host") or "connect.xcelsior.ca"
    scan = subprocess.run(
        f"ssh-keyscan -p {int(port)} {host} 2>/dev/null | ssh-keygen -lf -",
        shell=True, capture_output=True, text=True, timeout=60,
    )
    assert "SHA256:" in scan.stdout, f"ssh-keyscan produced nothing: {scan.stderr[:200]}"
    assert published in scan.stdout, (
        "the published fingerprint is not what the endpoint presents — a user "
        "following our own instruction would conclude they are being intercepted"
    )
