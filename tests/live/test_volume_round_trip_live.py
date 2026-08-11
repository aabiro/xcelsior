"""Gate P3 clause 3: a volume written in one instance is readable in a second.

The clause is about the *round trip*, so it needs two instances and one volume.
Writing and reading in the same container proves the mount works and nothing
about whether the data survived the boundary — which is the only interesting
question.

Skips without a fleet. Launches two real instances.
"""

from __future__ import annotations

import time
import uuid

import pytest

requests = pytest.importorskip("requests")

from tests.live._fleet import (  # noqa: E402
    BASE,
    FLEET_EXPECTED,
    MISSING_CREDENTIALS,
    MISSING_FLEET,
    TOKEN,
    auth,
)

pytestmark = [
    pytest.mark.skipif(not BASE or not TOKEN, reason=MISSING_CREDENTIALS),
    pytest.mark.skipif(not FLEET_EXPECTED, reason=MISSING_FLEET),
]

MARKER = f"xcelsior-round-trip-{uuid.uuid4().hex[:12]}"


@pytest.fixture(scope="module")
def two_instances():
    """Two interactive instances, both torn down."""
    ids = []
    for n in (1, 2):
        created = requests.post(
            f"{BASE}/instance",
            headers=auth(),
            json={
                "name": f"volume-round-trip-{n}-{int(time.time())}",
                "vram_needed_gb": 1, "num_gpus": 1, "interactive": True,
            },
            timeout=60,
        )
        assert created.status_code == 200, created.text
        body = created.json()
        ids.append(body.get("job_id") or body.get("instance", {}).get("job_id"))

    deadline = time.time() + 600
    while time.time() < deadline:
        states = []
        for job_id in ids:
            b = requests.get(f"{BASE}/instance/{job_id}", headers=auth(), timeout=30).json()
            states.append(str(b.get("instance", b).get("status") or ""))
        if all(s == "running" for s in states):
            break
        if any(s in ("failed", "cancelled") for s in states):
            pytest.fail(f"an instance did not start: {states}")
        time.sleep(10)

    yield ids[0], ids[1]
    for job_id in ids:
        requests.post(f"{BASE}/instances/{job_id}/cancel", headers=auth(), timeout=60)


@pytest.fixture(scope="module")
def volume():
    created = requests.post(
        f"{BASE}/api/v2/volumes",
        headers=auth(),
        json={"name": f"volume-round-trip-{int(time.time())}", "size_gb": 1},
        timeout=60,
    )
    assert created.status_code in (200, 201), created.text
    volume_id = created.json().get("volume_id") or created.json().get("id")
    assert volume_id, created.text
    yield volume_id
    requests.delete(f"{BASE}/api/v2/volumes/{volume_id}", headers=auth(), timeout=60)


def _attach(volume_id: str, job_id: str):
    return requests.post(
        f"{BASE}/api/v2/volumes/{volume_id}/attach",
        headers=auth(), json={"instance_id": job_id}, timeout=120,
    )


def _detach(volume_id: str, job_id: str):
    return requests.post(
        f"{BASE}/api/v2/volumes/{volume_id}/detach",
        headers=auth(), json={"instance_id": job_id}, timeout=120,
    )


def test_one_volume_attaches_to_a_second_instance_after_the_first(volume, two_instances):
    """The clause's structural half: the same volume reaches a *second* container.

    A volume that can only ever be attached to the instance that created it
    cannot round-trip anything, and that failure is invisible to a
    write-then-read inside one container.
    """
    first, second = two_instances
    attached = _attach(volume, first)
    assert attached.status_code in (200, 202), attached.text

    released = _detach(volume, first)
    assert released.status_code in (200, 202), released.text

    reattached = _attach(volume, second)
    assert reattached.status_code in (200, 202), (
        f"the volume would not attach to a second instance: {reattached.text[:300]}"
    )


def test_the_volume_reports_the_second_instance_as_its_holder(volume, two_instances):
    """State the control plane can be held to, after the move."""
    _, second = two_instances
    body = requests.get(f"{BASE}/api/v2/volumes/{volume}", headers=auth(), timeout=30).json()
    holder = str(body.get("attached_to") or body.get("instance_id") or "")
    assert holder == second, f"volume reports holder {holder!r}, expected {second!r}"


@pytest.mark.skipif(
    not __import__("os").environ.get("XCELSIOR_LIVE_SSH_KEY"),
    reason=(
        "set XCELSIOR_LIVE_SSH_KEY to a private key registered on the account to "
        "assert the byte-level round trip; the attachment assertions above run "
        "without it"
    ),
)
def test_bytes_written_in_the_first_instance_are_read_in_the_second(volume, two_instances):
    """The clause in full: a value that actually crossed the boundary.

    Marker is random per run — a stale file from an earlier run would otherwise
    pass this with nothing written today.
    """
    import os
    import subprocess

    key = os.environ["XCELSIOR_LIVE_SSH_KEY"]
    first, second = two_instances

    def _ssh(job_id: str, command: str) -> subprocess.CompletedProcess:
        inst = requests.get(
            f"{BASE}/instance/{job_id}", headers=auth(), timeout=30
        ).json()
        inst = inst.get("instance", inst)
        port = int(inst["ssh_port"])
        host = inst.get("ssh_host") or "connect.xcelsior.ca"
        return subprocess.run(
            ["ssh", "-i", key, "-p", str(port),
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", "ConnectTimeout=20", f"root@{host}", command],
            capture_output=True, text=True, timeout=120,
        )

    _attach(volume, first)
    wrote = _ssh(first, f"mkdir -p /mnt/volume && echo {MARKER} > /mnt/volume/marker.txt && sync")
    assert wrote.returncode == 0, wrote.stderr[:300]

    _detach(volume, first)
    _attach(volume, second)
    read = _ssh(second, "cat /mnt/volume/marker.txt")
    assert MARKER in read.stdout, (
        "the marker did not survive the move between instances — the volume "
        f"attached but its contents did not cross: {read.stderr[:200]}"
    )
