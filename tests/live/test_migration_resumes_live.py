"""Gate P5 clause 1: a migrated job resumes from its checkpoint.

*"Proven by comparing state before and after — not by the absence of an error."*

`migrate_job` refuses to report `resumed=True` without a probe having run on
both sides, so the whole clause reduces to: supply a probe that reads something
only a genuinely-resumed process could still know, and assert it matched.

The probe here is the container's own uptime counter. A process that was
checkpointed and restored keeps counting from where it stopped; a container that
was merely *restarted* begins again. That distinction is the clause: a restart
looks exactly like a successful migration to anything that only checks for
errors.

Skips without a fleet and without two hosts — a same-host move would not
exercise the transfer, which is where the interesting failures are.
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
    TOKEN,
    auth,
)

pytestmark = [
    pytest.mark.skipif(not BASE or not TOKEN, reason=MISSING_CREDENTIALS),
    pytest.mark.skipif(not FLEET_EXPECTED, reason=MISSING_FLEET),
]


def _hosts() -> list[dict]:
    body = requests.get(f"{BASE}/hosts", headers=auth(), timeout=30).json()
    hosts = body.get("hosts", body if isinstance(body, list) else [])
    return [h for h in hosts if str(h.get("status")) == "active"]


@pytest.fixture(scope="module")
def two_hosts():
    hosts = _hosts()
    if len(hosts) < 2:
        pytest.skip(
            f"{len(hosts)} active host(s); a migration needs two, and a same-host "
            "move would not exercise the checkpoint transfer"
        )
    return hosts[0]["host_id"], hosts[1]["host_id"]


@pytest.fixture(scope="module")
def running_job(two_hosts):
    source, _ = two_hosts
    created = requests.post(
        f"{BASE}/instance",
        headers=auth(),
        json={
            "name": f"p5-migrate-{int(time.time())}",
            "vram_needed_gb": 1, "num_gpus": 1, "interactive": True,
            "host_id": source,
        },
        timeout=60,
    )
    assert created.status_code == 200, created.text
    body = created.json()
    job_id = body.get("job_id") or body.get("instance", {}).get("job_id")

    deadline = time.time() + 600
    while time.time() < deadline:
        b = requests.get(f"{BASE}/instance/{job_id}", headers=auth(), timeout=30).json()
        if str(b.get("instance", b).get("status")) == "running":
            break
        time.sleep(10)
    else:
        pytest.fail("instance never reached running")

    # Let it accumulate state worth comparing.
    time.sleep(30)
    yield job_id
    requests.post(f"{BASE}/instances/{job_id}/cancel", headers=auth(), timeout=60)


def _uptime_probe(job_id: str) -> int:
    """Whole minutes of container uptime, as the container itself reports.

    Read over the platform's own access path rather than from the control
    plane: the control plane's view of a job survives a restart, so it cannot
    tell a resumed process from a fresh one.
    """
    inst = requests.get(f"{BASE}/instance/{job_id}", headers=auth(), timeout=30).json()
    inst = inst.get("instance", inst)
    import os

    key = os.environ.get("XCELSIOR_LIVE_SSH_KEY")
    if not key:
        pytest.skip("set XCELSIOR_LIVE_SSH_KEY to read in-container state")
    out = subprocess.run(
        ["ssh", "-i", key, "-p", str(int(inst["ssh_port"])),
         "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
         f"root@{inst.get('ssh_host') or 'connect.xcelsior.ca'}",
         "cut -d. -f1 /proc/uptime"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr[:300]
    return int(out.stdout.strip()) // 60


def test_a_migrated_job_resumes_with_its_state(running_job, two_hosts):
    """The clause, and the only assertion in this file that matters."""
    from control_plane.scheduler.migration_executor import migrate_job

    _, target = two_hosts
    outcome = migrate_job(running_job, target, state_probe=_uptime_probe)

    assert outcome.ok, f"{outcome.failure_code}: {outcome.detail}"
    assert outcome.resumed is True, (
        f"state did not survive the move: before={outcome.state_before} "
        f"after={outcome.state_after}. `resumed is None` would mean nothing "
        "checked, which the clause rejects as proof."
    )
    assert outcome.target_host_id == target


def test_the_control_plane_agrees_the_job_moved(running_job, two_hosts):
    _, target = two_hosts
    inst = requests.get(
        f"{BASE}/instance/{running_job}", headers=auth(), timeout=30
    ).json()
    inst = inst.get("instance", inst)
    assert str(inst.get("host_id")) == target
    assert str(inst.get("status")) == "running"


def test_the_fingerprint_moved_with_the_container(running_job):
    """A migrated container has new host keys; the old one must not be served."""
    inst = requests.get(
        f"{BASE}/instance/{running_job}", headers=auth(), timeout=30
    ).json()
    inst = inst.get("instance", inst)
    published = inst.get("host_key_fingerprint")
    if not published:
        return  # null is legitimate; A1's honest path
    port = int(inst["ssh_port"])
    host = inst.get("ssh_host") or "connect.xcelsior.ca"
    scan = subprocess.run(
        f"ssh-keyscan -p {port} {host} 2>/dev/null | ssh-keygen -lf -",
        shell=True, capture_output=True, text=True, timeout=60,
    )
    assert published in scan.stdout, (
        "the published fingerprint is the pre-migration container's — a user "
        "verifying it would conclude they are being intercepted"
    )
