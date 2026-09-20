"""Gate P7, against a live server: N nodes from one snapshot, byte-identical.

The clause: *"a sweep of N nodes from one snapshot is byte-identical in
environment; a snapshot records its lineage."*

Both halves need real nodes. `tests/test_a_sweep_is_a_record.py` and
`tests/test_sweep_fingerprints_are_compared.py` already pin the bookkeeping and
the comparison logic in-process — what they cannot establish is that two
machines launched from one image actually come up the same, which is the only
reason the clause exists. An image that drifts between launches is invisible to
every in-process test and ruins exactly the workload sweeps are for.

Launches N real instances and is billed like it. Skips without a fleet.

**A missing fingerprint is never agreement.** `compare_fingerprints` is explicit
about this and so is the assertion below: a collector that errored and reported
nothing must not read as "all members agree", because that is the phantom pass
this directory exists to refuse.
"""

from __future__ import annotations

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

#: Two is the smallest N that can disagree. A sweep of one proves nothing about
#: identity and would pass a broken comparison.
SWEEP_SIZE = 2

#: Sweeps launch real machines and pull an image; this is generous on purpose.
READY_TIMEOUT_SEC = 900
POLL_SEC = 20


def _get(path: str) -> dict:
    r = requests.get(f"{BASE}{path}", headers=auth(), timeout=30)
    r.raise_for_status()
    return r.json()


@pytest.fixture(scope="module")
def sweep() -> dict:
    """A snapshot of a running instance, swept to SWEEP_SIZE nodes."""
    images = _get("/user-images").get("images") or []
    if not images:
        pytest.skip(
            "no user image to sweep from. Snapshot a configured instance first "
            "(create_instance_snapshot); this gate is about launching from one, "
            "not about producing one."
        )
    image_id = images[0].get("image_id") or images[0].get("id")

    created = requests.post(
        f"{BASE}/api/v1/image-sweeps",
        headers=auth(),
        json={"image_id": image_id, "count": SWEEP_SIZE},
        timeout=60,
    )
    if created.status_code == 402:
        pytest.skip("wallet has no balance; a sweep launches real instances")
    created.raise_for_status()
    body = created.json()
    sweep_id = body.get("sweep_id") or (body.get("sweep") or {}).get("sweep_id")
    assert sweep_id, f"no sweep_id in {body}"

    yield {"sweep_id": sweep_id, "image_id": image_id}

    # A sweep of real nodes bills until it is stopped, and this gate fails in the
    # interesting cases — exactly when nobody is watching the bill.
    for member in (_get(f"/api/v1/image-sweeps/{sweep_id}").get("sweep") or {}).get("members", []):
        job_id = member.get("job_id")
        if job_id:
            requests.post(f"{BASE}/instances/{job_id}/cancel", headers=auth(), timeout=60)


def _await_reported(sweep_id: str) -> dict:
    """Wait until every member has reported a fingerprint, or time out saying so."""
    deadline = time.time() + READY_TIMEOUT_SEC
    last: dict = {}
    while time.time() < deadline:
        last = _get(f"/api/v1/image-sweeps/{sweep_id}")
        verification = last.get("verification") or {}
        if int(verification.get("reported") or 0) >= SWEEP_SIZE:
            return last
        time.sleep(POLL_SEC)
    pytest.fail(
        f"only {(last.get('verification') or {}).get('reported', 0)}/{SWEEP_SIZE} "
        f"members reported a fingerprint within {READY_TIMEOUT_SEC}s. The sweep "
        f"cannot be judged identical or otherwise: {last.get('verification')}"
    )


def test_every_member_reports_a_fingerprint(sweep) -> None:
    """The precondition, asserted separately so a timeout is not read as agreement."""
    body = _await_reported(sweep["sweep_id"])
    verification = body.get("verification") or {}
    assert int(verification.get("reported") or 0) == SWEEP_SIZE, (
        f"{verification.get('reported')} of {SWEEP_SIZE} members reported. A "
        f"missing fingerprint is never agreement: {verification}"
    )


def test_the_swept_nodes_are_byte_identical(sweep) -> None:
    """Gate P7's first half."""
    body = _await_reported(sweep["sweep_id"])
    verification = body.get("verification") or {}
    assert verification.get("verified") is True, (
        f"nodes launched from one image do not agree on their environment: "
        f"{verification}. That is the defect this gate exists for — an image "
        f"that drifts between launches ruins the workload sweeps are for."
    )


def test_the_snapshot_records_its_lineage(sweep) -> None:
    """Gate P7's second half: provenance, on the live record."""
    images = _get("/user-images").get("images") or []
    mine = next(
        (i for i in images if (i.get("image_id") or i.get("id")) == sweep["image_id"]),
        None,
    )
    assert mine is not None, f"image {sweep['image_id']} vanished from the library"
    lineage = {k: mine.get(k) for k in ("source_job_id", "created_at", "created_by")}
    missing = [k for k, v in lineage.items() if not v]
    assert not missing, (
        f"the image records no {missing} — without provenance nobody can say what "
        f"this environment was built from or by which run: {mine}"
    )
