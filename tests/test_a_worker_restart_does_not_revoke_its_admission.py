"""A worker restart must not destroy the operator's admission decision.

`worker_agent.py` used to call `DELETE /host/{HOST_ID}` on every graceful
shutdown. That endpoint removes the `hosts` row, and both
`host_admission_decisions` and `host_admission_evidence` are ON DELETE CASCADE —
so an ordinary `systemctl restart` deleted the operator-signed admission
decision and every evidence record, then re-registered as a fresh `pending`
host that is refused work with 403.

Measured on production 2026-08-29: `GET /agent/v2/work/{host}` returned 204 up
to 09:53:21 and 403 from 09:53:27 — the moment of a restart — with
`admission_version` back to 0 and zero rows in `host_admission_decisions`. It
had happened at least once before and looked like a database losing writes.

This is a privilege inversion: the host revoking an operator's decision about
itself. Admission is versioned with `expected_version` and idempotency keys
precisely because it is meant to be durable and auditable, and nothing about a
restart is a decommission.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "worker_agent.py"


def _source() -> str:
    return WORKER.read_text(encoding="utf-8")


def test_the_shutdown_path_does_not_unconditionally_delete_the_host() -> None:
    src = _source()
    # Find the DELETE /host/ call and walk back to the nearest guard.
    idx = src.find('_api_url(f"/host/{HOST_ID}")')
    assert idx != -1, "the deregistration call was removed entirely; this test needs updating"

    preceding = src[:idx]
    # The call must sit inside an explicit opt-in, not run on every exit.
    assert "_DEREGISTER_ON_EXIT" in preceding[-1500:], (
        "DELETE /host/{HOST_ID} is not gated by _DEREGISTER_ON_EXIT — a graceful "
        "shutdown will cascade-delete this host's admission decision and evidence"
    )


def test_deregistration_is_off_by_default() -> None:
    """A default-on flag would preserve the bug for every existing deployment."""
    src = _source()
    match = re.search(
        r'_DEREGISTER_ON_EXIT\s*=\s*\(\s*\n?\s*os\.environ\.get\(\s*\n?\s*"XCELSIOR_WORKER_DEREGISTER_ON_EXIT",\s*"([^"]*)"',
        src,
    )
    assert match is not None, "could not find the _DEREGISTER_ON_EXIT definition"
    default = match.group(1).strip().lower()
    assert default not in ("1", "true", "yes", "on"), (
        f"deregistration defaults to {default!r}, i.e. still on — every restart "
        "would keep destroying admission"
    )


def test_the_reason_is_recorded_where_someone_would_reintroduce_it() -> None:
    """The next person to 'clean up' this shutdown path must see why it is guarded."""
    src = _source()
    idx = src.find('_api_url(f"/host/{HOST_ID}")')
    context = src[max(0, idx - 2000):idx]
    assert "CASCADE" in context, (
        "the shutdown path does not explain that host deletion cascades to "
        "admission decisions — without that, this is trivially reintroduced"
    )


# ── Server side ────────────────────────────────────────────────────────
#
# The agent fix cannot bind a client. Old worker binaries in the field still
# call DELETE /host/{id} on shutdown, so the invariant is enforced by the API
# as well: a machine credential may not delete an *admitted* host.

# Importing the application from a test module is not free, and the preamble
# below is the established way to do it safely — copied from
# `tests/test_hosts_endpoints_coverage.py`, which imports `api` and coexists
# with the full suite.
#
# Without it, importing `routes.hosts` here broke
# `tests/test_token_billing_closure.py` with `invalid_client / Unknown OAuth
# client` — three tests that pass perfectly on their own. The env has to be
# pinned *before* the import, because the app freezes auth configuration at
# import time in three separate places (`routes._deps`, `routes.auth`, `api`).
import os  # noqa: E402

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

from routes.hosts import machine_credential_may_not_delete as _blocked  # noqa: E402

_WORKER = {"grant_type": "client_credentials"}
_ADMIN = {"grant_type": "client_credentials", "is_admin": True, "role": "admin"}
_HUMAN = {"grant_type": "authorization_code", "user_id": "u-1"}
_ADMITTED = {"host_id": "h", "admission_state": "admitted"}
_PENDING = {"host_id": "h", "admission_state": "pending"}


def test_a_worker_credential_cannot_delete_an_admitted_host() -> None:
    assert _blocked(_WORKER, _ADMITTED) is True


def test_a_worker_credential_may_still_delete_a_pending_host() -> None:
    """Registration cleanup must keep working — this guard is narrow on purpose."""
    assert _blocked(_WORKER, _PENDING) is False


def test_a_human_operator_is_not_blocked() -> None:
    """Removing your own admitted host from the dashboard must still work."""
    assert _blocked(_HUMAN, _ADMITTED) is False


def test_an_unknown_host_is_not_blocked_here() -> None:
    """A missing row is a 404 concern, not this guard's."""
    assert _blocked(_WORKER, None) is False


def test_the_state_comparison_is_not_case_or_whitespace_fragile() -> None:
    assert _blocked(_WORKER, {"admission_state": " Admitted "}) is True
