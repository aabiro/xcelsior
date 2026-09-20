"""In-memory auth with more than one worker is never correct, so it must not boot.

`XCELSIOR_PERSISTENT_AUTH=false` makes `routes._deps._users_db` — a plain dict
in the worker's own memory — the user *store*, not a cache. With more than one
gunicorn worker, registration lands on whichever worker served that request and
the next login round-robins elsewhere, so a worker that never saw the
registration answers 401.

Measured on this repo's own test stack, default two workers:

    401 401 401 401 200 200 401 200 401 401

Nothing logged an error. Each worker was behaving correctly. The user existed in
no database because it had never been written to one, and the register response
carried a complete user object because the worker that served it really had
created one. It presents as a password that works about half the time, which is
close to the hardest thing to diagnose from the outside — the search goes to the
database, the cache, the load balancer, and the answer is in a module-level
dict.

Guarded at boot rather than documented, because a warning in a log nobody reads
is how this survived in the first place.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "gunicorn.conf.py"


def _load(env: dict[str, str]) -> subprocess.CompletedProcess:
    """Execute the gunicorn config the way gunicorn does: as a module."""
    return subprocess.run(
        [sys.executable, "-c", f"exec(open({str(CONF)!r}).read())"],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("workers", ["2", "4"])
def test_in_memory_auth_with_many_workers_refuses_to_boot(workers: str) -> None:
    result = _load({"XCELSIOR_PERSISTENT_AUTH": "false", "GUNICORN_WORKERS": workers})
    assert result.returncode != 0, (
        f"gunicorn accepted in-memory auth with {workers} workers. Logins would "
        "fail on every worker that did not serve the registration, with nothing "
        "logged."
    )
    assert "XCELSIOR_PERSISTENT_AUTH" in result.stderr
    assert "in-memory auth is per-process" in result.stderr


def test_in_memory_auth_with_one_worker_is_allowed() -> None:
    """A single worker makes the in-memory store coherent — that is a real setup."""
    result = _load({"XCELSIOR_PERSISTENT_AUTH": "false", "GUNICORN_WORKERS": "1"})
    assert result.returncode == 0, (
        f"a single-worker in-memory setup was refused, which is over-blocking: "
        f"{result.stderr[-300:]}"
    )


@pytest.mark.parametrize("workers", ["1", "2", "8"])
def test_persistent_auth_is_never_blocked(workers: str) -> None:
    """The production posture must not be affected at any worker count."""
    for value in ("true", "TRUE", ""):
        result = _load({"XCELSIOR_PERSISTENT_AUTH": value, "GUNICORN_WORKERS": workers})
        assert result.returncode == 0, (
            f"persistent auth with {workers} workers was refused "
            f"(XCELSIOR_PERSISTENT_AUTH={value!r}): {result.stderr[-300:]}"
        )


def test_the_default_posture_boots() -> None:
    """No env at all is production's shape: persistent auth, two workers."""
    assert _load({}).returncode == 0
