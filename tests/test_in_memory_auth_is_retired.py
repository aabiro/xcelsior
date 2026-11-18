"""`XCELSIOR_PERSISTENT_AUTH=false` is refused, not ignored.

The flag used to keep users, sessions and API keys in dicts inside the worker.
Since 164 every table a user owns refers to `users.user_id` by foreign key, and
a user that exists only in a dict cannot satisfy one: the first insert that
names it fails. The mode was retired instead of being kept half-working.

An environment that still sets it was written for the old behaviour, so the API
refuses to import rather than quietly running persistent auth under a setting
that says otherwise.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parents[1] / "api"


def _import_deps(extra_env: dict[str, str]) -> subprocess.CompletedProcess:
    env = {**os.environ, **extra_env}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(API_DIR), str(API_DIR.parent), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", "import routes._deps"],
        cwd=API_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("off", ["false", "FALSE", "0", "no", "off"])
def test_the_retired_setting_refuses_to_start(off: str) -> None:
    result = _import_deps({"XCELSIOR_PERSISTENT_AUTH": off})
    assert result.returncode != 0, (
        f"XCELSIOR_PERSISTENT_AUTH={off!r} was accepted; the environment asked for "
        "in-memory users and silently got PostgreSQL ones"
    )
    assert "XCELSIOR_PERSISTENT_AUTH" in result.stderr, result.stderr[-500:]
    assert "Remove XCELSIOR_PERSISTENT_AUTH" in result.stderr, (
        "the refusal must say what to do:\n" + result.stderr[-500:]
    )


@pytest.mark.parametrize("value", ["true", ""])
def test_leaving_it_on_or_unset_still_starts(value: str) -> None:
    """Existing environments say `true`; that must keep booting."""
    result = _import_deps({"XCELSIOR_PERSISTENT_AUTH": value})
    assert result.returncode == 0, result.stderr[-800:]
