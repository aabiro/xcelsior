"""The SSH gateway hostname is written in four places. Pin them together.

`open_instance_access` hands a user the host they will type into `ssh`. That
hostname is not something the API returns per instance — the job record carries
`host_ip`, the tailnet address the dashboard shows under *"Direct SSH (requires
mesh network)"*, which a user off the mesh cannot reach. So every surface that
needs the connectable hostname carries its own default:

* `worker_agent.py` — injected into the container as `XCELSIOR_PUBLIC_SSH_HOST`
* `ai_assistant_config.py` — what the assistant tells people to type
* `frontend/.../instances/[id]/page.tsx` — the copy-to-clipboard command
* `mcp/src/config.ts` — what the MCP tool returns

Four copies of one string, and copies drift. `MCP_QUICK_CONNECT_SCOPES` drifted
from the frontend's list, then from the MCP suite's, in the same week — both
times a comment asking for them to be kept in sync was the only mechanism, and
a comment is not a mechanism.

The failure this prevents is quiet: the dashboard would keep showing the old
hostname while the tool returned the new one, and the mismatch would surface as
a user's `ssh` hanging with no error anyone can attribute. Gate P2 asks for
exactly the opposite — *"connection details in the instance view that match
exactly what the tool returns"*.
"""

from __future__ import annotations

import os
import pathlib
import re

os.environ.setdefault("XCELSIOR_ENV", "test")

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Each surface, and the pattern that extracts its default.
SOURCES = {
    "worker_agent.py": re.compile(r'"XCELSIOR_PUBLIC_SSH_HOST":\s*"([^"]+)"'),
    "ai_assistant_config.py": re.compile(
        r'SSH_HOST\s*=\s*os\.environ\.get\(\s*"XCELSIOR_SSH_HOST",\s*"([^"]+)"'
    ),
    "frontend/src/app/(dashboard)/dashboard/instances/[id]/page.tsx": re.compile(
        r'SSH_HOST\s*=\s*process\.env\.NEXT_PUBLIC_SSH_HOST\s*\|\|\s*"([^"]+)"'
    ),
    "mcp/src/config.ts": re.compile(
        r'process\.env\.XCELSIOR_SSH_HOST\s*\|\|\s*"([^"]+)"'
    ),
}


def _defaults() -> dict[str, str]:
    found: dict[str, str] = {}
    for rel, pattern in SOURCES.items():
        path = ROOT / rel
        assert path.exists(), f"{rel} is gone; update SOURCES rather than deleting the pin"
        match = pattern.search(path.read_text(encoding="utf-8"))
        assert match, (
            f"no SSH host default found in {rel}. Either it was renamed — in "
            "which case fix the pattern — or the default was removed, in which "
            "case that surface now has no fallback at all."
        )
        found[rel] = match.group(1)
    return found


def test_every_surface_still_declares_one():
    """Prove the reach: four patterns, four matches, or the pin is vacuous."""
    found = _defaults()
    assert len(found) == len(SOURCES)
    assert all(found.values())


def test_all_four_agree():
    """The pin itself."""
    found = _defaults()
    distinct = set(found.values())
    assert len(distinct) == 1, (
        "the SSH gateway hostname disagrees across surfaces, so the dashboard "
        "and the MCP tool would tell a user to connect to different hosts: "
        f"{found}"
    )


def test_it_is_not_the_tailnet_address():
    """The specific wrong answer.

    `host_ip` is a `100.64.0.0/10` CGNAT address on the tailnet. Handing one to
    a user is the failure mode `open_instance_access` is tested against, and a
    default that drifted into that range would defeat the tool's own guard.
    """
    host = next(iter(set(_defaults().values())))
    assert not re.match(r"^100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.", host), (
        f"the SSH gateway default is a tailnet address ({host}); users off the "
        "mesh cannot reach it"
    )
    assert not host.startswith("127."), f"the SSH gateway default is loopback ({host})"
