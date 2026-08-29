"""The worker's v1→v2 map and the server's alias table must not drift.

`routes/agent_v2_aliases.ALIASES` mounts each worker endpoint at a `/agent/v2/`
path. `worker_agent._AGENT_V2_PATH_MAP` tells the worker to *call* those paths.
They are the same mapping written twice.

The duplication is not laziness — `worker_agent.py` ships as a single file to
GPU hosts (`scripts/deploy_worker_agent.sh`) and cannot import from `routes/`.
But two copies of one fact is precisely the shape this codebase keeps paying
for, and the failure here is bad: the worker calls a path the server does not
serve, gets a 404 through a gateway that serves `/agent/v2/*` only, and the
symptom is a host that goes quiet rather than an error anyone can attribute.

So the copy is allowed and pinned.

## Why the check is by translation rather than by table equality

The two tables are shaped differently on purpose. The server declares exact
routes with typed parameters (`/agent/logs/{job_id}`); the worker matches
literal prefixes it is handed at runtime (`/agent/logs/`). Comparing them
key-by-key would force one to adopt the other's shape for the guard's
convenience.

Instead every server alias is run through the worker's translator and must come
out at the server's v2 path. That tests the property that matters — *would the
worker reach this endpoint* — rather than that two dicts look alike.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("XCELSIOR_ENV", "test")

import worker_agent  # noqa: E402
from routes.agent_v2_aliases import ALIASES, SUPERSEDED  # noqa: E402

#: `{param}` → a value, so a declared route becomes a path the worker would
#: actually build.
_SAMPLE = {
    "host_id": "h-probe",
    "job_id": "j-probe",
    "image_id": "img-probe",
    "promotion_id": "p-probe",
    "sweep_id": "s-probe",
    "member_index": "0",
    "worker_id": "w-probe",
}


def _fill(path: str) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: _SAMPLE.get(m.group(1), "x"), path)


def test_the_translator_is_reachable_and_enabled_for_the_check():
    """Calibration — the map must exist and the translator must do something."""
    assert worker_agent._AGENT_V2_PATH_MAP, "the worker's v2 map is empty"
    assert worker_agent._to_agent_v2_path("/host") == "/agent/v2/hosts/heartbeat"


def test_every_server_alias_is_reachable_through_the_worker_translator():
    wrong = []
    for v1_path, _method, v2_path in ALIASES:
        got = worker_agent._to_agent_v2_path(_fill(v1_path))
        want = _fill(v2_path)
        if got != want:
            wrong.append(f"{v1_path}\n      worker builds: {got}\n      server serves: {want}")
    assert not wrong, (
        "the worker would not reach these aliases — it calls a path the server "
        "does not serve, and behind an /agent/v2-only gateway that is a 404 "
        "that reads as a dead host:\n    " + "\n    ".join(wrong)
    )


def test_the_worker_map_has_no_entry_the_server_does_not_serve():
    """A stale worker entry sends traffic nowhere."""
    served = {_fill(v2) for _v1, _m, v2 in ALIASES}
    served_prefixes = tuple(sorted({v2.split("{")[0] for _v1, _m, v2 in ALIASES}, key=len))
    orphans = [
        f"{v1} -> {v2}"
        for v1, v2 in worker_agent._AGENT_V2_PATH_MAP.items()
        if not any(v2.startswith(p) or p.startswith(v2) for p in served_prefixes)
        and v2 not in served
    ]
    assert not orphans, f"worker maps to paths the server does not serve: {orphans}"


def test_the_translator_never_rewrites_a_superseded_path():
    """Those need a *different* call, not a different URL.

    `/agent/commands/{host}` is poll+drain and `/agent/lease/*` is unfenced.
    Rewriting either to a `/agent/v2/` URL would make the worker look migrated
    while still speaking the protocol the fence replaced — the one failure this
    migration must not produce quietly.
    """
    for v1 in SUPERSEDED:
        filled = _fill(v1)
        assert worker_agent._to_agent_v2_path(filled) == filled, (
            f"{v1} was rewritten to a v2 URL; it is superseded, not relocated — "
            "the worker must call the fenced endpoint instead"
        )


def test_translation_is_off_unless_opted_in():
    """A worker that switches before the API serves v2 loses its heartbeat."""
    assert worker_agent._AGENT_V2_PATHS is False or os.environ.get("XCELSIOR_AGENT_V2_PATHS"), (
        "v2 paths must be opt-in via XCELSIOR_AGENT_V2_PATHS, not the default"
    )


def test_an_unmapped_path_is_left_alone():
    """Guesswork would turn an uncovered call into an unattributable 404."""
    assert worker_agent._to_agent_v2_path("/oauth/token") == "/oauth/token"
    assert worker_agent._to_agent_v2_path("/some/new/thing") == "/some/new/thing"
