"""Local-state boundary (Track B B9.4, companion §10.2).

SQLite and local files are permitted in exactly four places:

  1. a host worker's local command/execution journal;
  2. local developer/test mode;
  3. CLI credentials/config on a user's machine;
  4. one-time migration staging that is never served concurrently.

Everything else — anything shared between processes, anything that decides
money or execution authority — belongs in PostgreSQL. This gate keeps that
list explicit, so a new module cannot quietly reintroduce a second source
of truth the way `lightning.py` (SQLite deposits) and `slurm_adapter.py`
(a JSON mapping file) did before migration 060.

Companion §10.1/§10.2; Track B §B9.4.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Modules allowed to import sqlite3, each for a §10.2 reason.
SQLITE_ALLOWED = {
    # The backend abstraction itself. Production selection of the SQLite
    # backend is refused by control_plane.startup_validation.
    "db.py": "backend abstraction; production use blocked by the startup validator",
    # Reads the SQLite side during the one-time consolidation.
    "scripts/migrate_sqlite_to_pg.py": "one-time migration staging (§10.2 item 4)",
    # Test/dev-only path, gated on XCELSIOR_DB_BACKEND != postgres.
    "bitcoin.py": "dev/test backend branch; production refuses a non-postgres backend",
}

# Developer tooling that runs on a workstation, never in a deployed image
# (§10.2 items 2 and 3). Exempted as a directory rather than file-by-file,
# with `test_local_tooling_is_not_deployed` keeping the exemption honest.
LOCAL_TOOLING_PREFIX = "scripts/local/"

# Directories that are not production control-plane code. Note this repo
# has BOTH `.venv` and `venv`; any directory containing `pyvenv.cfg` is
# treated as a virtualenv so a third one cannot silently slip through.
_SKIP_PARTS = {
    ".venv", "venv", "node_modules", "build", "dist", "__pycache__",
    "tests", "migrations", "frontend", "mcp", "desktop", "sprites",
    ".git", "site-packages",
}

_SQLITE_IMPORT_RE = re.compile(r"^\s*(?:import\s+sqlite3|from\s+sqlite3\s+import)", re.M)


def _virtualenv_roots() -> set[Path]:
    return {cfg.parent for cfg in PROJECT_ROOT.glob("*/pyvenv.cfg")}


def _python_sources() -> list[Path]:
    venvs = _virtualenv_roots()
    out = []
    for path in PROJECT_ROOT.glob("**/*.py"):
        if set(path.parts) & _SKIP_PARTS:
            continue
        if any(venv in path.parents for venv in venvs):
            continue
        out.append(path)
    return out


def _rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def test_only_approved_modules_import_sqlite() -> None:
    offenders = []
    for path in _python_sources():
        rel = _rel(path)
        if rel in SQLITE_ALLOWED or rel.startswith(LOCAL_TOOLING_PREFIX):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if _SQLITE_IMPORT_RE.search(text):
            offenders.append(rel)

    assert not offenders, (
        "these modules import sqlite3 without being on the §10.2 allowed "
        f"list: {offenders}. Shared control-plane or money state belongs in "
        "PostgreSQL — SQLite is permitted only for a worker's node-local "
        "journal, dev/test mode, CLI config, or one-time migration staging. "
        "If this is a legitimate new case, add it to SQLITE_ALLOWED with the "
        "reason."
    )


def test_allowed_list_has_no_stale_entries() -> None:
    """A retired exception must be removed, not left as dead config."""
    stale = []
    for rel in SQLITE_ALLOWED:
        path = PROJECT_ROOT / rel
        if not path.is_file():
            stale.append(f"{rel} (file is gone)")
            continue
        if not _SQLITE_IMPORT_RE.search(path.read_text(encoding="utf-8")):
            stale.append(f"{rel} (no longer imports sqlite3)")
    assert not stale, f"remove stale SQLITE_ALLOWED entries: {stale}"


def test_retired_shared_state_modules_are_clean() -> None:
    """The two modules companion §2.7 named must stay off local state.

    `lightning.py` kept deposits in SQLite and `slurm_adapter.py` kept the
    job mapping in a JSON file; both are financial or control-plane facts
    shared between processes. Migration 060 moved them and Track B B9.3
    finished the contract — this asserts they do not drift back.
    """
    for name in ("lightning.py", "slurm_adapter.py"):
        text = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        assert not _SQLITE_IMPORT_RE.search(text), (
            f"{name} imports sqlite3 again; deposit and mapping state is "
            f"owned by PostgreSQL"
        )
        for dead in ("LN_DB_PATH", "SLURM_MAP_FILE", "slurm_jobs.json"):
            assert dead not in text, (
                f"{name} reintroduced file-backed state authority ({dead})"
            )


def test_worker_journal_is_documented_as_non_authority() -> None:
    """§10.2's one allowed worker-local store must say what it is not.

    The journal decides whether a running container is re-adopted after a
    restart. Someone reading it must not mistake it for a place that can
    grant a lease or extend a fence — those are PostgreSQL rows, and
    adoption still requires the API to confirm the attempt is current.
    """
    text = (PROJECT_ROOT / "worker_agent.py").read_text(encoding="utf-8")
    assert "NOT CONTROL-PLANE AUTHORITY" in text, (
        "the worker journal must be explicitly documented as node-local "
        "recovery state (companion §10.2)"
    )
    assert "_V2_JOURNAL_PATH" in text


def test_local_tooling_is_not_deployed() -> None:
    """Keeps the `scripts/local/` SQLite exemption honest.

    Those scripts may use SQLite because they run on a workstation. The
    moment one is copied into an image or wired into compose, that reason
    evaporates and it becomes shared state on a shared host.
    """
    referenced = []
    for artifact in ("docker-compose.yml", "Dockerfile", "mcp/Dockerfile"):
        path = PROJECT_ROOT / artifact
        if not path.is_file():
            continue
        if LOCAL_TOOLING_PREFIX in path.read_text(encoding="utf-8"):
            referenced.append(artifact)
    assert not referenced, (
        f"{LOCAL_TOOLING_PREFIX} is referenced by {referenced}. It is exempt "
        "from the SQLite boundary only because it runs on a developer "
        "workstation; deployed code must use PostgreSQL."
    )


def test_production_refuses_a_non_postgres_backend() -> None:
    """What makes the dev/test SQLite branches unreachable in production.

    `bitcoin.py` and `db.py` both branch on XCELSIOR_DB_BACKEND. They are
    on the allowed list only because the startup validator rejects a
    SQLite or dual backend in production, so this gate asserts that check
    still exists rather than trusting the comment.
    """
    text = (PROJECT_ROOT / "control_plane" / "startup_validation.py").read_text(
        encoding="utf-8"
    )
    assert "XCELSIOR_DB_BACKEND=postgres" in text, (
        "the production startup validator no longer names the required "
        "backend; the SQLite exceptions in SQLITE_ALLOWED depend on it"
    )
