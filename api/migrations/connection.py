"""Migration credentials stay separate from long-running service credentials."""

import os

from control_plane.db_roles import roles_required


def migration_dsn(fallback: str | None = None) -> str | None:
    dedicated = os.environ.get("XCELSIOR_POSTGRES_DSN_MIGRATOR")
    if dedicated:
        return dedicated
    if roles_required():
        raise RuntimeError(
            "XCELSIOR_POSTGRES_DSN_MIGRATOR is required; shared migration fallback refused"
        )
    return (
        os.environ.get("XCELSIOR_POSTGRES_DSN")
        or os.environ.get("XCELSIOR_PG_DSN")
        or os.environ.get("DATABASE_URL")
        or fallback
    )


def require_migration_identity(current: str, login: str) -> None:
    if roles_required() or os.environ.get("XCELSIOR_POSTGRES_DSN_MIGRATOR"):
        if (current, login) != ("xcelsior_migrator", "xcelsior_migrator"):
            raise RuntimeError("migrations require a direct xcelsior_migrator login")
