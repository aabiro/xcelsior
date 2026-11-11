"""Who may start a container from a snapshot image in the platform registry.

Snapshots are pushed to ``<XCELSIOR_REGISTRY_URL>/<owner slug>/<name>:<tag>``
and every worker pulls from that registry with one shared platform login. The
owner slug is derived from the owner's id (an email or user id), so a
reference is guessable by anyone who knows whose it is. ``template_image_id``
launches were already checked against ``user_images``; a launch that typed the
same reference straight into ``image`` was not, and the worker would have
pulled another tenant's filesystem for it (CVP-01 T-22 residual, test T13b).

The rule: a reference inside a namespace that holds someone's snapshots may be
launched by that owner, or by anyone when that exact reference is a ready,
public image. Namespaces with no snapshots in them (the platform's own images)
are not this module's business.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable

#: What the caller is told. Deliberately the same for "exists but is not yours"
#: and "does not exist", so the answer does not confirm another tenant's image.
REFUSAL = "Image not found, or it is a private snapshot that belongs to another account"

Row = tuple[str, bool, str, str]  # owner_id, is_public, image_ref, status


def _registry_prefix() -> str:
    reg = os.environ.get("XCELSIOR_REGISTRY_URL", "").strip().rstrip("/").lower()
    for scheme in ("https://", "http://"):
        if reg.startswith(scheme):
            reg = reg[len(scheme) :]
    return reg


def snapshot_namespace(image: str | None) -> str | None:
    """``<registry>/<namespace>/`` when ``image`` sits in a registry namespace, else None."""
    reg = _registry_prefix()
    ref = (image or "").strip()
    for scheme in ("https://", "http://"):
        if ref.lower().startswith(scheme):
            ref = ref[len(scheme) :]
    if not reg or not ref.lower().startswith(reg + "/"):
        return None
    rest = ref[len(reg) + 1 :]
    namespace, sep, remainder = rest.partition("/")
    if not sep or not namespace or not remainder:
        return None
    return f"{reg}/{namespace.lower()}/"


def _rows_in_namespace(namespace: str) -> list[Row]:
    from db import _get_pg_pool

    pattern = namespace.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    with _get_pg_pool().connection() as conn:
        rows = conn.execute(
            """SELECT owner_id, is_public, image_ref, status FROM user_images
               WHERE deleted_at = 0 AND lower(image_ref) LIKE %s ESCAPE '\\'""",
            (pattern,),
        ).fetchall()
    out: list[Row] = []
    for r in rows:
        if isinstance(r, dict):
            out.append(
                (str(r["owner_id"]), bool(r["is_public"]), str(r["image_ref"]), str(r["status"]))
            )
        else:
            out.append((str(r[0]), bool(r[1]), str(r[2]), str(r[3])))
    return out


def foreign_snapshot_problem(
    image: str | None,
    principals: Iterable[str],
    *,
    lookup: Callable[[str], list[Row]] | None = None,
) -> str | None:
    """None when ``image`` may be launched by ``principals``, else the refusal text.

    A database error propagates: a launch that cannot be checked is not started.
    """
    namespace = snapshot_namespace(image)
    if namespace is None:
        return None
    rows = (lookup or _rows_in_namespace)(namespace)
    if not rows:
        return None
    mine = {str(p).strip() for p in principals if p and str(p).strip()}
    if any(owner in mine for owner, _public, _ref, _status in rows):
        return None
    wanted = (image or "").strip()
    for _owner, is_public, ref, status in rows:
        if is_public and status == "ready" and ref == wanted:
            return None
    return REFUSAL
