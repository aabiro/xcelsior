#!/usr/bin/env python3
"""Re-open encrypted LUKS volumes after NFS host reboot (VPS or legacy Mac).

After Ganesha/container restart, per-volume LUKS devices must be luksOpen'd
again before workers can mount exports. Safe to re-run (idempotent when
already open).

Usage (on VPS):
  docker compose exec -T api-blue python scripts/volumes_reopen_luks.py
  docker compose exec -T api-blue python scripts/volumes_reopen_luks.py --volume-id vol-abc123

Usage (local):
  python3 scripts/volumes_reopen_luks.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def _list_encrypted_volumes(volume_id: str | None) -> list[dict]:
    from db import _get_pg_pool
    from psycopg.rows import dict_row

    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        if volume_id:
            rows = conn.execute(
                """SELECT volume_id, name, status, encrypted
                   FROM volumes
                   WHERE volume_id = %s AND encrypted = true AND status != 'deleted'""",
                (volume_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT volume_id, name, status, encrypted
                   FROM volumes
                   WHERE encrypted = true AND status IN ('available', 'attached', 'error')""",
            ).fetchall()
    return [dict(r) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-open encrypted LUKS volumes on NFS host")
    parser.add_argument("--volume-id", help="Single volume to reopen (default: all encrypted)")
    parser.add_argument("--dry-run", action="store_true", help="List targets only; do not reopen")
    args = parser.parse_args()

    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))

    from volumes import NFS_SERVER, get_volume_engine

    if not NFS_SERVER:
        print(json.dumps({"error": "XCELSIOR_NFS_SERVER not configured", "pass": False}))
        return 1

    volumes = _list_encrypted_volumes(args.volume_id)
    if args.volume_id and not volumes:
        print(json.dumps({"error": f"Encrypted volume not found: {args.volume_id}", "pass": False}))
        return 1

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "count": len(volumes),
                    "volumes": volumes,
                    "pass": True,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    ve = get_volume_engine()
    results: list[dict] = []
    ok_count = 0
    for vol in volumes:
        vid = vol["volume_id"]
        try:
            ok = ve.reopen_luks_volume(vid)
            results.append({"volume_id": vid, "name": vol.get("name"), "ok": ok})
            if ok:
                ok_count += 1
        except Exception as e:
            results.append({"volume_id": vid, "name": vol.get("name"), "ok": False, "error": str(e)})

    report = {
        "total": len(volumes),
        "reopened": ok_count,
        "failed": len(volumes) - ok_count,
        "results": results,
        "pass": ok_count == len(volumes),
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())