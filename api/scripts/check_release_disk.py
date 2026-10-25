#!/usr/bin/env python3
"""Refuse release writes that would consume shared filesystem headroom."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

GIB = 1024**3
MINIMUM_RESERVE = 5 * GIB
DEFAULT_RELEASE_BUDGET = 8 * GIB


def existing_parent(path: Path) -> Path:
    path = path.absolute()
    while not path.exists():
        if path == path.parent:
            raise OSError("no existing filesystem ancestor")
        path = path.parent
    return path


def check_headroom(paths: list[Path], required_bytes: int) -> dict:
    if type(required_bytes) is not int or required_bytes < 0:
        raise ValueError("release budget must be a nonnegative integer")
    filesystems: dict[int, dict] = {}
    for path in paths:
        parent = existing_parent(path)
        device = parent.stat().st_dev
        if device in filesystems:
            filesystems[device]["paths"].append(str(path))
            continue
        space = os.statvfs(parent)
        total = space.f_blocks * space.f_frsize
        # Root-reserved blocks are unavailable to application/Redis users.
        available = space.f_bavail * space.f_frsize
        if total <= 0 or available < 0:
            raise ValueError("filesystem returned invalid capacity")
        reserve = max(MINIMUM_RESERVE, (total + 9) // 10)
        filesystems[device] = {
            "paths": [str(path)],
            "total_bytes": total,
            "available_bytes": available,
            "reserved_bytes": reserve,
            "release_budget_bytes": required_bytes,
            "passed": available >= reserve + required_bytes,
        }
    if not filesystems:
        raise ValueError("at least one release path is required")
    return {
        "passed": all(row["passed"] for row in filesystems.values()),
        "filesystems": list(filesystems.values()),
        "read_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", type=Path)
    parser.add_argument("--required-bytes", type=int, default=DEFAULT_RELEASE_BUDGET)
    args = parser.parse_args(argv)
    try:
        result = check_headroom(
            args.path or [Path("/opt/xcelsior"), Path("/var/lib/docker"), Path("/tmp")],
            args.required_bytes,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"passed": False, "probe_error": str(exc), "read_only": True}))
        return 1
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
