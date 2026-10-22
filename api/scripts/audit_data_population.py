#!/usr/bin/env python3
"""Count missing values across all application tables without emitting records."""

import argparse
import json
from pathlib import Path
import sys

for directory in (
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[2] / "core",
    Path(__file__).resolve().parents[2],
):
    sys.path.insert(0, str(directory))

from control_plane.data_population import audit_population
from db import _get_pg_pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-database", required=True)
    parser.add_argument("--expected-port", required=True, type=int)
    args = parser.parse_args()
    with _get_pg_pool().connection() as conn:
        result = audit_population(
            conn, expected_database=args.expected_database, expected_port=args.expected_port
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
