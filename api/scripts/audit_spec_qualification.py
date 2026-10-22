#!/usr/bin/env python3
"""Report spec qualification and column gaps without emitting workload records."""

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

from control_plane.spec_qualification import audit_spec_qualification
from db import _get_pg_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-database", required=True)
    parser.add_argument("--expected-port", type=int, required=True)
    parser.add_argument("--require-complete-specs", action="store_true")
    args = parser.parse_args()
    with _get_pg_pool().connection() as conn:
        result = audit_spec_qualification(
            conn, expected_database=args.expected_database, expected_port=args.expected_port
        )
    print(json.dumps(result, sort_keys=True))
    return 2 if args.require_complete_specs and not result["all_job_specs_qualified"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
