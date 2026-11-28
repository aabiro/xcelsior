"""Publish a timer failure even if startup failed before the script ran."""

import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import time


def main() -> int:
    result = os.environ.get("SERVICE_RESULT")
    if not result or result == "success":
        return 0
    operation = sys.argv[1]
    if operation not in {"backup", "restore"}:
        return 1
    metrics = Path(
        os.environ.get(
            "XCELSIOR_NODE_EXPORTER_TEXTFILE_DIR", "/var/lib/node_exporter/textfile_collector"
        )
    )
    metrics.mkdir(parents=True, exist_ok=True)
    filename = f"xcelsior_{operation}_failure.prom"
    name = f"xcelsior_{operation}_last_failure_timestamp_seconds"
    with NamedTemporaryFile(mode="w", dir=metrics, delete=False) as output:
        output.write(f"# TYPE {name} gauge\n{name} {int(time.time())}\n")
        os.fchmod(output.fileno(), 0o644)
        temporary = Path(output.name)
    temporary.replace(metrics / filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
