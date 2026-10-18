#!/usr/bin/env python3
"""Read-only host security collector; Prometheus text is written to stdout.

Use --ingress on the gateway and --egress on workers. Scheduling and atomic
textfile publication are deployment steps; this command never repairs state.
"""

import argparse
from pathlib import Path
import re
import sys
import time

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    sys.path.insert(0, str(_p))
from scripts.check_ingress_policy import public_servers_protected


def ingress_ready(root: Path) -> bool:
    try:
        snippet = root / "snippets/xcelsior-abuse-deny.conf"
        policy = root / "conf.d/xcelsior-abuse-ips.conf"
        active_snippet = "\n".join(
            line.split("#", 1)[0] for line in snippet.read_text().splitlines()
        )
        active_policy = "\n".join(line.split("#", 1)[0] for line in policy.read_text().splitlines())
        if not re.search(
            r"if\s*\(\s*\$xcelsior_abuse_ip\s*\)\s*\{\s*return\s+(403|444)\s*;\s*\}", active_snippet
        ):
            return False
        if not re.search(r"(?:geo|map)\s+[^{};]*\$xcelsior_abuse_ip\s*\{", active_policy):
            return False
        # Includes in sites-available alone do not protect a disabled vhost.
        for name in ("xcelsior", "mcp-xcelsior", "agent-xcelsior"):
            if not public_servers_protected((root / "sites-enabled" / name).read_text()):
                return False
        return True
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingress", action="store_true")
    parser.add_argument("--egress", action="store_true")
    parser.add_argument("--nginx-root", type=Path, default=Path("/etc/nginx"))
    args = parser.parse_args()
    if not (args.ingress or args.egress):
        parser.error("select --ingress and/or --egress")
    values = {}
    if args.ingress:
        values["ingress_protected"] = int(ingress_ready(args.nginx_root))
    if args.egress:
        import egress_isolation

        try:
            values["egress_fresh"] = int(egress_isolation.isolation_ready())
        except (OSError, ValueError, TypeError, KeyError):
            values["egress_fresh"] = 0
    values["collection_timestamp_seconds"] = time.time()
    for key, value in values.items():
        print(f"# TYPE xcelsior_host_security_{key} gauge")
        print(f"xcelsior_host_security_{key} {value}")
    return (
        0
        if all(value for key, value in values.items() if key != "collection_timestamp_seconds")
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
