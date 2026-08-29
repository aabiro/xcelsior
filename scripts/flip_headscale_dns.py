#!/usr/bin/env python3
"""Point `hs.xcelsior.ca` at the host that actually runs Headscale.

Step 3 of 3 in removing the 45.76.3.128 single point of failure. Steps 1 and 2
(issue a Let's Encrypt certificate on 149.28.121.61 by DNS-01, and repoint that
host's nginx at it) are already done and verified — this only moves the record.

Run it with no arguments to see the current record and what would change.
Pass --apply to make the change.

    python scripts/flip_headscale_dns.py            # show, change nothing
    python scripts/flip_headscale_dns.py --apply    # do it

Reads CLOUDFLARE_API_TOKEN / CLOUDFLARE_ZONE_ID from the repo `.env`; prints
neither. Reversible: re-run with `--target 45.76.3.128 --apply`.
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headscale_failover as hf  # noqa: E402

HEADSCALE_HOST = "149.28.121.61"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=HEADSCALE_HOST)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    settings = hf.settings_from_env(os.environ)
    if not (settings.cloudflare_token and settings.cloudflare_zone_id):
        print("CLOUDFLARE_API_TOKEN / CLOUDFLARE_ZONE_ID are not set", file=sys.stderr)
        return 2

    record = hf.dns_a_record(settings)
    current = record.get("content")
    print(f"  {settings.dns_name}: {current} (proxied={record.get('proxied')}, ttl={record.get('ttl')})")

    if current == args.target:
        print(f"  already {args.target} — nothing to do")
        return 0

    # Refuse to point the name at a box that cannot serve it. The whole failure
    # this removes was a name pointing somewhere that could not answer for it.
    #
    # `curl --resolve` rather than a hand-rolled SNI client: it pins the name to
    # one IP while keeping ordinary certificate validation, which is precisely
    # the question here — can *this* host prove *this* name to a normal client.
    if shutil.which("curl") is None:
        print("  curl is required for the pre-flight check")
        return 1
    probe = subprocess.run(
        [
            "curl", "-s", "--max-time", "15",
            "--resolve", f"{settings.dns_name}:443:{args.target}",
            f"https://{settings.dns_name}/health",
        ],
        capture_output=True,
        text=True,
    )
    body = (probe.stdout or "").strip()
    if probe.returncode != 0 or "pass" not in body.lower():
        print(f"  PRE-FLIGHT FAILED: {args.target} cannot serve {settings.dns_name}")
        print(f"    curl exit={probe.returncode} body={body[:120]!r} err={(probe.stderr or '')[:160]!r}")
        print("  refusing to move the record")
        return 1
    print(f"  pre-flight OK: {args.target} serves {settings.dns_name} with a trusted certificate")

    if not args.apply:
        print(f"  would change {current} -> {args.target}   (re-run with --apply)")
        return 0

    result = hf.set_dns_a(settings, args.target).get("result", {})
    print(f"  CHANGED: {result.get('name')} -> {result.get('content')} "
          f"(proxied={result.get('proxied')}, ttl={result.get('ttl')})")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
