#!/usr/bin/env python3
"""Render NFS export lines for the admitted provider hosts only (CVP-01 T-01 step 3).

Both storage setups exported the volume tree to the whole mesh subnet, so any
machine on the tailnet, admitted or not, could mount every tenant's volume.
This prints one export per admitted host address instead. It changes nothing
by itself: the output is reviewed and installed on the storage host by hand.

    render_nfs_exports.py --format linux --base /exports/volumes 100.64.0.6 100.64.0.4
    render_nfs_exports.py --format mac   --base /Volumes/InferenceData 100.64.0.6 100.64.0.4

Squash options are left exactly as each setup script has them; narrowing who
may mount and changing whose files they become are separate changes.
"""

from __future__ import annotations

import argparse
import ipaddress
import sys

MESH = ipaddress.ip_network("100.64.0.0/10")
LINUX_OPTS = "rw,sync,no_subtree_check,no_root_squash"
MAC_OPTS = "-mapall=501:20 -alldirs"


def admitted(addresses: list[str]) -> list[str]:
    """Sorted, de-duplicated host addresses; refuses anything that is not one mesh host."""
    out = set()
    for raw in addresses:
        text = raw.strip()
        if not text:
            continue
        try:
            ip = ipaddress.ip_address(text)
        except ValueError:
            raise ValueError(f"not a host address: {raw!r}") from None
        if ip not in MESH:
            raise ValueError(f"{ip} is outside the mesh {MESH}")
        out.add(ip)
    if not out:
        raise ValueError("no admitted hosts given; refusing to render an empty export list")
    return [str(ip) for ip in sorted(out)]


def render(fmt: str, base: str, addresses: list[str]) -> str:
    if not base.startswith("/") or any(c.isspace() for c in base):
        raise ValueError(f"export base must be an absolute path without spaces: {base!r}")
    hosts = admitted(addresses)
    if fmt == "linux":
        # exports(5): one path, then each client with its own options.
        return base + " " + " ".join(f"{h}({LINUX_OPTS})" for h in hosts) + "\n"
    if fmt == "mac":
        # BSD exports(5): hosts are listed after the options on one line.
        return f"{base} {MAC_OPTS} " + " ".join(hosts) + "\n"
    raise ValueError(f"unknown format {fmt!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--format", choices=("linux", "mac"), required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("hosts", nargs="+")
    args = parser.parse_args(argv)
    try:
        sys.stdout.write(render(args.format, args.base, args.hosts))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
