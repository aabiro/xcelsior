#!/usr/bin/env python3
"""Render the repo's nginx vhosts for the host they are about to be installed on.

The repo keeps one copy of each vhost, written for the SNI-passthrough ingress:
TLS listeners on 8444 with `proxy_protocol`, behind `stream-tls-router.conf`.
Production does not run that router yet, so `deploy.sh` used to hold every
nginx install, and production's vhosts drifted from the repo with each release:
a connection cap that refused ordinary page loads, an error page pointing at a
missing file, and a headscale certificate the repo had moved off weeks before.

So the deploy renders instead of holding:

* `--topology router`: the files as written.
* `--topology direct`: TLS listeners on 443 without `proxy_protocol`, and the
  server-level `set_real_ip_from 127.0.0.1 / ::1` + `real_ip_header
  proxy_protocol` lines dropped. Those read the client address from the
  router's PROXY header, and with no router there is none; the http-level
  Cloudflare real-IP config does that job instead.

Either way, the API and MCP upstreams are written for the colour that is live
now. The blue-green swaps edit the installed file in place, so a fresh copy of
the repo's "9500 primary" over a host serving from 9501 would route every
request to the drained replica.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

VHOSTS = ("xcelsior", "headscale", "headscale-http", "docs-xcelsior", "downloads-xcelsior")

#: upstream name -> (port when green is live, port when blue is live)
UPSTREAMS = {
    "xcelsior_api": (9500, 9501),
    "xcelsior_mcp": (8770, 8771),
}

_ROUTER_LISTEN = re.compile(r"^(\s*listen\s+[^;]*?)\b8444\b([^;]*?)\s+proxy_protocol\b([^;]*;.*)$")
_ROUTER_REAL_IP = re.compile(
    r"^\s*(set_real_ip_from\s+(127\.0\.0\.1|::1)|real_ip_header\s+proxy_protocol)\s*;\s*(#.*)?$"
)


def to_direct(text: str) -> str:
    out = []
    for line in text.splitlines():
        if _ROUTER_REAL_IP.match(line):
            continue
        m = _ROUTER_LISTEN.match(line)
        if m:
            line = f"{m.group(1)}443{m.group(2)}{m.group(3)}"
        out.append(line)
    rendered = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    for line in rendered.splitlines():
        stripped = line.split("#", 1)[0]
        if re.match(r"^\s*listen\b", stripped) and "proxy_protocol" in stripped:
            raise ValueError(f"a listener still expects PROXY protocol: {line.strip()}")
    return rendered


def set_live(text: str, upstream: str, live: int, standby: int) -> str:
    """Within `upstream <name> { }`, make `live` the primary and `standby` the backup."""
    block = re.compile(rf"(upstream\s+{re.escape(upstream)}\s*\{{)(.*?)(\}})", re.S)
    m = block.search(text)
    if not m:
        return text
    body = m.group(2)
    for port, backup in ((live, False), (standby, True)):
        pattern = re.compile(rf"(^\s*server\s+127\.0\.0\.1:{port})(\s+backup)?\s*;", re.M)
        if not pattern.search(body):
            raise ValueError(f"upstream {upstream} has no server for port {port}")
        body = pattern.sub(lambda mm: mm.group(1) + (" backup;" if backup else ";"), body)
    primaries = re.findall(r"^\s*server\s+[^;]*;", body, re.M)
    if sum(1 for s in primaries if "backup" not in s) != 1:
        raise ValueError(f"upstream {upstream} must have exactly one primary server")
    return text[: m.start(2)] + body + text[m.end(2):]


def render(text: str, *, topology: str, api_live: str, mcp_live: str) -> str:
    if topology == "direct":
        text = to_direct(text)
    for upstream, colour in (("xcelsior_api", api_live), ("xcelsior_mcp", mcp_live)):
        green, blue = UPSTREAMS[upstream]
        live, standby = (green, blue) if colour == "green" else (blue, green)
        text = set_live(text, upstream, live, standby)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("dst", type=pathlib.Path)
    ap.add_argument("--topology", choices=("direct", "router"), required=True)
    ap.add_argument("--api-live", choices=("green", "blue"), required=True)
    ap.add_argument("--mcp-live", choices=("green", "blue"), required=True)
    args = ap.parse_args(argv)
    args.dst.mkdir(parents=True, exist_ok=True)
    for name in VHOSTS:
        src = args.src / f"{name}.conf"
        if not src.is_file():
            print(f"missing {src}", file=sys.stderr)
            return 1
        out = render(src.read_text(), topology=args.topology, api_live=args.api_live, mcp_live=args.mcp_live)
        (args.dst / f"{name}.conf").write_text(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
