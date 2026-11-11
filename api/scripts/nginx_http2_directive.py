#!/usr/bin/env python3
"""Rewrite nginx's deprecated `listen ... http2` into `listen ...` plus `http2 on;`.

For vhosts this repository does not render (other projects on the same host).
Prints what it would change; writes only with --write, keeping `<file>.bak-http2`.
Run `nginx -t` afterwards and reload only if it passes.

    nginx_http2_directive.py /etc/nginx/sites-enabled/grafana-xcelsior            # dry run
    nginx_http2_directive.py --write /etc/nginx/sites-enabled/grafana-xcelsior
"""

from __future__ import annotations

import os
import re
import shutil
import sys

_LISTEN_HTTP2 = re.compile(r"^(\s*)listen\s+(.*?)\s+http2\b(.*;.*)$")


def convert(text: str) -> tuple[str, int]:
    """The converted config and how many listen lines changed."""
    lines = text.split("\n")
    out: list[str] = []
    changed = 0
    i = 0
    while i < len(lines):
        match = _LISTEN_HTTP2.match(lines[i])
        if not match:
            out.append(lines[i])
            i += 1
            continue
        indent = match.group(1)
        while i < len(lines) and (m := _LISTEN_HTTP2.match(lines[i])):
            out.append(f"{m.group(1)}listen {m.group(2)}{m.group(3)}")
            changed += 1
            i += 1
        out.append(f"{indent}http2 on;")
    return "\n".join(out), changed


def main(argv: list[str]) -> int:
    write = "--write" in argv
    paths = [a for a in argv if a != "--write"]
    if not paths:
        print(__doc__)
        return 2
    for path in paths:
        real = os.path.realpath(path)
        with open(real) as fh:
            before = fh.read()
        after, changed = convert(before)
        print(f"{path}: {changed} listen line(s)" + (" rewritten" if write and changed else ""))
        if write and changed:
            shutil.copy2(real, real + ".bak-http2")
            with open(real, "w") as fh:
                fh.write(after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
