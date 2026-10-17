#!/usr/bin/env python3
"""Check every public Nginx server for its server-level abuse include.

Read-only; shared by deployment preflight and the security status collector.
This validates the containment include, not general Nginx syntax (nginx -t).
"""

from pathlib import Path
import shlex
import sys


def public_servers_protected(text: str) -> bool:
    lexer = shlex.shlex(text, posix=True, punctuation_chars="{};")
    lexer.whitespace_split = True
    # The configuration's punctuation can coalesce ('};'). Expand before
    # recursion so closing a child never discards its parent's delimiter.
    tokens = iter(
        part
        for token in lexer
        for part in (list(token) if token and all(c in "{};" for c in token) else [token])
    )
    servers = []

    def block(nested=False):
        directives, words = [], []
        for token in tokens:
            if token == "{":
                server = words == ["server"]
                child = block(True)
                if server:
                    servers.append(child)
                words = []
            elif token == ";":
                directives.append(words)
                words = []
            elif token == "}":
                if not nested:
                    raise ValueError("unmatched brace")
                return directives
            else:
                words.append(token)
        if nested:
            raise ValueError("unclosed block")
        return directives

    try:
        block()
    except ValueError:
        return False
    public = 0
    for directives in servers:
        listens = [d[1] for d in directives if len(d) > 1 and d[0] == "listen"]
        if listens and all(s.startswith(("127.", "[::1]:", "unix:")) for s in listens):
            continue
        public += 1
        allowed = {
            "/etc/nginx/snippets/xcelsior-abuse-deny*.conf",
            "/etc/nginx/snippets/xcelsior-abuse-deny.conf",
        }
        if not any(len(d) == 2 and d[0] == "include" and d[1] in allowed for d in directives):
            return False
    return public > 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("provide at least one vhost file")
    for filename in sys.argv[1:]:
        if not public_servers_protected(Path(filename).read_text()):
            raise SystemExit("public server omits its abuse policy or configuration is malformed")
