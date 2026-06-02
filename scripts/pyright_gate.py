#!/usr/bin/env python3
"""CI gate: fail if pyright reports MORE wrong-call findings than the baseline.

``reportCallIssue`` catches the wrong-kwarg / wrong-arg-count / no-such-parameter
class of bug — e.g. passing ``customer_id=`` to a function whose parameter is
``user=``, or calling ``foo(a, b, c)`` when ``foo`` takes two arguments. A scan
in June 2026 found nine such bugs in route/CLI handlers (all silently broken).

The legacy backlog of these findings has been fully cleared (56 -> 0). The
bulk were pyright mis-inferring scheduler.py's JSONB host/job dicts as lists,
fixed by typing the _decode_payload boundary as Any; the rest were wrong
key-type / loose arg typing in a handful of handlers. So BASELINE is now 0
and this is a zero-tolerance gate: it fails if ANY reportCallIssue appears,
catching the whole class of bug on a PR. Keep it at 0 — only ratchet DOWN.

Run locally:  python scripts/pyright_gate.py
"""

import json
import subprocess
import sys

# reportCallIssue count: 0 as of 2026-06-02 (pyright 1.1.410, deps installed).
# The full backlog (56 -> 0) has been cleared, so this is a zero-tolerance
# gate. Keep it at 0; only ever ratchet DOWN.
BASELINE = 0


def main() -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pyright", "--outputjson"],
        capture_output=True,
        text=True,
    )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print("pyright did not return JSON. stderr:\n" + proc.stderr, file=sys.stderr)
        return 2

    call_issues = [
        d for d in data.get("generalDiagnostics", []) if d.get("rule") == "reportCallIssue"
    ]
    count = len(call_issues)
    print(f"reportCallIssue: {count} (baseline {BASELINE})")

    if count > BASELINE:
        print(f"\n[FAIL] {count - BASELINE} new wrong-call finding(s) — likely a bad kwarg/arg.")
        print("All current reportCallIssue locations:")
        for d in sorted(call_issues, key=lambda x: (x["file"], x["range"]["start"]["line"])):
            line = d["range"]["start"]["line"] + 1
            msg = d["message"].splitlines()[0]
            print(f"  {d['file']}:{line}: {msg}")
        return 1

    if count < BASELINE:
        print(
            f"\n[OK] {BASELINE - count} fewer than baseline — lower BASELINE in "
            f"scripts/pyright_gate.py to {count} to lock in the win."
        )
    else:
        print("[OK] no new wrong-call findings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
