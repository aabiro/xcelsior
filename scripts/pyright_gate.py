#!/usr/bin/env python3
"""CI gate: fail if pyright reports MORE wrong-call findings than the baseline.

``reportCallIssue`` catches the wrong-kwarg / wrong-arg-count / no-such-parameter
class of bug — e.g. passing ``customer_id=`` to a function whose parameter is
``user=``, or calling ``foo(a, b, c)`` when ``foo`` takes two arguments. A scan
in June 2026 found nine such bugs in route/CLI handlers (all silently broken).

The codebase still carries a backlog of these findings, most of them pyright
mis-inferring dict variables as lists in scheduler.py — not real bugs, but not
worth annotating away in a 150 KB hot-path file right now. So instead of
blocking on the legacy backlog, this gate *ratchets*: it fails only when the
count rises above BASELINE, catching any NEW bug of this class on a PR. Lower
BASELINE whenever the count drops to lock in the improvement.

Run locally:  python scripts/pyright_gate.py
"""

import json
import subprocess
import sys

# reportCallIssue count as of 2026-06-02 (pyright 1.1.410, deps installed).
# Only ever ratchet this DOWN.
BASELINE = 56


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
