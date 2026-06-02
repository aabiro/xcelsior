#!/usr/bin/env python3
"""CI gate: fail on any wrong-call / wrong-argument finding pyright reports.

Two related rule families catch the bug class where a call is structurally
wrong — the kind that is silently broken at runtime:

  * reportCallIssue   — wrong kwarg name / wrong arg count / no-such-parameter
    (e.g. passing customer_id= when the parameter is user=).
  * reportArgumentType — a value of the wrong type passed to a parameter
    (e.g. None into a str param, float into int, or a list-typed record where
    a dict was meant — which is how the scheduler crash bugs surfaced).

Both backlogs have been cleared, so this is a zero-tolerance gate: it fails if
ANY tracked finding appears, catching this whole class of bug on a PR.

One sub-class is deliberately filtered out: psycopg types execute()'s query
parameter as LiteralString (to discourage SQL injection), so every intentional
dynamic-SQL f-string trips reportArgumentType with a "QueryNoTemplate" message.
Parameterising that ~two-dozen-site pattern is a separate initiative; until
then those findings are skipped by message match so they don't mask real
argument-type bugs.

Run locally:  python scripts/pyright_gate.py
"""

import json
import subprocess
import sys

# Rules enforced at zero tolerance. The tuple lists message substrings to
# ignore for that rule (known, accepted noise). Keep each list short and
# justified — every entry is a hole in the gate.
TRACKED_RULES: dict[str, tuple[str, ...]] = {
    "reportCallIssue": (),
    "reportArgumentType": ("QueryNoTemplate",),  # psycopg dynamic-SQL LiteralString noise
}


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

    offenders = []
    for d in data.get("generalDiagnostics", []):
        rule = d.get("rule")
        if rule not in TRACKED_RULES:
            continue
        if any(skip in d.get("message", "") for skip in TRACKED_RULES[rule]):
            continue
        offenders.append(d)

    print(f"tracked findings: {len(offenders)} (must be 0)")
    if offenders:
        print("\n[FAIL] wrong-call / wrong-argument finding(s) — likely a bad kwarg/arg/type:")
        for d in sorted(offenders, key=lambda x: (x["file"], x["range"]["start"]["line"])):
            line = d["range"]["start"]["line"] + 1
            msg = d["message"].splitlines()[0]
            print(f"  {d['file']}:{line}: [{d.get('rule')}] {msg}")
        return 1

    print("[OK] no wrong-call / wrong-argument findings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
