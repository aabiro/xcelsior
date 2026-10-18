#!/usr/bin/env python3
"""Regenerate the privacy-policy translation lock after syncing French to English.

    python3 scripts/sync_privacy_translations.py          # check, exit 1 if stale
    python3 scripts/sync_privacy_translations.py --write  # after updating French

## Why a lock file rather than a derivation

Everything else guarding this document is derived from two live sources, because
a hand-kept copy of a fact drifts. This is not a copy of a fact. It records
**when the two languages were last reconciled**, and that is not recoverable
from the files themselves: nothing in `en-public.ts` or `fr-public.ts` says
whether the French text was written against the English text currently sitting
next to it, or against a sentence that has since been rewritten.

That is exactly what went wrong. English `s6_p3` was replaced with "compute may
run on independent hosts in any country" and French kept promising that a
province's residents' data "restent au Canada en tout temps". Both files were
valid, both had the same keys, and every check that existed passed. The missing
fact was the reconciliation, so the lock records the reconciliation.

Its staleness is the signal, in the way a dependency lockfile's is: a stale
entry is not a maintenance chore, it is the report that an English sentence
changed and the French one did not follow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
I18N = ROOT / "frontend" / "src" / "lib" / "i18n"
LOCK = I18N / "privacy-translation.lock.json"

KEY = re.compile(r'"(privacy\.[a-z0-9_]+)":\s*"((?:[^"\\]|\\.)*)"')


def strings(name: str) -> dict[str, str]:
    return dict(KEY.findall((I18N / f"{name}.ts").read_text(encoding="utf-8")))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def current() -> dict[str, str]:
    """`{key: digest of the English source}` for every privacy string."""
    return {key: digest(value) for key, value in sorted(strings("en-public").items())}


def stored() -> dict[str, str]:
    if not LOCK.exists():
        return {}
    return json.loads(LOCK.read_text(encoding="utf-8")).get("english_digests", {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write",
        action="store_true",
        help="record the current English as reconciled — only after French is updated",
    )
    args = ap.parse_args()

    now, was = current(), stored()
    if args.write:
        LOCK.write_text(
            json.dumps(
                {
                    "_comment": (
                        "Digests of the ENGLISH privacy strings as of the last time "
                        "French was reconciled to them. A mismatch means an English "
                        "sentence changed and French did not follow. Update "
                        "fr-public.ts and fr.ts, then rerun with --write."
                    ),
                    "english_digests": now,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"recorded {len(now)} reconciled strings -> {LOCK.relative_to(ROOT)}")
        return 0

    changed = sorted(k for k in now if k in was and was[k] != now[k])
    added = sorted(set(now) - set(was))
    removed = sorted(set(was) - set(now))
    if changed or added or removed:
        for k in changed:
            print(f"changed in English, French not reconciled: {k}")
        for k in added:
            print(f"new English string, no French reconciliation: {k}")
        for k in removed:
            print(f"gone from English, still locked: {k}")
        return 1
    print(f"{len(now)} strings reconciled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
