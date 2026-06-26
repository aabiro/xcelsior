#!/usr/bin/env python3
"""Remove duplicate KEY= lines from an env file.

Rule (preserves current load-time behavior — python-dotenv keeps the last value):
- identical values  -> keep the FIRST occurrence (cleaner; stays in its section)
- differing values  -> keep the LAST occurrence (the value the loader currently uses)
"""
import collections
import re
import sys

KEYLINE = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")


def dedup(path: str) -> int:
    with open(path) as f:
        lines = f.readlines()
    occ: "collections.OrderedDict[str, list]" = collections.OrderedDict()
    for i, ln in enumerate(lines):
        m = KEYLINE.match(ln.rstrip("\n"))
        if m:
            occ.setdefault(m.group(1), []).append((i, m.group(2)))
    remove: set[int] = set()
    for _k, lst in occ.items():
        if len(lst) < 2:
            continue
        if len({v for _, v in lst}) == 1:
            remove.update(idx for idx, _ in lst[1:])   # keep first
        else:
            remove.update(idx for idx, _ in lst[:-1])  # keep last
    if not remove:
        return 0
    with open(path, "w") as f:
        f.writelines(ln for i, ln in enumerate(lines) if i not in remove)
    return len(remove)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{p}: removed {dedup(p)} duplicate line(s)")
