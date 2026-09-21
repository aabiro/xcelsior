"""No source file may contain a stray control character.

Six had crept in, in three files, all where a dash or an arrow belonged:

    routes/inference.py   "Insufficient wallet balance \\x192 please deposit credits"
    routes/sla.py         "- 95\\x1999% uptime \\x192 10% credit"   (×3)
    routes/compliance.py  "GST/HST registration REQUIRED \\x14 revenue exceeds …"

Two of them are strings the API returns to callers — a 402 body and a
GST-registration message — so the corruption was being served, not merely
stored. Nothing catches this: the files import, the tests pass, the bytes are
valid UTF-8, and a reviewer reading a diff in a terminal sees a dash-ish smudge
or nothing at all. In `routes/inference.py` the same sentence appears three
times and only one copy was damaged, which is exactly how long it can survive.

Tabs, newlines and carriage returns are ordinary source characters and are
allowed. Everything else in the C0 range, and the C1 range that a mis-decoded
Windows-1252 round-trip leaves behind, is a finding.
"""

from __future__ import annotations

import re

from tests._source_tree import iter_source_files, read_source

#: C0 minus tab/newline/carriage-return, plus DEL and the C1 block.
FORBIDDEN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

SUFFIXES = ("*.py", "*.yml", "*.yaml", "*.toml", "*.md")


def test_no_source_file_contains_a_control_character() -> None:
    offenders: list[str] = []
    for suffix in SUFFIXES:
        for path, rel in iter_source_files(suffix, include_tests=True):
            try:
                text = read_source(path)
            except UnicodeDecodeError:
                offenders.append(f"{rel}: is not valid UTF-8")
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                found = FORBIDDEN.findall(line)
                if found:
                    codes = ", ".join(f"U+{ord(c):04X}" for c in dict.fromkeys(found))
                    offenders.append(f"{rel}:{lineno}: {codes} in {line.strip()[:90]!r}")

    assert not offenders, (
        "control characters in source — almost always a dash, quote or arrow "
        "that lost its encoding somewhere. Where the text is returned to a "
        "caller it is being served corrupted:\n  " + "\n  ".join(offenders)
    )
