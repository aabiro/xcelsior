"""SECURITY.md promises what the website's security page promises.

Two places tell a researcher how to report a vulnerability: `SECURITY.md`, which
GitHub shows on the repository's Security tab, and the "Reporting a
vulnerability" section of xcelsior.ca/security, whose English copy is in
`api/frontend/src/lib/i18n/en.ts`. If one changed the address or the response
window and the other did not, a researcher would be told two different things
by the same company. The website is the source: SECURITY.md must quote it.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "SECURITY.md"
EN = ROOT / "api" / "frontend" / "src" / "lib" / "i18n" / "en.ts"


def _en_string(key: str) -> str:
    match = re.search(
        rf'^\s*"{re.escape(key)}":\s*"((?:[^"\\]|\\.)*)",?\s*$',
        EN.read_text(encoding="utf-8"),
        re.M,
    )
    assert match, f"{key} is no longer in en.ts; update this test with the new key"
    return match.group(1)


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_security_md_gives_the_reporting_address_the_website_gives():
    address = _en_string("security.s7_email")
    assert address in POLICY.read_text(encoding="utf-8")


def test_security_md_quotes_the_websites_response_promise_and_safe_harbour():
    promise = _en_string("security.s7_p1")
    # The website's paragraph opens with the same sentence SECURITY.md opens
    # with; the commitments are the sentences after it.
    commitments = promise.split(". ", 1)[1]
    assert commitments in _flat(POLICY.read_text(encoding="utf-8"))
