"""An authorization assertion must not accept success as a pass.

`tests/test_health_endpoints_coverage.py::test_ssh_keygen_requires_auth` read:

    assert r.status_code in (401, 403, 200)

The endpoint being wide open and the endpoint being locked both satisfied it, so
the test could not fail. It sat in the coverage file for `/ssh/keygen` — one of
the four endpoints §0.1 names as authenticated but not authorized — and passed
throughout the period the defect existed.

The sharp class is narrow and worth stating precisely: a tuple containing a
success code **and** 401 or 403. Such an assertion claims something about
authorization while admitting the outcome that would prove it absent.

**Not swept up here:** tuples mixing success with 404, 503, 400 or 422. Those
are usually a legitimate tolerance — an endpoint that may be disabled, a feature
flag off in this environment, a resource that may not exist yet. There are 19 of
those, they are a weaker pattern, and treating them as the same defect would
inflate a real finding into a false one. If they should be tightened, that is a
separate piece of work with a separate argument.

The rule enforced: **zero tuples mix success with an authorization refusal.**
"""

from __future__ import annotations

import pathlib
import re

TESTS = pathlib.Path(__file__).resolve().parent

SUCCESS_CODES = {200, 201, 202, 204}
AUTHZ_REFUSALS = {401, 403}

_TUPLE = re.compile(r"status_code\s+in\s+\(([^)]*)\)")


def _mixed_authz_assertions() -> list[tuple[str, int, list[int], str]]:
    found = []
    for path in sorted(TESTS.glob("*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            match = _TUPLE.search(line)
            if not match:
                continue
            codes = {int(c) for c in re.findall(r"\b([1-5]\d\d)\b", match.group(1))}
            if codes & SUCCESS_CODES and codes & AUTHZ_REFUSALS:
                found.append((path.name, n, sorted(codes), line.strip()[:100]))
    return found


def test_no_authorization_assertion_accepts_success():
    """The load-bearing rule."""
    offenders = _mixed_authz_assertions()
    assert not offenders, (
        "these assertions accept both a success and an authorization refusal, "
        "so they pass whether the endpoint is guarded or wide open:\n"
        + "\n".join(f"  {f}:{n} {codes}\n      {line}" for f, n, codes, line in offenders)
    )


def test_the_scanner_detects_a_planted_assertion():
    """Prove the reach rather than trusting the silence.

    A regex that matches nothing reports clean, and clean is what a broken
    scanner looks like — the vocabulary guard reported zero while four blog
    posts were full of what it was hunting.
    """
    planted = "    assert r.status_code in (401, 403, 200)"
    match = _TUPLE.search(planted)
    assert match, "the tuple pattern no longer matches the original defect"
    codes = {int(c) for c in re.findall(r"\b([1-5]\d\d)\b", match.group(1))}
    assert codes & SUCCESS_CODES and codes & AUTHZ_REFUSALS


def test_the_scanner_does_not_flag_a_legitimate_tolerance():
    """`(200, 404)` for an endpoint that may be disabled is not this defect."""
    tolerant = "    assert r.status_code in (200, 404)"
    match = _TUPLE.search(tolerant)
    codes = {int(c) for c in re.findall(r"\b([1-5]\d\d)\b", match.group(1))}
    assert not (codes & AUTHZ_REFUSALS)


def test_the_scanner_does_not_flag_a_pure_refusal_tuple():
    """`(401, 403)` is correct — either refusal code is acceptable."""
    refusal = "    assert r.status_code in (401, 403)"
    match = _TUPLE.search(refusal)
    codes = {int(c) for c in re.findall(r"\b([1-5]\d\d)\b", match.group(1))}
    assert not (codes & SUCCESS_CODES)
