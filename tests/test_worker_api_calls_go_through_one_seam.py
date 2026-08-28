"""Every worker→API request goes through `_api_request`, or the cert is optional.

The private agent gateway (`nginx/agent-xcelsior.conf`) verifies a **client
certificate** and derives the worker's `host_id` from its CN. A request without
one is answered 400 by nginx before it reaches the API, which reads as a network
fault rather than a missing credential.

So the certificate is not a per-call detail — it is a property of "this is a
call to our API". `worker_agent._api_request` owns it, along with the choice to
send `cert` **only when one is configured**, so a worker still on a public
ingress calls `requests` with exactly the arguments it always did.

## Why a guard rather than a convention

The first attempt added `cert=` to all 31 call sites. That is the shape this
repository keeps finding and paying for: nothing stops the 32nd site omitting
it, and the omission is invisible — a worker that silently cannot authenticate
looks like a flaky network.

It also broke 52 tests at once, which is the more interesting failure. The
worker's test fakes are declared `(url, json=None, headers=None, timeout=None)`;
an unexpected keyword raises `TypeError`, and the worker's `except` swallows it
into a **silent no-op**. A credential whose absence makes a call quietly do
nothing is the worst available failure mode, and the mechanical fix produced it
31 times.

A `requests.Session` carrying the cert was the second attempt. It broke the same
tests a different way: `mock.patch.object(worker_agent.requests, "post", …)`
cannot see a call made through a session object. Hence
`getattr(requests, method)` inside the seam — one indirection, still patchable.

## What is deliberately *not* covered

Calls to things that are not our API: agent-upgrade downloads from arbitrary
URLs, the tower serverless admin endpoints, and artifact fetches. Sending a
fleet client certificate to an arbitrary host would leak this host's identity to
whoever it downloads from.
"""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKER = ROOT / "worker_agent.py"

#: Direct `requests.*` calls that are correct: none of them target our API.
#: Keyed by the URL expression as written, so a *new* direct call has to be
#: added here deliberately rather than inheriting someone else's exemption.
#: Written exactly as `ast.unparse` renders them — it normalises quoting, so
#: hand-typed doubles never match and every entry reads as stale.
ALLOWED_DIRECT = {
    "url",  # agent upgrade download + serverless health probe: arbitrary URLs
    "f'{TOWER_SERVERLESS_URL}/admin/drain'",
    "f'{TOWER_SERVERLESS_URL}/admin/resume'",
    "str(f.get('url') or '')",  # artifact fetch
}

_VERBS = {"get", "post", "put", "patch", "delete"}


def _direct_requests_calls() -> list[tuple[int, str]]:
    """`(lineno, url-expression)` for every bare `requests.<verb>(...)`."""
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr in _VERBS):
            continue
        if not (isinstance(fn.value, ast.Name) and fn.value.id == "requests"):
            continue
        url = ast.unparse(node.args[0]) if node.args else "<no positional url>"
        found.append((node.lineno, url))
    return found


def test_the_seam_exists_and_only_sends_a_cert_when_configured():
    """Calibration, and the property that keeps public-ingress workers working."""
    src = WORKER.read_text(encoding="utf-8")
    assert "def _api_request(" in src, "the transport seam is gone"
    assert "def _api_client_cert(" in src
    # Conditional, not unconditional: `cert=None` is harmless to `requests` but
    # an unconditional kwarg changes the call signature every test fake pins.
    assert "if cert is not None:" in src, (
        "`_api_request` must add `cert` only when one is configured — passing it "
        "always changes the call signature the worker's test fakes declare, and "
        "a TypeError there is swallowed into a silent no-op"
    )
    assert "getattr(requests, method)" in src, (
        "the seam must dispatch through `requests` by name so "
        "`mock.patch.object(worker_agent.requests, ...)` still intercepts; a "
        "Session hides the call from those mocks"
    )


def test_the_seam_is_actually_used():
    """A seam nothing calls is decoration."""
    src = WORKER.read_text(encoding="utf-8")
    assert src.count("_api_request(") > 20, (
        f"only {src.count('_api_request(')} uses of the seam; the API calls "
        "have drifted back to direct `requests.*`"
    )


def test_no_api_call_bypasses_the_seam():
    offenders = [(ln, url) for ln, url in _direct_requests_calls() if url not in ALLOWED_DIRECT]
    assert not offenders, (
        "these `requests.*` calls bypass `_api_request`, so they send no client "
        "certificate and the agent gateway answers 400 before the API sees them:\n  "
        + "\n  ".join(f"worker_agent.py:{ln}  {url}" for ln, url in offenders)
        + "\nUse `_api_request(...)`, or add the URL to ALLOWED_DIRECT with the "
        "reason it is not our API."
    )


def test_the_exemptions_are_still_real():
    """An exemption for a call that no longer exists hides the next one."""
    live = {url for _, url in _direct_requests_calls()}
    stale = ALLOWED_DIRECT - live
    assert not stale, (
        f"ALLOWED_DIRECT lists calls that no longer exist: {sorted(stale)}. "
        "Remove them, or the list stops describing the code."
    )
