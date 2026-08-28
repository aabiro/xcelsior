"""A raw `fetch` to an API route silently opts out of session recovery.

`apiFetch` does three things a bare `fetch` does not: it sets the JSON content
type, and on a **401** it refreshes the session once (through a shared promise,
so concurrent calls do not stampede) and retries the original request, falling
back to a login redirect only if the refresh fails.

The reputation page fetched `/api/reputation/me` with a raw `fetch` and rejected
on any non-ok response. So an expired access token produced:

* leaderboard — `apiFetch`, refreshed, correct;
* history — `apiFetch`, refreshed, correct;
* the user's own score — rejected, `myRep` null, and
  `finalScore = myRep?.final_score ?? 0` rendering **"0 points"**.

A working page telling someone they have no reputation. The same silent-default
shape as `?? 0` on the marketplace stats and `|| {}` on the SLA table: nothing
throws, and the wrong answer is indistinguishable from a real one.

## Why this is a list rather than a ban

Streaming genuinely needs the raw `Response` — `useAiChat`, `useChatStream` and
the chat widget read `res.body` incrementally, which `apiFetch` cannot return
because it resolves `.json()`. Those are correct and stay.

So the allowlist below is the point of the test: every remaining raw call is
either recorded with a reason, or it is a call that will not recover a session.
The count is a **downward ratchet** — it may shrink, never grow.
"""

from __future__ import annotations

import pathlib
import re

from tests._source_tree import iter_source_files, read_source, strip_ts_comments

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Reads the response body incrementally; `apiFetch` resolves `.json()` and
#: cannot return a stream. Checked below against the code, not taken on trust.
STREAMING = {
    "frontend/src/hooks/useAiChat.ts",
    "frontend/src/hooks/useChatStream.ts",
    "frontend/src/components/ai/xcel-ai-onboarding.tsx",
    "frontend/src/app/(dashboard)/dashboard/analytics/analytics-ai-panel.tsx",
}

#: Branches on a specific HTTP status. `apiFetch` throws an `ApiError`, so the
#: status is still reachable — but only in a catch, which turns a two-branch
#: read into exception control flow for no gain.
#:
#: `ChatWidget` clears a stale conversation id from `localStorage` on a 404.
#: It was in `STREAMING` until the check was tightened, and it does not stream —
#: `useChatStream` is the file that does. It passed the old loose check because
#: a `className="site-legal-body"` elsewhere matched a substring of "body".
STATUS_BRANCHING = {
    "frontend/src/components/ChatWidget.tsx",
}

#: Public pages. `apiFetch` redirects to login on a failed refresh, which would
#: bounce a logged-out visitor off a status page — worse than the bug this file
#: is about. A different reason from streaming, so a different list: the first
#: draft put this in `STREAMING` and it **passed**, because `site-legal-body` in
#: a className satisfied a substring check for "body". A guard that passes for
#: the wrong reason is the thing this whole session kept finding.
PUBLIC_UNAUTHENTICATED = {
    "frontend/src/app/(marketing)/status/content.tsx",
}

EXEMPT = STREAMING | PUBLIC_UNAUTHENTICATED | STATUS_BRANCHING

#: Raw API calls that are not streaming and have not been converted yet.
#: Each one cannot recover an expired session. Shrink this; never extend it.
#:
#: Measured, not guessed. The first draft said 20 against an actual 17, and the
#: control that adds a raw fetch passed — a ratchet with slack is not a ratchet,
#: it is a number that happens to be true.
KNOWN_UNCONVERTED = 0


def _raw_api_fetches() -> dict[str, list[str]]:
    """`file -> [call lines]` for `fetch("/api…")` outside the api client."""
    found: dict[str, list[str]] = {}
    sources = [
        *iter_source_files("*.ts", include_prefixes=("frontend/src/",)),
        *iter_source_files("*.tsx", include_prefixes=("frontend/src/",)),
    ]
    assert sources, "no frontend sources found; the include prefix is wrong"
    for path, rel in sources:
        if rel == "frontend/src/lib/api.ts":
            continue  # `apiFetch` itself is the one place a raw fetch belongs
        text = strip_ts_comments(read_source(path))
        for line in text.splitlines():
            if re.search(r'\bfetch\(\s*[`"]/(api|v1)/', line):
                found.setdefault(rel, []).append(line.strip()[:100])
    return found


def test_the_scan_finds_raw_calls_at_all():
    """Calibration — an empty result would pass the ratchet trivially."""
    found = _raw_api_fetches()
    assert found, "no raw API fetches found; the pattern is wrong"


def test_the_streaming_exemptions_really_stream():
    """An exemption for a file that no longer streams is an exemption to drop.

    Matched on `res.body` / `getReader`, not a substring of "body" — a
    `className="site-legal-body"` satisfied that and let a non-streaming file
    sit in this list unnoticed.
    """
    for rel in sorted(STREAMING):
        path = ROOT / rel
        assert path.exists(), f"{rel} is gone; remove it from the exemption list"
        text = read_source(path)
        assert re.search(r"\.body\b|getReader\(", text), (
            f"{rel} is exempted as streaming but never reads a response body. "
            "If it stopped streaming it should go through `apiFetch` and "
            "recover sessions like everything else."
        )


def test_the_status_branching_exemptions_really_branch_on_status():
    """Otherwise it is an ordinary read that should recover its session."""
    for rel in sorted(STATUS_BRANCHING):
        path = ROOT / rel
        assert path.exists(), f"{rel} is gone; remove it from the exemption list"
        text = strip_ts_comments(read_source(path))
        assert re.search(r"\.status\s*===", text), (
            f"{rel} is exempted for branching on an HTTP status and does not. "
            "It should go through `apiFetch`."
        )


def test_the_public_exemptions_are_actually_public():
    """A dashboard page is behind auth and has no business on this list."""
    for rel in sorted(PUBLIC_UNAUTHENTICATED):
        path = ROOT / rel
        assert path.exists(), f"{rel} is gone; remove it from the exemption list"
        assert "(dashboard)" not in rel, (
            f"{rel} is exempted as a public page but lives under (dashboard), "
            "which is authenticated. It should use `apiFetch`."
        )


def test_raw_api_fetches_do_not_grow():
    found = {k: v for k, v in _raw_api_fetches().items() if k not in EXEMPT}
    total = sum(len(v) for v in found.values())
    assert total <= KNOWN_UNCONVERTED, (
        f"{total} non-streaming raw API fetches, up from {KNOWN_UNCONVERTED}. "
        "Each one skips `apiFetch`'s 401 refresh-and-retry, so an expired token "
        "makes it fail while sibling calls on the same page recover — which "
        "renders as a working page showing wrong data. New code should call "
        f"`apiFetch`.\n  " + "\n  ".join(f"{k}: {len(v)}" for k, v in sorted(found.items()))
    )


def test_the_reputation_score_goes_through_the_api_client():
    """The one traced end to end: it rendered "0 points" for a stale token."""
    page = strip_ts_comments(
        (ROOT / "frontend/src/app/(dashboard)/dashboard/reputation/page.tsx").read_text(
            encoding="utf-8"
        )
    )
    assert 'fetch("/api/reputation/me"' not in page, (
        "the reputation page fetches its own score raw again, so an expired "
        "token renders 0 beside a leaderboard that refreshed successfully"
    )
    assert "fetchMyReputation()" in page
