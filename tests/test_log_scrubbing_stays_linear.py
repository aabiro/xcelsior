"""Scrubbing a log line costs time proportional to its length, not its square.

`PIIScrubFilter` runs on **every log record**, so its cost is paid by every
request, and it is paid worst on the longest lines — which are tracebacks and
JSON payloads, the records that matter most when something is wrong.

The email pattern was unbounded on both halves:

    [A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}

With no `@` anywhere in the line, the engine still starts at every position,
consumes the run of local-part characters to the end, fails, and restarts one
character along. Measured on this machine:

    1 KB, no @   1.2 ms
    4 KB, no @    29 ms

Quadrupling the line multiplied the cost by twenty-four. Nothing in the suite
noticed, because a slow filter produces correct output.

Two changes, and the tests below are named for what each buys:

* **Length bounds.** RFC 5321 caps the local part at 64 octets and a DNS label
  at 63, so the retry window at each position is a constant rather than the rest
  of the line.
* **An `@` check before the pass.** Most log lines contain no address at all,
  and a substring scan settles that far more cheaply than a regex.

The bounds also fixed an accuracy bug, asserted separately below: unbounded, the
local part could swallow a long dotted prefix and redact it as part of the
address.

**Timing assertions are deliberately loose** — a hundredfold margin over what
was measured, because this box also runs the dev pool and CI. They are here to
catch a return to quadratic behaviour, which was seconds, not to police
microseconds.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402

from log_pii_filter import _EMAIL_RE, _scrub  # noqa: E402


def test_a_long_line_with_no_address_is_not_rescanned_from_every_position():
    """The defect, at the size that made it obvious.

    64 KB with no `@`. Under the old pattern this was several seconds; the
    substring check now settles it without a regex pass at all.
    """
    line = "abcdefghij." * 6000  # ~66 KB, no '@' anywhere
    started = time.perf_counter()
    out = _scrub(line)
    elapsed = time.perf_counter() - started

    assert out == line, "a line with nothing sensitive in it was modified"
    assert elapsed < 1.0, (
        f"scrubbing a 66 KB line with no address took {elapsed:.2f}s — the email "
        "pattern is rescanning from every position again"
    )


def test_a_long_line_containing_an_at_sign_is_still_bounded():
    """The `@` check cannot save this one; only the length bounds do.

    The input matters, and an earlier version of this test got it wrong. A long
    line built from `"GET /api/... 200 12ms "` passes against the *unbounded*
    pattern too, because the spaces break the run of local-part characters into
    short pieces and there is nothing to backtrack over. It looked like a
    regression test and tested nothing.

    The shape that actually hurts is an unbroken run of characters the local
    part accepts, followed by an `@` that does not begin a valid address — a
    base64 blob or a dotted identifier chain in a traceback. The engine consumes
    the run to the `@`, fails on the domain, restarts one character along, and
    does it again. Measured on the unbounded pattern:

        2 KB    4 ms
        8 KB   72 ms
        20 KB  496 ms

    Ten times the input for a hundred and twenty times the cost. Bounded, the
    same 20 KB is 3.7 ms.
    """
    line = "a." * 20_000 + "@ nothing"  # 40 KB, no valid address in it
    started = time.perf_counter()
    out = _scrub(line)
    elapsed = time.perf_counter() - started

    assert out == line, "a line with no address in it was modified"
    assert elapsed < 0.5, (
        f"scrubbing a 40 KB line took {elapsed:.2f}s — unbounded, this shape is "
        "quadratic and costs seconds; bounded it is milliseconds"
    )


def test_the_local_part_cannot_swallow_the_line_before_it():
    """The accuracy half, which is why this is not only a performance change.

    `job.step.retry.attempt.…user@example.com` is one unbroken run of characters
    the local part accepts. Unbounded, the match began at the first character
    and 147 characters of diagnostic text were replaced by a single tag — the
    log line lost the very context it was written to carry.
    """
    prefix = "job.step.retry.attempt." * 6
    line = prefix + "user@example.com finished"
    out = _scrub(line)

    assert "user@example.com" not in out, "the address survived"
    assert "finished" in out, "text after the address was destroyed"
    assert "job.step.retry" in out, (
        "the local part swallowed the dotted prefix and redacted it as part of "
        "the address; the bound on the local part is gone"
    )


@pytest.mark.parametrize(
    "address",
    [
        "user@example.com",
        "a.b+tag@mail.example.co.uk",
        "x@y.io",
        "UPPER@EXAMPLE.COM",
        "dash-user@sub-domain.example.org",
    ],
)
def test_ordinary_addresses_are_still_redacted(address):
    """The bounds must not have narrowed what counts as an address.

    A faster pattern that misses real addresses is a leak, not an optimisation.
    """
    out = _scrub(f"contacting {address} now")
    assert address not in out, f"{address} was not redacted"
    assert "contacting" in out and "now" in out


def test_an_address_at_the_end_of_a_sentence_still_matches():
    """The trailing dot is why the domain half is not possessive.

    `user@example.com.` ends a sentence. A possessive repeat over `label.`
    groups consumes the final dot, leaves nothing for the TLD, and refuses to
    give it back — so the address would not match at all.
    """
    out = _scrub("mail went to user@example.com.")
    assert "user@example.com" not in out


def test_both_halves_of_the_pattern_are_bounded():
    """Guards the fix rather than its effect.

    A timing assertion tells you something regressed; this says what. If either
    quantifier loses its upper bound the quadratic behaviour returns, and it
    returns silently because the output stays correct.
    """
    source = _EMAIL_RE.pattern
    assert "{1,64}" in source, "the local part is unbounded again"
    assert "{1,63}" in source, "the domain label is unbounded again"
    assert "+@" not in source, (
        "the local part ends in an unbounded repeat before the @, which is the "
        "exact shape that rescans every position"
    )
