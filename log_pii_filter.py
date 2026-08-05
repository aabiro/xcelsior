"""P3/C3 — PII log scrubbing.

Redacts sensitive identifiers from log records emitted anywhere under
the `xcelsior` logger tree before they reach handlers. Applied as a
``logging.Filter`` so it catches both ``log.info("...%s...", email)``
style messages (via record.getMessage()) and pre-formatted strings.

Patterns redacted:
    - email addresses  → ``<email:ab12cd>`` (first 6 chars of sha256)
    - Stripe customer IDs (cus_XXXXXXXXXXXXXX) → ``<cus:XXXX...>``
    - API keys / bearer tokens with 32+ hex or base64 chars → ``<token:…>``
    - JWT-shaped strings (xxx.yyy.zzz)                       → ``<jwt:…>``

Disable entirely with ``XCELSIOR_PII_SCRUB=0`` (useful for local debug).

Design notes:
    - Hash-prefix the email so the same user's actions can still be
      correlated across log lines without leaking the email itself.
    - Keep scrubbing fast: single compiled regex with a dispatch dict.
      Logging is on the hot path; a slow filter adds latency to every
      request.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re

# Order matters: match longer/specific patterns first so we don't eat
# substrings that would also match a broader pattern.
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
)
_CUS_RE = re.compile(r"\bcus_[A-Za-z0-9]{14,}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{16,}\b")

# ── Payment and credential material ──────────────────────────────────────
#
# Everything below was reaching logs unredacted. The billing surface now has
# more places that touch a `client_secret` than it did — a manual top-up, an
# SCA decline carrying the declined PaymentIntent, a SetupIntent for the
# dashboard — and the plan's gate is that none of it appears "in any tool
# result, log, trace, audit row, or error string".

#: A Stripe client secret. `pi_..._secret_...` and `seti_..._secret_...` are
#: bearer credentials for *confirming a payment*: anyone holding one can
#: complete or cancel that intent. They travel in error strings, which is
#: exactly where an exception handler logs them.
_STRIPE_CLIENT_SECRET_RE = re.compile(r"\b(pi|seti)_[A-Za-z0-9]+_secret_[A-Za-z0-9]+\b")

#: Stripe API keys and webhook signing secrets. `sk_live_` is the one that
#: matters most and the one most likely to end up in a traceback from a
#: misconfigured client.
_STRIPE_KEY_RE = re.compile(r"\b(sk|rk|whsec)_(live|test)?_?[A-Za-z0-9]{16,}\b")

#: This platform's own credentials. `xcel_ai_` keys are pasted into editor
#: configs by users and never expire, so one in a log line is a durable
#: credential sitting in a file someone else can read.
_XCELSIOR_TOKEN_RE = re.compile(r"\b(xcel_ai|xoa)_[A-Za-z0-9]{16,}\b")

#: A candidate primary account number: 13–19 digits, optionally split by
#: spaces or hyphens the way humans and forms write them.
#:
#: Deliberately *not* the final word — a Luhn check decides, because this
#: pattern also matches timestamps, job ids and byte counts, and a scrubber
#: that mangles every long number in the logs gets turned off.
_PAN_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


def _luhn_ok(digits: str) -> bool:
    """The check digit every real card number satisfies.

    Used to tell a card number from a long integer that happens to be nearby.
    A false positive here costs a redacted timestamp; a false negative leaks a
    PAN, so the pattern is broad and this is what narrows it.
    """
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = ord(char) - 48
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _scrub_pans(text: str) -> str:
    def _replace(match: re.Match) -> str:
        digits = re.sub(r"[ -]", "", match.group(0))
        if not (13 <= len(digits) <= 19) or not _luhn_ok(digits):
            return match.group(0)
        # Last four only, which is what a human needs to identify the card and
        # is what Stripe itself stores for display.
        return f"<pan:...{digits[-4:]}>"

    return _PAN_CANDIDATE_RE.sub(_replace, text)


def _hash_tag(value: str, prefix: str, length: int = 6) -> str:
    h = hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:length]
    return f"<{prefix}:{h}>"


def _scrub(text: str) -> str:
    if not text:
        return text
    # Credential material first: these are the highest-consequence matches and
    # some of them contain substrings the broader patterns would otherwise eat.
    text = _STRIPE_CLIENT_SECRET_RE.sub("<stripe_client_secret:redacted>", text)
    text = _STRIPE_KEY_RE.sub("<stripe_key:redacted>", text)
    text = _XCELSIOR_TOKEN_RE.sub("<xcelsior_token:redacted>", text)
    text = _JWT_RE.sub("<jwt:redacted>", text)
    text = _BEARER_RE.sub("Bearer <token:redacted>", text)
    # PANs last: the candidate pattern is digit-only, so it cannot damage the
    # placeholder text the substitutions above have already inserted.
    text = _scrub_pans(text)
    text = _EMAIL_RE.sub(lambda m: _hash_tag(m.group(0), "email"), text)
    text = _CUS_RE.sub(
        lambda m: f"<cus:{m.group(0)[4:8]}...>",
        text,
    )
    return text


class PIIScrubFilter(logging.Filter):
    """Rewrite LogRecord.msg (post-format) to remove PII."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            formatted = record.getMessage()
        except Exception:
            # Malformed args — let the handler see the raw record.
            return True
        scrubbed = _scrub(formatted)
        if scrubbed != formatted:
            # Replace msg + clear args so handlers don't re-format.
            record.msg = scrubbed
            record.args = ()
        return True


_INSTALLED = False


def install(logger_name: str = "xcelsior") -> None:
    """Attach the scrub filter to ``logger_name`` (idempotent).

    The filter attaches to the *logger* rather than individual handlers
    so new handlers added later (e.g., by uvicorn or a test harness)
    inherit scrubbing automatically.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("XCELSIOR_PII_SCRUB", "1") == "0":
        return
    logging.getLogger(logger_name).addFilter(PIIScrubFilter())
    # Also install on root so non-xcelsior modules (uvicorn, fastapi)
    # that log request data get scrubbed too.
    logging.getLogger().addFilter(PIIScrubFilter())
    _INSTALLED = True
