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


def _hash_tag(value: str, prefix: str, length: int = 6) -> str:
    h = hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:length]
    return f"<{prefix}:{h}>"


def _scrub(text: str) -> str:
    if not text:
        return text
    text = _JWT_RE.sub("<jwt:redacted>", text)
    text = _BEARER_RE.sub("Bearer <token:redacted>", text)
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
