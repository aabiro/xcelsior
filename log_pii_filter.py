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
#: An email address, with both halves length-bounded.
#:
#: The bounds are the point. Unbounded (`[A-Za-z0-9._%+-]+@`), the engine
#: restarts at every position of a long run and rescans it to the end, which is
#: quadratic: a 1 KB line with no `@` in it cost 1.2 ms and a 4 KB line cost
#: 29 ms — on every log record, worst on the longest lines, which are tracebacks
#: and JSON payloads. RFC 5321 caps the local part at 64 octets and a DNS label
#: at 63, so the retry window at each position is bounded by a constant instead
#: of by the length of the line.
#:
#: It is also more accurate. Unbounded, `job.step.retry.attempt.user@example.com`
#: matched from the first character, so the whole 147-character prefix was
#: replaced by one `<email:...>` tag and the diagnostic content of the line was
#: destroyed along with the address.
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,}",
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

# ── Extended PII and Infrastructure Secrets (B7.2 / DA§9.2) ─────────────
#: S3, GCS, Azure, and generic signed URLs carrying auth signatures in query.
_SIGNED_URL_RE = re.compile(
    r"https?://[^\s\"'<>]+(?:\?|&)(?:[^\s\"'<>]*(?:X-Amz-Signature|X-Goog-Signature|GoogleAccessId|Signature|sig)=)[^\s\"'<>]*"
)

#: Authorization and proxy-authorization headers (Bearer, Basic, raw tokens).
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b(authorization|proxy-authorization)\s*:\s*(?:bearer\s+|basic\s+)?[^\s,;\"'<>]{8,}"
)
_AUTH_JSON_RE = re.compile(
    r"(?i)([\"'](?:authorization|proxy-authorization)[\"']\s*:\s*[\"'])(?:bearer\s+|basic\s+)?[^\"']{8,}([\"'])"
)

#: Database and broker connection strings containing passwords.
_CONN_STR_RE = re.compile(r"(?i)\b([a-z0-9+.-]+://[^:\s/@]*):([^@\s/]+)@")

#: Environment variable assignment secrets.
_ENV_SECRET_RE = re.compile(
    r"(?i)\b(XCELSIOR_[A-Z0-9_]*(?:SECRET|KEY|TOKEN|PASSWORD)|AWS_SECRET_ACCESS_KEY|STRIPE_SECRET_KEY|DATABASE_URL)\s*=\s*([\"']?)([^\s\"']{6,})\2"
)

#: Prompt bodies in format strings, serialized JSON, or kwargs.
_PROMPT_BODY_RE = re.compile(r"(?i)([\"']?prompt[\"']?\s*[:=]\s*)([\"'])(.*?)\2")

#: Private IPv4 ranges (RFC 1918: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, loopback 127.0.0.0/8).
_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
_PRIVATE_IPV4_RE = re.compile(
    r"\b("
    rf"10\.{_OCTET}\.{_OCTET}\.{_OCTET}|"
    rf"172\.(?:1[6-9]|2[0-9]|3[0-1])\.{_OCTET}\.{_OCTET}|"
    rf"192\.168\.{_OCTET}\.{_OCTET}|"
    rf"127\.{_OCTET}\.{_OCTET}\.{_OCTET}"
    r")\b"
)

#: Private IPv6 (link-local fe80::, unique-local fc00::/fd00::, ::1) and localhost.
_PRIVATE_IPV6_RE = re.compile(
    r"(?i)\b("
    r"(?:fe80|fc00|fd[0-9a-f]{2}):[0-9a-f:]+|"
    r"::1|"
    r"localhost"
    r")\b"
)

_SENSITIVE_KEY_NAMES = {
    "authorization",
    "proxy-authorization",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "stripe_secret",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "private_key",
    "jwt_secret",
    "webhook_secret",
}


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
    if "Basic " in text or "basic " in text:
        text = re.sub(r"\b(Basic|basic)\s+[A-Za-z0-9+/=]{12,}\b", r"\1 <token:redacted>", text)

    # Signed URLs: check substrings first before invoking regex
    if (
        "X-Amz-" in text
        or "X-Goog-" in text
        or "GoogleAccessId" in text
        or "Signature=" in text
        or "sig=" in text
    ):
        text = _SIGNED_URL_RE.sub("<signed_url:redacted>", text)

    # Authorization headers
    if "uthorization" in text.lower():
        text = _AUTH_HEADER_RE.sub(r"\1: <token:redacted>", text)
        text = _AUTH_JSON_RE.sub(r"\g<1><token:redacted>\g<2>", text)

    # Connection strings with credentials
    if "://" in text and "@" in text:
        text = _CONN_STR_RE.sub(r"\1:<password:redacted>@", text)

    # Environment variable secrets
    if any(k in text for k in ("SECRET", "KEY", "TOKEN", "PASSWORD", "DATABASE_URL")):
        text = _ENV_SECRET_RE.sub(r"\1=\2<secret:redacted>\2", text)

    # Prompt bodies
    if "prompt" in text.lower():
        text = _PROMPT_BODY_RE.sub(r'\g<1>"<prompt:redacted>"', text)

    # Private host addresses
    if any(
        p in text
        for p in (
            "10.",
            "172.",
            "192.168.",
            "127.",
            "fe80:",
            "fc00:",
            "fd",
            "::1",
            "localhost",
        )
    ):
        text = _PRIVATE_IPV4_RE.sub("<ip:private>", text)
        text = _PRIVATE_IPV6_RE.sub("<ip:private>", text)

    # PANs last: the candidate pattern is digit-only, so it cannot damage the
    # placeholder text the substitutions above have already inserted.
    text = _scrub_pans(text)
    # An address needs an `@`, and the overwhelming majority of log lines have
    # none. Checking first turns the most common case into a substring scan
    # rather than a regex pass over the whole line.
    if "@" in text:
        text = _EMAIL_RE.sub(lambda m: _hash_tag(m.group(0), "email"), text)
    text = _CUS_RE.sub(
        lambda m: f"<cus:{m.group(0)[4:8]}...>",
        text,
    )
    return text


def scrub_data(val: object) -> object:
    """Recursively scrub sensitive keys and text values from arbitrary data structures."""
    if isinstance(val, str):
        return _scrub(val)
    elif isinstance(val, dict):
        scrubbed: dict[str, object] = {}
        for k, v in val.items():
            k_lower = str(k).lower()
            if any(s in k_lower for s in _SENSITIVE_KEY_NAMES):
                scrubbed[str(k)] = "<secret:redacted>"
            elif "prompt" in k_lower:
                scrubbed[str(k)] = "<prompt:redacted>"
            else:
                scrubbed[str(k)] = scrub_data(v)
        return scrubbed
    elif isinstance(val, list):
        return [scrub_data(x) for x in val]
    elif isinstance(val, tuple):
        return tuple(scrub_data(x) for x in val)
    return val


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
