"""Wise provider payouts: one approved transfer per provider, from the ledger.

Wise charges per transfer, so this rail does not pay job by job. A provider's
queued settlements (``payout_splits`` rows on the ``wise`` rail) are added up
into one ``wise_payouts`` row, a platform admin approves it, and only then is a
transfer created at Wise. By default the transfer is left unfunded and a person
releases it in the Wise app, so the API token on this server cannot move money
on its own. ``XCELSIOR_WISE_AUTO_FUND=1`` funds from the Wise balance instead.

What the design refuses to do, and why:

* **Nobody supplies an amount.** The amount is the sum of ledger rows that
  ``provider_settlement.prepare_settlement`` derived from the paid wallet
  charge. Routes pass a provider id or a payout id, never money.
* **A destination that was just set cannot be paid.** Setting or changing it
  starts a cool-off (72 h by default) and emails the provider, so a stolen
  session cannot redirect a payout and collect it the same day.
* **An approved payout cannot be re-pointed.** The destination is copied onto
  the payout when it is batched, the transfer is created against that copy,
  and Wise's recipient is read back and compared with the stored fingerprint
  before any transfer is created.
* **A timeout never becomes a second transfer.** The idempotency id is on the
  row before Wise is called. An unknown outcome is retried with the same id.
* **A webhook is a hint, not a fact.** Its signature is verified against a
  pinned Wise key and its delivery id is recorded once, but the state that is
  applied is always read back from Wise's API.
* **A transfer that came back is never re-paid automatically.** Cancelled,
  refunded and bounced transfers stop at ``needs_review``; an admin confirms
  the money is back in the balance and releases the rows to be paid again.

Bank details are never stored here. The provider's destination lives at Wise;
this side keeps Wise's recipient id, a masked hint and a fingerprint.

Operator steps (setup, adding a provider, what each stuck state means) are in
``api/docs/WISE_PAYOUTS_RUNBOOK.md``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import uuid
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from money import MICROS_PER_CAD

log = logging.getLogger("xcelsior.wise")

MICROS_PER_CENT = MICROS_PER_CAD // 100

_API_BASE = {
    "production": "https://api.wise.com",
    "sandbox": "https://api.wise-sandbox.com",
}

#: Wise's webhook signing keys, as published at
#: https://docs.wise.com/guides/developer/webhooks/event-handling (read
#: 2026-11-10). Public keys, kept in source because ``*.pem`` is git-ignored
#: and a deploy from a clean checkout must still be able to verify.
_WEBHOOK_KEYS = {
    "production": """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvO8vXV+JksBzZAY6GhSO
XdoTCfhXaaiZ+qAbtaDBiu2AGkGVpmEygFmWP4Li9m5+Ni85BhVvZOodM9epgW3F
bA5Q1SexvAF1PPjX4JpMstak/QhAgl1qMSqEevL8cmUeTgcMuVWCJmlge9h7B1CS
D4rtlimGZozG39rUBDg6Qt2K+P4wBfLblL0k4C4YUdLnpGYEDIth+i8XsRpFlogx
CAFyH9+knYsDbR43UJ9shtc42Ybd40Afihj8KnYKXzchyQ42aC8aZ/h5hyZ28yVy
Oj3Vos0VdBIs/gAyJ/4yyQFCXYte64I7ssrlbGRaco4nKF3HmaNhxwyKyJafz19e
HwIDAQAB
-----END PUBLIC KEY-----
""",
    "sandbox": """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwpb91cEYuyJNQepZAVfP
ZIlPZfNUefH+n6w9SW3fykqKu938cR7WadQv87oF2VuT+fDt7kqeRziTmPSUhqPU
ys/V2Q1rlfJuXbE+Gga37t7zwd0egQ+KyOEHQOpcTwKmtZ81ieGHynAQzsn1We3j
wt760MsCPJ7GMT141ByQM+yW1Bx+4SG3IGjXWyqOWrcXsxAvIXkpUD/jK/L958Cg
nZEgz0BSEh0QxYLITnW1lLokSx/dTianWPFEhMC9BgijempgNXHNfcVirg1lPSyg
z7KqoKUN0oHqWLr2U1A+7kqrl6O2nx3CKs1bj1hToT1+p4kcMoHXA7kA+VBLUpEs
VwIDAQAB
-----END PUBLIC KEY-----
""",
}

_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,24}$")
_NAME_RE = re.compile(r"^[^\x00-\x1f<>{}\[\]\\|]{2,140}$")

#: Open states: a provider has at most one payout in any of these (the
#: partial unique index ``uq_wise_payouts_one_open`` says the same).
OPEN_STATES = (
    "pending_approval",
    "approved",
    "creating",
    "unknown",
    "awaiting_funding",
    "funded",
    "needs_review",
)

#: Wise transfer states and what each means for a payout. Anything not listed
#: is treated as needing a person, never as success.
_WISE_STATE_TO_STATUS = {
    "incoming_payment_waiting": "awaiting_funding",
    "incoming_payment_initiated": "awaiting_funding",
    "waiting_recipient_input_to_proceed": "funded",
    "processing": "funded",
    "funds_converted": "funded",
    "outgoing_payment_sent": "sent",
}
_WISE_RETURNED_STATES = frozenset(
    {"cancelled", "funds_refunded", "bounced_back", "charged_back", "unknown"}
)


class WiseError(RuntimeError):
    """A refusal with a stable, API-safe reason."""

    def __init__(self, reason: str, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.reason = reason
        self.status_code = status_code


class WiseAPIError(WiseError):
    """Wise answered and said no. The request did not take effect."""

    def __init__(self, http_status: int, message: str):
        super().__init__("wise_api_error", message, status_code=502)
        self.http_status = http_status


class WiseUnknownOutcome(WiseError):
    """No answer, or a 5xx: the request may or may not have taken effect."""

    def __init__(self, message: str):
        super().__init__("wise_unknown_outcome", message, status_code=502)


# ── Configuration ────────────────────────────────────────────────────


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_cad_micros(name: str, default_cad: str) -> int:
    raw = os.environ.get(name, default_cad)
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise WiseError("bad_config", f"{name} must be a decimal amount", status_code=503) from exc
    if value < 0:
        raise WiseError("bad_config", f"{name} cannot be negative", status_code=503)
    return int(value * MICROS_PER_CAD)


def wise_env() -> str:
    env = os.environ.get("XCELSIOR_WISE_ENV", "sandbox").strip().lower()
    if env not in _API_BASE:
        raise WiseError(
            "bad_config", "XCELSIOR_WISE_ENV must be sandbox or production", status_code=503
        )
    return env


def wise_configured() -> bool:
    """A token and a profile id are present. Says nothing about being enabled."""

    return bool(
        os.environ.get("XCELSIOR_WISE_API_TOKEN", "").strip()
        and os.environ.get("XCELSIOR_WISE_PROFILE_ID", "").strip().isdigit()
    )


def payouts_enabled() -> bool:
    """The kill switch. Off unless set, whatever else is configured."""

    return wise_configured() and _env_flag("XCELSIOR_WISE_PAYOUTS_ENABLED")


def auto_fund_enabled() -> bool:
    return _env_flag("XCELSIOR_WISE_AUTO_FUND")


def profile_id() -> int:
    raw = os.environ.get("XCELSIOR_WISE_PROFILE_ID", "").strip()
    if not raw.isdigit():
        raise WiseError("not_configured", "Wise is not configured", status_code=503)
    return int(raw)


def min_payout_micros() -> int:
    return _env_cad_micros("XCELSIOR_WISE_MIN_PAYOUT_CAD", "20")


def max_payout_micros() -> int:
    return _env_cad_micros("XCELSIOR_WISE_MAX_PAYOUT_CAD", "2000")


def daily_cap_micros() -> int:
    return _env_cad_micros("XCELSIOR_WISE_DAILY_CAP_CAD", "5000")


def cooloff_hours() -> int:
    raw = os.environ.get("XCELSIOR_WISE_RECIPIENT_COOLOFF_HOURS", "72").strip()
    try:
        hours = int(raw)
    except ValueError as exc:
        raise WiseError(
            "bad_config",
            "XCELSIOR_WISE_RECIPIENT_COOLOFF_HOURS must be whole hours",
            status_code=503,
        ) from exc
    # A cool-off of zero is the control switched off. Sandbox may; production
    # may not, whatever the variable says.
    if wise_env() == "production":
        return max(24, hours)
    return max(0, hours)


def allowed_target_currencies() -> frozenset[str]:
    raw = os.environ.get("XCELSIOR_WISE_TARGET_CURRENCIES", "CAD,USD,EUR,GBP")
    return frozenset(
        c.strip().upper() for c in raw.split(",") if re.fullmatch(r"[A-Za-z]{3}", c.strip())
    )


def micros_to_amount(micros: int) -> str:
    """Whole cents as a decimal string. Refuses a sub-cent amount."""

    micros = int(micros)
    if micros <= 0 or micros % MICROS_PER_CENT:
        raise WiseError("bad_amount", "payout amount must be a positive whole number of cents")
    cents = micros // MICROS_PER_CENT
    return f"{cents // 100}.{cents % 100:02d}"


def amount_to_micros(value: Any) -> int:
    try:
        return int((Decimal(str(value)) * MICROS_PER_CAD).to_integral_value())
    except (InvalidOperation, ValueError) as exc:
        raise WiseError("bad_amount", "Wise returned an amount that is not a number") from exc


def mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}"


def destination_fingerprint(*, recipient_id: int, currency: str, wise_hash: str) -> str:
    """Identity of a destination: Wise's id, its currency, and Wise's own hash.

    Wise recomputes ``hash`` whenever the recipient's name or account details
    change, so a fingerprint that still matches means the money goes where it
    went when the destination was approved.
    """

    canonical = f"wise|{int(recipient_id)}|{currency.upper()}|{wise_hash.strip().lower()}"
    return hashlib.sha256(canonical.encode()).hexdigest()


def email_recipients_enabled() -> bool:
    """Provider self-service by Wise email. Off until Wise has agreed to it.

    Wise's docs: "Contact us before attempting to use email recipients". They
    are also per-route and cannot be exercised in sandbox.
    """

    return _env_flag("XCELSIOR_WISE_EMAIL_RECIPIENTS")


# ── Webhook signature ────────────────────────────────────────────────


def _webhook_public_key_pem() -> bytes:
    """The pinned Wise signing key for this environment.

    An override in ``XCELSIOR_WISE_WEBHOOK_PUBLIC_KEY`` wins (for a key
    rotation ahead of a release). There is no fallback to "accept unsigned".
    """

    override = os.environ.get("XCELSIOR_WISE_WEBHOOK_PUBLIC_KEY", "").strip()
    if override:
        return override.replace("\\n", "\n").encode()
    return _WEBHOOK_KEYS[wise_env()].encode()


def verify_webhook_signature(body: bytes, signature_b64: str) -> bool:
    """RSA PKCS#1 v1.5 over SHA-256 of the raw body, against the pinned key."""

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    if not signature_b64 or not body:
        return False
    try:
        signature = base64.b64decode(signature_b64, validate=True)
        key = serialization.load_pem_public_key(_webhook_public_key_pem())
    except Exception:
        return False
    if not isinstance(key, rsa.RSAPublicKey):
        return False
    try:
        key.verify(signature, body, padding.PKCS1v15(), hashes.SHA256())
    except InvalidSignature:
        return False
    return True


# ── HTTP client ──────────────────────────────────────────────────────


class WiseClient:
    """The few Wise calls this rail makes. The token never reaches a log line."""

    def __init__(self, *, transport: httpx.BaseTransport | None = None):
        token = os.environ.get("XCELSIOR_WISE_API_TOKEN", "").strip()
        if not token:
            raise WiseError("not_configured", "Wise is not configured", status_code=503)
        self._profile = profile_id()
        self._http = httpx.Client(
            base_url=_API_BASE[wise_env()],
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "User-Agent": "xcelsior-payouts/1",
            },
            timeout=httpx.Timeout(20.0, connect=5.0),
            follow_redirects=False,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "WiseClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _call(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
        uncertain_on_failure: bool = False,
    ) -> Any:
        try:
            resp = self._http.request(method, path, json=json_body, params=params)
        except httpx.HTTPError as exc:
            # The exception text can carry the URL but never the headers.
            message = f"Wise {method} {path} did not answer ({type(exc).__name__})"
            if uncertain_on_failure:
                raise WiseUnknownOutcome(message) from None
            raise WiseAPIError(0, message) from None
        if resp.status_code >= 500 and uncertain_on_failure:
            raise WiseUnknownOutcome(f"Wise {method} {path} answered {resp.status_code}")
        if resp.status_code >= 400:
            raise WiseAPIError(
                resp.status_code,
                f"Wise {method} {path} answered {resp.status_code}: {_safe_error(resp)}",
            )
        if not resp.content:
            return {}
        try:
            return resp.json()
        except ValueError:
            raise WiseAPIError(
                resp.status_code, f"Wise {method} {path} answered non-JSON"
            ) from None

    # Recipients -------------------------------------------------------

    @property
    def profile(self) -> int:
        return self._profile

    def create_email_recipient(
        self, *, holder_name: str, currency: str, email: str, legal_type: str = "PRIVATE"
    ) -> dict:
        return self._call(
            "POST",
            "/v1/accounts",
            json_body={
                "profile": self._profile,
                "accountHolderName": holder_name,
                "currency": currency,
                "type": "email",
                "legalType": legal_type,
                "details": {"email": email},
            },
        )

    def get_recipient(self, recipient_id: int) -> dict:
        return self._call("GET", f"/v2/accounts/{int(recipient_id)}")

    def deactivate_recipient(self, recipient_id: int) -> None:
        self._call("DELETE", f"/v1/accounts/{int(recipient_id)}")

    # Transfers --------------------------------------------------------

    def create_quote(
        self, *, source_currency: str, target_currency: str, source_amount: str, recipient_id: int
    ) -> dict:
        return self._call(
            "POST",
            f"/v3/profiles/{self._profile}/quotes",
            json_body={
                "sourceCurrency": source_currency,
                "targetCurrency": target_currency,
                # A JSON number, as the API requires. The string is whole cents
                # (micros_to_amount), which a float holds exactly at these sizes.
                "sourceAmount": float(source_amount),
                "targetAccount": int(recipient_id),
                # Without this Wise prices a bank-funded transfer and the fee
                # recorded here would not be the fee charged at funding.
                "preferredPayIn": "BALANCE",
            },
        )

    def create_transfer(
        self, *, recipient_id: int, quote_id: str, customer_transaction_id: str, reference: str
    ) -> dict:
        return self._call(
            "POST",
            "/v1/transfers",
            json_body={
                "targetAccount": int(recipient_id),
                "quoteUuid": quote_id,
                "customerTransactionId": customer_transaction_id,
                "details": {"reference": reference},
            },
            uncertain_on_failure=True,
        )

    def get_transfer(self, transfer_id: int) -> dict:
        return self._call("GET", f"/v1/transfers/{int(transfer_id)}")

    def fund_from_balance(self, transfer_id: int) -> dict:
        return self._call(
            "POST",
            f"/v3/profiles/{self._profile}/transfers/{int(transfer_id)}/payments",
            json_body={"type": "BALANCE"},
            uncertain_on_failure=True,
        )

    def cancel_transfer(self, transfer_id: int) -> dict:
        return self._call("PUT", f"/v1/transfers/{int(transfer_id)}/cancel")

    def balances(self) -> list[dict]:
        result = self._call(
            "GET", f"/v4/profiles/{self._profile}/balances", params={"types": "STANDARD"}
        )
        return result if isinstance(result, list) else []

    def available_micros(self, currency: str) -> int:
        for balance in self.balances():
            amount = balance.get("amount") or {}
            if str(amount.get("currency") or balance.get("currency") or "").upper() == currency:
                return amount_to_micros(amount.get("value") or 0)
        return 0


def _safe_error(resp: httpx.Response) -> str:
    """Wise's own error code and message, bounded. Never the request."""

    try:
        body = resp.json()
    except ValueError:
        return "no detail"
    parts: list[str] = []
    if isinstance(body, dict):
        for item in body.get("errors") or []:
            if isinstance(item, dict):
                parts.append(f"{item.get('code', '')} {item.get('message', '')}".strip())
        for key in ("error", "error_description", "message"):
            if body.get(key):
                parts.append(str(body[key]))
    return "; ".join(parts)[:300] or "no detail"


# ── Database plumbing ────────────────────────────────────────────────


@contextmanager
def _conn():
    from db import _get_pg_pool
    from psycopg.rows import dict_row

    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def _public_payout(row: dict) -> dict:
    """A payout for an admin screen. No idempotency id, no fingerprint."""

    keys = (
        "id",
        "provider_id",
        "source_currency",
        "amount_micros",
        "split_count",
        "target_currency",
        "wise_transfer_id",
        "status",
        "wise_state",
        "fee_micros",
        "target_amount",
        "approved_by",
        "approved_at",
        "attempt_count",
        "error",
        "created_at",
        "updated_at",
        "sent_at",
    )
    result = {key: row.get(key) for key in keys}
    result["amount_cad"] = int(row["amount_micros"]) / MICROS_PER_CAD
    return result


# ── Destination ──────────────────────────────────────────────────────


def get_recipient_status(provider_id: str) -> dict:
    """What a provider may see about their own Wise destination."""

    with _conn() as conn:
        row = conn.execute(
            """
            SELECT target_currency, recipient_type, display_hint, status,
                   payouts_enabled_after, updated_at,
                   payouts_enabled_after <= clock_timestamp() AS payable
              FROM provider_wise_recipients
             WHERE provider_id = %s
            """,
            (provider_id,),
        ).fetchone()
        queued = conn.execute(
            """
            SELECT COALESCE(SUM(provider_share_micros), 0)::bigint AS micros,
                   COUNT(*)::integer AS jobs
              FROM payout_splits
             WHERE provider_id = %s AND payment_rail = 'wise'
               AND settlement_status IN ('queued', 'failed')
            """,
            (provider_id,),
        ).fetchone()
        payout = conn.execute(
            """
            SELECT id, amount_micros, target_currency, status, created_at, sent_at
              FROM wise_payouts
             WHERE provider_id = %s
             ORDER BY id DESC
             LIMIT 1
            """,
            (provider_id,),
        ).fetchone()
    return {
        "enabled": payouts_enabled(),
        "recipient": dict(row) if row and row["status"] == "active" else None,
        "queued_micros": int(queued["micros"]),
        "queued_cad": int(queued["micros"]) / MICROS_PER_CAD,
        "queued_jobs": int(queued["jobs"]),
        "minimum_payout_cad": min_payout_micros() / MICROS_PER_CAD,
        "latest_payout": dict(payout) if payout else None,
    }


def _refuse_if_payout_in_progress(provider_id: str) -> None:
    with _conn() as conn:
        busy = conn.execute(
            """
            SELECT id FROM wise_payouts
             WHERE provider_id = %s
               AND status IN ('approved', 'creating', 'unknown', 'awaiting_funding',
                              'funded', 'needs_review')
             LIMIT 1
            """,
            (provider_id,),
        ).fetchone()
    if busy:
        raise WiseError(
            "payout_in_progress",
            "A payout is in progress. The destination can be changed once it completes.",
        )


def _destination_hint(recipient: dict) -> str:
    """What a provider sees to recognise the destination. No full details."""

    email = str((recipient.get("details") or {}).get("email") or "")
    if email and _EMAIL_RE.fullmatch(email):
        return mask_email(email)
    summary = str(recipient.get("longAccountSummary") or "").strip()
    return summary[:80] or f"Wise recipient ending {str(recipient.get('id') or '')[-4:]}"


def _bind_recipient(*, provider_id: str, recipient: dict, actor: str, wise: WiseClient) -> dict:
    """Bind a recipient Wise holds to a provider, and start the cool-off.

    The recipient is the one read back from Wise, never the caller's
    description of it. It has to belong to this business profile: an id from
    some other Wise account must not become a payout destination here.
    """

    recipient_id = int(recipient.get("id") or 0)
    currency = str(recipient.get("currency") or "").upper()
    wise_hash = str(recipient.get("hash") or "")
    owner_profile = recipient.get("profileId", recipient.get("profile"))
    if recipient_id <= 0 or not wise_hash:
        raise WiseError("wise_api_error", "Wise did not return a usable recipient", status_code=502)
    if owner_profile is None or int(owner_profile) != wise.profile:
        raise WiseError(
            "foreign_recipient",
            "That recipient does not belong to this Wise account",
            status_code=400,
        )
    if recipient.get("active") is False:
        raise WiseError(
            "inactive_recipient", "That recipient is not active at Wise", status_code=400
        )
    if currency not in allowed_target_currencies():
        raise WiseError(
            "unsupported_currency",
            "Wise payouts are available in: " + ", ".join(sorted(allowed_target_currencies())),
            status_code=400,
        )
    fingerprint = destination_fingerprint(
        recipient_id=recipient_id, currency=currency, wise_hash=wise_hash
    )

    with _conn() as conn:
        taken = conn.execute(
            "SELECT provider_id FROM provider_wise_recipients "
            "WHERE wise_recipient_id = %s AND provider_id <> %s AND status = 'active'",
            (recipient_id, provider_id),
        ).fetchone()
        if taken:
            raise WiseError("recipient_in_use", "That recipient is bound to another provider")
        previous = conn.execute(
            "SELECT wise_recipient_id, fingerprint, status FROM provider_wise_recipients "
            "WHERE provider_id = %s FOR UPDATE",
            (provider_id,),
        ).fetchone()
        if previous and previous["status"] == "active" and previous["fingerprint"] == fingerprint:
            # Same destination again: nothing changed, so no new cool-off.
            row = conn.execute(
                "SELECT target_currency, recipient_type, display_hint, status, "
                "payouts_enabled_after FROM provider_wise_recipients WHERE provider_id = %s",
                (provider_id,),
            ).fetchone()
            return {"recipient": dict(row), "changed": False, "first": False}
        _dissolve_pending(conn, provider_id, reason="destination changed")
        row = conn.execute(
            """
            INSERT INTO provider_wise_recipients (
                provider_id, wise_recipient_id, target_currency, recipient_type,
                display_hint, fingerprint, status, payouts_enabled_after, updated_by
            )
            VALUES (
                %(provider_id)s, %(recipient_id)s, %(currency)s, %(type)s,
                %(hint)s, %(fingerprint)s, 'active',
                clock_timestamp() + make_interval(hours => %(cooloff)s), %(actor)s
            )
            ON CONFLICT (provider_id) DO UPDATE
               SET wise_recipient_id = EXCLUDED.wise_recipient_id,
                   target_currency = EXCLUDED.target_currency,
                   recipient_type = EXCLUDED.recipient_type,
                   display_hint = EXCLUDED.display_hint,
                   fingerprint = EXCLUDED.fingerprint,
                   status = 'active',
                   payouts_enabled_after = EXCLUDED.payouts_enabled_after,
                   updated_at = clock_timestamp(),
                   updated_by = EXCLUDED.updated_by
            RETURNING target_currency, recipient_type, display_hint, status,
                      payouts_enabled_after
            """,
            {
                "provider_id": provider_id,
                "recipient_id": recipient_id,
                "currency": currency,
                "type": str(recipient.get("type") or "unknown")[:40],
                "hint": _destination_hint(recipient),
                "fingerprint": fingerprint,
                "cooloff": cooloff_hours(),
                "actor": actor[:200],
            },
        ).fetchone()
    return {"recipient": dict(row), "changed": True, "first": previous is None}


def link_recipient(
    *,
    provider_id: str,
    recipient_id: int,
    actor: str,
    client: WiseClient | None = None,
) -> dict:
    """Admin path: bind a recipient that already exists in the Wise account.

    The admin adds the provider as a recipient in Wise (where Wise collects and
    checks the bank details), then links its id here. Bank details never pass
    through this server.
    """

    if not wise_configured():
        raise WiseError("not_configured", "Wise payouts are not available", status_code=503)
    if int(recipient_id) <= 0:
        raise WiseError("bad_recipient", "A Wise recipient id is required", status_code=400)
    _refuse_if_payout_in_progress(provider_id)
    owns_client = client is None
    wise = client or WiseClient()
    try:
        recipient = wise.get_recipient(int(recipient_id))
        return _bind_recipient(provider_id=provider_id, recipient=recipient, actor=actor, wise=wise)
    finally:
        if owns_client:
            wise.close()


def set_email_recipient(
    *,
    provider_id: str,
    email: str,
    currency: str,
    holder_name: str,
    legal_type: str = "PRIVATE",
    actor: str,
    client: WiseClient | None = None,
) -> dict:
    """Provider path: be paid to the email address of a Wise account.

    A payout that is still waiting for approval was batched against the old
    destination, so it is dissolved and its rows go back to the queue. One that
    was already approved is past the point where the destination may change.
    """

    email = str(email or "").strip()
    currency = str(currency or "").strip().upper()
    holder_name = " ".join(str(holder_name or "").split())
    legal_type = str(legal_type or "").strip().upper()
    if not wise_configured():
        raise WiseError("not_configured", "Wise payouts are not available", status_code=503)
    if not email_recipients_enabled():
        raise WiseError(
            "email_recipients_disabled",
            "Wise payouts are set up by Xcelsior support for now. Contact support to be paid by Wise.",
            status_code=403,
        )
    if not _EMAIL_RE.fullmatch(email):
        raise WiseError(
            "bad_email", "Enter the email address of your Wise account", status_code=400
        )
    if currency not in allowed_target_currencies():
        raise WiseError(
            "unsupported_currency",
            "Wise payouts are available in: " + ", ".join(sorted(allowed_target_currencies())),
            status_code=400,
        )
    if not _NAME_RE.fullmatch(holder_name):
        raise WiseError("bad_name", "Enter the account holder's full name", status_code=400)
    if legal_type not in {"PRIVATE", "BUSINESS"}:
        raise WiseError("bad_legal_type", "legal_type must be PRIVATE or BUSINESS", status_code=400)
    _refuse_if_payout_in_progress(provider_id)

    owns_client = client is None
    wise = client or WiseClient()
    try:
        created = wise.create_email_recipient(
            holder_name=holder_name, currency=currency, email=email, legal_type=legal_type
        )
        # Read it back: the hash and owning profile come from Wise's record.
        recipient = wise.get_recipient(int(created.get("id") or 0))
        return _bind_recipient(provider_id=provider_id, recipient=recipient, actor=actor, wise=wise)
    finally:
        if owns_client:
            wise.close()


def disable_recipient(*, provider_id: str, actor: str) -> dict:
    with _conn() as conn:
        busy = conn.execute(
            """
            SELECT id FROM wise_payouts
             WHERE provider_id = %s
               AND status IN ('approved', 'creating', 'unknown', 'awaiting_funding', 'funded')
             LIMIT 1
            """,
            (provider_id,),
        ).fetchone()
        if busy:
            raise WiseError(
                "payout_in_progress",
                "A payout is in progress. Wise can be disconnected once it completes.",
            )
        _dissolve_pending(conn, provider_id, reason="destination removed")
        row = conn.execute(
            """
            UPDATE provider_wise_recipients
               SET status = 'disabled', updated_at = clock_timestamp(), updated_by = %s
             WHERE provider_id = %s
            RETURNING provider_id
            """,
            (actor[:200], provider_id),
        ).fetchone()
    return {"status": "disabled" if row else "not_found"}


def _dissolve_pending(conn, provider_id: str, *, reason: str) -> None:
    """Return a not-yet-approved payout's rows to the queue."""

    rows = conn.execute(
        """
        UPDATE wise_payouts
           SET status = 'rejected', error = %s, updated_at = clock_timestamp()
         WHERE provider_id = %s AND status = 'pending_approval'
        RETURNING id
        """,
        (reason[:500], provider_id),
    ).fetchall()
    for row in rows:
        _requeue_splits(conn, int(row["id"]), expected_status="in_transit")


def _requeue_splits(conn, payout_id: int, *, expected_status: str) -> int:
    done = conn.execute(
        """
        UPDATE payout_splits
           SET settlement_status = 'queued',
               settlement_error = '',
               wise_payout_id = NULL,
               next_attempt_at = NULL,
               updated_at = clock_timestamp()
         WHERE wise_payout_id = %s AND settlement_status = %s
        RETURNING id
        """,
        (payout_id, expected_status),
    ).fetchall()
    return len(done)


# ── Batching ─────────────────────────────────────────────────────────


def batch_queued_settlements(*, provider_id: str | None = None, limit: int = 200) -> dict:
    """Add each provider's queued Wise settlements into one pending payout.

    Runs with or without the kill switch: a pending payout moves no money, and
    an admin should be able to see what would be paid before enabling the rail.
    """

    created: list[int] = []
    skipped: dict[str, str] = {}
    with _conn() as conn:
        providers = conn.execute(
            """
            SELECT DISTINCT ps.provider_id
              FROM payout_splits ps
             WHERE ps.payment_rail = 'wise'
               AND ps.settlement_key IS NOT NULL
               AND ps.settlement_status IN ('queued', 'failed')
               AND (%(provider_id)s::text IS NULL OR ps.provider_id = %(provider_id)s::text)
             ORDER BY ps.provider_id
             LIMIT %(limit)s
            """,
            {"provider_id": provider_id, "limit": max(1, min(int(limit), 1000))},
        ).fetchall()
    for entry in providers:
        pid = str(entry["provider_id"])
        try:
            with _conn() as conn:
                payout_id, why = _batch_one_provider(conn, pid)
        except Exception as exc:  # one provider's fault must not stop the rest
            log.error("wise batch failed provider=%s: %s", pid, type(exc).__name__)
            skipped[pid] = "error"
            continue
        if payout_id:
            created.append(payout_id)
        else:
            skipped[pid] = why
    return {"created": created, "skipped": skipped}


def _batch_one_provider(conn, provider_id: str) -> tuple[int | None, str]:
    recipient = conn.execute(
        """
        SELECT wise_recipient_id, target_currency, fingerprint,
               payouts_enabled_after <= clock_timestamp() AS payable
          FROM provider_wise_recipients
         WHERE provider_id = %s AND status = 'active'
           FOR UPDATE
        """,
        (provider_id,),
    ).fetchone()
    if recipient is None:
        return None, "no_destination"
    if not recipient["payable"]:
        return None, "destination_cooling_off"
    if conn.execute(
        "SELECT 1 FROM wise_payouts WHERE provider_id = %s AND status = ANY(%s) LIMIT 1",
        (provider_id, list(OPEN_STATES)),
    ).fetchone():
        return None, "payout_open"

    rows = conn.execute(
        """
        SELECT id, provider_share_micros, currency
          FROM payout_splits
         WHERE provider_id = %s
           AND payment_rail = 'wise'
           AND settlement_key IS NOT NULL
           AND settlement_status IN ('queued', 'failed')
           AND wise_payout_id IS NULL
           AND provider_share_micros > 0
         ORDER BY created_at ASC, id ASC
           FOR UPDATE SKIP LOCKED
        """,
        (provider_id,),
    ).fetchall()
    cap = max_payout_micros()
    chosen: list[int] = []
    total = 0
    for row in rows:
        if str(row["currency"]).upper() != "CAD":
            continue
        share = int(row["provider_share_micros"])
        if total + share > cap:
            break
        chosen.append(int(row["id"]))
        total += share
    if not chosen:
        return None, "over_cap" if rows else "nothing_queued"
    if total < min_payout_micros():
        return None, "below_minimum"

    payout = conn.execute(
        """
        INSERT INTO wise_payouts (
            provider_id, source_currency, amount_micros, split_count,
            target_currency, wise_recipient_id, recipient_fingerprint,
            customer_transaction_id, status
        )
        VALUES (%s, 'CAD', %s, %s, %s, %s, %s, %s, 'pending_approval')
        RETURNING id
        """,
        (
            provider_id,
            total,
            len(chosen),
            recipient["target_currency"],
            recipient["wise_recipient_id"],
            recipient["fingerprint"],
            str(uuid.uuid4()),
        ),
    ).fetchone()
    payout_id = int(payout["id"])
    moved = conn.execute(
        """
        UPDATE payout_splits
           SET settlement_status = 'in_transit',
               settlement_error = '',
               wise_payout_id = %s,
               updated_at = clock_timestamp()
         WHERE id = ANY(%s) AND wise_payout_id IS NULL
        RETURNING id
        """,
        (payout_id, chosen),
    ).fetchall()
    if len(moved) != len(chosen):
        raise WiseError("batch_race", "settlement rows changed while batching")
    return payout_id, ""


# ── Approval ─────────────────────────────────────────────────────────


def list_payouts(*, status: str | None = None, limit: int = 100) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM wise_payouts
             WHERE (%(status)s::text IS NULL OR status = %(status)s::text)
             ORDER BY id DESC
             LIMIT %(limit)s
            """,
            {"status": status, "limit": max(1, min(int(limit), 500))},
        ).fetchall()
    return [_public_payout(dict(row)) for row in rows]


def approve_payout(*, payout_id: int, approver: str) -> dict:
    """A person says this payout may be sent. Checks the switch and the caps."""

    approver = str(approver or "").strip()
    if not approver:
        raise WiseError("no_approver", "An approver is required", status_code=400)
    if not payouts_enabled():
        raise WiseError("payouts_disabled", "Wise payouts are switched off", status_code=503)
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM wise_payouts WHERE id = %s FOR UPDATE", (int(payout_id),)
        ).fetchone()
        if row is None:
            raise WiseError("not_found", "Payout not found", status_code=404)
        if row["status"] != "pending_approval":
            raise WiseError("not_pending", f"Payout is {row['status']}, not awaiting approval")
        if int(row["amount_micros"]) > max_payout_micros():
            raise WiseError("over_cap", "Payout is above the per-payout cap")

        # The ledger may not have moved under the batch: same rows, same sum.
        check = conn.execute(
            """
            SELECT COALESCE(SUM(provider_share_micros), 0)::bigint AS micros,
                   COUNT(*)::integer AS n
              FROM payout_splits
             WHERE wise_payout_id = %s AND settlement_status = 'in_transit'
               AND payment_rail = 'wise' AND provider_id = %s
            """,
            (row["id"], row["provider_id"]),
        ).fetchone()
        if int(check["micros"]) != int(row["amount_micros"]) or int(check["n"]) != int(
            row["split_count"]
        ):
            raise WiseError("ledger_mismatch", "Payout no longer matches its settlement rows")

        # And the destination must still be the one it was batched against.
        dest = conn.execute(
            """
            SELECT 1 FROM provider_wise_recipients
             WHERE provider_id = %s AND status = 'active'
               AND wise_recipient_id = %s AND fingerprint = %s
               AND payouts_enabled_after <= clock_timestamp()
            """,
            (row["provider_id"], row["wise_recipient_id"], row["recipient_fingerprint"]),
        ).fetchone()
        if dest is None:
            raise WiseError("destination_changed", "The provider's destination has changed")

        # Serialise approvals so two cannot both fit under the daily cap.
        conn.execute("SELECT pg_advisory_xact_lock(hashtext('wise_payout_daily_cap'))")
        spent = conn.execute(
            """
            SELECT COALESCE(SUM(amount_micros), 0)::bigint AS micros
              FROM wise_payouts
             WHERE approved_at > clock_timestamp() - interval '24 hours'
               AND status NOT IN ('rejected', 'released')
            """
        ).fetchone()
        if int(spent["micros"]) + int(row["amount_micros"]) > daily_cap_micros():
            raise WiseError("daily_cap", "Approving this would exceed the 24-hour payout cap")

        updated = conn.execute(
            """
            UPDATE wise_payouts
               SET status = 'approved', approved_by = %s, approved_at = clock_timestamp(),
                   error = '', updated_at = clock_timestamp()
             WHERE id = %s AND status = 'pending_approval'
            RETURNING *
            """,
            (approver[:200], row["id"]),
        ).fetchone()
    return _public_payout(dict(updated))


def reject_payout(*, payout_id: int, actor: str, reason: str = "") -> dict:
    with _conn() as conn:
        updated = conn.execute(
            """
            UPDATE wise_payouts
               SET status = 'rejected', error = %s, updated_at = clock_timestamp()
             WHERE id = %s AND status = 'pending_approval'
            RETURNING *
            """,
            (f"rejected by {actor}: {reason}"[:500], int(payout_id)),
        ).fetchone()
        if updated is None:
            raise WiseError("not_pending", "Only a payout awaiting approval can be rejected")
        _requeue_splits(conn, int(updated["id"]), expected_status="in_transit")
    return _public_payout(dict(updated))


def release_payout(
    *,
    payout_id: int,
    actor: str,
    confirm_no_transfer: bool = False,
    client: WiseClient | None = None,
) -> dict:
    """After a transfer came back: put its rows back in the queue.

    Refused unless Wise itself says the transfer is cancelled or refunded, so a
    payout that is merely slow cannot be released and paid twice. A payout with
    no transfer id on record needs the admin to state that they looked in Wise
    and found no transfer carrying its reference (``XCL-<id>``).
    """

    with _conn() as conn:
        row = conn.execute("SELECT * FROM wise_payouts WHERE id = %s", (int(payout_id),)).fetchone()
    if row is None:
        raise WiseError("not_found", "Payout not found", status_code=404)
    if row["status"] != "needs_review":
        raise WiseError("not_reviewable", "Only a payout in review can be released")
    if row["wise_transfer_id"] is None and not confirm_no_transfer:
        raise WiseError(
            "confirm_no_transfer",
            f"No transfer id is on record. Check Wise for reference XCL-{int(payout_id)} "
            "and confirm that none exists before releasing.",
        )
    if row["wise_transfer_id"] is not None:
        owns_client = client is None
        wise = client or WiseClient()
        try:
            state = str(wise.get_transfer(int(row["wise_transfer_id"])).get("status") or "")
        finally:
            if owns_client:
                wise.close()
        if state not in {"cancelled", "funds_refunded"}:
            raise WiseError(
                "money_not_returned",
                f"Wise reports the transfer as {state or 'unknown'}; it can be released "
                "once it is cancelled or refunded",
            )
    with _conn() as conn:
        updated = conn.execute(
            """
            UPDATE wise_payouts
               SET status = 'released', error = %s, updated_at = clock_timestamp()
             WHERE id = %s AND status = 'needs_review'
            RETURNING *
            """,
            (f"released by {actor}"[:500], int(payout_id)),
        ).fetchone()
        if updated is None:
            raise WiseError("not_reviewable", "Payout changed while it was being released")
        _requeue_splits(conn, int(updated["id"]), expected_status="in_transit")
        _requeue_splits(conn, int(updated["id"]), expected_status="manual_review")
    return _public_payout(dict(updated))


# ── Execution ────────────────────────────────────────────────────────


def execute_approved_payouts(*, limit: int = 20, client: WiseClient | None = None) -> dict:
    """Create transfers for approved payouts. Safe to run again at any time."""

    if not payouts_enabled():
        return {"created": 0, "unknown": 0, "review": 0, "reason": "payouts_disabled"}
    with _conn() as conn:
        rows = conn.execute(
            """
            WITH picked AS (
                SELECT id FROM wise_payouts
                 WHERE status = 'approved'
                    OR (status IN ('creating', 'unknown')
                        AND updated_at < clock_timestamp()
                            - make_interval(secs => LEAST(3600, 60 * GREATEST(1, attempt_count))))
                 ORDER BY id
                   FOR UPDATE SKIP LOCKED
                 LIMIT %s
            )
            UPDATE wise_payouts p
               SET status = 'creating', attempt_count = attempt_count + 1,
                   updated_at = clock_timestamp()
              FROM picked
             WHERE p.id = picked.id
            RETURNING p.*
            """,
            (max(1, min(int(limit), 100)),),
        ).fetchall()
    if not rows:
        return {"created": 0, "unknown": 0, "review": 0}

    counts = {"created": 0, "unknown": 0, "review": 0}
    owns_client = client is None
    wise = client or WiseClient()
    try:
        for row in rows:
            outcome = _execute_one(wise, dict(row))
            counts[outcome] += 1
    finally:
        if owns_client:
            wise.close()
    return counts


def _set_status(payout_id: int, status: str, *, expect: tuple[str, ...], **fields: Any) -> bool:
    allowed = {
        "error",
        "wise_quote_id",
        "wise_transfer_id",
        "wise_state",
        "fee_micros",
        "target_amount",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"unsupported payout fields: {sorted(unknown)}")
    with _conn() as conn:
        row = conn.execute(
            """
            UPDATE wise_payouts
               SET status = %(status)s,
                   error = COALESCE(%(error)s, error),
                   wise_quote_id = COALESCE(%(wise_quote_id)s, wise_quote_id),
                   wise_transfer_id = COALESCE(%(wise_transfer_id)s, wise_transfer_id),
                   wise_state = COALESCE(%(wise_state)s, wise_state),
                   fee_micros = COALESCE(%(fee_micros)s, fee_micros),
                   target_amount = COALESCE(%(target_amount)s, target_amount),
                   updated_at = clock_timestamp()
             WHERE id = %(id)s AND status = ANY(%(expect)s)
            RETURNING id
            """,
            {
                "id": payout_id,
                "status": status,
                "expect": list(expect),
                "error": fields.get("error"),
                "wise_quote_id": fields.get("wise_quote_id"),
                "wise_transfer_id": fields.get("wise_transfer_id"),
                "wise_state": fields.get("wise_state"),
                "fee_micros": fields.get("fee_micros"),
                "target_amount": fields.get("target_amount"),
            },
        ).fetchone()
    return row is not None


def _recipient_matches(recipient: dict, payout: dict, *, profile: int) -> bool:
    """Wise's record of the destination, against the copy made at batching."""

    if recipient.get("active") is False:
        return False
    owner = recipient.get("profileId", recipient.get("profile"))
    if owner is None or int(owner) != int(profile):
        return False
    currency = str(recipient.get("currency") or "")
    wise_hash = str(recipient.get("hash") or "")
    if not currency or not wise_hash:
        return False
    return (
        int(recipient.get("id") or 0) == int(payout["wise_recipient_id"])
        and currency.upper() == str(payout["target_currency"]).upper()
        and destination_fingerprint(
            recipient_id=int(payout["wise_recipient_id"]), currency=currency, wise_hash=wise_hash
        )
        == payout["recipient_fingerprint"]
    )


def _execute_one(wise: WiseClient, payout: dict) -> str:
    payout_id = int(payout["id"])
    amount = micros_to_amount(int(payout["amount_micros"]))
    try:
        recipient = wise.get_recipient(int(payout["wise_recipient_id"]))
        if not _recipient_matches(recipient, payout, profile=wise.profile):
            _set_status(
                payout_id,
                "needs_review",
                expect=("creating",),
                error="destination at Wise no longer matches the approved destination",
            )
            return "review"
        quote = wise.create_quote(
            source_currency=str(payout["source_currency"]),
            target_currency=str(payout["target_currency"]),
            source_amount=amount,
            recipient_id=int(payout["wise_recipient_id"]),
        )
        quote_id = str(quote.get("id") or "")
        if not quote_id:
            raise WiseAPIError(0, "Wise returned a quote without an id")
        transfer = wise.create_transfer(
            recipient_id=int(payout["wise_recipient_id"]),
            quote_id=quote_id,
            customer_transaction_id=str(payout["customer_transaction_id"]),
            reference=f"XCL-{payout_id}",
        )
    except WiseUnknownOutcome as exc:
        _set_status(payout_id, "unknown", expect=("creating",), error=str(exc)[:500])
        return "unknown"
    except WiseAPIError as exc:
        if exc.http_status in (0, 408, 429) or exc.http_status >= 500:
            # Transient. Retried later with the same idempotency id.
            _set_status(payout_id, "unknown", expect=("creating",), error=str(exc)[:500])
            return "unknown"
        # A definite refusal (bad recipient, an approval rule on the Wise
        # account, a validation error). Retrying will not change the answer.
        _set_status(
            payout_id,
            "needs_review",
            expect=("creating",),
            error=f"Wise refused the payout: {exc}"[:500],
        )
        return "review"

    transfer_id = int(transfer.get("id") or 0)
    problems = _transfer_problems(transfer, payout, amount)
    if transfer_id <= 0 or problems:
        if transfer_id > 0:
            try:
                wise.cancel_transfer(transfer_id)
            except WiseError:
                log.error("wise payout %s: could not cancel mismatched transfer", payout_id)
        _set_status(
            payout_id,
            "needs_review",
            expect=("creating",),
            wise_transfer_id=transfer_id or None,
            error=("transfer does not match the approved payout: " + ", ".join(problems))[:500],
        )
        return "review"

    _set_status(
        payout_id,
        "awaiting_funding",
        expect=("creating",),
        wise_quote_id=quote_id,
        wise_transfer_id=transfer_id,
        wise_state=str(transfer.get("status") or ""),
        target_amount=str(transfer.get("targetValue") or ""),
        fee_micros=_quote_fee_micros(quote),
        error="",
    )
    if auto_fund_enabled():
        _fund(
            wise,
            payout_id,
            transfer_id,
            str(payout["source_currency"]),
            int(payout["amount_micros"]),
        )
    return "created"


def _transfer_problems(transfer: dict, payout: dict, amount: str) -> list[str]:
    problems: list[str] = []
    if int(transfer.get("targetAccount") or 0) != int(payout["wise_recipient_id"]):
        problems.append("recipient")
    if str(transfer.get("sourceCurrency") or "").upper() != str(payout["source_currency"]):
        problems.append("source currency")
    if str(transfer.get("targetCurrency") or "").upper() != str(payout["target_currency"]):
        problems.append("target currency")
    try:
        if Decimal(str(transfer.get("sourceValue"))) != Decimal(amount):
            problems.append("amount")
    except (InvalidOperation, ValueError):
        problems.append("amount")
    return problems


def _quote_fee_micros(quote: dict) -> int | None:
    for option in quote.get("paymentOptions") or []:
        if option.get("payIn") == "BALANCE" and not option.get("disabled"):
            price = (((option.get("price") or {}).get("total") or {}).get("value") or {}).get(
                "amount"
            )
            fee = price if price is not None else (option.get("fee") or {}).get("total")
            if fee is not None:
                return amount_to_micros(fee)
    return None


def _fund(
    wise: WiseClient, payout_id: int, transfer_id: int, currency: str, amount_micros: int
) -> None:
    """Fund from the Wise balance. Never from a linked bank account."""

    try:
        if wise.available_micros(currency) < amount_micros:
            _set_status(
                payout_id,
                "awaiting_funding",
                expect=("awaiting_funding",),
                error=f"Wise {currency} balance is below this payout; top up and fund it",
            )
            return
        result = wise.fund_from_balance(transfer_id)
    except WiseError as exc:
        # Includes the 403 Wise answers when the account needs a person to
        # approve the payment. The transfer stays, unfunded, for the Wise app.
        _set_status(
            payout_id,
            "awaiting_funding",
            expect=("awaiting_funding",),
            error=f"not funded automatically: {exc}"[:500],
        )
        return
    if str(result.get("status") or "").upper() == "COMPLETED":
        _set_status(payout_id, "funded", expect=("awaiting_funding",), error="")
    else:
        _set_status(
            payout_id,
            "awaiting_funding",
            expect=("awaiting_funding",),
            error=f"funding answered {result.get('status')}: {result.get('errorCode') or ''}"[:500],
        )


# ── State sync ───────────────────────────────────────────────────────


def sync_payout(*, payout_id: int, client: WiseClient | None = None) -> dict:
    """Read the transfer back from Wise and apply its state to the ledger."""

    with _conn() as conn:
        row = conn.execute("SELECT * FROM wise_payouts WHERE id = %s", (int(payout_id),)).fetchone()
    if row is None:
        raise WiseError("not_found", "Payout not found", status_code=404)
    if row["wise_transfer_id"] is None:
        return _public_payout(dict(row))
    owns_client = client is None
    wise = client or WiseClient()
    try:
        transfer = wise.get_transfer(int(row["wise_transfer_id"]))
    finally:
        if owns_client:
            wise.close()
    return _apply_transfer(dict(row), transfer)


def _apply_transfer(payout: dict, transfer: dict) -> dict:
    payout_id = int(payout["id"])
    if int(transfer.get("id") or 0) != int(payout["wise_transfer_id"]):
        raise WiseError("transfer_mismatch", "Wise returned a different transfer")
    state = str(transfer.get("status") or "").lower()
    target = _WISE_STATE_TO_STATUS.get(state)
    with _conn() as conn:
        current = conn.execute(
            "SELECT * FROM wise_payouts WHERE id = %s FOR UPDATE", (payout_id,)
        ).fetchone()
        if current is None or current["status"] in ("released", "rejected"):
            return _public_payout(dict(current or payout))
        if target == "sent":
            # Also the way back from a bounce: Wise can redeliver a bounced
            # transfer, and then the rows that were un-paid are paid again.
            updated = conn.execute(
                """
                UPDATE wise_payouts
                   SET status = 'sent', wise_state = %s, error = '',
                       sent_at = COALESCE(sent_at, clock_timestamp()),
                       updated_at = clock_timestamp()
                 WHERE id = %s
                RETURNING *
                """,
                (state, payout_id),
            ).fetchone()
            conn.execute(
                """
                UPDATE payout_splits
                   SET settlement_status = 'paid', settlement_error = '',
                       settled_at = clock_timestamp(), updated_at = clock_timestamp()
                 WHERE wise_payout_id = %s
                   AND settlement_status IN ('in_transit', 'manual_review')
                """,
                (payout_id,),
            )
        elif target is not None:
            # Never move a sent payout backwards on a late or reordered read.
            updated = conn.execute(
                """
                UPDATE wise_payouts
                   SET status = CASE WHEN status = 'sent' THEN status ELSE %s END,
                       wise_state = %s, updated_at = clock_timestamp()
                 WHERE id = %s
                RETURNING *
                """,
                (target, state, payout_id),
            ).fetchone()
        else:
            reason = (
                f"Wise reports the transfer as {state}"
                if state in _WISE_RETURNED_STATES
                else f"Wise reports an unrecognised transfer state: {state[:40]}"
            )
            updated = conn.execute(
                """
                UPDATE wise_payouts
                   SET status = 'needs_review', wise_state = %s, error = %s,
                       updated_at = clock_timestamp()
                 WHERE id = %s
                RETURNING *
                """,
                (state[:60], reason, payout_id),
            ).fetchone()
            # A payout that was marked paid and then bounced is no longer paid.
            conn.execute(
                """
                UPDATE payout_splits
                   SET settlement_status = 'manual_review', settlement_error = %s,
                       settled_at = NULL, updated_at = clock_timestamp()
                 WHERE wise_payout_id = %s AND settlement_status = 'paid'
                """,
                (reason, payout_id),
            )
    return _public_payout(dict(updated))


def reconcile_open_payouts(*, limit: int = 50, client: WiseClient | None = None) -> dict:
    """Poll Wise for every payout that has a transfer and is not finished.

    The webhook is the fast path; this is what makes a lost delivery harmless.
    Recently sent payouts are re-read for a week, because a transfer can bounce
    after Wise has reported it sent.
    """

    if not wise_configured():
        return {"checked": 0, "reason": "not_configured"}
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM wise_payouts
             WHERE wise_transfer_id IS NOT NULL
               AND (status IN ('awaiting_funding', 'funded')
                    OR (status = 'sent' AND sent_at > clock_timestamp() - interval '7 days'))
             ORDER BY updated_at ASC
             LIMIT %s
            """,
            (max(1, min(int(limit), 200)),),
        ).fetchall()
    if not rows:
        return {"checked": 0}
    checked = failed = 0
    owns_client = client is None
    wise = client or WiseClient()
    try:
        for row in rows:
            try:
                _apply_transfer(dict(row), wise.get_transfer(int(row["wise_transfer_id"])))
                checked += 1
            except WiseError as exc:
                failed += 1
                log.warning("wise reconcile payout=%s: %s", row["id"], exc.reason)
    finally:
        if owns_client:
            wise.close()
    return {"checked": checked, "failed": failed}


# ── Webhook ──────────────────────────────────────────────────────────


def handle_webhook(
    *, body: bytes, signature: str, delivery_id: str, test_notification: bool = False
) -> dict:
    """Verify and record one delivery. Calls nothing, so it answers at once.

    Wise allows five seconds before it counts a delivery as failed. The state
    is applied later by ``drain_webhook_inbox``, from Wise's API rather than
    from this payload, so nothing here depends on what the payload claims.
    """

    if not verify_webhook_signature(body, signature):
        raise WiseError("bad_signature", "Invalid Wise webhook signature", status_code=400)
    try:
        event = json.loads(body)
    except ValueError:
        raise WiseError("bad_body", "Webhook body is not JSON", status_code=400) from None
    if not isinstance(event, dict):
        raise WiseError("bad_body", "Webhook body is not an object", status_code=400)
    event_type = str(event.get("event_type") or "")[:80]
    raw_data = event.get("data")
    data: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}
    raw_resource = data.get("resource")
    resource: dict[str, Any] = raw_resource if isinstance(raw_resource, dict) else {}
    try:
        transfer_id = int(resource.get("id") or 0) if event_type.startswith("transfers#") else 0
    except (TypeError, ValueError):
        transfer_id = 0
    # Without Wise's delivery id, the body's own hash is the identity.
    delivery = (delivery_id or "").strip()[:120] or "sha256:" + hashlib.sha256(body).hexdigest()
    outcome = "test" if test_notification else ("pending" if transfer_id else "ignored")
    # Kept: what happened and to which transfer. Not kept: the rest of the body.
    summary = {
        "event_type": event_type,
        "schema_version": str(event.get("schema_version") or "")[:20],
        "current_state": str(data.get("current_state") or "")[:60],
        "previous_state": str(data.get("previous_state") or "")[:60],
        "failure_reason_code": str(data.get("failure_reason_code") or "")[:80],
        "occurred_at": str(data.get("occurred_at") or "")[:40],
        "sent_at": str(event.get("sent_at") or "")[:40],
    }
    with _conn() as conn:
        first = conn.execute(
            """
            INSERT INTO wise_webhook_inbox (
                delivery_id, event_type, transfer_id, payload, outcome, processed_at
            )
            VALUES (
                %(delivery)s, %(event_type)s, %(transfer_id)s, %(payload)s::jsonb, %(outcome)s,
                CASE WHEN %(outcome)s = 'pending' THEN NULL ELSE clock_timestamp() END
            )
            ON CONFLICT (delivery_id) DO NOTHING
            RETURNING delivery_id
            """,
            {
                "delivery": delivery,
                "event_type": event_type,
                "transfer_id": transfer_id or None,
                "payload": json.dumps(summary),
                "outcome": outcome,
            },
        ).fetchone()
    return {"ok": True, "outcome": outcome if first is not None else "duplicate"}


def drain_webhook_inbox(*, limit: int = 50, client: WiseClient | None = None) -> dict:
    """Apply recorded deliveries by reading each transfer back from Wise."""

    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT i.delivery_id, p.id AS payout_id
              FROM wise_webhook_inbox i
              LEFT JOIN wise_payouts p ON p.wise_transfer_id = i.transfer_id
             WHERE i.outcome = 'pending'
             ORDER BY i.received_at ASC
             LIMIT %s
            """,
            (max(1, min(int(limit), 200)),),
        ).fetchall()
    if not rows:
        return {"applied": 0, "ignored": 0, "deferred": 0}
    counts = {"applied": 0, "ignored": 0, "deferred": 0}
    owns_client = client is None
    wise: WiseClient | None = client
    try:
        for row in rows:
            if row["payout_id"] is None:
                outcome = "ignored"  # a transfer this rail did not create
            else:
                try:
                    if wise is None:
                        wise = WiseClient()
                    sync_payout(payout_id=int(row["payout_id"]), client=wise)
                    outcome = "applied"
                except WiseError as exc:
                    log.warning("wise inbox payout=%s: %s", row["payout_id"], exc.reason)
                    counts["deferred"] += 1
                    continue  # stays pending; the next pass tries again
            counts[outcome] += 1
            with _conn() as conn:
                conn.execute(
                    "UPDATE wise_webhook_inbox SET outcome = %s, processed_at = clock_timestamp() "
                    "WHERE delivery_id = %s AND outcome = 'pending'",
                    (outcome, row["delivery_id"]),
                )
    finally:
        if owns_client and wise is not None:
            wise.close()
    return counts


def run_cycle() -> dict:
    """One background pass: apply deliveries, poll open payouts, send approved."""

    if not wise_configured():
        return {"reason": "not_configured"}
    with WiseClient() as wise:
        return {
            "inbox": drain_webhook_inbox(client=wise),
            "reconciled": reconcile_open_payouts(client=wise),
            "executed": execute_approved_payouts(client=wise),
        }


# ── Operator view ────────────────────────────────────────────────────


def rail_status(*, client: WiseClient | None = None) -> dict:
    """What an admin needs before approving: switches, caps, balance, queue."""

    status: dict[str, Any] = {
        "configured": wise_configured(),
        "enabled": payouts_enabled(),
        "environment": wise_env(),
        "auto_fund": auto_fund_enabled(),
        "min_payout_cad": min_payout_micros() / MICROS_PER_CAD,
        "max_payout_cad": max_payout_micros() / MICROS_PER_CAD,
        "daily_cap_cad": daily_cap_micros() / MICROS_PER_CAD,
        "cooloff_hours": cooloff_hours(),
        "email_recipients": email_recipients_enabled(),
        "target_currencies": sorted(allowed_target_currencies()),
    }
    with _conn() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*)::integer AS n, COALESCE(SUM(amount_micros), 0)::bigint AS micros "
            "FROM wise_payouts GROUP BY status"
        ).fetchall()
    status["payouts"] = {
        row["status"]: {"count": row["n"], "cad": int(row["micros"]) / MICROS_PER_CAD}
        for row in rows
    }
    if status["configured"]:
        owns_client = client is None
        try:
            wise = client or WiseClient()
            try:
                status["balance_cad"] = wise.available_micros("CAD") / MICROS_PER_CAD
            finally:
                if owns_client:
                    wise.close()
        except WiseError as exc:
            status["balance_error"] = exc.reason
    return status
