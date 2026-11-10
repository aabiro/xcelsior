"""Routes: Wise provider payouts.

Three audiences, three levels of trust:

* A **provider** may see their own Wise status and, when email recipients are
  enabled, set where they are paid. Setting a destination needs a person at a
  first-party session: an API key or a connector token may not redirect money.
* A **platform admin**, also a person at a first-party session, links
  destinations, approves, rejects and releases payouts. Every one of those is
  written to the audit log with who did it.
* **Wise** posts webhooks. The signature is the credential.

No route here takes an amount. See ``billing/wise_payouts.py``.
"""

import logging

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

import wise_payouts as wise
from routes._deps import (
    _is_interactive_human,
    _is_platform_admin,
    _require_auth,
    _require_scope,
    append_user_audit_event,
)
from routes.providers import _require_provider_access

router = APIRouter()
log = logging.getLogger("xcelsior.wise")


def _raise(exc: wise.WiseError):
    raise HTTPException(exc.status_code, {"code": exc.reason, "message": str(exc)}) from exc


#: Credentials that are a secret in a file or an environment, not a person at
#: a session. `_is_interactive_human` already excludes client-credentials
#: tokens, agent keys and connector tokens; a user API key and the static
#: master token are added here because either one, once leaked, would
#: otherwise be enough to redirect or approve a payout.
_STATIC_CREDENTIALS = frozenset({"master_token", "api_key", "agent_api_key"})


def _person_present(user: dict) -> bool:
    return user.get("auth_type") not in _STATIC_CREDENTIALS and _is_interactive_human(user)


def _require_wise_admin(request: Request) -> dict:
    """A human platform admin at a first-party session. No machine principal,
    agent key or connector token may approve or direct a payout."""

    user = _require_auth(request)
    if not _is_platform_admin(user) or not _person_present(user):
        raise HTTPException(403, "A platform admin signed in to the dashboard is required")
    return user


def _actor(user: dict) -> str:
    return str(user.get("user_id") or user.get("email") or "unknown")


def _notify_destination_change(provider_id: str, hint: str, payable_after) -> None:
    """Tell the provider, at the address on file, that the destination changed.

    This is the control that lets the real owner notice a change they did not
    make while the cool-off still holds the money.
    """

    try:
        from scheduler import send_email
        from stripe_connect import get_stripe_manager

        provider = get_stripe_manager().get_provider(provider_id) or {}
        email = str(provider.get("email") or "").strip()
        if not email:
            return
        send_email(
            "Your Wise payout destination was changed",
            "The Wise destination for your Xcelsior payouts was set to "
            f"{hint}.\n\nPayouts to it start after {payable_after}.\n\n"
            "If you did not make this change, reply to this email or contact "
            "support@xcelsior.ca now. No payout will be sent to the new "
            "destination before that time.",
            to_email=email,
        )
    except Exception as exc:  # the change is already committed and audited
        log.warning("wise destination notice failed provider=%s: %s", provider_id, exc)


# ── Provider ─────────────────────────────────────────────────────────


class WiseEmailRecipientRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    currency: str = Field(min_length=3, max_length=3)
    account_holder_name: str = Field(min_length=2, max_length=140)
    legal_type: str = Field(default="PRIVATE", max_length=16)


@router.get("/api/providers/{provider_id}/wise", tags=["Providers"])
def api_provider_wise_status(provider_id: str, request: Request):
    """Wise payout status: destination hint, cool-off, queued earnings."""
    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:read")
    status = wise.get_recipient_status(provider_id)
    status["self_service"] = wise.email_recipients_enabled()
    return {"ok": True, "wise": status}


@router.post("/api/providers/{provider_id}/wise/recipient", tags=["Providers"])
def api_provider_wise_set_recipient(
    provider_id: str, body: WiseEmailRecipientRequest, request: Request
):
    """Be paid to the email address of a Wise account. Starts a cool-off."""
    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:write")
    if not _person_present(user):
        raise HTTPException(403, "Sign in to the dashboard to change where payouts are sent")
    try:
        result = wise.set_email_recipient(
            provider_id=provider_id,
            email=body.email,
            currency=body.currency,
            holder_name=body.account_holder_name,
            legal_type=body.legal_type,
            actor=f"user:{_actor(user)}",
        )
    except wise.WiseError as exc:
        _raise(exc)
    recipient = result["recipient"]
    if result["changed"]:
        append_user_audit_event(
            "provider.wise.destination_set",
            "provider",
            provider_id,
            user,
            data={
                "hint": recipient["display_hint"],
                "currency": recipient["target_currency"],
                "first": result["first"],
                "payouts_enabled_after": str(recipient["payouts_enabled_after"]),
            },
        )
        _notify_destination_change(
            provider_id, recipient["display_hint"], recipient["payouts_enabled_after"]
        )
    return {"ok": True, "recipient": recipient, "changed": result["changed"]}


@router.post("/api/providers/{provider_id}/wise/disconnect", tags=["Providers"])
def api_provider_wise_disconnect(provider_id: str, request: Request):
    """Stop Wise payouts for this provider. Queued earnings stay queued."""
    user = _require_provider_access(request, provider_id)
    _require_scope(user, "providers:write")
    if not _person_present(user):
        raise HTTPException(403, "Sign in to the dashboard to change where payouts are sent")
    try:
        result = wise.disable_recipient(provider_id=provider_id, actor=f"user:{_actor(user)}")
    except wise.WiseError as exc:
        _raise(exc)
    append_user_audit_event(
        "provider.wise.destination_removed", "provider", provider_id, user, data=result
    )
    return {"ok": True, **result}


# ── Admin ────────────────────────────────────────────────────────────


class WiseLinkRecipientRequest(BaseModel):
    recipient_id: int = Field(gt=0)


class WiseRejectRequest(BaseModel):
    reason: str = Field(default="", max_length=300)


class WiseReleaseRequest(BaseModel):
    confirm_no_transfer: bool = False


@router.get("/api/admin/wise/status", tags=["Admin"])
def api_admin_wise_status(request: Request):
    """Switches, caps, Wise balance and payout counts."""
    _require_wise_admin(request)
    try:
        return {"ok": True, "wise": wise.rail_status()}
    except wise.WiseError as exc:
        _raise(exc)


@router.get("/api/admin/wise/payouts", tags=["Admin"])
def api_admin_wise_payouts(
    request: Request,
    status: str | None = Query(default=None, max_length=32),
    limit: int = Query(default=100, ge=1, le=500),
):
    _require_wise_admin(request)
    return {"ok": True, "payouts": wise.list_payouts(status=status, limit=limit)}


@router.post("/api/admin/wise/providers/{provider_id}/recipient", tags=["Admin"])
def api_admin_wise_link_recipient(
    provider_id: str, body: WiseLinkRecipientRequest, request: Request
):
    """Bind a recipient that exists in the Wise account to a provider."""
    user = _require_wise_admin(request)
    from stripe_connect import get_stripe_manager

    if not get_stripe_manager().get_provider(provider_id):
        raise HTTPException(404, f"Provider {provider_id} not found")
    try:
        result = wise.link_recipient(
            provider_id=provider_id,
            recipient_id=body.recipient_id,
            actor=f"admin:{_actor(user)}",
        )
    except wise.WiseError as exc:
        _raise(exc)
    recipient = result["recipient"]
    if result["changed"]:
        append_user_audit_event(
            "admin.wise.destination_linked",
            "provider",
            provider_id,
            user,
            data={
                "wise_recipient_id": body.recipient_id,
                "hint": recipient["display_hint"],
                "currency": recipient["target_currency"],
                "payouts_enabled_after": str(recipient["payouts_enabled_after"]),
            },
        )
        _notify_destination_change(
            provider_id, recipient["display_hint"], recipient["payouts_enabled_after"]
        )
    return {"ok": True, "recipient": recipient, "changed": result["changed"]}


@router.post("/api/admin/wise/payouts/batch", tags=["Admin"])
def api_admin_wise_batch(request: Request, provider_id: str | None = None):
    """Add queued Wise settlements into pending payouts. Moves no money."""
    user = _require_wise_admin(request)
    result = wise.batch_queued_settlements(provider_id=provider_id)
    append_user_audit_event("admin.wise.batched", "wise_payouts", "batch", user, data=result)
    return {"ok": True, **result}


@router.post("/api/admin/wise/payouts/{payout_id}/approve", tags=["Admin"])
def api_admin_wise_approve(payout_id: int, request: Request):
    """Approve one payout. The transfer is created on the next worker pass."""
    user = _require_wise_admin(request)
    try:
        payout = wise.approve_payout(payout_id=payout_id, approver=_actor(user))
    except wise.WiseError as exc:
        append_user_audit_event(
            "admin.wise.approval_refused",
            "wise_payout",
            str(payout_id),
            user,
            data={"reason": exc.reason},
        )
        _raise(exc)
    append_user_audit_event(
        "admin.wise.payout_approved",
        "wise_payout",
        str(payout_id),
        user,
        data={
            "provider_id": payout["provider_id"],
            "amount_micros": payout["amount_micros"],
            "target_currency": payout["target_currency"],
        },
    )
    return {"ok": True, "payout": payout}


@router.post("/api/admin/wise/payouts/{payout_id}/reject", tags=["Admin"])
def api_admin_wise_reject(payout_id: int, body: WiseRejectRequest, request: Request):
    user = _require_wise_admin(request)
    try:
        payout = wise.reject_payout(payout_id=payout_id, actor=_actor(user), reason=body.reason)
    except wise.WiseError as exc:
        _raise(exc)
    append_user_audit_event(
        "admin.wise.payout_rejected",
        "wise_payout",
        str(payout_id),
        user,
        data={"reason": body.reason},
    )
    return {"ok": True, "payout": payout}


@router.post("/api/admin/wise/payouts/{payout_id}/release", tags=["Admin"])
def api_admin_wise_release(payout_id: int, body: WiseReleaseRequest, request: Request):
    """Return a payout's rows to the queue after its transfer came back."""
    user = _require_wise_admin(request)
    try:
        payout = wise.release_payout(
            payout_id=payout_id,
            actor=_actor(user),
            confirm_no_transfer=body.confirm_no_transfer,
        )
    except wise.WiseError as exc:
        _raise(exc)
    append_user_audit_event(
        "admin.wise.payout_released",
        "wise_payout",
        str(payout_id),
        user,
        data={"confirm_no_transfer": body.confirm_no_transfer},
    )
    return {"ok": True, "payout": payout}


@router.post("/api/admin/wise/payouts/{payout_id}/sync", tags=["Admin"])
def api_admin_wise_sync(payout_id: int, request: Request):
    """Read the transfer's state back from Wise now."""
    _require_wise_admin(request)
    try:
        return {"ok": True, "payout": wise.sync_payout(payout_id=payout_id)}
    except wise.WiseError as exc:
        _raise(exc)


# ── Webhook ──────────────────────────────────────────────────────────

_MAX_WEBHOOK_BYTES = 64 * 1024


@router.post("/api/billing/wise/webhook", tags=["Billing"])
async def api_wise_webhook(request: Request):
    """Wise webhook receiver. Verifies, records, and answers; applies nothing."""
    if not wise.wise_configured():
        raise HTTPException(503, "Wise is not configured")
    body = await request.body()
    if len(body) > _MAX_WEBHOOK_BYTES:
        raise HTTPException(413, "Webhook body too large")
    try:
        result = wise.handle_webhook(
            body=body,
            signature=request.headers.get("x-signature-sha256", ""),
            delivery_id=request.headers.get("x-delivery-id", ""),
            test_notification=request.headers.get("x-test-notification", "").lower() == "true",
        )
    except wise.WiseError as exc:
        _raise(exc)
    return result
