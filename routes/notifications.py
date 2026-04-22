"""Routes: notifications."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from routes._deps import (
    _get_current_user,
)
from db import NotificationStore, WebPushSubscriptionStore
from web_push import get_web_push_public_key, is_web_push_configured

router = APIRouter()


class WebPushKeys(BaseModel):
    p256dh: str = Field(min_length=1)
    auth: str = Field(min_length=1)


class WebPushSubscriptionPayload(BaseModel):
    endpoint: str = Field(min_length=1)
    expirationTime: int | None = None
    keys: WebPushKeys


class WebPushSubscriptionDeletePayload(BaseModel):
    endpoint: str = Field(min_length=1)


@router.get("/api/notifications", tags=["Notifications"])
def api_list_notifications(request: Request, unread: bool = False, limit: int = 50):
    """List notifications for the current user."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "notifications:read")
    notifications = NotificationStore.list_for_user(user["email"], unread_only=unread, limit=limit)
    unread_count = NotificationStore.unread_count(user["email"])
    return {"ok": True, "notifications": notifications, "unread_count": unread_count}


@router.get("/api/notifications/unread-count", tags=["Notifications"])
def api_notification_unread_count(request: Request):
    """Get the unread notification count for the current user."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "notifications:read")
    return {"ok": True, "unread_count": NotificationStore.unread_count(user["email"])}


@router.post("/api/notifications/{notification_id}/read", tags=["Notifications"])
def api_mark_notification_read(request: Request, notification_id: str):
    """Mark a single notification as read."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "notifications:write")
    ok = NotificationStore.mark_read(notification_id, user["email"])
    if not ok:
        raise HTTPException(404, "Notification not found")
    return {"ok": True}


@router.post("/api/notifications/read-all", tags=["Notifications"])
def api_mark_all_read(request: Request):
    """Mark all notifications as read for the current user."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "notifications:write")
    count = NotificationStore.mark_all_read(user["email"])
    return {"ok": True, "marked": count}


@router.delete("/api/notifications/{notification_id}", tags=["Notifications"])
def api_delete_notification(request: Request, notification_id: str):
    """Delete a single notification."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "notifications:write")
    ok = NotificationStore.delete(notification_id, user["email"])
    if not ok:
        raise HTTPException(404, "Notification not found")
    return {"ok": True}


@router.get("/api/notifications/push/subscription", tags=["Notifications"])
def api_get_push_subscription_status(request: Request):
    """Get push subscription capability for the current user."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "notifications:read")
    configured = is_web_push_configured()
    return {
        "ok": True,
        "configured": configured,
        "vapid_public_key": get_web_push_public_key() if configured else "",
        "active_subscription_count": WebPushSubscriptionStore.count_active_for_user(user["email"]),
    }


@router.post("/api/notifications/push/subscription", tags=["Notifications"])
def api_upsert_push_subscription(request: Request, body: WebPushSubscriptionPayload):
    """Create or refresh the current browser's push subscription."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "notifications:write")
    if not is_web_push_configured():
        raise HTTPException(503, "Web push notifications are not configured")

    subscription_id = WebPushSubscriptionStore.upsert(
        user["email"],
        body.endpoint,
        body.keys.p256dh,
        body.keys.auth,
        user_agent=request.headers.get("user-agent", "")[:512],
    )
    return {"ok": True, "subscription_id": subscription_id}


@router.delete("/api/notifications/push/subscription", tags=["Notifications"])
def api_delete_push_subscription(request: Request, body: WebPushSubscriptionDeletePayload):
    """Revoke the current browser's push subscription."""
    from routes._deps import _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    _require_scope(user, "notifications:write")

    revoked = WebPushSubscriptionStore.revoke(user["email"], body.endpoint)
    return {"ok": True, "revoked": revoked}
