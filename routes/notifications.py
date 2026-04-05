"""Routes: notifications."""

from fastapi import APIRouter, HTTPException, Request

from routes._deps import (
    _get_current_user,
)
from db import NotificationStore

router = APIRouter()

@router.get("/api/notifications", tags=["Notifications"])
def api_list_notifications(request: Request, unread: bool = False, limit: int = 50):
    """List notifications for the current user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    notifications = NotificationStore.list_for_user(user["email"], unread_only=unread, limit=limit)
    unread_count = NotificationStore.unread_count(user["email"])
    return {"ok": True, "notifications": notifications, "unread_count": unread_count}

@router.get("/api/notifications/unread-count", tags=["Notifications"])
def api_notification_unread_count(request: Request):
    """Get the unread notification count for the current user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    return {"ok": True, "unread_count": NotificationStore.unread_count(user["email"])}

@router.post("/api/notifications/{notification_id}/read", tags=["Notifications"])
def api_mark_notification_read(request: Request, notification_id: str):
    """Mark a single notification as read."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ok = NotificationStore.mark_read(notification_id, user["email"])
    if not ok:
        raise HTTPException(404, "Notification not found")
    return {"ok": True}

@router.post("/api/notifications/read-all", tags=["Notifications"])
def api_mark_all_read(request: Request):
    """Mark all notifications as read for the current user."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    count = NotificationStore.mark_all_read(user["email"])
    return {"ok": True, "marked": count}

@router.delete("/api/notifications/{notification_id}", tags=["Notifications"])
def api_delete_notification(request: Request, notification_id: str):
    """Delete a single notification."""
    user = _get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    ok = NotificationStore.delete(notification_id, user["email"])
    if not ok:
        raise HTTPException(404, "Notification not found")
    return {"ok": True}

