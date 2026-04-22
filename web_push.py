"""Web push delivery for desktop and installed PWA notifications."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

try:
    from pywebpush import WebPushException, webpush
except ImportError:  # pragma: no cover - dependency rollout safety
    WebPushException = Exception  # type: ignore[assignment]
    webpush = None

log = logging.getLogger("xcelsior.webpush")

VAPID_PUBLIC_KEY = os.environ.get("XCELSIOR_WEB_PUSH_VAPID_PUBLIC_KEY", "").strip()
VAPID_PRIVATE_KEY = os.environ.get("XCELSIOR_WEB_PUSH_VAPID_PRIVATE_KEY", "").strip()
VAPID_SUBJECT = os.environ.get(
    "XCELSIOR_WEB_PUSH_VAPID_SUBJECT", "mailto:hello@xcelsior.ca"
).strip()
WEB_PUSH_STALE_AFTER_DAYS = int(os.environ.get("XCELSIOR_WEB_PUSH_STALE_AFTER_DAYS", "30"))

_runtime_metrics_lock = threading.Lock()


def _new_runtime_metrics() -> dict[str, float | int]:
    return {
        "delivery_attempts_total": 0,
        "delivery_success_total": 0,
        "delivery_failure_total": 0,
        "delivery_revoked_total": 0,
        "deliveries_skipped_unconfigured_total": 0,
        "deliveries_skipped_no_subscriptions_total": 0,
        "last_delivery_attempt_at": 0.0,
        "last_delivery_success_at": 0.0,
        "last_delivery_failure_at": 0.0,
        "last_failure_status_code": 0,
    }


_runtime_metrics = _new_runtime_metrics()


def is_web_push_configured() -> bool:
    return bool(webpush is not None and VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY and VAPID_SUBJECT)


def get_web_push_public_key() -> str:
    return VAPID_PUBLIC_KEY


def reset_web_push_runtime_metrics() -> None:
    global _runtime_metrics
    with _runtime_metrics_lock:
        _runtime_metrics = _new_runtime_metrics()


def get_web_push_runtime_metrics() -> dict[str, float | int]:
    with _runtime_metrics_lock:
        return dict(_runtime_metrics)


def _record_runtime_metric(
    metric: str,
    *,
    delta: int = 0,
    timestamp_field: str | None = None,
    timestamp: float | None = None,
    last_failure_status_code: int | None = None,
) -> None:
    ts = time.time() if timestamp is None else timestamp
    with _runtime_metrics_lock:
        if delta:
            _runtime_metrics[metric] = int(_runtime_metrics.get(metric, 0)) + delta
        if timestamp_field:
            _runtime_metrics[timestamp_field] = float(ts)
        if last_failure_status_code is not None:
            _runtime_metrics["last_failure_status_code"] = int(last_failure_status_code)


def get_web_push_observability_snapshot(
    stale_after_days: int | None = None,
) -> dict[str, float | int]:
    snapshot: dict[str, float | int] = {
        "configured": 1 if is_web_push_configured() else 0,
        "active_subscriptions": 0,
        "revoked_subscriptions": 0,
        "stale_subscriptions": 0,
        **get_web_push_runtime_metrics(),
    }

    try:
        from db import WebPushSubscriptionStore

        stale_window = max(stale_after_days or WEB_PUSH_STALE_AFTER_DAYS, 1)
        snapshot["active_subscriptions"] = WebPushSubscriptionStore.count_active()
        snapshot["revoked_subscriptions"] = WebPushSubscriptionStore.count_revoked()
        snapshot["stale_subscriptions"] = WebPushSubscriptionStore.count_stale_active(stale_window)
    except Exception as exc:
        log.debug("Web push observability snapshot degraded: %s", exc)

    return snapshot


def _notification_url(notification: dict[str, Any]) -> str:
    action_url = notification.get("action_url")
    if isinstance(action_url, str) and action_url:
        return action_url
    return "/dashboard/notifications"


def _notification_payload(notification: dict[str, Any]) -> str:
    return json.dumps(
        {
            "title": notification.get("title") or "Xcelsior",
            "body": notification.get("body") or "You have a new Xcelsior notification.",
            "tag": f"xcelsior-{notification.get('type') or 'notification'}",
            "url": _notification_url(notification),
            "data": {
                "notification_id": notification.get("id"),
                "type": notification.get("type"),
                "entity_type": notification.get("entity_type"),
                "entity_id": notification.get("entity_id"),
                "priority": notification.get("priority", 0),
                **(notification.get("data") or {}),
            },
        }
    )


def _status_code(exc: WebPushException) -> int | None:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def deliver_web_push_notification(user_email: str, notification: dict[str, Any]) -> None:
    if not is_web_push_configured():
        _record_runtime_metric("deliveries_skipped_unconfigured_total", delta=1)
        return

    from db import WebPushSubscriptionStore

    subscriptions = WebPushSubscriptionStore.list_active_for_user(user_email)
    if not subscriptions:
        _record_runtime_metric("deliveries_skipped_no_subscriptions_total", delta=1)
        return

    payload = _notification_payload(notification)
    vapid_claims = {"sub": VAPID_SUBJECT}

    for subscription in subscriptions:
        endpoint = subscription["endpoint"]
        _record_runtime_metric(
            "delivery_attempts_total",
            delta=1,
            timestamp_field="last_delivery_attempt_at",
        )
        try:
            if webpush is None:
                return
            webpush(
                subscription_info={
                    "endpoint": endpoint,
                    "keys": {
                        "p256dh": subscription["p256dh"],
                        "auth": subscription["auth"],
                    },
                },
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims=vapid_claims,
                ttl=3600,
            )
            WebPushSubscriptionStore.touch(endpoint)
            _record_runtime_metric(
                "delivery_success_total",
                delta=1,
                timestamp_field="last_delivery_success_at",
            )
        except WebPushException as exc:
            status_code = _status_code(exc)
            _record_runtime_metric(
                "delivery_failure_total",
                delta=1,
                timestamp_field="last_delivery_failure_at",
                last_failure_status_code=status_code or 0,
            )
            if status_code in {404, 410}:
                WebPushSubscriptionStore.revoke_endpoint(endpoint)
                _record_runtime_metric("delivery_revoked_total", delta=1)
                continue

            log.warning(
                "Web push delivery failed for %s (%s): %s",
                user_email,
                status_code or "unknown",
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            _record_runtime_metric(
                "delivery_failure_total",
                delta=1,
                timestamp_field="last_delivery_failure_at",
                last_failure_status_code=0,
            )
            log.warning("Unexpected web push delivery failure for %s: %s", user_email, exc)
