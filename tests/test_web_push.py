import importlib

import pytest


class _DummyResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _DummyWebPushException(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"web push error {status_code}")
        self.response = _DummyResponse(status_code)


def _load_web_push_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XCELSIOR_WEB_PUSH_VAPID_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("XCELSIOR_WEB_PUSH_VAPID_PRIVATE_KEY", "test-private")
    monkeypatch.setenv("XCELSIOR_WEB_PUSH_VAPID_SUBJECT", "mailto:hello@xcelsior.ca")
    import web_push

    module = importlib.reload(web_push)
    module.reset_web_push_runtime_metrics()
    return module


def test_deliver_web_push_records_success_and_revokes_gone_endpoint(monkeypatch: pytest.MonkeyPatch):
    module = _load_web_push_module(monkeypatch)
    import db

    touched: list[str] = []
    revoked: list[str] = []

    class FakeWebPushSubscriptionStore:
        @staticmethod
        def list_active_for_user(_user_email: str):
            return [
                {
                    "endpoint": "https://push.example.test/ok",
                    "p256dh": "p256dh-ok",
                    "auth": "auth-ok",
                },
                {
                    "endpoint": "https://push.example.test/gone",
                    "p256dh": "p256dh-gone",
                    "auth": "auth-gone",
                },
            ]

        @staticmethod
        def touch(endpoint: str):
            touched.append(endpoint)

        @staticmethod
        def revoke_endpoint(endpoint: str):
            revoked.append(endpoint)
            return True

        @staticmethod
        def count_active():
            return 1

        @staticmethod
        def count_revoked():
            return 1

        @staticmethod
        def count_stale_active(_days: int):
            return 0

    def fake_webpush(*, subscription_info, **_kwargs):
        endpoint = subscription_info["endpoint"]
        if endpoint.endswith("/gone"):
            raise _DummyWebPushException(410)
        return None

    monkeypatch.setattr(db, "WebPushSubscriptionStore", FakeWebPushSubscriptionStore)
    monkeypatch.setattr(module, "WebPushException", _DummyWebPushException)
    monkeypatch.setattr(module, "webpush", fake_webpush)

    module.deliver_web_push_notification(
        "user@xcelsior.ca",
        {
            "id": "notif-1",
            "type": "instance",
            "title": "Instance ready",
            "body": "Your instance is live.",
            "action_url": "/dashboard/instances/job-1",
        },
    )

    metrics = module.get_web_push_runtime_metrics()
    assert metrics["delivery_attempts_total"] == 2
    assert metrics["delivery_success_total"] == 1
    assert metrics["delivery_failure_total"] == 1
    assert metrics["delivery_revoked_total"] == 1
    assert metrics["last_failure_status_code"] == 410
    assert touched == ["https://push.example.test/ok"]
    assert revoked == ["https://push.example.test/gone"]


def test_web_push_observability_snapshot_merges_runtime_and_store_counts(monkeypatch: pytest.MonkeyPatch):
    module = _load_web_push_module(monkeypatch)
    import db

    class FakeWebPushSubscriptionStore:
        @staticmethod
        def list_active_for_user(_user_email: str):
            return []

        @staticmethod
        def count_active():
            return 4

        @staticmethod
        def count_revoked():
            return 2

        @staticmethod
        def count_stale_active(days: int):
            assert days == 45
            return 1

    monkeypatch.setattr(db, "WebPushSubscriptionStore", FakeWebPushSubscriptionStore)
    monkeypatch.setattr(module, "webpush", lambda **_kwargs: None)

    module.deliver_web_push_notification(
        "user@xcelsior.ca",
        {
            "id": "notif-2",
            "type": "billing",
            "title": "Wallet low",
        },
    )

    snapshot = module.get_web_push_observability_snapshot(stale_after_days=45)
    assert snapshot["configured"] == 1
    assert snapshot["active_subscriptions"] == 4
    assert snapshot["revoked_subscriptions"] == 2
    assert snapshot["stale_subscriptions"] == 1
    assert snapshot["deliveries_skipped_no_subscriptions_total"] == 1
