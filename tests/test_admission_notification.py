"""Tests for provider admission failure notification pipeline.

Covers:
- _notify_provider_admission_failure throttle logic
- Notification content (type, priority, reasons)
- Edge cases: missing owner, missing email, unknown user
- Throttle expiry after 1 hour
"""

import os
import time
import tempfile

import pytest

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_admit_test_")
_tmpdir = _tmp_ctx.name
os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from routes.agent import _notify_provider_admission_failure, _admission_notified


def _create_test_user(email="provider@xcelsior.ca", user_id=None):
    from db import UserStore
    if user_id is None:
        user_id = email  # use email as user_id for simplicity
    try:
        UserStore.create_user({"email": email, "user_id": user_id, "name": "Test Provider", "role": "provider"})
    except Exception:
        pass  # already exists
    return user_id


@pytest.fixture(autouse=True)
def clean_throttle():
    _admission_notified.clear()
    yield


class TestAdmissionNotification:
    """Test _notify_provider_admission_failure logic."""

    def test_creates_notification_for_owner(self):
        email = _create_test_user()
        host = {"host_id": "h-admit-1", "owner": email, "gpu_model": "RTX 4090"}
        details = {"admitted": False, "rejection_reasons": ["CUDA version too old (11.8 < 12.0)"]}

        _notify_provider_admission_failure(host, details)

        from db import NotificationStore
        notifs = NotificationStore.list_for_user(email, limit=10)
        admit_notifs = [n for n in notifs if n.get("type") == "host_admission_failed"]
        assert len(admit_notifs) >= 1
        assert "RTX 4090" in admit_notifs[0]["title"]
        assert "CUDA" in admit_notifs[0]["body"]
        assert admit_notifs[0].get("priority") == 2  # critical

    def test_throttle_within_1_hour(self):
        email = _create_test_user("throttle-provider@test.ca")
        host = {"host_id": "h-throttle-admit", "owner": email, "gpu_model": "RTX 4090"}
        details = {"admitted": False, "rejection_reasons": ["stale driver"]}

        _notify_provider_admission_failure(host, details)
        from db import NotificationStore
        count1 = len([n for n in NotificationStore.list_for_user(email, limit=50) if n.get("type") == "host_admission_failed"])

        # Second call within 1 hour — should be throttled
        _notify_provider_admission_failure(host, details)
        count2 = len([n for n in NotificationStore.list_for_user(email, limit=50) if n.get("type") == "host_admission_failed"])
        assert count2 == count1

    def test_throttle_expires_after_1_hour(self):
        email = _create_test_user("expire-provider@test.ca")
        host = {"host_id": "h-expire-admit", "owner": email, "gpu_model": "RTX 4090"}
        details = {"admitted": False, "rejection_reasons": ["old runtime"]}

        _notify_provider_admission_failure(host, details)
        from db import NotificationStore
        count1 = len([n for n in NotificationStore.list_for_user(email, limit=50) if n.get("type") == "host_admission_failed"])

        # Simulate passage of 1 hour by backdating the throttle entry
        _admission_notified["h-expire-admit"] = time.time() - 3601

        _notify_provider_admission_failure(host, details)
        count2 = len([n for n in NotificationStore.list_for_user(email, limit=50) if n.get("type") == "host_admission_failed"])
        assert count2 == count1 + 1

    def test_no_crash_on_missing_owner(self):
        host = {"host_id": "h-no-owner", "gpu_model": "RTX 4090"}
        details = {"admitted": False, "rejection_reasons": ["test"]}
        # Should not raise
        _notify_provider_admission_failure(host, details)

    def test_no_crash_on_empty_owner(self):
        host = {"host_id": "h-empty-owner", "owner": "", "gpu_model": "RTX 4090"}
        details = {"admitted": False, "rejection_reasons": ["test"]}
        _notify_provider_admission_failure(host, details)

    def test_no_crash_on_unknown_user_id(self):
        host = {"host_id": "h-unknown", "owner": "nonexistent@test.ca", "gpu_model": "RTX 4090"}
        details = {"admitted": False, "rejection_reasons": ["test"]}
        _notify_provider_admission_failure(host, details)

    def test_multiple_rejection_reasons_in_body(self):
        email = _create_test_user("multi-reason@test.ca")
        host = {"host_id": "h-multi", "owner": email, "gpu_model": "A100"}
        details = {"admitted": False, "rejection_reasons": ["CUDA too old", "runc unpatched", "toolkit outdated"]}

        _notify_provider_admission_failure(host, details)

        from db import NotificationStore
        notifs = [n for n in NotificationStore.list_for_user(email, limit=10) if n.get("type") == "host_admission_failed"]
        assert len(notifs) >= 1
        body = notifs[0]["body"]
        assert "CUDA too old" in body
        assert "runc unpatched" in body
        assert "toolkit outdated" in body

    def test_action_url_points_to_hosts(self):
        email = _create_test_user("action-url@test.ca")
        host = {"host_id": "h-action", "owner": email, "gpu_model": "RTX 4090"}
        details = {"admitted": False, "rejection_reasons": ["test"]}

        _notify_provider_admission_failure(host, details)

        from db import NotificationStore
        notifs = [n for n in NotificationStore.list_for_user(email, limit=10) if n.get("type") == "host_admission_failed"]
        assert notifs[0].get("action_url") == "/dashboard/hosts"

    def test_notification_data_has_host_id(self):
        email = _create_test_user("data-check@test.ca")
        host = {"host_id": "h-data-check", "owner": email, "gpu_model": "H100"}
        details = {"admitted": False, "rejection_reasons": ["outdated"], "recommended_runtime": "nvidia"}

        _notify_provider_admission_failure(host, details)

        from db import NotificationStore
        notifs = [n for n in NotificationStore.list_for_user(email, limit=10) if n.get("type") == "host_admission_failed"]
        data = notifs[0].get("data", {})
        assert data.get("host_id") == "h-data-check"
        assert data.get("recommended_runtime") == "nvidia"

    def test_uses_hostname_when_no_gpu_model(self):
        email = _create_test_user("hostname-test@test.ca")
        host = {"host_id": "h-hostname", "owner": email, "hostname": "my-server"}
        details = {"admitted": False, "rejection_reasons": ["test"]}

        _notify_provider_admission_failure(host, details)

        from db import NotificationStore
        notifs = [n for n in NotificationStore.list_for_user(email, limit=10) if n.get("type") == "host_admission_failed"]
        assert "my-server" in notifs[0]["title"]
