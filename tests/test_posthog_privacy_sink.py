from __future__ import annotations

import privacy_sinks


def test_posthog_deletion_uses_private_api_host(monkeypatch):
    monkeypatch.setenv("NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN", "phc_test")
    monkeypatch.setenv("NEXT_PUBLIC_POSTHOG_HOST", "https://us.i.posthog.com")
    monkeypatch.setenv("XCELSIOR_POSTHOG_PERSONAL_API_KEY", "phx_test")
    monkeypatch.setenv("XCELSIOR_POSTHOG_PROJECT_ID", "12345")
    monkeypatch.setenv("XCELSIOR_POSTHOG_API_HOST", "https://us.posthog.com")
    requests: list[dict[str, object]] = []

    def fake_request(method, path, **kwargs):
        requests.append({"method": method, "path": path, **kwargs})
        if path.endswith("/persons/"):
            return {"results": []}
        return {
            "persons_found": 0,
            "persons_deleted": 0,
            "events_queued_for_deletion": False,
            "recordings_queued_for_deletion": False,
            "deletion_errors": [],
        }

    monkeypatch.setattr(privacy_sinks, "_posthog_request", fake_request)
    outcome = privacy_sinks.delete_posthog_subject(
        {
            "subject_user_id": "user-123",
            "subject_email": "user@example.test",
            "subject_customer_ids": [],
        },
        {},
    )

    assert outcome.status == "completed"
    assert requests
    assert {request["base_url"] for request in requests} == {
        "https://us.posthog.com"
    }
    assert all(
        ".i.posthog.com" not in str(request["base_url"])
        for request in requests
    )
