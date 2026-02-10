import worker_agent


def test_validate_config_allows_missing_api_token(monkeypatch):
    monkeypatch.setattr(worker_agent, "HOST_ID", "rig-01")
    monkeypatch.setattr(worker_agent, "SCHEDULER_URL", "http://localhost:8000")
    monkeypatch.setattr(worker_agent, "API_TOKEN", None)

    # Should not exit even without API token because API auth can be disabled.
    worker_agent.validate_config()


def test_register_or_update_host_skips_auth_header_without_token(monkeypatch):
    monkeypatch.setattr(worker_agent, "HOST_ID", "rig-01")
    monkeypatch.setattr(worker_agent, "SCHEDULER_URL", "http://localhost:8000")
    monkeypatch.setattr(worker_agent, "API_TOKEN", None)
    monkeypatch.setattr(worker_agent, "COST_PER_HOUR", 0.10)

    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

    def fake_put(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(worker_agent.requests, "put", fake_put)

    ok = worker_agent.register_or_update_host(
        {"gpu_model": "RTX 2060", "total_vram_gb": 6.0, "free_vram_gb": 4.5},
        "127.0.0.1",
    )

    assert ok is True
    assert captured["url"] == "http://localhost:8000/host"
    assert captured["headers"] == {}
    assert captured["json"]["host_id"] == "rig-01"
