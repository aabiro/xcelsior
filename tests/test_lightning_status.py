import pytest
import lightning as ln


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setattr(ln, "LN_ENABLED", True)
    monkeypatch.setattr(ln, "LN_RUNE", "test-rune")


@pytest.mark.parametrize("warning, reason", [
    ("warning_bitcoind_sync", "Bitcoin node is synchronizing with the network"),
    ("warning_lightningd_sync", "Lightning node is synchronizing Bitcoin blocks"),
])
def test_reports_actual_sync_warnings(monkeypatch, warning, reason):
    monkeypatch.setattr(ln, "get_node_info", lambda: {"num_active_channels": 1, warning: "syncing"})
    assert ln.get_service_status()["reason"] == reason
    assert ln.get_service_status()["available"] is False


def test_requires_an_active_payment_channel(monkeypatch):
    monkeypatch.setattr(ln, "get_node_info", lambda: {"num_active_channels": 0, "blockheight": 975702})
    status = ln.get_service_status()
    assert status["available"] is False
    assert status["reason"] == "Lightning node has no active payment channels"


def test_ready_node_is_available(monkeypatch):
    monkeypatch.setattr(ln, "get_node_info", lambda: {"num_active_channels": 1})
    assert ln.get_service_status()["available"] is True


@pytest.mark.parametrize("error, reason", [
    ("CLN REST connection failed: [SSL: CERTIFICATE_VERIFY_FAILED] IP address mismatch", "Lightning connection certificate could not be verified"),
    ("CLN REST connection failed: connection refused", "Lightning node connection is unavailable"),
])
def test_transport_errors_are_not_reported_as_bitcoin_startup(monkeypatch, error, reason):
    def fail():
        raise RuntimeError(error)
    monkeypatch.setattr(ln, "get_node_info", fail)
    assert ln.get_service_status()["reason"] == reason
