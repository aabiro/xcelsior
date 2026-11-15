"""A legacy live harness must never create accounts while being imported."""

import importlib.util
from pathlib import Path
import runpy

import pytest
import requests

HARNESS = Path(__file__).with_name("test_e2e_live.py")


@pytest.fixture
def harness(monkeypatch):
    def refuse_request(*args, **kwargs):
        pytest.fail("Import/configuration attempted a real HTTP request")

    monkeypatch.setattr(requests.sessions.Session, "request", refuse_request)
    spec = importlib.util.spec_from_file_location("manual_live_e2e", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configured_environment():
    return {
        "XCELSIOR_RUN_LIVE_E2E": "1",
        "XCELSIOR_E2E_BASE_URL": "https://api.example.test",
        "XCELSIOR_E2E_EMAIL": "qa+explicit-run@example.test",
        "XCELSIOR_API_TOKEN": "synthetic-test-token",
    }


def test_import_has_no_requests_output_generated_recipient_or_credentials(harness, capsys):
    assert capsys.readouterr().out == ""
    assert harness.TEST_EMAIL == ""
    assert harness.TOKEN == ""
    assert harness.AUTH == {}
    assert harness.results == []


def test_import_stays_inert_even_when_live_configuration_is_present(harness, monkeypatch, capsys):
    for name, value in configured_environment().items():
        monkeypatch.setenv(name, value)
    namespace = runpy.run_path(str(HARNESS))
    assert namespace["results"] == []
    assert capsys.readouterr().out == ""


def test_direct_execution_without_opt_in_stops_before_any_request(harness, monkeypatch, capsys):
    for name, value in configured_environment().items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("XCELSIOR_RUN_LIVE_E2E")
    assert harness.main() == 2
    assert "Live E2E is disabled" in capsys.readouterr().err
    assert harness.results == []


@pytest.mark.parametrize("missing", list(configured_environment()))
def test_every_live_setting_must_be_explicit(harness, missing):
    environment = configured_environment()
    environment.pop(missing)
    with pytest.raises(ValueError):
        harness.configure_run(environment)


@pytest.mark.parametrize(
    "address",
    [
        "",
        "e2e-test-user",
        "user@",
        "Display Name <user@example.test>",
        "user@example.test\nBcc: other@example.test",
    ],
)
def test_invalid_mailbox_is_refused_before_any_request(harness, address):
    environment = configured_environment()
    environment["XCELSIOR_E2E_EMAIL"] = address
    with pytest.raises(ValueError, match="real, unused test mailbox"):
        harness.configure_run(environment)


@pytest.mark.parametrize(
    "base",
    [
        "",
        "api.example.test",
        "ftp://api.example.test",
        "https://user:password@api.example.test",
        "https://api.example.test?unexpected=1",
    ],
)
def test_invalid_target_is_refused_before_any_request(harness, base):
    environment = configured_environment()
    environment["XCELSIOR_E2E_BASE_URL"] = base
    with pytest.raises(ValueError, match="intended API URL"):
        harness.configure_run(environment)


def test_registration_uses_only_the_explicit_mailbox(harness, monkeypatch):
    environment = configured_environment()
    harness.BASE, harness.TOKEN, harness.TEST_EMAIL = harness.configure_run(environment)
    submitted = []

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"ok": True, "user_id": "test-user"}

    def api(method, path, **kwargs):
        submitted.append((method, path, kwargs))
        return Response()

    monkeypatch.setattr(harness, "api", api)
    harness._e2e_register()
    assert len(submitted) == 1
    assert submitted[0][0:2] == ("post", "/api/auth/register")
    assert submitted[0][2]["json"]["email"] == environment["XCELSIOR_E2E_EMAIL"]
