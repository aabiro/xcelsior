"""The real CUDA acceptance must refuse non-disposable database targets."""

import pytest
from psycopg.conninfo import make_conninfo

from scripts.worker_billing_acceptance import validate_isolation


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    root = tmp_path / "xcl-sec-pg-guard"
    socket_dir, state = root / "socket", root / "state"
    socket_dir.mkdir(parents=True)
    state.mkdir()
    dsn = make_conninfo(host=str(socket_dir), dbname="xcelsior_security_test", user="security_test")
    monkeypatch.setenv("XCELSIOR_ISOLATED_ACCEPTANCE", "1")
    monkeypatch.setenv("XCELSIOR_ENV", "test")
    monkeypatch.setenv("XCELSIOR_SKIP_DOTENV", "1")
    monkeypatch.setenv("XCELSIOR_ACCEPTANCE_STATE", str(state))
    for key in (
        "XCELSIOR_POSTGRES_DSN",
        "DATABASE_URL",
        "XCELSIOR_PG_DSN",
        "XCELSIOR_TEST_POSTGRES_DSN",
    ):
        monkeypatch.setenv(key, dsn)
    return dsn, state


def test_disposable_target_passes_without_connecting(isolated):
    assert validate_isolation() == isolated[1]


@pytest.mark.parametrize(
    "override",
    [
        {"host": "production.example.test"},
        {"hostaddr": "203.0.113.1"},
        {"service": "production"},
        {"dbname": "production"},
        {"user": "production"},
    ],
)
def test_connection_overrides_are_refused(isolated, monkeypatch, override):
    monkeypatch.setenv("XCELSIOR_POSTGRES_DSN", make_conninfo(isolated[0], **override))
    with pytest.raises(RuntimeError):
        validate_isolation()


@pytest.mark.parametrize(
    "key,value",
    [
        ("XCELSIOR_ISOLATED_ACCEPTANCE", ""),
        ("XCELSIOR_ENV", "production"),
        ("XCELSIOR_SKIP_DOTENV", ""),
        ("DATABASE_URL", "postgresql://production.example.test/live"),
        ("XCELSIOR_ACCEPTANCE_STATE", "/tmp"),
    ],
)
def test_guard_refuses_uncontrolled_environment(isolated, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError):
        validate_isolation()
