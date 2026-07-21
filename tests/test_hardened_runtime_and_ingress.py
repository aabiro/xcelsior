"""Unprivileged API image and the public worker-surface cutover.

Blueprint §19.4 (privilege separation), §21.1 (unprivileged API),
§22.1/§22.3 (separate public endpoints; the agent gateway serves the
worker protocol, the public origin does not).

The ingress tests drive the real ASGI middleware, so they prove the
surface is closed at the *API*, not only at the edge — an edge-only
cutover leaves the endpoints live for anything that reaches the API
directly (a second origin, a container on the host network, an Nginx
include that lands in the wrong order).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

from api import app  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
client = TestClient(app)


def _compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())


# ── Unprivileged image (§19.4 / §21.1) ────────────────────────────────


def test_dockerfile_runs_as_a_non_root_user():
    df = (ROOT / "Dockerfile").read_text()
    user_lines = [ln.strip() for ln in df.splitlines() if ln.strip().startswith("USER ")]
    assert user_lines, "image must declare a runtime USER"
    # Last USER wins; it must not be root or 0.
    final = user_lines[-1].split(None, 1)[1].strip()
    assert final not in ("root", "0", "0:0")
    assert "${APP_UID}" in final or re.match(r"^\d{3,}", final), final

    # The uid is pinned so host bind mounts and the data volume can be
    # granted to it deterministically across rebuilds.
    assert re.search(r"ARG APP_UID=(\d+)", df)
    uid = int(re.search(r"ARG APP_UID=(\d+)", df).group(1))
    assert uid >= 1000, "runtime uid must be unprivileged"


def test_dockerfile_prepares_writable_paths_for_the_runtime_user():
    """A read-only rootfs only works if the writable paths are real."""
    df = (ROOT / "Dockerfile").read_text()
    assert "mkdir -p /home/xcelsior/.ssh /data" in df
    assert "chown -R ${APP_UID}:${APP_GID} /home/xcelsior /data" in df
    # /app must NOT be writable by the runtime user.
    assert "chown -R root:${APP_GID} /app" in df
    assert "chmod -R g+rX /app" in df


def test_dockerfile_drops_privileged_volume_tooling():
    """§19.4: LUKS/NFS binaries belong in the volume-provisioner image."""
    df = (ROOT / "Dockerfile").read_text()
    install = "\n".join(
        ln for ln in df.splitlines() if "apt-get install" in ln or ln.startswith("    ")
    )
    for pkg in ("cryptsetup", "e2fsprogs"):
        assert pkg not in install, f"{pkg} must not ship in the unprivileged API image"
    # ...and must still be present where the privilege actually lives.
    provisioner = (ROOT / "infra" / "volume-provisioner" / "Dockerfile").read_text()
    assert "cryptsetup" in provisioner


def test_compose_api_services_are_fully_hardened():
    services = _compose()["services"]
    for name in ("api", "api-blue", "bg-worker"):
        svc = services[name]
        assert svc.get("read_only") is True, f"{name} must have a read-only rootfs"
        assert svc.get("cap_drop") == ["ALL"], f"{name} must drop all capabilities"
        assert "no-new-privileges:true" in svc.get("security_opt", []), name
        assert "cap_add" not in svc, f"{name} must not add capabilities"
        user = str(svc.get("user", ""))
        assert user, f"{name} must pin a runtime user"
        assert "root" not in user and not user.startswith("0:"), user


def test_compose_grants_tmpfs_where_the_process_must_write():
    """Read-only is only safe if the genuinely-writable paths are provided."""
    api = _compose()["services"]["api"]
    mounts = {entry.split(":", 1)[0] for entry in api.get("tmpfs", [])}
    # gunicorn's worker heartbeat file lives in /dev/shm (gunicorn.conf.py).
    assert "/dev/shm" in mounts
    assert "/tmp" in mounts
    tmp_opts = next(e for e in api["tmpfs"] if e.startswith("/tmp"))
    assert "noexec" in tmp_opts and "nosuid" in tmp_opts

    gunicorn_conf = (ROOT / "gunicorn.conf.py").read_text()
    tmp_dir = re.search(r'worker_tmp_dir\s*=\s*"([^"]+)"', gunicorn_conf)
    assert (
        tmp_dir and tmp_dir.group(1) in mounts
    ), "gunicorn worker_tmp_dir must be one of the writable tmpfs mounts"


def test_scheduler_known_hosts_moved_out_of_root_home():
    """Non-root cannot read /root — SSH host verification would break."""
    svc = _compose()["services"]["scheduler-worker"]
    known_hosts = [v for v in svc["volumes"] if "known_hosts" in v]
    assert known_hosts, "scheduler must still pin known_hosts"
    assert all("/root/" not in v for v in known_hosts), known_hosts
    assert any("/home/xcelsior/.ssh/known_hosts" in v for v in known_hosts)
    assert svc.get("cap_drop") == ["ALL"]


def test_only_the_volume_provisioner_is_privileged():
    """§19.4: exactly one narrow service holds SYS_ADMIN."""
    services = _compose()["services"]
    privileged = [
        name
        for name, svc in services.items()
        if "SYS_ADMIN" in (svc.get("cap_add") or []) or svc.get("privileged")
    ]
    assert privileged == ["volume-provisioner"], privileged
    # ...and it is opt-in, never part of the default `up`.
    assert services["volume-provisioner"].get("profiles") == ["volume-provisioner"]


def test_file_logging_degrades_instead_of_crashing(tmp_path):
    """A read-only rootfs must not stop the process from starting.

    Regression test for a real failure found while running the hardened
    image: the default log path is inside /app, which is read-only, and
    ``logging.FileHandler`` raised before the app could serve anything.
    """
    import logging

    import scheduler

    unwritable = tmp_path / "nope" / "xcelsior.log"  # parent does not exist
    xcelsior_logger = logging.getLogger("xcelsior")
    saved = list(xcelsior_logger.handlers)
    xcelsior_logger.handlers.clear()
    try:
        result = scheduler.setup_logging(log_file=str(unwritable), level=logging.INFO)
        handler_types = {type(h) for h in result.handlers}
        assert logging.StreamHandler in handler_types, "console logging must survive"
        assert (
            logging.FileHandler not in handler_types
        ), "unwritable log path must be skipped, not fatal"

        # ...and a writable path still gets the permanent record.
        writable = tmp_path / "xcelsior.log"
        xcelsior_logger.handlers.clear()
        result = scheduler.setup_logging(log_file=str(writable), level=logging.INFO)
        assert any(isinstance(h, logging.FileHandler) for h in result.handlers)
    finally:
        for handler in list(xcelsior_logger.handlers):
            handler.close()
        xcelsior_logger.handlers.clear()
        xcelsior_logger.handlers.extend(saved)


# ── Public worker-surface cutover (§22.1 / §22.3) ─────────────────────


WORKER_PATHS = ("/agent/popular-images", "/agent/telemetry", "/host")


def _bearer() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("path", WORKER_PATHS)
def test_allow_mode_keeps_the_public_worker_surface(monkeypatch, path):
    monkeypatch.setenv("XCELSIOR_AGENT_PUBLIC_INGRESS", "allow")
    resp = client.get(path, headers=_bearer())
    assert resp.status_code != 410, resp.text


@pytest.mark.parametrize("path", WORKER_PATHS)
def test_deny_mode_retires_the_public_worker_surface(monkeypatch, path):
    monkeypatch.setenv("XCELSIOR_AGENT_PUBLIC_INGRESS", "deny")
    resp = client.get(path, headers=_bearer())
    assert resp.status_code == 410, resp.text
    body = resp.json()
    assert body["error"]["code"] == "agent_ingress_retired"
    # The message must tell an operator exactly what to change.
    assert "agent.xcelsior.ca" in body["error"]["message"]


def test_deny_mode_covers_every_worker_path_shape(monkeypatch):
    monkeypatch.setenv("XCELSIOR_AGENT_PUBLIC_INGRESS", "deny")
    for path in (
        "/agent/",
        "/agent/versions",
        "/agent/v2/negotiate/gpu-01",
        "/agent/commands/gpu-01",
        "/host",
        "/host/anything",
    ):
        assert client.get(path).status_code == 410, path


def test_deny_mode_does_not_touch_the_product_api(monkeypatch):
    """The cutover must not take the dashboard down with it."""
    monkeypatch.setenv("XCELSIOR_AGENT_PUBLIC_INGRESS", "deny")
    for path in ("/healthz", "/api/status", "/api/pricing/gpus"):
        assert client.get(path).status_code != 410, path
    # A path that merely starts with the same letters is not worker surface.
    assert client.get("/api/hosts").status_code != 410


def test_deny_mode_still_admits_the_private_gateway(monkeypatch):
    """ "Public ingress retired" must not mean "the gateway is retired"."""
    monkeypatch.setenv("XCELSIOR_AGENT_PUBLIC_INGRESS", "deny")
    monkeypatch.setenv("XCELSIOR_AGENT_GATEWAY_SECRET", "gw-secret")

    blocked = client.get("/agent/popular-images", headers=_bearer())
    assert blocked.status_code == 410

    through = client.get(
        "/agent/popular-images",
        headers={**_bearer(), "X-Xcelsior-Gateway-Auth": "gw-secret"},
    )
    assert through.status_code != 410, through.text


def test_deny_mode_rejects_a_wrong_gateway_secret(monkeypatch):
    monkeypatch.setenv("XCELSIOR_AGENT_PUBLIC_INGRESS", "deny")
    monkeypatch.setenv("XCELSIOR_AGENT_GATEWAY_SECRET", "gw-secret")
    resp = client.get(
        "/agent/popular-images",
        headers={**_bearer(), "X-Xcelsior-Gateway-Auth": "not-the-secret"},
    )
    assert resp.status_code == 410, resp.text


def test_ingress_mode_defaults_to_allow(monkeypatch):
    """A repo checkout must never silently cut a live fleet off."""
    from api import AgentIngressMiddleware

    monkeypatch.delenv("XCELSIOR_AGENT_PUBLIC_INGRESS", raising=False)
    assert AgentIngressMiddleware._mode() == "allow"
    for value in ("allow", "on", "yes", "anything-else"):
        monkeypatch.setenv("XCELSIOR_AGENT_PUBLIC_INGRESS", value)
        assert AgentIngressMiddleware._mode() == "allow", value
    for value in ("deny", "0", "false", "off", "closed"):
        monkeypatch.setenv("XCELSIOR_AGENT_PUBLIC_INGRESS", value)
        assert AgentIngressMiddleware._mode() == "deny", value


# ── Nginx: the edge half of the same cutover ──────────────────────────


def test_public_nginx_strips_forgeable_identity_headers():
    conf = (ROOT / "nginx" / "xcelsior.conf").read_text()
    from control_plane.identity import UNTRUSTED_IDENTITY_HEADERS

    agent_block = conf[conf.index("location /host") : conf.index("location /api/")]
    for header in UNTRUSTED_IDENTITY_HEADERS:
        # Nginx header names are title-cased in the config.
        assert re.search(
            rf'proxy_set_header\s+{re.escape(header)}\s+""', agent_block, re.I
        ), f"public ingress does not strip {header}"


def test_retirement_snippet_is_a_drop_in_replacement():
    snippet = (ROOT / "nginx" / "snippets" / "xcelsior-agent-retired.conf").read_text()
    # Same two locations the public conf currently proxies.
    assert "location /host" in snippet
    assert "location /agent/" in snippet
    assert snippet.count("return 410") == 2
    assert "agent_ingress_retired" in snippet
    # The bootstrap must stay reachable — a host that cannot fetch the
    # agent cannot be migrated onto the gateway in the first place.
    assert "location = /static/worker_agent.py" in snippet
    assert "proxy_pass" in snippet


def test_public_conf_documents_the_cutover_procedure():
    conf = (ROOT / "nginx" / "xcelsior.conf").read_text()
    assert "xcelsior-agent-retired.conf" in conf
    assert "XCELSIOR_AGENT_PUBLIC_INGRESS=deny" in conf
