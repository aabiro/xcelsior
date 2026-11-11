"""The worker pulls and pushes with its own registry credentials, never the host account's.

A provider host's ~/.docker/config.json belongs to the operator. Two things go
wrong when a tenant-named image is pulled with it: a desktop credential helper
has no session bus under systemd, so every pull fails ("error getting
credentials ... Cannot autolaunch D-Bus"), and the operator's own registry
logins would authenticate a pull the tenant chose.
"""

import base64
import json
import os
import stat

import worker_agent


def _config(monkeypatch, tmp_path, **env):
    for name in (
        "XCELSIOR_REGISTRY_URL",
        "XCELSIOR_REGISTRY_USERNAME",
        "XCELSIOR_REGISTRY_PASSWORD",
        "NGC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(worker_agent, "_WORKER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(worker_agent, "_registry_config_dir_cached", None)
    path = worker_agent._registry_config_dir()
    with open(os.path.join(path, "config.json")) as fh:
        return path, json.load(fh)


def test_config_carries_no_credential_helper(monkeypatch, tmp_path):
    path, config = _config(monkeypatch, tmp_path)
    assert config == {"auths": {}}
    assert "credsStore" not in config and "credHelpers" not in config
    assert path.startswith(str(tmp_path)), "config must live in the worker's state dir"


def test_config_is_private_to_the_worker(monkeypatch, tmp_path):
    path, _ = _config(monkeypatch, tmp_path, NGC_API_KEY="k")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(os.path.join(path, "config.json")).st_mode) == 0o600


def test_only_provisioned_registries_are_authenticated(monkeypatch, tmp_path):
    _, config = _config(
        monkeypatch,
        tmp_path,
        XCELSIOR_REGISTRY_URL="https://registry.example.test/xcelsior",
        XCELSIOR_REGISTRY_USERNAME="worker",
        XCELSIOR_REGISTRY_PASSWORD="pw",
        NGC_API_KEY="ngc-key",
    )
    assert set(config["auths"]) == {"registry.example.test", "nvcr.io"}
    decoded = base64.b64decode(config["auths"]["registry.example.test"]["auth"]).decode()
    assert decoded == "worker:pw"
    ngc = base64.b64decode(config["auths"]["nvcr.io"]["auth"]).decode()
    assert ngc == "$oauthtoken:ngc-key"


def test_half_configured_registry_is_left_out(monkeypatch, tmp_path):
    _, config = _config(
        monkeypatch,
        tmp_path,
        XCELSIOR_REGISTRY_URL="registry.example.test",
        XCELSIOR_REGISTRY_USERNAME="worker",
    )
    assert config == {"auths": {}}


def test_unwritable_state_dir_never_falls_back_to_the_host_config(monkeypatch, tmp_path):
    blocker = tmp_path / "state"
    blocker.write_text("a file where the directory should be")
    monkeypatch.setattr(worker_agent, "_WORKER_STATE_DIR", str(blocker))
    monkeypatch.setattr(worker_agent, "_registry_config_dir_cached", None)
    path = worker_agent._registry_config_dir()
    assert os.path.isfile(os.path.join(path, "config.json"))
    assert os.path.realpath(path) != os.path.realpath(os.path.expanduser("~/.docker"))
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o700


def test_command_names_the_worker_config(monkeypatch, tmp_path):
    path, _ = _config(monkeypatch, tmp_path)
    assert worker_agent.registry_docker_cmd("pull", "img:tag") == [
        "docker",
        "--config",
        path,
        "pull",
        "img:tag",
    ]


def test_no_registry_operation_bypasses_the_worker_config():
    """Every `docker pull` / `docker push` in the agent bundle goes through the helper."""
    root = os.path.dirname(os.path.abspath(worker_agent.__file__))
    offenders = []
    for name in ("worker_agent.py", "worker_image_cache.py"):
        with open(os.path.join(root, name)) as fh:
            for lineno, line in enumerate(fh, 1):
                compact = line.replace(" ", "")
                if '"docker","pull"' in compact or '"docker","push"' in compact:
                    offenders.append(f"{name}:{lineno}")
    assert not offenders, f"bare registry command, host credentials would be used: {offenders}"


def test_pull_failure_says_whose_fault_it_was():
    msg = worker_agent._pull_failure_message(
        1, "error getting credentials - err: exit status 1, out: `Cannot autolaunch D-Bus`"
    )
    assert "host could not read its own registry credentials" in msg
    assert "D-Bus" not in msg, "host internals stay in the host journal"
    assert "not found" in worker_agent._pull_failure_message(1, "manifest unknown")
    assert "refused access" in worker_agent._pull_failure_message(
        1, "pull access denied for x, repository does not exist or may require 'docker login'"
    ) or "not found" in worker_agent._pull_failure_message(
        1, "pull access denied for x, repository does not exist or may require 'docker login'"
    )
    assert worker_agent._pull_failure_message(7, "something new") == "Image pull failed (exit 7)"
