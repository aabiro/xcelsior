"""The registry-built SSH transport (workers/slurm_transport.py, plan item 1.2).

What it must guarantee, without a cluster: the only host key ssh will accept
is the pinned one, nothing comes from ~/.ssh, a key held in the environment
exists on disk only while the runner is in use, and nothing a registry row
holds can add a line to the generated config.
"""

from __future__ import annotations

import base64
import os
import re
import stat

import pytest

import slurm_adapter as sa
import slurm_transport as st

KEY = base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519" + b"k" * 36).decode()
PIN = f"203.0.113.7 ssh-ed25519 {KEY}"
PEM = "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----\n"


@pytest.fixture(autouse=True)
def _runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("XCELSIOR_SLURM_RUNTIME_DIR", str(tmp_path))
    return tmp_path


def _cluster(**kw):
    row = {
        "cluster_id": "lab",
        "ssh_host": "203.0.113.7",
        "ssh_port": 2222,
        "remote_user": "xclslurm",
        "known_hosts_line": PIN,
        "credential_ref": "file:/etc/xcelsior/slurm/lab.key",
    }
    row.update(kw)
    return row


def _config(runner) -> str:
    return open(runner.ssh_options[1]).read()


class TestPinnedKey:
    def test_takes_the_key_from_any_known_hosts_form(self):
        assert st.pinned_key(PIN) == ("ssh-ed25519", KEY)
        assert st.pinned_key(f"ssh-ed25519 {KEY}") == ("ssh-ed25519", KEY)
        hashed = f"|1|c2FsdA==|aGFzaA== ssh-ed25519 {KEY} comment"
        assert st.pinned_key(hashed) == ("ssh-ed25519", KEY)

    @pytest.mark.parametrize(
        "line",
        [
            "",
            f"@cert-authority *.example ssh-ed25519 {KEY}",
            f"@revoked host ssh-ed25519 {KEY}",
            "host ssh-ed25519 not*base64",
            "host ssh-dss AAAA",
            f"host ssh-ed25519 {KEY}\nother ssh-ed25519 {KEY}",
        ],
    )
    def test_refuses_what_does_not_pin_one_key(self, line):
        with pytest.raises(sa.SlurmAdapterError) as e:
            st.pinned_key(line)
        assert e.value.code == "invalid_host_key"


class TestGeneratedConfig:
    def test_only_the_pinned_key_and_nothing_from_home(self, _runtime):
        with st.cluster_runner(_cluster()) as runner:
            assert runner.ssh_host == "xcl-lab" and runner.ssh_options[0] == "-F"
            conf = _config(runner)
            known = open(os.path.join(os.path.dirname(runner.ssh_options[1]), "known_hosts")).read()
        assert known == f"xcl-lab ssh-ed25519 {KEY}\n"
        for line in (
            "HostName 203.0.113.7",
            "Port 2222",
            "User xclslurm",
            "HostKeyAlias xcl-lab",
            "GlobalKnownHostsFile /dev/null",
            "StrictHostKeyChecking yes",
            "UpdateHostKeys no",
            "IdentityFile /etc/xcelsior/slurm/lab.key",
            "IdentitiesOnly yes",
            "IdentityAgent none",
            "BatchMode yes",
            "ControlMaster auto",
            f"ControlPersist {st.CONTROL_PERSIST_SEC}",
        ):
            assert f"    {line}\n" in conf, line
        base = f"{_runtime}/xcelsior-slurm-{os.getuid()}"
        assert re.search(rf"    ControlPath {re.escape(base)}/cm-[0-9a-f]{{12}}-%C\n", conf)

    def test_each_use_cleans_up_but_the_master_socket_dir_stays(self, _runtime):
        with st.cluster_runner(_cluster()) as runner:
            workdir = os.path.dirname(runner.ssh_options[1])
            assert os.path.isdir(workdir)
        assert not os.path.exists(workdir)
        base = _runtime / f"xcelsior-slurm-{os.getuid()}"
        assert base.is_dir() and stat.S_IMODE(base.stat().st_mode) == 0o700

    @pytest.mark.parametrize(
        "field,value,code",
        [
            ("ssh_host", "lab\n    ProxyCommand evil", "invalid_host"),
            ("ssh_host", "lab host", "invalid_host"),
            ("remote_user", "root\nHost *", "invalid_user"),
            ("ssh_port", 70000, "invalid_port"),
            ("cluster_id", "../lab", "invalid_cluster"),
            ("credential_ref", "file:relative/key", "invalid_credential_ref"),
            ("credential_ref", "file:/key with space", "invalid_credential_ref"),
            ("credential_ref", "vault:slurm/lab", "invalid_credential_ref"),
        ],
    )
    def test_a_row_cannot_write_its_own_config_lines(self, field, value, code):
        with pytest.raises(sa.SlurmAdapterError) as e:
            with st.cluster_runner(_cluster(**{field: value})):
                pass
        assert e.value.code == code

    def test_without_an_ssh_host_there_is_no_registry_transport(self):
        with pytest.raises(sa.SlurmAdapterError) as e:
            with st.cluster_runner(_cluster(ssh_host=None)):
                pass
        assert e.value.code == "no_registry_transport"


def _control_path(row) -> str:
    with st.cluster_runner(row) as runner:
        return next(
            line.split()[1] for line in _config(runner).splitlines() if "ControlPath" in line
        )


def test_a_new_pin_or_key_never_reuses_the_old_master():
    # A mux client skips host key checks, so the old master must be unreachable.
    first = _control_path(_cluster())
    assert _control_path(_cluster()) == first  # same trust: share the master
    other_key = base64.b64encode(b"\x00\x00\x00\x0bssh-ed25519" + b"z" * 36).decode()
    assert _control_path(_cluster(known_hosts_line=f"h ssh-ed25519 {other_key}")) != first
    assert _control_path(_cluster(credential_ref="file:/etc/xcelsior/slurm/new.key")) != first


class TestEnvironmentKey:
    def test_written_0600_for_the_use_and_removed_after(self, monkeypatch):
        monkeypatch.setenv("XCL_LAB_KEY", PEM)
        with st.cluster_runner(_cluster(credential_ref="env:XCL_LAB_KEY")) as runner:
            identity = next(
                line.split()[1]
                for line in _config(runner).splitlines()
                if line.strip().startswith("IdentityFile")
            )
            assert open(identity).read() == PEM
            assert stat.S_IMODE(os.stat(identity).st_mode) == 0o600
        assert not os.path.exists(identity)

    def test_escaped_newlines_are_restored(self, monkeypatch):
        monkeypatch.setenv("XCL_LAB_KEY", PEM.replace("\n", "\\n"))
        with st.cluster_runner(_cluster(credential_ref="env:XCL_LAB_KEY")) as runner:
            identity = os.path.join(os.path.dirname(runner.ssh_options[1]), "key")
            assert open(identity).read() == PEM

    def test_a_missing_key_is_refused(self, monkeypatch):
        monkeypatch.delenv("XCL_LAB_KEY", raising=False)
        with pytest.raises(sa.SlurmAdapterError) as e:
            with st.cluster_runner(_cluster(credential_ref="env:XCL_LAB_KEY")):
                pass
        assert e.value.code == "credential_unavailable"


def test_an_unsafe_runtime_dir_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("XCELSIOR_SLURM_RUNTIME_DIR", str(tmp_path))
    base = tmp_path / f"xcelsior-slurm-{os.getuid()}"
    base.mkdir(mode=0o755)
    base.chmod(0o755)
    with pytest.raises(sa.SlurmAdapterError) as e:
        st.runtime_dir()
    assert e.value.code == "unsafe_runtime_dir"


def test_a_changed_host_key_is_its_own_error():
    class R:
        remote = True

    class P:
        returncode = 255
        stderr = "Host key verification failed.\r\n"

    assert sa._transport_failure(P(), R())["error_code"] == "host_key_mismatch"
    P.stderr = "ssh: connect to host lab port 22: Connection refused"
    assert sa._transport_failure(P(), R())["error_code"] == "transport_error"
