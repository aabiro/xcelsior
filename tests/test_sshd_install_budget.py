"""The sshd install is given time according to how the container reaches its mirror (CVP-01 D8, D9)."""

import inspect
from types import SimpleNamespace

import worker_agent


def _probe(monkeypatch, returncode=0, raises=None):
    seen = []

    def run(cmd, **kwargs):
        seen.append(cmd)
        if raises:
            raise raises
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(worker_agent.subprocess, "run", run)
    return seen


def test_a_proxied_container_gets_the_long_allowance(monkeypatch):
    seen = _probe(monkeypatch, returncode=0)
    assert (
        worker_agent._sshd_install_budget("xcl-a") == worker_agent.SSHD_INSTALL_BUDGET_PROXIED_SEC
    )
    assert seen[0][:3] == ["docker", "exec", "xcl-a"]


def test_a_direct_container_gets_the_short_one(monkeypatch):
    _probe(monkeypatch, returncode=1)
    assert worker_agent._sshd_install_budget("xcl-a") == worker_agent.SSHD_INSTALL_BUDGET_DIRECT_SEC


def test_a_failed_probe_is_not_fatal(monkeypatch):
    _probe(monkeypatch, raises=worker_agent.subprocess.TimeoutExpired("docker", 5))
    assert worker_agent._sshd_install_budget("xcl-a") == worker_agent.SSHD_INSTALL_BUDGET_DIRECT_SEC


def test_the_allowances_cover_what_staging_measured():
    """T2 funded: 90 s end to end. T2 restricted: not finished at 38 s."""
    assert worker_agent.SSHD_INSTALL_BUDGET_DIRECT_SEC >= 90
    assert (
        worker_agent.SSHD_INSTALL_BUDGET_PROXIED_SEC
        >= 2 * worker_agent.SSHD_INSTALL_BUDGET_DIRECT_SEC
    )


def test_the_install_uses_its_allowance_and_tools_do_not_block_readiness():
    body = inspect.getsource(worker_agent._inject_ssh_keys)
    assert "timeout=_remaining(install_budget)" in body
    assert "timeout=_remaining(30)" not in body
    assert '"docker", "exec", "-d", container_name, "sh", "-c", tools_script' in body
