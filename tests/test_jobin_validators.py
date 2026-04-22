"""P2.1 — validators on JobIn for init_script, git_repo, auto_launch, exposed_ports.

These fields are user-controlled, so validation hardens the API boundary
before anything lands in the jobs.payload JSONB column or gets shipped to
the GPU agent. See PLAN_EXECUTION.md §P2.1.
"""

import pytest
from pydantic import ValidationError

from routes.instances import JobIn


def _make(**overrides):
    base = dict(name="test", vram_needed_gb=0, interactive=True)
    base.update(overrides)
    return JobIn(**base)


# ---------- init_script -----------------------------------------------------


def test_init_script_rejects_null_byte():
    with pytest.raises(ValidationError):
        _make(init_script="echo hi\x00evil")


def test_init_script_rejects_bel():
    with pytest.raises(ValidationError):
        _make(init_script="echo hi\x07")


def test_init_script_accepts_multiline():
    j = _make(init_script="echo one\necho two\n\tindented")
    assert "echo two" in j.init_script


def test_init_script_too_long_rejected():
    with pytest.raises(ValidationError):
        _make(init_script="a" * 5000)


# ---------- git_repo --------------------------------------------------------


def test_git_repo_accepts_https():
    j = _make(git_repo="https://github.com/acme/repo.git")
    assert j.git_repo.startswith("https://")


def test_git_repo_rejects_http():
    with pytest.raises(ValidationError):
        _make(git_repo="http://github.com/acme/repo.git")


def test_git_repo_rejects_ssh():
    with pytest.raises(ValidationError):
        _make(git_repo="git@github.com:acme/repo.git")


def test_git_repo_rejects_embedded_credentials():
    with pytest.raises(ValidationError):
        _make(git_repo="https://user:token@github.com/acme/repo.git")


# ---------- auto_launch -----------------------------------------------------


def test_auto_launch_accepts_known_services():
    j = _make(auto_launch=["jupyter", "vscode"])
    assert j.auto_launch == ["jupyter", "vscode"]


def test_auto_launch_rejects_unknown_service():
    with pytest.raises(ValidationError):
        _make(auto_launch=["emacs"])


def test_auto_launch_dedupes_and_lowercases():
    j = _make(auto_launch=["JUPYTER", "jupyter", "VSCode"])
    assert j.auto_launch == ["jupyter", "vscode"]


# ---------- exposed_ports ---------------------------------------------------


def test_exposed_ports_accepts_valid():
    j = _make(exposed_ports=[8888, 8080, 3000])
    assert j.exposed_ports == [8888, 8080, 3000]


def test_exposed_ports_reserves_22():
    # Port 22 is the platform SSH path — surface a clear error rather than
    # silently dropping it.
    with pytest.raises(ValidationError):
        _make(exposed_ports=[22])


def test_exposed_ports_rejects_out_of_range():
    with pytest.raises(ValidationError):
        _make(exposed_ports=[0])
    with pytest.raises(ValidationError):
        _make(exposed_ports=[70000])


def test_exposed_ports_dedupes():
    j = _make(exposed_ports=[8080, 8080, 8888])
    assert j.exposed_ports == [8080, 8888]


def test_exposed_ports_rejects_non_integer():
    with pytest.raises(ValidationError):
        _make(exposed_ports=["not-a-port"])
