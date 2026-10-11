"""The worker keeps its attempt journal in systemd's StateDirectory when it has one."""

import os

import worker_agent


def test_state_directory_from_systemd(monkeypatch):
    monkeypatch.setenv("STATE_DIRECTORY", "/var/lib/xcelsior:/var/lib/other")
    assert worker_agent._worker_state_dir() == "/var/lib/xcelsior"


def test_falls_back_to_home_when_var_lib_is_not_writable(monkeypatch, tmp_path):
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(os, "access", lambda path, mode: False)
    assert worker_agent._worker_state_dir() == str(tmp_path / ".xcelsior" / "state")


def test_uses_var_lib_when_writable(monkeypatch):
    monkeypatch.delenv("STATE_DIRECTORY", raising=False)
    monkeypatch.setattr(os, "access", lambda path, mode: True)
    assert worker_agent._worker_state_dir() == "/var/lib/xcelsior"
