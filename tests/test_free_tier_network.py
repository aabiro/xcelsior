"""Tests for free-tier network boundary validation and container arg injection."""

from unittest.mock import Mock, patch

import pytest

import free_tier_network


def test_prepare_free_tier_non_free_is_noop():
    job = {"job_id": "job-1", "tier": "on-demand"}
    env = {"KEY": "VAL"}
    args, new_env = free_tier_network.prepare_free_tier(job, env)
    assert args == []
    assert new_env == env


def test_prepare_free_tier_injects_network_and_proxy(monkeypatch):
    monkeypatch.setattr(
        free_tier_network,
        "validate_free_tier_network",
        lambda: {
            "http_proxy": "http://172.30.242.1:3128",
            "https_proxy": "http://172.30.242.1:3128",
        },
    )
    job = {"job_id": "job-free-1", "tier": "free"}
    env = {"USER_ENV": "1"}

    args, new_env = free_tier_network.prepare_free_tier(job, env)
    assert args == ["--network", "xcelsior-free", "--dns", "127.0.0.1"]
    assert new_env["http_proxy"] == "http://172.30.242.1:3128"
    assert new_env["https_proxy"] == "http://172.30.242.1:3128"
    assert new_env["HTTP_PROXY"] == "http://172.30.242.1:3128"
    assert new_env["HTTPS_PROXY"] == "http://172.30.242.1:3128"
    assert "127.0.0.1" in new_env["NO_PROXY"]
    assert new_env["USER_ENV"] == "1"


def test_firewall_rules_defined():
    rules = free_tier_network.firewall_rules()
    assert len(rules) == 4
    assert rules[0][0] == "INPUT"
    assert rules[-1][-1] == "DROP"
