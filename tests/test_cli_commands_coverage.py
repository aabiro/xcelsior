"""Smoke coverage for CLI commands missing from the regenerate worklist."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cli


def test_cmd_token_gen(capsys):
    cli.cmd_token_gen(SimpleNamespace())
    out = capsys.readouterr().out
    assert "Token:" in out
    assert "XCELSIOR_API_TOKEN=" in out


def test_cmd_ssh_pubkey(capsys):
    with patch("cli.get_public_key", return_value="ssh-ed25519 AAAA test"):
        cli.cmd_ssh_pubkey(SimpleNamespace())
    assert "ssh-ed25519" in capsys.readouterr().out


def test_cmd_ssh_pubkey_missing_exits():
    with patch("cli.get_public_key", return_value=None):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_ssh_pubkey(SimpleNamespace())
        assert exc.value.code == 1


def test_cmd_host_add(capsys):
    entry = {
        "host_id": "h1",
        "ip": "10.0.0.1",
        "gpu_model": "RTX 4090",
        "total_vram_gb": 24,
        "cost_per_hour": 1.5,
        "country": "CA",
        "province": "ON",
    }
    args = SimpleNamespace(
        id="h1",
        ip="10.0.0.1",
        gpu="RTX 4090",
        vram=24,
        free_vram=24,
        rate=1.5,
        country="CA",
        province="ON",
    )
    with patch("cli.register_host", return_value=entry):
        cli.cmd_host_add(args)
    assert "h1" in capsys.readouterr().out


def test_cmd_host_rm(capsys):
    with patch("cli.remove_host") as rm:
        cli.cmd_host_rm(SimpleNamespace(id="h1"))
    rm.assert_called_once_with("h1")
    assert "h1" in capsys.readouterr().out


def test_cmd_host_add_ca(capsys):
    entry = {
        "host_id": "ca1",
        "ip": "10.0.0.2",
        "gpu_model": "A100",
        "country": "CA",
        "province": "QC",
        "cost_per_hour": 3.0,
    }
    args = SimpleNamespace(
        id="ca1",
        ip="10.0.0.2",
        gpu="A100",
        vram=80,
        free_vram=80,
        rate=3.0,
        country="CA",
        province="QC",
    )
    with patch("cli.register_host_ca", return_value=entry):
        cli.cmd_host_add_ca(args)
    assert "ca1" in capsys.readouterr().out


def test_cmd_hosts_ca(capsys):
    hosts = [
        {
            "status": "active",
            "host_id": "ca1",
            "ip": "10.0.0.2",
            "gpu_model": "A100",
            "country": "CA",
            "free_vram_gb": 80,
            "cost_per_hour": 3.0,
        }
    ]
    with patch("cli.list_hosts_filtered", return_value=hosts):
        cli.cmd_hosts_ca(SimpleNamespace(all=False))
    assert "ca1" in capsys.readouterr().out


def test_cmd_pool_add(capsys):
    entry = {"host_id": "p1", "gpu_model": "RTX 4090", "vram_gb": 24}
    args = SimpleNamespace(host_id="p1", ip="10.0.0.3", gpu="RTX 4090", vram=24, rate=2.0, country="CA")
    with patch("cli.add_to_pool", return_value=entry):
        cli.cmd_pool_add(args)
    assert "p1" in capsys.readouterr().out


def test_cmd_pool_rm(capsys):
    with patch("cli.remove_from_pool") as rm:
        cli.cmd_pool_rm(SimpleNamespace(host_id="p1"))
    rm.assert_called_once_with("p1")


def test_cmd_market_stats(capsys):
    stats = {
        "active_listings": 2,
        "total_listings": 5,
        "total_jobs_completed": 10,
        "total_host_payouts": 100.0,
        "platform_revenue": 20.0,
        "default_platform_cut_pct": 0.15,
    }
    with patch("cli.marketplace_stats", return_value=stats):
        cli.cmd_market_stats(SimpleNamespace())
    out = capsys.readouterr().out
    assert "Active listings" in out
    assert "2" in out


def test_cmd_market_unlist(capsys):
    with patch("cli.unlist_rig", return_value=True):
        cli.cmd_market_unlist(SimpleNamespace(host_id="rig1"))
    assert "rig1" in capsys.readouterr().out


def test_cmd_market_unlist_not_found():
    with patch("cli.unlist_rig", return_value=False):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_market_unlist(SimpleNamespace(host_id="missing"))
        assert exc.value.code == 1


def test_cmd_provider_register(capsys):
    result = {
        "status": "pending",
        "stripe_account_id": "acct_123",
        "onboarding_url": "https://stripe.example/onboard",
    }
    args = SimpleNamespace(
        provider_id="prov1",
        email="p@example.com",
        type="individual",
        corp_name="",
        bn="",
        gst="",
        province="ON",
    )
    mock_mgr = MagicMock()
    mock_mgr.create_provider_account.return_value = result
    with patch("stripe_connect.get_stripe_manager", return_value=mock_mgr):
        cli.cmd_provider_register(args)
    out = capsys.readouterr().out
    assert "prov1" in out
    assert "acct_123" in out


def test_cmd_provider_info(capsys):
    mock_mgr = MagicMock()
    mock_mgr.get_provider.return_value = {"email": "p@example.com", "status": "active"}
    with patch("stripe_connect.get_stripe_manager", return_value=mock_mgr):
        cli.cmd_provider_info(SimpleNamespace(provider_id="prov1"))
    assert "prov1" in capsys.readouterr().out


def test_cmd_provider_info_not_found(capsys):
    mock_mgr = MagicMock()
    mock_mgr.get_provider.return_value = None
    with patch("stripe_connect.get_stripe_manager", return_value=mock_mgr):
        cli.cmd_provider_info(SimpleNamespace(provider_id="missing"))
    assert "not found" in capsys.readouterr().out


def test_cmd_slurm_submit(capsys):
    args = SimpleNamespace(
        model="llama",
        vram=8,
        priority=0,
        tier="free",
        gpus=1,
        profile="default",
        dry_run=True,
        image="",
    )
    with patch("slurm_adapter.slurm_submit_cli", return_value={"ok": True, "dry_run": True}) as submit:
        cli.cmd_slurm_submit(args)
    submit.assert_called_once()
    assert "ok" in capsys.readouterr().out.lower() or "dry_run" in capsys.readouterr().out.lower()


def test_cmd_slurm_cancel(capsys):
    with patch("slurm_adapter.cancel_slurm_job", return_value={"cancelled": True}):
        cli.cmd_slurm_cancel(SimpleNamespace(slurm_id="12345"))
    assert "12345" in capsys.readouterr().out


def test_cmd_slurm_cancel_error(capsys):
    with patch("slurm_adapter.cancel_slurm_job", return_value={"cancelled": False, "error": "gone"}):
        cli.cmd_slurm_cancel(SimpleNamespace(slurm_id="999"))
    assert "gone" in capsys.readouterr().out


def test_cmd_health_start_keyboard_interrupt(capsys):
    args = SimpleNamespace(interval=30)
    with patch("cli.start_health_monitor") as start:
        with patch("cli.time.sleep", side_effect=KeyboardInterrupt):
            cli.cmd_health_start(args)
    start.assert_called_once_with(interval=30)
    assert "stopped" in capsys.readouterr().out


def test_cmd_host_profile_list(capsys):
    profiles = [
        {
            "profile_id": "rtx4090",
            "gpu_model": "RTX 4090",
            "total_vram_gb": 24,
            "default_rate_cad_per_hour": 1.5,
            "description": "Consumer",
        }
    ]
    with patch("host_profiles.list_host_profiles", return_value=profiles):
        cli.cmd_host_profile(SimpleNamespace(profile=None, host_id="h1", ip="10.0.0.1", json=False, country="CA", province="ON", owner="me"))
    assert "rtx4090" in capsys.readouterr().out


def test_cmd_host_accept_ready(capsys):
    report = {
        "host_id": "h1",
        "ready": True,
        "checks": [{"name": "gpu", "ok": True, "severity": "error", "detail": "ok"}],
    }
    args = SimpleNamespace(
        host_id="h1",
        expected_gpu="RTX 4090",
        min_vram=20,
        docker_probe=False,
        docker_image="",
        json=False,
    )
    with patch("host_acceptance.probe_local_host", return_value=report):
        cli.cmd_host_accept(args)
    assert "Ready: yes" in capsys.readouterr().out


def test_cmd_host_accept_not_ready_exits():
    report = {
        "host_id": "h1",
        "ready": False,
        "checks": [{"name": "gpu", "ok": False, "severity": "error", "detail": "missing"}],
    }
    args = SimpleNamespace(
        host_id="h1",
        expected_gpu=None,
        min_vram=0,
        docker_probe=False,
        docker_image="",
        json=False,
    )
    with patch("host_acceptance.probe_local_host", return_value=report):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_host_accept(args)
        assert exc.value.code == 1