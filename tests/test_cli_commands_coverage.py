"""Smoke coverage for CLI commands missing from the regenerate worklist."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cli


def test_cmd_run_no_assign(capsys):
    job = {
        "job_id": "j1",
        "name": "train",
        "vram_needed_gb": 8,
        "tier": "free",
        "priority": 0,
        "num_gpus": 1,
    }
    args = SimpleNamespace(
        model="llama",
        vram=8,
        priority=0,
        tier="free",
        gpus=1,
        gpu=None,
        nfs_server=None,
        nfs_path=None,
        image=None,
        no_assign=True,
    )
    with patch("cli.submit_job", return_value=job):
        with patch("cli.process_queue") as pq:
            cli.cmd_run(args)
    pq.assert_not_called()
    assert "j1" in capsys.readouterr().out


def test_cmd_serve_uvicorn(capsys):
    import builtins

    args = SimpleNamespace(bind="127.0.0.1", port=9500)
    mock_uvicorn = MagicMock()
    mock_api = MagicMock()
    mock_api.app = MagicMock()
    real_import = builtins.__import__

    def import_mock(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvicorn":
            return mock_uvicorn
        if name == "api":
            return mock_api
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=import_mock):
        cli.cmd_serve(args)
    mock_uvicorn.run.assert_called_once_with(mock_api.app, host="127.0.0.1", port=9500)
    assert "9500" in capsys.readouterr().out


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


def test_cmd_jobs_lists(capsys):
    jobs = [
        {
            "status": "queued",
            "job_id": "j1",
            "name": "train",
            "vram_needed_gb": 8,
            "tier": "free",
            "host_id": None,
        }
    ]
    with patch("cli.list_jobs", return_value=jobs):
        cli.cmd_jobs(SimpleNamespace(status=None))
    assert "j1" in capsys.readouterr().out


def test_cmd_jobs_empty(capsys):
    with patch("cli.list_jobs", return_value=[]):
        cli.cmd_jobs(SimpleNamespace(status=None))
    assert "No jobs" in capsys.readouterr().out


def test_cmd_job_found(capsys):
    job = {"job_id": "j1", "name": "train", "status": "running"}
    with patch("cli.list_jobs", return_value=[job]):
        cli.cmd_job(SimpleNamespace(job_id="j1"))
    assert "j1" in capsys.readouterr().out


def test_cmd_job_missing_exits():
    with patch("cli.list_jobs", return_value=[]):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_job(SimpleNamespace(job_id="missing"))
        assert exc.value.code == 1


def test_cmd_cancel(capsys):
    with patch("cli.update_job_status") as upd:
        cli.cmd_cancel(SimpleNamespace(job_id="j1"))
    upd.assert_called_once_with("j1", "failed")
    assert "j1" in capsys.readouterr().out


def test_cmd_process(capsys):
    job = {"job_id": "j1", "name": "train"}
    host = {"host_id": "h1", "gpu_model": "RTX 4090"}
    with patch("cli.process_queue", return_value=[(job, host)]):
        cli.cmd_process(SimpleNamespace())
    out = capsys.readouterr().out
    assert "h1" in out


def test_cmd_hosts(capsys):
    hosts = [
        {
            "status": "active",
            "host_id": "h1",
            "ip": "10.0.0.1",
            "gpu_model": "RTX 4090",
            "free_vram_gb": 20,
            "cost_per_hour": 1.5,
        }
    ]
    with patch("cli.list_hosts", return_value=hosts):
        cli.cmd_hosts(SimpleNamespace(all=False))
    assert "h1" in capsys.readouterr().out


def test_cmd_ping(capsys):
    with patch("cli.check_hosts", return_value={"h1": "ok"}):
        cli.cmd_ping(SimpleNamespace())
    assert "h1" in capsys.readouterr().out


def test_cmd_bill_single(capsys):
    record = {"job_name": "train", "duration_sec": 60, "cost": 1.25}
    with patch("cli.bill_job", return_value=record):
        cli.cmd_bill(SimpleNamespace(job_id="j1"))
    assert "train" in capsys.readouterr().out


def test_cmd_bill_all(capsys):
    with patch("cli.bill_all_completed", return_value=[]):
        cli.cmd_bill(SimpleNamespace(job_id=None))
    assert "Nothing to bill" in capsys.readouterr().out


def test_cmd_revenue(capsys):
    with patch("cli.get_total_revenue", return_value=42.5):
        with patch("cli.load_billing", return_value=[{"x": 1}]):
            cli.cmd_revenue(SimpleNamespace())
    out = capsys.readouterr().out
    assert "42.5" in out


def test_cmd_failover(capsys):
    job = {"job_id": "j1", "name": "train", "retries": 1}
    host = {"host_id": "h2", "gpu_model": "A100"}
    with patch("cli.failover_and_reassign", return_value=([job], [(job, host)])):
        cli.cmd_failover(SimpleNamespace())
    out = capsys.readouterr().out
    assert "Requeued" in out


def test_cmd_requeue(capsys):
    with patch("cli.requeue_job", return_value={"retries": 1}):
        cli.cmd_requeue(SimpleNamespace(job_id="j1"))
    assert "j1" in capsys.readouterr().out


def test_cmd_ssh_keygen(capsys):
    with patch("cli.generate_ssh_keypair", return_value="/tmp/id_test"):
        with patch("cli.get_public_key", return_value="ssh-ed25519 AAAA"):
            cli.cmd_ssh_keygen(SimpleNamespace())
    out = capsys.readouterr().out
    assert "id_test" in out


def test_cmd_canada_on(capsys):
    with patch("cli.set_canada_only") as setter:
        cli.cmd_canada(SimpleNamespace(on=True, off=False))
    setter.assert_called_once_with(True)
    assert "ON" in capsys.readouterr().out


def test_cmd_pool_list(capsys):
    pool = [
        {
            "host_id": "p1",
            "ip": "10.0.0.3",
            "gpu_model": "RTX 4090",
            "vram_gb": 24,
            "cost_per_hour": 2.0,
            "country": "CA",
            "provisioned": False,
        }
    ]
    with patch("cli.load_autoscale_pool", return_value=pool):
        cli.cmd_pool(SimpleNamespace())
    assert "p1" in capsys.readouterr().out


def test_cmd_autoscale(capsys):
    host = {"host_id": "p1", "gpu_model": "RTX 4090"}
    job = {"name": "train"}
    with patch("cli.autoscale_cycle", return_value=([host], [(job, host)], ["idle1"])):
        cli.cmd_autoscale(SimpleNamespace())
    out = capsys.readouterr().out
    assert "Provisioned" in out


def test_cmd_market(capsys):
    listings = [
        {
            "host_id": "rig1",
            "gpu_model": "RTX 4090",
            "vram_gb": 24,
            "price_per_hour": 2.0,
            "owner": "alice",
            "total_jobs": 3,
            "total_earned": 10.0,
        }
    ]
    with patch("cli.get_marketplace", return_value=listings):
        cli.cmd_market(SimpleNamespace(all=False))
    assert "rig1" in capsys.readouterr().out


def test_cmd_market_list(capsys):
    listing = {
        "host_id": "rig1",
        "gpu_model": "RTX 4090",
        "price_per_hour": 2.0,
        "owner": "alice",
    }
    args = SimpleNamespace(host_id="rig1", gpu="RTX 4090", vram=24, price=2.0, desc="", owner="alice")
    with patch("cli.list_rig", return_value=listing):
        cli.cmd_market_list(args)
    assert "rig1" in capsys.readouterr().out


def test_cmd_build_dockerfile_only(capsys):
    with patch("cli.generate_dockerfile", return_value="FROM nvidia/cuda:12.0\n"):
        args = SimpleNamespace(
            model="llama",
            dockerfile_only=True,
            base="",
            quantize=False,
            context=".",
            push=False,
        )
        cli.cmd_build(args)
    assert "FROM" in capsys.readouterr().out


def test_cmd_builds(capsys):
    builds = [{"model": "llama", "path": "/tmp/llama", "has_dockerfile": True}]
    with patch("cli.list_builds", return_value=builds):
        cli.cmd_builds(SimpleNamespace())
    assert "llama" in capsys.readouterr().out


def test_cmd_tiers(capsys):
    tiers = {"free": {"priority": 0, "multiplier": 1.0, "label": "Free"}}
    with patch("cli.list_tiers", return_value=tiers):
        cli.cmd_tiers(SimpleNamespace())
    assert "free" in capsys.readouterr().out


def test_cmd_reputation(capsys):
    score = MagicMock()
    score.to_dict.return_value = {
        "tier": "gold",
        "total_score": 90,
        "reliability_score": 0.95,
        "components": {"uptime": 40},
    }
    mock_re = MagicMock()
    mock_re.compute_score.return_value = score
    with patch("reputation.get_reputation_engine", return_value=mock_re):
        cli.cmd_reputation(SimpleNamespace(entity_id="host1"))
    assert "gold" in capsys.readouterr().out


def test_cmd_verify_no_record(capsys):
    mock_ve = MagicMock()
    mock_ve.store.get_verification.return_value = None
    with patch("verification.get_verification_engine", return_value=mock_ve):
        cli.cmd_verify(SimpleNamespace(host_id="h1"))
    assert "No verification" in capsys.readouterr().out


def test_cmd_wallet(capsys):
    mock_be = MagicMock()
    mock_be.get_wallet.return_value = {
        "balance_cad": 10.0,
        "total_deposits_cad": 20.0,
        "total_spent_cad": 10.0,
    }
    with patch("billing.get_billing_engine", return_value=mock_be):
        cli.cmd_wallet(SimpleNamespace(customer_id="cust1"))
    assert "10.00" in capsys.readouterr().out


def test_cmd_deposit(capsys):
    mock_be = MagicMock()
    mock_be.deposit.return_value = {"balance_cad": 25.0}
    with patch("billing.get_billing_engine", return_value=mock_be):
        cli.cmd_deposit(SimpleNamespace(customer_id="cust1", amount=15.0))
    assert "25.00" in capsys.readouterr().out


def test_cmd_invoice(capsys):
    invoice = MagicMock(
        subtotal_cad=10.0,
        tax_amount_cad=1.3,
        total_cad=11.3,
        line_items=[{"description": "GPU", "subtotal_cad": 5.0}],
    )
    mock_be = MagicMock()
    mock_be.generate_invoice.return_value = invoice
    with patch("billing.get_billing_engine", return_value=mock_be):
        with patch("cli.time.time", return_value=1_700_000_000):
            cli.cmd_invoice(SimpleNamespace(customer_id="cust1"))
    assert "cust1" in capsys.readouterr().out


def test_cmd_sla(capsys):
    mock_engine = MagicMock()
    mock_engine.get_host_uptime_pct.return_value = 99.5
    mock_engine.get_violations.return_value = []
    with patch("sla.get_sla_engine", return_value=mock_engine):
        cli.cmd_sla(SimpleNamespace(host_id="h1", month=None))
    assert "99.50" in capsys.readouterr().out


def test_cmd_leaderboard(capsys):
    mock_re = MagicMock()
    mock_re.get_leaderboard.return_value = [{"entity_id": "h1", "tier": "gold", "total_score": 90}]
    with patch("reputation.get_reputation_engine", return_value=mock_re):
        cli.cmd_leaderboard(SimpleNamespace(type="host", limit=5))
    assert "h1" in capsys.readouterr().out


def test_cmd_compliance(capsys):
    with patch("billing.PROVINCE_TAX_RATES", {"ON": 0.13}):
        with patch("jurisdiction.PROVINCE_COMPLIANCE", {}):
            with patch("jurisdiction.Province", side_effect=lambda x: x):
                cli.cmd_compliance(SimpleNamespace())
    assert "ON" in capsys.readouterr().out


def test_cmd_logout_removes_token(capsys, tmp_path):
    token_file = tmp_path / "token.json"
    token_file.write_text("{}", encoding="utf-8")
    with patch.object(cli, "TOKEN_FILE", str(token_file)):
        cli.cmd_logout(SimpleNamespace())
    assert not token_file.exists()
    assert "Logged out" in capsys.readouterr().out


def test_cmd_whoami_not_authenticated(capsys):
    with patch("cli.get_api_token", return_value=""):
        cli.cmd_whoami(SimpleNamespace())
    assert "Not authenticated" in capsys.readouterr().out


def test_cmd_whoami_env_token(capsys):
    with patch("cli.get_api_token", return_value="abcdefghijklmnop"):
        with patch.dict("os.environ", {"XCELSIOR_API_TOKEN": "abcdefghijklmnop"}):
            cli.cmd_whoami(SimpleNamespace())
    assert "env bearer" in capsys.readouterr().out


def test_cmd_login_success(capsys):
    device_resp = MagicMock()
    device_resp.raise_for_status = MagicMock()
    device_resp.json.return_value = {
        "device_code": "dc",
        "user_code": "ABCD-1234",
        "verification_uri": "https://xcelsior.ca/verify",
        "interval": 0.01,
        "expires_in": 60,
    }
    token_resp = MagicMock(status_code=200)
    token_resp.json.return_value = {"access_token": "tok", "token_type": "bearer"}
    with patch("webbrowser.open"):
        with patch("requests.post", side_effect=[device_resp, token_resp]):
            with patch("cli._save_token") as save:
                with patch("cli.get_api_url", return_value="https://xcelsior.ca"):
                    cli.cmd_login(SimpleNamespace(api_url=None))
    save.assert_called_once()
    assert "Authenticated" in capsys.readouterr().out


def test_cmd_slurm_status(capsys):
    with patch("slurm_adapter.slurm_status_cli", return_value={"state": "RUNNING"}):
        cli.cmd_slurm_status(SimpleNamespace(job_id="j1", slurm_id="99"))
    assert "RUNNING" in capsys.readouterr().out


def test_cmd_config_set_and_list(capsys, tmp_path):
    cfg_file = tmp_path / "config.toml"
    with patch.object(cli, "CONFIG_FILE", str(cfg_file)):
        with patch.object(cli, "CONFIG_DIR", str(tmp_path)):
            cli.cmd_config(SimpleNamespace(action="set", key="api.url", value="https://xcelsior.ca"))
            cli.cmd_config(SimpleNamespace(action="list", key=None, value=None))
    out = capsys.readouterr().out
    assert "api.url" in out or "xcelsior.ca" in out