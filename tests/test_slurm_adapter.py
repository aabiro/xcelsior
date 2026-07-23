"""Tests for Xcelsior Slurm adapter — job translation, sbatch generation, state mapping."""

import json
import os
import uuid
import subprocess
import tempfile

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from slurm_adapter import (
    CLUSTER_PROFILES,
    GPU_VRAM_TABLE,
    SLURM_STATE_MAP,
    _estimate_gpus_needed,
    _estimate_walltime,
    _get_profile,
    _priority_to_qos,
    cancel_slurm_job,
    get_slurm_job_status,
    is_slurm_available,
    register_slurm_job,
    slurm_submit_cli,
    submit_to_slurm,
    xcelsior_job_to_sbatch,
)

# ── Cluster Profiles ─────────────────────────────────────────────────


class TestClusterProfiles:
    """Verify all cluster profiles are well-formed."""

    def test_required_profiles_exist(self):
        for name in ("nibi", "graham", "narval", "generic"):
            assert name in CLUSTER_PROFILES, f"Missing profile: {name}"

    def test_profile_keys(self):
        required_keys = {
            "partition_gpu",
            "gpu_type",
            "modules",
            "container_runtime",
            "account_env",
            "scratch_dir",
            "project_dir",
            "gpus_per_node",
        }
        for name, profile in CLUSTER_PROFILES.items():
            for key in required_keys:
                assert key in profile, f"{name} missing key: {key}"

    def test_nibi_profile(self):
        p = CLUSTER_PROFILES["nibi"]
        assert p["gpu_type"] == "a100"
        assert p["gpus_per_node"] == 4
        assert p["container_runtime"] == "apptainer"

    def test_graham_profile(self):
        p = CLUSTER_PROFILES["graham"]
        assert p["container_runtime"] == "apptainer"
        assert "gpus_per_node" in p

    def test_get_profile_default(self):
        profile = _get_profile()
        # Should return generic when no env is set
        assert profile is not None
        assert "partition_gpu" in profile

    def test_get_profile_by_name(self):
        profile = _get_profile("nibi")
        assert profile["gpu_type"] == "a100"


# ── GPU VRAM Table ────────────────────────────────────────────────────


class TestGPUVRAMTable:
    def test_common_gpus_present(self):
        for gpu in ["a100", "h100", "rtx_4090", "rtx_3090", "t4", "v100"]:
            assert gpu in GPU_VRAM_TABLE, f"Missing GPU: {gpu}"

    def test_a100_vram(self):
        assert GPU_VRAM_TABLE["a100"] == 40

    def test_rtx_4090_vram(self):
        assert GPU_VRAM_TABLE["rtx_4090"] == 24


# ── GPU Estimation ────────────────────────────────────────────────────


class TestEstimateGPUs:
    """Test ceil-division GPU estimation logic."""

    def test_zero_vram_returns_one(self):
        assert _estimate_gpus_needed(0, "a100") == 1

    def test_negative_vram_returns_one(self):
        assert _estimate_gpus_needed(-10, "a100") == 1

    def test_exact_fit_single_gpu(self):
        assert _estimate_gpus_needed(40, "a100") == 1

    def test_needs_two_gpus(self):
        assert _estimate_gpus_needed(60, "a100") == 2  # 60/40 = ceil 2

    def test_small_model_single_gpu(self):
        assert _estimate_gpus_needed(8, "rtx_4090") == 1

    def test_large_model_multiple_gpus(self):
        assert _estimate_gpus_needed(48, "rtx_4090") == 2  # 48/24 = 2

    def test_unknown_gpu_uses_16gb_default(self):
        assert _estimate_gpus_needed(32, "unknown_gpu") == 2  # 32/16 = 2
        assert _estimate_gpus_needed(48, "unknown_gpu") == 3  # 48/16 = 3


# ── Priority to QoS Mapping ──────────────────────────────────────────


class TestPriorityToQoS:
    def test_premium_qos(self):
        assert _priority_to_qos(90) == "premium"
        assert _priority_to_qos(100) == "premium"

    def test_normal_qos(self):
        assert _priority_to_qos(50) == "normal"
        assert _priority_to_qos(89) == "normal"

    def test_low_qos(self):
        assert _priority_to_qos(10) == "low"
        assert _priority_to_qos(49) == "low"

    def test_default_qos(self):
        assert _priority_to_qos(0) == "default"
        assert _priority_to_qos(9) == "default"
        assert _priority_to_qos(-1) == "default"


# ── Walltime Estimation ──────────────────────────────────────────────


class TestEstimateWalltime:
    def test_large_model_24h(self):
        assert _estimate_walltime("llama-70b-chat") == "24:00:00"
        assert _estimate_walltime("falcon-180b") == "24:00:00"

    def test_medium_model_12h(self):
        assert _estimate_walltime("llama-13b-finetune") == "12:00:00"
        assert _estimate_walltime("mistral-7b-v0.1") == "12:00:00"
        assert _estimate_walltime("codellama-34b") == "12:00:00"

    def test_small_model_6h(self):
        assert _estimate_walltime("llama-7b-base") == "6:00:00"
        assert _estimate_walltime("phi-2-instruct") == "6:00:00"
        assert _estimate_walltime("gemma-2b") == "6:00:00"

    def test_unknown_model_default(self):
        assert _estimate_walltime("custom-model-xyz") == "4:00:00"

    def test_custom_default(self):
        assert _estimate_walltime("custom", default="8:00:00") == "8:00:00"


# ── Slurm State Mapping ──────────────────────────────────────────────


class TestSlurmStateMap:
    """Verify all expected Slurm states map to valid Xcelsior states."""

    VALID_XCELSIOR_STATES = {
        "queued",
        "running",
        "completed",
        "failed",
        "cancelled",
        "preempted",
    }

    def test_all_states_mapped(self):
        expected_slurm = {
            "PENDING",
            "RUNNING",
            "COMPLETING",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "NODE_FAIL",
            "PREEMPTED",
            "SUSPENDED",
            "OUT_OF_MEMORY",
        }
        for state in expected_slurm:
            assert state in SLURM_STATE_MAP, f"Missing mapping: {state}"

    def test_all_values_are_valid(self):
        for slurm_state, xcelsior_state in SLURM_STATE_MAP.items():
            assert (
                xcelsior_state in self.VALID_XCELSIOR_STATES
            ), f"{slurm_state} maps to invalid state: {xcelsior_state}"

    def test_specific_mappings(self):
        assert SLURM_STATE_MAP["PENDING"] == "queued"
        assert SLURM_STATE_MAP["RUNNING"] == "running"
        assert SLURM_STATE_MAP["COMPLETED"] == "completed"
        assert SLURM_STATE_MAP["FAILED"] == "failed"
        assert SLURM_STATE_MAP["PREEMPTED"] == "preempted"
        assert SLURM_STATE_MAP["TIMEOUT"] == "failed"
        assert SLURM_STATE_MAP["OUT_OF_MEMORY"] == "failed"


# ── Job Translation (sbatch generation) ──────────────────────────────


class TestJobTranslation:
    """Test xcelsior_job_to_sbatch for various profiles and job configs."""

    def _make_job(self, **overrides):
        job = {
            "job_id": "test-123",
            "name": "llama-7b",
            "vram_needed_gb": 16,
            "priority": 50,
            "tier": "premium",
            "image": "ghcr.io/xcelsior/llama:latest",
        }
        job.update(overrides)
        return job

    def test_generates_sbatch_header(self):
        script, meta = xcelsior_job_to_sbatch(self._make_job(), "generic")
        assert script.startswith("#!/bin/bash")
        assert "#SBATCH --job-name=xcl-test-123" in script
        assert "#SBATCH --time=" in script
        assert "#SBATCH --gres=gpu:" in script

    def test_metadata_returned(self):
        script, meta = xcelsior_job_to_sbatch(self._make_job(), "nibi")
        assert meta["job_id"] == "test-123"
        assert meta["profile"] == "nibi"
        assert meta["gpu_type"] == "a100"
        assert meta["nodes"] >= 1
        assert meta["qos"] == "normal"  # priority=50

    def test_nibi_uses_apptainer(self):
        script, _ = xcelsior_job_to_sbatch(self._make_job(), "nibi")
        assert "apptainer" in script
        assert "module load" in script

    def test_generic_uses_docker(self):
        script, _ = xcelsior_job_to_sbatch(self._make_job(), "generic")
        assert "docker" in script.lower() or "Inference Script" in script

    def test_gpu_count_in_gres(self):
        job = self._make_job(vram_needed_gb=60)  # Needs 2 A100s (40GB each)
        script, meta = xcelsior_job_to_sbatch(job, "nibi")
        assert meta["num_gpus"] == 2
        assert ":2" in script  # gpu:a100:2

    def test_explicit_num_gpus(self):
        job = self._make_job(num_gpus=4)
        script, meta = xcelsior_job_to_sbatch(job, "nibi")
        assert meta["num_gpus"] == 4

    def test_walltime_heuristic(self):
        job = self._make_job(name="llama-70b-chat")
        _, meta = xcelsior_job_to_sbatch(job, "generic")
        assert meta["walltime"] == "24:00:00"

    def test_env_vars_in_script(self):
        script, _ = xcelsior_job_to_sbatch(self._make_job(), "generic")
        assert "XCELSIOR_JOB_ID=test-123" in script
        assert 'XCELSIOR_TIER="premium"' in script

    def test_extra_env_vars(self):
        extra = {"MY_VAR": "hello", "NUM_WORKERS": "4"}
        script, _ = xcelsior_job_to_sbatch(self._make_job(), "generic", extra_env=extra)
        assert 'MY_VAR="hello"' in script
        assert 'NUM_WORKERS="4"' in script

    def test_multi_node_allocation(self):
        # 8 GPUs on a cluster with 4 per node → 2 nodes
        job = self._make_job(num_gpus=8)
        script, meta = xcelsior_job_to_sbatch(job, "nibi")
        assert meta["nodes"] == 2
        assert "#SBATCH --nodes=2" in script

    def test_memory_calculation(self):
        job = self._make_job(num_gpus=2)
        script, _ = xcelsior_job_to_sbatch(job, "generic")
        # 4 + (8 * 2) = 20G
        assert "#SBATCH --mem=20G" in script

    def test_premium_qos(self):
        job = self._make_job(priority=95)
        script, meta = xcelsior_job_to_sbatch(job, "generic")
        assert meta["qos"] == "premium"
        assert "#SBATCH --qos=premium" in script

    def test_no_image_uses_inference_script(self):
        job = self._make_job(image="")
        script, _ = xcelsior_job_to_sbatch(job, "generic")
        assert "Inference Script" in script


# ── Slurm Submission (dry run) ────────────────────────────────────────


class TestSubmitDryRun:
    def test_dry_run_returns_script(self):
        job = {"job_id": "dry-1", "name": "test", "vram_needed_gb": 8}
        result = submit_to_slurm(job, profile_name="generic", dry_run=True)
        assert result["dry_run"] is True
        assert "script" in result
        assert "metadata" in result

    def test_no_slurm_returns_error(self, monkeypatch):
        monkeypatch.setattr(
            "slurm_adapter.is_slurm_available",
            lambda: False,
        )
        job = {"job_id": "no-slurm", "name": "test", "vram_needed_gb": 8}
        result = submit_to_slurm(job, profile_name="generic", dry_run=False)
        assert "error" in result


# ── Slurm CLI Wrapper ────────────────────────────────────────────────


class TestSlurmCLI:
    def test_dry_run_output(self):
        job = {"job_id": "cli-1", "name": "test-model", "vram_needed_gb": 16}
        output = slurm_submit_cli(job, profile="generic", dry_run=True)
        assert "sbatch script" in output.lower() or "SBATCH" in output


# ── Slurm Job Map Persistence ────────────────────────────────────────


class TestSlurmJobMap:
    def test_register_and_load(self, monkeypatch):
        """Mappings round-trip through PostgreSQL, not a JSON file.

        This test used to monkeypatch `SLURM_MAP_FILE` — a leftover from
        the file-backed era that migration 060 replaced. The constant is
        gone (companion §10.1: "A JSON file cannot safely coordinate
        several processes"), so the assertion now drives the real table.
        """
        import slurm_adapter
        from db import pg_transaction

        job_id = f"xcelsior-{uuid.uuid4().hex[:8]}"
        monkeypatch.setattr(slurm_adapter, "_slurm_job_map", {})
        try:
            register_slurm_job(job_id, "12345")

            slurm_adapter._slurm_job_map = {}  # prove it reloads from the DB
            slurm_adapter._load_slurm_map()
            assert slurm_adapter._slurm_job_map[job_id] == "12345"
        finally:
            with pg_transaction() as conn:
                conn.execute(
                    "DELETE FROM slurm_job_mappings WHERE xcelsior_job_id = %s",
                    (job_id,),
                )


# ── is_slurm_available ───────────────────────────────────────────────


class TestSlurmAvailability:
    def test_returns_false_when_sinfo_missing(self, monkeypatch):
        def mock_run(*args, **kwargs):
            raise FileNotFoundError("sinfo not found")

        monkeypatch.setattr(subprocess, "run", mock_run)
        assert is_slurm_available() is False


# ── get_slurm_job_status ─────────────────────────────────────────────


def _completed(stdout="", returncode=0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class TestSlurmJobStatus:
    def test_no_slurm_returns_error(self, monkeypatch):
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)

        def mock_run(argv, *a, **k):
            return _completed("", returncode=0)  # empty squeue + empty sacct

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = get_slurm_job_status("12345")
        assert result["xcelsior_state"] == "failed"
        assert "not found" in result["error"]

    def test_running_job_parsed_from_squeue(self, monkeypatch):
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)
        # %i %T %M %l %D %R  — running job, %R is a single-token nodelist
        line = "12345 RUNNING 1:02:03 4:00:00 1 node007"

        def mock_run(argv, *a, **k):
            assert argv[0] == "squeue"
            return _completed(line + "\n")

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = get_slurm_job_status("12345")
        assert result["slurm_state"] == "RUNNING"
        assert result["xcelsior_state"] == SLURM_STATE_MAP.get("RUNNING", "running")
        assert result["elapsed"] == "1:02:03"
        assert result["time_limit"] == "4:00:00"
        assert result["nodes"] == "1"
        assert result["reason"] == "node007"

    def test_pending_reason_with_spaces_not_truncated(self, monkeypatch):
        """Regression: %R can contain spaces; it must not be split apart."""
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)
        # Pending job whose reason is a multi-word parenthesised string.
        line = "12346 PENDING 0:00 8:00:00 2 (launch failed requeued held)"

        def mock_run(argv, *a, **k):
            return _completed(line + "\n")

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = get_slurm_job_status("12346")
        assert result["slurm_state"] == "PENDING"
        assert result["nodes"] == "2"
        assert result["reason"] == "(launch failed requeued held)"

    def test_falls_back_to_sacct_for_completed(self, monkeypatch):
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)
        calls = {"n": 0}

        def mock_run(argv, *a, **k):
            calls["n"] += 1
            if argv[0] == "squeue":
                return _completed("")  # not in queue anymore
            # sacct --parsable2; skip the ".batch" sub-step row
            return _completed(
                "12347.batch|COMPLETED|0:30|0:0|1024K|2048K\n"
                "12347|COMPLETED|0:30|0:0|1024K|2048K\n"
            )

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = get_slurm_job_status("12347")
        assert calls["n"] == 2  # squeue then sacct
        assert result["slurm_state"] == "COMPLETED"
        assert result["exit_code"] == "0:0"
        assert result["max_rss"] == "1024K"

    def test_timeout_returns_error(self, monkeypatch):
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)

        def mock_run(argv, *a, **k):
            raise subprocess.TimeoutExpired(cmd="squeue", timeout=10)

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = get_slurm_job_status("12348")
        assert "timed out" in result["error"]


class TestCancelSlurmJob:
    def test_cancel_success(self, monkeypatch):
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed("", returncode=0))
        result = cancel_slurm_job("12345")
        assert result["cancelled"] is True

    def test_cancel_failure_surfaces_stderr(self, monkeypatch):
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)

        def mock_run(*a, **k):
            return subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="Invalid job id"
            )

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = cancel_slurm_job("99999")
        assert "Invalid job id" in result["error"]
