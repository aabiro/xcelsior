from unittest.mock import MagicMock, patch

from host_acceptance import DEFAULT_DOCKER_PROBE_IMAGE, probe_local_host


GOOD_VERSIONS = {
    "runc": "1.2.4",
    "docker": "25.0.0",
    "nvidia_driver": "555.0",
    "nvidia_toolkit": "1.17.8",
}

RTX3060_GPU = {
    "index": 0,
    "gpu_model": "NVIDIA GeForce RTX 3060",
    "memory_total_gb": 12.0,
    "memory_free_gb": 11.2,
    "compute_capability": "8.6",
    "driver_version": "555.0",
    "temperature_c": 54,
}


def _patch_probe(gpus=None, versions=None, admitted=True):
    version_payload = GOOD_VERSIONS if versions is None else versions
    admission = {
        "host_id": "tower-server",
        "admitted": admitted,
        "rejection_reasons": [] if admitted else ["old driver"],
        "recommended_runtime": "runc",
    }
    return (
        patch("host_acceptance.get_local_versions", return_value=version_payload),
        patch("host_acceptance.admit_node", return_value=(admitted, admission)),
        patch("host_acceptance.recommend_runtime", return_value=("runc", "test runtime")),
        patch("nvml_telemetry.nvml_init", return_value=True),
        patch("nvml_telemetry.collect_all_gpus", return_value=gpus or [RTX3060_GPU]),
    )


def test_probe_local_host_ready_for_rtx3060():
    patches = _patch_probe()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        report = probe_local_host(
            host_id="tower-server",
            expected_gpu_model="RTX 3060",
            min_vram_gb=12,
        )

    assert report["ready"] is True
    assert report["recommended_runtime"] == "runc"
    assert report["gpus"][0]["gpu_model"] == "NVIDIA GeForce RTX 3060"
    assert {check["name"]: check["ok"] for check in report["checks"]}["expected_gpu"] is True


def test_probe_local_host_fails_when_versions_missing_even_if_admission_is_permissive():
    patches = _patch_probe(versions={})
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        report = probe_local_host(host_id="tower-server", expected_gpu_model="RTX 3060")

    by_name = {check["name"]: check for check in report["checks"]}
    assert report["ready"] is False
    assert by_name["versions_present"]["ok"] is False
    assert "missing:" in by_name["versions_present"]["detail"]


def test_probe_local_host_fails_expected_gpu_mismatch():
    patches = _patch_probe(gpus=[{**RTX3060_GPU, "gpu_model": "NVIDIA GeForce RTX 2060"}])
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        report = probe_local_host(host_id="tower-server", expected_gpu_model="RTX 3060")

    by_name = {check["name"]: check for check in report["checks"]}
    assert report["ready"] is False
    assert by_name["expected_gpu"]["ok"] is False


def test_probe_local_host_fails_minimum_vram():
    patches = _patch_probe()
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        report = probe_local_host(host_id="tower-server", min_vram_gb=24)

    by_name = {check["name"]: check for check in report["checks"]}
    assert report["ready"] is False
    assert by_name["minimum_vram"]["ok"] is False


def test_docker_probe_uses_configured_image():
    patches = _patch_probe()
    mock_result = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 3060, 12288", stderr="")
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patch("host_acceptance.subprocess.run", return_value=mock_result) as run,
    ):
        report = probe_local_host(host_id="tower-server", docker_probe=True)

    assert report["ready"] is True
    assert run.call_args.args[0][5] == DEFAULT_DOCKER_PROBE_IMAGE
    by_name = {check["name"]: check for check in report["checks"]}
    assert by_name["docker_gpu_probe"]["ok"] is True
