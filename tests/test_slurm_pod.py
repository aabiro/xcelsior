"""Pod listing spec (workers/slurm_pod.py).

Probe outputs below are verbatim from the hpc/slurm-local rehearsal run
(Slurm 25.11, RTX 2060 GPU node); sinfo lines are the adapter's format.
"""

import copy

import pytest

import slurm_pod as sp

GPU_PROBE_OUT = (
    "0, NVIDIA GeForce RTX 3060, 12288, GPU-f22a551d-2bee-2f17-22af-454481edcdc7\n"
    "---nvlink\n"
    "Xcelsior job r1-podg-tower completed at Sat Nov  7 08:44:34 UTC 2026\n"
)
NIC_OUT = "iface=enp38s0\nspeed=1000\nXcelsior job r1-podn completed\n"
INVENTORY = {
    "partition": "xcl",
    "nodes": [
        {
            "name": "asus",
            "partitions": ["xcl"],
            "cpus": 16,
            "memory_mb": 4096,
            "gres": "(null)",
            "state": "idle",
        },
        {
            "name": "tower",
            "partitions": ["xcl"],
            "cpus": 32,
            "memory_mb": 8192,
            "gres": "gpu:rtx_3060:1(S:0)",
            "state": "idle",
        },
    ],
}
DECLARED = {
    "listing_title": "xcl-local reference pod",
    "facility": "home lab",
    "interconnect": {
        "type": "ethernet",
        "overlay": "wireguard",
        "nic_speed_gbps": 1,
        "switch_speed_gbps": None,
        "switch_count": None,
    },
}


def _spec(declared=DECLARED, inventory=INVENTORY, gpu_out=GPU_PROBE_OUT, nic=NIC_OUT):
    return sp.build_pod_spec(
        cluster_id="xcl-local",
        partition="xcl",
        inventory=inventory,
        gpu_probes={"tower": {"probe": sp.parse_gpu_probe(gpu_out), "slurm_job_id": "31"}},
        nic_probes={
            "tower": {"probe": sp.parse_nic_probe(nic), "slurm_job_id": "32"},
            "asus": {"probe": sp.parse_nic_probe(nic), "slurm_job_id": "33"},
        },
        declared=declared,
        scheduler_version="slurm 25.11.2",
    )


@pytest.mark.parametrize(
    "gres,expected",
    [
        ("gpu:rtx_3060:1(S:0)", [{"type": "rtx_3060", "count": 1}]),
        ("gpu:4", [{"type": None, "count": 4}]),
        (
            "gpu:h100:8(S:0-1),gpu:a100:2",
            [{"type": "h100", "count": 8}, {"type": "a100", "count": 2}],
        ),
        ("(null)", []),
    ],
)
def test_parse_gres(gres, expected):
    assert sp.parse_gres(gres) == expected


def test_parse_gpu_probe_no_nvlink():
    p = sp.parse_gpu_probe(GPU_PROBE_OUT)
    assert p["gpus"][0]["model"] == "NVIDIA GeForce RTX 3060"
    assert p["gpus"][0]["memory_mib"] == 12288
    assert p["nvlink_active_links"] == 0


def test_parse_gpu_probe_counts_active_links():
    out = "0, NVIDIA H100 80GB HBM3, 81559, GPU-a\n1, NVIDIA H100 80GB HBM3, 81559, GPU-b\n---nvlink\nGPU 0: NVIDIA H100\n\t Link 0: 26.562 GB/s\n\t Link 1: 26.562 GB/s\n"
    p = sp.parse_gpu_probe(out)
    assert len(p["gpus"]) == 2 and p["nvlink_active_links"] == 2


def test_parse_nic_probe_unknown_speed():
    assert sp.parse_nic_probe("iface=wlp3s0\nspeed=unknown\n") == {
        "interface": "wlp3s0",
        "speed_mbps": None,
    }
    assert sp.parse_nic_probe("iface=eth0\nspeed=-1\n")["speed_mbps"] is None


def test_verified_spec_from_real_shapes():
    spec = _spec()
    assert sp.validate_pod_spec(spec) == []
    assert spec["verification"]["status"] == "verified", spec["verification"]
    assert spec["node_count"]["value"] == 2
    assert spec["gpu"]["count"]["value"] == 1
    assert spec["gpu"]["model"]["value"] == "NVIDIA GeForce RTX 3060"
    assert spec["gpu"]["memory_gb"]["value"] == 12.0
    assert spec["gpu"]["nvlink_domain_size"]["value"] == 1
    assert spec["interconnect"]["nic_speed_gbps"]["value"] == 1.0
    assert spec["gpu"]["model"]["evidence"] == "slurm jobs 31"


def test_every_required_field_carries_provenance():
    spec = _spec()
    assert spec["interconnect"]["switch_speed_gbps"]["source"] == "declared"
    assert spec["gpu"]["count"]["source"] == "observed"


def test_overclaimed_nic_speed_fails():
    d = copy.deepcopy(DECLARED)
    d["interconnect"]["nic_speed_gbps"] = 800
    spec = _spec(declared=d)
    failed = {c["check"] for c in spec["verification"]["checks"] if not c["ok"]}
    assert spec["verification"]["status"] == "failed"
    assert "nic_speed_declared_not_above_observed" in failed


def test_gres_without_a_working_gpu_fails():
    spec = _spec(gpu_out="---nvlink\n")
    failed = {c["check"] for c in spec["verification"]["checks"] if not c["ok"]}
    assert "gpu_count_matches_probes" in failed


def test_drained_node_fails():
    inv = copy.deepcopy(INVENTORY)
    inv["nodes"][0]["state"] = "drained"
    spec = _spec(inventory=inv)
    assert spec["verification"]["status"] == "failed"


def test_unobservable_nic_is_not_verified():
    spec = _spec(nic="iface=wlp3s0\nspeed=unknown\n")
    failed = {c["check"] for c in spec["verification"]["checks"] if not c["ok"]}
    assert "nic_speed_observed" in failed


def test_validate_catches_shape_errors():
    spec = _spec()
    del spec["gpu"]["model"]["source"]
    spec["spec_version"] = "v-1"
    errs = sp.validate_pod_spec(spec)
    assert any("spec_version" in e for e in errs)
    assert any("gpu.model" in e for e in errs)
