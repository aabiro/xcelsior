import pytest

from control_plane.contracts import (
    CONTRACT_WHOLE_GPU,
    WHOLE_GPU_REQUIRED,
    require_whole_gpu_contract,
)
from control_plane.scheduler.filters import device_outside_use


def test_hardware_capacity_and_minimum_vram_are_valid_with_whole_gpus():
    require_whole_gpu_contract(
        {
            "gpu_contract": CONTRACT_WHOLE_GPU,
            "total_vram_gb": 11.5,
            "vram_needed_gb": 4,
            "spot_gpu_slots": 0,
            "num_gpus": 2,
        }
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"gpu_contract": "vram_budget"},
        {"gpu_count_total": "0.5"},
        {"capabilities": {"vram_budget_mb": 4096}},
        {"num_gpus": 1.5},
    ],
)
def test_unsupported_gpu_contract_is_rejected(payload):
    with pytest.raises(ValueError, match=WHOLE_GPU_REQUIRED):
        require_whole_gpu_contract(payload)


def test_vram_usage_without_a_compute_process_does_not_block_placement():
    assert device_outside_use(0, 4096) is None
    assert device_outside_use(1, 0) == {"processes": 1, "vram_mb": 0}
