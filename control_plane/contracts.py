"""The GPU allocation contracts the scheduler can enforce."""

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any


class GPUContract(StrEnum):
    WHOLE_GPU = "whole_gpu"


CONTRACT_WHOLE_GPU = GPUContract.WHOLE_GPU.value
WHOLE_GPU_REQUIRED = "VRAM budget contracts are not enforceable on this GPU; whole-GPU only."
_BUDGET_FIELDS = frozenset(
    {
        "vram_budget",
        "vram_budget_gb",
        "vram_budget_mb",
        "vram_limit_gb",
        "vram_limit_mb",
        "gpu_fraction",
        "fractional_capacity",
        "fractional_gpu",
    }
)
_COUNT_FIELDS = ("spot_gpu_slots", "gpu_count", "gpu_count_total", "num_gpus")


def require_whole_gpu_count(value: object) -> None:
    """Reject a fractional count before any integer coercion."""
    if value is None:
        return
    try:
        count = Decimal(str(value))
    except InvalidOperation:
        # Leave ordinary type errors to the request schema.
        return
    if count.is_finite() and count != count.to_integral_value():
        raise ValueError(WHOLE_GPU_REQUIRED)


def _validate_capacity(payload: Mapping[str, Any]) -> None:
    if _BUDGET_FIELDS.intersection(payload) or any(
        payload.get(field) not in (None, CONTRACT_WHOLE_GPU)
        for field in ("gpu_contract", "allocation_mode", "contract")
    ):
        raise ValueError(WHOLE_GPU_REQUIRED)
    for field in _COUNT_FIELDS:
        require_whole_gpu_count(payload.get(field))


def require_whole_gpu_contract(payload: Mapping[str, Any]) -> None:
    """Reject explicit budgets for every model while whole GPU is the sole contract.

    Hardware VRAM and minimum VRAM requirements are independent of the contract.
    """
    _validate_capacity(payload)
    capabilities = payload.get("capabilities")
    if isinstance(capabilities, Mapping):
        # Agent registration accepts a capabilities envelope as well.
        _validate_capacity(capabilities)
