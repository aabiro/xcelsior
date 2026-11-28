"""Request validation shared by the listing and launch routes."""

from fastapi import HTTPException
from pydantic import BaseModel, model_validator

from control_plane.contracts import require_whole_gpu_contract


class WholeGPURequest(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def validate_gpu_contract(cls, data):
        if isinstance(data, dict):
            try:
                require_whole_gpu_contract(data)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return data
