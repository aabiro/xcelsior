"""The serverless model presets are the same list in the API and the dashboard.

2026-11-22: the dashboard's Quick Deploy offered Voyage-3, Voyage-3-code and
Cohere Rerank v3.5. None has public weights, so an endpoint on one could never
start. The lists now hold open-weight models only, and must agree: a preset the
dashboard shows but the API does not price or stage is a dead button.
"""

from __future__ import annotations

import re
from pathlib import Path

from serverless.registry_models import DOC_PRESET_MODELS

ROOT = Path(__file__).resolve().parents[1]
CONSTANTS = ROOT / "api" / "frontend" / "src" / "features" / "serverless" / "constants.ts"

# Hosted-API models with no downloadable weights. None may come back as a preset.
CLOSED_WEIGHT = ("voyage-3", "cohere-rerank")


def _dashboard_presets() -> list[str]:
    source = CONSTANTS.read_text(encoding="utf-8")
    block = source.split("export const PRESET_MODELS = [", 1)[1].split("] as const;", 1)[0]
    return re.findall(r'\{\s*id:\s*"([^"]+)"', block)


def test_the_dashboard_lists_exactly_the_api_presets():
    assert _dashboard_presets() == list(DOC_PRESET_MODELS)


def test_no_preset_is_a_closed_weight_model():
    for ref in (*DOC_PRESET_MODELS, *_dashboard_presets()):
        assert not any(name in ref.lower() for name in CLOSED_WEIGHT), ref
