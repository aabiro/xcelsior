#!/usr/bin/env python3
"""Export only GPU model names and minimum VRAM from the private Stripe snapshot.

Run after changing the compute catalog; --check verifies that hardware metadata
is current without writing anything. No prices or Stripe identifiers are copied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def hardware_catalog(manifest: dict) -> dict[str, int]:
    models: dict[str, int] = {}
    for row in manifest["service_tier_catalog"]["prices"]:
        model = row["variant"]["gpu_model"]
        vram = row["variant"]["vram_gb"]
        if not isinstance(model, str) or not model.strip() or type(vram) is not int or vram <= 0:
            raise ValueError("Invalid GPU hardware metadata in compute catalog")
        models[model] = min(models.get(model, vram), vram)
    if not models:
        raise ValueError("Compute catalog has no GPU hardware metadata")
    return dict(sorted(models.items()))


def write_hardware_catalog(manifest: dict, output: Path) -> None:
    output.write_text(json.dumps(hardware_catalog(manifest), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "billing/stripe_catalog.json")
    parser.add_argument("--output", type=Path, default=ROOT / "hosts/gpu_catalog.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.check:
        if hardware_catalog(manifest) != json.loads(args.output.read_text()):
            parser.error("Public GPU hardware catalog is stale; regenerate and commit it")
    else:
        write_hardware_catalog(manifest, args.output)


if __name__ == "__main__":
    main()
