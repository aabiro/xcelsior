#!/usr/bin/env python3
"""Glamified Canada map for GPU marketing — no top fade, richer glow."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "frontend" / "public" / "xcelsior-canada-map-0.svg"
OUT = ROOT / "frontend" / "public" / "gpu-fleet" / "canada-sovereign.svg"


def main() -> None:
    svg = SRC.read_text(encoding="utf-8")

    # Drop top fade mask — keep map fully visible at the top
    svg = re.sub(r"<linearGradient id=\"topFadeGrad\"[\s\S]*?</linearGradient>\s*", "", svg)
    svg = re.sub(r"<mask id=\"topFade\"[\s\S]*?</mask>\s*", "", svg)
    svg = svg.replace('<g mask="url(#topFade)">', "<g>")
    svg = svg.replace("mask=\"url(#topFade)\"", "")

    # Richer fills + aurora backdrop
    inject = """
    <radialGradient id="auroraGlow" cx="50%" cy="42%" r="65%">
      <stop offset="0%" stop-color="#00d4ff" stop-opacity="0.22"/>
      <stop offset="45%" stop-color="#7c3aed" stop-opacity="0.12"/>
      <stop offset="100%" stop-color="#dc2626" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="provinceFill" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00d4ff" stop-opacity="0.14"/>
      <stop offset="50%" stop-color="#7c3aed" stop-opacity="0.10"/>
      <stop offset="100%" stop-color="#dc2626" stop-opacity="0.08"/>
    </linearGradient>
    """
    svg = svg.replace("<linearGradient id=\"provinceFill\"", inject + "\n    <!-- replaced provinceFill below -->\n    <linearGradient id=\"provinceFillLegacy\"", 1)

    svg = svg.replace(
        ".arc{fill:none;stroke:url(#arcGrad);stroke-width:1.6;opacity:.55;",
        ".arc{fill:none;stroke:url(#arcGrad);stroke-width:2;opacity:.72;",
    )
    svg = svg.replace(
        'viewBox="-40 120 1080 650"',
        'viewBox="-40 80 1080 690"',
    )

    # Aurora plate behind geography
    svg = svg.replace(
        '<rect class="bg" width="1000" height="750"/>',
        '<rect class="bg" width="1000" height="750"/>'
        '\n  <rect x="-40" y="80" width="1080" height="690" fill="url(#auroraGlow)" opacity="0.9"/>',
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(svg.strip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()