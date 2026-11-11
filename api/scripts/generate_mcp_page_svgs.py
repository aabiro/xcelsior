#!/usr/bin/env python3
"""Generate MCP marketing + settings SVG assets.

Premium agent->MCP->GPU visuals matching the gpu-fleet / features style
(glass panels, radial lighting, glows).

The three "How it works" illustrations come from generate_site_illustrations.py,
and the client marks (Claude, Cursor, VS Code) are the real logos in
frontend/public/logos/clients, tinted by the page with a CSS mask.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "mcp"
OUT.mkdir(parents=True, exist_ok=True)

DEFS = """
    <linearGradient id="brand" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#22e0ff"/>
      <stop offset="48%" stop-color="#7c3aed"/>
      <stop offset="100%" stop-color="#f43f5e"/>
    </linearGradient>
    <linearGradient id="brandSoft" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0%" stop-color="#22e0ff" stop-opacity="0.14"/>
      <stop offset="100%" stop-color="#7c3aed" stop-opacity="0.06"/>
    </linearGradient>
    <radialGradient id="spot" cx="50%" cy="34%" r="75%">
      <stop offset="0%" stop-color="#1b2740"/>
      <stop offset="100%" stop-color="#070b15"/>
    </radialGradient>
    <linearGradient id="glass" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.14"/>
      <stop offset="24%" stop-color="#ffffff" stop-opacity="0.04"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="panel" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#16233c"/>
      <stop offset="100%" stop-color="#0b1322"/>
    </linearGradient>
    <radialGradient id="core" cx="50%" cy="42%" r="62%">
      <stop offset="0%" stop-color="#5eeaff"/>
      <stop offset="42%" stop-color="#2a8cff"/>
      <stop offset="100%" stop-color="#0b1b3a"/>
    </radialGradient>
    <linearGradient id="violet" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#a78bfa"/>
      <stop offset="100%" stop-color="#7c3aed"/>
    </linearGradient>
    <linearGradient id="gold" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#fde68a"/>
      <stop offset="100%" stop-color="#d4a233"/>
    </linearGradient>
    <linearGradient id="emerald" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#34d399"/>
      <stop offset="100%" stop-color="#10b981"/>
    </linearGradient>
    <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="bloom" x="-120%" y="-120%" width="340%" height="340%">
      <feGaussianBlur stdDeviation="22"/>
    </filter>
    <filter id="cardShadow" x="-40%" y="-40%" width="180%" height="180%">
      <feDropShadow dx="0" dy="9" stdDeviation="13" flood-color="#000" flood-opacity="0.45"/>
    </filter>
    <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">
      <path d="M24 0H0V24" fill="none" stroke="#22e0ff" stroke-opacity="0.06" stroke-width="1"/>
    </pattern>
"""


def mcp_hex(cx, cy, s=1.0):
    return f"""
  <g transform="translate({cx},{cy}) scale({s})" filter="url(#cardShadow)">
    <polygon points="0,-46 40,-23 40,23 0,46 -40,23 -40,-23" fill="url(#panel)" stroke="url(#brand)" stroke-width="1.75"/>
    <polygon points="0,-46 40,-23 40,23 0,46 -40,23 -40,-23" fill="url(#glass)"/>
    <polygon points="0,-26 22,-13 22,13 0,26 -22,13 -22,-13" fill="none" stroke="#7cf2ff" stroke-opacity="0.5" stroke-width="1.5"/>
    <text x="0" y="6" text-anchor="middle" fill="#eafcff" font-family="ui-monospace,monospace" font-size="14" font-weight="700">MCP</text>
  </g>"""


def agent_node(cx, cy):
    return f"""
  <g transform="translate({cx},{cy})" filter="url(#cardShadow)">
    <rect x="-44" y="-34" width="88" height="60" rx="14" fill="url(#panel)" stroke="url(#violet)" stroke-width="1.5"/>
    <rect x="-44" y="-34" width="88" height="60" rx="14" fill="url(#glass)"/>
    <path d="M-22,26 L-22,40 L-4,26 Z" fill="url(#panel)" stroke="url(#violet)" stroke-width="1.5"/>
    <g fill="#c4b5fd" opacity="0.85"><rect x="-30" y="-18" width="48" height="6" rx="3"/><rect x="-30" y="-4" width="60" height="6" rx="3" opacity="0.6"/><rect x="-30" y="10" width="34" height="6" rx="3" opacity="0.4"/></g>
  </g>"""


def gpu_node(cx, cy):
    return f"""
  <g transform="translate({cx},{cy})" filter="url(#cardShadow)">
    <rect x="-46" y="-36" width="92" height="72" rx="12" fill="url(#panel)" stroke="url(#brand)" stroke-width="1.5"/>
    <rect x="-46" y="-36" width="92" height="72" rx="12" fill="url(#glass)"/>
    <rect x="-30" y="-22" width="60" height="44" rx="7" fill="url(#core)" opacity="0.85" stroke="#7cf2ff" stroke-opacity="0.4"/>
    <g stroke="#0a1322" stroke-opacity="0.5" stroke-width="1"><path d="M-10,-22 V22 M10,-22 V22 M-30,0 H30"/></g>
    <circle cx="0" cy="0" r="7" fill="#eafcff" opacity="0.9" filter="url(#glow)"/>
  </g>"""


def beam(x1, x2, y):
    return f'<path d="M{x1},{y} H{x2}" stroke="url(#brand)" stroke-width="2.5" stroke-dasharray="2 9" stroke-linecap="round" opacity="0.7" filter="url(#glow)"/>'


def frame(inner, w, h, rx=20):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" fill="none">
  <defs>{DEFS}</defs>
  <rect width="{w}" height="{h}" rx="{rx}" fill="url(#spot)"/>
  <rect width="{w}" height="{h}" rx="{rx}" fill="url(#grid)"/>
{inner}
  <rect width="{w}" height="{h}" rx="{rx}" fill="url(#brandSoft)"/>
</svg>"""


SLIDES = {
    # Hero: agent -> MCP -> GPU bridge, no baked title
    "hero-agent-gpu.svg": frame(f"""
  <circle cx="400" cy="60" r="78" fill="#7c3aed" filter="url(#bloom)" opacity="0.16"/>
  <circle cx="80" cy="220" r="70" fill="#22e0ff" filter="url(#bloom)" opacity="0.12"/>
  {beam(118, 196, 140)}{beam(284, 372, 140)}
  {agent_node(74, 140)}
  {mcp_hex(240, 140, 1.0)}
  {gpu_node(418, 140)}
""", 480, 280, 24),

    # Guardrails: shield + wallet/estimate gauges
    "flow-guardrails.svg": frame("""
  <circle cx="160" cy="70" r="64" fill="#fbbf24" filter="url(#bloom)" opacity="0.12"/>
  <g filter="url(#cardShadow)">
    <path d="M160,30 L206,48 V92 C206,124 160,146 160,146 C160,146 114,124 114,92 V48 Z" fill="url(#panel)" stroke="url(#gold)" stroke-width="1.75"/>
    <path d="M160,30 L206,48 V92 C206,124 160,146 160,146 C160,146 114,124 114,92 V48 Z" fill="url(#glass)"/>
    <path d="M146,86 L156,98 L178,72" stroke="url(#gold)" stroke-width="3.5" fill="none" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)"/>
  </g>
  <rect x="52" y="160" width="84" height="10" rx="5" fill="#1a2740"/>
  <rect x="52" y="160" width="54" height="10" rx="5" fill="url(#gold)"/>
  <rect x="184" y="160" width="84" height="10" rx="5" fill="#1a2740"/>
  <rect x="184" y="160" width="66" height="10" rx="5" fill="url(#brand)"/>
""", 320, 200),

    # Settings + OG keep their text (standalone banners)
    "settings-hero.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 160" fill="none">
  <defs>{DEFS}</defs>
  <rect width="480" height="160" rx="16" fill="url(#spot)"/>
  <rect width="480" height="160" rx="16" fill="url(#grid)"/>
  <circle cx="92" cy="46" r="80" fill="#7c3aed" filter="url(#bloom)" opacity="0.18"/>
  <ellipse cx="92" cy="80" rx="62" ry="44" fill="none" stroke="url(#brand)" stroke-width="1" opacity="0.28"/>
  <circle cx="154" cy="80" r="3.5" fill="#22e0ff" filter="url(#glow)"/>
  <circle cx="30" cy="80" r="3.5" fill="#f43f5e" filter="url(#glow)"/>
  {mcp_hex(92, 80, 0.92)}
  <text x="198" y="72" fill="#eafcff" font-family="system-ui,sans-serif" font-size="22" font-weight="800">Connect AI Agents</text>
  <text x="198" y="100" fill="#9fb3c8" font-family="system-ui,sans-serif" font-size="13">Natural language to real GPUs, in seconds</text>
  <rect width="480" height="160" rx="16" fill="url(#brandSoft)"/>
</svg>""",
    "og-mcp-card.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 630" fill="none">
  <defs>{DEFS}</defs>
  <rect width="1200" height="630" fill="url(#spot)"/>
  <rect width="1200" height="630" fill="url(#grid)"/>
  <circle cx="980" cy="150" r="220" fill="#7c3aed" filter="url(#bloom)" opacity="0.14"/>
  <g transform="translate(150,150) scale(1.5)">{beam(118,196,140)}{beam(284,372,140)}{agent_node(74,140)}{mcp_hex(240,140,1.0)}{gpu_node(418,140)}</g>
  <text x="120" y="470" fill="#eafcff" font-family="system-ui,sans-serif" font-size="56" font-weight="800">Let AI agents control real GPUs</text>
  <text x="120" y="540" fill="#9fb3c8" font-family="system-ui,sans-serif" font-size="28">Xcelsior MCP — natural language to compute · xcelsior.ca/mcp</text>
</svg>""",
}

for name, svg in SLIDES.items():
    (OUT / name).write_text(svg.strip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT / name}")

print(f"Done — {len(SLIDES)} assets in {OUT}")
