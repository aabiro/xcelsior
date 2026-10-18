#!/usr/bin/env python3
"""Generate premium product-section SVG assets for the Features page.

One dimensional illustration per platform product (Serverless, Instances,
Hosting, Xcel AI, Volumes). GPUs and MCP reuse their existing hero art.
No baked-in text labels — the page renders the titles.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "features"
OUT.mkdir(parents=True, exist_ok=True)

DEFS = """
    <linearGradient id="brand" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#22e0ff"/>
      <stop offset="48%" stop-color="#7c3aed"/>
      <stop offset="100%" stop-color="#f43f5e"/>
    </linearGradient>
    <linearGradient id="brandSoft" x1="0" y1="1" x2="1" y2="0">
      <stop offset="0%" stop-color="#22e0ff" stop-opacity="0.14"/>
      <stop offset="100%" stop-color="#f43f5e" stop-opacity="0.04"/>
    </linearGradient>
    <radialGradient id="spot" cx="50%" cy="36%" r="72%">
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
    <linearGradient id="emerald" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#34d399"/>
      <stop offset="100%" stop-color="#10b981"/>
    </linearGradient>
    <linearGradient id="violet" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#a78bfa"/>
      <stop offset="100%" stop-color="#7c3aed"/>
    </linearGradient>
    <linearGradient id="gold" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#fde68a"/>
      <stop offset="100%" stop-color="#d4a233"/>
    </linearGradient>
    <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="bloom" x="-120%" y="-120%" width="340%" height="340%">
      <feGaussianBlur stdDeviation="24"/>
    </filter>
    <filter id="cardShadow" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#000" flood-opacity="0.45"/>
    </filter>
    <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">
      <path d="M24 0H0V24" fill="none" stroke="#22e0ff" stroke-opacity="0.06" stroke-width="1"/>
    </pattern>
"""


def frame(inner, w=480, h=320):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" fill="none">
  <defs>{DEFS}</defs>
  <rect width="{w}" height="{h}" rx="28" fill="url(#spot)"/>
  <rect width="{w}" height="{h}" rx="28" fill="url(#grid)"/>
{inner}
  <rect width="{w}" height="{h}" rx="28" fill="url(#brandSoft)"/>
</svg>"""


# Serverless — autoscaling endpoint: stacked function cards fanning out + bolt + scale bars
SERVERLESS = frame("""
  <circle cx="372" cy="78" r="78" fill="#22e0ff" filter="url(#bloom)" opacity="0.14"/>
  <g filter="url(#cardShadow)">
    <rect x="150" y="78" width="150" height="92" rx="14" fill="url(#panel)" stroke="url(#brand)" stroke-width="1.5" transform="rotate(-7 225 124)"/>
    <rect x="170" y="96" width="150" height="92" rx="14" fill="url(#panel)" stroke="url(#brand)" stroke-width="1.5" transform="rotate(-2 245 142)"/>
    <rect x="190" y="116" width="150" height="92" rx="14" fill="url(#panel)" stroke="#22e0ff" stroke-width="1.5"/>
    <rect x="190" y="116" width="150" height="92" rx="14" fill="url(#glass)"/>
    <path d="M262,140 L250,166 H266 L258,188 L284,156 H266 L274,140 Z" fill="url(#brand)" filter="url(#glow)"/>
  </g>
  <!-- autoscale bars -->
  <g>
    <rect x="60" y="206" width="16" height="34" rx="3" fill="url(#brand)" opacity="0.5"/>
    <rect x="84" y="190" width="16" height="50" rx="3" fill="url(#brand)" opacity="0.7"/>
    <rect x="108" y="170" width="16" height="70" rx="3" fill="url(#brand)" opacity="0.9"/>
    <rect x="132" y="154" width="16" height="86" rx="3" fill="#34d399"/>
  </g>
  <circle cx="360" cy="234" r="6" fill="#34d399" filter="url(#glow)"/>
""")

# Instances — a live instance monitor: window with metric lines + status dot
INSTANCES = frame("""
  <circle cx="110" cy="80" r="74" fill="#7c3aed" filter="url(#bloom)" opacity="0.16"/>
  <g filter="url(#cardShadow)">
    <rect x="88" y="74" width="304" height="172" rx="16" fill="url(#panel)" stroke="url(#brand)" stroke-width="1.5"/>
    <rect x="88" y="74" width="304" height="172" rx="16" fill="url(#glass)"/>
    <path d="M88,98 H392" stroke="#22324d" stroke-width="1"/>
    <circle cx="106" cy="86" r="3.5" fill="#f43f5e"/><circle cx="118" cy="86" r="3.5" fill="#fbbf24"/><circle cx="130" cy="86" r="3.5" fill="#34d399"/>
    <!-- util line -->
    <path d="M108,196 L140,170 L168,182 L200,150 L232,160 L264,132 L300,144 L340,120 L372,128" stroke="url(#brand)" stroke-width="2.4" fill="none" stroke-linecap="round" filter="url(#glow)"/>
    <path d="M108,214 L150,206 L196,210 L244,200 L300,204 L372,196" stroke="#34d399" stroke-width="1.6" fill="none" stroke-linecap="round" opacity="0.7"/>
    <g fill="#7cf2ff" opacity="0.55"><rect x="108" y="116" width="34" height="6" rx="3"/><rect x="150" y="116" width="22" height="6" rx="3" opacity="0.6"/></g>
  </g>
  <circle cx="372" cy="92" r="5" fill="#34d399" filter="url(#glow)"/>
""")

# Hosting / providing — a host node uploading capacity + earnings coin
HOSTING = frame("""
  <circle cx="360" cy="240" r="76" fill="#34d399" filter="url(#bloom)" opacity="0.12"/>
  <g filter="url(#cardShadow)">
    <!-- server tower -->
    <rect x="150" y="80" width="120" height="170" rx="14" fill="url(#panel)" stroke="url(#brand)" stroke-width="1.5"/>
    <rect x="150" y="80" width="120" height="170" rx="14" fill="url(#glass)"/>
    <g>
      <rect x="166" y="100" width="88" height="30" rx="6" fill="#0c1526" stroke="#26405f"/>
      <rect x="166" y="138" width="88" height="30" rx="6" fill="#0c1526" stroke="#26405f"/>
      <rect x="166" y="176" width="88" height="30" rx="6" fill="#0c1526" stroke="#26405f"/>
      <circle cx="180" cy="115" r="3.5" fill="#34d399"/><circle cx="180" cy="153" r="3.5" fill="#34d399"/><circle cx="180" cy="191" r="3.5" fill="#22e0ff"/>
      <g stroke="#22e0ff" stroke-opacity="0.5" stroke-width="2"><path d="M196,115 H242 M196,153 H242 M196,191 H236"/></g>
    </g>
  </g>
  <!-- upload arrow to cloud -->
  <path d="M286,150 C330,150 330,120 360,120" stroke="url(#brand)" stroke-width="2" stroke-dasharray="2 8" fill="none" opacity="0.6"/>
  <g transform="translate(360,118)">
    <circle r="30" fill="#0c1526" stroke="url(#gold)" stroke-width="1.5"/>
    <circle r="30" fill="url(#glass)"/>
    <path d="M0,-12 V12 M-7,-5 L0,-12 L7,-5" stroke="url(#gold)" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  </g>
""")

# Xcel AI — assistant: chat bubble with neural spark + reply
XCELAI = frame("""
  <circle cx="120" cy="90" r="78" fill="#7c3aed" filter="url(#bloom)" opacity="0.18"/>
  <g filter="url(#cardShadow)">
    <rect x="86" y="92" width="220" height="96" rx="18" fill="url(#panel)" stroke="url(#violet)" stroke-width="1.5"/>
    <rect x="86" y="92" width="220" height="96" rx="18" fill="url(#glass)"/>
    <path d="M120,188 L120,210 L146,188 Z" fill="url(#panel)" stroke="url(#violet)" stroke-width="1.5"/>
    <g fill="#c4b5fd" opacity="0.85"><rect x="110" y="118" width="120" height="8" rx="4"/><rect x="110" y="136" width="170" height="8" rx="4" opacity="0.6"/><rect x="110" y="154" width="92" height="8" rx="4" opacity="0.4"/></g>
  </g>
  <g filter="url(#cardShadow)">
    <rect x="250" y="176" width="150" height="74" rx="18" fill="#0c1526" stroke="url(#brand)" stroke-width="1.5"/>
    <rect x="250" y="176" width="150" height="74" rx="18" fill="url(#glass)"/>
    <g fill="#7cf2ff" opacity="0.85"><rect x="270" y="198" width="92" height="8" rx="4"/><rect x="270" y="216" width="60" height="8" rx="4" opacity="0.6"/></g>
  </g>
  <g transform="translate(360,96)" filter="url(#glow)">
    <path d="M0,-22 L6,-6 L22,0 L6,6 L0,22 L-6,6 L-22,0 L-6,-6 Z" fill="url(#violet)"/>
    <circle r="4" fill="#eafcff"/>
  </g>
""")

# Volumes — persistent storage: stacked disks + attach connector
VOLUMES = frame("""
  <circle cx="360" cy="84" r="74" fill="#22e0ff" filter="url(#bloom)" opacity="0.14"/>
  <g filter="url(#cardShadow)" transform="translate(170,86)">
    <ellipse cx="0" cy="0" rx="74" ry="22" fill="url(#panel)" stroke="url(#brand)" stroke-width="1.5"/>
    <path d="M-74,0 V46 A74,22 0 0,0 74,46 V0" fill="url(#panel)" stroke="url(#brand)" stroke-width="1.5"/>
    <ellipse cx="0" cy="0" rx="74" ry="22" fill="url(#glass)"/>
    <path d="M-74,46 V92 A74,22 0 0,0 74,92 V46" fill="url(#panel)" stroke="url(#brand)" stroke-width="1.5"/>
    <ellipse cx="0" cy="46" rx="74" ry="22" fill="none" stroke="#22e0ff" stroke-opacity="0.4"/>
    <path d="M-74,92 V126 A74,22 0 0,0 74,126 V92" fill="url(#panel)" stroke="url(#brand)" stroke-width="1.5"/>
    <ellipse cx="0" cy="92" rx="74" ry="22" fill="none" stroke="#22e0ff" stroke-opacity="0.4"/>
    <ellipse cx="44" cy="-2" rx="5" ry="3.5" fill="#34d399" filter="url(#glow)"/>
  </g>
  <!-- attach connector to an instance -->
  <path d="M250,150 C300,150 300,210 348,210" stroke="url(#brand)" stroke-width="2" stroke-dasharray="2 8" fill="none" opacity="0.6"/>
  <g transform="translate(372,210)">
    <rect x="-26" y="-22" width="52" height="44" rx="10" fill="#0c1526" stroke="url(#brand)" stroke-width="1.5"/>
    <rect x="-26" y="-22" width="52" height="44" rx="10" fill="url(#core)" opacity="0.5"/>
    <g stroke="#0a1322" stroke-opacity="0.5" stroke-width="1"><path d="M-9,-22 V22 M9,-22 V22 M-26,-2 H26"/></g>
  </g>
""")

ASSETS = {
    "serverless.svg": SERVERLESS,
    "instances.svg": INSTANCES,
    "hosting.svg": HOSTING,
    "xcel-ai.svg": XCELAI,
    "volumes.svg": VOLUMES,
}

for name, svg in ASSETS.items():
    (OUT / name).write_text(svg.strip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT / name}")

print(f"Done — {len(ASSETS)} assets in {OUT}")
