#!/usr/bin/env python3
"""Generate GPU fleet marketing SVG assets.

Aim: premium, dimensional artwork (layered radial lighting, glass panels, soft
shadows, detailed GPU silhouettes) rather than flat primitives. No baked-in
text labels — titles are rendered by the page so the art stays reusable.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "gpu-fleet"
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared defs: gradients, lighting, glass, depth ──────────────────────────
DEFS = """
    <linearGradient id="brand" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#22e0ff"/>
      <stop offset="48%" stop-color="#7c3aed"/>
      <stop offset="100%" stop-color="#f43f5e"/>
    </linearGradient>
    <linearGradient id="brandSoft" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#22e0ff" stop-opacity="0.16"/>
      <stop offset="100%" stop-color="#f43f5e" stop-opacity="0.05"/>
    </linearGradient>
    <radialGradient id="spot" cx="50%" cy="38%" r="70%">
      <stop offset="0%" stop-color="#1b2740"/>
      <stop offset="100%" stop-color="#070b15"/>
    </radialGradient>
    <linearGradient id="glass" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.14"/>
      <stop offset="22%" stop-color="#ffffff" stop-opacity="0.04"/>
      <stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="pcb" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#16233c"/>
      <stop offset="100%" stop-color="#0a1322"/>
    </linearGradient>
    <radialGradient id="core" cx="50%" cy="42%" r="62%">
      <stop offset="0%" stop-color="#5eeaff"/>
      <stop offset="40%" stop-color="#2a8cff"/>
      <stop offset="100%" stop-color="#0b1b3a"/>
    </radialGradient>
    <linearGradient id="gold" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#fde68a"/>
      <stop offset="100%" stop-color="#b8862f"/>
    </linearGradient>
    <linearGradient id="emerald" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#34d399"/>
      <stop offset="100%" stop-color="#10b981"/>
    </linearGradient>
    <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="3.2" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="bloom" x="-120%" y="-120%" width="340%" height="340%">
      <feGaussianBlur stdDeviation="22"/>
    </filter>
    <filter id="cardShadow" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#000000" flood-opacity="0.45"/>
    </filter>
    <pattern id="grid" width="22" height="22" patternUnits="userSpaceOnUse">
      <path d="M22 0H0V22" fill="none" stroke="#22e0ff" stroke-opacity="0.06" stroke-width="1"/>
    </pattern>
"""

# ── A detailed GPU card (PCB + heatsink fins + die + gold pins) ──────────────
def gpu_card(cx, cy, s=1.0):
    return f"""
  <g transform="translate({cx},{cy}) scale({s})" filter="url(#cardShadow)">
    <rect x="-104" y="-58" width="208" height="116" rx="14" fill="url(#pcb)" stroke="url(#brand)" stroke-width="1.5"/>
    <rect x="-104" y="-58" width="208" height="116" rx="14" fill="url(#glass)"/>
    <!-- heatsink fins -->
    <g stroke="#22e0ff" stroke-opacity="0.5" stroke-width="2">
      <path d="M-86,-40 V40 M-74,-40 V40 M-62,-40 V40 M-50,-40 V40 M-38,-40 V40 M-26,-40 V40"/>
    </g>
    <rect x="-92" y="-44" width="78" height="88" rx="8" fill="none" stroke="#334a6b" stroke-width="1"/>
    <!-- die -->
    <rect x="6" y="-34" width="78" height="68" rx="8" fill="#0a1322"/>
    <rect x="6" y="-34" width="78" height="68" rx="8" fill="url(#core)" opacity="0.92"/>
    <rect x="6" y="-34" width="78" height="68" rx="8" fill="none" stroke="#7cf2ff" stroke-opacity="0.4"/>
    <g stroke="#0a1322" stroke-opacity="0.55" stroke-width="1.2">
      <path d="M19,-34 V34 M32,-34 V34 M45,-34 V34 M58,-34 V34 M71,-34 V34 M6,-21 H84 M6,-8 H84 M6,5 H84 M6,18 H84"/>
    </g>
    <circle cx="45" cy="0" r="20" fill="none" stroke="#7cf2ff" stroke-opacity="0.45" stroke-width="1.5"/>
    <circle cx="45" cy="0" r="12" fill="#eafcff" opacity="0.95" filter="url(#glow)"/>
    <path d="M10,-30 Q45,-37 80,-30 L80,-19 Q45,-25 10,-19 Z" fill="#ffffff" opacity="0.10"/>
    <!-- gold pin edge -->
    <rect x="-104" y="50" width="208" height="8" rx="2" fill="url(#gold)" opacity="0.8"/>
    <g stroke="#0a1322" stroke-width="1.4">
      <path d="M-92,50 V58 M-80,50 V58 M-68,50 V58 M-56,50 V58 M-44,50 V58 M-32,50 V58 M-20,50 V58 M-8,50 V58 M4,50 V58 M16,50 V58 M28,50 V58 M40,50 V58 M52,50 V58 M64,50 V58 M76,50 V58 M88,50 V58"/>
    </g>
  </g>"""

SLIDES = {
    "hero-power.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 320" fill="none">
  <defs>{DEFS}</defs>
  <rect width="480" height="320" rx="28" fill="url(#spot)"/>
  <rect width="480" height="320" rx="28" fill="url(#grid)"/>
  <circle cx="392" cy="70" r="80" fill="#22e0ff" filter="url(#bloom)" opacity="0.16"/>
  <circle cx="86" cy="250" r="66" fill="#7c3aed" filter="url(#bloom)" opacity="0.18"/>
  <g opacity="0.5" stroke="url(#brand)" fill="none">
    <ellipse cx="240" cy="160" rx="196" ry="78" stroke-width="1" opacity="0.22"/>
    <ellipse cx="240" cy="160" rx="252" ry="104" stroke-width="0.75" opacity="0.12"/>
  </g>
  {gpu_card(240, 160, 1.18)}
  <rect width="480" height="320" rx="28" fill="url(#brandSoft)"/>
</svg>""",

    # LLM training — a smooth loss curve descending over a faint net, gradient area fill
    "workload-training.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 120" fill="none">
  <defs>{DEFS}
    <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#22e0ff" stop-opacity="0.32"/>
      <stop offset="100%" stop-color="#22e0ff" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <rect width="200" height="120" rx="16" fill="url(#spot)" stroke="#22324d" stroke-width="0.75"/>
  <rect width="200" height="120" rx="16" fill="url(#grid)"/>
  <g stroke="#22e0ff" stroke-opacity="0.18" stroke-width="0.75">
    <path d="M24,30 H176 M24,58 H176 M24,86 H176"/>
  </g>
  <path d="M24,40 C60,42 64,80 96,84 C128,88 140,96 176,98 L176,104 L24,104 Z" fill="url(#area)"/>
  <path d="M24,40 C60,42 64,80 96,84 C128,88 140,96 176,98" stroke="url(#brand)" stroke-width="2.6" stroke-linecap="round" fill="none" filter="url(#glow)"/>
  <circle cx="176" cy="98" r="4.5" fill="#34d399" filter="url(#glow)"/>
  <g fill="#7cf2ff" opacity="0.5">
    <circle cx="46" cy="34" r="2"/><circle cx="92" cy="30" r="2"/><circle cx="138" cy="36" r="2"/>
  </g>
</svg>""",

    # LLM inference — chip emitting a token stream + latency arc
    "workload-inference.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 120" fill="none">
  <defs>{DEFS}</defs>
  <rect width="200" height="120" rx="16" fill="url(#spot)" stroke="#22324d" stroke-width="0.75"/>
  <rect width="200" height="120" rx="16" fill="url(#grid)"/>
  <g transform="translate(58,60)">
    <rect x="-26" y="-26" width="52" height="52" rx="10" fill="#0a1322" stroke="url(#brand)" stroke-width="1.5"/>
    <rect x="-26" y="-26" width="52" height="52" rx="10" fill="url(#core)" opacity="0.55"/>
    <g stroke="#0a1322" stroke-opacity="0.5" stroke-width="1"><path d="M-13,-26 V26 M0,-26 V26 M13,-26 V26 M-26,-13 H26 M-26,0 H26 M-26,13 H26"/></g>
    <g stroke="url(#brand)" stroke-width="1.5" stroke-linecap="round" opacity="0.7">
      <path d="M-26,-14 H-36 M-26,0 H-38 M-26,14 H-36 M26,-14 H36 M26,0 H38 M26,14 H36"/>
    </g>
  </g>
  <g fill="url(#brand)">
    <rect x="104" y="46" width="18" height="7" rx="3.5" opacity="0.9"/>
    <rect x="128" y="56" width="26" height="7" rx="3.5" opacity="0.7"/>
    <rect x="104" y="66" width="20" height="7" rx="3.5" opacity="0.5"/>
    <rect x="130" y="76" width="14" height="7" rx="3.5" opacity="0.35"/>
  </g>
  <path d="M150,30 A40,40 0 0,1 150,90" stroke="#34d399" stroke-width="1.5" stroke-dasharray="3 6" fill="none" opacity="0.6"/>
</svg>""",

    # Pick -> Provision -> Pulse pipeline (no titles): mini card, energy gear, live heartbeat
    "deploy-pipeline.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 200" fill="none">
  <defs>{DEFS}</defs>
  <rect width="640" height="200" rx="24" fill="url(#spot)"/>
  <rect width="640" height="200" rx="24" fill="url(#grid)"/>
  <circle cx="540" cy="56" r="70" fill="#10b981" filter="url(#bloom)" opacity="0.12"/>
  <!-- energy flow -->
  <path d="M150,100 H270 M370,100 H490" stroke="url(#brand)" stroke-width="2.5" stroke-dasharray="2 10" stroke-linecap="round" opacity="0.6" filter="url(#glow)"/>
  <!-- node 1: pick a card -->
  <g transform="translate(96,100)">
    <circle r="46" fill="#0c1526" stroke="url(#brand)" stroke-width="1.5"/>
    <circle r="46" fill="url(#glass)"/>
    <rect x="-26" y="-17" width="52" height="34" rx="6" fill="url(#pcb)"/>
    <rect x="-26" y="-17" width="52" height="34" rx="6" fill="url(#core)" opacity="0.75" stroke="#7cf2ff" stroke-opacity="0.4"/>
    <rect x="-26" y="12" width="52" height="5" rx="2" fill="url(#gold)" opacity="0.85"/>
  </g>
  <!-- node 2: provision -->
  <g transform="translate(320,100)" filter="url(#glow)">
    <circle r="54" fill="#0c1526" stroke="url(#brand)" stroke-width="2"/>
    <circle r="54" fill="url(#glass)"/>
    <g stroke="url(#brand)" stroke-width="3" fill="none" stroke-linecap="round">
      <circle r="18"/>
      <path d="M0,-30 V-22 M0,30 V22 M-30,0 H-22 M30,0 H22 M-21,-21 l5,5 M21,21 l-5,-5 M21,-21 l-5,5 M-21,21 l5,-5"/>
    </g>
    <path d="M-6,-9 L8,-1 L-6,7 Z" fill="#eafcff"/>
  </g>
  <!-- node 3: live pulse -->
  <g transform="translate(544,100)">
    <circle r="46" fill="#0c1526" stroke="url(#emerald)" stroke-width="1.5"/>
    <circle r="46" fill="url(#glass)"/>
    <path d="M-28,0 H-12 L-4,-16 L6,16 L14,-6 L20,0 H28" stroke="url(#emerald)" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)"/>
    <circle cx="34" cy="-30" r="6" fill="#34d399" filter="url(#glow)"/>
  </g>
</svg>""",

    "spot-pulse.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 120" fill="none">
  <defs>{DEFS}
    <linearGradient id="save" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#10b981" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="#10b981" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <rect width="480" height="120" rx="20" fill="url(#spot)" stroke="#22324d" stroke-width="0.75"/>
  <rect width="480" height="120" rx="20" fill="url(#grid)"/>
  <path d="M40,80 L80,64 L120,70 L160,46 L200,54 L240,38 L280,50 L320,34 L360,42 L400,30 L440,36 L440,96 L40,96 Z" fill="url(#save)"/>
  <path d="M40,80 L80,64 L120,70 L160,46 L200,54 L240,38 L280,50 L320,34 L360,42 L400,30 L440,36" stroke="url(#brand)" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" fill="none" filter="url(#glow)"/>
  <circle cx="440" cy="36" r="5" fill="#34d399" filter="url(#glow)"/>
</svg>""",

    "accent-flare.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <defs>{DEFS}</defs>
  <circle cx="60" cy="60" r="48" fill="#22e0ff" filter="url(#bloom)" opacity="0.18"/>
  <path d="M60,10 L67,46 L103,53 L67,60 L60,96 L53,60 L17,53 L53,46 Z" fill="url(#brand)" opacity="0.4" filter="url(#glow)"/>
  <circle cx="60" cy="60" r="6" fill="#eafcff"/>
</svg>""",
}

for name, svg in SLIDES.items():
    (OUT / name).write_text(svg.strip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT / name}")

print(f"Done — {len(SLIDES)} assets in {OUT}")
