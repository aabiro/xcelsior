#!/usr/bin/env python3
"""Premium dashboard illustrations (theme-aware), matching the gpu-fleet/mcp style.

Replaces the flat line-art overview + host assets with dimensional glass/glow
versions. Each asset is emitted in a dark and a light variant so it reads on
both the dark dashboard and the white light-mode cards.

  rocket(.svg/-light)                      -> overview "launch" card
  gpu(.svg/-light)                         -> overview "provider" card
  xcelsior-hosts-setup-transparent(.svg/-light) -> hosts setup illustration
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "frontend" / "public"
OUT.mkdir(parents=True, exist_ok=True)


def defs(theme):
    """Theme-aware <defs>. Brand/emerald/gold gradients read on both themes;
    panel fills, detail strokes and depth (glow vs soft shadow) flip per theme."""
    if theme == "dark":
        panel = ('<linearGradient id="panel" x1="0" y1="0" x2="0" y2="1">'
                 '<stop offset="0%" stop-color="#16233c"/><stop offset="100%" stop-color="#0b1322"/></linearGradient>')
        glass_op, detail, depth = "0.14", "#7cf2ff", (
            '<filter id="depth" x="-60%" y="-60%" width="220%" height="220%">'
            '<feGaussianBlur stdDeviation="3" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>')
        bloom = ('<filter id="bloom" x="-120%" y="-120%" width="340%" height="340%">'
                 '<feGaussianBlur stdDeviation="20"/></filter>')
    else:
        panel = ('<linearGradient id="panel" x1="0" y1="0" x2="0" y2="1">'
                 '<stop offset="0%" stop-color="#ffffff"/><stop offset="100%" stop-color="#eef4fb"/></linearGradient>')
        glass_op, detail, depth = "0.5", "#3b6fb0", (
            '<filter id="depth" x="-40%" y="-40%" width="180%" height="180%">'
            '<feDropShadow dx="0" dy="6" stdDeviation="7" flood-color="#1e3a5f" flood-opacity="0.18"/></filter>')
        bloom = ('<filter id="bloom" x="-120%" y="-120%" width="340%" height="340%">'
                 '<feGaussianBlur stdDeviation="20"/></filter>')
    return f"""
    <linearGradient id="brand" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#22e0ff"/><stop offset="48%" stop-color="#7c3aed"/><stop offset="100%" stop-color="#f43f5e"/>
    </linearGradient>
    <linearGradient id="emerald" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#34d399"/><stop offset="100%" stop-color="#10b981"/></linearGradient>
    <linearGradient id="gold" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#fde68a"/><stop offset="100%" stop-color="#d4a233"/></linearGradient>
    <radialGradient id="core" cx="50%" cy="40%" r="62%">
      <stop offset="0%" stop-color="#5eeaff"/><stop offset="42%" stop-color="#2a8cff"/><stop offset="100%" stop-color="#0b1b3a"/></radialGradient>
    <linearGradient id="glassHi" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="{glass_op}"/><stop offset="40%" stop-color="#ffffff" stop-opacity="0.03"/><stop offset="100%" stop-color="#ffffff" stop-opacity="0"/></linearGradient>
    {panel}{depth}{bloom}
    <!--detail:{detail}-->"""


def rocket(theme):
    glow = '#22e0ff' if theme == 'dark' else '#7c3aed'
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 320" fill="none">
  <defs>{defs(theme)}</defs>
  <circle cx="150" cy="84" r="70" fill="{glow}" filter="url(#bloom)" opacity="0.16"/>
  <!-- exhaust trail -->
  <path d="M120,196 C96,232 70,250 40,300" stroke="url(#brand)" stroke-width="7" stroke-linecap="round" fill="none" opacity="0.55"/>
  <g filter="url(#depth)" transform="rotate(36 132 132)">
    <!-- body -->
    <path d="M132,66 C156,66 168,96 168,134 L168,176 C168,190 156,202 132,202 C108,202 96,190 96,176 L96,134 C96,96 108,66 132,66 Z" fill="url(#panel)" stroke="url(#brand)" stroke-width="2.5"/>
    <path d="M132,66 C156,66 168,96 168,134 L168,176 C168,190 156,202 132,202 C108,202 96,190 96,176 L96,134 C96,96 108,66 132,66 Z" fill="url(#glassHi)"/>
    <!-- window -->
    <circle cx="132" cy="118" r="20" fill="url(#core)" stroke="#7cf2ff" stroke-opacity="0.5" stroke-width="1.5"/>
    <circle cx="132" cy="118" r="8" fill="#eafcff" opacity="0.9" filter="url(#depth)"/>
    <!-- fins -->
    <path d="M96,168 L72,206 L96,194 Z" fill="url(#brand)" opacity="0.85"/>
    <path d="M168,168 L192,206 L168,194 Z" fill="url(#brand)" opacity="0.85"/>
    <!-- flame -->
    <path d="M118,202 L132,238 L146,202 Z" fill="url(#gold)" filter="url(#depth)"/>
  </g>
</svg>"""


def gpu(theme):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 220" fill="none">
  <defs>{defs(theme)}</defs>
  <circle cx="300" cy="60" r="76" fill="#7c3aed" filter="url(#bloom)" opacity="0.14"/>
  <g filter="url(#depth)">
    <!-- PCB -->
    <rect x="30" y="56" width="300" height="120" rx="16" fill="url(#panel)" stroke="url(#brand)" stroke-width="2"/>
    <rect x="30" y="56" width="300" height="120" rx="16" fill="url(#glassHi)"/>
    <!-- bracket -->
    <rect x="18" y="44" width="14" height="132" rx="3" fill="url(#panel)" stroke="url(#brand)" stroke-width="1.5"/>
    <!-- shroud -->
    <rect x="44" y="70" width="272" height="78" rx="10" fill="#0c1526" opacity="{0.0 if theme=='light' else 0.55}"/>
    <!-- fans -->
    <g>
      <circle cx="108" cy="110" r="38" fill="none" stroke="url(#brand)" stroke-width="2"/>
      <circle cx="108" cy="110" r="9" fill="url(#core)"/>
      <g stroke="url(#brand)" stroke-width="3" stroke-linecap="round" opacity="0.8">
        <path d="M108,110 C118,96 132,98 130,110"/><path d="M108,110 C122,120 120,134 108,132"/><path d="M108,110 C94,120 80,118 82,108"/><path d="M108,110 C98,96 96,84 108,86"/>
      </g>
      <circle cx="220" cy="110" r="38" fill="none" stroke="url(#brand)" stroke-width="2"/>
      <circle cx="220" cy="110" r="9" fill="url(#core)"/>
      <g stroke="url(#brand)" stroke-width="3" stroke-linecap="round" opacity="0.8">
        <path d="M220,110 C230,96 244,98 242,110"/><path d="M220,110 C234,120 232,134 220,132"/><path d="M220,110 C206,120 192,118 194,108"/><path d="M220,110 C210,96 208,84 220,86"/>
      </g>
    </g>
    <!-- gold pin edge -->
    <rect x="30" y="168" width="300" height="8" rx="2" fill="url(#gold)" opacity="0.85"/>
    <!-- earning glow -->
    <circle cx="300" cy="78" r="8" fill="#34d399" filter="url(#depth)"/>
  </g>
</svg>"""


def hosts(theme):
    rack = '#0c1526' if theme == 'dark' else '#f1f5f9'
    line = '#26405f' if theme == 'dark' else '#cbd8e8'
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 360" fill="none">
  <defs>{defs(theme)}</defs>
  <circle cx="120" cy="100" r="100" fill="#22e0ff" filter="url(#bloom)" opacity="0.12"/>
  <circle cx="380" cy="280" r="90" fill="#34d399" filter="url(#bloom)" opacity="0.1"/>
  <!-- server tower -->
  <g filter="url(#depth)">
    <rect x="70" y="70" width="150" height="220" rx="16" fill="url(#panel)" stroke="url(#brand)" stroke-width="2"/>
    <rect x="70" y="70" width="150" height="220" rx="16" fill="url(#glassHi)"/>
    <g>
      <rect x="90" y="94" width="110" height="40" rx="8" fill="{rack}" stroke="{line}"/>
      <rect x="90" y="146" width="110" height="40" rx="8" fill="{rack}" stroke="{line}"/>
      <rect x="90" y="198" width="110" height="40" rx="8" fill="{rack}" stroke="{line}"/>
      <circle cx="106" cy="114" r="4.5" fill="#34d399"/><circle cx="106" cy="166" r="4.5" fill="#34d399"/><circle cx="106" cy="218" r="4.5" fill="#22e0ff"/>
      <g stroke="url(#brand)" stroke-width="2.5" stroke-linecap="round" opacity="0.7"><path d="M124,114 H190 M124,166 H190 M124,218 H178"/></g>
    </g>
  </g>
  <!-- network fabric to a node -->
  <g stroke="url(#brand)" stroke-width="2" fill="none" opacity="0.55" stroke-linecap="round">
    <path d="M220,150 H300 V210 H360"/>
    <path d="M220,210 H264 V120 H320"/>
  </g>
  <circle cx="300" cy="150" r="4" fill="#7c3aed"/><circle cx="264" cy="210" r="4" fill="#22e0ff"/>
  <g filter="url(#depth)" transform="translate(360,210)">
    <rect x="-34" y="-30" width="68" height="60" rx="12" fill="url(#panel)" stroke="url(#brand)" stroke-width="1.75"/>
    <rect x="-34" y="-30" width="68" height="60" rx="12" fill="url(#glassHi)"/>
    <rect x="-20" y="-16" width="40" height="32" rx="6" fill="url(#core)" opacity="0.7"/>
    <g stroke="#0a1322" stroke-opacity="0.4" stroke-width="1"><path d="M-7,-16 V16 M7,-16 V16 M-20,0 H20"/></g>
  </g>
</svg>"""


ASSETS = {
    "rocket.svg": rocket("dark"),
    "rocket-light.svg": rocket("light"),
    "gpu.svg": gpu("dark"),
    "gpu-light.svg": gpu("light"),
    "xcelsior-hosts-setup-transparent.svg": hosts("dark"),
    "xcelsior-hosts-setup-transparent-light.svg": hosts("light"),
}

for name, svg in ASSETS.items():
    (OUT / name).write_text(svg.strip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT / name}")

print(f"Done — {len(ASSETS)} assets in {OUT}")
