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


def hosts_palette(theme):
    if theme == "dark":
        return dict(
            cyan="#38dbff", violet="#9b6dff", emerald="#34d399",
            major_grid="#2A3548", minor_grid="#1A2235",
            major_grid_op="0.42", minor_grid_op="0.28",
            glow_c_op="0.14", glow_v_op="0.11", glow_e_op="0.09",
            rack_hi="#152033", rack_mid="#0b1220", rack_lo="#06111c",
            bay_hi="#111b2e", bay_lo="#08111d",
            iso_top_hi="#11223A", iso_top_lo="#0A1526",
            iso_face_hi="#0F1D32", iso_face_lo="#2A123B",
            iso_side_hi="#0E2438", iso_side_lo="#1B1033",
            glass_op="0.14", rail_hi_op="0.05",
            detail_stroke="#38dbff", detail_muted="#9b6dff",
            node_stroke="#06111c",
        )
    return dict(
        cyan="#0891b2", violet="#7c3aed", emerald="#059669",
        major_grid="#94A3B8", minor_grid="#CBD5E1",
        major_grid_op="0.18", minor_grid_op="0.16",
        glow_c_op="0.14", glow_v_op="0.11", glow_e_op="0.08",
        rack_hi="#f8fafc", rack_mid="#e2e8f0", rack_lo="#cbd5e1",
        bay_hi="#f1f5f9", bay_lo="#e8eef5",
        iso_top_hi="#FFFFFF", iso_top_lo="#E0F2FE",
        iso_face_hi="#FFFFFF", iso_face_lo="#E9D5FF",
        iso_side_hi="#E0F2FE", iso_side_lo="#DDD6FE",
        glass_op="0.55", rail_hi_op="0.08",
        detail_stroke="#0891b2", detail_muted="#7c3aed",
        node_stroke="#0f172a",
    )


def hosts_defs(theme):
    p = hosts_palette(theme)
    c, v, e = p["cyan"], p["violet"], p["emerald"]
    if theme == "dark":
        depth = (
            '<filter id="depth" x="-30%" y="-30%" width="160%" height="160%" color-interpolation-filters="sRGB">'
            f'<feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="{c}" flood-opacity="0.48"/>'
            f'<feDropShadow dx="0" dy="12" stdDeviation="16" flood-color="{v}" flood-opacity="0.28"/>'
            "</filter>"
        )
        bloom = (
            '<filter id="bloom" x="-120%" y="-120%" width="340%" height="340%">'
            '<feGaussianBlur stdDeviation="22"/></filter>'
        )
    else:
        depth = (
            '<filter id="depth" x="-25%" y="-25%" width="150%" height="150%" color-interpolation-filters="sRGB">'
            '<feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#0f172a" flood-opacity="0.18"/>'
            f'<feDropShadow dx="0" dy="0" stdDeviation="3" flood-color="{c}" flood-opacity="0.2"/>'
            "</filter>"
        )
        bloom = (
            '<filter id="bloom" x="-120%" y="-120%" width="340%" height="340%">'
            '<feGaussianBlur stdDeviation="20"/></filter>'
        )
    return f"""
    <linearGradient id="brand" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="{c}"/><stop offset="62%" stop-color="{v}"/><stop offset="100%" stop-color="{e}"/>
    </linearGradient>
    <linearGradient id="panel" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{p['rack_hi']}"/><stop offset="52%" stop-color="{p['rack_mid']}"/><stop offset="100%" stop-color="{p['rack_lo']}"/>
    </linearGradient>
    <linearGradient id="bay" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{p['bay_hi']}"/><stop offset="100%" stop-color="{p['bay_lo']}"/>
    </linearGradient>
    <radialGradient id="core" cx="50%" cy="40%" r="62%">
      <stop offset="0%" stop-color="{c}" stop-opacity="0.55"/><stop offset="42%" stop-color="{v}" stop-opacity="0.35"/><stop offset="100%" stop-color="{p['rack_lo']}"/>
    </radialGradient>
    <linearGradient id="glassHi" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="{p['glass_op']}"/><stop offset="40%" stop-color="#ffffff" stop-opacity="0.03"/><stop offset="100%" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
    <radialGradient id="gC" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(340 88) rotate(90) scale(190 240)">
      <stop offset="0%" stop-color="{c}" stop-opacity="{p['glow_c_op']}"/><stop offset="100%" stop-color="{c}" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="gV" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(400 210) rotate(90) scale(160 200)">
      <stop offset="0%" stop-color="{v}" stop-opacity="{p['glow_v_op']}"/><stop offset="100%" stop-color="{v}" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="gE" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(420 300) rotate(90) scale(120 150)">
      <stop offset="0%" stop-color="{e}" stop-opacity="{p['glow_e_op']}"/><stop offset="100%" stop-color="{e}" stop-opacity="0"/>
    </radialGradient>
    <pattern id="majorGrid" width="34" height="34" patternUnits="userSpaceOnUse">
      <path d="M34 0H0V34" stroke="{p['major_grid']}" stroke-opacity="{p['major_grid_op']}" stroke-width="0.6"/>
    </pattern>
    <pattern id="minorGrid" width="17" height="17" patternUnits="userSpaceOnUse">
      <path d="M17 0H0V17" stroke="{p['minor_grid']}" stroke-opacity="{p['minor_grid_op']}" stroke-width="0.45"/>
    </pattern>
    <linearGradient id="gridFade" x1="0" y1="0" x2="480" y2="360" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="white" stop-opacity="0.12"/>
      <stop offset="22%" stop-color="white" stop-opacity="0.55"/>
      <stop offset="48%" stop-color="white" stop-opacity="0.82"/>
      <stop offset="72%" stop-color="white" stop-opacity="0.95"/>
      <stop offset="100%" stop-color="white"/>
    </linearGradient>
    <mask id="gridMask"><rect width="480" height="360" fill="url(#gridFade)"/></mask>
    <linearGradient id="brandH" x1="20" y1="0" x2="460" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{c}" stop-opacity="0"/><stop offset="28%" stop-color="{c}" stop-opacity="0.38"/><stop offset="70%" stop-color="{v}" stop-opacity="0.26"/><stop offset="100%" stop-color="{v}" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="brandV" x1="0" y1="16" x2="0" y2="344" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{c}" stop-opacity="0.34"/><stop offset="58%" stop-color="{v}" stop-opacity="0.2"/><stop offset="100%" stop-color="{c}" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="arcGrad" x1="60" y1="72" x2="460" y2="168" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{c}" stop-opacity="0.78"/><stop offset="42%" stop-color="{c}" stop-opacity="0.68"/><stop offset="100%" stop-color="{v}" stop-opacity="0.64"/>
    </linearGradient>
    <linearGradient id="railGrad" x1="40" y1="0" x2="440" y2="0" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{c}" stop-opacity="{p['rail_hi_op']}"/><stop offset="50%" stop-color="{c}"/><stop offset="100%" stop-color="{v}" stop-opacity="0.1"/>
    </linearGradient>
    <linearGradient id="isoTop" x1="0" y1="0" x2="1" y2="1" gradientUnits="objectBoundingBox">
      <stop offset="0%" stop-color="{p['iso_top_hi']}" stop-opacity="0.88"/><stop offset="100%" stop-color="{p['iso_top_lo']}" stop-opacity="0.28"/>
    </linearGradient>
    <linearGradient id="isoFace" x1="0" y1="0" x2="1" y2="1" gradientUnits="objectBoundingBox">
      <stop offset="0%" stop-color="{p['iso_face_hi']}" stop-opacity="0.76"/><stop offset="100%" stop-color="{p['iso_face_lo']}" stop-opacity="0.16"/>
    </linearGradient>
    <linearGradient id="isoSide" x1="0" y1="0" x2="1" y2="1" gradientUnits="objectBoundingBox">
      <stop offset="0%" stop-color="{p['iso_side_hi']}" stop-opacity="0.32"/><stop offset="100%" stop-color="{p['iso_side_lo']}" stop-opacity="0.14"/>
    </linearGradient>
    <linearGradient id="isoStroke" x1="0" y1="0" x2="1" y2="1" gradientUnits="objectBoundingBox">
      <stop offset="0%" stop-color="{c}" stop-opacity="0.7"/><stop offset="58%" stop-color="{v}" stop-opacity="0.58"/><stop offset="100%" stop-color="{v}" stop-opacity="0.34"/>
    </linearGradient>
    <filter id="glow" x="-200%" y="-200%" width="500%" height="500%">
      <feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="arcGlow" x="-100%" y="-100%" width="300%" height="300%">
      <feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    {depth}{bloom}
    <style>
      .tw{{animation:tw 3.2s ease-in-out infinite}}
      .tw2{{animation:tw 5.2s ease-in-out infinite}}
      .tw3{{animation:tw 4.4s ease-in-out infinite}}
      .landing{{animation:landing 4.6s ease-in-out infinite}}
      .energy{{fill:none;stroke-linecap:round;stroke-dasharray:110 720;animation:flow 11s linear infinite}}
      @keyframes tw{{0%,100%{{opacity:1}}50%{{opacity:.2}}}}
      @keyframes landing{{0%,100%{{opacity:1}}50%{{opacity:.18}}}}
      @keyframes flow{{from{{stroke-dashoffset:830}}to{{stroke-dashoffset:0}}}}
    </style>"""


def _iso_stack(opacity, y_offset=0, stroke_scale=1.0):
    sw = lambda w: w * stroke_scale
    return f"""
    <g opacity="{opacity}" transform="translate(0 {y_offset})">
      <path d="M18 42L126 26L170 52L62 68Z" fill="url(#isoTop)" stroke="url(#isoStroke)" stroke-width="{sw(1.1)}"/>
      <path d="M62 68L170 52V84L62 100Z" fill="url(#isoFace)" stroke="url(#isoStroke)" stroke-width="{sw(0.95)}"/>
      <path d="M126 26L170 52V84L126 58Z" fill="url(#isoSide)" stroke="url(#isoStroke)" stroke-width="{sw(0.9)}"/>
      <path d="M82 80L144 70" stroke="url(#brand)" stroke-opacity="0.72" stroke-width="{sw(1.65)}" stroke-linecap="round"/>
      <path d="M82 88L128 80" stroke="url(#brand)" stroke-opacity="0.24" stroke-width="{sw(0.86)}" stroke-linecap="round"/>
    </g>"""


def hosts(theme):
    p = hosts_palette(theme)
    c, v, e = p["cyan"], p["violet"], p["emerald"]
    if theme == "dark":
        bay_stroke, bay_stroke_op = c, "0.32"
        rack_edge, rack_edge_op = c, "0.32"
        node_grid, node_grid_op = c, "0.35"
    else:
        bay_stroke, bay_stroke_op = "#cbd5e1", "1"
        rack_edge, rack_edge_op = p["node_stroke"], "0.18"
        node_grid, node_grid_op = p["node_stroke"], "0.22"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 360" fill="none">
  <defs>{hosts_defs(theme)}</defs>

  <rect width="480" height="360" fill="url(#gC)"/>
  <rect width="480" height="360" fill="url(#gV)"/>
  <rect width="480" height="360" fill="url(#gE)"/>

  <g mask="url(#gridMask)">
    <rect width="480" height="360" fill="url(#majorGrid)" opacity="0.8"/>
    <rect width="480" height="360" fill="url(#minorGrid)" opacity="0.68"/>
    <g opacity="0.66">
      <path d="M20 78H460" stroke="url(#brandH)" stroke-width="0.88"/>
      <path d="M20 152H460" stroke="url(#brandH)" stroke-width="1.05"/>
      <path d="M20 226H460" stroke="url(#brandH)" stroke-width="0.88"/>
      <path d="M20 300H460" stroke="url(#brandH)" stroke-width="0.7"/>
    </g>
    <g opacity="0.58">
      <path d="M64 16V344" stroke="url(#brandV)" stroke-width="0.64"/>
      <path d="M148 16V344" stroke="url(#brandV)" stroke-width="0.78"/>
      <path d="M240 16V344" stroke="url(#brandV)" stroke-width="1.0"/>
      <path d="M332 16V344" stroke="url(#brandV)" stroke-width="0.82"/>
      <path d="M416 16V344" stroke="url(#brandV)" stroke-width="0.64"/>
    </g>
  </g>

  <g opacity="0.72">
    <path d="M48 318h52l10-10h24l9 10h58" stroke="url(#railGrad)" stroke-width="3" stroke-linecap="round"/>
    <path d="M56 328h368" stroke="{c}" stroke-opacity="0.22" stroke-width="2" stroke-linecap="round"/>
    <g stroke="{c}" stroke-opacity="0.3" stroke-width="2" stroke-linecap="round">
      <path d="M118 321v12M168 321v17M218 321v12"/>
      <circle cx="118" cy="337" r="2.5" fill="{c}" fill-opacity="0.5"/>
      <circle cx="168" cy="342" r="2.5" fill="{v}" fill-opacity="0.55"/>
      <circle cx="218" cy="337" r="2.5" fill="{e}" fill-opacity="0.5"/>
    </g>
  </g>

  <g filter="url(#glow)" opacity="0.82">
    <circle cx="108" cy="92" r="3.1" fill="{c}" class="tw"/>
    <circle cx="142" cy="218" r="3.1" fill="{v}" class="tw2"/>
    <circle cx="198" cy="286" r="3.1" fill="{c}" class="tw3"/>
    <circle cx="186" cy="128" r="3.1" fill="{c}" class="tw"/>
    <circle cx="268" cy="78" r="3.1" fill="{v}" class="tw2"/>
    <circle cx="312" cy="108" r="4.2" fill="{c}" class="landing"/>
    <circle cx="388" cy="86" r="3.2" fill="{v}" class="tw3"/>
  </g>

  <g fill="none" stroke-linecap="round" filter="url(#arcGlow)">
    <path d="M108 92 Q188 108 298 148" stroke="url(#arcGrad)" stroke-width="1.75" stroke-opacity="0.62"/>
    <path d="M142 218 Q214 178 298 148" stroke="{v}" stroke-width="1.58" stroke-opacity="0.52"/>
    <path d="M198 286 Q252 218 298 148" stroke="{c}" stroke-width="1.48" stroke-opacity="0.42"/>
    <path d="M186 128 Q238 132 298 148" stroke="{c}" stroke-width="1.34" stroke-opacity="0.36"/>
    <path d="M298 148 Q372 102 458 118" stroke="url(#arcGrad)" stroke-width="1.82" stroke-opacity="0.5"/>
    <path d="M298 148 Q302 164 308 180" stroke="url(#arcGrad)" stroke-width="1.34" stroke-opacity="0.4"/>
    <path class="energy" d="M108 92 Q188 108 298 148" stroke="url(#arcGrad)" stroke-width="2.2" stroke-opacity="0.62"/>
    <path class="energy" d="M142 218 Q214 178 298 148" stroke="{v}" stroke-width="1.95" stroke-opacity="0.48" style="animation-delay:4s;animation-duration:14s"/>
    <path class="energy" d="M298 148 Q372 102 458 118" stroke="url(#arcGrad)" stroke-width="2" stroke-opacity="0.45" style="animation-delay:2s;animation-duration:13s"/>
  </g>

  <g filter="url(#glow)">
    <line x1="298" y1="148" x2="298" y2="118" stroke="{c}" stroke-width="1.05" stroke-opacity="0.54" stroke-linecap="round"/>
    <line x1="298" y1="148" x2="320" y2="130" stroke="{v}" stroke-width="0.86" stroke-opacity="0.28" stroke-linecap="round"/>
    <line x1="298" y1="148" x2="274" y2="130" stroke="{c}" stroke-width="0.86" stroke-opacity="0.22" stroke-linecap="round"/>
  </g>
  <circle cx="298" cy="148" r="6" fill="{c}" fill-opacity="0.82" filter="url(#glow)" class="landing"/>
  <circle cx="298" cy="148" r="3.5" fill="{c}" opacity="0.94"/>
  <circle cx="298" cy="148" r="18" fill="none" stroke="{c}" stroke-opacity="0.14" stroke-width="0.8"/>
  <circle cx="298" cy="148" r="34" fill="none" stroke="{v}" stroke-opacity="0.08" stroke-width="0.7"/>

  <g transform="translate(308 158)" filter="url(#depth)">
    {_iso_stack(0.28, 96, 0.86)}
    {_iso_stack(0.56, 48, 0.92)}
    <g opacity="0.94">
      {_iso_stack(0.94, 0, 1.0)}
    </g>
    <g opacity="0.98" filter="url(#glow)" transform="translate(154 18)">
      <circle cx="0" cy="0" r="13" fill="#FFFFFF" fill-opacity="0.14" stroke="#FFFFFF" stroke-opacity="0.2" stroke-width="0.7"/>
      <path d="M0 5V-5M0 -5L-4 -1M0 -5L4 -1" stroke="#FFFFFF" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </g>
  </g>

  <g filter="url(#depth)" transform="translate(88 72)">
    <path d="M23 108h82l8 6H31l-8-6Z" fill="{c}" fill-opacity="0.1"/>
    <path d="M29 20 64 10l36 10v86l-36 12-35-12V20Z" fill="url(#panel)" stroke="url(#brand)" stroke-width="2.4" stroke-linejoin="round"/>
    <path d="M64 10v108M29 20l35 10 36-10" stroke="{rack_edge}" stroke-opacity="{rack_edge_op}" stroke-width="2"/>
    <path d="M38 33h18M38 44h18M38 58h18M38 69h18M38 83h18M38 94h18" stroke="{c}" stroke-width="4" stroke-linecap="round" opacity="0.85"/>
    <g fill="{e}">
      <circle cx="49" cy="35" r="2.5"/><circle cx="54" cy="60" r="2.5"/><circle cx="44" cy="85" r="2.5"/>
    </g>
    <g stroke="{p['detail_muted']}" stroke-width="2" stroke-linecap="round" opacity="0.8">
      <path d="M75 34h12M75 45h16M75 80h15M75 92h10"/>
      <path d="M88 52 78 72h9l-4 19 17-27h-9l4-12h-7Z" fill="{c}" fill-opacity="0.18" stroke="{c}"/>
    </g>
    <path d="M32 111c16 7 47 7 68 0" stroke="{c}" stroke-opacity="0.38" stroke-width="2" stroke-linecap="round"/>
  </g>

  <g filter="url(#depth)" transform="translate(400 248)">
    <rect x="-38" y="-34" width="76" height="68" rx="14" fill="url(#panel)" stroke="url(#brand)" stroke-width="2"/>
    <rect x="-38" y="-34" width="76" height="68" rx="14" fill="url(#glassHi)"/>
    <rect x="-24" y="-18" width="48" height="36" rx="7" fill="url(#core)" opacity="0.85"/>
    <rect x="-18" y="-10" width="36" height="10" rx="3" fill="url(#bay)" stroke="{bay_stroke}" stroke-opacity="{bay_stroke_op}" stroke-width="1"/>
    <rect x="-18" y="4" width="36" height="10" rx="3" fill="url(#bay)" stroke="{bay_stroke}" stroke-opacity="{bay_stroke_op}" stroke-width="1"/>
    <circle cx="-10" cy="-5" r="2.2" fill="{e}"/><circle cx="-10" cy="9" r="2.2" fill="{c}"/>
    <g stroke="{node_grid}" stroke-opacity="{node_grid_op}" stroke-width="1">
      <path d="M-8 -18V18M8 -18V18M-24 0H24"/>
    </g>
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
