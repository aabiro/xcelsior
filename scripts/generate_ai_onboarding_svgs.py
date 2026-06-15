#!/usr/bin/env python3
"""Generate Xcel AI onboarding hero SVG assets (brand-aligned, minimal futuristic)."""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "ai-onboarding"
OUT.mkdir(parents=True, exist_ok=True)

GRAD = """
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00d4ff"/>
      <stop offset="50%" stop-color="#7c3aed"/>
      <stop offset="100%" stop-color="#dc2626"/>
    </linearGradient>
    <linearGradient id="gSoft" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#00d4ff" stop-opacity="0.15"/>
      <stop offset="100%" stop-color="#7c3aed" stop-opacity="0.08"/>
    </linearGradient>
    <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="8" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
"""

LOGO_MARK = """
  <g transform="translate(160,28) scale(0.42)">
    <rect width="200" height="200" rx="40" fill="#0a0e1a"/>
    <path d="M102.6,56.9 L117.4,59.5 L115.4,54.0 L115.8,52.8 L131.8,39.8 L128.1,38.1 L127.6,36.8 L130.8,27.0 L121.5,28.9 L120.2,28.2 L118.5,24.0 L111.2,31.8 L109.4,30.9 L112.8,12.9 L107.2,16.1 L105.6,15.6 L100.0,4.5 L94.4,15.6 L92.8,16.1 L87.2,12.9 L90.6,30.9 L88.8,31.8 L81.5,24.0 L79.8,28.2 L78.5,28.9 L69.2,27.0 L72.4,36.8 L71.9,38.1 L68.2,39.8 L84.2,52.8 L84.6,54.0 L82.6,59.5 L97.4,56.9 Z" fill="url(#g)" opacity="0.9"/>
    <path d="M48,64 L72,64 L100,108 L128,64 L152,64 L112,124 L152,186 L128,186 L100,140 L72,186 L48,186 L88,124 Z" fill="url(#g)"/>
  </g>
"""

SLIDES = {
    "hero.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 280" fill="none">
  <defs>{GRAD}</defs>
  <rect width="480" height="280" rx="24" fill="#0a0e1a"/>
  <rect width="480" height="280" rx="24" fill="url(#gSoft)"/>
  <circle cx="400" cy="60" r="90" fill="#00d4ff" opacity="0.06"/>
  <circle cx="80" cy="220" r="70" fill="#7c3aed" opacity="0.08"/>
  {LOGO_MARK}
  <text x="240" y="200" text-anchor="middle" fill="#e2e8f0" font-family="system-ui,sans-serif" font-size="22" font-weight="700">Xcel AI</text>
  <text x="240" y="228" text-anchor="middle" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="13">Your sovereign compute copilot</text>
  <g filter="url(#glow)" transform="translate(200,108)">
    <path d="M0,32 L18,0 L36,32 L54,0 L72,32" stroke="url(#g)" stroke-width="3" fill="none" stroke-linecap="round"/>
  </g>
</svg>""",
    "actions.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 280" fill="none">
  <defs>{GRAD}</defs>
  <rect width="480" height="280" rx="24" fill="#0a0e1a"/>
  <rect width="480" height="280" rx="24" fill="url(#gSoft)"/>
  {LOGO_MARK}
  <g transform="translate(48,100)">
    <rect x="0" y="0" width="120" height="72" rx="14" stroke="#00d4ff" stroke-opacity="0.35" fill="#00d4ff" fill-opacity="0.06"/>
    <text x="60" y="32" text-anchor="middle" fill="#00d4ff" font-size="11" font-family="system-ui">Launch</text>
    <text x="60" y="50" text-anchor="middle" fill="#cbd5e1" font-size="10" font-family="system-ui">Instances</text>
    <rect x="136" y="0" width="120" height="72" rx="14" stroke="#7c3aed" stroke-opacity="0.35" fill="#7c3aed" fill-opacity="0.06"/>
    <text x="196" y="32" text-anchor="middle" fill="#a78bfa" font-size="11" font-family="system-ui">Register</text>
    <text x="196" y="50" text-anchor="middle" fill="#cbd5e1" font-size="10" font-family="system-ui">Hosts</text>
    <rect x="272" y="0" width="120" height="72" rx="14" stroke="#10b981" stroke-opacity="0.35" fill="#10b981" fill-opacity="0.06"/>
    <text x="332" y="32" text-anchor="middle" fill="#34d399" font-size="11" font-family="system-ui">Billing</text>
    <text x="332" y="50" text-anchor="middle" fill="#cbd5e1" font-size="10" font-family="system-ui">Wallet</text>
  </g>
  <path d="M60,200 H420" stroke="#334155" stroke-width="1"/>
  <text x="240" y="232" text-anchor="middle" fill="#94a3b8" font-size="12" font-family="system-ui">Ask in plain English — Xcel executes across the dashboard</text>
</svg>""",
    "api.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 280" fill="none">
  <defs>{GRAD}</defs>
  <rect width="480" height="280" rx="24" fill="#0a0e1a"/>
  <rect width="480" height="280" rx="24" fill="url(#gSoft)"/>
  {LOGO_MARK}
  <g transform="translate(72,96)">
    <rect x="0" y="0" width="336" height="120" rx="16" stroke="#334155" fill="#111827"/>
    <text x="20" y="28" fill="#00d4ff" font-family="ui-monospace,monospace" font-size="11">POST /api/instances/launch</text>
    <text x="20" y="52" fill="#64748b" font-family="ui-monospace,monospace" font-size="10">GET  /api/hosts</text>
    <text x="20" y="76" fill="#64748b" font-family="ui-monospace,monospace" font-size="10">GET  /api/billing/wallet</text>
    <text x="20" y="100" fill="#64748b" font-family="ui-monospace,monospace" font-size="10">POST /api/artifacts/upload</text>
    <circle cx="300" cy="60" r="28" fill="url(#g)" fill-opacity="0.2" stroke="url(#g)" stroke-width="2"/>
    <path d="M288,60 L296,68 L314,50" stroke="#00d4ff" stroke-width="3" stroke-linecap="round" fill="none"/>
  </g>
  <text x="240" y="248" text-anchor="middle" fill="#94a3b8" font-size="12" font-family="system-ui">Full API access — confirmations before destructive actions</text>
</svg>""",
    "ready.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 280" fill="none">
  <defs>{GRAD}</defs>
  <rect width="480" height="280" rx="24" fill="#0a0e1a"/>
  <rect width="480" height="280" rx="24" fill="url(#gSoft)"/>
  <circle cx="240" cy="120" r="64" stroke="url(#g)" stroke-width="2" fill="#00d4ff" fill-opacity="0.05"/>
  {LOGO_MARK.replace('translate(160,28)', 'translate(192,52)').replace('scale(0.42)', 'scale(0.5)')}
  <text x="240" y="210" text-anchor="middle" fill="#e2e8f0" font-size="18" font-weight="600" font-family="system-ui">You're all set</text>
  <text x="240" y="236" text-anchor="middle" fill="#94a3b8" font-size="12" font-family="system-ui">Tell Xcel what you need — it handles the rest</text>
</svg>""",
}

for name, svg in SLIDES.items():
    (OUT / name).write_text(svg.strip() + "\n", encoding="utf-8")
    print(f"wrote {OUT / name}")