#!/usr/bin/env python3
"""Generate MCP marketing + settings SVG assets (brand-aligned agent-bridge visuals)."""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "mcp"
OUT.mkdir(parents=True, exist_ok=True)

GRAD = """
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00d4ff"/>
      <stop offset="45%" stop-color="#7c3aed"/>
      <stop offset="100%" stop-color="#dc2626"/>
    </linearGradient>
    <linearGradient id="gSoft" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#00d4ff" stop-opacity="0.18"/>
      <stop offset="100%" stop-color="#7c3aed" stop-opacity="0.1"/>
    </linearGradient>
    <linearGradient id="gold" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#fbbf24"/>
      <stop offset="100%" stop-color="#f59e0b"/>
    </linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="5" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="softGlow" x="-80%" y="-80%" width="260%" height="260%">
      <feGaussianBlur stdDeviation="14"/>
    </filter>
"""

SPARK = """
  <g filter="url(#glow)" transform="translate(72,140) scale(1.8)" fill="url(#g)">
    <path d="M12 1.8c.38 3.07 1.05 4.96 2.18 6.1 1.13 1.13 3.02 1.8 6.02 2.1-3 .38-4.89 1.05-6.02 2.18-1.13 1.13-1.8 3.02-2.18 6.02-.38-3-1.05-4.89-2.18-6.02-1.13-1.13-3.02-1.8-6.02-2.18 3-.3 4.89-.97 6.02-2.1C10.95 6.76 11.62 4.87 12 1.8Z"/>
  </g>
"""

MCP_HEX = """
  <g filter="url(#glow)" transform="translate(200,108)">
    <polygon points="40,0 80,22 80,66 40,88 0,66 0,22" fill="#111827" stroke="url(#g)" stroke-width="1.5"/>
    <text x="40" y="48" text-anchor="middle" fill="#e2e8f0" font-family="ui-monospace,monospace" font-size="11" font-weight="700">MCP</text>
  </g>
"""

GPU_DIE = """
  <g filter="url(#glow)" transform="translate(328,96)">
    <rect width="112" height="80" rx="10" fill="#0f172a" stroke="url(#g)" stroke-width="1.25"/>
    <rect x="12" y="12" width="88" height="56" rx="5" fill="#0a0e1a" stroke="#334155"/>
    <path d="M56,28 L56,52 M44,40 L68,40" stroke="url(#g)" stroke-width="2" stroke-linecap="round"/>
  </g>
"""

FLOW_PATH = """
  <path d="M108,152 C160,152 180,132 240,132" stroke="url(#g)" stroke-width="2" stroke-dasharray="6 8" fill="none" opacity="0.85"/>
  <path d="M280,152 C320,152 340,168 384,168" stroke="url(#g)" stroke-width="2" stroke-dasharray="6 8" fill="none" opacity="0.85"/>
  <circle cx="108" cy="152" r="4" fill="#00d4ff"/>
  <circle cx="240" cy="132" r="4" fill="#7c3aed"/>
  <circle cx="384" cy="168" r="4" fill="#dc2626"/>
"""

SLIDES = {
    "hero-agent-gpu.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 280" fill="none">
  <defs>{GRAD}</defs>
  <rect width="480" height="280" rx="24" fill="#0a0e1a"/>
  <rect width="480" height="280" rx="24" fill="url(#gSoft)"/>
  <circle cx="400" cy="56" r="72" fill="#7c3aed" filter="url(#softGlow)" opacity="0.12"/>
  {SPARK}{MCP_HEX}{GPU_DIE}{FLOW_PATH}
  <text x="240" y="248" text-anchor="middle" fill="#e2e8f0" font-family="system-ui,sans-serif" font-size="18" font-weight="700">Agent → MCP → GPU</text>
</svg>""",
    "flow-discover.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 200" fill="none">
  <defs>{GRAD}</defs>
  <rect width="320" height="200" rx="20" fill="#0a0e1a" stroke="#334155" stroke-width="0.75"/>
  <rect width="320" height="200" rx="20" fill="url(#gSoft)"/>
  <circle cx="100" cy="88" r="36" stroke="url(#g)" stroke-width="2" fill="#111827"/>
  <path d="M118,88 L148,88" stroke="#00d4ff" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M128,78 L148,88 L128,98" stroke="#00d4ff" stroke-width="2" fill="none"/>
  <g transform="translate(168,52)">
    <rect width="56" height="36" rx="6" fill="#1e293b" stroke="#475569"/>
    <rect x="72" y="8" width="56" height="36" rx="6" fill="#1e293b" stroke="#475569"/>
    <rect x="36" y="56" width="56" height="36" rx="6" fill="#1e293b" stroke="#475569"/>
    <text x="28" y="24" text-anchor="middle" fill="#94a3b8" font-size="8" font-family="system-ui">4090</text>
    <text x="100" y="32" text-anchor="middle" fill="#94a3b8" font-size="8" font-family="system-ui">H100</text>
  </g>
  <text x="160" y="176" text-anchor="middle" fill="#64748b" font-size="10" font-weight="600" letter-spacing="0.1em">DISCOVER</text>
</svg>""",
    "flow-launch.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 200" fill="none">
  <defs>{GRAD}</defs>
  <rect width="320" height="200" rx="20" fill="#0a0e1a" stroke="#334155"/>
  <rect x="32" y="40" width="256" height="56" rx="10" fill="#111827" stroke="#334155"/>
  <text x="48" y="62" fill="#00d4ff" font-family="ui-monospace,monospace" font-size="10">$ spin up 4x A100...</text>
  <path d="M160,96 V120" stroke="url(#g)" stroke-width="2" marker-end="url(#arr)"/>
  <rect x="72" y="124" width="176" height="48" rx="10" fill="#0f172a" stroke="url(#g)" stroke-width="1.25" filter="url(#glow)"/>
  <text x="160" y="152" text-anchor="middle" fill="#e2e8f0" font-size="11" font-weight="600">instance running</text>
  <text x="160" y="184" text-anchor="middle" fill="#64748b" font-size="10" font-weight="600" letter-spacing="0.1em">LAUNCH</text>
</svg>""",
    "flow-guardrails.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 200" fill="none">
  <defs>{GRAD}</defs>
  <rect width="320" height="200" rx="20" fill="#0a0e1a" stroke="#334155"/>
  <path d="M160,44 L200,60 V96 C200,120 160,140 160,140 C160,140 120,120 120,96 V60 Z" fill="#1e293b" stroke="url(#gold)" stroke-width="1.5"/>
  <rect x="56" y="152" width="80" height="10" rx="5" fill="#334155"/>
  <rect x="56" y="152" width="52" height="10" rx="5" fill="url(#gold)"/>
  <text x="96" y="178" text-anchor="middle" fill="#94a3b8" font-size="9">wallet</text>
  <rect x="184" y="152" width="80" height="10" rx="5" fill="#334155"/>
  <rect x="184" y="152" width="64" height="10" rx="5" fill="url(#g)"/>
  <text x="224" y="178" text-anchor="middle" fill="#94a3b8" font-size="9">estimate</text>
  <text x="160" y="24" text-anchor="middle" fill="#fbbf24" font-size="10" font-weight="700" letter-spacing="0.12em">GUARDRAILS</text>
</svg>""",
    "flow-monitor.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 200" fill="none">
  <defs>{GRAD}</defs>
  <rect width="320" height="200" rx="20" fill="#0a0e1a" stroke="#334155"/>
  <path d="M40,120 L72,100 L104,108 L136,72 L168,80 L200,48 L232,56 L264,40 L280,36" stroke="url(#g)" stroke-width="2.5" fill="none" filter="url(#glow)"/>
  <rect x="48" y="136" width="224" height="40" rx="8" fill="#111827" stroke="#334155"/>
  <text x="160" y="162" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="ui-monospace">GPU 94% · 22GB · 68°C</text>
  <text x="160" y="24" text-anchor="middle" fill="#64748b" font-size="10" font-weight="600" letter-spacing="0.1em">MONITOR</text>
</svg>""",
    "agent-cursor.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none">
  <defs>{GRAD}</defs>
  <rect width="64" height="64" rx="14" fill="#0a0e1a" stroke="#334155"/>
  <rect x="12" y="14" width="40" height="36" rx="6" fill="#111827" stroke="url(#g)" stroke-width="1"/>
  <path d="M18,42 L28,32 L34,38 L46,24" stroke="#00d4ff" stroke-width="2" fill="none"/>
</svg>""",
    "agent-claude.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none">
  <defs>{GRAD}</defs>
  <rect width="64" height="64" rx="14" fill="#0a0e1a" stroke="#334155"/>
  <ellipse cx="32" cy="32" rx="18" ry="14" fill="#111827" stroke="url(#g)"/>
  <path d="M22,32 Q32,22 42,32 Q32,42 22,32" fill="url(#g)" opacity="0.35"/>
</svg>""",
    "agent-vscode.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none">
  <defs>{GRAD}</defs>
  <rect width="64" height="64" rx="14" fill="#0a0e1a" stroke="#334155"/>
  <path d="M16,20 L40,32 L16,44 Z" fill="url(#g)" opacity="0.5"/>
  <path d="M40,20 L48,32 L40,44" stroke="#00d4ff" stroke-width="2" fill="none"/>
</svg>""",
    "settings-hero.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 160" fill="none">
  <defs>{GRAD}</defs>
  <rect width="480" height="160" rx="16" fill="#0a0e1a"/>
  <rect width="480" height="160" rx="16" fill="url(#gSoft)"/>
  {SPARK}
  <g transform="translate(200,36) scale(0.7)">{MCP_HEX}</g>
  <text x="300" y="72" fill="#e2e8f0" font-family="system-ui,sans-serif" font-size="20" font-weight="700">Connect AI Agents</text>
  <text x="300" y="98" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="13">From prompt to compute in seconds</text>
</svg>""",
    "og-mcp-card.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 630" fill="none">
  <defs>{GRAD}</defs>
  <rect width="1200" height="630" fill="#0a0e1a"/>
  <rect width="1200" height="630" fill="url(#gSoft)"/>
  <g transform="translate(120,140) scale(1.4)">{SPARK}{MCP_HEX}{GPU_DIE}{FLOW_PATH}</g>
  <text x="640" y="280" fill="#e2e8f0" font-family="system-ui,sans-serif" font-size="56" font-weight="800">Let AI agents control real GPUs</text>
  <text x="640" y="360" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="28">Xcelsior MCP — natural language to compute</text>
  <text x="640" y="480" fill="#00d4ff" font-family="system-ui,sans-serif" font-size="22" font-weight="600">xcelsior.ca/mcp</text>
</svg>""",
}

for name, svg in SLIDES.items():
    (OUT / name).write_text(svg.strip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT / name}")

print(f"Done — {len(SLIDES)} assets in {OUT}")