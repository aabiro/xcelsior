#!/usr/bin/env python3
"""Generate GPU fleet marketing SVG assets (brand-aligned, high-energy)."""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "gpu-fleet"
OUT.mkdir(parents=True, exist_ok=True)

GRAD = """
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00d4ff"/>
      <stop offset="45%" stop-color="#7c3aed"/>
      <stop offset="100%" stop-color="#dc2626"/>
    </linearGradient>
    <linearGradient id="gSoft" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#00d4ff" stop-opacity="0.2"/>
      <stop offset="100%" stop-color="#dc2626" stop-opacity="0.08"/>
    </linearGradient>
    <linearGradient id="chip" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="6" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="softGlow" x="-80%" y="-80%" width="260%" height="260%">
      <feGaussianBlur stdDeviation="18"/>
    </filter>
"""

# Stylized GPU die with heat traces
GPU_DIE = """
  <g filter="url(#glow)">
    <rect x="168" y="88" width="144" height="104" rx="12" fill="url(#chip)" stroke="url(#g)" stroke-width="1.5" opacity="0.95"/>
    <rect x="180" y="100" width="120" height="80" rx="6" fill="#0a0e1a" stroke="#334155" stroke-width="0.75"/>
    <path d="M168,116 H152 M168,136 H148 M168,156 H152 M168,176 H148" stroke="#00d4ff" stroke-opacity="0.35" stroke-width="2" stroke-linecap="round"/>
    <path d="M312,116 H328 M312,136 H332 M312,156 H328 M312,176 H332" stroke="#dc2626" stroke-opacity="0.35" stroke-width="2" stroke-linecap="round"/>
    <circle cx="240" cy="140" r="22" fill="url(#g)" opacity="0.25"/>
    <path d="M228,140 L240,128 L252,140 L240,152 Z" fill="url(#g)" opacity="0.9"/>
  </g>
"""

POWER_WAVES = """
  <g opacity="0.55">
    <ellipse cx="240" cy="140" rx="200" ry="80" fill="none" stroke="url(#g)" stroke-width="1" opacity="0.25"/>
    <ellipse cx="240" cy="140" rx="260" ry="110" fill="none" stroke="url(#g)" stroke-width="0.75" opacity="0.15"/>
    <ellipse cx="240" cy="140" rx="320" ry="140" fill="none" stroke="url(#g)" stroke-width="0.5" opacity="0.08"/>
  </g>
  <circle cx="420" cy="48" r="64" fill="#00d4ff" filter="url(#softGlow)" opacity="0.12"/>
  <circle cx="56" cy="232" r="48" fill="#7c3aed" filter="url(#softGlow)" opacity="0.14"/>
"""

SLIDES = {
    "hero-power.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 320" fill="none">
  <defs>{GRAD}</defs>
  <rect width="480" height="320" rx="28" fill="#0a0e1a"/>
  <rect width="480" height="320" rx="28" fill="url(#gSoft)"/>
  {POWER_WAVES}
  {GPU_DIE}
</svg>""",
    "workload-training.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 120" fill="none">
  <defs>{GRAD}</defs>
  <rect width="200" height="120" rx="16" fill="#0a0e1a" stroke="#334155" stroke-width="0.75"/>
  <rect width="200" height="120" rx="16" fill="url(#gSoft)"/>
  <path d="M28,88 L48,72 L68,76 L88,52 L108,58 L128,38 L148,44 L172,28" stroke="url(#g)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none" filter="url(#glow)"/>
  <circle cx="172" cy="28" r="4" fill="#00d4ff"/>
</svg>""",
    "workload-inference.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 120" fill="none">
  <defs>{GRAD}</defs>
  <rect width="200" height="120" rx="16" fill="#0a0e1a" stroke="#334155" stroke-width="0.75"/>
  <rect width="200" height="120" rx="16" fill="url(#gSoft)"/>
  <g transform="translate(100,52)">
    <circle r="28" stroke="url(#g)" stroke-width="1.5" fill="#111827" opacity="0.9"/>
    <path d="M-8,-4 L8,-4 L4,8 L-4,8 Z" fill="url(#g)" opacity="0.85"/>
    <path d="M-18,0 A18,18 0 1,1 18,0" stroke="#00d4ff" stroke-width="1.5" fill="none" stroke-dasharray="4 6" opacity="0.6"/>
  </g>
</svg>""",
    "deploy-pipeline.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 200" fill="none">
  <defs>{GRAD}</defs>
  <rect width="640" height="200" rx="24" fill="#0a0e1a"/>
  <rect width="640" height="200" rx="24" fill="url(#gSoft)"/>
  <path d="M88,100 H200 M440,100 H552" stroke="url(#g)" stroke-width="2" stroke-dasharray="6 8" opacity="0.45"/>
  <circle cx="88" cy="100" r="36" fill="#111827" stroke="url(#g)" stroke-width="1.5"/>
  <circle cx="320" cy="100" r="44" fill="#111827" stroke="url(#g)" stroke-width="2" filter="url(#glow)"/>
  <circle cx="552" cy="100" r="36" fill="#111827" stroke="url(#g)" stroke-width="1.5"/>
  <path d="M308,92 L320,76 L332,92 L324,92 L324,108 L316,108 L316,92 Z" fill="url(#g)"/>
  <rect x="68" y="86" width="40" height="28" rx="6" fill="url(#chip)" stroke="#334155"/>
  <rect x="532" y="86" width="40" height="28" rx="6" fill="url(#chip)" stroke="#334155"/>
  <circle cx="552" cy="100" r="8" fill="#10b981" opacity="0.9"/>
</svg>""",
    "spot-pulse.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 480 120" fill="none">
  <defs>{GRAD}</defs>
  <rect width="480" height="120" rx="20" fill="#0a0e1a" stroke="#334155" stroke-width="0.75"/>
  <rect width="480" height="120" rx="20" fill="url(#gSoft)"/>
  <path d="M40,78 L80,62 L120,68 L160,44 L200,52 L240,36 L280,48 L320,32 L360,40 L400,28 L440,34" stroke="url(#g)" stroke-width="2.5" stroke-linecap="round" filter="url(#glow)"/>
  <circle cx="440" cy="34" r="5" fill="#10b981"/>
  <ellipse cx="240" cy="60" rx="180" ry="40" fill="#10b981" opacity="0.06"/>
</svg>""",
    "accent-flare.svg": f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">
  <defs>{GRAD}</defs>
  <circle cx="60" cy="60" r="50" fill="#00d4ff" filter="url(#softGlow)" opacity="0.18"/>
  <path d="M60,8 L68,44 L104,52 L68,60 L60,96 L52,60 L16,52 L52,44 Z" fill="url(#g)" opacity="0.35"/>
</svg>""",
}

for name, svg in SLIDES.items():
    (OUT / name).write_text(svg.strip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT / name}")

print(f"Done — {len(SLIDES)} assets in {OUT}")