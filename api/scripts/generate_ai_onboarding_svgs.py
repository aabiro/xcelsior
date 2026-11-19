#!/usr/bin/env python3
"""Generate the Xcel AI onboarding illustrations, dark and light.

The four slides in `components/ai/xcel-ai-onboarding.tsx` are drawn in the same
family as `public/gpu.svg`, `public/gpu-fleet/fleet-pulse.*.svg` and the site
icons: glossy navy (or white) bodies with the cyan → violet → emerald edge,
filled gradient glyphs like the fleet slabs, a soft glow, a ground shadow and a
fading horizon. The X is the brand mark with the brand gradient
(`brand-system/icon-gradient.svg`).

They carry no text; the modal prints each slide's title and body below the
picture, in the reader's language. Motion (flowing wires, pulsing rings) is CSS
inside the SVG, so it runs when the file is shown through an <img>, and it is
switched off under `prefers-reduced-motion`.

Each slide is written twice, `<name>.svg` (dark) and `<name>-light.svg`. The
output is deterministic, and `tests/test_svg_generators_that_can_be_gated_are.py`
fails when the committed files differ from a fresh run, so edit this file and
regenerate rather than editing the SVGs:

    python api/scripts/generate_ai_onboarding_svgs.py

The modal requests these with `?v=<ASSET_VERSION>`; bump it there whenever the
artwork changes, because `/:path*.svg` is served `immutable` and a browser that
cached the old file would otherwise never ask again.
"""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "ai-onboarding"

W, H = 480, 240
HORIZON_Y = 214

THEMES = {
    "dark": {
        "suffix": "",
        "body": ("#1b2a46", "#0e182b", "#070f1c"),
        "edge": ("#38dbff", "#9b6dff", "#34d399"),
        "fills": {
            "cyan": ("#38dbff", "#9b6dff"),
            "violet": ("#9b6dff", "#ff8f8f"),
            "green": ("#34d399", "#38dbff"),
            "warm": ("#ffbf5e", "#ff8f8f"),
            "badge": ("#38dbff", "#9b6dff"),
        },
        "brand": ("#00d4ff", "#7c3aed"),
        "glow": (("0", "0", "5", "#38dbff", ".5"), ("0", "10", "14", "#9b6dff", ".34")),
        "node_glow": ("#38dbff", ".75"),
        "shadow": ("#38dbff", ".3"),
        "bloom": ("#38dbff", "#9b6dff", ".11"),
        "gloss": ("#ffffff", ".16"),
        "screen": "#060c18",
        "ink": "#ffffff",
        "ink_opacity": ".85",
        "spark_opacity": ".5",
        "line_opacity": ".5",
    },
    "light": {
        "suffix": "-light",
        "body": ("#ffffff", "#f4f7fb", "#e6ecf4"),
        "edge": ("#0891b2", "#7c3aed", "#059669"),
        "fills": {
            "cyan": ("#06b6d4", "#7c3aed"),
            "violet": ("#8b5cf6", "#f43f5e"),
            "green": ("#10b981", "#0891b2"),
            "warm": ("#f59e0b", "#f43f5e"),
            "badge": ("#06b6d4", "#7c3aed"),
        },
        "brand": ("#00c2ef", "#7c3aed"),
        "glow": (("0", "0", "4", "#0891b2", ".22"), ("0", "9", "12", "#0f172a", ".16")),
        "node_glow": ("#0891b2", ".35"),
        "shadow": ("#1e293b", ".18"),
        "bloom": ("#0891b2", "#7c3aed", ".07"),
        "gloss": ("#ffffff", ".95"),
        "screen": "#eef2f8",
        "ink": "#ffffff",
        "ink_opacity": ".95",
        "spark_opacity": ".55",
        "line_opacity": ".55",
    },
}

#: The Xcelsior mark, from brand-system/icon-gradient.svg (240-unit box).
X_MARK = "M56.5,54 L92.5,54 L122.9,102 L168.5,54 L204.5,54 L138,120 L183.6,186 L147.6,186 L117.2,138 L71.6,186 L35.5,186 L102,120 Z"

#: Motion, off for readers who ask for less. Each SVG is its own document when
#: shown through <img>, so the class names cannot collide with the page.
STYLE = (
    "<style>@media (prefers-reduced-motion:no-preference){"
    ".flow{animation:flow 1.8s linear infinite}"
    "@keyframes flow{to{stroke-dashoffset:-24}}"
    ".ring{transform-box:fill-box;transform-origin:center;animation:ring 3.2s ease-out infinite}"
    ".ring.d2{animation-delay:1.07s}.ring.d3{animation-delay:2.13s}"
    "@keyframes ring{0%{transform:scale(.72);opacity:.9}100%{transform:scale(1.32);opacity:0}}"
    ".twinkle{transform-box:fill-box;transform-origin:center;animation:twinkle 2.6s ease-in-out infinite}"
    ".twinkle.d2{animation-delay:1.3s}"
    "@keyframes twinkle{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.45;transform:scale(.78)}}"
    ".blink{animation:blink 1.1s steps(1) infinite}@keyframes blink{50%{opacity:0}}"
    ".breathe{animation:breathe 2.4s ease-in-out infinite}@keyframes breathe{0%,100%{opacity:.35}50%{opacity:1}}"
    "}</style>"
)


def _n(v: float) -> str:
    """Numbers without float noise, so a rerun is byte-identical."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


class Slide:
    """One SVG, its <defs> collected as the shapes ask for them."""

    def __init__(self, name: str, theme: dict) -> None:
        self.t = theme
        self.p = f"ao-{name}{'-l' if theme['suffix'] else '-d'}"
        self.defs: list[str] = []
        self.body: list[str] = []
        e = theme["edge"]
        g = theme["glow"]
        gc, go = theme["gloss"]
        self.defs += [
            f'<linearGradient id="{self.p}-edge" x1="40" y1="20" x2="440" y2="220" gradientUnits="userSpaceOnUse">'
            f'<stop stop-color="{e[0]}"/><stop offset=".55" stop-color="{e[1]}"/><stop offset="1" stop-color="{e[2]}"/></linearGradient>',
            f'<linearGradient id="{self.p}-horizon" x1="40" y1="0" x2="440" y2="0" gradientUnits="userSpaceOnUse">'
            f'<stop stop-color="{e[0]}" stop-opacity="0"/><stop offset=".5" stop-color="{e[0]}" stop-opacity=".5"/>'
            f'<stop offset="1" stop-color="{e[1]}" stop-opacity="0"/></linearGradient>',
            f'<radialGradient id="{self.p}-shadow" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="{theme["shadow"][0]}" '
            f'stop-opacity="{theme["shadow"][1]}"/><stop offset="100%" stop-color="{theme["shadow"][0]}" stop-opacity="0"/></radialGradient>',
            # The glass highlight across the top of every body, in its own bounding box.
            f'<linearGradient id="{self.p}-gloss" x1="0" y1="0" x2="0" y2="1">'
            f'<stop stop-color="{gc}" stop-opacity="{go}"/><stop offset=".48" stop-color="{gc}" stop-opacity="0"/></linearGradient>',
            f'<linearGradient id="{self.p}-brand" x1="0" y1="0" x2="0" y2="1">'
            f'<stop stop-color="{theme["brand"][0]}"/><stop offset="1" stop-color="{theme["brand"][1]}"/></linearGradient>',
            f'<filter id="{self.p}-glow" x="-40%" y="-40%" width="180%" height="180%" color-interpolation-filters="sRGB">'
            + "".join(
                f'<feDropShadow dx="{dx}" dy="{dy}" stdDeviation="{sd}" flood-color="{c}" flood-opacity="{o}"/>'
                for dx, dy, sd, c, o in g
            )
            + "</filter>",
            f'<filter id="{self.p}-nglow" x="-140%" y="-140%" width="380%" height="380%" color-interpolation-filters="sRGB">'
            f'<feDropShadow dx="0" dy="0" stdDeviation="3" flood-color="{theme["node_glow"][0]}" flood-opacity="{theme["node_glow"][1]}"/></filter>',
            f'<filter id="{self.p}-bloom" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="30"/></filter>',
        ]
        for key, (a, b) in theme["fills"].items():
            self.defs.append(
                f'<linearGradient id="{self.p}-{key}" x1=".14" y1=".06" x2=".8" y2=".96">'
                f'<stop stop-color="{a}"/><stop offset="1" stop-color="{b}"/></linearGradient>'
            )
        self._grads = 0
        self._b = theme["body"]

    # ── defs on demand ────────────────────────────────────────────────
    def body_grad(self, x1: float, y1: float, x2: float, y2: float) -> str:
        self._grads += 1
        gid = f"{self.p}-body{self._grads}"
        b = self._b
        self.defs.append(
            f'<linearGradient id="{gid}" x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}" gradientUnits="userSpaceOnUse">'
            f'<stop stop-color="{b[0]}"/><stop offset=".55" stop-color="{b[1]}"/><stop offset="1" stop-color="{b[2]}"/></linearGradient>'
        )
        return gid

    def edge_grad(self, x1: float, y1: float, x2: float, y2: float) -> str:
        self._grads += 1
        gid = f"{self.p}-fill{self._grads}"
        e = self.t["edge"]
        self.defs.append(
            f'<linearGradient id="{gid}" x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}" gradientUnits="userSpaceOnUse">'
            f'<stop stop-color="{e[0]}"/><stop offset=".58" stop-color="{e[1]}"/><stop offset="1" stop-color="{e[2]}"/></linearGradient>'
        )
        return gid

    # ── shared scenery ────────────────────────────────────────────────
    def backdrop(self, blooms: tuple[tuple[float, float, float, int], ...]) -> None:
        c0, c1, op = self.t["bloom"]
        for cx, cy, r, which in blooms:
            self.body.append(
                f'<circle cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}" fill="{c0 if which == 0 else c1}" opacity="{op}" filter="url(#{self.p}-bloom)"/>'
            )

    def sparks(self, points: tuple[tuple[float, float], ...]) -> None:
        e = self.t["edge"][0]
        for x, y in points:
            self.body.append(
                f'<path d="M{_n(x - 4)} {_n(y)}h8M{_n(x)} {_n(y - 4)}v8" stroke="{e}" stroke-opacity="{self.t["spark_opacity"]}" '
                'stroke-width="1.4" stroke-linecap="round"/>'
            )

    def ground(self, cx: float, rx: float, y: float) -> None:
        self.body.append(
            f'<ellipse cx="{_n(cx)}" cy="{_n(y)}" rx="{_n(rx)}" ry="8" fill="url(#{self.p}-shadow)"/>'
        )

    def horizon(self) -> None:
        self.body.append(
            f'<path d="M40 {HORIZON_Y}H440" stroke="url(#{self.p}-horizon)" stroke-width="2" stroke-linecap="round"/>'
        )

    def dot(self, cx: float, cy: float, color: int, r: float = 2.6, cls: str = "") -> None:
        c = f' class="{cls}"' if cls else ""
        self.body.append(
            f'<circle cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}" fill="{self.t["edge"][color]}"{c}/>'
        )

    def node(self, cx: float, cy: float, r: float, fill: str) -> None:
        """A glowing bead: a gradient sphere with a highlight."""
        self.body.append(
            f'<circle cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}" fill="url(#{self.p}-{fill})" filter="url(#{self.p}-nglow)"/>'
        )
        self.body.append(
            f'<circle cx="{_n(cx - r * 0.3)}" cy="{_n(cy - r * 0.32)}" r="{_n(r * 0.34)}" fill="#ffffff" fill-opacity=".75"/>'
        )

    def gloss(self, shape: str) -> None:
        """The glass highlight over a body; `shape` is the body's own element, unfilled."""
        self.body.append(shape.replace("/>", f' fill="url(#{self.p}-gloss)"/>'))

    def tile(self, x: float, y: float, w: float, h: float, rx: float, *, glow: bool = True) -> None:
        bg = self.body_grad(x, y, x + w, y + h)
        f = f' filter="url(#{self.p}-glow)"' if glow else ""
        self.body.append(
            f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" rx="{_n(rx)}" fill="url(#{bg})" '
            f'stroke="url(#{self.p}-edge)" stroke-width="3"{f}/>'
        )
        i = 2.5
        self.gloss(
            f'<rect x="{_n(x + i)}" y="{_n(y + i)}" width="{_n(w - 2 * i)}" height="{_n(h - 2 * i)}" rx="{_n(max(rx - i, 2))}"/>'
        )

    def sparkle(self, cx: float, cy: float, s: float, *, glow: bool = True, cls: str = "") -> None:
        """A four-point star, the Xcel AI sign."""
        k = s * 0.22
        d = (
            f"M{_n(cx)} {_n(cy - s)}Q{_n(cx + k)} {_n(cy - k)} {_n(cx + s)} {_n(cy)}"
            f"Q{_n(cx + k)} {_n(cy + k)} {_n(cx)} {_n(cy + s)}"
            f"Q{_n(cx - k)} {_n(cy + k)} {_n(cx - s)} {_n(cy)}"
            f"Q{_n(cx - k)} {_n(cy - k)} {_n(cx)} {_n(cy - s)}Z"
        )
        f = f' filter="url(#{self.p}-glow)"' if glow else ""
        open_g = f'<g class="{cls}">' if cls else "<g>"
        self.body.append(
            f'{open_g}<path d="{d}" fill="url(#{self.p}-cyan)"{f}/>'
            f'<path d="{d}" stroke="{self.t["ink"]}" stroke-opacity="{self.t["ink_opacity"]}" stroke-width="1.2" stroke-linejoin="round"/>'
            f'<circle cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(max(s * 0.14, 1))}" fill="#ffffff"/></g>'
        )

    def glyph(self, cx: float, cy: float, parts: list[str], scale: float = 1.0) -> None:
        """Filled art on a 32-unit grid centred on (0, 0). `{p}` in a part is this slide's id prefix."""
        inner = "".join(
            part.replace("{p}", self.p).replace("{screen}", self.t["screen"]) for part in parts
        )
        sc = f" scale({_n(scale)})" if scale != 1 else ""
        self.body.append(f'<g transform="translate({_n(cx)},{_n(cy)}){sc}">{inner}</g>')

    def render(self) -> str:
        header = "<!-- Generated by api/scripts/generate_ai_onboarding_svgs.py; edit the generator, not this file. -->"
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" fill="none">\n{header}\n{STYLE}\n<defs>\n  '
            + "\n  ".join(self.defs)
            + "\n</defs>\n"
            + "\n".join(self.body)
            + "\n</svg>\n"
        )


# ── The four slides ───────────────────────────────────────────────────


def _orbit_point(
    cx: float, cy: float, rx: float, ry: float, rot: float, deg: float
) -> tuple[float, float]:
    a, t = math.radians(deg), math.radians(rot)
    x, y = rx * math.cos(a), ry * math.sin(a)
    return cx + x * math.cos(t) - y * math.sin(t), cy + x * math.sin(t) + y * math.cos(t)


def hero(s: Slide) -> None:
    """Meet Xcel: the Xcelsior app tile with an AI sparkle, held in two orbits."""
    s.backdrop(((150, 96, 96, 0), (330, 126, 104, 1)))
    s.sparks(((70, 58), (424, 66), (108, 180), (392, 184), (176, 26)))
    cx, cy = 240, 106
    orbits = ((160, 46, -10), (122, 70, 16))
    for (rx, ry, rot), op in zip(orbits, (".42", ".26")):
        ell = f'cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" transform="rotate({rot} {cx} {cy})"'
        s.body.append(
            f'<ellipse {ell} stroke="url(#{s.p}-edge)" stroke-opacity="{op}" stroke-width="1.6"/>'
        )
        s.body.append(
            f'<ellipse {ell} class="flow" stroke-dasharray="3 9" stroke="url(#{s.p}-edge)" stroke-opacity=".9" stroke-width="1.6"/>'
        )
    # Beads riding the orbits; the ones in front of the tile are drawn after it.
    beads = (
        (0, 200, 7, "cyan", False),
        (0, 22, 6, "green", True),
        (1, 300, 6, "violet", False),
        (1, 140, 4.5, "cyan", True),
    )
    for i, deg, r, fill, front in beads:
        if not front:
            s.node(*_orbit_point(cx, cy, *orbits[i], deg), r, fill)
    s.ground(cx, 80, 188)
    size = 108
    s.tile(cx - size / 2, cy - size / 2, size, size, 28)
    scale = 0.44
    xf = f'transform="translate({_n(cx - 120 * scale)},{_n(cy - 120 * scale)}) scale({_n(scale)})"'
    s.body.append(f'<path {xf} d="{X_MARK}" fill="url(#{s.p}-brand)" filter="url(#{s.p}-glow)"/>')
    for i, deg, r, fill, front in beads:
        if front:
            s.node(*_orbit_point(cx, cy, *orbits[i], deg), r, fill)
    s.sparkle(cx + 62, cy - 58, 17, cls="twinkle")
    s.sparkle(cx + 88, cy - 76, 7, glow=False, cls="twinkle d2")
    s.horizon()


#: Filled glyphs for the action tiles, after the fleet slabs.
ROCKET = [
    '<path d="M-6 4L-13 12L-12 4L-6 -2Z" fill="url(#{p}-violet)"/>',
    '<path d="M6 4L13 12L12 4L6 -2Z" fill="url(#{p}-violet)"/>',
    '<path d="M-4 11Q0 22 4 11Z" fill="url(#{p}-warm)" class="breathe"/>',
    '<path d="M0 -17C9 -11 10 2 6.5 11H-6.5C-10 2 -9 -11 0 -17Z" fill="url(#{p}-cyan)"/>',
    '<path d="M0 -17C9 -11 10 2 6.5 11H-6.5C-10 2 -9 -11 0 -17Z" fill="url(#{p}-gloss)"/>',
    '<circle cx="0" cy="-3" r="4" fill="{screen}" stroke="#ffffff" stroke-opacity=".85" stroke-width="1.6"/>',
]
SERVER = [
    '<path d="M-13 -15H15L13 -3H-15Z" fill="url(#{p}-cyan)"/>',
    '<path d="M-13 -15H15L13 -3H-15Z" fill="url(#{p}-gloss)"/>',
    '<path d="M-13 1H15L13 13H-15Z" fill="url(#{p}-violet)"/>',
    '<path d="M-13 1H15L13 13H-15Z" fill="url(#{p}-gloss)"/>',
    '<path d="M-9 -9H0M-9 7H0" stroke="#ffffff" stroke-opacity=".9" stroke-width="2.2" stroke-linecap="round"/>',
    '<circle cx="6" cy="-9" r="1.8" fill="#ffffff"/><circle cx="10" cy="-9" r="1.8" fill="#ffffff" fill-opacity=".6"/>',
    '<circle cx="6" cy="7" r="1.8" fill="#ffffff"/><circle cx="10" cy="7" r="1.8" fill="#ffffff" fill-opacity=".6"/>',
]
WALLET = [
    '<rect x="-15" y="-12" width="26" height="8" rx="3" fill="url(#{p}-warm)" transform="rotate(-8)"/>',
    '<rect x="-15" y="-8" width="30" height="21" rx="5" fill="url(#{p}-green)"/>',
    '<rect x="-15" y="-8" width="30" height="21" rx="5" fill="url(#{p}-gloss)"/>',
    '<path d="M15 -1H7a4.5 4.5 0 0 0 0 9H15Z" fill="{screen}" fill-opacity=".55"/>',
    '<circle cx="8" cy="3.5" r="2" fill="#ffffff"/>',
]
TERMINAL = [
    '<rect x="-15" y="-12" width="30" height="24" rx="5" fill="url(#{p}-violet)"/>',
    '<rect x="-12.5" y="-6" width="25" height="15.5" rx="2.5" fill="{screen}"/>',
    '<circle cx="-10.5" cy="-9" r="1.2" fill="#ffffff"/><circle cx="-6.5" cy="-9" r="1.2" fill="#ffffff" fill-opacity=".7"/>',
    '<path d="M-8 -2L-3.5 1.5L-8 5" stroke="url(#{p}-cyan)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
    '<path d="M0 6H8" stroke="#38dbff" stroke-width="2.2" stroke-linecap="round" class="blink"/>',
]


def actions(s: Slide) -> None:
    """Xcel works the dashboard: an AI hub wired to instances, hosts, billing and settings."""
    s.backdrop(((140, 110, 96, 0), (340, 110, 100, 1)))
    s.sparks(((62, 30), (420, 34), (240, 16), (40, 196), (440, 194)))
    hub = (240, 106)
    tiles = (
        ((110, 62), ROCKET, 0),
        ((110, 152), SERVER, 1),
        ((370, 62), WALLET, 2),
        ((370, 152), TERMINAL, 0),
    )
    for (tx, ty), _glyph, color in tiles:
        wire = f'd="M{hub[0]} {hub[1]}L{tx} {ty}"'
        s.body.append(
            f'<path {wire} stroke="url(#{s.p}-edge)" stroke-opacity="{s.t["line_opacity"]}" stroke-width="1.8"/>'
        )
        s.body.append(
            f'<path {wire} class="flow" stroke-dasharray="3 9" stroke="{s.t["edge"][color]}" stroke-width="2.2" stroke-linecap="round"/>'
        )
    s.ground(240, 156, 194)
    for (tx, ty), glyph, _color in tiles:
        s.tile(tx - 31, ty - 31, 62, 62, 17)
        s.glyph(tx, ty, glyph, 1.25)
    for cls in ("ring", "ring d2"):
        s.body.append(
            f'<circle class="{cls}" opacity="0" cx="{hub[0]}" cy="{hub[1]}" r="38" stroke="url(#{s.p}-edge)" stroke-opacity=".5" stroke-width="1.6"/>'
        )
    bg = s.body_grad(hub[0] - 31, hub[1] - 31, hub[0] + 31, hub[1] + 31)
    s.body.append(
        f'<circle cx="{hub[0]}" cy="{hub[1]}" r="31" fill="url(#{bg})" stroke="url(#{s.p}-edge)" stroke-width="3" filter="url(#{s.p}-glow)"/>'
    )
    s.gloss(f'<circle cx="{hub[0]}" cy="{hub[1]}" r="28.5"/>')
    s.sparkle(hub[0], hub[1], 17, glow=False, cls="twinkle")
    s.horizon()


def api(s: Slide) -> None:
    """Full API reach, with a guard: a console whose destructive call waits on a confirm shield."""
    s.backdrop(((170, 100, 100, 0), (330, 140, 100, 1)))
    s.sparks(((70, 46), (420, 40), (96, 188), (424, 184)))
    x, y, w, h = 124, 34, 214, 152
    s.ground(236, 128, 198)
    s.tile(x, y, w, h, 16)
    e = s.t["edge"]
    for i, dx in enumerate((16, 29, 42)):
        s.body.append(f'<circle cx="{x + dx}" cy="{y + 16}" r="3.6" fill="{e[i]}"/>')
    s.body.append(
        f'<path d="M{x + 3} {y + 30}H{x + w - 3}" stroke="url(#{s.p}-edge)" stroke-opacity=".3" stroke-width="1.5"/>'
    )
    # Request rows: a method pill, then the path. The last is destructive and held.
    rows = (
        (y + 50, 30, 96, "cyan", ".85"),
        (y + 74, 30, 74, "green", ".4"),
        (y + 98, 38, 66, "violet", ".4"),
        (y + 124, 46, 54, "warm", ".4"),
    )
    for i, (ry, pill, path_len, fill, op) in enumerate(rows):
        x0 = x + 26
        if i == 0:
            s.body.append(
                f'<path d="M{x + 12} {ry - 4.5}l5 4.5l-5 4.5" stroke="{e[0]}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        if i == len(rows) - 1:
            s.body.append(
                f'<rect x="{x0 - 7}" y="{ry - 11}" width="{pill + path_len + 26}" height="22" rx="7" '
                f'stroke="url(#{s.p}-warm)" stroke-width="1.5" stroke-dasharray="4 4" class="breathe"/>'
            )
        s.body.append(
            f'<rect x="{x0}" y="{ry - 5}" width="{pill}" height="10" rx="5" fill="url(#{s.p}-{fill})"/>'
        )
        s.body.append(
            f'<path d="M{x0 + pill + 10} {ry}h{path_len}" stroke="url(#{s.p}-edge)" stroke-opacity="{op}" '
            'stroke-width="5" stroke-linecap="round"/>'
        )
    s.body.append(
        f'<rect class="blink" x="{x + 26 + 30 + 10 + 96 + 8}" y="{y + 44}" width="7" height="12" rx="1.5" fill="{e[0]}"/>'
    )
    # The confirm shield, standing over the console's corner.
    sx, sy = 340, 152
    shield = (
        f"M{sx} {sy - 34}L{sx + 27} {sy - 24}V{sy - 3}C{sx + 27} {sy + 16} {sx + 15} {sy + 27} {sx} {sy + 34}"
        f"C{sx - 15} {sy + 27} {sx - 27} {sy + 16} {sx - 27} {sy - 3}V{sy - 24}Z"
    )
    for cls in ("ring", "ring d2"):
        s.body.append(
            f'<circle class="{cls}" opacity="0" cx="{sx}" cy="{sy}" r="40" stroke="url(#{s.p}-edge)" stroke-opacity=".45" stroke-width="1.5"/>'
        )
    s.body.append(f'<path d="{shield}" fill="url(#{s.p}-badge)" filter="url(#{s.p}-glow)"/>')
    s.gloss(f'<path d="{shield}"/>')
    s.body.append(
        f'<path d="{shield}" stroke="{s.t["ink"]}" stroke-opacity="{s.t["ink_opacity"]}" stroke-width="2" stroke-linejoin="round"/>'
    )
    s.body.append(
        f'<path d="M{sx - 11} {sy + 1}l8 8l15 -17" stroke="{s.t["ink"]}" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    s.horizon()


def ready(s: Slide) -> None:
    """All set: a check badge pulsing over a live circuit rail."""
    s.backdrop(((180, 96, 100, 0), (310, 110, 100, 1)))
    s.sparks(((84, 50), (400, 44), (130, 166), (360, 170)))
    cx, cy = 240, 96
    for r, op in ((92, ".1"), (72, ".2"), (56, ".34")):
        s.body.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="url(#{s.p}-edge)" stroke-opacity="{op}" stroke-width="1.5"/>'
        )
    for cls in ("ring", "ring d2", "ring d3"):
        s.body.append(
            f'<circle class="{cls}" opacity="0" cx="{cx}" cy="{cy}" r="60" stroke="url(#{s.p}-edge)" stroke-opacity=".55" stroke-width="2"/>'
        )
    e = s.t["edge"]
    # The rail the badge stands on, after the bolt icon's.
    rail = s.edge_grad(132, 0, 348, 0)
    track = 'd="M132 186h58l12 -12h76l12 12h58"'
    s.body.append(
        f'<path {track} stroke="url(#{rail})" stroke-opacity=".75" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    s.body.append(
        f'<path {track} class="flow" stroke-dasharray="3 9" stroke="#ffffff" stroke-opacity=".7" stroke-width="1.6" stroke-linecap="round"/>'
    )
    for i, (px, depth) in enumerate(((204, 12), (240, 18), (276, 12))):
        top = 176 if i != 1 else 174
        s.body.append(
            f'<path d="M{px} {top}v{depth}" stroke="{e[0]}" stroke-opacity=".4" stroke-width="2" stroke-linecap="round"/>'
        )
        s.dot(px, top + depth + 3.5, i, 3)
    s.ground(cx, 74, 160)
    s.body.append(
        f'<circle cx="{cx}" cy="{cy}" r="44" fill="url(#{s.p}-badge)" filter="url(#{s.p}-glow)"/>'
    )
    s.gloss(f'<circle cx="{cx}" cy="{cy}" r="42"/>')
    s.body.append(
        f'<circle cx="{cx}" cy="{cy}" r="44" stroke="{s.t["ink"]}" stroke-opacity="{s.t["ink_opacity"]}" stroke-width="2.2"/>'
    )
    s.body.append(
        f'<path d="M{cx - 18} {cy + 1}l12 12l25 -27" stroke="{s.t["ink"]}" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    s.sparkle(cx + 54, cy - 50, 11, glow=False, cls="twinkle")
    s.sparkle(cx - 60, cy - 40, 7, glow=False, cls="twinkle d2")
    s.horizon()


SLIDES = {"hero": hero, "actions": actions, "api": api, "ready": ready}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, draw in SLIDES.items():
        for theme in THEMES.values():
            slide = Slide(name, theme)
            draw(slide)
            path = OUT / f"{name}{theme['suffix']}.svg"
            path.write_text(slide.render(), encoding="utf-8")
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
