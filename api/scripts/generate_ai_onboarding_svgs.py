#!/usr/bin/env python3
"""Generate the Xcel AI onboarding illustrations, dark and light.

The four slides in `components/ai/xcel-ai-onboarding.tsx` are drawn in the same
family as `public/gpu.svg`, `public/mesh-*.svg` and the site icons
(`site-assets/assets/icons/bolt.*.svg`): navy or white bodies with a
cyan → violet → emerald edge, a soft glow, a ground shadow and a fading horizon.
They carry no text; the modal prints each slide's title and body below the
picture, in the reader's language.

Each slide is written twice, `<name>.svg` (dark) and `<name>-light.svg`. The
output is deterministic, and `tests/test_svg_generators_that_can_be_gated_are.py`
fails when the committed files differ from a fresh run, so edit this file and
regenerate rather than editing the SVGs:

    python api/scripts/generate_ai_onboarding_svgs.py
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
        "body": ("#17243d", "#0c1526", "#070f1c"),
        "edge": ("#38dbff", "#9b6dff", "#34d399"),
        "glow": (("0", "0", "4.5", "#38dbff", ".55"), ("0", "9", "13", "#9b6dff", ".3")),
        "node_glow": ("#38dbff", ".7"),
        "shadow": ("#38dbff", ".28"),
        "bloom": ("#38dbff", "#9b6dff", ".07"),
        "panel": ("#ecfeff", ".05"),
        "ink": "#ecfeff",
        "ink_opacity": ".68",
        "spark_opacity": ".45",
        "line_opacity": ".42",
    },
    "light": {
        "suffix": "-light",
        "body": ("#ffffff", "#f1f5f9", "#e2e8f0"),
        "edge": ("#0891b2", "#7c3aed", "#059669"),
        "glow": (("0", "0", "4", "#0891b2", ".24"), ("0", "8", "10", "#0f172a", ".16")),
        "node_glow": ("#0891b2", ".32"),
        "shadow": ("#1e293b", ".16"),
        "bloom": ("#0891b2", "#7c3aed", ".05"),
        "panel": ("#0f172a", ".03"),
        "ink": "#ffffff",
        "ink_opacity": ".9",
        "spark_opacity": ".5",
        "line_opacity": ".5",
    },
}

#: The Xcelsior mark, from site-assets/assets/icon-gradient.svg (240-unit box).
X_MARK = "M56.5,54 L92.5,54 L122.9,102 L168.5,54 L204.5,54 L138,120 L183.6,186 L147.6,186 L117.2,138 L71.6,186 L35.5,186 L102,120 Z"


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
        b, e = theme["body"], theme["edge"]
        g = theme["glow"]
        self.defs += [
            f'<linearGradient id="{self.p}-edge" x1="40" y1="20" x2="440" y2="220" gradientUnits="userSpaceOnUse">'
            f'<stop stop-color="{e[0]}"/><stop offset=".55" stop-color="{e[1]}"/><stop offset="1" stop-color="{e[2]}"/></linearGradient>',
            f'<linearGradient id="{self.p}-horizon" x1="40" y1="0" x2="440" y2="0" gradientUnits="userSpaceOnUse">'
            f'<stop stop-color="{e[0]}" stop-opacity="0"/><stop offset=".5" stop-color="{e[0]}" stop-opacity=".45"/>'
            f'<stop offset="1" stop-color="{e[1]}" stop-opacity="0"/></linearGradient>',
            f'<radialGradient id="{self.p}-shadow" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="{theme["shadow"][0]}" '
            f'stop-opacity="{theme["shadow"][1]}"/><stop offset="100%" stop-color="{theme["shadow"][0]}" stop-opacity="0"/></radialGradient>',
            f'<filter id="{self.p}-glow" x="-35%" y="-35%" width="170%" height="170%" color-interpolation-filters="sRGB">'
            + "".join(
                f'<feDropShadow dx="{dx}" dy="{dy}" stdDeviation="{sd}" flood-color="{c}" flood-opacity="{o}"/>'
                for dx, dy, sd, c, o in g
            )
            + "</filter>",
            f'<filter id="{self.p}-nglow" x="-140%" y="-140%" width="380%" height="380%" color-interpolation-filters="sRGB">'
            f'<feDropShadow dx="0" dy="0" stdDeviation="3" flood-color="{theme["node_glow"][0]}" flood-opacity="{theme["node_glow"][1]}"/></filter>',
            f'<filter id="{self.p}-bloom" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="30"/></filter>',
        ]
        self._grads = 0
        self._b = b

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
            f'<ellipse cx="{_n(cx)}" cy="{_n(y)}" rx="{_n(rx)}" ry="7" fill="url(#{self.p}-shadow)"/>'
        )

    def horizon(self) -> None:
        self.body.append(
            f'<path d="M40 {HORIZON_Y}H440" stroke="url(#{self.p}-horizon)" stroke-width="2" stroke-linecap="round"/>'
        )

    def dot(self, cx: float, cy: float, color: int, r: float = 2.6) -> None:
        self.body.append(f'<circle cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}" fill="{self.t["edge"][color]}"/>')

    def node(self, cx: float, cy: float, r: float, color: int) -> None:
        bg = self.body_grad(cx - r, cy - r, cx + r, cy + r)
        self.body.append(
            f'<circle cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}" fill="url(#{bg})" stroke="url(#{self.p}-edge)" '
            f'stroke-width="2" filter="url(#{self.p}-nglow)"/>'
        )
        self.dot(cx, cy, color, r * 0.55)

    def tile(self, x: float, y: float, w: float, h: float, rx: float, *, glow: bool = True) -> None:
        bg = self.body_grad(x, y, x + w, y + h)
        f = f' filter="url(#{self.p}-glow)"' if glow else ""
        self.body.append(
            f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}" rx="{_n(rx)}" fill="url(#{bg})" '
            f'stroke="url(#{self.p}-edge)" stroke-width="2.4"{f}/>'
        )
        pc, po = self.t["panel"]
        inset = max(4.0, min(w, h) * 0.06)
        self.body.append(
            f'<rect x="{_n(x + inset)}" y="{_n(y + inset)}" width="{_n(w - 2 * inset)}" height="{_n(h - 2 * inset)}" '
            f'rx="{_n(max(rx - inset, 2))}" fill="{pc}" fill-opacity="{po}"/>'
        )

    def sparkle(self, cx: float, cy: float, s: float, *, glow: bool = True) -> None:
        """A four-point star, the Xcel AI sign."""
        k = s * 0.22
        d = (
            f"M{_n(cx)} {_n(cy - s)}Q{_n(cx + k)} {_n(cy - k)} {_n(cx + s)} {_n(cy)}"
            f"Q{_n(cx + k)} {_n(cy + k)} {_n(cx)} {_n(cy + s)}"
            f"Q{_n(cx - k)} {_n(cy + k)} {_n(cx - s)} {_n(cy)}"
            f"Q{_n(cx - k)} {_n(cy - k)} {_n(cx)} {_n(cy - s)}Z"
        )
        fill = self.edge_grad(cx - s, cy - s, cx + s, cy + s)
        f = f' filter="url(#{self.p}-glow)"' if glow else ""
        self.body.append(f'<path d="{d}" fill="url(#{fill})"{f}/>')
        self.body.append(
            f'<path d="{d}" stroke="{self.t["ink"]}" stroke-opacity="{self.t["ink_opacity"]}" stroke-width="1.2" stroke-linejoin="round"/>'
        )

    def icon(self, cx: float, cy: float, paths: list[str]) -> None:
        """Line art on a 32-unit grid centred on (0, 0), stroked with the edge."""
        g = self.edge_grad(cx - 16, cy - 16, cx + 16, cy + 16)
        inner = "".join(paths)
        self.body.append(
            f'<g transform="translate({_n(cx)},{_n(cy)})" stroke="url(#{g})" stroke-width="2.4" '
            f'stroke-linecap="round" stroke-linejoin="round" fill="none">{inner}</g>'
        )

    def render(self) -> str:
        header = "<!-- Generated by api/scripts/generate_ai_onboarding_svgs.py; edit the generator, not this file. -->"
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" fill="none">\n{header}\n<defs>\n  '
            + "\n  ".join(self.defs)
            + "\n</defs>\n"
            + "\n".join(self.body)
            + "\n</svg>\n"
        )


# ── The four slides ───────────────────────────────────────────────────


def hero(s: Slide) -> None:
    """Meet Xcel: the Xcelsior tile with an AI sparkle, held in two orbits."""
    s.backdrop(((150, 96, 92, 0), (330, 126, 100, 1)))
    s.sparks(((70, 58), (424, 66), (108, 176), (392, 184), (176, 28)))
    cx, cy = 240, 108
    for rx, ry, rot, op in ((156, 44, -10, ".34"), (120, 66, 16, ".22")):
        s.body.append(
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" transform="rotate({rot} {cx} {cy})" '
            f'stroke="url(#{s.p}-edge)" stroke-opacity="{op}" stroke-width="1.5"/>'
        )
    # Nodes riding the orbits, behind and beside the tile.
    for rx, ry, rot, deg, r, color in ((156, 44, -10, 200, 6, 0), (156, 44, -10, 20, 5, 2), (120, 66, 16, 300, 5, 1), (120, 66, 16, 135, 4, 0)):
        a, t = math.radians(deg), math.radians(rot)
        x, y = rx * math.cos(a), ry * math.sin(a)
        s.node(cx + x * math.cos(t) - y * math.sin(t), cy + x * math.sin(t) + y * math.cos(t), r, color)
    s.ground(cx, 74, 186)
    s.tile(cx - 46, cy - 46, 92, 92, 22)
    fill = s.edge_grad(cx - 24, cy - 26, cx + 24, cy + 26)
    scale = 74 / 240
    s.body.append(
        f'<path transform="translate({_n(cx - 120 * scale)},{_n(cy - 120 * scale)}) scale({_n(scale)})" d="{X_MARK}" fill="url(#{fill})"/>'
    )
    s.sparkle(cx + 58, cy - 54, 15)
    s.sparkle(cx + 80, cy - 70, 6, glow=False)
    s.horizon()


ROCKET = [
    '<path d="M0 -15C7 -10 8.5 1 6 9H-6C-8.5 1 -7 -10 0 -15Z"/>',
    '<circle cx="0" cy="-4" r="3"/>',
    '<path d="M-6 3L-11 11L-6 9M6 3L11 11L6 9M-3 13L0 17L3 13"/>',
]
SERVER = [
    '<rect x="-13" y="-13" width="26" height="11" rx="3"/>',
    '<rect x="-13" y="2" width="26" height="11" rx="3"/>',
    '<path d="M-8 -7.5h.01M-8 7.5h.01M2 -7.5h6M2 7.5h6"/>',
]
WALLET = [
    '<rect x="-14" y="-10" width="28" height="21" rx="4"/>',
    '<path d="M-14 -4H8"/>',
    '<path d="M14 0H6a3 3 0 0 0 0 6H14"/>',
    '<path d="M8 3h.01"/>',
]
TERMINAL = [
    '<rect x="-14" y="-11" width="28" height="22" rx="4"/>',
    '<path d="M-8 -3L-3 1L-8 5M0 6H8"/>',
]


def actions(s: Slide) -> None:
    """Xcel works the dashboard: an AI hub wired to instances, hosts, billing and settings."""
    s.backdrop(((140, 110, 90, 0), (340, 110, 96, 1)))
    s.sparks(((62, 30), (420, 34), (240, 18), (40, 196), (440, 194)))
    hub = (240, 106)
    tiles = (((112, 64), ROCKET, 0), ((112, 150), SERVER, 1), ((368, 64), WALLET, 2), ((368, 150), TERMINAL, 0))
    for (tx, ty), _icon, color in tiles:
        s.body.append(
            f'<path d="M{hub[0]} {hub[1]}L{tx} {ty}" stroke="url(#{s.p}-edge)" stroke-opacity="{s.t["line_opacity"]}" stroke-width="1.5"/>'
        )
        s.dot(hub[0] + (tx - hub[0]) * 0.55, hub[1] + (ty - hub[1]) * 0.55, color)
    s.ground(240, 150, 190)
    for (tx, ty), icon, _color in tiles:
        s.tile(tx - 28, ty - 28, 56, 56, 14)
        s.icon(tx, ty, icon)
    s.body.append(
        f'<circle cx="{hub[0]}" cy="{hub[1]}" r="40" stroke="url(#{s.p}-edge)" stroke-opacity=".28" stroke-width="1.5"/>'
    )
    bg = s.body_grad(hub[0] - 28, hub[1] - 28, hub[0] + 28, hub[1] + 28)
    s.body.append(
        f'<circle cx="{hub[0]}" cy="{hub[1]}" r="28" fill="url(#{bg})" stroke="url(#{s.p}-edge)" stroke-width="2.6" filter="url(#{s.p}-glow)"/>'
    )
    s.sparkle(hub[0], hub[1], 15, glow=False)
    s.horizon()


def api(s: Slide) -> None:
    """Full API reach, with a guard: a console whose destructive call waits on a confirm shield."""
    s.backdrop(((170, 100, 96, 0), (330, 140, 96, 1)))
    s.sparks(((70, 46), (420, 40), (96, 186), (420, 182)))
    x, y, w, h = 138, 40, 196, 140
    s.ground(240, 120, 196)
    s.tile(x, y, w, h, 14)
    e = s.t["edge"]
    for i, cx in enumerate((154, 166, 178)):
        s.body.append(f'<circle cx="{cx}" cy="{y + 16}" r="3.2" fill="{e[i]}"/>')
    s.body.append(
        f'<path d="M{x + 2} {y + 30}H{x + w - 2}" stroke="url(#{s.p}-edge)" stroke-opacity=".25" stroke-width="1.5"/>'
    )
    # Request lines: a method tag and a path, the first one being sent.
    rows = ((y + 50, 26, 92, 0, ".9"), (y + 70, 20, 70, 1, ".45"), (y + 90, 26, 84, 2, ".45"), (y + 110, 20, 58, 1, ".45"))
    for i, (ry, tag, path_len, color, op) in enumerate(rows):
        if i == 0:
            s.body.append(
                f'<path d="M{x + 14} {ry - 4}l5 4l-5 4" stroke="{e[0]}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        x0 = x + 26
        s.body.append(
            f'<path d="M{x0} {ry}h{tag}" stroke="{e[color]}" stroke-opacity="{op}" stroke-width="5" stroke-linecap="round"/>'
        )
        s.body.append(
            f'<path d="M{x0 + tag + 10} {ry}h{path_len}" stroke="url(#{s.p}-edge)" stroke-opacity="{".7" if i == 0 else ".28"}" '
            'stroke-width="5" stroke-linecap="round"/>'
        )
    s.body.append(f'<rect x="{x + 26 + 26 + 10 + 92 + 8}" y="{y + 45}" width="7" height="10" rx="1.5" fill="{e[0]}" fill-opacity=".8"/>')
    # The confirm shield.
    sx, sy = 334, 150
    shield = (
        f"M{sx} {sy - 30}L{sx + 24} {sy - 21}V{sy - 2}C{sx + 24} {sy + 14} {sx + 13} {sy + 24} {sx} {sy + 30}"
        f"C{sx - 13} {sy + 24} {sx - 24} {sy + 14} {sx - 24} {sy - 2}V{sy - 21}Z"
    )
    fill = s.edge_grad(sx - 24, sy - 30, sx + 24, sy + 30)
    s.body.append(f'<path d="{shield}" fill="url(#{fill})" filter="url(#{s.p}-glow)"/>')
    s.body.append(
        f'<path d="{shield}" stroke="{s.t["ink"]}" stroke-opacity="{s.t["ink_opacity"]}" stroke-width="2" stroke-linejoin="round"/>'
    )
    s.body.append(
        f'<path d="M{sx - 10} {sy + 1}l7 7l13 -15" stroke="{s.t["ink"]}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    s.horizon()


def ready(s: Slide) -> None:
    """All set: a check badge pulsing over a live circuit rail."""
    s.backdrop(((180, 96, 96, 0), (310, 110, 96, 1)))
    s.sparks(((84, 50), (400, 44), (130, 162), (360, 168)))
    cx, cy = 240, 98
    for r, op in ((92, ".08"), (72, ".16"), (54, ".3")):
        s.body.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="url(#{s.p}-edge)" stroke-opacity="{op}" stroke-width="1.5"/>'
        )
    e = s.t["edge"]
    # The rail the badge stands on, after the bolt icon's.
    rail = s.edge_grad(140, 0, 340, 0)
    s.body.append(
        f'<path d="M140 184h52l12 -12h72l12 12h52" stroke="url(#{rail})" stroke-opacity=".7" stroke-width="3" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )
    for i, (px, depth) in enumerate(((204, 12), (240, 18), (276, 12))):
        s.body.append(
            f'<path d="M{px} {176 if i != 1 else 172}v{depth}" stroke="{e[0]}" stroke-opacity=".34" stroke-width="2" stroke-linecap="round"/>'
        )
        s.dot(px, (176 if i != 1 else 172) + depth + 3, i, 2.6)
    s.ground(cx, 70, 160)
    fill = s.edge_grad(cx - 40, cy - 40, cx + 40, cy + 40)
    s.body.append(f'<circle cx="{cx}" cy="{cy}" r="40" fill="url(#{fill})" filter="url(#{s.p}-glow)"/>')
    s.body.append(
        f'<circle cx="{cx}" cy="{cy}" r="40" stroke="{s.t["ink"]}" stroke-opacity="{s.t["ink_opacity"]}" stroke-width="2"/>'
    )
    s.body.append(
        f'<path d="M{cx - 17} {cy + 1}l11 11l23 -25" stroke="{s.t["ink"]}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    s.sparkle(cx + 50, cy - 46, 9, glow=False)
    s.sparkle(cx - 56, cy - 38, 6, glow=False)
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
