#!/usr/bin/env python3
"""Generate the marketing illustrations drawn like the bolt icon.

The reference is public/brand-system/assets/icons/bolt.{dark,light}.svg:

  * one solid glyph cut from straight edges, leaning to the right
  * filled cyan -> violet -> green, with a thin bright edge
  * a smaller copy of the shape laid inside it as a pale facet
  * a tight cyan glow and a wide violet one underneath
  * thin circuitry behind it: a rail with a raised notch, a second line,
    probe ticks ending in ring nodes. The glyph overlaps the rail.

No backgrounds, and every illustration is written for both themes.

  public/mcp/flow-{discover,launch,monitor}.{dark,light}.svg   MCP "How it works"
  public/mcp/cost-guardrails.{dark,light}.svg                  MCP guardrails panel
  public/gpu-fleet/{training,inference,fleet-pulse,pipeline}.{dark,light}.svg

public/ is served `immutable` for a year, so changed art needs a new file name.

    python api/scripts/generate_site_illustrations.py
"""

import math
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / "frontend" / "public"

Pts = list[tuple[float, float]]

THEMES = {
    "dark": {
        "a": "#38dbff", "b": "#9b6dff", "c": "#34d399", "warm": "#ffbf5e", "hot": "#ff8f8f",
        "mid": ".58",
        "rail": ("#38dbff", ".05", "1", "#9b6dff", ".08"),
        "rail2": ("#38dbff", ".24"),
        "tick": ".34", "dot": (".5", ".55", ".5"),
        "outline": ("#ecfeff", ".68"),
        "facet": ("#ecfeff", ".14"),
        "ink": "#ecfeff", "ink_edge": "#38dbff",
        "glow": (
            '<feDropShadow dx="0" dy="0" stdDeviation="5" flood-color="#38dbff" flood-opacity=".72"/>'
            '<feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#9b6dff" flood-opacity=".38"/>'
        ),
        "soft": (
            '<feDropShadow dx="0" dy="0" stdDeviation="3" flood-color="#38dbff" flood-opacity=".6"/>'
            '<feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#9b6dff" flood-opacity=".3"/>'
        ),
    },
    "light": {
        "a": "#0891b2", "b": "#7c3aed", "c": "#059669", "warm": "#d97706", "hot": "#e11d48",
        "mid": ".56",
        "rail": ("#0891b2", ".08", ".7", "#7c3aed", ".1"),
        "rail2": ("#0f172a", ".14"),
        "tick": ".45", "dot": (".65", ".65", ".65"),
        "outline": ("#0f172a", ".42"),
        "facet": ("#ffffff", ".18"),
        "ink": "#ffffff", "ink_edge": "#0f172a",
        "glow": (
            '<feDropShadow dx="0" dy="9" stdDeviation="10" flood-color="#0f172a" flood-opacity=".2"/>'
            '<feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#0891b2" flood-opacity=".32"/>'
        ),
        "soft": (
            '<feDropShadow dx="0" dy="5" stdDeviation="6" flood-color="#0f172a" flood-opacity=".18"/>'
            '<feDropShadow dx="0" dy="0" stdDeviation="2.5" flood-color="#0891b2" flood-opacity=".3"/>'
        ),
    },
}

# How far the top of a glyph leans right per unit of height, as the bolt does.
LEAN = 0.16


def n(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def d_of(pts: Pts) -> str:
    return "M" + "L".join(f"{n(x)} {n(y)}" for x, y in pts) + "Z"


def place(pts: Pts, cx: float, cy: float, scale: float = 1.0, rot: float = 0.0, lean: float = 0.0) -> Pts:
    """Scale, rotate (degrees, clockwise) and lean a shape drawn around 0,0, then move it."""
    r = math.radians(rot)
    out = []
    for x, y in pts:
        x, y = x * scale, y * scale
        x, y = x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)
        out.append((cx + x - y * lean, cy + y))
    return out


def inset(pts: Pts, k: float, dx: float = 0.0, dy: float = 0.0) -> Pts:
    """The same shape, smaller, about its centre: the facet."""
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    return [(cx + (x - cx) * k + dx, cy + (y - cy) * k + dy) for x, y in pts]


def ribbon(line: Pts, width: float) -> Pts:
    """A polyline thickened into one mitred polygon (the pulse, the check)."""
    h = width / 2
    left, right = [], []
    for i, (x, y) in enumerate(line):
        dirs = []
        if i > 0:
            dirs.append((x - line[i - 1][0], y - line[i - 1][1]))
        if i < len(line) - 1:
            dirs.append((line[i + 1][0] - x, line[i + 1][1] - y))
        normals = []
        for dx, dy in dirs:
            length = math.hypot(dx, dy)
            normals.append((-dy / length, dx / length))
        mx = sum(a for a, _ in normals) / len(normals)
        my = sum(b for _, b in normals) / len(normals)
        ml = math.hypot(mx, my)
        mx, my = mx / ml, my / ml
        # Mitre length, capped so a sharp turn does not spike.
        k = min(h / max(mx * normals[0][0] + my * normals[0][1], 0.35), h * 2.6)
        left.append((x + mx * k, y + my * k))
        right.append((x - mx * k, y - my * k))
    return left + right[::-1]


def regular(sides: int, r: float, turn: float = 0.0) -> Pts:
    return [
        (r * math.cos(math.radians(turn + 360 * i / sides)), r * math.sin(math.radians(turn + 360 * i / sides)))
        for i in range(sides)
    ]


def die(s: float) -> Pts:
    """A square with its corners cut: a chip seen from above."""
    h, c = s / 2, s * 0.2
    return [(-h + c, -h), (h - c, -h), (h, -h + c), (h, h - c), (h - c, h), (-h + c, h), (-h, h - c), (-h, -h + c)]


def slab(w: float, h: float) -> Pts:
    c = h * 0.3
    return [(-w / 2 + c, -h / 2), (w / 2, -h / 2), (w / 2, h / 2 - c), (w / 2 - c, h / 2), (-w / 2, h / 2), (-w / 2, -h / 2 + c)]


CHEVRON: Pts = [(-12, -30), (4, -30), (30, 0), (4, 30), (-12, 30), (14, 0)]
BOLT: Pts = [(7, -56), (-37, 6), (-6, 6), (-14, 56), (37, -13), (6, -13)]
ROCKET: Pts = [
    (0, -88), (20, -52), (24, 26), (50, 64), (50, 80), (20, 62),
    (12, 78), (-12, 78), (-20, 62), (-50, 80), (-50, 64), (-24, 26), (-20, -52),
]
SHIELD: Pts = [(0, -124), (92, -90), (92, 6), (68, 70), (0, 124), (-68, 70), (-92, 6), (-92, -90)]


def solid(pts: Pts, p: str, t: dict, glow: str = "glow", width: float = 2.4, fill: str = "fill", facet: float = 0.6) -> str:
    """The bolt recipe: glowing gradient fill, bright edge, pale inner facet."""
    d = d_of(pts)
    oc, oo = t["outline"]
    fc, fo = t["facet"]
    out = (
        f'  <path d="{d}" fill="url(#{p}-{fill})" filter="url(#{p}-{glow})"/>\n'
        f'  <path d="{d}" stroke="{oc}" stroke-opacity="{oo}" stroke-width="{n(width)}" stroke-linejoin="round"/>\n'
    )
    if facet:
        out += f'  <path d="{d_of(inset(pts, facet))}" fill="{fc}" fill-opacity="{fo}"/>\n'
    return out


def ink(pts: Pts, p: str, t: dict, opacity: str = ".92") -> str:
    """A detail cut into a glyph in the edge colour (a check, a die's core)."""
    return f'  <path d="{d_of(pts)}" fill="{t["ink"]}" fill-opacity="{opacity}" filter="url(#{p}-soft)"/>\n'


def circuit(p: str, t: dict, x0: float, x1: float, y: float, notch: tuple[float, float], ticks: list[float]) -> str:
    """The bolt's ground: rail with a raised notch, second line, probe ticks."""
    c2, o2 = t["rail2"]
    a, b = notch
    rail = f"M{n(x0)} {n(y)}H{n(a - 10)}l10-10H{n(b)}l9 10H{n(x1)}"
    out = (
        f'  <path d="{rail}" stroke="url(#{p}-rail)" stroke-width="3" stroke-linecap="round"/>\n'
        f'  <path d="M{n(x0 + 9)} {n(y + 10)}H{n(x1 - 9)}" stroke="{c2}" stroke-opacity="{o2}" stroke-width="2" stroke-linecap="round"/>\n'
        f'  <g stroke="{t["a"]}" stroke-opacity="{t["tick"]}" stroke-width="2" stroke-linecap="round">\n'
        "    <path d=\""
        + "".join(f"M{n(x)} {n(y + 3)}v{17 if i % 2 else 12}" for i, x in enumerate(ticks))
        + '"/>\n'
    )
    colours = (t["a"], t["b"], t["c"])
    for i, x in enumerate(ticks):
        out += (
            f'    <circle cx="{n(x)}" cy="{n(y + (24 if i % 2 else 19))}" r="2.5" '
            f'fill="{colours[i % 3]}" fill-opacity="{t["dot"][i % 3]}"/>\n'
        )
    return out + "  </g>\n"


def traces(t: dict, paths: list[tuple[str, tuple[float, float], int]]) -> str:
    """Extra thin board traces, each ending in a ring node."""
    colours = (t["a"], t["b"], t["c"])
    out = f'  <g stroke="{t["a"]}" stroke-opacity="{t["tick"]}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">\n'
    for d, (nx, ny), ci in paths:
        out += (
            f'    <path d="{d}"/><circle cx="{n(nx)}" cy="{n(ny)}" r="2.5" '
            f'fill="{colours[ci % 3]}" fill-opacity="{t["dot"][ci % 3]}"/>\n'
        )
    return out + "  </g>\n"


def floor(p: str, cx: float, y: float, rx: float) -> str:
    """Light pooled on the board under a glyph."""
    return f'  <ellipse cx="{n(cx)}" cy="{n(y)}" rx="{n(rx)}" ry="{n(rx * 0.16)}" fill="url(#{p}-floor)"/>\n'


def packets(p: str, t: dict, y: float, xs: list[float]) -> str:
    """Bright packets travelling the rail."""
    colours = (t["a"], t["c"], t["warm"], t["b"])
    return "".join(
        f'  <path d="{d_of(place(slab(11, 5), x, y, lean=LEAN))}" fill="{colours[i % 4]}" filter="url(#{p}-soft)"/>\n'
        for i, x in enumerate(xs)
    )


def chip(p: str, t: dict, cx: float, cy: float, s: float, glow: str = "glow", fill: str = "fill") -> str:
    """A die with thin leads on every side and a bright core."""
    lead = s * 0.2
    leads = ""
    for k in (-0.24, 0.0, 0.24):
        o = k * s
        # Leads follow the lean, like the body.
        leads += (
            f"M{n(cx + o + LEAN * (s / 2 + lead))} {n(cy - s / 2 - lead)}L{n(cx + o + LEAN * s / 2)} {n(cy - s / 2)}"
            f"M{n(cx + o - LEAN * s / 2)} {n(cy + s / 2)}L{n(cx + o - LEAN * (s / 2 + lead))} {n(cy + s / 2 + lead)}"
            f"M{n(cx - s / 2 - lead - LEAN * o)} {n(cy + o)}h{n(lead)}"
            f"M{n(cx + s / 2 - LEAN * o)} {n(cy + o)}h{n(lead)}"
        )
    body = place(die(s), cx, cy, lean=LEAN)
    return (
        f'  <path d="{leads}" stroke="{t["a"]}" stroke-opacity=".7" stroke-width="{n(max(2, s * 0.035))}" stroke-linecap="round"/>\n'
        + solid(body, p, t, glow, 2.4 if s > 70 else 1.8, fill=fill, facet=0.62)
        + ink(place(die(s * 0.26), cx, cy, lean=LEAN), p, t)
    )


# ── MCP: how it works ────────────────────────────────────────────────

def flow_discover(p: str, t: dict) -> str:
    # A cut-glass lens on a handle: the agent searching what is available.
    lens = place(regular(6, 62, 12), 138, 78)
    ux, uy = math.cos(math.radians(48)), math.sin(math.radians(48))
    s, e, h = (176, 120), (228, 177), 12
    handle = [(s[0] - uy * h, s[1] + ux * h), (s[0] + uy * h, s[1] - ux * h), (e[0] + uy * h, e[1] - ux * h), (e[0] - uy * h, e[1] + ux * h)]
    glass = inset(lens, 0.64)
    fc, _ = t["facet"]
    return (
        circuit(p, t, 26, 294, 150, (98, 214), [70, 160, 250])
        + traces(t, [("M40 60h22l14 14", (40, 60), 0), ("M284 96h-22l-12-12", (284, 96), 2), ("M268 40v18", (268, 40), 1)])
        + floor(p, 150, 152, 96) + packets(p, t, 150, [54, 268])
        + solid(handle, p, t, fill="violet", facet=0)
        + solid(lens, p, t, facet=0)
        + f'  <path d="{d_of(glass)}" fill="{t["ink_edge"]}" fill-opacity=".22"/>\n'
        f'  <path d="{d_of(glass)}" fill="{fc}" fill-opacity=".1" stroke="{t["outline"][0]}" stroke-opacity=".5" stroke-width="1.6" stroke-linejoin="round"/>\n'
        # A glint across the glass, and three results inside it.
        + f'  <path d="{d_of([(108, 60), (128, 46), (138, 52), (114, 70)])}" fill="{fc}" fill-opacity=".5"/>\n'
        + ink(place(slab(34, 7), 142, 74, lean=LEAN), p, t)
        + ink(place(slab(24, 7), 136, 88, lean=LEAN), p, t, ".7")
        + ink(place(slab(28, 7), 137, 102, lean=LEAN), p, t, ".5")
    )


def flow_launch(p: str, t: dict) -> str:
    # A rocket leaving the board: one sentence becomes a running instance.
    body = place(ROCKET, 164, 80, 0.8, rot=26)
    flame = place([(-12, 78), (12, 78), (5, 104), (0, 126), (-6, 100)], 164, 80, 0.8, rot=26)
    fc, fo = t["facet"]
    oc, oo = t["outline"]
    # The lit side of the hull: the bolt's facet, as a bevel.
    bevel = place([(0, -88), (-20, -52), (-24, 26), (-10, 62), (-4, 62), (-4, -70)], 164, 80, 0.8, rot=26)
    port = place([(-9, -34), (9, -34), (9, -16), (-9, -16)], 164, 80, 0.8, rot=26)
    return (
        circuit(p, t, 26, 294, 150, (104, 196), [70, 160, 250])
        + traces(t, [("M52 44h20l12 12", (52, 44), 1), ("M276 122h-18", (276, 122), 2), ("M60 110h16", (60, 110), 0)])
        + floor(p, 150, 152, 84) + packets(p, t, 150, [48, 226, 276])
        + f'  <path d="{d_of(flame)}" fill="url(#{p}-warm)" filter="url(#{p}-soft)"/>\n'
        f'  <path d="{d_of(flame)}" stroke="{oc}" stroke-opacity="{oo}" stroke-width="1.8" stroke-linejoin="round"/>\n'
        + solid(body, p, t, facet=0)
        + f'  <path d="{d_of(bevel)}" fill="{fc}" fill-opacity="{fo}"/>\n'
        + ink(port, p, t)
        # Speed lines, thin like the traces.
        + f'  <g stroke="{t["a"]}" stroke-opacity=".5" stroke-width="2" stroke-linecap="round">'
        f'<path d="M100 92l-20 40M122 118l-11 22M228 34l9-18"/></g>\n'
    )


def flow_monitor(p: str, t: dict) -> str:
    # A heartbeat cut as one solid band: the agent watching the job it started.
    line = [(22, 100), (94, 100), (122, 44), (164, 150), (196, 78), (214, 106), (284, 106)]
    band = ribbon(line, 19)
    fc, fo = t["facet"]
    return (
        circuit(p, t, 26, 294, 150, (116, 204), [70, 160, 250])
        + traces(t, [("M44 52h22l12 12", (44, 52), 0), ("M272 62h-20l-12 12", (272, 62), 2), ("M250 150v-18", (250, 132), 1)])
        + floor(p, 160, 152, 110) + packets(p, t, 150, [52, 232])
        + solid(band, p, t, fill="green", facet=0)
        + f'  <path d="{d_of(ribbon(line, 7))}" fill="{fc}" fill-opacity="{float(fo) + 0.08:.2f}"/>\n'
        # Live: the reading at the end of the line.
        f'  <circle cx="296" cy="106" r="9" fill="{t["c"]}" filter="url(#{p}-soft)"/>\n'
        f'  <circle cx="296" cy="106" r="9" stroke="{t["outline"][0]}" stroke-opacity="{t["outline"][1]}" stroke-width="2"/>\n'
        f'  <circle cx="296" cy="106" r="16" stroke="{t["c"]}" stroke-opacity=".4" stroke-width="2"/>\n'
    )


def mcp_guardrails(p: str, t: dict) -> str:
    # One shield, and the three checks wired into it: the cost estimate
    # (the estimate card), the confirm switch, and the scoped key.
    shield = place(SHIELD, 240, 138, lean=0.0)
    fc, fo = t["facet"]
    bevel = [(240, 14), (148, 48), (148, 144), (172, 208), (240, 262), (240, 232), (190, 192), (170, 140), (170, 64), (240, 38)]
    check = ribbon([(204, 138), (232, 168), (284, 104)], 22)
    out = circuit(p, t, 44, 436, 250, (178, 302), [112, 240, 368])
    out += traces(t, [
        ("M112 122h36", (130, 122), 0), ("M332 122h42", (353, 122), 2),
        ("M70 190v22l14 14h36", (70, 190), 0),
        ("M410 196v14", (410, 210), 2),
    ])
    out += floor(p, 240, 252, 150) + packets(p, t, 250, [84, 150, 396])
    out += solid(shield, p, t, facet=0)
    out += f'  <path d="{d_of(bevel)}" fill="{fc}" fill-opacity="{fo}"/>\n'
    out += f'  <path d="{d_of(inset(shield, 0.66))}" fill="{t["ink_edge"]}" fill-opacity=".18"/>\n'
    out += ink(check, p, t)
    # Estimate: what it will cost, before anything starts.
    out += solid(place(slab(92, 60), 70, 122, lean=LEAN), p, t, "soft", 2, fill="warm", facet=0)
    out += ink(place(slab(32, 7), 58, 106, lean=LEAN), p, t)
    out += ink(place(die(10), 96, 106, lean=LEAN), p, t, ".75")
    out += ink(ribbon([(36, 140), (52, 129), (65, 135), (94, 120)], 5.5), p, t)
    # Switch: nothing destructive without confirm:true.
    out += solid(place(slab(84, 42), 410, 122, lean=LEAN), p, t, "soft", 2, fill="green", facet=0)
    out += ink(place(die(26), 428, 122, lean=LEAN), p, t)
    out += ink(place(slab(20, 6), 388, 122, lean=LEAN), p, t, ".7")
    # Key: the scope the agent was given.
    key: Pts = [(-20, 0), (-6, -14), (6, -4), (52, -4), (52, 12), (44, 12), (44, 5), (36, 5), (36, 14), (27, 14), (27, 5), (6, 5), (-6, 14)]
    out += solid(place(key, 356, 224, 1.5, rot=-8), p, t, "soft", 2, fill="violet", facet=0)
    out += ink(place(regular(4, 7), 346, 226), p, t)
    return out


# ── GPU page ─────────────────────────────────────────────────────────

def gpu_training(p: str, t: dict) -> str:
    # Three dies on one bus: a multi-GPU cluster training together.
    out = circuit(p, t, 30, 450, 132, (188, 296), [100, 240, 380])
    out += traces(t, [("M40 44h22l12 12", (40, 44), 0), ("M440 52h-22l-12 12", (440, 52), 2)])
    # The bus between them: two lines, as on a board.
    out += (
        f'  <g stroke="url(#{p}-rail)" stroke-width="3" stroke-linecap="round">'
        f'<path d="M150 78h38M150 92h38M292 78h38M292 92h38"/></g>\n'
    )
    out += floor(p, 240, 134, 150) + packets(p, t, 132, [62, 168, 318, 420])
    out += chip(p, t, 110, 86, 60, "soft", "cyan")
    out += chip(p, t, 370, 86, 60, "soft", "green")
    out += chip(p, t, 240, 84, 92, fill="violet")
    return out


def gpu_inference(p: str, t: dict) -> str:
    # One die, answers leaving it at speed.
    out = circuit(p, t, 30, 450, 132, (78, 190), [130, 258, 386])
    out += traces(t, [("M452 40h-22l-12 12", (452, 40), 2), ("M36 60h18", (36, 60), 0), ("M36 104h14", (36, 104), 1)])
    out += floor(p, 132, 134, 100) + floor(p, 408, 134, 70) + packets(p, t, 132, [222, 300, 352])
    out += chip(p, t, 132, 84, 92, fill="cyan")
    for x, scale in ((236, 1.0), (286, 0.86), (330, 0.72)):
        out += solid(place(CHEVRON, x, 86, scale), p, t, "soft", 1.8, fill="violet", facet=0)
    out += solid(place(BOLT, 408, 82, 1.12), p, t, fill="warm", facet=0.56)
    return out


def gpu_fleet_health(p: str, t: dict) -> str:
    # A rack with a steady heartbeat over it. Drawn in the top of the frame:
    # the page lays a caption bar over the bottom.
    # The trace stays clear of the rack: its lowest point ends above the top unit.
    line = [(38, 62), (128, 62), (152, 26), (184, 84), (210, 44), (226, 62), (350, 62)]
    fc, fo = t["facet"]
    out = circuit(p, t, 40, 360, 236, (128, 276), [110, 200, 290])
    out += traces(t, [("M52 150h20l12 12", (52, 150), 0), ("M352 176h-20l-12-12", (352, 176), 2)])
    out += floor(p, 200, 240, 150) + packets(p, t, 236, [70, 322])
    for i, y in enumerate((126, 172, 218)):
        unit = place(slab(210, 36), 200, y, lean=LEAN)
        out += solid(unit, p, t, "glow" if i == 1 else "soft", 2.2, fill=("cyan", "violet", "fill")[i], facet=0)
        out += f'  <path d="{d_of(place(slab(196, 10), 201, y - 9, lean=LEAN))}" fill="{fc}" fill-opacity="{fo}"/>\n'
        out += ink(place(slab(58, 6), 152, y + 3, lean=LEAN), p, t, ".85")
        out += ink(place(die(9), 262, y + 2, lean=LEAN), p, t)
        out += ink(place(die(9), 280, y + 2, lean=LEAN), p, t, ".6")
    out += solid(ribbon(line, 15), p, t, fill="green", facet=0)
    out += f'  <path d="{d_of(ribbon(line, 5))}" fill="{fc}" fill-opacity="{float(fo) + 0.08:.2f}"/>\n'
    out += (
        f'  <circle cx="362" cy="62" r="8" fill="{t["c"]}" filter="url(#{p}-soft)"/>\n'
        f'  <circle cx="362" cy="62" r="8" stroke="{t["outline"][0]}" stroke-opacity="{t["outline"][1]}" stroke-width="2"/>\n'
        f'  <circle cx="362" cy="62" r="15" stroke="{t["c"]}" stroke-opacity=".4" stroke-width="2"/>\n'
    )
    return out


def gpu_pipeline(p: str, t: dict) -> str:
    # Pick, provision, pulse: three stations on one line.
    fc, fo = t["facet"]
    out = circuit(p, t, 34, 606, 148, (268, 372), [112, 320, 528])
    out += traces(t, [("M40 46h20l12 12", (40, 46), 0), ("M604 60h-20l-12 12", (604, 60), 2)])
    out += floor(p, 112, 150, 90) + floor(p, 320, 150, 110) + floor(p, 528, 150, 90)
    out += packets(p, t, 148, [200, 236, 420, 456])
    for x in (206, 432):
        out += solid(place(CHEVRON, x, 84, 0.62), p, t, "soft", 1.6, fill="warm", facet=0)
        out += solid(place(CHEVRON, x + 24, 84, 0.62), p, t, "soft", 1.6, fill="warm", facet=0)
    # Pick: a die under a pointer.
    out += chip(p, t, 112, 84, 92, fill="cyan")
    pointer: Pts = [(0, 0), (40, 16), (22, 22), (34, 42), (24, 48), (12, 28), (0, 40)]
    out += (
        f'  <path d="{d_of(place(pointer, 126, 92))}" fill="{t["ink"]}" filter="url(#{p}-soft)"/>\n'
        f'  <path d="{d_of(place(pointer, 126, 92))}" stroke="{t["b"]}" stroke-width="2" stroke-linejoin="round"/>\n'
    )
    # Provision: the rack it lands on.
    for i, y in enumerate((46, 84, 122)):
        out += solid(place(slab(128, 30), 320, y, lean=LEAN), p, t, "glow" if i == 1 else "soft", 2, fill="violet", facet=0)
        out += f'  <path d="{d_of(place(slab(118, 8), 321, y - 8, lean=LEAN))}" fill="{fc}" fill-opacity="{fo}"/>\n'
        out += ink(place(slab(36, 5), 292, y + 3, lean=LEAN), p, t, ".85")
        out += ink(place(die(8), 362, y + 2, lean=LEAN), p, t)
    # Pulse: a live reading.
    gem = place(regular(6, 56, 0), 528, 84, lean=LEAN)
    out += solid(gem, p, t, fill="green", facet=0.64)
    out += ink(ribbon([(494, 86), (512, 86), (521, 62), (535, 108), (545, 78), (551, 88), (564, 88)], 7.5), p, t)
    return out


def svg(prefix: str, theme: str, draw, w: int, h: int) -> str:
    t = THEMES[theme]
    p = f"{prefix}-{theme}"
    r1, r1o, rmo, r2, r2o = t["rail"]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" fill="none">\n'
        "  <defs>\n"
        # Per shape, so every glyph carries the whole cyan -> violet -> green sweep.
        f'    <linearGradient id="{p}-fill" x1=".14" y1=".06" x2=".8" y2=".96">\n'
        f'      <stop stop-color="{t["a"]}"/><stop offset="{t["mid"]}" stop-color="{t["b"]}"/><stop offset="1" stop-color="{t["c"]}"/>\n'
        "    </linearGradient>\n"
        # Single-family fills, so a scene is not one colour throughout.
        + "".join(
            f'    <linearGradient id="{p}-{name}" x1=".14" y1=".06" x2=".8" y2=".96">'
            f'<stop stop-color="{t[c0]}"/><stop offset="1" stop-color="{t[c1]}"/></linearGradient>\n'
            for name, c0, c1 in (("cyan", "a", "b"), ("violet", "b", "hot"), ("green", "c", "a"), ("warm", "warm", "hot"))
        )
        + f'    <radialGradient id="{p}-floor"><stop stop-color="{t["a"]}" stop-opacity=".34"/>'
        f'<stop offset="1" stop-color="{t["b"]}" stop-opacity="0"/></radialGradient>\n'
        f'    <linearGradient id="{p}-rail" x1="{n(w * 0.06)}" y1="0" x2="{n(w * 0.94)}" y2="0" gradientUnits="userSpaceOnUse">\n'
        f'      <stop stop-color="{r1}" stop-opacity="{r1o}"/><stop offset=".5" stop-color="{r1}" stop-opacity="{rmo}"/>'
        f'<stop offset="1" stop-color="{r2}" stop-opacity="{r2o}"/>\n'
        "    </linearGradient>\n"
        f'    <filter id="{p}-glow" x="-40%" y="-40%" width="180%" height="190%" color-interpolation-filters="sRGB">{t["glow"]}</filter>\n'
        f'    <filter id="{p}-soft" x="-50%" y="-50%" width="200%" height="210%" color-interpolation-filters="sRGB">{t["soft"]}</filter>\n'
        "  </defs>\n" + draw(p, t) + "</svg>\n"
    )


ART = (
    ("mcp", "flow-discover", flow_discover, 320, 200),
    ("mcp", "flow-launch", flow_launch, 320, 200),
    ("mcp", "flow-monitor", flow_monitor, 320, 200),
    ("mcp", "cost-guardrails", mcp_guardrails, 480, 290),
    ("gpu-fleet", "training", gpu_training, 480, 172),
    ("gpu-fleet", "inference", gpu_inference, 480, 172),
    ("gpu-fleet", "fleet-pulse", gpu_fleet_health, 400, 270),
    ("gpu-fleet", "pipeline", gpu_pipeline, 640, 190),
)


def main() -> None:
    for folder, name, draw, w, h in ART:
        for theme in THEMES:
            path = PUBLIC / folder / f"{name}.{theme}.svg"
            path.write_text(svg(name, theme, draw, w, h), encoding="utf-8")
            print(f"wrote {path.relative_to(PUBLIC)}")


if __name__ == "__main__":
    main()
