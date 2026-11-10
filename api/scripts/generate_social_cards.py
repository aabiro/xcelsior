"""Generate the agent-native social card set from the OG card design.

Usage: python3 make_social.py <og-agent-native.svg> <outdir>
Writes <name>.svg and <name>.html (for Chrome screenshotting) per card.
"""
import pathlib, re, sys

SRC = pathlib.Path(sys.argv[1]).read_text()
OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)

# ---- pieces lifted from the OG card -------------------------------------------------
DEFS = SRC[SRC.index("<defs>"): SRC.index("</defs>") + len("</defs>")]
# the og-compute gradient is in userSpace x 120..560; make it objectBoundingBox so it fits any headline width
DEFS = DEFS.replace(
    '<linearGradient id="og-compute" x1="120" y1="0" x2="560" y2="0" gradientUnits="userSpaceOnUse">',
    '<linearGradient id="og-compute" x1="0" y1="0" x2="1" y2="0">')
net_start = SRC.index('<g transform="translate(980 290)')
net_end = SRC.index('<g transform="translate(96,160)">')
NET_INNER = SRC[net_start:net_end]
NET_INNER = NET_INNER[NET_INNER.index("\n") + 1:]            # drop the outer transform line
NET_INNER = NET_INNER[: NET_INNER.rindex("</g>")]            # and its closing tag
NET_INNER = re.sub(r'<g stroke-linecap="round" opacity="\.5">.*?</g>', '', NET_INNER, flags=re.S)  # no streaks
MARK = 'M56.5,54 L92.5,54 L122.9,102 L168.5,54 L204.5,54 L138,120 L183.6,186 L147.6,186 L117.2,138 L71.6,186 L35.5,186 L102,120 Z'
GROT = "&#39;Space Grotesk&#39;,&#39;Helvetica Neue&#39;,Arial,sans-serif"
MONO = "&#39;Space Mono&#39;,ui-monospace,monospace"


def background(W, H):
    return f"""<rect width="{W}" height="{H}" fill="url(#og-bg)"></rect>
<rect width="{W}" height="{H}" fill="url(#og-grid)"></rect>
<ellipse cx="{W*0.94:.0f}" cy="{H*0.05:.0f}" rx="{W*0.35:.0f}" ry="{H*0.54:.0f}" fill="url(#og-tr)"></ellipse>
<ellipse cx="{W*0.03:.0f}" cy="{H*0.98:.0f}" rx="{W*0.30:.0f}" ry="{H*0.44:.0f}" fill="url(#og-bl)"></ellipse>
"""


def network(cx, cy, s):
    # the network is authored around (980,300) with the hex centre there
    return (f'<circle cx="{cx}" cy="{cy}" r="{260*s:.0f}" fill="url(#og-halo)"></circle>\n'
            f'<g transform="translate({cx} {cy}) scale({s}) translate(-980 -300)">\n{NET_INNER}</g>\n')


def eyebrow(x, baseline, size=20):
    sq = size * 0.65
    return (f'<g transform="translate({x},{baseline})"><rect x="0" y="{-sq*0.92:.1f}" width="{sq:.1f}" height="{sq:.1f}" rx="{sq*0.23:.1f}" fill="#34d399"></rect>'
            f'<text x="{size*1.5:.0f}" y="0" font-family="{MONO}" font-size="{size}" font-weight="700" letter-spacing="{size*0.25:.1f}" fill="#8ea3bd">XCELSIOR</text>'
            f'<text x="{size*8.8:.0f}" y="0" font-family="{MONO}" font-size="{size}" letter-spacing="{size*0.25:.1f}" fill="#5b7086">/ MCP</text></g>\n')


def headline(x, b1, b2, F, one_line=False):
    ls = -F / 40
    if one_line:
        return (f'<text x="{x}" y="{b1}" font-family="{GROT}" font-weight="700" font-size="{F}" letter-spacing="{ls:.1f}" fill="#f4f8ff">Agent-Native '
                f'<tspan fill="url(#og-compute)">Compute.</tspan></text>\n')
    return (f'<text x="{x}" y="{b1}" font-family="{GROT}" font-weight="700" font-size="{F}" letter-spacing="{ls:.1f}" fill="#f4f8ff">Agent-Native</text>\n'
            f'<text x="{x}" y="{b2}" font-family="{GROT}" font-weight="700" font-size="{F}" letter-spacing="{ls:.1f}" fill="url(#og-compute)">Compute.</text>\n')


def subtitle(x, baseline, size=23):
    return f'<text x="{x}" y="{baseline}" font-family="{MONO}" font-size="{size}" letter-spacing="1" fill="#aebfd4">Fast GPUs for your AI agents. Built for MCP + Claude.</text>\n'


def footer(x, baseline, size=26):
    k = size / 26
    return (f'<g transform="translate({x - 14*k:.1f},{baseline - 35*k:.1f}) scale({0.21*k:.3f})"><path d="{MARK}" fill="url(#og-mark)"></path></g>'
            f'<text x="{x + 32*k:.0f}" y="{baseline}" font-family="{GROT}" font-weight="700" font-size="{size}" fill="#e7eef8">xcelsior.ca</text>\n')


def card(W, H, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" fill="none">\n'
            f'{DEFS}\n{background(W, H)}{body}</svg>')


CARDS = {
    # name: (W, H, body)
    "twitter-header-1500x500": (1500, 500,
        network(1270, 235, 0.55) + eyebrow(120, 124, 18) + headline(118, 232, 0, 92, one_line=True)
        + subtitle(122, 286, 22)),
    "linkedin-banner-1584x396": (1584, 396,
        network(1430, 200, 0.42) + eyebrow(520, 112, 17) + headline(518, 212, 0, 80, one_line=True)
        + subtitle(522, 266, 20) + footer(534, 330, 22)),
    "facebook-cover-1640x624": (1640, 624,
        network(1280, 300, 0.75) + eyebrow(236, 186, 19) + headline(234, 292, 400, 104)
        + subtitle(238, 457, 22) + footer(250, 522, 25)),
    "x-post-1600x900": (1600, 900,
        network(1290, 430, 1.1) + eyebrow(128, 240, 22) + headline(126, 400, 540, 140)
        + subtitle(130, 612, 26) + footer(144, 690, 30)),
    "instagram-post-1080": (1080, 1080,
        network(540, 330, 1.0) + eyebrow(98, 656, 19) + headline(96, 768, 880, 112)
        + subtitle(100, 940, 21) + footer(114, 1000, 26)),
    "instagram-portrait-1080x1350": (1080, 1350,
        network(540, 420, 1.15) + eyebrow(98, 818, 19) + headline(96, 936, 1056, 118)
        + subtitle(100, 1120, 21) + footer(114, 1186, 26)),
}

LINK = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&family=Space+Mono:wght@400;700&display=swap">'
for name, (W, H, body) in CARDS.items():
    svg = card(W, H, body)
    (OUT / f"{name}.svg").write_text(svg)
    (OUT / f"{name}.html").write_text(
        f'<!doctype html><meta charset="utf-8">{LINK}<style>html,body{{margin:0;background:#0a0f1a}}svg{{display:block;width:{W}px;height:{H}px}}</style>{svg}')
    print(name, W, H)
