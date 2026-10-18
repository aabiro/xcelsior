#!/usr/bin/env python3
"""
Wizard Pixel Art Sprite Generator — Xcelsior Edition
Generates 129 animation frames for terminal sub-cell rendering.
Canvas: 40x27px, transparent background, no anti-aliasing, flat crisp pixels.

Usage:
  python3 sprites/wizard/generate_wizard_sprites.py
  python3 sprites/wizard/generate_wizard_sprites.py --only peek,type,nod
"""

from PIL import Image
import argparse
import os
import copy

# Output directory — PNGs land next to this script
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)


parser = argparse.ArgumentParser(description="Generate Hexara wizard sprite PNGs")
parser.add_argument(
    "--only",
    help="Comma-separated move groups to generate (e.g. peek,type,nod). Default: all.",
)
args = parser.parse_args()
ONLY = {s.strip() for s in args.only.split(",")} if args.only else None
SKIP_LEGACY = ONLY is not None


def should_generate(group: str) -> bool:
    return ONLY is None or group in ONLY

# Canvas dimensions
W, H = 40, 27

# ========== XCELSIOR BRAND PALETTE ==========
# Gradient: Blue #00d4ff → Purple #7c3aed → Red #dc2626
XC_BLUE   = (62, 135, 246, 255)    # #3E87F6 — deeper blue (mid-gradient from SVG)
XC_PURPLE = (124, 58, 237, 255)   # #7C3AED — brand purple (mid gradient)
XC_RED    = (220, 38, 38, 255)    # #DC2626 — brand red (bottom of gradient)
XC_DARK   = (10, 14, 26, 255)     # #0A0E1A — brand dark navy bg

# Color palette (RGBA) — Xcelsior-branded
T = (0, 0, 0, 0)                  # Transparent

# Hat uses Xcelsior purple
HAT     = XC_PURPLE                # Purple hat — brand purple
HAT_D   = (95, 40, 190, 255)      # Slightly darker purple (brim)
HAT_STAR = XC_BLUE                 # Blue star at hat tip

SKIN    = (255, 218, 185, 255)    # Pale skin
SKIN_S  = (230, 190, 155, 255)    # Skin shadow

# Long silver-white Dumbledore beard
BEARD   = (220, 220, 230, 255)    # Bright silver beard
BEARD_D = (180, 180, 195, 255)    # Darker silver
BEARD_L = (240, 240, 250, 255)    # Lightest beard highlight

# Robes use Xcelsior blue as primary
ROBE    = XC_BLUE                  # Bright blue robes — brand primary
ROBE_D  = (35, 95, 200, 255)       # Darker blue shadow
ROBE_L  = (110, 170, 250, 255)     # Light blue highlight

# Accents use Xcelsior red
BELT    = XC_RED                   # Xcelsior red belt/sash
BELT_D  = (180, 30, 30, 255)      # Darker red belt accent

# Glasses — half-moon style (Dumbledore)
GLASS_FRAME = (180, 170, 140, 255) # Gold/brass frame
GLASS_LENS  = (200, 220, 255, 255) # Slight blue tint lens

WAND     = (139, 69, 19, 255)     # Brown wand
WAND_TIP = XC_RED                  # Xcelsior red wand tip (magic!)

# Particles use all 3 brand colors
STAR    = XC_BLUE                  # Blue star/particle
STAR_W  = (255, 255, 255, 255)    # White sparkle
MAGIC_P = XC_PURPLE               # Purple magic particle
MAGIC_B = XC_BLUE                 # Blue magic particle
MAGIC_K = XC_RED                  # Red magic particle
FLASH   = (255, 235, 100, 255)    # Golden wand flash

# Spell energy colors
SPELL_GOLD   = (255, 215, 0, 255)     # Bright gold energy
SPELL_YELLOW = (255, 255, 60, 255)    # Bright yellow energy

EYE  = (20, 20, 60, 255)          # Dark eye behind glasses
SHOE = XC_DARK                     # Dark navy shoes (brand dark)


def create_canvas():
    """Create a blank transparent canvas."""
    return [[T for _ in range(W)] for _ in range(H)]


def set_pixel(canvas, x, y, color):
    """Safely set a pixel."""
    if 0 <= x < W and 0 <= y < H:
        canvas[y][x] = color


def draw_rect(canvas, x, y, w, h, color):
    """Draw filled rectangle."""
    for dy in range(h):
        for dx in range(w):
            set_pixel(canvas, x + dx, y + dy, color)


def canvas_to_image(canvas):
    """Convert 2D pixel array to PIL Image."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for y in range(H):
        for x in range(W):
            img.putpixel((x, y), canvas[y][x])
    return img


def draw_base_wizard(canvas, offset_y=0, offset_x=0):
    """
    Draw the Xcelsior wizard — Dumbledore-style with half-moon glasses,
    long flowing beard, and Xcelsior brand-colored robes/accents.
    Canvas: 48x32px. offset_y shifts for bob, offset_x for pacing.
    """
    bx = 10 + offset_x
    by = 2 + offset_y

    # === HAT (pointed wizard hat — Xcelsior violet) ===
    set_pixel(canvas, bx + 6, by + 0, HAT_STAR)       # Blue star at hat tip!
    draw_rect(canvas, bx + 5, by + 1, 3, 1, HAT)
    draw_rect(canvas, bx + 4, by + 2, 5, 1, HAT)
    draw_rect(canvas, bx + 3, by + 3, 7, 1, HAT)
    # Hat band — Xcelsior red accent stripe
    draw_rect(canvas, bx + 2, by + 4, 9, 1, XC_RED)
    # Hat brim — dark violet
    draw_rect(canvas, bx + 1, by + 5, 11, 1, HAT_D)

    # === FACE ===
    draw_rect(canvas, bx + 3, by + 6, 7, 1, SKIN)    # forehead
    draw_rect(canvas, bx + 3, by + 7, 7, 1, SKIN)    # eyes row
    draw_rect(canvas, bx + 3, by + 8, 7, 1, SKIN)    # nose area
    draw_rect(canvas, bx + 3, by + 9, 7, 1, SKIN_S)  # lower face / mouth area

    # === HALF-MOON GLASSES (Dumbledore-style) ===
    set_pixel(canvas, bx + 4, by + 7, GLASS_FRAME)    # left frame top
    set_pixel(canvas, bx + 5, by + 7, GLASS_LENS)     # left lens (over eye)
    set_pixel(canvas, bx + 4, by + 8, GLASS_FRAME)    # left frame bottom curve
    set_pixel(canvas, bx + 5, by + 8, GLASS_FRAME)    # left bottom
    set_pixel(canvas, bx + 6, by + 7, GLASS_FRAME)    # bridge
    set_pixel(canvas, bx + 7, by + 7, GLASS_LENS)     # right lens (over eye)
    set_pixel(canvas, bx + 8, by + 7, GLASS_FRAME)    # right frame top
    set_pixel(canvas, bx + 7, by + 8, GLASS_FRAME)    # right bottom
    set_pixel(canvas, bx + 8, by + 8, GLASS_FRAME)    # right frame bottom curve
    set_pixel(canvas, bx + 9, by + 7, GLASS_FRAME)    # right ear piece

    # Eyes (visible through lenses)
    set_pixel(canvas, bx + 5, by + 7, EYE)
    set_pixel(canvas, bx + 7, by + 7, EYE)

    # Nose
    set_pixel(canvas, bx + 7, by + 8, SKIN_S)

    # Mouth (small, on lower face)
    set_pixel(canvas, bx + 5, by + 9, (180, 100, 100, 255))  # left lip
    set_pixel(canvas, bx + 6, by + 9, (180, 100, 100, 255))  # right lip

    # === LONG DUMBLEDORE BEARD (flows over robes, reaches below belt) ===
    # Upper beard — full width under face
    draw_rect(canvas, bx + 3, by + 10, 7, 1, BEARD_L)  # bright silver top
    draw_rect(canvas, bx + 3, by + 11, 7, 1, BEARD)    # full width
    draw_rect(canvas, bx + 4, by + 12, 6, 1, BEARD)    # slightly narrower
    draw_rect(canvas, bx + 4, by + 13, 5, 1, BEARD)    # over belt area
    draw_rect(canvas, bx + 5, by + 14, 4, 1, BEARD_D)  # below belt
    draw_rect(canvas, bx + 5, by + 15, 4, 1, BEARD_D)  # continues
    draw_rect(canvas, bx + 5, by + 16, 3, 1, BEARD_D)  # tapers
    draw_rect(canvas, bx + 6, by + 17, 2, 1, BEARD_D)  # tapers more
    set_pixel(canvas, bx + 6, by + 18, BEARD_D)         # beard tip - very long!

    # === BODY / ROBES (thinner midsection) ===
    # Shoulders (narrow)
    set_pixel(canvas, bx + 2, by + 10, ROBE)           # left shoulder
    set_pixel(canvas, bx + 10, by + 10, ROBE)          # right shoulder

    # Torso (slim flanks beside beard)
    set_pixel(canvas, bx + 2, by + 11, ROBE)           # left
    draw_rect(canvas, bx + 10, by + 11, 2, 1, ROBE)   # right
    draw_rect(canvas, bx + 2, by + 12, 2, 1, ROBE)    # left
    draw_rect(canvas, bx + 10, by + 12, 2, 1, ROBE)   # right

    # Belt/sash — Xcelsior red (no center buckle pixel)
    draw_rect(canvas, bx + 2, by + 13, 2, 1, BELT)    # left belt
    draw_rect(canvas, bx + 9, by + 13, 3, 1, BELT)    # right belt

    # Lower robe (gradually flares out)
    draw_rect(canvas, bx + 2, by + 14, 3, 1, ROBE)
    draw_rect(canvas, bx + 9, by + 14, 3, 1, ROBE)
    draw_rect(canvas, bx + 2, by + 15, 3, 1, ROBE)
    draw_rect(canvas, bx + 9, by + 15, 3, 1, ROBE)
    draw_rect(canvas, bx + 2, by + 16, 10, 1, ROBE_D)
    draw_rect(canvas, bx + 2, by + 17, 4, 1, ROBE_D)
    draw_rect(canvas, bx + 8, by + 17, 4, 1, ROBE_D)

    # Robe bottom (wider, flared)
    draw_rect(canvas, bx + 1, by + 18, 5, 1, ROBE)
    draw_rect(canvas, bx + 8, by + 18, 5, 1, ROBE)
    draw_rect(canvas, bx + 0, by + 19, 14, 1, ROBE)
    draw_rect(canvas, bx + 0, by + 20, 14, 1, ROBE_D)

    # Robe highlight — blue glow on left edge
    set_pixel(canvas, bx + 2, by + 14, ROBE_L)
    set_pixel(canvas, bx + 2, by + 15, ROBE_L)
    set_pixel(canvas, bx + 2, by + 16, ROBE_L)

    # Red trim at robe bottom hem — Xcelsior accent
    draw_rect(canvas, bx + 0, by + 20, 14, 1, XC_RED)

    # === FEET ===
    draw_rect(canvas, bx + 2, by + 21, 4, 1, SHOE)
    draw_rect(canvas, bx + 8, by + 21, 4, 1, SHOE)

    # === RIGHT ARM + WAND (droopy triangle cuff) ===
    draw_rect(canvas, bx + 11, by + 11, 3, 1, ROBE)   # upper sleeve
    draw_rect(canvas, bx + 14, by + 11, 2, 1, SKIN)   # hand
    draw_rect(canvas, bx + 11, by + 12, 4, 1, ROBE_D) # cuff droops wider
    draw_rect(canvas, bx + 10, by + 13, 6, 1, ROBE_D) # cuff widest — triangle tip

    # Wand with Xcelsior red tip
    draw_rect(canvas, bx + 16, by + 11, 4, 1, WAND)
    set_pixel(canvas, bx + 20, by + 11, WAND_TIP)      # red wand tip!

    # === LEFT ARM (droopy triangle cuff) ===
    draw_rect(canvas, bx + 0, by + 11, 2, 1, ROBE)    # upper sleeve
    set_pixel(canvas, bx - 1, by + 11, SKIN)           # hand
    draw_rect(canvas, bx - 1, by + 12, 3, 1, ROBE_D)  # cuff droops wider
    draw_rect(canvas, bx - 2, by + 13, 4, 1, ROBE_D)  # cuff widest — triangle tip

    return bx, by


def draw_wand_angled(canvas, bx, by, angle="up"):
    """Redraw the wand at an angle. Clear existing wand area first."""
    # Clear existing wand area
    for x in range(bx + 10, bx + 22):
        for y in range(by + 8, by + 16):
            set_pixel(canvas, x, y, T)

    # Droopy triangle cuff (consistent across all angles)
    draw_rect(canvas, bx + 11, by + 12, 4, 1, ROBE_D)  # cuff mid
    draw_rect(canvas, bx + 10, by + 13, 6, 1, ROBE_D)  # cuff widest

    if angle == "up":
        # Arm raised
        draw_rect(canvas, bx + 11, by + 11, 3, 1, ROBE)
        draw_rect(canvas, bx + 14, by + 10, 2, 1, SKIN)
        # Wand angled up
        set_pixel(canvas, bx + 16, by + 10, WAND)
        set_pixel(canvas, bx + 17, by + 9, WAND)
        set_pixel(canvas, bx + 18, by + 9, WAND)
        set_pixel(canvas, bx + 19, by + 8, WAND)
        set_pixel(canvas, bx + 20, by + 8, WAND_TIP)
    elif angle == "mid_up":
        # Arm
        draw_rect(canvas, bx + 11, by + 11, 3, 1, ROBE)
        draw_rect(canvas, bx + 14, by + 11, 2, 1, SKIN)
        # Wand slightly angled up
        set_pixel(canvas, bx + 16, by + 11, WAND)
        set_pixel(canvas, bx + 17, by + 10, WAND)
        set_pixel(canvas, bx + 18, by + 10, WAND)
        set_pixel(canvas, bx + 19, by + 9, WAND)
        set_pixel(canvas, bx + 20, by + 9, WAND_TIP)
    elif angle == "down":
        # Arm lowered
        draw_rect(canvas, bx + 11, by + 11, 3, 1, ROBE)
        draw_rect(canvas, bx + 14, by + 12, 2, 1, SKIN)
        # Wand angled down
        set_pixel(canvas, bx + 16, by + 12, WAND)
        set_pixel(canvas, bx + 17, by + 12, WAND)
        set_pixel(canvas, bx + 18, by + 13, WAND)
        set_pixel(canvas, bx + 19, by + 13, WAND)
        set_pixel(canvas, bx + 20, by + 14, WAND_TIP)


def add_particles(canvas, bx, by, tip_x, tip_y, pattern=0):
    """Add sparkle particles using all 3 Xcelsior brand colors."""
    offsets_by_pattern = {
        0: [(2, -1, XC_BLUE), (3, 0, STAR_W), (1, -2, XC_PURPLE), (4, 1, XC_RED)],
        1: [(3, -2, STAR_W), (2, 1, XC_BLUE), (4, -1, XC_PURPLE), (1, 0, XC_RED), (5, -1, XC_BLUE)],
        2: [(1, -1, XC_RED), (3, 1, STAR_W), (4, -2, XC_BLUE), (2, 0, XC_PURPLE), (5, 0, STAR_W), (3, -3, FLASH)],
        3: [(2, -2, FLASH), (4, 0, XC_BLUE), (1, 1, XC_PURPLE), (5, -1, STAR_W), (3, -1, XC_RED), (6, 0, FLASH), (4, -3, XC_BLUE)],
    }
    for dx, dy, color in offsets_by_pattern.get(pattern, []):
        set_pixel(canvas, tip_x + dx, tip_y + dy, color)


def add_wand_flash(canvas, tip_x, tip_y):
    """Add a bright flash at the wand tip."""
    set_pixel(canvas, tip_x, tip_y, FLASH)
    set_pixel(canvas, tip_x + 1, tip_y, FLASH)
    set_pixel(canvas, tip_x, tip_y - 1, FLASH)
    set_pixel(canvas, tip_x + 1, tip_y - 1, STAR_W)
    set_pixel(canvas, tip_x - 1, tip_y, STAR_W)
    set_pixel(canvas, tip_x, tip_y + 1, STAR_W)


def draw_wand_glow(canvas, tip_x, tip_y, intensity=1):
    """Add a growing glow around the wand tip. intensity 1-4."""
    # Glow colors escalate from dim gold to bright yellow
    GLOW_DIM  = (255, 220, 60, 120)
    GLOW_MED  = (255, 235, 80, 180)
    GLOW_BRI  = (255, 245, 120, 220)
    GLOW_HOT  = (255, 255, 180, 255)

    if intensity >= 1:
        set_pixel(canvas, tip_x + 1, tip_y, GLOW_DIM)
    if intensity >= 2:
        set_pixel(canvas, tip_x, tip_y - 1, GLOW_MED)
        set_pixel(canvas, tip_x, tip_y + 1, GLOW_DIM)
    if intensity >= 3:
        set_pixel(canvas, tip_x + 1, tip_y - 1, GLOW_BRI)
        set_pixel(canvas, tip_x + 1, tip_y + 1, GLOW_MED)
        set_pixel(canvas, tip_x + 2, tip_y, GLOW_BRI)
    if intensity >= 4:
        set_pixel(canvas, tip_x + 2, tip_y - 1, GLOW_HOT)
        set_pixel(canvas, tip_x + 2, tip_y + 1, GLOW_MED)
        set_pixel(canvas, tip_x, tip_y - 2, GLOW_DIM)
        set_pixel(canvas, tip_x + 3, tip_y, GLOW_DIM)


def draw_smoke_cloud(canvas, cx, cy, radius, density=1.0):
    """Draw a smoke cloud centered at (cx, cy). density 0.0-1.0 controls coverage."""
    # Pre-baked smoke pixel offsets in concentric rings
    SMOKE_W = (230, 230, 240, 200)   # bright white smoke
    SMOKE_L = (200, 200, 210, 170)   # light gray smoke
    SMOKE_M = (160, 160, 170, 140)   # medium gray smoke
    SMOKE_D = (120, 120, 130, 100)   # dark edge smoke

    # Inner ring (always if density > 0)
    if density > 0.1:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if abs(dx) + abs(dy) <= radius:
                    set_pixel(canvas, cx + dx, cy + dy, SMOKE_W)

    # Middle ring
    if density > 0.3 and radius >= 2:
        ring2 = [(-2,0),(2,0),(0,-2),(0,2),(-1,-1),(1,-1),(-1,1),(1,1),
                 (-2,-1),(2,-1),(-2,1),(2,1)]
        for dx, dy in ring2:
            set_pixel(canvas, cx + dx, cy + dy, SMOKE_L)

    # Outer ring
    if density > 0.5 and radius >= 3:
        ring3 = [(-3,0),(3,0),(0,-3),(0,3),(-2,-2),(2,-2),(-2,2),(2,2),
                 (-3,-1),(3,-1),(-3,1),(3,1),(-1,-3),(1,-3),(-1,3),(1,3)]
        for dx, dy in ring3:
            set_pixel(canvas, cx + dx, cy + dy, SMOKE_M)

    # Wispy edges
    if density > 0.7 and radius >= 4:
        ring4 = [(-4,0),(4,0),(0,-4),(0,4),(-3,-2),(3,-2),(-3,2),(3,2),
                 (-2,-3),(2,-3),(-2,3),(2,3)]
        for dx, dy in ring4:
            set_pixel(canvas, cx + dx, cy + dy, SMOKE_D)


def draw_color_dots(canvas, dots):
    """Draw a list of colored dots. Each entry: (x, y, color)."""
    for x, y, color in dots:
        set_pixel(canvas, x, y, color)


# Pre-defined dot patterns for intro/outro (deterministic, brand-colored)
# Centered around wizard position (bx=10, by=2), wizard center ~(17, 12)

# Initial concentrated flash near center
INTRO_DOTS_FLASH = [
    (15, 10, STAR_W), (17, 9, FLASH), (19, 11, STAR_W), (16, 13, FLASH),
    (18, 8, XC_BLUE), (20, 12, XC_PURPLE), (14, 11, XC_RED), (21, 10, STAR_W),
]

INTRO_DOTS_FULL = [
    # Wide burst of all brand colors
    (3, 4, XC_BLUE), (5, 1, XC_PURPLE), (8, 0, STAR_W), (28, 2, XC_RED),
    (30, 5, XC_BLUE), (1, 10, XC_PURPLE), (32, 8, STAR_W), (35, 3, XC_RED),
    (25, 0, XC_BLUE), (0, 6, STAR_W), (33, 12, XC_PURPLE), (6, 15, XC_RED),
    (27, 14, STAR_W), (2, 2, XC_BLUE), (34, 0, XC_PURPLE), (38, 6, XC_RED),
    (22, 1, STAR_W), (12, 0, XC_BLUE), (39, 10, XC_PURPLE),
]

# Between FULL and MED — dots starting to spread outward
INTRO_DOTS_HEAVY = [
    (4, 6, XC_BLUE), (29, 4, XC_PURPLE), (2, 12, STAR_W), (33, 9, XC_RED),
    (36, 5, XC_BLUE), (1, 8, XC_PURPLE), (31, 13, STAR_W), (6, 16, XC_RED),
    (26, 2, STAR_W), (34, 14, XC_BLUE), (8, 3, XC_PURPLE), (38, 8, XC_RED),
    (23, 1, XC_BLUE), (0, 14, STAR_W),
]

INTRO_DOTS_MED = [
    (4, 8, XC_BLUE), (28, 6, XC_PURPLE), (2, 14, STAR_W), (32, 10, XC_RED),
    (35, 7, XC_BLUE), (0, 12, XC_PURPLE), (30, 15, STAR_W), (7, 18, XC_RED),
    (25, 4, STAR_W), (33, 16, XC_BLUE),
]

# Between MED and FEW — scattered remnants
INTRO_DOTS_LIGHT = [
    (5, 12, XC_BLUE), (30, 10, XC_PURPLE), (3, 18, STAR_W), (33, 14, XC_RED),
    (28, 8, STAR_W), (1, 16, XC_BLUE), (35, 12, XC_PURPLE),
]

INTRO_DOTS_FEW = [
    (5, 16, XC_BLUE), (28, 14, XC_PURPLE), (1, 20, STAR_W), (33, 18, XC_RED),
    (30, 20, STAR_W),
]

# Very sparse — just 2-3 sparkles drifting down
INTRO_DOTS_SPARSE = [
    (8, 22, XC_BLUE), (26, 20, STAR_W), (18, 24, XC_PURPLE),
]

OUTRO_DOTS_BURST = [
    (4, 6, XC_BLUE), (6, 2, XC_PURPLE), (9, 1, STAR_W), (27, 3, XC_RED),
    (30, 6, XC_BLUE), (2, 11, XC_PURPLE), (33, 9, STAR_W), (36, 4, XC_RED),
    (24, 1, XC_BLUE), (1, 7, STAR_W), (34, 13, XC_PURPLE), (7, 16, XC_RED),
    (28, 15, STAR_W), (3, 3, XC_BLUE), (35, 1, XC_PURPLE), (37, 7, XC_RED),
    (21, 2, STAR_W), (13, 1, XC_BLUE), (39, 11, XC_PURPLE),
]

# Heavier outro dots — used during peak smoke phase
OUTRO_DOTS_HEAVY = [
    (5, 4, XC_BLUE), (7, 1, XC_PURPLE), (10, 0, STAR_W), (29, 2, XC_RED),
    (31, 5, XC_BLUE), (3, 9, XC_PURPLE), (34, 8, STAR_W), (37, 3, XC_RED),
    (25, 0, XC_BLUE), (2, 6, STAR_W), (35, 12, XC_PURPLE), (8, 14, XC_RED),
    (26, 13, STAR_W), (4, 2, XC_BLUE), (36, 0, XC_PURPLE), (39, 5, XC_RED),
]

# Lighter outro dots — smoke thinning stage
OUTRO_DOTS_THIN = [
    (6, 10, XC_BLUE), (30, 8, XC_PURPLE), (2, 16, STAR_W), (34, 12, XC_RED),
    (28, 6, STAR_W), (1, 14, XC_BLUE), (36, 10, XC_PURPLE),
]

# Falling dot positions: same dots but shifted down progressively
def make_falling_dots(base_dots, fall):
    """Shift dots downward by 'fall' pixels, clamped to canvas."""
    return [(x, min(y + fall, H - 1), c) for x, y, c in base_dots]

# Ground-level dots (sitting at bottom rows, fading out)
GROUND_DOTS = [
    (5, 24, XC_BLUE), (10, 25, XC_PURPLE), (15, 24, STAR_W), (20, 25, XC_RED),
    (25, 24, XC_BLUE), (30, 25, XC_PURPLE), (8, 25, STAR_W), (22, 24, XC_RED),
    (13, 25, XC_BLUE), (28, 25, STAR_W),
]

GROUND_DOTS_FADE = [
    (5, 25, (62, 135, 246, 120)), (15, 25, (255, 255, 255, 100)),
    (25, 25, (62, 135, 246, 100)), (10, 26, (124, 58, 237, 80)),
    (20, 26, (220, 38, 38, 80)),
]


def save(canvas, name):
    """Save canvas as PNG."""
    img = canvas_to_image(canvas)
    path = os.path.join(OUT_DIR, name)
    img.save(path)
    print(f"  Saved: {path}")


# ============================================================
# NEW HELPER FUNCTIONS
# ============================================================

def draw_zzz(canvas, x, y, size=1, color=XC_BLUE):
    """Draw a 'Z' character at (x,y). size 1=3x3, size 2=4x4."""
    if size == 1:
        for dx in range(3):
            set_pixel(canvas, x + dx, y, color)       # top bar
        set_pixel(canvas, x + 1, y + 1, color)        # diagonal
        for dx in range(3):
            set_pixel(canvas, x + dx, y + 2, color)   # bottom bar
    elif size == 2:
        for dx in range(4):
            set_pixel(canvas, x + dx, y, color)
        set_pixel(canvas, x + 2, y + 1, color)
        set_pixel(canvas, x + 1, y + 2, color)
        for dx in range(4):
            set_pixel(canvas, x + dx, y + 3, color)


def draw_exclamation(canvas, x, y, color=STAR_W):
    """Draw a '!' at (x, y) — shaft at y/y+1, dot at y+3."""
    set_pixel(canvas, x, y, color)
    set_pixel(canvas, x, y + 1, color)
    # gap at y+2
    set_pixel(canvas, x, y + 3, color)


def draw_red_x(canvas, cx, cy):
    """Draw a red X centered at (cx, cy) — 3x3 diagonal cross."""
    set_pixel(canvas, cx - 1, cy - 1, XC_RED)
    set_pixel(canvas, cx + 1, cy - 1, XC_RED)
    set_pixel(canvas, cx, cy, XC_RED)
    set_pixel(canvas, cx - 1, cy + 1, XC_RED)
    set_pixel(canvas, cx + 1, cy + 1, XC_RED)


def draw_levitate_sparkles(canvas, ground_y, bx, intensity=1):
    """Draw sparkles at ground level during levitation."""
    colors = [XC_BLUE, XC_PURPLE, STAR_W, XC_RED]
    if intensity >= 1:
        set_pixel(canvas, bx + 4, ground_y, colors[0])
        set_pixel(canvas, bx + 9, ground_y, colors[2])
    if intensity >= 2:
        set_pixel(canvas, bx + 2, ground_y, colors[1])
        set_pixel(canvas, bx + 6, ground_y + 1, colors[3])
        set_pixel(canvas, bx + 11, ground_y, colors[0])
    if intensity >= 3:
        set_pixel(canvas, bx + 0, ground_y + 1, colors[2])
        set_pixel(canvas, bx + 7, ground_y, colors[1])
        set_pixel(canvas, bx + 13, ground_y + 1, colors[3])
        set_pixel(canvas, bx + 3, ground_y + 1, STAR_W)
        set_pixel(canvas, bx + 10, ground_y + 1, STAR_W)


def close_eyes(canvas, bx, by):
    """Override eye pixels to show closed eyelids."""
    set_pixel(canvas, bx + 5, by + 7, SKIN_S)
    set_pixel(canvas, bx + 7, by + 7, SKIN_S)


def draw_happy_eyes(canvas, bx, by):
    """Closed happy eyes — short curved lid lines."""
    set_pixel(canvas, bx + 4, by + 7, SKIN_S)
    set_pixel(canvas, bx + 5, by + 7, SKIN_S)
    set_pixel(canvas, bx + 7, by + 7, SKIN_S)
    set_pixel(canvas, bx + 8, by + 7, SKIN_S)


def draw_tilt_left_eyes(canvas, bx, by):
    """Head tilt left — right eye slightly larger."""
    set_pixel(canvas, bx + 5, by + 7, EYE)
    set_pixel(canvas, bx + 6, by + 7, EYE)
    set_pixel(canvas, bx + 7, by + 7, EYE)
    set_pixel(canvas, bx + 8, by + 7, EYE)


def draw_squint_peek(canvas, bx, by):
    """Peek past hat brim — squint with one eye larger."""
    set_pixel(canvas, bx + 5, by + 7, SKIN_S)
    set_pixel(canvas, bx + 7, by + 7, EYE)
    set_pixel(canvas, bx + 8, by + 7, EYE)
    set_pixel(canvas, bx + 4, by + 6, HAT_D)


def clear_right_arm_wand(canvas, bx, by):
    """Clear default right arm + wand for custom poses."""
    for x in range(bx + 10, bx + 22):
        for y in range(by + 8, by + 16):
            set_pixel(canvas, x, y, T)


def draw_wand_lowered(canvas, bx, by):
    """Wand held low — curious inspection pose."""
    clear_right_arm_wand(canvas, bx, by)
    draw_rect(canvas, bx + 11, by + 12, 3, 1, ROBE)
    draw_rect(canvas, bx + 14, by + 13, 2, 1, SKIN)
    draw_rect(canvas, bx + 11, by + 14, 4, 1, ROBE_D)
    set_pixel(canvas, bx + 16, by + 14, WAND)
    set_pixel(canvas, bx + 17, by + 15, WAND)
    set_pixel(canvas, bx + 18, by + 15, WAND_TIP)


def draw_stylus_wand(canvas, bx, by):
    """Wand angled forward like a stylus / pointer."""
    clear_right_arm_wand(canvas, bx, by)
    draw_rect(canvas, bx + 11, by + 11, 3, 1, ROBE)
    draw_rect(canvas, bx + 14, by + 12, 2, 1, SKIN)
    draw_rect(canvas, bx + 11, by + 13, 4, 1, ROBE_D)
    set_pixel(canvas, bx + 16, by + 12, WAND)
    set_pixel(canvas, bx + 17, by + 12, WAND)
    set_pixel(canvas, bx + 18, by + 13, WAND)
    set_pixel(canvas, bx + 19, by + 13, WAND_TIP)


def draw_keyboard(canvas, bx, by):
    """Tiny imaginary keyboard in front of the wizard."""
    draw_rect(canvas, bx + 1, by + 20, 12, 1, XC_DARK)
    for x in range(bx + 2, bx + 12, 2):
        set_pixel(canvas, x, by + 20, ROBE_L)


def draw_key_flash(canvas, bx, by, side="left"):
    """Yellow pixel flash on keyboard tap."""
    if side == "left":
        set_pixel(canvas, bx + 3, by + 19, FLASH)
        set_pixel(canvas, bx + 4, by + 19, SPELL_YELLOW)
    else:
        set_pixel(canvas, bx + 9, by + 19, FLASH)
        set_pixel(canvas, bx + 10, by + 19, SPELL_YELLOW)


def draw_code_glyph(canvas, bx, by):
    """Brief xoa_ glyph above hands."""
    glyph_y = by + 8
    set_pixel(canvas, bx + 3, glyph_y, STAR_W)
    set_pixel(canvas, bx + 5, glyph_y, STAR_W)
    set_pixel(canvas, bx + 4, glyph_y + 1, ROBE_L)
    set_pixel(canvas, bx + 7, glyph_y, XC_BLUE)
    set_pixel(canvas, bx + 8, glyph_y, XC_BLUE)
    set_pixel(canvas, bx + 7, glyph_y + 1, XC_BLUE)
    set_pixel(canvas, bx + 10, glyph_y, XC_PURPLE)
    set_pixel(canvas, bx + 11, glyph_y, XC_PURPLE)
    set_pixel(canvas, bx + 10, glyph_y + 1, XC_PURPLE)
    set_pixel(canvas, bx + 13, glyph_y + 1, STAR_W)
    set_pixel(canvas, bx + 14, glyph_y + 1, STAR_W)


def draw_left_hand_type(canvas, bx, by, raised=False):
    """Left hand reaching toward keyboard."""
    for x in range(bx - 3, bx + 1):
        for y in range(by + 10, by + 16):
            set_pixel(canvas, x, y, T)
    y = by + 12 if raised else by + 14
    draw_rect(canvas, bx - 1, y, 2, 1, SKIN)
    draw_rect(canvas, bx - 2, y + 1, 3, 1, ROBE_D)



# ============================================================
# FRAME GENERATION
# ============================================================

def generate_legacy_frames():
    """Generate the original 113-frame sprite set."""

    # --- INTRO FRAMES (wizard appears in smoke + colorful dots) ---
    print("=== Intro Frames (16) ===")

    # Wizard center for smoke: x~17, y~12
    SMOKE_CX, SMOKE_CY = 17, 12

    # intro-1: initial flash — concentrated burst of light at center, no wizard
    c = create_canvas()
    draw_color_dots(c, INTRO_DOTS_FLASH)
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=2, density=0.4)
    save(c, "wizard-intro-1.png")

    # intro-2: flash expands, smoke erupts outward, dots spray wide
    c = create_canvas()
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=3, density=0.7)
    draw_smoke_cloud(c, SMOKE_CX + 2, SMOKE_CY - 1, radius=2, density=0.5)
    draw_color_dots(c, INTRO_DOTS_FLASH)
    draw_color_dots(c, [(3, 6, XC_BLUE), (30, 4, XC_RED), (35, 8, STAR_W)])
    save(c, "wizard-intro-2.png")

    # intro-3: smoke at max density, dots at all edges
    c = create_canvas()
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=4, density=1.0)
    draw_smoke_cloud(c, SMOKE_CX - 3, SMOKE_CY - 2, radius=3, density=0.8)
    draw_smoke_cloud(c, SMOKE_CX + 4, SMOKE_CY + 1, radius=3, density=0.7)
    draw_color_dots(c, INTRO_DOTS_FULL)
    save(c, "wizard-intro-3.png")

    # intro-4: smoke still dense, extra puffs left and right
    c = create_canvas()
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=4, density=1.0)
    draw_smoke_cloud(c, SMOKE_CX - 5, SMOKE_CY, radius=3, density=0.7)
    draw_smoke_cloud(c, SMOKE_CX + 5, SMOKE_CY, radius=3, density=0.7)
    draw_color_dots(c, INTRO_DOTS_FULL)
    save(c, "wizard-intro-4.png")

    # intro-5: first hint of silhouette through smoke, dots still heavy
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=4, density=0.95)
    draw_smoke_cloud(c, SMOKE_CX - 3, SMOKE_CY - 1, radius=3, density=0.8)
    draw_smoke_cloud(c, SMOKE_CX + 4, SMOKE_CY + 1, radius=2, density=0.6)
    draw_color_dots(c, INTRO_DOTS_FULL)
    save(c, "wizard-intro-5.png")

    # intro-6: wizard silhouette emerging, smoke beginning to thin at center
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=4, density=0.85)
    draw_smoke_cloud(c, SMOKE_CX - 2, SMOKE_CY - 1, radius=3, density=0.6)
    draw_color_dots(c, INTRO_DOTS_HEAVY)
    save(c, "wizard-intro-6.png")

    # intro-7: smoke thinning outward from center, wizard more visible
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=3, density=0.7)
    draw_smoke_cloud(c, SMOKE_CX + 4, SMOKE_CY, radius=3, density=0.5)
    draw_color_dots(c, INTRO_DOTS_HEAVY)
    save(c, "wizard-intro-7.png")

    # intro-8: more thinning, wizard mostly visible, outer smoke
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX + 3, SMOKE_CY + 1, radius=3, density=0.5)
    draw_smoke_cloud(c, SMOKE_CX - 4, SMOKE_CY + 2, radius=2, density=0.4)
    draw_color_dots(c, INTRO_DOTS_MED)
    save(c, "wizard-intro-8.png")

    # intro-9: wizard mostly visible, thin wisps and medium dots
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX - 3, SMOKE_CY + 3, radius=2, density=0.4)
    draw_smoke_cloud(c, SMOKE_CX + 6, SMOKE_CY - 1, radius=2, density=0.3)
    draw_color_dots(c, INTRO_DOTS_MED)
    save(c, "wizard-intro-9.png")

    # intro-10: smoke clearing, scattered wisps, dots starting to fall
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX + 7, SMOKE_CY + 3, radius=2, density=0.3)
    draw_color_dots(c, INTRO_DOTS_LIGHT)
    save(c, "wizard-intro-10.png")

    # intro-11: almost clear, just edge wisps, dots falling further
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX - 5, SMOKE_CY + 5, radius=2, density=0.3)
    draw_color_dots(c, make_falling_dots(INTRO_DOTS_FEW, 2))
    save(c, "wizard-intro-11.png")

    # intro-12: thin wisps at edges only, dots near ground
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX + 8, SMOKE_CY + 6, radius=1, density=0.3)
    draw_color_dots(c, make_falling_dots(INTRO_DOTS_FEW, 5))
    save(c, "wizard-intro-12.png")

    # intro-13: last wisp drifting away, 3-4 sparkles settling
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX - 6, SMOKE_CY + 7, radius=1, density=0.2)
    draw_color_dots(c, INTRO_DOTS_SPARSE)
    save(c, "wizard-intro-13.png")

    # intro-14: wizard clear, 2 sparkles settling down
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    draw_color_dots(c, [(8, 24, STAR_W), (26, 22, XC_BLUE)])
    save(c, "wizard-intro-14.png")

    # intro-15: wizard clear, 1 last sparkle fading
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    draw_color_dots(c, [(20, 26, XC_PURPLE)])
    save(c, "wizard-intro-15.png")

    # intro-16: clean wizard, neutral pose (matches idle-1)
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    save(c, "wizard-intro-16.png")

    print()

    # --- IDLE FRAMES (subtle bob) ---
    print("=== Idle Frames (3) ===")

    # idle-1: neutral position
    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    save(c, "wizard-idle-1.png")

    # idle-2: bob down 1px
    c = create_canvas()
    draw_base_wizard(c, offset_y=1)
    save(c, "wizard-idle-2.png")

    # idle-3: bob up (back to neutral + hat star twinkle)
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    set_pixel(c, bx + 8, by + 1, XC_BLUE)
    save(c, "wizard-idle-3.png")

    print()

    # --- PACE FRAMES (reusable pacing shift — 16 frames, ~2s loop) ---
    print("=== Pace Frames (16) ===")

    # Data-driven: (offset_x, offset_y, hat_star_override)
    # Wizard paces right, dips, returns center, paces left, dips, returns.
    # Hat star cycles through brand colors for visual interest.
    PACE_DATA = [
        # rightward
        ( 0,  0, XC_BLUE),    # pace-1:  neutral start
        (+1,  0, XC_BLUE),    # pace-2:  step right
        (+2,  0, XC_PURPLE),  # pace-3:  far right, hat → purple
        (+2, +1, XC_PURPLE),  # pace-4:  dip at far right
        (+2,  0, XC_PURPLE),  # pace-5:  rise back
        (+1,  0, XC_BLUE),    # pace-6:  return step
        ( 0,  0, XC_BLUE),    # pace-7:  center
        ( 0, +1, XC_BLUE),    # pace-8:  settle dip
        # leftward
        ( 0,  0, XC_BLUE),    # pace-9:  center rise
        (-1,  0, XC_BLUE),    # pace-10: step left
        (-2,  0, XC_RED),     # pace-11: far left, hat → red
        (-2, +1, XC_RED),     # pace-12: dip at far left
        (-2,  0, XC_RED),     # pace-13: rise back
        (-1,  0, XC_BLUE),    # pace-14: return step
        ( 0,  0, XC_BLUE),    # pace-15: center
        ( 0, +1, XC_BLUE),    # pace-16: final settle (loops back to pace-1)
    ]

    for i, (ox, oy, hat_color) in enumerate(PACE_DATA, start=1):
        c = create_canvas()
        bx, by = draw_base_wizard(c, offset_y=oy, offset_x=ox)
        set_pixel(c, bx + 6, by + 0, hat_color)  # override hat star color
        save(c, f"wizard-pace-{i}.png")

    print()

    # --- THINK FRAMES (pacing + hat sparkle pulse + wand glow buildup) ---
    print("=== Think Frames (5) ===")

    # think-1: shift right 1px, hat star turns purple, wand begins to glow
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0, offset_x=1)
    set_pixel(c, bx + 6, by + 0, XC_PURPLE)  # hat star → purple
    tip_x, tip_y = bx + 20, by + 11  # wand tip (horizontal default)
    draw_wand_glow(c, tip_x, tip_y, intensity=1)
    save(c, "wizard-think-1.png")

    # think-2: shift right 1px + bob down, hat star turns red, wand glow grows
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1, offset_x=1)
    set_pixel(c, bx + 6, by + 0, XC_RED)  # hat star → red
    tip_x, tip_y = bx + 20, by + 11
    draw_wand_glow(c, tip_x, tip_y, intensity=2)
    save(c, "wizard-think-2.png")

    # think-3: back center, hat star flashes white, wand glow medium
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0, offset_x=0)
    set_pixel(c, bx + 6, by + 0, STAR_W)  # hat star → bright white flash
    set_pixel(c, bx + 5, by + 0, (200, 200, 255, 180))  # tiny flash halo left
    set_pixel(c, bx + 7, by + 0, (200, 200, 255, 180))  # tiny flash halo right
    tip_x, tip_y = bx + 20, by + 11
    draw_wand_glow(c, tip_x, tip_y, intensity=3)
    save(c, "wizard-think-3.png")

    # think-4: shift left 1px, hat star → purple (cycling back), wand glow peaks
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0, offset_x=-1)
    set_pixel(c, bx + 6, by + 0, XC_PURPLE)  # hat star → purple
    tip_x, tip_y = bx + 20, by + 11
    draw_wand_glow(c, tip_x, tip_y, intensity=4)
    save(c, "wizard-think-4.png")

    # think-5: back to center, hat star returns to blue, glow fades — clean base
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0, offset_x=0)
    # hat star already draws as XC_BLUE by default — neutral base pose
    tip_x, tip_y = bx + 20, by + 11
    draw_wand_glow(c, tip_x, tip_y, intensity=1)  # residual dim glow
    save(c, "wizard-think-5.png")

    print()

    # --- WAVE FRAMES (wand angles — smoother 4-step) ---
    print("=== Wave Frames (4) ===")

    # wave-1: wand slightly up
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="mid_up")
    save(c, "wizard-wave-1.png")

    # wave-2: wand fully up
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    save(c, "wizard-wave-2.png")

    # wave-3: wand back to mid-up (returning)
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="mid_up")
    # Add small sparkle trail at mid position for motion feel
    set_pixel(c, bx + 21, by + 8, STAR_W)
    save(c, "wizard-wave-3.png")

    # wave-4: wand sweeps down
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="down")
    save(c, "wizard-wave-4.png")

    print()

    # --- CAST FRAMES (wand flash + particles — 8-step buildup & cool-down) ---
    print("=== Cast Frames (8) ===")

    # Helper: paint hand golden during spell (overrides SKIN at up-angle hand position)
    def spell_hand(canvas, bx, by, color):
        draw_rect(canvas, bx + 14, by + 10, 2, 1, color)

    # cast-1: wand raised, tiny glow at tip
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    tip_x, tip_y = bx + 20, by + 8
    draw_wand_glow(c, tip_x, tip_y, intensity=1)
    set_pixel(c, tip_x, tip_y, FLASH)
    save(c, "wizard-cast-1.png")

    # cast-2: wand up, flash appears + glow building
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    tip_x, tip_y = bx + 20, by + 8
    draw_wand_glow(c, tip_x, tip_y, intensity=2)
    add_wand_flash(c, tip_x, tip_y)
    save(c, "wizard-cast-2.png")

    # cast-3: flash + first particles + golden hand
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    tip_x, tip_y = bx + 20, by + 8
    draw_wand_glow(c, tip_x, tip_y, intensity=2)
    add_wand_flash(c, tip_x, tip_y)
    add_particles(c, bx, by, tip_x, tip_y, pattern=0)
    save(c, "wizard-cast-3.png")

    # cast-4: flash + medium particles + hand glowing brighter
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    tip_x, tip_y = bx + 20, by + 8
    draw_wand_glow(c, tip_x, tip_y, intensity=3)
    add_wand_flash(c, tip_x, tip_y)
    add_particles(c, bx, by, tip_x, tip_y, pattern=1)
    save(c, "wizard-cast-4.png")

    # cast-5: flash + bigger burst + full spell energy
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    tip_x, tip_y = bx + 20, by + 8
    draw_wand_glow(c, tip_x, tip_y, intensity=4)
    add_wand_flash(c, tip_x, tip_y)
    add_particles(c, bx, by, tip_x, tip_y, pattern=2)
    save(c, "wizard-cast-5.png")

    # cast-6: maximum particle explosion + blazing spell hand
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    tip_x, tip_y = bx + 20, by + 8
    draw_wand_glow(c, tip_x, tip_y, intensity=4)
    add_wand_flash(c, tip_x, tip_y)
    add_particles(c, bx, by, tip_x, tip_y, pattern=3)
    save(c, "wizard-cast-6.png")

    # cast-7: cool-down — fading golden sparkles, glow receding
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    tip_x, tip_y = bx + 20, by + 8
    draw_wand_glow(c, tip_x, tip_y, intensity=1)
    set_pixel(c, tip_x + 3, tip_y - 2, (255, 235, 100, 180))
    set_pixel(c, tip_x + 5, tip_y, (255, 220, 60, 150))
    set_pixel(c, tip_x + 2, tip_y + 1, (255, 215, 0, 120))
    save(c, "wizard-cast-7.png")

    # cast-8: wand lowers to mid-up — spell finished, last golden sparkle
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="mid_up")
    set_pixel(c, bx + 22, by + 8, (255, 235, 100, 100))
    save(c, "wizard-cast-8.png")

    print()

    # --- OUTRO FRAMES (grand gesture → disappear in smoke + falling dots) ---
    print("=== Outro Frames (16) ===")

    # outro-1: wizard raises wand high — grand gesture begins
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    save(c, "wizard-outro-1.png")

    # outro-2: wand flash begins, first sparkle at tip
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    tip_x, tip_y = bx + 20, by + 8
    draw_wand_glow(c, tip_x, tip_y, intensity=2)
    save(c, "wizard-outro-2.png")

    # outro-3: big flash + first particles spray out
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    tip_x, tip_y = bx + 20, by + 8
    add_wand_flash(c, tip_x, tip_y)
    add_particles(c, bx, by, tip_x, tip_y, pattern=1)
    save(c, "wizard-outro-3.png")

    # outro-4: flash peak + more particles + first small smoke puff
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    tip_x, tip_y = bx + 20, by + 8
    add_wand_flash(c, tip_x, tip_y)
    add_particles(c, bx, by, tip_x, tip_y, pattern=2)
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY - 3, radius=2, density=0.3)
    save(c, "wizard-outro-4.png")

    # outro-5: smoke spreading from wand, dots spray near wizard
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="up")
    tip_x, tip_y = bx + 20, by + 8
    add_wand_flash(c, tip_x, tip_y)
    add_particles(c, bx, by, tip_x, tip_y, pattern=3)
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY - 2, radius=2, density=0.4)
    draw_smoke_cloud(c, SMOKE_CX + 3, SMOKE_CY, radius=2, density=0.3)
    save(c, "wizard-outro-5.png")

    # outro-6: smoke expanding, wizard starting to fade, dots bright
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=3, density=0.5)
    draw_smoke_cloud(c, SMOKE_CX + 3, SMOKE_CY - 1, radius=2, density=0.4)
    draw_color_dots(c, OUTRO_DOTS_THIN)
    save(c, "wizard-outro-6.png")

    # outro-7: more smoke, wizard half-visible, dots spraying
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=3, density=0.6)
    draw_smoke_cloud(c, SMOKE_CX - 2, SMOKE_CY + 1, radius=2, density=0.5)
    draw_smoke_cloud(c, SMOKE_CX + 4, SMOKE_CY, radius=2, density=0.4)
    draw_color_dots(c, OUTRO_DOTS_BURST)
    save(c, "wizard-outro-7.png")

    # outro-8: heavy smoke, wizard mostly hidden, burst dots
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=4, density=0.8)
    draw_smoke_cloud(c, SMOKE_CX - 2, SMOKE_CY + 1, radius=3, density=0.6)
    draw_color_dots(c, OUTRO_DOTS_BURST)
    save(c, "wizard-outro-8.png")

    # outro-9: dense smoke, only hat star faintly visible, dots heavy
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=4, density=0.9)
    draw_smoke_cloud(c, SMOKE_CX - 3, SMOKE_CY - 1, radius=3, density=0.7)
    draw_smoke_cloud(c, SMOKE_CX + 3, SMOKE_CY + 2, radius=3, density=0.6)
    draw_color_dots(c, OUTRO_DOTS_HEAVY)
    save(c, "wizard-outro-9.png")

    # outro-10: maximum smoke, wizard fully obscured, dots peak
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=4, density=1.0)
    draw_smoke_cloud(c, SMOKE_CX - 3, SMOKE_CY - 1, radius=3, density=0.8)
    draw_smoke_cloud(c, SMOKE_CX + 4, SMOKE_CY + 2, radius=3, density=0.7)
    draw_color_dots(c, OUTRO_DOTS_BURST)
    draw_color_dots(c, OUTRO_DOTS_HEAVY)
    save(c, "wizard-outro-10.png")

    # outro-11: smoke starting to thin, no wizard, dots begin falling
    c = create_canvas()
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY, radius=4, density=0.8)
    draw_smoke_cloud(c, SMOKE_CX - 2, SMOKE_CY + 2, radius=3, density=0.5)
    draw_color_dots(c, make_falling_dots(OUTRO_DOTS_BURST, 3))
    save(c, "wizard-outro-11.png")

    # outro-12: smoke thinning, dots falling mid-air
    c = create_canvas()
    draw_smoke_cloud(c, SMOKE_CX, SMOKE_CY + 2, radius=3, density=0.5)
    draw_smoke_cloud(c, SMOKE_CX + 4, SMOKE_CY + 3, radius=2, density=0.4)
    draw_color_dots(c, make_falling_dots(OUTRO_DOTS_BURST, 6))
    save(c, "wizard-outro-12.png")

    # outro-13: light smoke, dots falling further
    c = create_canvas()
    draw_smoke_cloud(c, SMOKE_CX + 3, SMOKE_CY + 4, radius=2, density=0.3)
    draw_smoke_cloud(c, SMOKE_CX - 3, SMOKE_CY + 5, radius=2, density=0.3)
    draw_color_dots(c, make_falling_dots(OUTRO_DOTS_BURST, 10))
    save(c, "wizard-outro-13.png")

    # outro-14: wisps of smoke, dots near ground
    c = create_canvas()
    draw_smoke_cloud(c, SMOKE_CX + 5, SMOKE_CY + 6, radius=1, density=0.3)
    draw_color_dots(c, make_falling_dots(OUTRO_DOTS_BURST, 14))
    draw_color_dots(c, GROUND_DOTS)
    save(c, "wizard-outro-14.png")

    # outro-15: smoke gone, dots settling on ground, fading
    c = create_canvas()
    draw_color_dots(c, make_falling_dots(OUTRO_DOTS_BURST, 18))
    draw_color_dots(c, GROUND_DOTS)
    save(c, "wizard-outro-15.png")

    # outro-16: last few faded dots at ground, nearly empty — wizard is gone
    c = create_canvas()
    draw_color_dots(c, GROUND_DOTS_FADE)
    save(c, "wizard-outro-16.png")

    print()

    # --- CELEBRATE FRAMES (victory jump + particles — 6 frames) ---
    print("=== Celebrate Frames (6) ===")

    # celebrate-1: wand rising, hat star turns purple — anticipation
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_wand_angled(c, bx, by, angle="mid_up")
    set_pixel(c, bx + 6, by + 0, XC_PURPLE)
    save(c, "wizard-celebrate-1.png")

    # celebrate-2: wizard jumps up 2px, wand raised, hat flashes white
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-2)
    draw_wand_angled(c, bx, by, angle="up")
    set_pixel(c, bx + 6, by + 0, STAR_W)
    tip_x, tip_y = bx + 20, by + 8
    add_wand_flash(c, tip_x, tip_y)
    save(c, "wizard-celebrate-2.png")

    # celebrate-3: peak jump (-3px), full particle burst, ground sparkles
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-3)
    draw_wand_angled(c, bx, by, angle="up")
    set_pixel(c, bx + 6, by + 0, STAR_W)
    set_pixel(c, bx + 5, by + 0, (200, 200, 255, 180))
    set_pixel(c, bx + 7, by + 0, (200, 200, 255, 180))
    tip_x, tip_y = bx + 20, by + 8
    add_wand_flash(c, tip_x, tip_y)
    add_particles(c, bx, by, tip_x, tip_y, pattern=3)
    draw_levitate_sparkles(c, 23, bx, intensity=2)
    save(c, "wizard-celebrate-3.png")

    # celebrate-4: falling back (-1px), particles spreading
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-1)
    draw_wand_angled(c, bx, by, angle="up")
    set_pixel(c, bx + 6, by + 0, XC_PURPLE)
    tip_x, tip_y = bx + 20, by + 8
    add_particles(c, bx, by, tip_x, tip_y, pattern=2)
    draw_levitate_sparkles(c, 23, bx, intensity=1)
    save(c, "wizard-celebrate-4.png")

    # celebrate-5: landing (dip +1), sparkle trail fading
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1)
    draw_wand_angled(c, bx, by, angle="mid_up")
    set_pixel(c, bx + 6, by + 0, XC_BLUE)
    set_pixel(c, bx + 22, by + 9, STAR_W)
    set_pixel(c, bx + 24, by + 8, (255, 255, 255, 150))
    save(c, "wizard-celebrate-5.png")

    # celebrate-6: settle back to neutral, last faint sparkle
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    set_pixel(c, bx + 22, by + 10, (255, 255, 255, 100))
    save(c, "wizard-celebrate-6.png")

    print()

    # --- ERROR FRAMES (frustrated shake + red effects — 6 frames) ---
    print("=== Error Frames (6) ===")

    DIM_RED = (220, 38, 38, 120)

    # error-1: wizard shifts right, wand droops down, red glow starts
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0, offset_x=1)
    draw_wand_angled(c, bx, by, angle="down")
    tip_x, tip_y = bx + 20, by + 14
    set_pixel(c, tip_x, tip_y, XC_RED)
    save(c, "wizard-error-1.png")

    # error-2: wizard shifts left (quick shake), red flash at wand
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0, offset_x=-1)
    draw_wand_angled(c, bx, by, angle="down")
    tip_x, tip_y = bx + 20, by + 14
    set_pixel(c, tip_x, tip_y, XC_RED)
    set_pixel(c, tip_x + 1, tip_y, XC_RED)
    set_pixel(c, tip_x, tip_y - 1, XC_RED)
    save(c, "wizard-error-2.png")

    # error-3: shift right again (shake 2), red X appears above hat
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0, offset_x=1)
    draw_wand_angled(c, bx, by, angle="down")
    draw_red_x(c, bx + 6, by - 1)
    save(c, "wizard-error-3.png")

    # error-4: center, dip down (frustrated sigh), red X drifts up
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1, offset_x=0)
    draw_wand_angled(c, bx, by, angle="down")
    draw_red_x(c, bx + 6, by - 2)
    save(c, "wizard-error-4.png")

    # error-5: still dipped, X fading out
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1, offset_x=0)
    set_pixel(c, bx + 5, by - 3, DIM_RED)
    set_pixel(c, bx + 7, by - 3, DIM_RED)
    set_pixel(c, bx + 6, by - 2, DIM_RED)
    set_pixel(c, bx + 5, by - 1, DIM_RED)
    set_pixel(c, bx + 7, by - 1, DIM_RED)
    save(c, "wizard-error-5.png")

    # error-6: return to neutral, last faint X remnant
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    set_pixel(c, bx + 6, by - 1, (220, 38, 38, 60))
    save(c, "wizard-error-6.png")

    print()

    # --- SLEEP FRAMES (dozing + Zzz — 6 frames) ---
    print("=== Sleep Frames (6) ===")

    # sleep-1: wizard dips slightly, eyes start closing
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1)
    close_eyes(c, bx, by)
    save(c, "wizard-sleep-1.png")

    # sleep-2: dipped, eyes closed, first small Z appears
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1)
    close_eyes(c, bx, by)
    draw_zzz(c, bx + 14, by - 2, size=1, color=XC_BLUE)
    save(c, "wizard-sleep-2.png")

    # sleep-3: small Z rises/fades, new purple Z appears lower
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1)
    close_eyes(c, bx, by)
    draw_zzz(c, bx + 16, by - 5, size=1, color=(62, 135, 246, 150))
    draw_zzz(c, bx + 13, by - 1, size=1, color=XC_PURPLE)
    save(c, "wizard-sleep-3.png")

    # sleep-4: deeper bob, three Z's floating up at different heights
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=2)
    close_eyes(c, bx, by)
    draw_zzz(c, bx + 18, by - 6, size=1, color=(62, 135, 246, 100))
    draw_zzz(c, bx + 15, by - 3, size=1, color=XC_BLUE)
    draw_zzz(c, bx + 12, by + 1, size=1, color=XC_PURPLE)
    save(c, "wizard-sleep-4.png")

    # sleep-5: bob back to 1, fresh Z cycle starting
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1)
    close_eyes(c, bx, by)
    draw_zzz(c, bx + 14, by - 2, size=1, color=XC_BLUE)
    draw_zzz(c, bx + 17, by - 6, size=1, color=(124, 58, 237, 130))
    save(c, "wizard-sleep-5.png")

    # sleep-6: gentle bob down, single Z (good loop point back to sleep-2)
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=2)
    close_eyes(c, bx, by)
    draw_zzz(c, bx + 13, by - 1, size=1, color=XC_BLUE)
    save(c, "wizard-sleep-6.png")

    print()

    # --- EUREKA FRAMES (aha moment + exclamation — 5 frames) ---
    print("=== Eureka Frames (5) ===")

    # eureka-1: neutral stance, hat star turns bright white (something stirring)
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    set_pixel(c, bx + 6, by + 0, STAR_W)
    save(c, "wizard-eureka-1.png")

    # eureka-2: wizard hops up (-1px), hat sparkle halo expanding
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-1)
    set_pixel(c, bx + 6, by + 0, STAR_W)
    set_pixel(c, bx + 5, by + 0, (200, 200, 255, 180))
    set_pixel(c, bx + 7, by + 0, (200, 200, 255, 180))
    set_pixel(c, bx + 6, by - 1, (255, 255, 200, 150))
    save(c, "wizard-eureka-2.png")

    # eureka-3: peak hop (-2px), exclamation beside head, wand flash
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-2)
    draw_wand_angled(c, bx, by, angle="up")
    draw_exclamation(c, bx + 13, by + 1, color=SPELL_GOLD)
    set_pixel(c, bx + 5, by + 0, STAR_W)
    set_pixel(c, bx + 7, by + 0, STAR_W)
    tip_x, tip_y = bx + 20, by + 8
    add_wand_flash(c, tip_x, tip_y)
    save(c, "wizard-eureka-3.png")

    # eureka-4: stars burst around head, maximum brightness (-1px)
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-1)
    draw_wand_angled(c, bx, by, angle="up")
    set_pixel(c, bx + 3, by - 1, XC_BLUE)
    set_pixel(c, bx + 9, by - 1, XC_PURPLE)
    set_pixel(c, bx + 6, by - 2, STAR_W)
    set_pixel(c, bx + 2, by + 2, XC_RED)
    set_pixel(c, bx + 11, by + 1, XC_BLUE)
    draw_exclamation(c, bx + 13, by + 0, color=SPELL_YELLOW)
    tip_x, tip_y = bx + 20, by + 8
    add_particles(c, bx, by, tip_x, tip_y, pattern=1)
    save(c, "wizard-eureka-4.png")

    # eureka-5: settle back to neutral, residual sparkle
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    set_pixel(c, bx + 6, by + 0, XC_BLUE)
    set_pixel(c, bx + 8, by - 1, (255, 255, 255, 120))
    set_pixel(c, bx + 4, by + 1, (255, 255, 255, 80))
    save(c, "wizard-eureka-5.png")

    print()

    # --- LEVITATE FRAMES (floating up and down — 8 frames) ---
    print("=== Levitate Frames (8) ===")

    GROUND_Y = 23  # Normal feet position (by=2, feet at by+21=23)

    # levitate-1: grounded, sparkle starts at feet
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_levitate_sparkles(c, GROUND_Y, bx, intensity=1)
    save(c, "wizard-levitate-1.png")

    # levitate-2: rising (-1px), sparkles grow
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-1)
    draw_levitate_sparkles(c, GROUND_Y, bx, intensity=2)
    save(c, "wizard-levitate-2.png")

    # levitate-3: rising (-2px), full sparkle cloud, hat → purple
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-2)
    draw_levitate_sparkles(c, GROUND_Y, bx, intensity=3)
    set_pixel(c, bx + 6, by + 0, XC_PURPLE)
    save(c, "wizard-levitate-3.png")

    # levitate-4: peak height (-3px), full sparkles, hat → white flash
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-3)
    draw_levitate_sparkles(c, GROUND_Y, bx, intensity=3)
    set_pixel(c, bx + 6, by + 0, STAR_W)
    set_pixel(c, bx + 5, by + 0, (200, 200, 255, 120))
    set_pixel(c, bx + 7, by + 0, (200, 200, 255, 120))
    save(c, "wizard-levitate-4.png")

    # levitate-5: hovering at peak (-3px), sparkle pattern shifts
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-3)
    set_pixel(c, bx + 3, GROUND_Y, XC_PURPLE)
    set_pixel(c, bx + 7, GROUND_Y, STAR_W)
    set_pixel(c, bx + 10, GROUND_Y + 1, XC_BLUE)
    set_pixel(c, bx + 5, GROUND_Y + 1, XC_RED)
    set_pixel(c, bx + 12, GROUND_Y, STAR_W)
    set_pixel(c, bx + 6, by + 0, XC_PURPLE)
    save(c, "wizard-levitate-5.png")

    # levitate-6: descending (-2px), sparkles fading
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-2)
    draw_levitate_sparkles(c, GROUND_Y, bx, intensity=2)
    set_pixel(c, bx + 6, by + 0, XC_BLUE)
    save(c, "wizard-levitate-6.png")

    # levitate-7: descending (-1px), light sparkles
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-1)
    draw_levitate_sparkles(c, GROUND_Y, bx, intensity=1)
    save(c, "wizard-levitate-7.png")

    # levitate-8: landed, last faint ground sparkle
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    set_pixel(c, bx + 5, GROUND_Y + 1, (62, 135, 246, 100))
    set_pixel(c, bx + 9, GROUND_Y + 1, (124, 58, 237, 80))
    save(c, "wizard-levitate-8.png")

    print()

    # --- DANCE FRAMES (groovy wizard shuffle — 8 frames) ---
    print("=== Dance Frames (8) ===")

    # dance-1: shift right, bob down, hat → purple
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1, offset_x=1)
    set_pixel(c, bx + 6, by + 0, XC_PURPLE)
    save(c, "wizard-dance-1.png")

    # dance-2: shift right more, bob up, wand raised, hat → red
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-1, offset_x=2)
    draw_wand_angled(c, bx, by, angle="up")
    set_pixel(c, bx + 6, by + 0, XC_RED)
    save(c, "wizard-dance-2.png")

    # dance-3: shift right, bob down, wand mid, sparkle trail
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1, offset_x=1)
    draw_wand_angled(c, bx, by, angle="mid_up")
    set_pixel(c, bx + 6, by + 0, XC_BLUE)
    set_pixel(c, bx + 22, by + 9, STAR_W)
    save(c, "wizard-dance-3.png")

    # dance-4: center, wand sweeps down, particles spray
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0, offset_x=0)
    draw_wand_angled(c, bx, by, angle="down")
    tip_x, tip_y = bx + 20, by + 14
    add_particles(c, bx, by, tip_x, tip_y, pattern=0)
    set_pixel(c, bx + 6, by + 0, STAR_W)
    save(c, "wizard-dance-4.png")

    # dance-5: shift left, bob down, hat → purple
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1, offset_x=-1)
    set_pixel(c, bx + 6, by + 0, XC_PURPLE)
    save(c, "wizard-dance-5.png")

    # dance-6: shift left more, bob up, wand raised, hat → red
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-1, offset_x=-2)
    draw_wand_angled(c, bx, by, angle="up")
    set_pixel(c, bx + 6, by + 0, XC_RED)
    save(c, "wizard-dance-6.png")

    # dance-7: shift left, bob down, wand mid, sparkle
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1, offset_x=-1)
    draw_wand_angled(c, bx, by, angle="mid_up")
    set_pixel(c, bx + 6, by + 0, XC_BLUE)
    set_pixel(c, bx + 22, by + 9, STAR_W)
    save(c, "wizard-dance-7.png")

    # dance-8: back to center, settle with fading sparkle
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0, offset_x=0)
    set_pixel(c, bx + 6, by + 0, STAR_W)
    set_pixel(c, bx + 8, by - 1, (255, 255, 255, 120))
    set_pixel(c, bx + 4, by + 1, (255, 255, 255, 80))
    save(c, "wizard-dance-8.png")

    print()

    # --- BOW FRAMES (graceful bow — 6 frames) ---
    print("=== Bow Frames (6) ===")

    # bow-1: wizard neutral, hat star turns purple (preparing)
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    set_pixel(c, bx + 6, by + 0, XC_PURPLE)
    save(c, "wizard-bow-1.png")

    # bow-2: slight dip (bob down 1px), beginning bow
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1)
    set_pixel(c, bx + 6, by + 0, XC_PURPLE)
    save(c, "wizard-bow-2.png")

    # bow-3: deep bow (bob down 3px), wand extends forward/down
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=3)
    draw_wand_angled(c, bx, by, angle="down")
    set_pixel(c, bx + 6, by + 0, XC_PURPLE)
    save(c, "wizard-bow-3.png")

    # bow-4: holds deep bow, hat star sparkles bright
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=3)
    draw_wand_angled(c, bx, by, angle="down")
    set_pixel(c, bx + 6, by + 0, STAR_W)
    set_pixel(c, bx + 5, by + 0, (200, 200, 255, 150))
    set_pixel(c, bx + 7, by + 0, (200, 200, 255, 150))
    save(c, "wizard-bow-4.png")

    # bow-5: rising back (bob 1px), star fading
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1)
    set_pixel(c, bx + 6, by + 0, XC_BLUE)
    save(c, "wizard-bow-5.png")

    # bow-6: back to neutral, last faint sparkle
    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    set_pixel(c, bx + 8, by + 1, (255, 255, 255, 80))
    save(c, "wizard-bow-6.png")

    print()


def generate_peek_frames():
    """Generate peek animation frames."""

    # --- PEEK FRAMES (curious inspection — 6 frames) ---
    print("=== Peek Frames (6) ===")

    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    save(c, "wizard-peek-1.png")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0, offset_x=-1)
    draw_tilt_left_eyes(c, bx, by)
    save(c, "wizard-peek-2.png")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-1, offset_x=1)
    set_pixel(c, bx + 4, by + 5, HAT_D)
    set_pixel(c, bx + 8, by + 5, HAT_D)
    draw_wand_lowered(c, bx, by)
    save(c, "wizard-peek-3.png")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-1, offset_x=1)
    set_pixel(c, bx + 3, by + 4, HAT_D)
    set_pixel(c, bx + 9, by + 4, HAT_D)
    draw_squint_peek(c, bx, by)
    draw_wand_lowered(c, bx, by)
    save(c, "wizard-peek-4.png")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    set_pixel(c, bx + 9, by + 6, STAR_W)
    set_pixel(c, bx + 10, by + 7, XC_BLUE)
    save(c, "wizard-peek-5.png")

    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    save(c, "wizard-peek-6.png")

    print()


def generate_type_frames():
    """Generate type animation frames."""

    # --- TYPE FRAMES (writing config — 6 frames) ---
    print("=== Type Frames (6) ===")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_stylus_wand(c, bx, by)
    save(c, "wizard-type-1.png")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_stylus_wand(c, bx, by)
    draw_left_hand_type(c, bx, by)
    draw_keyboard(c, bx, by)
    save(c, "wizard-type-2.png")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_stylus_wand(c, bx, by)
    draw_left_hand_type(c, bx, by, raised=True)
    draw_keyboard(c, bx, by)
    draw_key_flash(c, bx, by, side="left")
    save(c, "wizard-type-3.png")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_stylus_wand(c, bx, by)
    draw_left_hand_type(c, bx, by)
    draw_keyboard(c, bx, by)
    draw_key_flash(c, bx, by, side="right")
    save(c, "wizard-type-4.png")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_stylus_wand(c, bx, by)
    draw_left_hand_type(c, bx, by, raised=True)
    draw_keyboard(c, bx, by)
    draw_key_flash(c, bx, by, side="left")
    draw_key_flash(c, bx, by, side="right")
    draw_code_glyph(c, bx, by)
    save(c, "wizard-type-5.png")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=1)
    draw_keyboard(c, bx, by)
    set_pixel(c, bx + 6, by + 0, XC_BLUE)
    save(c, "wizard-type-6.png")

    print()


def generate_nod_frames():
    """Generate nod animation frames."""

    # --- NOD FRAMES (affirmative ack — 4 frames) ---
    print("=== Nod Frames (4) ===")

    c = create_canvas()
    draw_base_wizard(c, offset_y=0)
    save(c, "wizard-nod-1.png")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=2)
    set_pixel(c, bx + 6, by + 0, XC_PURPLE)
    save(c, "wizard-nod-2.png")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=-1)
    set_pixel(c, bx + 6, by + 0, STAR_W)
    save(c, "wizard-nod-3.png")

    c = create_canvas()
    bx, by = draw_base_wizard(c, offset_y=0)
    draw_happy_eyes(c, bx, by)
    set_pixel(c, bx + 5, by + 9, (180, 100, 100, 255))
    set_pixel(c, bx + 7, by + 9, (180, 100, 100, 255))
    save(c, "wizard-nod-4.png")

    print()


if __name__ == "__main__":
    print("Generating wizard sprite frames...")
    print(f"Output: {OUT_DIR}\n")
    if SKIP_LEGACY:
        print(f"(Skipping legacy frames — generating only: {', '.join(sorted(ONLY))})\n")
    else:
        generate_legacy_frames()
    if should_generate("peek"):
        generate_peek_frames()
    if should_generate("type"):
        generate_type_frames()
    if should_generate("nod"):
        generate_nod_frames()

    generated = [f for f in os.listdir(OUT_DIR) if f.endswith(".png")]
    print(f"\nDone! {len(generated)} PNG frames in {OUT_DIR}")
    print("\nFrame list:")
    for f in sorted(generated):
        fpath = os.path.join(OUT_DIR, f)
        img = Image.open(fpath)
        print(f"  {f} ({img.size[0]}x{img.size[1]})")

