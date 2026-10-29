#!/usr/bin/env python3
"""Apply horizontal spherical distortion + rotation to canada-map-copy.svg.

This transforms all path coordinates and circle positions to give the map
a globe-like horizontal curvature and a slight counterclockwise rotation.
"""
import re, math

SVG_PATH = '/Users/aaryn/Projects/aarynfans-mvp/app/svg/canada-map-copy.svg'

# --- Parameters ---
CENTER_X = 500
CENTER_Y = 450       # vertical center of visible area
HALF_WIDTH = 550      # horizontal half-extent for distortion calc
SPHERE_STRENGTH = 0.06  # additional horizontal globe curvature
ROTATION_DEG = 0      # no extra rotation (already applied)


def transform_point(x, y):
    """Apply horizontal spherical distortion then rotation."""
    # Step 1: Horizontal spherical distortion
    dx = x - CENTER_X
    normalized = max(-1.0, min(1.0, dx / HALF_WIDTH))
    curve = math.cos(normalized * math.pi / 2)  # 1 at center, 0 at edges

    # Horizontal: compress edges slightly
    new_x = CENTER_X + dx * (1.0 - SPHERE_STRENGTH * (1.0 - curve))
    # Vertical: edges curve upward (y decreases at edges)
    new_y = y - SPHERE_STRENGTH * HALF_WIDTH * 0.7 * (1.0 - curve)

    # Step 2: Rotation around center
    rad = math.radians(ROTATION_DEG)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    rx, ry = new_x - CENTER_X, new_y - CENTER_Y
    final_x = CENTER_X + rx * cos_r - ry * sin_r
    final_y = CENTER_Y + rx * sin_r + ry * cos_r

    return round(final_x, 1), round(final_y, 1)


def replace_coord_pair(m):
    """Replace a single x,y coordinate pair."""
    x, y = float(m.group(1)), float(m.group(2))
    nx, ny = transform_point(x, y)
    return f'{nx},{ny}'


coord_re = re.compile(r'(-?\d+\.?\d*),(-?\d+\.?\d*)')


def transform_d_attr(m):
    """Transform all coordinate pairs inside a path d attribute."""
    return f'd="{coord_re.sub(replace_coord_pair, m.group(1))}"'


def transform_circle(m):
    """Transform circle cx and cy attributes."""
    pre, cx_s, mid, cy_s, post = m.groups()
    nx, ny = transform_point(float(cx_s), float(cy_s))
    return f'<circle{pre}cx="{nx}"{mid}cy="{ny}"{post}'


def main():
    with open(SVG_PATH) as f:
        svg = f.read()

    # Split at </defs> — everything before is defs/styles/masks (don't touch)
    marker = '</defs>'
    idx = svg.index(marker) + len(marker)
    defs = svg[:idx]
    content = svg[idx:]

    # Transform path d="" attributes in content
    content = re.sub(r'd="([^"]*)"', transform_d_attr, content)

    # Transform circle cx/cy
    content = re.sub(
        r'<circle([^>]*?)cx="(-?\d+\.?\d*)"([^>]*?)cy="(-?\d+\.?\d*)"([^>]*/?>)',
        transform_circle,
        content
    )

    # Expand viewBox to accommodate additional curvature
    defs = defs.replace(
        'viewBox="-25 150 1050 600" width="1050" height="600"',
        'viewBox="-40 120 1080 650" width="1080" height="650"'
    )

    with open(SVG_PATH, 'w') as f:
        f.write(defs + content)

    print('Done! Applied horizontal globe distortion + 5° CCW rotation.')


if __name__ == '__main__':
    main()
