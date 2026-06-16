#!/usr/bin/env python3
"""Convert Canada GeoJSON to simplified SVG paths for province outlines."""
import json
import urllib.request
import math
import sys

# Fetch the GeoJSON data
url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/canada.geojson"
print("Fetching GeoJSON data...")
with urllib.request.urlopen(url) as response:
    data = json.loads(response.read().decode())

# Fetch Alaska from Natural Earth states/provinces (50m for better detail)
alaska_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson"
print("Fetching Natural Earth data for Alaska...")
try:
    with urllib.request.urlopen(alaska_url, timeout=60) as response:
        ne_data = json.loads(response.read().decode())
    alaska_features = [f for f in ne_data["features"] if f["properties"].get("name", "").lower() == "alaska"]
    if alaska_features:
        alaska_data = {"type": "FeatureCollection", "features": alaska_features}
        has_alaska = True
        print(f"  Found Alaska feature")
    else:
        print("  Warning: Alaska not found in Natural Earth data")
        has_alaska = False
except Exception as e:
    print(f"  Warning: Could not fetch Natural Earth data: {e}")
    has_alaska = False

# SVG dimensions
SVG_WIDTH = 1000
SVG_HEIGHT = 680

# Canada bounds - expanded to show Alaska mainland adjacent to Canada
MIN_LON = -170.0
MAX_LON = -48.0
MIN_LAT = 40.0
MAX_LAT = 72.0

# Spherical distortion settings
REF_LAT = 50.0         # Reference latitude (scale=1.0 here)
SPHERE_STRENGTH = 0.55 # 0=flat, 1=full spherical narrowing at poles

def lon_to_x(lon, lat=None):
    normalized = (lon - MIN_LON) / (MAX_LON - MIN_LON)
    if lat is not None:
        center_x = 0.5
        raw_scale = math.cos(math.radians(lat)) / math.cos(math.radians(REF_LAT))
        scale = 1.0 + (raw_scale - 1.0) * SPHERE_STRENGTH
        normalized = center_x + (normalized - center_x) * scale
    return normalized * SVG_WIDTH

def lat_to_y(lat):
    lat_rad = math.radians(lat)
    mercator_y = math.log(math.tan(math.pi/4 + lat_rad/2))
    min_lat_rad = math.radians(MIN_LAT)
    max_lat_rad = math.radians(MAX_LAT)
    min_merc = math.log(math.tan(math.pi/4 + min_lat_rad/2))
    max_merc = math.log(math.tan(math.pi/4 + max_lat_rad/2))
    return SVG_HEIGHT - (mercator_y - min_merc) / (max_merc - min_merc) * SVG_HEIGHT

def simplify_coords(coords, tolerance=0.15):
    """Douglas-Peucker line simplification."""
    if len(coords) <= 2:
        return coords
    
    # Find the point with maximum distance from line between first and last
    max_dist = 0
    max_idx = 0
    p1 = coords[0]
    p2 = coords[-1]
    
    for i in range(1, len(coords) - 1):
        p = coords[i]
        # Distance from point to line
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            dist = math.sqrt((p[0]-p1[0])**2 + (p[1]-p1[1])**2)
        else:
            t = ((p[0]-p1[0])*dx + (p[1]-p1[1])*dy) / (dx*dx + dy*dy)
            t = max(0, min(1, t))
            proj_x = p1[0] + t * dx
            proj_y = p1[1] + t * dy
            dist = math.sqrt((p[0]-proj_x)**2 + (p[1]-proj_y)**2)
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    
    if max_dist > tolerance:
        left = simplify_coords(coords[:max_idx+1], tolerance)
        right = simplify_coords(coords[max_idx:], tolerance)
        return left[:-1] + right
    else:
        return [coords[0], coords[-1]]

def coords_to_svg_path(rings, tolerance=0.15):
    """Convert coordinate rings to SVG path data with simplification."""
    path_parts = []
    for ring in rings:
        simplified = simplify_coords(ring, tolerance)
        if len(simplified) < 3:
            continue
        parts = []
        for i, (lon, lat) in enumerate(simplified):
            x = round(lon_to_x(lon, lat), 1)
            y = round(lat_to_y(lat), 1)
            if i == 0:
                parts.append(f"M{x},{y}")
            else:
                parts.append(f"L{x},{y}")
        parts.append("Z")
        path_parts.append("".join(parts))
    return "".join(path_parts)

# Province name to ID mapping
province_ids = {
    "British Columbia": "BC",
    "Alberta": "AB",
    "Saskatchewan": "SK",
    "Manitoba": "MB",
    "Ontario": "ON",
    "Quebec": "QC",
    "Québec": "QC",
    "New Brunswick": "NB",
    "Nova Scotia": "NS",
    "Prince Edward Island": "PE",
    "Newfoundland and Labrador": "NL",
    "Yukon": "YT",
    "Yukon Territory": "YT",
    "Northwest Territories": "NT",
    "Nunavut": "NU",
}

# Province colors (dark theme with purple/blue/violet palette)
province_colors = {
    "BC": "#4a1a6b",
    "AB": "#3d1a5e",
    "SK": "#351a52",
    "MB": "#2d1a47",
    "ON": "#3a1a5c",
    "QC": "#451a65",
    "NB": "#3f1a60",
    "NS": "#481a68",
    "PE": "#501a70",
    "NL": "#421a62",
    "YT": "#381a58",
    "NT": "#331a50",
    "NU": "#2a1a44",
}

# City locations (lon, lat)
cities = {
    "Vancouver": (-123.12, 49.28),
    "Calgary": (-114.07, 51.05),
    "Edmonton": (-113.49, 53.54),
    "Winnipeg": (-97.14, 49.90),
    "Toronto": (-79.38, 43.65),
    "Ottawa": (-75.70, 45.42),
    "Montreal": (-73.57, 45.50),
    "Halifax": (-63.57, 44.65),
    "London ON": (-81.25, 42.98),
}

# Process each province - only keep largest polygons
provinces = {}
for feature in data["features"]:
    name = feature["properties"]["name"]
    pid = province_ids.get(name, name[:2].upper())
    geom = feature["geometry"]
    
    if geom["type"] == "MultiPolygon":
        all_rings = []
        for polygon in geom["coordinates"]:
            for ring in polygon:
                all_rings.append(ring)
    elif geom["type"] == "Polygon":
        all_rings = geom["coordinates"]
    else:
        continue
    
    # Sort rings by area (keep larger ones for better detail vs file size)
    def ring_area(ring):
        area = 0
        for i in range(len(ring)):
            j = (i + 1) % len(ring)
            area += ring[i][0] * ring[j][1]
            area -= ring[j][0] * ring[i][1]
        return abs(area) / 2
    
    all_rings.sort(key=ring_area, reverse=True)
    
    # Keep top N rings per province (more for large provinces)
    if pid in ("NU", "NT", "NL", "BC", "QC", "ON"):
        max_rings = 25  # These have many important islands
    else:
        max_rings = 8
    
    # Also filter tiny rings
    if all_rings:
        largest_area = ring_area(all_rings[0])
        min_area_threshold = largest_area * 0.001  # At least 0.1% of largest
        filtered_rings = []
        for ring in all_rings[:max_rings]:
            if ring_area(ring) >= min_area_threshold:
                filtered_rings.append(ring)
        all_rings = filtered_rings
    
    # Use looser simplification for territories (they're huge)
    if pid in ("NU",):
        tol = 0.3
    elif pid in ("NT",):
        tol = 0.2
    elif pid in ("YT",):
        tol = 0.15
    elif pid in ("PE", "NB", "NS", "NL"):
        tol = 0.06
    elif pid in ("BC", "QC", "ON"):
        tol = 0.12
    else:
        tol = 0.1
    
    path_data = coords_to_svg_path(all_rings, tolerance=tol)
    provinces[pid] = {
        "name": name,
        "path": path_data,
        "color": province_colors.get(pid, "#3a1a5c"),
    }
    print(f"  {name} ({pid}): {len(all_rings)} rings, path length: {len(path_data)}")

print(f"\nTotal provinces: {len(provinces)}")

# Process Alaska border outline
alaska_path_data = ""
if has_alaska:
    print("\nProcessing Alaska...")
    alaska_rings = []
    for feature in alaska_data["features"]:
        geom = feature["geometry"]
        if geom["type"] == "MultiPolygon":
            for polygon in geom["coordinates"]:
                ring = polygon[0]  # outer ring only
                # Skip far Aleutian islands (west of -172)
                max_lon = max(c[0] for c in ring)
                if max_lon < -172:
                    continue
                alaska_rings.append(ring)
        elif geom["type"] == "Polygon":
            ring = geom["coordinates"][0]
            max_lon = max(c[0] for c in ring)
            if max_lon < -172:
                continue
            alaska_rings.append(ring)

    # Sort by area, keep only mainland + large islands
    def ring_area(ring):
        area = 0
        for i in range(len(ring)):
            j = (i + 1) % len(ring)
            area += ring[i][0] * ring[j][1]
            area -= ring[j][0] * ring[i][1]
        return abs(area) / 2

    alaska_rings.sort(key=ring_area, reverse=True)
    # Keep top rings, filter tiny ones
    if alaska_rings:
        largest = ring_area(alaska_rings[0])
        alaska_rings = [r for r in alaska_rings[:15] if ring_area(r) >= largest * 0.005]
    alaska_path_data = coords_to_svg_path(alaska_rings, tolerance=0.2)
    print(f"  Alaska: {len(alaska_rings)} rings, path length: {len(alaska_path_data)}")

# Generate city coordinates
city_svg = []
city_svg_dict = {}
for name, (lon, lat) in cities.items():
    x = round(lon_to_x(lon, lat), 1)
    y = round(lat_to_y(lat), 1)
    city_svg.append((name, x, y))
    city_svg_dict[name] = (x, y)

# Generate arc paths between cities
arcs = [
    ("Vancouver", "Calgary"),
    ("Calgary", "Toronto"),
    ("Toronto", "Montreal"),
    ("Montreal", "Halifax"),
]

arc_paths = []
for c1_name, c2_name in arcs:
    c1 = cities[c1_name]
    c2 = cities[c2_name]
    x1 = round(lon_to_x(c1[0], c1[1]), 1)
    y1 = round(lat_to_y(c1[1]), 1)
    x2 = round(lon_to_x(c2[0], c2[1]), 1)
    y2 = round(lat_to_y(c2[1]), 1)
    # Arc with curvature (flipped: arcs bow upward left-to-right)
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)
    curve = dist * 0.32
    # Perpendicular offset (flipped direction)
    nx = dy / dist * curve
    ny = -dx / dist * curve
    cx = round(mx + nx, 1)
    cy = round(my + ny, 1)
    arc_paths.append(f'M{x1},{y1} Q{cx},{cy} {x2},{y2}')

# Build SVG - matching original color scheme (outline-only, no labels, no animations)
svg_parts = []
svg_parts.append(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" role="img" aria-label="Canada compute network map">
  <defs>
    <linearGradient id="provinceFill" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00d4ff" stop-opacity="0.06"/>
      <stop offset="55%" stop-color="#7c3aed" stop-opacity="0.04"/>
      <stop offset="100%" stop-color="#dc2626" stop-opacity="0.03"/>
    </linearGradient>
    <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#00d4ff"/>
      <stop offset="100%" stop-color="#7c3aed"/>
    </linearGradient>
    <filter id="dotGlow" x="-200%" y="-200%" width="500%" height="500%">
      <feGaussianBlur stdDeviation="4" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="arcGlow" x="-100%" y="-100%" width="300%" height="300%">
      <feGaussianBlur stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <!-- Arctic fade: fades to transparent at top (for mask: white=visible, black=hidden) -->
    <linearGradient id="arcticFadeMask" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="black"/>
      <stop offset="15%" stop-color="black"/>
      <stop offset="40%" stop-color="#555"/>
      <stop offset="60%" stop-color="white"/>
      <stop offset="100%" stop-color="white"/>
    </linearGradient>
    <!-- Left edge fade (Alaska side, for mask) -->
    <linearGradient id="leftFadeMask" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="black"/>
      <stop offset="30%" stop-color="#444"/>
      <stop offset="60%" stop-color="white"/>
      <stop offset="100%" stop-color="white"/>
    </linearGradient>
    <!-- Right edge fade (Newfoundland side, for mask) -->
    <linearGradient id="rightFadeMask" x1="1" y1="0" x2="0" y2="0">
      <stop offset="0%" stop-color="#888"/>
      <stop offset="15%" stop-color="#ccc"/>
      <stop offset="35%" stop-color="white"/>
      <stop offset="100%" stop-color="white"/>
    </linearGradient>
    <!-- Combined edge mask -->
    <mask id="edgeMask" maskContentUnits="objectBoundingBox">
      <rect width="1" height="1" fill="white"/>
      <rect width="1" height=".55" fill="url(#arcticFadeMask)"/>
      <rect width=".22" height="1" fill="url(#leftFadeMask)"/>
      <rect x=".88" width=".12" height="1" fill="url(#rightFadeMask)"/>
    </mask>
    <style>
      .land{{fill:url(#provinceFill);stroke:currentColor;stroke-width:1.0;stroke-opacity:.38;stroke-linejoin:round;stroke-linecap:round}}
      .alaska{{fill:none;stroke:currentColor;stroke-width:0.8;stroke-opacity:.22;stroke-linejoin:round;stroke-linecap:round}}
      .ghost{{fill:none;stroke:#00d4ff;stroke-width:.5;stroke-opacity:.12;stroke-linejoin:round;stroke-linecap:round}}
      .ghost-lon{{fill:none;stroke:#7c3aed;stroke-width:.4;stroke-opacity:.08;stroke-linejoin:round;stroke-linecap:round}}
      .water{{fill:none}}
      .arc{{fill:none;stroke:url(#arcGrad);stroke-width:1.6;opacity:.55;stroke-linecap:round;filter:url(#arcGlow)}}
      .core{{filter:url(#dotGlow)}}
      .frame{{fill:none;stroke:#00d4ff;stroke-width:.4;stroke-opacity:.08;rx:4}}
    </style>
  </defs>

  <!-- All content wrapped in edge-fade mask -->
  <g mask="url(#edgeMask)">

  <!-- Ghost latitude lines -->
  <g class="ghost">''')

# Ghost latitude lines (curved for spherical projection)
for lat in range(45, 80, 5):
    y = round(lat_to_y(lat), 1)
    points = []
    for lon_step in range(0, 21):
        lon = MIN_LON + (MAX_LON - MIN_LON) * lon_step / 20
        x = round(lon_to_x(lon, lat), 1)
        if lon_step == 0:
            points.append(f'M{x},{y}')
        else:
            points.append(f'L{x},{y}')
    svg_parts.append(f'    <path d="{"".join(points)}"/>')

svg_parts.append('  </g>')

# Ghost longitude lines (faint vertical grid for tech look)
svg_parts.append('  <g class="ghost-lon">')
for lon in range(-140, -50, 10):
    points = []
    for lat_step in range(0, 21):
        lat = MIN_LAT + (MAX_LAT - MIN_LAT) * lat_step / 20
        x = round(lon_to_x(lon, lat), 1)
        y = round(lat_to_y(lat), 1)
        if lat_step == 0:
            points.append(f'M{x},{y}')
        else:
            points.append(f'L{x},{y}')
    svg_parts.append(f'    <path d="{"".join(points)}"/>')
svg_parts.append('  </g>')

# Water bodies
svg_parts.append('\n  <!-- Water Bodies -->')
svg_parts.append('  <g>')

# Hudson Bay (approximate)
hb_coords = [
    (-95.0, 63.0), (-94.5, 61.5), (-94.8, 60.0), (-93.5, 58.5),
    (-92.5, 57.5), (-89.0, 56.5), (-86.5, 56.0), (-85.0, 55.3),
    (-83.5, 55.2), (-82.0, 55.2), (-81.0, 52.5), (-80.0, 51.5),
    (-79.0, 51.0), (-79.0, 52.5), (-79.5, 54.0), (-79.5, 56.8),
    (-80.0, 58.0), (-81.0, 59.5), (-83.0, 61.5), (-85.0, 63.0),
    (-87.0, 63.5), (-89.0, 63.3), (-91.0, 63.0), (-93.0, 63.5),
    (-95.0, 63.0)
]
water_bodies = [("Hudson Bay", hb_coords)]

# Lake Superior
ls_coords = [
    (-92.1, 46.8), (-91.0, 46.9), (-89.5, 47.0), (-88.5, 47.3),
    (-87.5, 47.4), (-86.5, 47.1), (-85.5, 47.0), (-84.7, 46.5),
    (-84.5, 46.4), (-84.5, 46.8), (-85.0, 47.2), (-85.5, 47.5),
    (-86.0, 48.0), (-86.5, 48.5), (-87.5, 48.3), (-88.5, 48.5),
    (-89.5, 48.0), (-90.5, 47.5), (-91.0, 47.3), (-92.1, 46.8)
]
water_bodies.append(("Lake Superior", ls_coords))

# Lake Huron
lh_coords = [
    (-84.5, 46.0), (-83.5, 45.5), (-82.5, 44.5), (-82.0, 43.5),
    (-82.2, 43.0), (-82.5, 43.6), (-83.0, 44.0), (-83.5, 44.8),
    (-84.0, 45.0), (-84.5, 45.5), (-84.5, 46.0)
]
water_bodies.append(("Lake Huron", lh_coords))

# Lake Erie
le_coords = [
    (-83.5, 41.7), (-82.5, 41.7), (-81.5, 41.8), (-80.5, 42.0),
    (-79.5, 42.5), (-79.0, 42.8), (-79.5, 42.9), (-80.5, 42.6),
    (-81.5, 42.3), (-82.5, 42.0), (-83.5, 41.7)
]
water_bodies.append(("Lake Erie", le_coords))

# Lake Ontario
lo_coords = [
    (-79.8, 43.2), (-79.0, 43.3), (-78.0, 43.3), (-77.0, 43.5),
    (-76.5, 43.8), (-76.3, 44.0), (-76.5, 44.2), (-77.0, 44.0),
    (-78.0, 43.7), (-79.0, 43.5), (-79.8, 43.2)
]
water_bodies.append(("Lake Ontario", lo_coords))

for wb_name, wb_coords in water_bodies:
    parts = []
    for i, (lon, lat) in enumerate(wb_coords):
        x = round(lon_to_x(lon, lat), 1)
        y = round(lat_to_y(lat), 1)
        if i == 0:
            parts.append(f"M{x},{y}")
        else:
            parts.append(f"L{x},{y}")
    parts.append("Z")
    svg_parts.append(f'    <path class="water" d="{"".join(parts)}">')
    svg_parts.append(f'      <title>{wb_name}</title>')
    svg_parts.append(f'    </path>')

svg_parts.append('  </g>')
svg_parts.append('\n  <!-- Provinces -->')
svg_parts.append('  <g color="#d7e6ff">')

for pid, prov in provinces.items():
    svg_parts.append(f'    <path class="land" data-name="{prov["name"]}" d="{prov["path"]}"/>')

# Alaska outline (ghost style, fainter than provinces)
if alaska_path_data:
    svg_parts.append(f'    <path class="alaska" data-name="Alaska" d="{alaska_path_data}"/>')

svg_parts.append('  </g>')

# Ghost internal province borders
svg_parts.append('\n  <!-- Ghost internal borders -->')
svg_parts.append('  <g class="ghost">')
# AB/SK border, SK/MB border, MB/ON border, etc. - these are already in the land paths
svg_parts.append('  </g>')

# Arc paths
svg_parts.append('\n  <!-- Connection Arcs -->')
svg_parts.append('  <g>')
for arc in arc_paths:
    svg_parts.append(f'    <path class="arc" d="{arc}"/>')
# Add a long dashed arc Vancouver -> Toronto
van = city_svg_dict["Vancouver"]
tor = city_svg_dict["Toronto"]
mx = (van[0] + tor[0]) / 2
my = (van[1] + tor[1]) / 2
dx = tor[0] - van[0]
dy = tor[1] - van[1]
dist = math.sqrt(dx*dx + dy*dy)
curve = dist * 0.38
nx = dy / dist * curve
ny = -dx / dist * curve
cx2 = round(mx + nx, 1)
cy2 = round(my + ny, 1)
svg_parts.append(f'    <path class="arc" d="M{van[0]},{van[1]} Q{cx2},{cy2} {tor[0]},{tor[1]}" stroke-dasharray="6 8"/>')
svg_parts.append('  </g>')

# City dot colors matching original
city_colors = {
    "Vancouver": "#7c3aed",
    "Calgary": "#00d4ff",
    "Edmonton": "#00d4ff",
    "Winnipeg": "#00d4ff",
    "Toronto": "#00d4ff",
    "Ottawa": "#00d4ff",
    "Montreal": "#00d4ff",
    "Halifax": "#10b981",
    "London ON": "#00d4ff",
}
# Major cities get bigger dots
city_sizes = {
    "Vancouver": 4.5,
    "Toronto": 4.5,
    "Montreal": 4.5,
    "Calgary": 3,
    "Edmonton": 2,
    "Winnipeg": 2,
    "Ottawa": 3,
    "Halifax": 3,
    "London ON": 2,
}

# Cities (dots only, no labels)
svg_parts.append('\n  <!-- Cities -->')
svg_parts.append('  <g>')
for name, x, y in city_svg:
    color = city_colors.get(name, "#00d4ff")
    size = city_sizes.get(name, 2)
    svg_parts.append(f'    <circle class="core" cx="{x}" cy="{y}" r="{size}" fill="{color}"/>')
svg_parts.append('  </g>')

# Subtle tech frame (outside mask so it stays full opacity)
svg_parts.append('\n  <!-- Close edge-fade mask group -->')
svg_parts.append('  </g>')

svg_parts.append('\n  <!-- Tech frame -->')
svg_parts.append(f'  <rect class="frame" x="3" y="3" width="{SVG_WIDTH - 6}" height="{SVG_HEIGHT - 6}" rx="3"/>')

svg_parts.append('\n</svg>')

svg_content = "\n".join(svg_parts)

output_path = "/Users/aaryn/Projects/aarynfans-mvp/app/svg/canada_map.svg"
with open(output_path, "w") as f:
    f.write(svg_content)

size_kb = len(svg_content) / 1024
print(f"\nSVG written to {output_path}")
print(f"File size: {size_kb:.1f} KB")
