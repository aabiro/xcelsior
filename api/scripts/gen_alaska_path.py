#!/usr/bin/env python3
"""Generate Alaska SVG path data using the projection from the user's perfect SVG."""
import json
import urllib.request
import math

# Projection parameters reverse-engineered from the user's SVG:
# Grid lines at lat 45,50,55,60,65,70,75 match y positions 690.4,625.3,553.0,471.1,375.7,260.4,113.4
# City positions (Vancouver, Toronto, Halifax) confirm lon bounds
SVG_WIDTH = 1000
SVG_HEIGHT = 750
MIN_LON = -146.0
MAX_LON = -50.0
MIN_LAT = 40.0
MAX_LAT = 78.0
REF_LAT = 50.0
SPHERE_STRENGTH = 0.55

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
    min_merc = math.log(math.tan(math.pi/4 + math.radians(MIN_LAT)/2))
    max_merc = math.log(math.tan(math.pi/4 + math.radians(MAX_LAT)/2))
    return SVG_HEIGHT - (mercator_y - min_merc) / (max_merc - min_merc) * SVG_HEIGHT

def simplify_coords(coords, tolerance=0.15):
    if len(coords) <= 2:
        return coords
    max_dist = 0
    max_idx = 0
    p1, p2 = coords[0], coords[-1]
    for i in range(1, len(coords) - 1):
        p = coords[i]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if dx == 0 and dy == 0:
            dist = math.sqrt((p[0]-p1[0])**2 + (p[1]-p1[1])**2)
        else:
            t = max(0, min(1, ((p[0]-p1[0])*dx + (p[1]-p1[1])*dy) / (dx*dx + dy*dy)))
            proj_x, proj_y = p1[0] + t*dx, p1[1] + t*dy
            dist = math.sqrt((p[0]-proj_x)**2 + (p[1]-proj_y)**2)
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    if max_dist <= tolerance:
        return [coords[0], coords[-1]]
    left = simplify_coords(coords[:max_idx+1], tolerance)
    right = simplify_coords(coords[max_idx:], tolerance)
    return left[:-1] + right

# Verify projection against known city positions
cities = {
    "Vancouver": (-123.12, 49.28, 236.2, 635.0),
    "Toronto": (-79.38, 43.65, 707.4, 706.9),
    "Halifax": (-63.57, 44.65, 879.7, 694.7),
}
print("=== Verifying projection ===")
for name, (lon, lat, exp_x, exp_y) in cities.items():
    x = lon_to_x(lon, lat)
    y = lat_to_y(lat)
    print(f"  {name}: expected ({exp_x},{exp_y}), got ({x:.1f},{y:.1f}), delta=({abs(x-exp_x):.1f},{abs(y-exp_y):.1f})")

# Fetch Alaska
print("\nFetching Natural Earth 50m data for Alaska...")
url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_1_states_provinces.geojson"
with urllib.request.urlopen(url, timeout=60) as resp:
    ne_data = json.loads(resp.read().decode())

ak = [f for f in ne_data["features"] if f["properties"].get("name","").lower() == "alaska"]
if not ak:
    print("ERROR: Alaska not found!")
    exit(1)

print(f"Found Alaska feature, geometry type: {ak[0]['geometry']['type']}")

geom = ak[0]["geometry"]
if geom["type"] == "Polygon":
    all_polygons = [geom["coordinates"]]
elif geom["type"] == "MultiPolygon":
    all_polygons = geom["coordinates"]
else:
    print(f"Unexpected geometry: {geom['type']}")
    exit(1)

# Extract outer rings, filter far Aleutians
rings = []
for poly in all_polygons:
    ring = poly[0]  # outer ring only
    lons = [c[0] for c in ring]
    max_lon = max(lons)
    min_lon_ring = min(lons)
    # Skip far Aleutian islands (west of -170 or crossing date line)
    if max_lon < -170 or min_lon_ring > 0:
        continue
    area = abs(sum(ring[i][0]*ring[i+1][1] - ring[i+1][0]*ring[i][1] for i in range(len(ring)-1))) / 2
    rings.append((area, ring))

rings.sort(key=lambda x: x[0], reverse=True)
max_area = rings[0][0]
# Keep significant rings (>0.5% of largest)
rings = [(a, r) for a, r in rings if a > max_area * 0.005]
# Limit to top 15
rings = rings[:15]

print(f"Alaska: {len(rings)} significant rings")

# Generate SVG paths
paths = []
for area, ring in rings:
    simplified = simplify_coords(ring, tolerance=0.2)
    if len(simplified) < 3:
        continue
    points = [(lon_to_x(lon, lat), lat_to_y(lat)) for lon, lat in simplified]
    path_d = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
    for x, y in points[1:]:
        path_d += f"L{x:.1f},{y:.1f}"
    path_d += "Z"
    paths.append(path_d)
    # Show bounding box
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    print(f"  Ring: {len(simplified)} pts, x=[{min(xs):.0f},{max(xs):.0f}], y=[{min(ys):.0f},{max(ys):.0f}], path len={len(path_d)}")

combined = "".join(paths)
print(f"\nTotal Alaska path length: {len(combined)} chars")
print(f"\n--- ALASKA SVG PATH ---")
print(combined)
print(f"--- END ---")
