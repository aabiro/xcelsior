import math

MIN_LON, MAX_LON = -146.0, -50.0
MIN_LAT, MAX_LAT = 40.0, 78.0
SVG_W, SVG_H = 1000, 750
SPHERE = 0.55
REF_LAT = 50.0

def project(lon, lat):
    t = (lon - MIN_LON) / (MAX_LON - MIN_LON)
    x = t * SVG_W
    lat_r = math.radians(lat)
    min_r = math.radians(MIN_LAT)
    max_r = math.radians(MAX_LAT)
    my = math.log(math.tan(math.pi / 4 + lat_r / 2))
    mn = math.log(math.tan(math.pi / 4 + min_r / 2))
    mx = math.log(math.tan(math.pi / 4 + max_r / 2))
    ty = (my - mn) / (mx - mn)
    y = (1 - ty) * SVG_H
    cos_f = math.cos(math.radians(lat))
    ref_c = math.cos(math.radians(REF_LAT))
    sc = 1 - SPHERE * (1 - cos_f / ref_c)
    x = 500 + (x - 500) * sc
    return round(x, 1), round(y, 1)

cities = {
    'Whitehorse': (60.72, -135.05),
    'Yellowknife': (62.45, -114.37),
    'Goose Bay (Labrador)': (53.30, -60.33),
    'Quebec City': (46.81, -71.21),
    'Iqaluit': (63.75, -68.52),
    'Thunder Bay': (48.38, -89.25),
    'Saskatoon': (52.13, -106.67),
    'Regina': (50.45, -104.62),
    'Vancouver': (49.28, -123.12),
    'Winnipeg': (49.90, -97.14),
    'Toronto': (43.65, -79.38),
    'Halifax': (44.65, -63.57),
    'Montreal': (45.50, -73.57),
}

for name, (lat, lon) in cities.items():
    x, y = project(lon, lat)
    print(f'{name:25s}: ({x:7.1f}, {y:7.1f})  lat={lat}, lon={lon}')
