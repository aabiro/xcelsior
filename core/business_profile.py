"""A business account's public face: display name, website, tagline and logo.

The legal `business_name` is for invoices and tax. This is the name and mark a
business wants to be seen as in the dashboard.

Every upload is decoded and re-encoded here rather than stored as received:

* it proves the bytes are an image, not a file with an image's first bytes;
* it drops EXIF and other metadata (a phone photo carries GPS coordinates);
* it bounds what is stored and served to one 512 px square.

Logos are not avatars. A transparent wordmark must stay transparent, and a dark
one drawn on a dark dashboard disappears, so the tile it sits on is chosen from
the logo itself (`tile`): dark artwork gets a light tile and light artwork a
dark one. An opaque image needs no tile; it is squared instead, padded with its
own background colour when that is flat and centre-cropped when it is a photo.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
LOGO_SIZE = 512
# Share of the square a transparent logo's artwork spans. The tile has rounded
# corners and, in the app bar, the person's avatar on one of them, so the
# artwork keeps the margin an app icon keeps (about 14 % a side).
_LOGO_INSET = 0.72
_MIN_SIDE = 48
_MAX_PIXELS = 40_000_000

DISPLAY_NAME_MAX = 80
TAGLINE_MAX = 120
WEBSITE_MAX = 200

TILES = ("none", "light", "dark")


@dataclass(frozen=True)
class ProcessedLogo:
    data: bytes
    content_type: str
    tile: str


# Zero-width non-joiner and joiner are spelling in Persian, the Indic scripts and
# emoji sequences. Every other format character (bidi overrides above all) can
# make a name read differently from what is stored, so it goes.
_KEPT_FORMAT_CHARS = frozenset("\u200c\u200d")


def _clean_line(raw: str) -> str:
    """One line of visible text: no control or format characters, single spaces."""
    text = unicodedata.normalize("NFC", raw or "")
    text = "".join(" " if ch.isspace() else ch for ch in text)
    text = "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in _KEPT_FORMAT_CHARS
    )
    return " ".join(text.split())


def normalize_display_name(raw: str) -> str:
    name = _clean_line(raw)
    if len(name) > DISPLAY_NAME_MAX:
        raise ValueError(f"Display name must be {DISPLAY_NAME_MAX} characters or fewer")
    return name


def normalize_tagline(raw: str) -> str:
    tagline = _clean_line(raw)
    if len(tagline) > TAGLINE_MAX:
        raise ValueError(f"Tagline must be {TAGLINE_MAX} characters or fewer")
    return tagline


_HOST_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z][a-z0-9-]{0,61}[a-z0-9]$")


def normalize_website(raw: str) -> str:
    """A web address a browser can open, or blank. `example.com` gains `https://`."""
    value = (raw or "").strip()
    if not value:
        return ""
    if any(ch.isspace() or unicodedata.category(ch)[0] == "C" for ch in value):
        raise ValueError("Enter a web address such as https://example.com")
    if "://" not in value:
        value = f"https://{value}"
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise ValueError("Enter a web address such as https://example.com") from exc
    if parts.scheme.lower() not in ("http", "https"):
        raise ValueError("The website must start with https:// or http://")
    if parts.username or parts.password:
        raise ValueError("The website must not include a user name or password")
    try:
        host = (parts.hostname or "").encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("Enter a web address such as https://example.com") from exc
    if not _HOST_RE.match(host):
        raise ValueError("Enter a web address such as https://example.com")
    netloc = host if port is None else f"{host}:{port}"
    path = "" if parts.path == "/" else parts.path
    url = urlunsplit((parts.scheme.lower(), netloc, path, parts.query, ""))
    if len(url) > WEBSITE_MAX:
        raise ValueError(f"The website must be {WEBSITE_MAX} characters or fewer")
    return url


def _flat_border(image) -> tuple[int, int, int] | None:
    """The border colour of an opaque image when the border is one flat colour."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    step_x, step_y = max(1, w // 64), max(1, h // 64)
    samples = [rgb.getpixel((x, y)) for x in range(0, w, step_x) for y in (0, h - 1)]
    samples += [rgb.getpixel((x, y)) for y in range(0, h, step_y) for x in (0, w - 1)]
    mean = tuple(sum(px[i] for px in samples) // len(samples) for i in range(3))
    worst = max(max(abs(px[i] - mean[i]) for i in range(3)) for px in samples)
    return mean if worst <= 14 else None


def _artwork_box(rgb, background: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    """The box around what differs from a flat background, or None if nothing does.

    Measured on the largest per-channel difference, so a saturated mark on a
    background of similar brightness (blue on grey) still counts as artwork.
    """
    from PIL import Image, ImageChops

    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, background))
    red, green, blue = diff.split()
    strongest = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    return strongest.point(lambda v: 255 if v > 24 else 0).getbbox()


def _tile_for(image) -> str:
    """Light tile for dark artwork, dark tile for light artwork."""
    small = image.copy()
    small.thumbnail((96, 96))
    total = weight = 0.0
    raw = small.tobytes()  # RGBA, four bytes a pixel, on every Pillow version
    for i in range(0, len(raw), 4):
        r, g, b, a = raw[i], raw[i + 1], raw[i + 2], raw[i + 3]
        if a < 24:
            continue
        total += (0.2126 * r + 0.7152 * g + 0.0722 * b) * a
        weight += a
    if weight == 0:
        return "light"
    return "light" if total / weight < 150 else "dark"


def process_logo(data: bytes) -> ProcessedLogo:
    """Validate an uploaded logo and return the normalised image to store.

    Raises `ValueError` with a message fit to show the person who uploaded it.
    """
    if not data:
        raise ValueError("That file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("The logo must be 2 MB or smaller")

    from user_avatars import sniff_image_ext

    if sniff_image_ext(data) is None:
        raise ValueError("Use a PNG, JPEG or WebP image")

    from PIL import Image, ImageOps, features

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > _MAX_PIXELS:
            raise ValueError("That image is too large; use one under 6000 px on each side")
        image.seek(0)
        image = ImageOps.exif_transpose(image).convert("RGBA")
    except ValueError:
        raise
    except Exception as exc:  # Pillow raises many types for a damaged file.
        raise ValueError("That file could not be read as an image") from exc

    transparent = image.getchannel("A").getextrema()[0] < 250
    if transparent:
        # Transparent margins are not part of the mark: measure and size what is drawn.
        box = image.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
        if box is None:
            raise ValueError("That image is fully transparent")
        image = image.crop(box)
    if min(image.size) < _MIN_SIDE and max(image.size) < _MIN_SIDE * 2:
        raise ValueError(f"The logo is too small; use at least {_MIN_SIDE * 2} px on its longer side")

    canvas_size = (LOGO_SIZE, LOGO_SIZE)
    if transparent:
        tile = _tile_for(image)
        inner = int(LOGO_SIZE * _LOGO_INSET)
        image = ImageOps.contain(image, (inner, inner), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        canvas.alpha_composite(image, ((LOGO_SIZE - image.width) // 2, (LOGO_SIZE - image.height) // 2))
    else:
        tile = "none"
        rgb = image.convert("RGB")
        border = _flat_border(rgb)
        if border is not None:
            # Artwork on a flat background (a wordmark on white): the margin
            # is background, not logo. Trim it, then centre the artwork on a
            # square of that colour, sized like a transparent logo.
            artwork = _artwork_box(rgb, border)
            if artwork is not None:
                rgb = rgb.crop(artwork)
            inner = int(LOGO_SIZE * _LOGO_INSET)
            rgb = ImageOps.contain(rgb, (inner, inner), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", canvas_size, border)
            canvas.paste(rgb, ((LOGO_SIZE - rgb.width) // 2, (LOGO_SIZE - rgb.height) // 2))
        else:
            # A photo or a gradient runs to the edge: square it by cropping.
            canvas = ImageOps.fit(rgb, canvas_size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    out = io.BytesIO()
    if features.check("webp"):
        if transparent:
            canvas.save(out, "WEBP", lossless=True, method=4)
        else:
            canvas.save(out, "WEBP", quality=90, method=4)
        return ProcessedLogo(out.getvalue(), "image/webp", tile)
    canvas.save(out, "PNG", optimize=True)
    return ProcessedLogo(out.getvalue(), "image/png", tile)


def logo_url(updated_at: float) -> str:
    return f"/api/auth/me/business-logo?v={int(updated_at * 1000)}"
