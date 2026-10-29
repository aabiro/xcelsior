"""P7's lineage must survive the trip to the browser, not just exist in the row.

Gate P7: *"a snapshot records its lineage"*. It does — `user_images` carries
`source_job_id`, `base_image_ref`, `host_id`, `created_at` and `image_digest`,
and `tests/test_snapshot_records_its_lineage.py` pins that the columns are
written. `GET /api/v2/user-images` selects all of them.

And the dashboard showed **none of it**. The image library listed name, tag,
size, status and age; the plan's frontend clause asks for "what it was built
from, when, by which run", and only *when* was there. Worse, the frontend's own
`UserImage` type had simply omitted `base_image_ref` and `image_digest`, so a
component could not have rendered them even by trying — TypeScript would have
called them errors.

That is the failure this file guards: not a missing column, not a missing route,
but a **type that quietly drops fields the route returns**. Nothing fails when it
happens. The data arrives over the wire, the type says it does not exist, and the
UI is built around the smaller shape.

## What is asserted

The route's projection and the frontend's declared type agree on the lineage
fields. Either side dropping one breaks this, which is the only place the two are
compared — one is Python, the other TypeScript, and no compiler sees both.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROUTE = ROOT / "routes" / "instances.py"
API_TS = ROOT / "frontend" / "src" / "lib" / "api.ts"

#: The lineage the gate names, plus the digest a sweep pins.
LINEAGE_FIELDS = ("source_job_id", "base_image_ref", "host_id", "created_at", "image_digest")


def _list_route_body() -> str:
    source = ROUTE.read_text(encoding="utf-8")
    body = source.split("def api_list_user_images", 1)[1]
    return body.split("\n@router.", 1)[0]


def _user_image_type() -> str:
    source = API_TS.read_text(encoding="utf-8")
    body = source.split("export type UserImage = {", 1)[1]
    return body.split("};", 1)[0]


def test_the_route_returns_every_lineage_field():
    body = _list_route_body()
    missing = [f for f in LINEAGE_FIELDS if f'"{f}"' not in body]
    assert not missing, (
        f"GET /api/v2/user-images no longer returns {missing}. Gate P7 asks that "
        "a snapshot record its lineage; dropping it from the projection makes "
        "the record unreachable even though the column still holds it."
    )


def test_the_frontend_type_declares_every_lineage_field():
    """The half that was broken.

    `base_image_ref` and `image_digest` were absent here while the route
    returned both, so the dashboard could not show what an image was built from.
    A type narrower than its payload is invisible: nothing errors, the data is
    simply discarded at the boundary.
    """
    declared = _user_image_type()
    missing = [f for f in LINEAGE_FIELDS if not re.search(rf"^\s*{f}\??:", declared, re.M)]
    assert not missing, (
        f"frontend UserImage does not declare {missing}, so no component can "
        "render them however the API is shaped — TypeScript rejects the access. "
        "Add them to the type when the route grows a field."
    )


def test_the_dashboard_actually_renders_the_lineage():
    """Declaring a field is not showing it.

    The type carried `source_job_id` all along and the image library never
    rendered it, which is how "the data is there" and "the user can see it"
    drifted apart in the first place.
    """
    page = (
        ROOT / "frontend" / "src" / "app" / "(dashboard)" / "dashboard" / "templates" / "page.tsx"
    ).read_text(encoding="utf-8")
    for field in ("base_image_ref", "source_job_id", "image_digest"):
        assert f"img.{field}" in page, (
            f"the image library no longer renders {field}; P7's frontend clause "
            "asks for what an image was built from and by which run"
        )
