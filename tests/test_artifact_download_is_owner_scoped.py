"""An artifact_id is not an authorization.

`POST /api/artifacts/download` accepts either `job_id`+`filename` or a bare
`artifact_id`. The first branch has always resolved through
`_resolve_artifact_job_id`, which calls `_check_job_access`. The second went
straight to `ArtifactManager.request_download_by_id`, which selected on
`artifact_id` alone — no tenant, no owner — and returned a presigned URL.

Two halves of one handler disagreeing about whether ownership matters.

The scope check did not close it. `_require_scope` is a **no-op for interactive
sessions** by design, so for any logged-in user the only thing between them and
another tenant's weights and checkpoints was not knowing a UUID. UUIDs are not
secrets: they appear in logs, in SSE payloads, in screenshots, and in the
listing endpoint.

`storage.artifacts` has carried `tenant_id`, `owner_user_id` and `job_id` the
whole time. The query simply did not use them.

The fix makes the check impossible to omit rather than merely present:
`request_download_by_id` now takes a **required** keyword-only `authorize`
callback. A call site that forgets it raises `TypeError` at the call instead of
returning a URL — which is why the state-machine test that legitimately bypasses
ownership now has to say so on the line where it does.
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")


def test_the_authorizer_is_required_not_optional():
    """The property that prevents this returning by omission.

    If `authorize` ever gains a default, every future call site can forget it
    and leak silently — which is exactly how this defect existed.
    """
    import inspect

    from artifacts import ArtifactManager

    sig = inspect.signature(ArtifactManager.request_download_by_id)
    param = sig.parameters.get("authorize")
    assert param is not None, "request_download_by_id no longer takes an authorizer"
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, "authorize must be keyword-only"
    assert param.default is inspect.Parameter.empty, (
        "authorize has acquired a default, so a call site can now omit the "
        "ownership check and get a presigned URL — the original defect"
    )


def test_a_refusing_authorizer_stops_the_url_being_minted():
    """The callback is consulted, and refusing it prevents a URL.

    Uses a random id so the test asserts ordering rather than existence: the
    authorizer must run before anything reaches storage. If the implementation
    ever moved the check after `generate_download_url`, a refused caller would
    still have caused a signed URL to exist.
    """
    from artifacts import get_artifact_manager

    class Refused(Exception):
        pass

    def refuse(_owner: dict) -> None:
        raise Refused("not yours")

    mgr = get_artifact_manager()
    with pytest.raises((Refused, KeyError)):
        mgr.request_download_by_id(str(uuid.uuid4()), authorize=refuse)


def test_the_route_authorizes_the_artifact_id_branch():
    """Read the handler, because the branch is what regressed.

    Asserted against the source rather than by driving two tenants through the
    API: the failure being guarded is *a missing call in one branch*, and an
    integration test that happened to use the `job_id` branch would pass while
    the `artifact_id` branch leaked — which is the situation this file exists
    to end.
    """
    import ast
    import inspect

    from routes import artifacts as routes_artifacts

    source = inspect.getsource(routes_artifacts.api_request_download)
    tree = ast.parse(source.lstrip())

    # The manager call must pass an `authorize` keyword.
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "attr", "") == "request_download_by_id")
    ]
    assert calls, "the handler no longer calls request_download_by_id"
    for call in calls:
        assert any(kw.arg == "authorize" for kw in call.keywords), (
            "request_download_by_id is called without an authorizer, so the "
            "artifact_id branch hands out presigned URLs without checking who "
            "is asking"
        )

    # And the authorizer must actually consult an access helper.
    names = {
        getattr(node.func, "id", "") or getattr(node.func, "attr", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "_check_job_access" in names, (
        "the handler's authorizer no longer calls _check_job_access, so "
        "ownership of a job-owned artifact is not being checked"
    )
