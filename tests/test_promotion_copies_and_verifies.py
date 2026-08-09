"""The copy crosses the boundary, and a bad one leaves nothing behind.

A1 of `docs/artifact-promotion-plan.md` — "one file, one host, no resume",
built before holds and idempotency because if a host cannot stream from object
storage to a mount, everything after it is rework.

**Real HTTP, real files.** The object is served by an actual server, the bytes
travel over a socket, and the result is read back off disk and re-hashed by the
test rather than trusting what the code under test reported. A fake that returns
canned bytes would exercise the loop and prove nothing about the boundary A1
exists to test.

## The property that matters more than success

Each object streams to a `.part`, is verified, and only then renamed into place.
A rename within one filesystem is atomic, so an interrupted or corrupt copy
leaves **no file at all** rather than a truncated one — and a truncated
checkpoint that looks complete is the failure this whole design is arranged
around. `test_a_corrupt_copy_leaves_nothing_behind` is therefore the load-bearing
test here, not the happy path.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import socketserver
import threading

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

PAYLOAD = os.urandom(1024 * 1024 + 7919)  # deliberately not a chunk multiple
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class _Server:
    """A stand-in object store and control plane, on one port."""

    def __init__(self, tmp_path):
        self.reported: dict = {}
        self.manifest: dict = {}
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep pytest output readable
                pass

            def _json(self, payload: dict):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.endswith("/manifest"):
                    return self._json(outer.manifest)
                if self.path == "/object":
                    self.send_response(200)
                    self.send_header("content-length", str(len(PAYLOAD)))
                    self.end_headers()
                    self.wfile.write(PAYLOAD)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                outer.reported.update(json.loads(self.rfile.read(n) or b"{}"))
                self._json({"ok": True})

        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()


@pytest.fixture
def agent(tmp_path):
    """`worker_agent` pointed at a local server and a temp mount root."""
    import worker_agent as wa

    srv = _Server(tmp_path)
    mounts = tmp_path / "mounts"
    mounts.mkdir()

    saved = (wa._api_url, wa._api_headers, wa.MANAGED_VOLUME_HOST_DIR)
    wa._api_url = lambda p: f"http://127.0.0.1:{srv.port}{p}"
    wa._api_headers = lambda: {"Content-Type": "application/json"}
    wa.MANAGED_VOLUME_HOST_DIR = str(mounts)

    def manifest(*, sha256: str = DIGEST, name: str = "model.safetensors", volume: str = "vol-1"):
        srv.manifest = {
            "ok": True,
            "promotion_id": "prom-1",
            "volume_id": volume,
            "manifest_sha256": "d" * 64,
            "files": [{
                "artifact_id": "art-1",
                "logical_name": name,
                "size_bytes": len(PAYLOAD),
                "sha256": sha256,
                "url": f"http://127.0.0.1:{srv.port}/object",
            }],
        }

    srv.set_manifest = manifest
    manifest()
    yield wa, srv, mounts
    wa._api_url, wa._api_headers, wa.MANAGED_VOLUME_HOST_DIR = saved
    srv.stop()


def _dest(mounts, volume="vol-1", name="model.safetensors"):
    return mounts / volume / "promoted" / name


def test_the_file_crosses_the_boundary_intact(agent):
    """The question A1 exists to answer."""
    wa, srv, mounts = agent
    assert wa._promote_artifacts({"promotion_id": "prom-1"}) is True

    dest = _dest(mounts)
    assert dest.exists(), "nothing was written to the mount"
    assert dest.stat().st_size == len(PAYLOAD)
    # Re-hashed here rather than trusting what the agent reported.
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == DIGEST


def test_success_is_reported_with_the_byte_count(agent):
    wa, srv, mounts = agent
    wa._promote_artifacts({"promotion_id": "prom-1"})
    assert srv.reported.get("state") == "succeeded"
    assert srv.reported.get("bytes_written") == len(PAYLOAD)


def test_no_part_file_survives_a_good_copy(agent):
    wa, srv, mounts = agent
    wa._promote_artifacts({"promotion_id": "prom-1"})
    contents = os.listdir(_dest(mounts).parent)
    assert contents == ["model.safetensors"], f"leftover files: {contents}"


def test_a_corrupt_copy_leaves_nothing_behind(agent):
    """The load-bearing one.

    A digest that does not match must leave **no file**, not a quarantined or
    renamed one. A truncated checkpoint sitting in the destination is the
    outcome the `.part`-then-rename dance exists to prevent, and it is worse
    than an outright failure because the user believes the promotion worked.
    """
    wa, srv, mounts = agent
    srv.set_manifest(sha256=hashlib.sha256(b"a different file").hexdigest())

    assert wa._promote_artifacts({"promotion_id": "prom-1"}) is False
    assert srv.reported.get("state") == "failed"
    assert srv.reported.get("failure_code") == "digest_mismatch"

    dest_dir = _dest(mounts).parent
    assert os.listdir(dest_dir) == [], (
        f"a failed copy left files behind: {os.listdir(dest_dir)} — a partial "
        "checkpoint that looks complete is the failure this design prevents"
    )


def test_an_artifact_with_no_digest_is_refused_before_any_bytes_move(agent):
    """Consistent with the manifest, applied where it costs something."""
    wa, srv, mounts = agent
    srv.set_manifest(sha256="")

    assert wa._promote_artifacts({"promotion_id": "prom-1"}) is False
    assert srv.reported.get("failure_code") == "unverifiable_artifact"
    assert os.listdir(_dest(mounts).parent) == []


@pytest.mark.parametrize("evil", ["../../escape.bin", "/etc/passwd", "..", "."])
def test_a_logical_name_cannot_escape_the_destination(agent, evil):
    """`logical_name` is user-supplied at upload time.

    Without basename-ing it, a promotion would write wherever the name pointed —
    on a host that also runs other tenants' containers.
    """
    wa, srv, mounts = agent
    srv.set_manifest(name=evil)

    wa._promote_artifacts({"promotion_id": "prom-1"})
    escaped = mounts.parent / "escape.bin"
    assert not escaped.exists(), "a logical name escaped the promotion directory"
    for root, _dirs, files in os.walk(mounts):
        for f in files:
            assert not f.startswith(".."), f"suspicious file written: {root}/{f}"


def test_a_missing_promotion_id_is_refused(agent):
    wa, srv, mounts = agent
    assert wa._promote_artifacts({}) is False


def test_the_command_is_on_both_allowlists():
    """The API may enqueue it and the agent may run it — or neither should.

    A command allowed on one side only is either an instruction nothing
    executes, or an agent capability nothing can reach; both are worth failing
    on rather than discovering in production.
    """
    import inspect

    import routes.agent as agent_routes
    import worker_agent as wa

    assert "promote_artifacts" in agent_routes._AGENT_COMMAND_ALLOWED, (
        "the API will not enqueue promote_artifacts, so the route that creates "
        "a promotion would raise rather than dispatch it"
    )
    # The agent's allowlist is a module-level frozenset. Asserted against the
    # dispatcher's source because that is where a name is matched — an entry in
    # the allowlist with no `elif` branch is a command the agent accepts and
    # then does nothing about, which is worse than refusing it.
    dispatch = inspect.getsource(wa.drain_agent_commands)
    assert 'name == "promote_artifacts"' in dispatch, (
        "the agent has no dispatch branch for promote_artifacts, so it would "
        "accept the command and silently do nothing"
    )
    assert "_promote_artifacts" in dispatch
