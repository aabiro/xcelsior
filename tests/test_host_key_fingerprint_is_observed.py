"""A0 of the host-key plan: read the fingerprint, publish nothing.

`open_instance_access` returns `host_key_fingerprint: null` with a note saying
the platform publishes none. That was the honest thing to ship — but Gate P2 asks
for "the SSH endpoint **plus the fingerprint to verify**", and an agent walking a
user through their first SSH connection currently has to tell them it cannot be
verified.

The value was never unknowable **on an Xcelsior-managed host**. `worker_agent`
runs `ls /etc/ssh/ssh_host_*_key || ssh-keygen -A` inside every interactive
container, so each instance has a unique host key; and the public port is DNAT'd
to the container rather than terminated by a gateway (`ssh_port` is hardcoded to
22 container-side precisely because "sshd would refuse the gateway's relay"), so
there the key the container presents **is** the key the user's client sees.

**That is not universal, and this file used to imply it was.** Checked against a
live cloud GPU host on 2026-08-09: the container ran **no sshd and held no host
keys at all** — the provider's proxy terminates SSH and pipes in. On that host
class the key a client verifies belongs to the proxy, and reading the container
would publish the wrong one. Returning `""` there is the correct answer rather
than a degraded one, which is why the "empty is real" rule below is load-bearing
and not defensive coding.

The same check corrected a second assumption: the endpoint offered **RSA-2048 and
no Ed25519**, so a reader asking only for Ed25519 would have reported "unknown"
for a host whose key was perfectly verifiable.

## Why this stage stores nothing

Instance images vary far more than the control plane does. Before anything
depends on the value, the rate at which it comes back blank across real base
images has to be *known* rather than assumed. So A0 logs and returns; storage,
API, and the tool field come later, and each is useless without this one working.

## The rule this file mostly exists to pin

**Empty is a real answer.** A fingerprint the platform did not observe must never
be synthesised, defaulted, or borrowed from a sibling instance, because `null`
makes a model say "this cannot be verified" while a wrong value makes it say
"verified" — and the second is worse than having no feature at all.
"""

from __future__ import annotations

import os
import subprocess

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

REAL = "SHA256:hLtY9vQm2xKpR7wZ3nF8sJ4cB6dE0aG1iU5oT9yXvNw"

#: Genuine `ssh-keygen -lf` output for a freshly generated Ed25519 key, pasted
#: verbatim. The matcher was written from memory of the format, and a fixture
#: invented from the same memory would agree with it for the same wrong reasons.
#: Note the `+` — SHA256 fingerprints are unpadded base64, so the character class
#: has to include `+` and `/`, which is exactly the kind of detail a
#: hand-written fixture omits.
REAL_SSH_KEYGEN_OUTPUT = (
    "256 SHA256:cV3DxlKNScYYsCLm5gs+JeEpWL5AjD5b4zCQ7csKIMo "
    "aaryn@aaryn-ASUS-TUF-Gaming-A15-FA506IV-TUF506IV (ED25519)\n"
)
REAL_FROM_OUTPUT = "SHA256:cV3DxlKNScYYsCLm5gs+JeEpWL5AjD5b4zCQ7csKIMo"


class _Proc:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def _patch_run(monkeypatch, result):
    import worker_agent

    def fake_run(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    return worker_agent


def test_real_ssh_keygen_output_is_parsed(monkeypatch):
    """Against output a real `ssh-keygen` produced, not a fixture I invented.

    This is the calibration for every other case in the file: they all use a
    hand-written fingerprint, and if my mental model of the format were wrong
    they would agree with the matcher for the same wrong reason.
    """
    wa = _patch_run(monkeypatch, _Proc(REAL_SSH_KEYGEN_OUTPUT))
    assert wa.read_container_host_key_fingerprint("c1") == REAL_FROM_OUTPUT


def test_a_normal_ssh_keygen_line_yields_the_fingerprint(monkeypatch):
    """The shape `ssh-keygen -lf` actually prints."""
    wa = _patch_run(monkeypatch, _Proc(f"256 {REAL} root@a1b2c3 (ED25519)\n"))
    assert wa.read_container_host_key_fingerprint("c1") == REAL


def test_a_comment_containing_spaces_does_not_shift_the_field(monkeypatch):
    """The comment is user-influenced and can contain spaces.

    Splitting on whitespace and taking index 1 happens to work here, but a
    comment is not guaranteed to be one token, so the fingerprint is matched by
    its own shape instead of by position.
    """
    wa = _patch_run(monkeypatch, _Proc(f"256 {REAL} my laptop key, generated 2026 (ED25519)\n"))
    assert wa.read_container_host_key_fingerprint("c1") == REAL


def test_no_key_file_yields_empty_not_a_guess(monkeypatch):
    wa = _patch_run(monkeypatch, _Proc(""))
    assert wa.read_container_host_key_fingerprint("c1") == ""


def test_an_image_without_ssh_keygen_yields_empty(monkeypatch):
    wa = _patch_run(monkeypatch, _Proc("sh: ssh-keygen: not found\n", returncode=127))
    assert wa.read_container_host_key_fingerprint("c1") == ""


def test_a_docker_exec_failure_yields_empty_and_does_not_raise(monkeypatch):
    """The launch path calls this. It must never be the reason a launch fails."""
    wa = _patch_run(monkeypatch, subprocess.TimeoutExpired(cmd="docker", timeout=10))
    assert wa.read_container_host_key_fingerprint("c1") == ""


@pytest.mark.parametrize(
    "line",
    [
        "256 SHA256:tooshort root@x (ED25519)",
        "256 MD5:16:27:ac:a5:76:28:2d:36:63:1b:56:4d root@x (ED25519)",
        "not a fingerprint at all",
        "256 SHA256:has spaces in it here root@x (ED25519)",
    ],
)
def test_output_that_is_not_a_sha256_fingerprint_is_refused(monkeypatch, line: str):
    """Anything that is not the exact shape is unknown, not best-effort.

    The MD5 case matters: older `ssh-keygen` prints MD5 hex, which is a real
    fingerprint of the right key in the wrong format. Returning it would produce
    a value a user could never match against what their client shows.
    """
    wa = _patch_run(monkeypatch, _Proc(line + "\n"))
    assert wa.read_container_host_key_fingerprint("c1") == ""


def test_the_observer_never_raises(monkeypatch):
    """Calibration for the wrapper the launch path actually calls."""
    import worker_agent

    def explode(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(worker_agent, "read_container_host_key_fingerprint", explode)
    worker_agent._log_container_host_key_fingerprint("c1", "job-1")


def test_a0_publishes_nothing_yet():
    """The stage boundary, asserted so it cannot be skipped by accident.

    A0 observes. If a later change starts reporting this value, that is A1 and
    it needs the malformed-input refusal at the API boundary that A1 specifies —
    this column is served to users, and must never carry text a worker chose.
    """
    import inspect

    import worker_agent

    source = inspect.getsource(worker_agent._log_container_host_key_fingerprint)
    for reporting in ("requests.", "httpx.", "post(", "callback", "report"):
        assert reporting not in source, (
            f"A0 appears to be reporting the fingerprint ({reporting!r}); that is "
            "A1, and it needs input validation at the API boundary first"
        )


# ── What real hardware corrected ────────────────────────────────────


#: Genuine `ssh-keygen -lf` output for an **RSA** host key. Checked against a
#: live GPU host on 2026-08-09: the endpoint presented RSA-2048 and offered no
#: Ed25519 at all. The first version of the reader asked only for Ed25519, on
#: the reasoning that it is OpenSSH's default preference and therefore what a
#: modern client negotiates — true only when the server offers one.
REAL_RSA_OUTPUT = (
    "2048 SHA256:ISYTeX+iCU2BYWSSb/jBds0Nvhjn2F9DlpVnoxRdSqo "
    "root@d95c0069a7e42af6 (RSA)\n"
)
REAL_RSA_FINGERPRINT = "SHA256:ISYTeX+iCU2BYWSSb/jBds0Nvhjn2F9DlpVnoxRdSqo"


def test_an_rsa_only_host_still_yields_a_fingerprint(monkeypatch):
    """The correction. "Unknown" for a host whose key is verifiable is a bug.

    Reporting `null` here would tell a user their connection cannot be verified
    when it can — the same class of wrongness as reporting a value that was
    never observed, in the opposite direction.
    """
    wa = _patch_run(monkeypatch, _Proc(REAL_RSA_OUTPUT))
    assert wa.read_container_host_key_fingerprint("c1") == REAL_RSA_FINGERPRINT


def test_the_reader_asks_in_client_negotiation_order():
    """Order matters, not merely coverage.

    A host offering both must report the key a client will actually negotiate.
    Reading RSA first on a dual-key host would publish a fingerprint the user's
    client never sees — verification would fail on a host that is fine, which
    trains people to ignore the warning.
    """
    import inspect

    import worker_agent as wa

    source = inspect.getsource(wa.read_container_host_key_fingerprint)
    assert "for t in ed25519 ecdsa rsa" in source, (
        "the reader no longer follows the client's host-key preference order"
    )


def test_a_proxy_terminated_host_reports_unknown(monkeypatch):
    """The finding that limits the whole feature, asserted so it is not forgotten.

    A cloud GPU host checked on 2026-08-09 ran **no sshd and held no host keys
    in the container at all** — the provider's proxy terminates SSH and pipes
    in. The key a client verifies there belongs to the proxy, so reading the
    container would publish the wrong one.

    `""` is therefore not a degraded answer on that host class; it is the
    correct one, and any later change that "fixes" the blank by substituting
    something reintroduces exactly the failure this returns empty to avoid.
    """
    wa = _patch_run(monkeypatch, _Proc(""))
    assert wa.read_container_host_key_fingerprint("c1") == ""
