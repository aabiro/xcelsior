"""An instance's shell greets a phone as well as a desktop (agent 2.4.1).

The welcome banner was a 59-column box; a phone's web terminal is about 36
columns, where it wrapped into a mess. And NVIDIA's CDI spec gives a container
the host's render group id (992 on the tower) with no name inside the image,
so every shell printed ``groups: cannot find name for group ID 992``.

These run the shell the agent writes, with sh and bash, against temp files.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import worker_agent

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SH = shutil.which("sh")
BASH = shutil.which("bash")
# worker_agent.subprocess is this module; the setup capture patches its run().
RUN = subprocess.run


def _setup_command(monkeypatch) -> str:
    """The `sh -c` script the agent runs in a new interactive container."""
    monkeypatch.setattr(worker_agent, "SCHEDULER_URL", "http://localhost:8000")
    monkeypatch.setattr(worker_agent, "API_TOKEN", None)
    monkeypatch.setattr(worker_agent, "HOST_SSH_USER", "")
    monkeypatch.setattr(worker_agent, "_device_group_names", lambda: {992: "render", 44: "video"})
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, **kw: SimpleNamespace(
            status_code=200, ok=True, json=lambda: {"ok": True, "keys": []}
        ),
    )
    seen: list[list[str]] = []

    def run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(worker_agent.subprocess, "run", run)
    worker_agent._inject_ssh_keys("job-banner", "xcl-925934a5-b24e061c", interactive=True)
    scripts = [c[-1] for c in seen if isinstance(c, list) and "/etc/motd.xcelsior" in c[-1]]
    assert len(scripts) == 1, "one setup exec writes the banner"
    return scripts[0]


def _files(script: str, root: Path) -> dict[str, str]:
    """Run the file-writing half of the setup against `root` instead of /etc."""
    head = script.split(" ; ")[0].replace("/etc/", f"{root}/")
    RUN([SH, "-c", head], check=True)
    return {p.name: p.read_text() for p in root.iterdir() if p.is_file()}


@pytest.fixture
def written(monkeypatch, tmp_path) -> tuple[str, Path]:
    (tmp_path / "profile.d").mkdir()
    script = _setup_command(monkeypatch)
    _files(script, tmp_path)
    return script, tmp_path


def _greeting(root: Path, *, env_cols: str | None, shell: str) -> list[str]:
    profile = (root / "profile.d" / "xcelsior.sh").read_text().replace("/etc/", f"{root}/")
    # No tty here, so the profile's `[ -t 1 ]` guard is lifted; stty then has
    # no size to give, as on a PTY the browser has not resized yet.
    profile = profile.replace("[ -t 1 ] && ", "")
    env = {"PATH": "/usr/bin:/bin"}
    if env_cols is not None:
        env["XCELSIOR_TERM_COLS"] = env_cols
    out = RUN(
        [shell, "-c", f'. /dev/stdin; echo "left:${{XCELSIOR_TERM_COLS-unset}}"'],
        input=profile,
        capture_output=True,
        text=True,
        env=env,
        stdin=None,
        check=True,
    ).stdout
    return [ANSI.sub("", line) for line in out.splitlines()]


@pytest.mark.skipif(not (SH and BASH), reason="needs sh and bash")
@pytest.mark.parametrize("shell", ["sh", "bash"])
def test_a_phone_width_gets_the_banner_that_fits(written, shell):
    _, root = written
    lines = _greeting(root, env_cols="36", shell=shutil.which(shell) or shell)
    banner = [
        line for line in lines if line.strip() and not line.startswith(("Last login", "left:"))
    ]
    assert any("Xcelsior - GPU compute, on demand" in line for line in banner)
    assert not any("+---" in line for line in banner), "no box on a phone"
    # Every fixed line fits 36 columns (the GPU line holds nvidia-smi's answer,
    # here empty because there is no GPU in the test).
    assert max(len(line) for line in banner) <= 36, banner
    assert lines[-1] == "left:unset", "the width hint does not outlive the greeting"


@pytest.mark.skipif(not (SH and BASH), reason="needs sh and bash")
@pytest.mark.parametrize("cols", ["80", "120", None])
def test_a_desktop_or_unknown_width_keeps_the_box(written, cols):
    _, root = written
    lines = _greeting(root, env_cols=cols, shell=BASH or "bash")
    assert any("+-------" in line for line in lines)
    assert any("| Xcelsior - GPU compute, on demand" in line for line in lines)


def test_the_narrow_banner_is_written_beside_the_box(written):
    script, root = written
    assert "/etc/motd.xcelsior.narrow" in script
    assert (root / "motd.xcelsior.narrow").exists()
    assert "+---" not in (root / "motd.xcelsior.narrow").read_text()


@pytest.mark.skipif(not SH, reason="needs sh")
class TestGroupNames:
    def _run(self, tmp_path, ids: str, group_file: str, names: dict[int, str]) -> str:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "id").write_text(f"#!/bin/sh\necho {ids}\n")
        (bin_dir / "id").chmod(0o755)
        group = tmp_path / "group"
        group.write_text(group_file)
        script = worker_agent._name_supplementary_groups_script(names).replace(
            "/etc/group", str(group)
        )
        RUN(
            [SH, "-c", script],
            check=True,
            env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
        )
        return group.read_text()

    def test_the_render_group_gets_the_hosts_name(self, tmp_path):
        out = self._run(tmp_path, "0 992", "root:x:0:\n", {992: "render", 44: "video"})
        assert out == "root:x:0:\nrender:x:992:\n"

    def test_a_name_the_image_already_uses_is_not_reused(self, tmp_path):
        out = self._run(tmp_path, "0 992", "root:x:0:\nrender:x:110:\n", {992: "render"})
        assert out.endswith("hostgid992:x:992:\n")

    def test_an_id_with_no_host_name_still_gets_one(self, tmp_path):
        assert self._run(tmp_path, "0 4242", "root:x:0:\n", {}).endswith("hostgid4242:x:4242:\n")

    def test_named_groups_are_left_alone_and_it_is_idempotent(self, tmp_path):
        before = "root:x:0:\nrender:x:992:\n"
        assert self._run(tmp_path, "0 992", before, {992: "render"}) == before

    def test_unsafe_host_names_are_not_put_in_the_shell(self):
        script = worker_agent._name_supplementary_groups_script({992: "ren;der", 44: "video"})
        assert "ren;der" not in script and "44:video" in script


def test_device_groups_are_read_from_the_device_nodes(monkeypatch):
    stats = {"/dev/dri/renderD128": 992, "/dev/dri/card1": 44, "/dev/nvidia0": 0}
    monkeypatch.setattr(
        "glob.glob",
        lambda pattern: [p for p in stats if p.startswith(pattern.rstrip("*"))],
    )
    monkeypatch.setattr(worker_agent.os, "stat", lambda p: SimpleNamespace(st_gid=stats[p]))
    monkeypatch.setattr(
        "grp.getgrgid", lambda gid: SimpleNamespace(gr_name={992: "render", 44: "video"}[gid])
    )
    assert worker_agent._device_group_names() == {44: "video", 992: "render"}
