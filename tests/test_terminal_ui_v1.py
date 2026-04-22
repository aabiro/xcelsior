"""Terminal UI v1 — frozen snapshot.

This test locks in the exact text of the interactive-instance stdout UI
(the init-script banner, the SSH-setup status notes, and the six final
'Terminal ready …' summary lines) that the user explicitly approved on
2026-04-22 as baseline v1.

If you intend to change any of these strings, DO NOT just edit this test
to make it pass — bump TERMINAL_UI_VERSION to v2 in both worker_agent.py
and here, and explicitly document the change. The whole point of this
file is to make an accidental whitespace / emoji / copy change impossible
without a deliberate version bump.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKER = Path(__file__).parent.parent / "worker_agent.py"


@pytest.fixture(scope="module")
def source() -> str:
    return WORKER.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Banner emitted by the container's init-script (what the user sees first
#    in the web terminal / log stream when an interactive instance boots).
# ---------------------------------------------------------------------------

BANNER_LINES_V1 = (
    "echo '[xcelsior] Initialising interactive instance…';",
    "echo '[xcelsior] GPU:' $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not detected');",
    # \\K below is the literal 2-char sequence backslash-K as it appears in
    # the worker_agent.py source (Python would compile it to a single \K).
    r"echo '[xcelsior] CUDA:' $(nvcc --version 2>/dev/null | grep -oP 'release \\K[0-9.]+' || echo 'N/A');",
    "echo '[xcelsior] Python:' $(python3 --version 2>/dev/null || python --version 2>/dev/null || echo 'N/A');",
    "echo '[xcelsior] PyTorch:' $(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo 'N/A');",
    "echo '[xcelsior] Setting up SSH…';",
)


def test_banner_init_script_is_locked_v1(source: str) -> None:
    """The interactive init-script's six echo lines are frozen at v1."""
    for line in BANNER_LINES_V1:
        assert line in source, (
            f"Terminal UI v1 banner line missing from worker_agent.py:\n"
            f"  {line!r}\n"
            "Bump TERMINAL_UI_VERSION if you intentionally changed this."
        )


# ---------------------------------------------------------------------------
# 2. The six possible final summary lines emitted once SSH setup finishes.
#    Exact strings — any emoji/casing/wording change must be deliberate.
# ---------------------------------------------------------------------------

FINAL_SUMMARY_V1 = (
    # (template, is_f_string)
    ('[xcelsior] Terminal ready — SSH enabled ({len(keys)} key(s))', True),
    ('[xcelsior] Terminal ready — add SSH keys at xcelsior.ca/dashboard/settings to enable direct SSH', False),
    ('[xcelsior] Terminal ready — web terminal only (sshd failed to start)', False),
    ('[xcelsior] Terminal ready — web terminal only (image has no sshd)', False),
    ('[xcelsior] SSH setup timed out — web terminal still works', False),
)


def test_final_summary_lines_are_locked_v1(source: str) -> None:
    for template, is_f in FINAL_SUMMARY_V1:
        needle = ('f"' + template + '"') if is_f else ('"' + template + '"')
        assert needle in source, (
            f"Terminal UI v1 final-summary string missing:\n  {needle!r}\n"
            "If this is intentional, bump TERMINAL_UI_VERSION."
        )


# ---------------------------------------------------------------------------
# 3. The 'Tip: add SSH public key' note and the SSH-daemon status notes —
#    these appear in the UI between banner and final line. Frozen at v1.
# ---------------------------------------------------------------------------

NOTE_STRINGS_V1 = (
    "Tip: add an SSH public key at Settings → SSH Keys to enable direct SSH (root@host:port) into this instance. The web terminal works without a key.",
    "Installing OpenSSH server in container (one-time setup)…",
    "OpenSSH server installed",
    "SSH daemon ready — connections accepted",
    "SSH daemon started (add keys to connect)",
)


def test_note_strings_are_locked_v1(source: str) -> None:
    for note in NOTE_STRINGS_V1:
        assert note in source, (
            f"Terminal UI v1 note string missing:\n  {note!r}\n"
            "If this is intentional, bump TERMINAL_UI_VERSION."
        )


# ---------------------------------------------------------------------------
# 4. Version marker — editing any locked string above without also bumping
#    this marker will leave the test suite pointing at the wrong baseline.
# ---------------------------------------------------------------------------

TERMINAL_UI_VERSION = "v1"


def test_version_marker_present(source: str) -> None:
    """worker_agent.py must carry a matching TERMINAL_UI_VERSION marker so a
    grep reveals which baseline is currently shipped."""
    m = re.search(r"TERMINAL_UI_VERSION\s*=\s*['\"]([^'\"]+)['\"]", source)
    assert m is not None, (
        "worker_agent.py must define TERMINAL_UI_VERSION = 'v1' (or later). "
        "This marker is what lets future edits prove they're a deliberate bump."
    )
    assert m.group(1) == TERMINAL_UI_VERSION, (
        f"Version mismatch: worker_agent.py says {m.group(1)!r}, test expects "
        f"{TERMINAL_UI_VERSION!r}. Bump both together."
    )
