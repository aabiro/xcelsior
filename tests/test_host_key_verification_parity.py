"""The human and the agent get the same host-key check, or neither is trusted.

Gate P2: *"Connection details in the instance view that match exactly what the
tool returns — same host, same fingerprint, same expiry. A human and an agent
looking at the same instance must see the same truth."*

They did not. `routes/instances.py` `_enrich_instance` has served
`host_key_fingerprint` since the host-key rollout and `open_instance_access`
returns it to the model with a verification command beside it — while the
instance page dropped the field at the TypeScript type boundary and rendered
`ssh root@host -p port` with nothing to check it against. An agent could verify
the host key; the person looking at the same instance could not.

`tests/test_ssh_host_is_one_value.py` already pins the *hostname* across four
surfaces. This pins the other half: the **command** each surface tells its
reader to run, and the **fingerprint shape** each surface will accept.

## Why the command and not the prose

The two surfaces address different readers and should not be forced into
identical sentences. The tool's null-state note ends *"Tell the user that,
rather than telling them to accept the key"* — an instruction to a model, which
would be nonsense rendered on a page. Asserting string equality across them
would either freeze the UI's wording into machine-directed prose or invite
someone to weaken the guard until it passed.

What must be identical is the thing that decides whether verification actually
works: `ssh-keyscan` against the same host and port, piped to `ssh-keygen -lf -`.
Drift there is silent and dangerous in a specific way — a user runs a check, it
prints a different value than the page shows, and the correct response to that
is indistinguishable from the response to a real interception.

## Why the fingerprint shape too

`host_key_fingerprint.py` accepts `SHA256:` + exactly 43 unpadded-base64
characters, and `mcp/src/tools/compute.ts` re-validates with its own regex
before handing the value to the model. Two independently written regexes for one
wire format is exactly the shape that drifts: loosen one and it publishes a
value the other would have refused, tighten one and it silently nulls a
fingerprint the platform did observe.
"""

from __future__ import annotations

import pathlib
import re

from tests._source_tree import iter_source_files, read_source, strip_ts_comments

ROOT = pathlib.Path(__file__).resolve().parent.parent

MCP_TOOL = ROOT / "mcp/src/tools/compute.ts"
UI_COMPONENT = ROOT / "frontend/src/components/instances/host-key-verification.tsx"
PY_VALIDATOR = ROOT / "host_key_fingerprint.py"

#: `ssh-keyscan` through the end of the pipe, however the source spells it.
_COMMAND_SPAN = re.compile(r"ssh-keyscan.*?ssh-keygen\s+-lf\s+-", re.DOTALL)

#: A JS concatenation seam: `" +` newline indent backtick, or the reverse.
#: Splitting a template across `+` is a formatting choice, not a difference in
#: what the user is told to type.
_JS_SEAM = re.compile(r"[\"`]\s*\+\s*[\"`]")

#: Any `${...}` interpolation, whatever the local variable happens to be named.
#: `sshPort` and `port` are the same hole in the same sentence.
_INTERPOLATION = re.compile(r"\$\{[^}]*\}")


def _command_shape(path: pathlib.Path) -> str:
    """The verification command with its interpolations blanked out."""
    assert path.exists(), f"{path.relative_to(ROOT)} is gone; fix the pin, don't delete it"
    text = strip_ts_comments(path.read_text(encoding="utf-8"))
    match = _COMMAND_SPAN.search(text)
    assert match, (
        f"no ssh-keyscan…ssh-keygen command found in {path.relative_to(ROOT)}. "
        "Either it was reworded — in which case both surfaces must move "
        "together — or that surface stopped telling anyone how to verify."
    )
    shape = _JS_SEAM.sub("", match.group(0))
    shape = _INTERPOLATION.sub("{}", shape)
    return " ".join(shape.split())


def test_the_extraction_actually_finds_a_command():
    """Calibration. A pattern that matched nothing would pass the real test."""
    shape = _command_shape(MCP_TOOL)
    assert shape.startswith("ssh-keyscan"), shape
    assert shape.endswith("ssh-keygen -lf -"), shape
    assert "{}" in shape, "interpolations vanished; the normaliser is too greedy"


def test_the_extraction_is_sensitive_to_a_real_change():
    """Calibration. Prove the normaliser does not blank out the difference."""
    assert _command_shape(MCP_TOOL) != _INTERPOLATION.sub(
        "{}", "ssh-keyscan -p ${p} ${h} | ssh-keygen -lf -"
    ), "a command missing 2>/dev/null compared equal; the normaliser eats too much"


def test_agent_and_human_are_told_to_run_the_same_check():
    tool = _command_shape(MCP_TOOL)
    ui = _command_shape(UI_COMPONENT)
    assert tool == ui, (
        "the MCP tool and the instance view tell their readers to run different "
        f"commands.\n  open_instance_access: {tool}\n  instance view:       {ui}\n"
        "A user who runs one and compares against the other gets a mismatch "
        "whose correct response is indistinguishable from the response to a "
        "real interception."
    )


def test_the_command_still_silences_keyscan_progress_output():
    """`ssh-keyscan` writes progress to stderr; it interleaves with the value."""
    assert "2>/dev/null" in _command_shape(MCP_TOOL)


def test_both_languages_accept_exactly_the_same_fingerprint_shape():
    ts = MCP_TOOL.read_text(encoding="utf-8")
    py = PY_VALIDATOR.read_text(encoding="utf-8")

    ts_match = re.search(r"/\^SHA256:\[([^\]]+)\]\{(\d+)\}\$/", ts)
    assert ts_match, (
        "no anchored SHA256 fingerprint regex in mcp/src/tools/compute.ts. If "
        "validation moved, re-point this pin; if it was dropped, the tool now "
        "publishes whatever the API sends without re-checking it."
    )
    py_match = re.search(r'r"\^SHA256:\[([^\]]+)\]\{(\d+)\}\$"', py)
    assert py_match, "no anchored SHA256 fingerprint regex in host_key_fingerprint.py"

    assert ts_match.group(2) == py_match.group(2) == "43", (
        f"fingerprint length disagrees: TypeScript {ts_match.group(2)}, "
        f"Python {py_match.group(2)}. `ssh-keygen -lf` prints exactly 43 "
        "characters of unpadded base64; anything else can never match."
    )
    assert set(ts_match.group(1)) == set(py_match.group(1)), (
        f"character classes disagree: TypeScript [{ts_match.group(1)}], "
        f"Python [{py_match.group(1)}]. One of them will refuse a fingerprint "
        "the other accepted."
    )


def test_the_browser_never_stores_a_terminal_ticket():
    """P2's *"same expiry"* holds by construction — keep it that way.

    `open_instance_access` returns `expires_in_seconds` because the agent
    receives a ticket it must carry to a websocket itself. The browser does not:
    `WebTerminal` mints a ticket and opens the socket with it in the same
    function, and mints a fresh one on every reconnect. It reads `expires_in`
    and discards it. So there is no browser-side expiry that can disagree with
    the tool's — the human never holds a ticket between mint and use.

    That is a property, not an oversight, and the tempting "fix" would break it:
    displaying a countdown to match the tool means holding the ticket to count
    down *from*, which turns a credential consumed on arrival into one sitting
    in storage with a clock on it. A ticket in `localStorage` also outlives the
    tab, which is exactly what single-use was meant to prevent.

    Checked as storage rather than as shape, because that is the part with a
    blast radius.
    """
    offenders: list[str] = []
    sources = [
        *iter_source_files("*.ts", include_prefixes=("frontend/src/",)),
        *iter_source_files("*.tsx", include_prefixes=("frontend/src/",)),
    ]
    assert sources, "no frontend sources found; the include prefix is wrong"
    for path, _rel in sources:
        text = strip_ts_comments(read_source(path))
        for line in text.splitlines():
            if "ticket" not in line.lower():
                continue
            if re.search(r"(localStorage|sessionStorage)\.(set|get)Item", line):
                offenders.append(f"{path.relative_to(ROOT)}: {line.strip()[:90]}")
    assert not offenders, (
        "a terminal ticket is being written to or read from browser storage, "
        "which outlives the tab and defeats single-use:\n  " + "\n  ".join(offenders)
    )


def test_neither_surface_tells_anyone_to_accept_an_unverifiable_key():
    """The null state is honest on both sides, or it is worse than silence.

    A missing fingerprint is permanent for whole classes of instance. The one
    thing neither surface may do is resolve that by suggesting the key be
    trusted anyway — that converts an unverifiable connection into a verified-
    feeling one, which is the exact failure the fingerprint exists to prevent.
    """
    for path in (MCP_TOOL, UI_COMPONENT):
        text = strip_ts_comments(path.read_text(encoding="utf-8")).lower()
        for phrase in (
            "accept the key anyway",
            "safe to accept",
            "just accept",
            "accept it anyway",
        ):
            assert phrase not in text, (
                f"{path.relative_to(ROOT)} suggests accepting an unverified host key ({phrase!r})."
            )
