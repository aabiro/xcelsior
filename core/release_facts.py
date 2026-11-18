"""Figures and versions the product quotes in more than one place.

Each value is read from the one file that decides it, so a copy cannot drift:

- the MCP tool count: ``api/mcp/tool-surface.json``, which ``npm run
  surface:update`` writes from the server's code;
- the Xcel AI tool count: the assistant's own tool list;
- a package version: that package's manifest;
- the wizard release every instruction pins, and the SPIRE release:
  ``install.sh``'s pin lines.

Code reads these functions. Text that is served (``llms.txt``) holds
placeholders such as ``{mcpToolCount}`` and is rendered when it is read. Files
that cannot read anything at run time (the frontend build, the registry's
``server.json``) get their copy written by
``api/scripts/generate_release_facts.py``, and a test fails when a copy is not
what that script would write.

To fill the placeholders in a document by hand (a launch draft, a deck)::

    python3 core/release_facts.py render path/to/draft.md
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]

MCP_SURFACE = ROOT / "api" / "mcp" / "tool-surface.json"
API_PROJECT = ROOT / "pyproject.toml"
MCP_SERVER_PACKAGE = ROOT / "api" / "mcp" / "package.json"
MCP_CLI_PACKAGE = ROOT / "api" / "mcp-cli" / "package.json"
INSTALLER = ROOT / "api" / "scripts" / "install.sh"

_WIZARD_PIN = re.compile(r'^readonly WIZARD_PACKAGE="(@xcelsior-gpu/wizard@\d+\.\d+\.\d+)"$', re.M)


@lru_cache(maxsize=None)
def _manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mcp_tool_count() -> int:
    """Tools the hosted MCP server publishes to a customer."""
    return len(_manifest(MCP_SURFACE)["tools"])


def xcel_ai_tool_count() -> int:
    """Tools the in-app assistant can call.

    Imported when asked for: the assistant module is large and imports the
    database layer, which a caller that only wants a version should not pay for.
    """
    from ai_assistant import _build_tools

    return len(_build_tools())


@lru_cache(maxsize=None)
def api_version() -> str:
    with API_PROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]["version"]


def package_version(manifest: Path) -> str:
    return _manifest(manifest)["version"]


@lru_cache(maxsize=None)
def wizard_package() -> str:
    """The provider setup wizard every instruction pins: the release install.sh runs.

    Not ``api/wizard/package.json``: the source tree may already carry the next,
    unpublished version, and a pin has to name one people can install.
    install.sh needs the literal anyway (it runs on a provider's host, which has
    no repository), so that line is where the released version is recorded.
    """
    match = _WIZARD_PIN.search(INSTALLER.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"{INSTALLER} no longer pins the wizard to a release")
    return match.group(1)


@lru_cache(maxsize=None)
def spire_version() -> str:
    """The SPIRE release: install.sh pins the host agent's tarball (with its
    checksums), and the server images and the rollout script follow it."""
    match = re.search(
        r'^SPIRE_VERSION="(\d+\.\d+\.\d+)"$', INSTALLER.read_text(encoding="utf-8"), re.M
    )
    if match is None:
        raise RuntimeError(f"{INSTALLER} no longer pins SPIRE")
    return match.group(1)


#: Placeholder -> the function that fills it. Only the ones a text uses are
#: computed, so rendering a document that names no assistant count never
#: imports the assistant.
PLACEHOLDERS: dict[str, Callable[[], object]] = {
    "{mcpToolCount}": mcp_tool_count,
    "{xcelAiToolCount}": xcel_ai_tool_count,
    "{wizardPackage}": wizard_package,
    "{apiVersion}": api_version,
}

_PLACEHOLDER = re.compile(r"\{[A-Za-z]+\}")


def render(text: str) -> str:
    """Fill every known placeholder in ``text``; leave other braces alone."""
    for token in set(_PLACEHOLDER.findall(text)) & PLACEHOLDERS.keys():
        text = text.replace(token, str(PLACEHOLDERS[token]()))
    return text


def frontend_facts() -> dict:
    """What the Next.js build reads. Its build context is ``api/frontend`` only."""
    return {
        "$comment": (
            "Written by api/scripts/generate_release_facts.py from the files that "
            "decide each value (core/release_facts.py names them). Do not edit: "
            "regenerate, and tests/test_release_facts.py fails until you do."
        ),
        "mcpToolCount": mcp_tool_count(),
        "wizardPackage": wizard_package(),
    }


def _main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "render":
        sys.stdout.write(render(Path(argv[1]).read_text(encoding="utf-8")))
        return 0
    sys.stderr.write("usage: python3 core/release_facts.py render FILE\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
