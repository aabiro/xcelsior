"""Every count and version the product quotes comes from one file.

`core/release_facts.py` names the file that decides each value. Code reads it;
served text renders placeholders from it; the few files that cannot read
anything at run time get a copy from `api/scripts/generate_release_facts.py`,
checked here. What remains (base-image and SPIRE pins in Dockerfiles and compose
files, values drawn into images) is held to its source by a test below.

How this came about: `llms.txt` told agents 87 MCP tools for a week after the
surface reached 89, while the website said 89. The website's number was right
only because someone remembered to edit a default in TypeScript and its twin in
`.env.example`, and an environment override could still have put a third number
on the page. The MCP server reported 2.0.0 while its registry entry said 2.0.2.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import tomllib
from pathlib import Path

import release_facts as facts

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "api" / "frontend"
LAUNCH_KIT = FRONTEND / "public" / "brand-system" / "assets" / "social" / "mcp-launch"


def _generator():
    spec = importlib.util.spec_from_file_location(
        "generate_release_facts", ROOT / "api" / "scripts" / "generate_release_facts.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tracked(*pathspec: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", *pathspec],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


# ── the sources ──────────────────────────────────────────────────────


def test_each_fact_reads_the_file_that_decides_it() -> None:
    surface = json.loads((ROOT / "api/mcp/tool-surface.json").read_text())
    assert facts.mcp_tool_count() == len(surface["tools"]) > 0
    with (ROOT / "pyproject.toml").open("rb") as fh:
        assert facts.api_version() == tomllib.load(fh)["project"]["version"]
    server = json.loads((ROOT / "api/mcp/package.json").read_text())
    assert facts.package_version(facts.MCP_SERVER_PACKAGE) == server["version"]
    installer = (ROOT / "api/scripts/install.sh").read_text()
    assert f'readonly WIZARD_PACKAGE="{facts.wizard_package()}"' in installer
    assert f'SPIRE_VERSION="{facts.spire_version()}"' in installer


def test_render_fills_known_placeholders_and_leaves_other_braces() -> None:
    text = "{mcpToolCount} tools, {wizardPackage}, api {apiVersion}, and {notAFact}"
    assert facts.render(text) == (
        f"{facts.mcp_tool_count()} tools, {facts.wizard_package()}, "
        f"api {facts.api_version()}, and {{notAFact}}"
    )


def test_llms_txt_holds_placeholders_and_renders_them_all() -> None:
    raw = (ROOT / "llms.txt").read_text()
    for token in ("{mcpToolCount}", "{xcelAiToolCount}", "{wizardPackage}"):
        assert token in raw, token
    rendered = facts.render(raw)
    assert not re.search(r"\{(?:mcpToolCount|xcelAiToolCount|wizardPackage|apiVersion)\}", rendered)
    assert f"{facts.mcp_tool_count()} customer tools" in rendered
    assert f"{facts.xcel_ai_tool_count()} tools across" in rendered


# ── the written copies ───────────────────────────────────────────────


def test_generate_release_facts_copies_are_current() -> None:
    """The currency gate for `api/scripts/generate_release_facts.py`."""
    stale = [str(p.relative_to(ROOT)) for p in _generator().stale()]
    assert not stale, f"stale copies: {stale}. Run `python3 api/scripts/generate_release_facts.py`."


def test_the_registry_entry_names_the_versions_in_the_manifests() -> None:
    entry = json.loads((ROOT / "api/mcp/server.json").read_text())
    cli = json.loads((ROOT / "api/mcp-cli/package.json").read_text())
    assert entry["version"] == facts.package_version(facts.MCP_SERVER_PACKAGE)
    assert [p["version"] for p in entry["packages"] if p["identifier"] == cli["name"]] == [
        cli["version"]
    ]


# ── no hand-typed copies ─────────────────────────────────────────────


def test_no_tracked_file_types_either_tool_count() -> None:
    """A count followed by "tools" is a copy; it should be a placeholder or a read.

    The plan doc keeps a dated count history, and changelogs record what
    each release had; neither claims to be current.
    """
    history = {"api/docs/mcp-agent-native-implementation-plan.md"}
    counts = "|".join(sorted({str(facts.mcp_tool_count()), str(facts.xcel_ai_tool_count())}))
    out = subprocess.run(
        [
            "git",
            "grep",
            "-nIE",
            rf"\b({counts})[ -](MCP |customer |scoped |published |agent )?(tools|outils|Tools|Outils)\b",
            "--",
            ".",
            ":!**/package-lock.json",
            ":!**/CHANGELOG.md",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    hits = [line for line in out.stdout.splitlines() if line.split(":", 1)[0] not in history]
    assert not hits, "typed tool counts:\n" + "\n".join(hits)


def test_the_website_reads_the_count_and_has_no_override() -> None:
    config = (FRONTEND / "src/lib/mcp-tool-count.ts").read_text()
    assert 'from "./release-facts.json"' in config
    assert not re.search(r"=\s*\d+;", config), "a typed default is a second source"
    for path in ("docker-compose.yml", ".env.example", "README.md", "api/frontend/Dockerfile"):
        assert "NEXT_PUBLIC_MCP_TOOL_COUNT" not in (ROOT / path).read_text(), path


def test_translations_use_the_count_placeholder() -> None:
    for locale in ("en", "fr"):
        for suffix in ("", "-public"):
            text = (FRONTEND / f"src/lib/i18n/{locale}{suffix}.ts").read_text()
            assert re.search(r'"home\.cmp_agents_x":\s*"\{mcpToolCount\}', text)


def test_static_docs_do_not_quote_a_tool_count() -> None:
    """Fern and the README cannot render placeholders, so they name no count."""
    for relative in ("README.md", "api/fern/pages/mcp.mdx", "api/fern/pages/llm-integration.mdx"):
        text = (ROOT / relative).read_text()
        assert not re.search(
            r"\b\d+[ -](?:MCP tools|customer tools|tool reference|tools across)\b", text
        ), relative


def test_version_literals_live_only_in_their_manifests() -> None:
    api = (ROOT / "api/api.py").read_text()
    assert "version=api_version()" in api
    assert not re.search(r'(?:version=|get_tracer\("xcelsior\.api",\s*)"\d+\.\d+\.\d+"', api)
    assert not re.search(r"\bv\d+\.\d+\.\d+\b", (ROOT / "core/cli.py").read_text().splitlines()[1])
    for relative in (
        "api/mcp/src/server.ts",
        "api/mcp/src/index.ts",
        "api/mcp/src/observability/tracing.ts",
        "api/mcp/src/protocol-smoke.ts",
    ):
        text = (ROOT / relative).read_text()
        assert "PACKAGE_VERSION" in text, relative
        assert not re.search(
            r'version:\s*"\d+\.\d+\.\d+"|getTracer\("[^"]+",\s*"\d', text
        ), relative
    assert not re.search(r'protocol:\s*"\d{4}-', (ROOT / "api/mcp/src/index.ts").read_text())


# ── copies that cannot read: held to the source ──────────────────────


def test_launch_kit_images_show_the_current_count() -> None:
    """Pixels cannot read a file. When this fails, re-cut the images named in
    baked-in.json from the design canvas, then update the value there."""
    baked = json.loads((LAUNCH_KIT / "baked-in.json").read_text())
    tools = json.loads((ROOT / "api/mcp/tool-surface.json").read_text())["tools"]
    assert baked["mcpToolCount"]["value"] == len(tools), baked["mcpToolCount"]["images"]
    read_only = sum(1 for t in tools if t["annotations"].get("readOnlyHint"))
    confirm = sum(1 for t in tools if t["annotations"].get("destructiveHint"))
    split = baked["toolsListSplit"]
    assert (split["readOnly"], split["previewThenConfirm"], split["otherActions"]) == (
        read_only,
        confirm,
        len(tools) - read_only - confirm,
    ), split["images"]
    for section in ("mcpToolCount", "toolsListSplit", "capturedOnly"):
        for image in baked[section]["images"]:
            assert (LAUNCH_KIT / f"{image}.png").is_file(), image


def test_one_node_base_image_for_every_node_build() -> None:
    pins = {}
    for relative in ("api/frontend/Dockerfile", "api/mcp/Dockerfile"):
        text = (ROOT / relative).read_text()
        match = re.search(r"^ARG NODE_IMAGE=(\S+)$", text, re.M)
        assert match, relative
        assert "@sha256:" in match[1], f"{relative}: pin the digest"
        froms = re.findall(r"^FROM (\S+)", text, re.M)
        assert froms and all(f == "${NODE_IMAGE}" for f in froms), (relative, froms)
        pins[relative] = match[1]
    assert len(set(pins.values())) == 1, pins


def test_every_spire_reference_names_the_installers_release() -> None:
    version = facts.spire_version()
    seen = 0
    for relative in _tracked("api/infra/**", "docker-compose*.yml", "api/scripts/**"):
        text = (ROOT / relative).read_text(errors="replace")
        for found in re.findall(r"spiffe/spire-(?:server|agent):([0-9][^\s\"']*)", text):
            seen += 1
            assert found == version, f"{relative}: SPIRE {found}, install.sh pins {version}"
    assert seen >= 3
    rollout = (ROOT / "api/infra/spire/rollout-enrollment.sh").read_text()
    assert "scripts/install.sh" in rollout and "spire-server:${SPIRE_VERSION" in rollout


def test_ci_service_images_agree() -> None:
    images = set()
    for relative in _tracked(".github/workflows/*.yml"):
        images.update(re.findall(r"image:\s*(postgres:\S+)", (ROOT / relative).read_text()))
    assert len(images) == 1, images


def test_the_desktop_app_has_one_version() -> None:
    desktop = ROOT / "api" / "desktop"
    npm = json.loads((desktop / "package.json").read_text())["version"]
    tauri = json.loads((desktop / "src-tauri" / "tauri.conf.json").read_text())
    assert tauri["version"] == "../package.json", "Tauri reads the version from package.json"
    with (desktop / "src-tauri" / "Cargo.toml").open("rb") as fh:
        assert tomllib.load(fh)["package"]["version"] == npm
