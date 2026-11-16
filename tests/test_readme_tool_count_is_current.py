"""Keep the configurable marketing count aligned with the customer MCP surface."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "api" / "frontend"


def test_default_and_example_tool_count_match_the_published_surface() -> None:
    published = len(json.loads((ROOT / "api/mcp/tool-surface.json").read_text())["tools"])
    assert published > 0
    config = (FRONTEND / "src/lib/mcp-tool-count.ts").read_text()
    default = re.search(r"DEFAULT_MCP_TOOL_COUNT = (\d+)", config)
    example = re.search(
        r"^NEXT_PUBLIC_MCP_TOOL_COUNT=(\d+)$", (ROOT / ".env.example").read_text(), re.M
    )
    assert default and example
    assert int(default[1]) == int(example[1]) == published


def test_translations_share_the_configured_tool_count() -> None:
    for locale in ("en", "fr"):
        for suffix in ("", "-public"):
            text = (FRONTEND / f"src/lib/i18n/{locale}{suffix}.ts").read_text()
            assert re.search(r'"home\.cmp_agents_x":\s*"\{mcpToolCount\}', text)


def test_static_docs_do_not_duplicate_the_current_mcp_total() -> None:
    for relative in ("README.md", "api/fern/pages/mcp.mdx", "api/fern/pages/llm-integration.mdx"):
        text = (ROOT / relative).read_text()
        assert not re.search(
            r"\b\d+[ -](?:MCP tools|customer tools|tool reference)\b", text
        ), relative
    assert "NEXT_PUBLIC_MCP_TOOL_COUNT" in (ROOT / "README.md").read_text()


def test_docker_build_passes_the_tool_count_to_nextjs() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "NEXT_PUBLIC_MCP_TOOL_COUNT: ${NEXT_PUBLIC_MCP_TOOL_COUNT:-}" in compose
    dockerfile = (FRONTEND / "Dockerfile").read_text()
    assert "ARG NEXT_PUBLIC_MCP_TOOL_COUNT=" in dockerfile
    assert "ENV NEXT_PUBLIC_MCP_TOOL_COUNT=${NEXT_PUBLIC_MCP_TOOL_COUNT}" in dockerfile
