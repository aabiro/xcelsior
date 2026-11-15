import { afterEach, describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ refresh: vi.fn() }),
  usePathname: () => "/mcp",
}));

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
  localStorage.clear();
});

describe("configured MCP tool count", () => {
  it("uses the override in the English and French website copy, SSR, and blog", async () => {
    vi.stubEnv("NEXT_PUBLIC_MCP_TOOL_COUNT", "101");
    vi.resetModules();
    const { MCP_TOOL_COUNT } = await import("./mcp-tool-count");
    const { LocaleProvider, useLocale } = await import("./locale");
    const { createTranslator } = await import("./i18n/server");
    const { getAllPosts, getPostBySlug } = await import("./blog");
    expect(MCP_TOOL_COUNT).toBe(101);
    const { result } = renderHook(() => useLocale(), { wrapper: LocaleProvider });
    expect(result.current.t("home.cmp_agents_x")).toMatch(/^101 tools/);
    act(() => result.current.toggleLocale());
    await waitFor(() => expect(result.current.t("home.cmp_agents_x")).toMatch(/^101 outils/));
    expect(createTranslator("en")("home.cmp_agents_x")).toMatch(/^101 tools/);
    expect(createTranslator("fr")("home.cmp_agents_x")).toMatch(/^101 outils/);
    const post = getPostBySlug("mcp-server-release");
    expect(post?.title).toContain("101 Scoped Tools");
    expect(post?.content).toContain("101 scoped customer tools");
    expect(post?.content).not.toContain("{mcpToolCount}");
    expect(getAllPosts().find((entry) => entry.slug === post?.slug)?.title).toBe(post?.title);
  });

  it.each([undefined, "", "  "])("uses the default when the override is %s", async (value) => {
    vi.stubEnv("NEXT_PUBLIC_MCP_TOOL_COUNT", value);
    vi.resetModules();
    const { DEFAULT_MCP_TOOL_COUNT, MCP_TOOL_COUNT } = await import("./mcp-tool-count");
    expect(MCP_TOOL_COUNT).toBe(DEFAULT_MCP_TOOL_COUNT);
  });

  it.each(["0", "-1", "1.5", "invalid"])("rejects the invalid count %s", async (value) => {
    vi.stubEnv("NEXT_PUBLIC_MCP_TOOL_COUNT", value);
    vi.resetModules();
    await expect(import("./mcp-tool-count")).rejects.toThrow("must be a positive integer");
  });
});
