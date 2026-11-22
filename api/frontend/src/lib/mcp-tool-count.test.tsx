import { describe, expect, it, vi } from "vitest";
import { act, renderHook, waitFor } from "@testing-library/react";
import releaseFacts from "./release-facts.json";
import { MCP_TOOL_COUNT, VERIFICATION_CHECK_COUNT, withMcpToolCount } from "./mcp-tool-count";
import { LocaleProvider, useLocale } from "./locale";
import { createTranslator } from "./i18n/server";
import { getAllPosts, getPostBySlug } from "./blog";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ refresh: vi.fn() }),
  usePathname: () => "/mcp",
}));

// release-facts.json is written from api/mcp/tool-surface.json; the Python test
// tests/test_release_facts.py holds the two together. This one holds every page
// to the file.
describe("MCP tool count", () => {
  it("is the published surface's count, with no other source", () => {
    expect(MCP_TOOL_COUNT).toBe(releaseFacts.mcpToolCount);
    expect(Number.isSafeInteger(MCP_TOOL_COUNT) && MCP_TOOL_COUNT > 0).toBe(true);
    expect(withMcpToolCount("{mcpToolCount} tools")).toBe(`${MCP_TOOL_COUNT} tools`);
  });

  it("reaches the English and French copy, server-rendered and client-side, and the blog", async () => {
    const n = String(MCP_TOOL_COUNT);
    const { result } = renderHook(() => useLocale(), { wrapper: LocaleProvider });
    expect(result.current.t("home.cmp_agents_x")).toMatch(new RegExp(`^${n} tools`));
    act(() => result.current.toggleLocale());
    await waitFor(() => expect(result.current.t("home.cmp_agents_x")).toMatch(new RegExp(`^${n} outils`)));
    expect(createTranslator("en")("home.cmp_agents_x")).toMatch(new RegExp(`^${n} tools`));
    expect(createTranslator("fr")("home.cmp_agents_x")).toMatch(new RegExp(`^${n} outils`));
    const post = getPostBySlug("mcp-server-release");
    expect(post?.title).toContain(`${n} Scoped Tools`);
    expect(post?.content).toContain(`${n} scoped customer tools`);
    expect(post?.content).not.toContain("{mcpToolCount}");
    expect(getAllPosts().find((entry) => entry.slug === post?.slug)?.title).toBe(post?.title);
  });
});

describe("verification check count", () => {
  it("is hosts/verification.py's count, and the comparison table quotes it", () => {
    expect(VERIFICATION_CHECK_COUNT).toBe(releaseFacts.verificationCheckCount);
    expect(Number.isSafeInteger(VERIFICATION_CHECK_COUNT) && VERIFICATION_CHECK_COUNT > 0).toBe(true);
    const n = String(VERIFICATION_CHECK_COUNT);
    expect(createTranslator("en")("home.cmp_verification_x")).toBe(`${n} checks`);
    expect(createTranslator("fr")("home.cmp_verification_x")).toBe(`${n} vérifications`);
  });
});
