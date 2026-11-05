import { describe, it, expect } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import type { XcelsiorApiClient } from "../../src/client/api.js";
import { createMcpServer } from "../../src/server.js";
import { ICON_PATH, httpIcons, readIcon, stdioIcons } from "../../src/icon.js";

/**
 * VS Code accepts an HTTP server's icon only from that server's own authority
 * and a stdio server's only from `file:`. An icon pointing anywhere else is
 * silently dropped, so the URL shape is the contract, not a detail.
 */
describe("server icon", () => {
  it("ships the PNG the server advertises", () => {
    const icon = readIcon();
    expect(icon).not.toBeNull();
    expect(icon!.subarray(0, 8)).toEqual(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  });

  it("serves the hosted icon from the connector URL's own origin", () => {
    expect(httpIcons("https://mcp.xcelsior.ca/mcp")).toEqual([
      { src: `https://mcp.xcelsior.ca${ICON_PATH}`, mimeType: "image/png", sizes: ["512x512"] },
    ]);
  });

  it("points the stdio icon at the file on disk", () => {
    const [icon] = stdioIcons();
    expect(icon.src).toMatch(/^file:\/\/.*\/assets\/icon-gradient-512\.png$/);
  });

  it("advertises the icons in the initialize response", async () => {
    const icons = httpIcons("https://mcp.xcelsior.ca/mcp");
    const server = createMcpServer({} as XcelsiorApiClient, undefined, "streamable_http", "customer", {}, icons);
    const client = new Client({ name: "icon-test", version: "1" });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    expect(client.getServerVersion()?.icons).toEqual(icons);
    await client.close();
  });

  it("describes itself fully, with a title short enough for VS Code's tool prefix", async () => {
    const server = createMcpServer({} as XcelsiorApiClient);
    const client = new Client({ name: "icon-test", version: "1" });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    const info = client.getServerVersion()!;
    expect(info).toMatchObject({ name: "xcelsior-mcp", title: "Xcelsior", websiteUrl: "https://xcelsior.ca/mcp" });
    expect(info.description).toBeTruthy();
    // VS Code: "mcp_" + title lowercased, cut to 13 chars, + "_" (MaxPrefixLen 18).
    expect(info.title!.toLowerCase().replace(/[^a-z0-9_.-]+/g, "_").length).toBeLessThanOrEqual(13);
    await client.close();
  });

  it("omits icons rather than advertising an empty list", async () => {
    const server = createMcpServer({} as XcelsiorApiClient);
    const client = new Client({ name: "icon-test", version: "1" });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
    expect(client.getServerVersion()).not.toHaveProperty("icons");
    await client.close();
  });
});
