import { describe, expect, it } from "vitest";
import {
  MCP_CONNECTOR_URL,
  MCP_RESOURCE,
  configJson,
  configPath,
  mcpHealthUrl,
  mcpUrl,
  oneClickInstalls,
} from "./mcp";

describe("connector URL", () => {
  it("is the exact URL a user pastes, path included", () => {
    // A connector echoes this back as its RFC 8707 `resource` parameter, so
    // the origin alone is a different identifier and gets rejected.
    expect(MCP_CONNECTOR_URL).toBe("https://mcp.xcelsior.ca/mcp");
    expect(MCP_RESOURCE).toBe(MCP_CONNECTOR_URL);
  });

  it("substitutes the local server only when actually on localhost", () => {
    // The test DOM is localhost, so this asserts the dev branch here and the
    // production branch by construction — there are only two.
    expect(window.location.hostname).toBe("localhost");
    expect(mcpUrl()).toBe("http://localhost:8770/mcp");
    expect(mcpHealthUrl()).toBe("http://localhost:8770/mcp/health");
  });

  it("never probes health against the connector host", () => {
    // mcp.xcelsior.ca serves no CORS headers on purpose; a browser fetch there
    // would be blocked before reaching us and would read as an outage.
    expect(mcpHealthUrl()).not.toContain("mcp.xcelsior.ca");
  });
});

describe("one-click installs", () => {
  const installs = oneClickInstalls();

  it("covers the clients the plan names", () => {
    expect(installs.map((i) => i.id).sort()).toEqual(
      ["claude-code", "copilot-cli", "cursor", "npx", "vscode"],
    );
  });

  it("never puts a credential in a link or a command", () => {
    // A deep link with a token in it is a token in a URL bar, a bookmark, a
    // browser history, and whatever chat it gets pasted into.
    for (const install of installs) {
      const payload = `${install.href ?? ""} ${install.command ?? ""}`;
      expect(payload, install.id).not.toMatch(/Bearer|token|secret|xoa_|xcel_ai_/i);
    }
  });

  it("gives every entry exactly one way to run it", () => {
    for (const install of installs) {
      expect(Boolean(install.href) !== Boolean(install.command), install.id).toBe(true);
      expect(install.label.length, install.id).toBeGreaterThan(0);
    }
  });

  it("points every command at the canonical connector URL", () => {
    for (const install of installs.filter((i) => i.command)) {
      if (install.id === "npx") continue; // stdio takes its URL from the package
      expect(install.command, install.id).toContain(MCP_CONNECTOR_URL);
    }
  });

  it("encodes the Cursor deep link so the config survives the URL", () => {
    const cursor = installs.find((i) => i.id === "cursor")!;
    expect(cursor.href).toMatch(/^cursor:\/\/anysphere\.cursor-deeplink\/mcp\/install\?/);
    const config = new URL(cursor.href!).searchParams.get("config")!;
    expect(JSON.parse(atob(config))).toEqual({ url: MCP_CONNECTOR_URL });
  });

  it("encodes the VS Code deep link as the config object it expects", () => {
    const vscode = installs.find((i) => i.id === "vscode")!;
    const raw = decodeURIComponent(vscode.href!.split("?")[1]);
    expect(JSON.parse(raw)).toEqual({
      name: "xcelsior", type: "http", url: MCP_CONNECTOR_URL,
    });
  });
});

describe("hand-written config snippets", () => {
  it("uses each client's own key name", () => {
    // VS Code reads `servers`; Cursor and Claude read `mcpServers`. Getting
    // this wrong produces a config that loads and silently does nothing.
    expect(JSON.parse(configJson("vscode"))).toHaveProperty("servers.xcelsior");
    expect(JSON.parse(configJson("claude"))).toHaveProperty("mcpServers.xcelsior");
    expect(JSON.parse(configJson("cursor"))).toHaveProperty("mcpServers.xcelsior");
  });

  it("points every snippet at one connector URL", () => {
    for (const agent of ["cursor", "claude", "vscode", "github"]) {
      expect(configJson(agent), agent).toContain(mcpUrl());
    }
  });

  it("names where each config file lives", () => {
    expect(configPath("vscode")).toBe(".vscode/mcp.json");
    expect(configPath("cursor")).toBe("~/.cursor/mcp.json");
    expect(configPath("claude")).toContain(".mcp.json");
  });
});
