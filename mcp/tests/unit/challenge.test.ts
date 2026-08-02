import { describe, expect, it } from "vitest";
import { buildWwwAuthenticate } from "../../src/auth/challenge.js";
import { acceptedAudiences, loadConfig } from "../../src/config.js";

const METADATA = "https://mcp.xcelsior.ca/.well-known/oauth-protected-resource";

describe("RFC 9728 challenge", () => {
  it("names the resource metadata URL on the no-credentials branch", () => {
    const header = buildWwwAuthenticate({ realm: "xcelsior", resourceMetadataUrl: METADATA });
    expect(header.startsWith("Bearer ")).toBe(true);
    expect(header).toContain(`resource_metadata="${METADATA}"`);
    expect(header).toContain('realm="xcelsior"');
    // RFC 6750 §3: the no-credentials branch carries no error code.
    expect(header).not.toContain("error=");
  });

  it("adds error and description only on the invalid-token branch", () => {
    const header = buildWwwAuthenticate({
      realm: "xcelsior",
      resourceMetadataUrl: METADATA,
      error: "invalid_token",
      errorDescription: "Bearer token invalid, expired, or bound to another resource.",
    });
    expect(header).toContain('error="invalid_token"');
    expect(header).toContain('error_description="Bearer token invalid');
  });

  it("cannot be split by a description containing quotes or newlines", () => {
    const header = buildWwwAuthenticate({
      realm: "xcelsior",
      resourceMetadataUrl: METADATA,
      error: "invalid_token",
      errorDescription: 'bad "quoted"\r\nX-Injected: yes',
    });
    expect(header).not.toContain("\r");
    expect(header).not.toContain("\n");
    expect(header).toContain('bad \\"quoted\\"');
  });
});

describe("resource identifier configuration", () => {
  const withEnv = <T>(env: Record<string, string | undefined>, run: () => T): T => {
    const saved = { ...process.env };
    Object.assign(process.env, env);
    for (const [key, value] of Object.entries(env)) if (value === undefined) delete process.env[key];
    try {
      return run();
    } finally {
      process.env = saved;
    }
  };

  it("defaults the canonical resource to the exact connector URL", () => {
    const config = withEnv({ XCELSIOR_MCP_RESOURCE_AUDIENCE: undefined }, loadConfig);
    expect(config.resourceAudience).toBe("https://mcp.xcelsior.ca/mcp");
    expect(config.resourceMetadataUrl).toBe(METADATA);
  });

  it("derives the legacy origin audience from the canonical value", () => {
    const config = withEnv(
      {
        XCELSIOR_MCP_RESOURCE_AUDIENCE: "https://mcp.xcelsior.ca/mcp",
        XCELSIOR_MCP_LEGACY_RESOURCE_AUDIENCE: undefined,
        XCELSIOR_MCP_LEGACY_AUDIENCE_SUNSET: "2999-01-01T00:00:00Z",
      },
      loadConfig,
    );
    expect(config.legacyResourceAudience).toBe("https://mcp.xcelsior.ca");
    expect(acceptedAudiences(config)).toEqual([
      "https://mcp.xcelsior.ca/mcp",
      "https://mcp.xcelsior.ca",
    ]);
  });

  it("drops the legacy audience once the sunset passes", () => {
    const config = withEnv(
      {
        XCELSIOR_MCP_RESOURCE_AUDIENCE: "https://mcp.xcelsior.ca/mcp",
        XCELSIOR_MCP_LEGACY_AUDIENCE_SUNSET: "2020-01-01T00:00:00Z",
      },
      loadConfig,
    );
    expect(acceptedAudiences(config)).toEqual(["https://mcp.xcelsior.ca/mcp"]);
  });

  it("never treats a path-less resource as its own legacy alias", () => {
    const config = withEnv(
      {
        XCELSIOR_MCP_RESOURCE_AUDIENCE: "https://mcp.test",
        XCELSIOR_MCP_LEGACY_RESOURCE_AUDIENCE: undefined,
      },
      loadConfig,
    );
    expect(config.legacyResourceAudience).toBe("");
    expect(acceptedAudiences(config)).toEqual(["https://mcp.test"]);
  });

  it("defaults to the customer trust surface", () => {
    const config = withEnv({ XCELSIOR_MCP_TOOL_PROFILE: undefined }, loadConfig);
    expect(config.toolProfile).toBe("customer");
  });
});
