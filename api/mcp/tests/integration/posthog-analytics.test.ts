import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { PostHog } from "posthog-node";
import { describe, expect, it, vi } from "vitest";
import { z } from "zod";
import {
  instrumentMcpAnalyticsWithClient,
  posthogIdentity,
  stripPrivateMcpPayloads,
} from "../../src/analytics.js";

describe("PostHog MCP analytics", () => {
  it("uses the browser identity and never falls back to email", () => {
    expect(posthogIdentity({ user_id: "user-7", subject: "subject-7", email: "person@example.test" }))
      .toEqual({ distinctId: "user-7" });
    expect(posthogIdentity({ subject: "machine-client", email: "machine@example.test" }))
      .toEqual({ distinctId: "machine-client" });
    expect(posthogIdentity({ email: "only-email@example.test" })).toBeNull();
  });

  it("removes arguments, results, and error details before send", () => {
    const safe = stripPrivateMcpPayloads({
      distinct_id: "user-7",
      event: "$mcp_tool_call",
      properties: {
        "$mcp_parameters": { init_script: "super-secret" },
        "$mcp_response": { environment: { TOKEN: "super-secret" } },
        "$mcp_error_message": "upstream returned super-secret",
        "$exception_list": [{ value: "super-secret" }],
        "$mcp_tool_name": "create_instance",
      },
      timestamp: new Date().toISOString(),
      type: "capture",
    });
    expect(safe.properties).toEqual({ "$mcp_tool_name": "create_instance" });
    expect(JSON.stringify(safe)).not.toContain("super-secret");
  });

  it("captures real protocol events without changing schemas or retaining tool content", async () => {
    const capture = vi.fn();
    const fakePostHog = { capture } as unknown as PostHog;
    const server = new McpServer({ name: "analytics-test", version: "1.0.0" });
    server.registerTool(
      "secret_echo",
      {
        description: "Test-only tool",
        inputSchema: {
          init_script: z.string(),
          environment: z.record(z.string()),
        },
      },
      async ({ init_script, environment }) => ({
        content: [{ type: "text", text: JSON.stringify({ init_script, environment }) }],
      }),
    );

    instrumentMcpAnalyticsWithClient(server, fakePostHog, {
      transport: "streamable_http",
      profile: "customer",
      user: { user_id: "user-analytics" },
    });

    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "analytics-test-client", version: "1.0.0" });
    await server.connect(serverTransport);
    await client.connect(clientTransport);

    const listed = await client.listTools();
    const tool = listed.tools.find((candidate) => candidate.name === "secret_echo");
    expect(tool).toBeDefined();
    expect(tool?.inputSchema.properties).not.toHaveProperty("context");
    expect(tool?.inputSchema.properties).not.toHaveProperty("conversation_id");

    await client.callTool({
      name: "secret_echo",
      arguments: {
        init_script: "do-not-retain-this",
        environment: { ACCESS_TOKEN: "also-do-not-retain-this" },
      },
    });

    await vi.waitFor(() => {
      expect(capture.mock.calls.some(([event]) => event.event === "$mcp_tool_call")).toBe(true);
    });
    const events = capture.mock.calls.map(([event]) => event as {
      distinctId: string;
      event: string;
      properties: Record<string, unknown>;
    });
    const toolCall = events.find((event) => event.event === "$mcp_tool_call");
    expect(toolCall).toMatchObject({
      distinctId: "user-analytics",
      properties: {
        "$mcp_tool_name": "secret_echo",
        xcelsior_transport: "streamable_http",
        xcelsior_tool_profile: "customer",
      },
    });
    expect(toolCall?.properties).not.toHaveProperty("$mcp_parameters");
    expect(toolCall?.properties).not.toHaveProperty("$mcp_response");
    expect(JSON.stringify(events)).not.toContain("do-not-retain-this");
    expect(JSON.stringify(events)).not.toContain("also-do-not-retain-this");

    await client.close();
    await server.close();
  });
});
