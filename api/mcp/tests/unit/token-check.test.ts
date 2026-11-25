/**
 * Two faults the live rehearsal found on 2026-11-24.
 *
 * The API limits requests per client address. Every MCP call reached it from
 * this server's address, so all MCP users shared one limit, and once it filled
 * the token check came back 429. validateBearer turned that into "invalid
 * token", which tells a connector to discard a good token and sign in again.
 */
import type { IncomingMessage } from "node:http";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TokenCheckUnavailable, forwardedFor, validateBearer } from "../../src/auth/bearer.js";
import { XcelsiorApiClient } from "../../src/client/api.js";

afterEach(() => vi.unstubAllGlobals());

function request(headers: Record<string, string>, remoteAddress?: string): IncomingMessage {
  return { headers, socket: { remoteAddress } } as unknown as IncomingMessage;
}

const principal = { customer_id: "cust-1", audience: "https://mcp.example/mcp" };

describe("the caller's address reaches the API", () => {
  it("appends this server's peer to the chain it received, as a proxy does", () => {
    expect(forwardedFor(request({ "x-forwarded-for": "203.0.113.7" }, "127.0.0.1"))).toBe(
      "203.0.113.7, 127.0.0.1",
    );
  });

  it("keeps a chain a caller made up on the left, where the API's trust walk skips it", () => {
    expect(forwardedFor(request({ "x-forwarded-for": "1.2.3.4, 203.0.113.7" }, "::ffff:127.0.0.1"))).toBe(
      "1.2.3.4, 203.0.113.7, 127.0.0.1",
    );
  });

  it("uses the peer alone when nothing was forwarded", () => {
    expect(forwardedFor(request({}, "198.51.100.2"))).toBe("198.51.100.2");
    expect(forwardedFor(request({}))).toBeUndefined();
  });

  it("is sent on the token check and on every tool call", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(principal), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await validateBearer("https://api.example", "t", undefined, "203.0.113.7, 127.0.0.1");
    const client = new XcelsiorApiClient({
      baseUrl: "https://api.example",
      bearer: "t",
      forwardedFor: "203.0.113.7, 127.0.0.1",
    });
    await client.get("/api/v1/read");
    for (const [, init] of fetchMock.mock.calls) {
      expect((init as RequestInit).headers).toMatchObject({ "X-Forwarded-For": "203.0.113.7, 127.0.0.1" });
    }
  });
});

describe("a token check that could not happen is not a rejected token", () => {
  it("answers null when the API rejects the token", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}", { status: 401 })));
    await expect(validateBearer("https://api.example", "t")).resolves.toBeNull();
  });

  it("throws TokenCheckUnavailable when the API throttles the check", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}", { status: 429 })));
    await expect(validateBearer("https://api.example", "t")).rejects.toBeInstanceOf(TokenCheckUnavailable);
  });

  it("throws TokenCheckUnavailable when the API fails or does not answer", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("down", { status: 503 })));
    await expect(validateBearer("https://api.example", "t")).rejects.toBeInstanceOf(TokenCheckUnavailable);
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fetch failed")));
    await expect(validateBearer("https://api.example", "t")).rejects.toBeInstanceOf(TokenCheckUnavailable);
  });

  it("still rejects a token bound to another resource", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(principal), { status: 200 })));
    await expect(validateBearer("https://api.example", "t", ["https://other.example/mcp"])).resolves.toBeNull();
  });
});
