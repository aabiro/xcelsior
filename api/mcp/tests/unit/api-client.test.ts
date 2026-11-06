import { afterEach, describe, expect, it, vi } from "vitest";
import { XcelsiorApiClient } from "../../src/client/api.js";
import { ApiError } from "../../src/client/errors.js";
import { traced, currentToolTraceparent } from "../../src/observability/tracing.js";

afterEach(() => vi.unstubAllGlobals());

describe("typed API transport behavior", () => {
  it("decodes RFC 9457 problems", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      type: "https://docs.xcelsior.ca/problems/version-conflict",
      title: "Conflict", status: 409, detail: "stale version", code: "version_conflict",
      retryable: false, trace_id: "abc",
    }), { status: 409, headers: { "content-type": "application/problem+json" } })));
    const client = new XcelsiorApiClient({ baseUrl: "https://api.example", bearer: "redacted" });
    await expect(client.post("/api/v1/test", {})).rejects.toMatchObject({
      status: 409,
      problem: { code: "version_conflict", detail: "stale version" },
    } satisfies Partial<ApiError>);
  });

  it("retries safe reads but never blindly replays a write", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response("unavailable", { status: 503 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new XcelsiorApiClient({ baseUrl: "https://api.example", bearer: "redacted" });
    await expect(client.get("/api/v1/read")).resolves.toEqual({ ok: true });
    expect(fetchMock).toHaveBeenCalledTimes(2);

    fetchMock.mockReset();
    fetchMock.mockResolvedValue(new Response("unavailable", { status: 503 }));
    await expect(client.post("/api/v1/write", {})).rejects.toBeInstanceOf(ApiError);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});


describe("tool trace propagation without an exporter", () => {
  it("keeps concurrent requests correlated with their own tool audit", async () => {
    const sent = new Map<string, string>();
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      sent.set(new URL(url).pathname, new Headers(init.headers).get("traceparent") || "");
      return new Response(JSON.stringify({ok:true}));
    }));
    const client = new XcelsiorApiClient({baseUrl:"https://api.example",bearer:"redacted"});
    const traces = await Promise.all(["first","second"].map(name => traced(name, async (parent, traceId) => {
      await Promise.resolve();
      await client.get(`/api/v1/${name}`);
      expect(sent.get(`/api/v1/${name}`)).toBe(parent);
      expect(parent?.split("-")[1]).toBe(traceId);
      expect(parent).toMatch(/^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$/);
      return traceId;
    })));
    expect(traces[0]).not.toBe(traces[1]);
    expect(currentToolTraceparent()).toBeUndefined();
  });

  it("honors an explicit request context", async () => {
    const fetchMock=vi.fn(async (_url: string, _init?: RequestInit) => new Response(JSON.stringify({ok:true})));
    vi.stubGlobal("fetch",fetchMock);
    const explicit="00-11111111111111111111111111111111-2222222222222222-00";
    const client=new XcelsiorApiClient({baseUrl:"https://api.example",bearer:"redacted"});
    await traced("tool",async () => { await client.get("/api/v1/test",undefined,{traceparent:explicit}); });
    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get("traceparent")).toBe(explicit);
  });
});
