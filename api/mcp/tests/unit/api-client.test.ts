import { afterEach, describe, expect, it, vi } from "vitest";
import { XcelsiorApiClient } from "../../src/client/api.js";
import { ApiError } from "../../src/client/errors.js";

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
