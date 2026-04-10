import { describe, expect, it } from "vitest";
import { isEventStreamRequest, isPublicDataPath, isSensitiveDataPath } from "@/lib/pwa/runtime-caching";

describe("desktop PWA cache policy", () => {
  it("allows short-lived caching only for explicit public read models", () => {
    expect(isPublicDataPath("/marketplace")).toBe(true);
    expect(isPublicDataPath("/marketplace/search")).toBe(true);
    expect(isPublicDataPath("/spot-prices")).toBe(true);
    expect(isPublicDataPath("/compute-score/host-1")).toBe(true);
    expect(isPublicDataPath("/api/pricing/reference")).toBe(true);
    expect(isPublicDataPath("/api/images/templates")).toBe(true);

    expect(isPublicDataPath("/billing")).toBe(false);
    expect(isPublicDataPath("/instances")).toBe(false);
    expect(isPublicDataPath("/api/notifications")).toBe(false);
  });

  it("treats user-sensitive and control-plane endpoints as network only", () => {
    expect(isSensitiveDataPath("/billing")).toBe(true);
    expect(isSensitiveDataPath("/instances")).toBe(true);
    expect(isSensitiveDataPath("/instance/abc")).toBe(true);
    expect(isSensitiveDataPath("/api/auth/refresh")).toBe(true);
    expect(isSensitiveDataPath("/api/billing/wallet/customer-1")).toBe(true);
    expect(isSensitiveDataPath("/api/notifications")).toBe(true);
    expect(isSensitiveDataPath("/api/chat/conversations")).toBe(true);

    expect(isSensitiveDataPath("/dashboard")).toBe(false);
    expect(isSensitiveDataPath("/marketplace")).toBe(false);
  });

  it("keeps event streams off the cache entirely", () => {
    const streamRequest = new Request("https://xcelsior.ca/instances/abc/logs/stream", {
      headers: { Accept: "text/event-stream" },
    });
    const normalRequest = new Request("https://xcelsior.ca/marketplace", {
      headers: { Accept: "application/json" },
    });

    expect(isEventStreamRequest(streamRequest, "/instances/abc/logs/stream")).toBe(true);
    expect(isEventStreamRequest(normalRequest, "/marketplace")).toBe(false);
  });
});
