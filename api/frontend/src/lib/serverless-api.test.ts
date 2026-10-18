import { beforeEach, describe, expect, it, vi } from "vitest";
import { createServerlessJobStream } from "@/lib/api";

class MockEventSource {
  url: string;
  constructor(url: string, _opts?: { withCredentials?: boolean }) {
    this.url = url;
  }
  close() {}
}

describe("serverless API helpers", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.stubGlobal("EventSource", MockEventSource);
  });

  it("createServerlessJobStream builds a credentialed EventSource URL", () => {
    const es = createServerlessJobStream("sep-abc/123", "job-xyz", 5);
    expect(es.url).toContain("/v1/serverless/sep-abc%2F123/stream/job-xyz");
    expect(es.url).toContain("after_seq=5");
    es.close();
  });

  it("createServerlessJobStream omits query when afterSeq is zero", () => {
    const es = createServerlessJobStream("sep-1", "job-1", 0);
    expect(es.url).toBe("/v1/serverless/sep-1/stream/job-1");
    es.close();
  });
});