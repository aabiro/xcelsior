import { afterEach, describe, expect, it, vi } from "vitest";

const { init } = vi.hoisted(() => ({ init: vi.fn() }));

vi.mock("posthog-js", () => ({
  default: { init },
}));

describe("PostHog browser instrumentation", () => {
  afterEach(() => {
    init.mockReset();
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it("captures initial and client-side pageviews plus pageleaves", async () => {
    vi.stubEnv("NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN", "phc_test_project_token");
    vi.stubEnv("NODE_ENV", "production");

    await import("../../instrumentation-client");

    expect(init).toHaveBeenCalledWith(
      "phc_test_project_token",
      expect.objectContaining({
        api_host: "/ingest",
        defaults: "2026-05-30",
        capture_pageview: "history_change",
        capture_pageleave: true,
      }),
    );
  });

  it("does not initialize without a project token", async () => {
    vi.stubEnv("NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN", "");

    await import("../../instrumentation-client");

    expect(init).not.toHaveBeenCalled();
  });
});
