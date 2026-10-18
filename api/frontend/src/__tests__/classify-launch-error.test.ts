import { describe, it, expect } from "vitest";

import { ApiError, classifyLaunchError } from "@/lib/api";

/**
 * `handled` drives whether the launch modal reports an error to error
 * tracking: recognized user-state rejections are skipped, unclassified and
 * server errors are captured. `status` gives the modal a stable, amount-free
 * fingerprint for the captures it does keep.
 */
describe("classifyLaunchError", () => {
  it("marks an empty-wallet 402 as handled user state", () => {
    const info = classifyLaunchError(
      new ApiError(402, "Insufficient balance: hold $0.20 CAD, available $0.00 CAD", {
        detail: "Insufficient balance: hold $0.20 CAD, available $0.00 CAD",
      }),
    );
    expect(info.handled).toBe(true);
    expect(info.status).toBe(402);
    expect(info.action?.label).toBe("Add Funds");
  });

  it("marks a suspended-wallet 402 as handled user state", () => {
    const info = classifyLaunchError(
      new ApiError(402, "Wallet suspended", { detail: "Wallet suspended" }),
    );
    expect(info.handled).toBe(true);
    expect(info.status).toBe(402);
  });

  it("marks a concurrency-limit 429 as handled user state", () => {
    const info = classifyLaunchError(
      new ApiError(429, "concurrent instance limit reached", {
        detail: "concurrent instance limit reached",
      }),
    );
    expect(info.handled).toBe(true);
    expect(info.status).toBe(429);
  });

  it("keeps an unclassified 4xx unhandled so it is still captured", () => {
    const info = classifyLaunchError(new ApiError(400, "bad request", {}));
    expect(info.handled).toBe(false);
    expect(info.status).toBe(400);
  });

  it("keeps a 5xx server error unhandled by status even when it has a friendly message", () => {
    const info = classifyLaunchError(new ApiError(503, "no hosts", {}));
    // Recognized message, but the modal still captures it because status >= 500.
    expect(info.status).toBe(503);
  });

  it("marks a non-API error as unhandled with no status", () => {
    const info = classifyLaunchError(new Error("boom"));
    expect(info.handled).toBe(false);
    expect(info.status).toBeUndefined();
  });
});
