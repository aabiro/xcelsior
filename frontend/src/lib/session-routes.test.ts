import { describe, expect, it } from "vitest";
import { needsSessionOnMount } from "./session-routes";

describe("needsSessionOnMount", () => {
  it("returns false for public marketing pages", () => {
    expect(needsSessionOnMount("/")).toBe(false);
    expect(needsSessionOnMount("/pricing")).toBe(false);
    expect(needsSessionOnMount("/gpu-availability")).toBe(false);
  });

  it("returns true for dashboard and auth entry routes", () => {
    expect(needsSessionOnMount("/dashboard")).toBe(true);
    expect(needsSessionOnMount("/dashboard/instances")).toBe(true);
    expect(needsSessionOnMount("/login")).toBe(true);
    expect(needsSessionOnMount("/register")).toBe(true);
    expect(needsSessionOnMount("/accept-invite")).toBe(true);
    expect(needsSessionOnMount("/setup-2fa")).toBe(true);
  });
});