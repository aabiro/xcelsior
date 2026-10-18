import { describe, expect, it } from "vitest";
import { parseDesktopDeepLink } from "../lib/deep-links";

describe("desktop deep link parsing", () => {
  it("resolves control-center routes from path-based deep links", () => {
    expect(parseDesktopDeepLink("xcelsior://desktop/settings")).toEqual({
      target: "desktop",
      route: "/desktop/settings",
    });
  });

  it("resolves shared-app routes from path-based deep links", () => {
    expect(parseDesktopDeepLink("xcelsior://dashboard/instances?tab=active")).toEqual({
      target: "remote",
      route: "/dashboard/instances?tab=active",
    });
  });

  it("prefers explicit route query overrides", () => {
    expect(parseDesktopDeepLink("xcelsior://open?route=/desktop/activity&mode=debug")).toEqual({
      target: "desktop",
      route: "/desktop/activity",
    });
  });

  it("merges non-route query params onto explicit remote routes", () => {
    expect(parseDesktopDeepLink("xcelsior://open?route=/dashboard/billing&invoice=inv_123")).toEqual({
      target: "remote",
      route: "/dashboard/billing?invoice=inv_123",
    });
  });

  it("normalizes invalid deep-link targets to safe defaults", () => {
    expect(parseDesktopDeepLink("xcelsior://desktop/unknown")).toEqual({
      target: "desktop",
      route: "/desktop",
    });
    expect(parseDesktopDeepLink("xcelsior://desktop?route=dashboard")).toEqual({
      target: "remote",
      route: "/dashboard",
    });
  });

  it("rejects malformed or non-xcelsior URLs", () => {
    expect(parseDesktopDeepLink("https://xcelsior.ca/dashboard")).toBeNull();
    expect(parseDesktopDeepLink("not-a-url")).toBeNull();
  });
});
