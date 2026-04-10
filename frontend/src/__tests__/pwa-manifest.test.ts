import { describe, expect, it } from "vitest";
import manifest from "@/app/manifest";

describe("desktop PWA manifest", () => {
  it("exposes standalone install metadata for the desktop app", () => {
    const appManifest = manifest();

    expect(appManifest.id).toBe("/");
    expect(appManifest.scope).toBe("/");
    expect(appManifest.start_url).toBe("/");
    expect(appManifest.display).toBe("standalone");
    expect(appManifest.background_color).toBe("#060a13");
    expect(appManifest.theme_color).toBe("#060a13");
  });

  it("includes maskable install icons and desktop shortcuts", () => {
    const appManifest = manifest();

    expect(appManifest.icons).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          src: "/xcelsior_icon_192x192.png",
          sizes: "192x192",
          purpose: "maskable",
        }),
        expect.objectContaining({
          src: "/xcelsior_icon_512x512.png",
          sizes: "512x512",
          purpose: "maskable",
        }),
      ]),
    );

    expect(appManifest.shortcuts).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ url: "/dashboard/marketplace" }),
        expect.objectContaining({ url: "/dashboard/instances" }),
        expect.objectContaining({ url: "/dashboard/notifications" }),
        expect.objectContaining({ url: "/dashboard/billing" }),
      ]),
    );
  });

  it("includes screenshots and a share target for installed surfaces", () => {
    const appManifest = manifest();

    expect(appManifest.screenshots).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          form_factor: "wide",
          src: "/desktop-dashboard-screenshot.svg",
        }),
        expect.objectContaining({
          form_factor: "narrow",
          src: "/desktop-control-center-screenshot.svg",
        }),
      ]),
    );

    expect(appManifest.share_target).toEqual(
      expect.objectContaining({
        action: "/dashboard/marketplace",
        method: "GET",
      }),
    );
  });
});
