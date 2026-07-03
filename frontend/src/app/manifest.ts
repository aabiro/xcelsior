import type { MetadataRoute } from "next";
import { SITE_ASSETS } from "@/lib/brand-assets";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Xcelsior, Cheap, Compliant GPU Compute in Canada",
    short_name: "Xcelsior",
    description:
      "Rent verified GPUs by the hour in Canadian dollars, PIPEDA compliance and clean hydro built in. From $0.30 CAD/hr, with dynamic spot pricing up to 70% off.",
    id: "/",
    start_url: "/",
    scope: "/",
    display: "standalone",
    background_color: "#060a13",
    theme_color: "#060a13",
    orientation: "any",
    categories: ["business", "productivity", "utilities"],
    icons: [
      {
        src: SITE_ASSETS.icon192,
        sizes: "192x192",
        type: "image/png",
        purpose: "any",
      },
      {
        src: SITE_ASSETS.icon512,
        sizes: "512x512",
        type: "image/png",
        purpose: "any",
      },
      {
        src: SITE_ASSETS.iconMaskable512,
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
      {
        src: SITE_ASSETS.appGradientRounded512,
        sizes: "512x512",
        type: "image/png",
        purpose: "any",
      },
    ],
    shortcuts: [
      {
        name: "Marketplace",
        short_name: "Market",
        url: "/dashboard/marketplace",
        icons: [{ src: SITE_ASSETS.icon192, sizes: "192x192", type: "image/png" }],
      },
      {
        name: "Instances",
        short_name: "Instances",
        url: "/dashboard/instances",
        icons: [{ src: SITE_ASSETS.icon192, sizes: "192x192", type: "image/png" }],
      },
      {
        name: "Notifications",
        short_name: "Alerts",
        url: "/dashboard/notifications",
        icons: [{ src: SITE_ASSETS.icon192, sizes: "192x192", type: "image/png" }],
      },
      {
        name: "Billing",
        short_name: "Billing",
        url: "/dashboard/billing",
        icons: [{ src: SITE_ASSETS.icon192, sizes: "192x192", type: "image/png" }],
      },
    ],
    screenshots: [
      {
        src: "/desktop-dashboard-screenshot.svg",
        sizes: "1200x630",
        type: "image/svg+xml",
        form_factor: "wide",
        label: "Xcelsior dashboard in the desktop shell",
      },
      {
        src: "/desktop-control-center-screenshot.svg",
        sizes: "900x1600",
        type: "image/svg+xml",
        form_factor: "narrow",
        label: "Xcelsior Control Center for desktop operations",
      },
    ],
    share_target: {
      action: "/dashboard/marketplace",
      method: "GET",
      enctype: "application/x-www-form-urlencoded",
      params: {
        title: "title",
        text: "text",
        url: "url",
      },
    },
  };
}
