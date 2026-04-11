import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Xcelsior — Sovereign GPU Compute",
    short_name: "Xcelsior",
    description:
      "Canada-first GPU compute marketplace with transparent pricing, compliance-aware operations, and infrastructure for teams worldwide.",
    id: "/",
    start_url: "/",
    scope: "/",
    display: "standalone",
    background_color: "#060a13",
    theme_color: "#060a13",
    orientation: "any",
    categories: ["business", "productivity", "utilities"],
    icons: [
      { src: "/favicon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" },
      { src: "/xcelsior_icon_16x16.png", sizes: "16x16", type: "image/png" },
      { src: "/xcelsior_icon_32x32.png", sizes: "32x32", type: "image/png" },
      { src: "/xcelsior_icon_48x48.png", sizes: "48x48", type: "image/png" },
      { src: "/xcelsior_icon_60x60.png", sizes: "60x60", type: "image/png" },
      { src: "/xcelsior_icon_120x120.png", sizes: "120x120", type: "image/png" },
      { src: "/xcelsior_icon_150x150.png", sizes: "150x150", type: "image/png" },
      { src: "/xcelsior_icon_180x180.png", sizes: "180x180", type: "image/png" },
      {
        src: "/xcelsior_icon_192x192.png",
        sizes: "192x192",
        type: "image/png",
        purpose: "maskable",
      },
      {
        src: "/xcelsior_icon_512x512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
    shortcuts: [
      {
        name: "Marketplace",
        short_name: "Market",
        url: "/dashboard/marketplace",
        icons: [{ src: "/xcelsior_icon_192x192.png", sizes: "192x192", type: "image/png" }],
      },
      {
        name: "Instances",
        short_name: "Instances",
        url: "/dashboard/instances",
        icons: [{ src: "/xcelsior_icon_192x192.png", sizes: "192x192", type: "image/png" }],
      },
      {
        name: "Notifications",
        short_name: "Alerts",
        url: "/dashboard/notifications",
        icons: [{ src: "/xcelsior_icon_192x192.png", sizes: "192x192", type: "image/png" }],
      },
      {
        name: "Billing",
        short_name: "Billing",
        url: "/dashboard/billing",
        icons: [{ src: "/xcelsior_icon_192x192.png", sizes: "192x192", type: "image/png" }],
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
