import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Xcelsior — Sovereign GPU Compute",
    short_name: "Xcelsior",
    description:
      "Canada-first GPU compute marketplace with data sovereignty, compliance automation, and competitive pricing.",
    start_url: "/",
    display: "standalone",
    background_color: "#060a13",
    theme_color: "#060a13",
    icons: [
      { src: "/favicon.svg", sizes: "any", type: "image/svg+xml" },
      { src: "/xcelsior_icon_16x16.png", sizes: "16x16", type: "image/png" },
      { src: "/xcelsior_icon_32x32.png", sizes: "32x32", type: "image/png" },
      { src: "/xcelsior_icon_48x48.png", sizes: "48x48", type: "image/png" },
      { src: "/xcelsior_icon_60x60.png", sizes: "60x60", type: "image/png" },
      { src: "/xcelsior_icon_120x120.png", sizes: "120x120", type: "image/png" },
      { src: "/xcelsior_icon_150x150.png", sizes: "150x150", type: "image/png" },
      { src: "/xcelsior_icon_180x180.png", sizes: "180x180", type: "image/png" },
      { src: "/xcelsior_icon_192x192.png", sizes: "192x192", type: "image/png", purpose: "maskable" },
      { src: "/xcelsior_icon_512x512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
    ],
  };
}
