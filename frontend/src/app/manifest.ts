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
      { src: "/logo.svg", sizes: "512x512", type: "image/svg+xml", purpose: "maskable" },
    ],
  };
}
