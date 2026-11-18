import type { Metadata } from "next";
import dynamic from "next/dynamic";
import { SITE_ASSETS } from "@/lib/brand-assets";

const McpLandingContent = dynamic(
  () => import("./content").then((mod) => mod.McpLandingContent),
  { loading: () => <div className="min-h-[50vh]" aria-hidden /> },
);

// Link previews (LinkedIn, Facebook, Slack, X) don't render SVG, so the card
// must be a raster image. Metadata merges shallowly: this page's `openGraph`
// replaces the root one whole (so siteName/type are restated here), and
// without its own `twitter` block X would show the homepage's title and card.
const SHARE_TITLE = "Let AI agents control real GPUs | Xcelsior MCP";
const SHARE_DESCRIPTION =
  "Connect Cursor, Claude, and other agents to the Xcelsior GPU network with one hosted MCP server.";
const SHARE_IMAGE = {
  url: SITE_ASSETS.ogMcp1200x630,
  width: 1200,
  height: 630,
  alt: "Xcelsior MCP: Your agent rents the GPU. You approve the bill.",
};

export const metadata: Metadata = {
  title: "Xcelsior MCP, AI Agent GPU Control",
  description:
    "Let AI agents control real GPUs. Xcelsior MCP turns natural language into compute, instantly, securely, and globally.",
  alternates: { canonical: "https://xcelsior.ca/mcp" },
  openGraph: {
    title: SHARE_TITLE,
    description: SHARE_DESCRIPTION,
    url: "https://xcelsior.ca/mcp",
    siteName: "Xcelsior",
    type: "website",
    images: [SHARE_IMAGE],
  },
  twitter: {
    card: "summary_large_image",
    title: SHARE_TITLE,
    description: SHARE_DESCRIPTION,
    images: [SHARE_IMAGE],
  },
};

export default function McpLandingPage() {
  return <McpLandingContent />;
}
