import type { Metadata } from "next";
import dynamic from "next/dynamic";

const McpLandingContent = dynamic(
  () => import("./content").then((mod) => mod.McpLandingContent),
  { loading: () => <div className="min-h-[50vh]" aria-hidden /> },
);

export const metadata: Metadata = {
  title: "Xcelsior MCP — AI Agent GPU Control",
  description:
    "Let AI agents control real GPUs. Xcelsior MCP turns natural language into compute — instantly, securely, and globally.",
  alternates: { canonical: "https://xcelsior.ca/mcp" },
  openGraph: {
    title: "Let AI agents control real GPUs | Xcelsior MCP",
    description:
      "Connect Cursor, Claude, and other agents to the Xcelsior GPU network with one hosted MCP server.",
    url: "https://xcelsior.ca/mcp",
    images: [{ url: "/mcp/og-mcp-card.svg", width: 1200, height: 630, alt: "Xcelsior MCP" }],
  },
};

export default function McpLandingPage() {
  return <McpLandingContent />;
}