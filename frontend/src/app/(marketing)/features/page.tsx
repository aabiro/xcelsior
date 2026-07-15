import type { Metadata } from "next";
import dynamic from "next/dynamic";

const FeaturesContent = dynamic(
  () => import("./content").then((mod) => mod.FeaturesContent),
  { loading: () => <div className="min-h-[40vh]" aria-hidden /> },
);

export const metadata: Metadata = {
  title: "Features",
  description:
    "Explore Xcelsior's agent-native GPU compute platform: transparent pricing, real-time telemetry, jurisdiction controls, and provider tooling for teams worldwide.",
  alternates: { canonical: "https://xcelsior.ca/features" },
  openGraph: {
    title: "Features | Xcelsior",
    description:
      "Native MCP for AI agents, real-time telemetry, transparent pricing, and provider tooling.",
    url: "https://xcelsior.ca/features",
  },
  twitter: {
    title: "Features | Xcelsior",
    description:
      "Native MCP for AI agents, real-time telemetry, transparent pricing, and provider tooling.",
  },
};

export default function FeaturesPage() {
  return <FeaturesContent />;
}
