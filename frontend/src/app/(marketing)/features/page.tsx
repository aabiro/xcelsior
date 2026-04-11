import type { Metadata } from "next";
import { FeaturesContent } from "./content";

export const metadata: Metadata = {
  title: "Features",
  description:
    "Explore Xcelsior's Canada-first GPU compute platform: transparent pricing, compliance-aware operations, sovereignty controls, telemetry, and provider tooling for teams worldwide.",
  alternates: { canonical: "https://xcelsior.ca/features" },
  openGraph: {
    title: "Features | Xcelsior",
    description:
      "Transparent pricing, compliance-aware operations, sovereignty controls, telemetry, and provider tooling.",
    url: "https://xcelsior.ca/features",
  },
  twitter: {
    title: "Features | Xcelsior",
    description:
      "Transparent pricing, compliance-aware operations, sovereignty controls, telemetry, and provider tooling.",
  },
};

export default function FeaturesPage() {
  return <FeaturesContent />;
}
