import type { Metadata } from "next";
import dynamic from "next/dynamic";

const FeaturesContent = dynamic(
  () => import("./content").then((mod) => mod.FeaturesContent),
  { loading: () => <div className="min-h-[40vh]" aria-hidden /> },
);

export const metadata: Metadata = {
  title: "Features",
  description:
    "Explore Xcelsior's Canada-first GPU compute platform: transparent pricing, compliance-aware operations, jurisdiction controls, telemetry, and provider tooling for teams worldwide.",
  alternates: { canonical: "https://xcelsior.ca/features" },
  openGraph: {
    title: "Features | Xcelsior",
    description:
      "Transparent pricing, compliance-aware operations, jurisdiction controls, telemetry, and provider tooling.",
    url: "https://xcelsior.ca/features",
  },
  twitter: {
    title: "Features | Xcelsior",
    description:
      "Transparent pricing, compliance-aware operations, jurisdiction controls, telemetry, and provider tooling.",
  },
};

export default function FeaturesPage() {
  return <FeaturesContent />;
}
