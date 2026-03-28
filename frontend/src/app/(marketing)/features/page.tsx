import type { Metadata } from "next";
import { FeaturesContent } from "./content";

export const metadata: Metadata = {
  title: "Features",
  description:
    "Explore Xcelsior's GPU compute features: data sovereignty, PIPEDA compliance, Slurm HPC, real-time telemetry, and AI Compute Access Fund rebates.",
  alternates: { canonical: "https://xcelsior.ca/features" },
  openGraph: {
    title: "Features | Xcelsior",
    description:
      "Data sovereignty, compliance automation, Slurm HPC, real-time telemetry, and 67% federal rebates.",
    url: "https://xcelsior.ca/features",
  },
  twitter: {
    title: "Features | Xcelsior",
    description:
      "Data sovereignty, compliance automation, Slurm HPC, real-time telemetry, and 67% federal rebates.",
  },
};

export default function FeaturesPage() {
  return <FeaturesContent />;
}
