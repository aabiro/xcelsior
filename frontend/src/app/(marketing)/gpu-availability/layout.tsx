import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "GPU Availability",
  description:
    "Live GPU availability and CAD pricing across the Xcelsior network. See on-demand and spot rates, regions, and deploy when capacity is open.",
  alternates: { canonical: "https://xcelsior.ca/gpu-availability" },
  openGraph: {
    title: "GPU Availability | Xcelsior",
    description: "Live GPU availability and transparent CAD pricing across Canada-first compute.",
    url: "https://xcelsior.ca/gpu-availability",
  },
  robots: { index: true, follow: true },
};

export default function GPUAvailabilityLayout({ children }: { children: React.ReactNode }) {
  return children;
}