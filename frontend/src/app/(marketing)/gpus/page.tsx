import type { Metadata } from "next";
import dynamic from "next/dynamic";

const GPUAvailabilityContent = dynamic(
  () => import("../gpu-availability/content").then((mod) => mod.GPUAvailabilityContent),
  { loading: () => <div className="site-container" style={{ minHeight: "50vh" }} aria-hidden /> },
);

export const metadata: Metadata = {
  title: "GPU Availability",
  description:
    "Live GPU availability across the Xcelsior network. See which models are online, pricing in CAD, and spot rates updated in real time.",
  alternates: { canonical: "https://xcelsior.ca/gpu-availability" },
  openGraph: {
    title: "GPU Availability | Xcelsior",
    description: "Live GPU availability and pricing across the Xcelsior sovereign compute network.",
    url: "https://xcelsior.ca/gpu-availability",
  },
};

export default function GpusPage() {
  return <GPUAvailabilityContent />;
}
