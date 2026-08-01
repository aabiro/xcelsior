import type { Metadata } from "next";
import { StatusContent } from "./content";

export const metadata: Metadata = {
  title: "Status & Service Levels",
  description:
    "Live service status for the Xcelsior platform and the mcp.xcelsior.ca connector, and the availability and latency objectives they are measured against.",
  alternates: { canonical: "https://xcelsior.ca/status" },
  openGraph: {
    title: "Status & Service Levels | Xcelsior",
    description:
      "Live service status and published availability and latency objectives.",
    url: "https://xcelsior.ca/status",
  },
};

export default function StatusPage() {
  return <StatusContent />;
}
