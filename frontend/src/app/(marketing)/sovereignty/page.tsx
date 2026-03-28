import type { Metadata } from "next";
import { SovereigntyContent } from "./content";

export const metadata: Metadata = {
  title: "Data Sovereignty",
  description: "Understand the difference between data residency and true data sovereignty. How Xcelsior protects Canadian data from the US CLOUD Act.",
  alternates: { canonical: "https://xcelsior.ca/sovereignty" },
  openGraph: {
    title: "Data Sovereignty | Xcelsior",
    description: "True Canadian data sovereignty — not just residency. Protection from the US CLOUD Act.",
    url: "https://xcelsior.ca/sovereignty",
  },
  twitter: {
    title: "Data Sovereignty | Xcelsior",
    description: "True Canadian data sovereignty — not just residency. Protection from the US CLOUD Act.",
  },
};

export default function SovereigntyPage() {
  return <SovereigntyContent />;
}
