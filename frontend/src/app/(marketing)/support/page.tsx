import type { Metadata } from "next";
import { SupportContent } from "./content";

export const metadata: Metadata = {
  title: "Support — Xcelsior",
  description:
    "Get help with Xcelsior GPU compute. Chat with our AI assistant, reach our team by email, or browse documentation.",
  alternates: { canonical: "https://xcelsior.ca/support" },
  openGraph: {
    title: "Support — Xcelsior",
    description:
      "Get help with Xcelsior GPU compute. Chat with our AI assistant or reach our team directly.",
    url: "https://xcelsior.ca/support",
  },
  twitter: {
    title: "Support — Xcelsior",
    description:
      "Get help with Xcelsior GPU compute. Chat with our AI assistant or reach our team directly.",
  },
};

export default function SupportPage() {
  return <SupportContent />;
}
