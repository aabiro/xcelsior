import type { Metadata } from "next";
import { SecurityContent } from "./content";

export const metadata: Metadata = {
  title: "Security & Trust",
  description:
    "How Xcelsior secures agent access to real infrastructure: server-bound action plans for destructive operations, scoped tokens, per-tool rate limits, and a customer-readable audit trail.",
  alternates: { canonical: "https://xcelsior.ca/security" },
  openGraph: {
    title: "Security & Trust | Xcelsior",
    description:
      "Server-bound action plans, scoped tokens, and a customer-readable audit trail — how an agent gets near real infrastructure safely.",
    url: "https://xcelsior.ca/security",
  },
  twitter: {
    title: "Security & Trust | Xcelsior",
    description:
      "Server-bound action plans, scoped tokens, and a customer-readable audit trail.",
  },
};

export default function SecurityPage() {
  return <SecurityContent />;
}
