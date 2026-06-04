import type { Metadata } from "next";
import { PrivacyContent } from "./content";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description:
    "How Xcelsior Computing Inc. collects, uses, and protects your personal information under PIPEDA and Canadian provincial privacy law.",
  alternates: { canonical: "https://xcelsior.ca/privacy" },
  openGraph: {
    title: "Privacy Policy | Xcelsior",
    description:
      "How Xcelsior protects your personal information under PIPEDA and Canadian privacy law.",
    url: "https://xcelsior.ca/privacy",
  },
  twitter: {
    title: "Privacy Policy | Xcelsior",
    description:
      "How Xcelsior protects your personal information under PIPEDA and Canadian privacy law.",
  },
};

export default function PrivacyPage() {
  return <PrivacyContent />;
}
