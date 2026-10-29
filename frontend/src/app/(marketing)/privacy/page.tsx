import type { Metadata } from "next";
import { PrivacyContent } from "./content";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description:
    "How Xcelsior Compute Inc. collects, uses, and protects your personal information.",
  alternates: { canonical: "https://xcelsior.ca/privacy" },
  openGraph: {
    title: "Privacy Policy | Xcelsior",
    description:
      "How Xcelsior protects your personal information.",
    url: "https://xcelsior.ca/privacy",
  },
  twitter: {
    title: "Privacy Policy | Xcelsior",
    description:
      "How Xcelsior protects your personal information.",
  },
};

export default function PrivacyPage() {
  return <PrivacyContent />;
}
