import type { Metadata } from "next";
import { AboutContent } from "./content";

export const metadata: Metadata = {
  title: "About Xcelsior",
  description:
    "Xcelsior is a Canada-first GPU compute marketplace with transparent pricing, compliance-aware operations, and infrastructure for teams worldwide.",
  alternates: { canonical: "https://xcelsior.ca/about" },
  openGraph: {
    title: "About Xcelsior",
    description:
      "Canada-first GPU compute marketplace. Built in Canada, open to teams worldwide.",
    url: "https://xcelsior.ca/about",
  },
  twitter: {
    title: "About Xcelsior",
    description:
      "Canada-first GPU compute marketplace. Built in Canada, open to teams worldwide.",
  },
};

export default function AboutPage() {
  return <AboutContent />;
}
