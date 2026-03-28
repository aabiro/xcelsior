import type { Metadata } from "next";
import { AboutContent } from "./content";

export const metadata: Metadata = {
  title: "About Xcelsior",
  description:
    "Xcelsior is a Canadian-owned GPU compute marketplace. Our mission: keep Canada's AI infrastructure sovereign, affordable, and green.",
  alternates: { canonical: "https://xcelsior.ca/about" },
  openGraph: {
    title: "About Xcelsior",
    description:
      "Canadian-owned GPU compute marketplace. Sovereign, affordable, and green AI infrastructure.",
    url: "https://xcelsior.ca/about",
  },
  twitter: {
    title: "About Xcelsior",
    description:
      "Canadian-owned GPU compute marketplace. Sovereign, affordable, and green AI infrastructure.",
  },
};

export default function AboutPage() {
  return <AboutContent />;
}
