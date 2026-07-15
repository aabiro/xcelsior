import type { Metadata } from "next";
import dynamic from "next/dynamic";

const AboutContent = dynamic(
  () => import("./content").then((mod) => mod.AboutContent),
  { loading: () => <div className="min-h-[40vh]" aria-hidden /> },
);

export const metadata: Metadata = {
  title: "About Xcelsior",
  description:
    "Xcelsior is an agent-native GPU compute marketplace with transparent pricing, real-time telemetry, and infrastructure for teams worldwide.",
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
