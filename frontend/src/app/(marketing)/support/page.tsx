import type { Metadata } from "next";
import dynamic from "next/dynamic";

const SupportContent = dynamic(
  () => import("./content").then((mod) => mod.SupportContent),
  { loading: () => <div className="min-h-[40vh]" aria-hidden /> },
);

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
