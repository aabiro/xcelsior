import type { Metadata } from "next";
import { TermsContent } from "./content";

export const metadata: Metadata = {
  title: "Terms of Service",
  description:
    "Terms governing your use of the Xcelsior GPU compute marketplace. Canadian jurisdiction, CAD billing, and dispute resolution.",
  alternates: { canonical: "https://xcelsior.ca/terms" },
  openGraph: {
    title: "Terms of Service | Xcelsior",
    description:
      "Terms governing the Xcelsior GPU compute marketplace. Canadian jurisdiction and CAD billing.",
    url: "https://xcelsior.ca/terms",
  },
  twitter: {
    title: "Terms of Service | Xcelsior",
    description:
      "Terms governing the Xcelsior GPU compute marketplace. Canadian jurisdiction and CAD billing.",
  },
};

import { getServerLocale } from "@/lib/i18n/server";

export default async function TermsPage() {
  const locale = await getServerLocale();
  return <TermsContent locale={locale} />;
}
