import type { Metadata } from "next";
import { Suspense } from "react";
import { DeactivateContent } from "@/components/marketing/account-removal";

export const metadata: Metadata = {
  title: "Deactivate your account",
  description: "Deactivate your Xcelsior account and start permanent deletion of your personal data.",
  alternates: { canonical: "https://xcelsior.ca/deactivate" },
  openGraph: {
    title: "Deactivate your account | Xcelsior",
    description: "Start permanent deletion of your Xcelsior account.",
    url: "https://xcelsior.ca/deactivate",
  },
};

export default function DeactivatePage() {
  return (
    <Suspense fallback={null}>
      <DeactivateContent />
    </Suspense>
  );
}
