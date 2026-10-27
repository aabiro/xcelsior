import type { Metadata } from "next";
import { Suspense } from "react";
import { DeleteDataContent } from "@/components/marketing/account-removal";

export const metadata: Metadata = {
  title: "Delete your data",
  description: "How to delete your Xcelsior account and check a Facebook data-deletion request.",
  alternates: { canonical: "https://xcelsior.ca/delete-data" },
  openGraph: {
    title: "Delete your data | Xcelsior",
    description: "Request deletion of your Xcelsior account and check the status of a confirmation code.",
    url: "https://xcelsior.ca/delete-data",
  },
};

export default function DeleteDataPage() {
  return (
    <Suspense fallback={null}>
      <DeleteDataContent />
    </Suspense>
  );
}
