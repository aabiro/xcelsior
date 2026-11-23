import type { Metadata } from "next";
import dynamic from "next/dynamic";
import { loadPricing } from "./rates";

const PricingContent = dynamic(
  () => import("./content").then((mod) => mod.PricingContent),
  { loading: () => <div className="min-h-[40vh]" aria-hidden /> },
);

/** Savings in the description come from the rate card, like the page's own figures. */
function describe(spot: number | null, reserved: number | null): string {
  const parts = [
    spot ? `spot up to ${spot}% off` : null,
    reserved ? `reserved plans up to ${reserved}% off` : null,
  ].filter(Boolean);
  return parts.length ? `One published rate per GPU, ${parts.join(", ")}.` : "One published rate per GPU.";
}

export async function generateMetadata(): Promise<Metadata> {
  const { maxSpotSaving, maxReservedSaving } = await loadPricing();
  const summary = describe(maxSpotSaving, maxReservedSaving);
  return {
    title: "Pricing",
    description: `GPU compute pricing. ${summary}`,
    alternates: { canonical: "https://xcelsior.ca/pricing" },
    openGraph: {
      title: "Pricing | Xcelsior",
      description: `GPU compute at ${summary.charAt(0).toLowerCase()}${summary.slice(1)}`,
      url: "https://xcelsior.ca/pricing",
    },
    twitter: {
      title: "Pricing | Xcelsior",
      description: `GPU compute at ${summary.charAt(0).toLowerCase()}${summary.slice(1)}`,
    },
  };
}

export default async function PricingPage() {
  const { rows: gpus, fleet, maxReservedSaving } = await loadPricing();

  const productJsonLd = {
    "@context": "https://schema.org",
    "@type": "Product",
    name: "Xcelsior GPU Compute",
    description: "GPU compute marketplace. On-demand, spot, and reserved pricing.",
    brand: { "@type": "Brand", name: "Xcelsior" },
    offers: gpus.map((g) => ({
      "@type": "Offer",
      name: `${g.model} On-Demand`,
      price: g.onDemand.toFixed(2),
      priceCurrency: "CAD",
      unitText: "per hour",
      // In stock only when the live fleet has one free right now.
      availability: fleet[g.model] ? "https://schema.org/InStock" : "https://schema.org/OutOfStock",
    })),
  };

  const faqJsonLd = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: [
      {
        "@type": "Question",
        name: "What currency are prices in?",
        acceptedAnswer: { "@type": "Answer", text: "All pricing is in Canadian Dollars (CAD). No USD conversion." },
      },
      {
        "@type": "Question",
        name: "How much can I save with reserved pricing?",
        acceptedAnswer: {
          "@type": "Answer",
          text: maxReservedSaving
            ? `Reserved 1-year plans save up to ${maxReservedSaving}% compared to on-demand pricing.`
            : "Reserved plans are priced below on-demand; see the published rates on this page.",
        },
      },
    ],
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(productJsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(faqJsonLd) }}
      />
      <PricingContent gpus={gpus} />
    </>
  );
}
