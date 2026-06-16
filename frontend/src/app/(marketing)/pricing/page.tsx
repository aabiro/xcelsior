import type { Metadata } from "next";
import dynamic from "next/dynamic";
import { curatePricingGpus } from "@/lib/marketing-gpu";

const PricingContent = dynamic(
  () => import("./content").then((mod) => mod.PricingContent),
  { loading: () => <div className="min-h-[40vh]" aria-hidden /> },
);

export const metadata: Metadata = {
  title: "Pricing",
  description: "GPU compute pricing in Canadian dollars. From $0.30 CAD/hr, with dynamic spot pricing up to 70% off and reserved plans up to 45% off.",
  alternates: { canonical: "https://xcelsior.ca/pricing" },
  openGraph: {
    title: "Pricing | Xcelsior",
    description: "GPU compute from $0.30 CAD/hr. Spot pricing up to 70% off, reserved plans up to 45% off.",
    url: "https://xcelsior.ca/pricing",
  },
  twitter: {
    title: "Pricing | Xcelsior",
    description: "GPU compute from $0.30 CAD/hr. Spot pricing up to 70% off, reserved plans up to 45% off.",
  },
};

interface GpuPricing {
  gpu_model: string;
  vram_gb?: number;
  base_rate_cad: number;
  spot_cad?: number;
  reserved_1mo_cad?: number;
  reserved_3mo_cad?: number;
  reserved_1yr_cad?: number;
  subsidized_starter_cad?: number;
  min_rate_cad?: number;
  premium_rate_cad?: number;
}

async function fetchPricing(): Promise<GpuPricing[]> {
  const BACKEND = process.env.NEXT_PUBLIC_API_URL || "https://xcelsior.ca";
  try {
    const res = await fetch(`${BACKEND}/api/pricing/reference`, {
      next: { revalidate: 3600 },
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return [];
    const data = await res.json();
    const pricing = data.pricing || {};
    return Object.entries(pricing).map(([model, rates]) => ({
      gpu_model: model,
      ...(rates as Omit<GpuPricing, "gpu_model">),
    }));
  } catch {
    return [];
  }
}

async function fetchLiveSpotRates(): Promise<Record<string, number>> {
  const BACKEND = process.env.NEXT_PUBLIC_API_URL || "https://xcelsior.ca";
  try {
    const res = await fetch(`${BACKEND}/spot-prices`, {
      next: { revalidate: 300 },
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return {};
    const data = await res.json();
    return (data.prices || data.spot_prices || {}) as Record<string, number>;
  } catch {
    return {};
  }
}

const FALLBACK_GPUS = [
  { model: "RTX 3090", vram: 24, onDemand: 0.30, spot: 0.12, reserved1m: 0.24, reserved1y: 0.17 },
  { model: "RTX 4090", vram: 24, onDemand: 0.55, spot: 0.22, reserved1m: 0.44, reserved1y: 0.30 },
  { model: "A100 40GB", vram: 40, onDemand: 1.50, spot: 0.60, reserved1m: 1.20, reserved1y: 0.83 },
  { model: "A100 80GB", vram: 80, onDemand: 2.20, spot: 0.88, reserved1m: 1.76, reserved1y: 1.21 },
  { model: "H100 80GB", vram: 80, onDemand: 3.50, spot: 1.40, reserved1m: 2.80, reserved1y: 1.93 },
  { model: "L40S", vram: 48, onDemand: 1.80, spot: 0.72, reserved1m: 1.44, reserved1y: 0.99 },
];

export default async function PricingPage() {
  const [apiPricing, liveSpot] = await Promise.all([fetchPricing(), fetchLiveSpotRates()]);

  const gpus = curatePricingGpus(
    apiPricing.length > 0
      ? apiPricing
          .filter((p) => p.base_rate_cad > 0)
          .map((p) => ({
            model: p.gpu_model,
            vram: p.vram_gb ?? (p.gpu_model.includes("A100 80") ? 80 : p.gpu_model.includes("A100") ? 40 : p.gpu_model.includes("H100") ? 80 : p.gpu_model.includes("L40") ? 48 : 24),
            onDemand: p.base_rate_cad,
            spot: +(liveSpot[p.gpu_model] ?? p.spot_cad ?? p.min_rate_cad ?? p.base_rate_cad * 0.4).toFixed(2),
            reserved1m: +(p.reserved_1mo_cad ?? p.base_rate_cad * 0.8).toFixed(2),
            reserved1y: +(p.reserved_1yr_cad ?? p.base_rate_cad * 0.55).toFixed(2),
          }))
      : FALLBACK_GPUS,
    10,
  );

  const productJsonLd = {
    "@context": "https://schema.org",
    "@type": "Product",
    name: "Xcelsior GPU Compute",
    description: "Sovereign GPU compute marketplace. On-demand, spot, and reserved pricing in CAD.",
    brand: { "@type": "Brand", name: "Xcelsior" },
    offers: gpus.map((g) => ({
      "@type": "Offer",
      name: `${g.model} On-Demand`,
      price: g.onDemand.toFixed(2),
      priceCurrency: "CAD",
      unitText: "per hour",
      availability: "https://schema.org/InStock",
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
        acceptedAnswer: { "@type": "Answer", text: "Reserved 1-year plans save up to 45% compared to on-demand pricing." },
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
