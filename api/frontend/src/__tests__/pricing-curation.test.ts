import { describe, expect, it } from "vitest";
import { curatePricingGpus, type PricingCardRow } from "@/lib/marketing-gpu";

const row = (model: string, vram: number, onDemand: number): PricingCardRow => ({
  model,
  vram,
  onDemand,
  spot: +(onDemand * 0.4).toFixed(3),
  reserved1m: +(onDemand * 0.8).toFixed(3),
  reserved1y: +(onDemand * 0.55).toFixed(3),
});

const MARQUEE = [
  row("H100", 80, 3.5),
  row("A100", 80, 2.2),
  row("A100", 40, 1.5),
  row("L40S", 48, 1.8),
  row("RTX 4090", 24, 0.6),
  row("RTX 5090", 32, 0.9),
  row("RTX 3090", 24, 0.62),
  row("L4", 24, 0.7),
  row("A40", 48, 1.1),
  row("RTX 6000 Ada", 48, 1.3),
  row("H200", 141, 4.2),
];

describe("curatePricingGpus", () => {
  it("keeps the cheapest GPU so the From price is the real minimum", () => {
    const rows = [...MARQUEE, row("RTX 3060", 12, 0.12), row("RTX 2060", 6, 0.08)];
    const curated = curatePricingGpus(rows, 10);
    expect(curated).toHaveLength(10);
    expect(Math.min(...curated.map((r) => r.onDemand))).toBe(0.08);
    expect(curated.some((r) => r.model.includes("2060"))).toBe(true);
  });

  it("leaves a short list alone", () => {
    const rows = [row("RTX 4090", 24, 0.6), row("RTX 2060", 6, 0.08)];
    expect(curatePricingGpus(rows, 10)).toHaveLength(2);
  });
});
