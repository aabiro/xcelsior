import { getGpuModel } from "./gpu-models";

/** Canonical marketing order — curated fleet, not every SKU in the catalog. */
export const MARKETING_GPU_PRIORITY = [
  "B200",
  "H200",
  "H100",
  "H100 NVL",
  "A100",
  "L40S",
  "L40",
  "L4",
  "A40",
  "RTX 5090",
  "RTX 4090",
  "RTX 3090",
  "T4",
  "MI300X",
] as const;

/** Fix common API / host mislabels before display. */
const MODEL_ALIASES: Record<string, string> = {
  L4OS: "L40S",
  l4os: "L40S",
  "L4 OS": "L40S",
  "NVIDIA L40S": "L40S",
  "NVIDIA L40": "L40",
  "NVIDIA L4": "L4",
};

export interface MarketingGpuRow {
  gpu_model: string;
  available: number;
  total: number;
  vram_gb: number;
  price_cad: number;
  spot_cad: number;
  locations: string[];
}

export function normalizeGpuModel(raw: string, vram_gb?: number): string {
  const trimmed = raw.trim();
  const alias = MODEL_ALIASES[trimmed];
  if (alias) return alias;
  // Mislabeled L4 with 48GB VRAM is almost always L40S
  if (trimmed === "L4" && vram_gb === 48) return "L40S";
  const known = getGpuModel(trimmed);
  return known?.value ?? trimmed;
}

/** User-facing short label (canonical catalog title). */
export function marketingGpuLabel(model: string): string {
  const normalized = normalizeGpuModel(model);
  return getGpuModel(normalized)?.label ?? normalized;
}

function priorityIndex(model: string): number {
  const idx = MARKETING_GPU_PRIORITY.indexOf(model as (typeof MARKETING_GPU_PRIORITY)[number]);
  return idx === -1 ? 999 : idx;
}

/**
 * Curate fleet for marketing: priority tiers first, then at most two
 * in-stock extras, capped at 12 rows.
 */
export function curateMarketingGpus(rows: MarketingGpuRow[], max = 12): MarketingGpuRow[] {
  const byModel = new Map<string, MarketingGpuRow>();

  for (const row of rows) {
    const model = normalizeGpuModel(row.gpu_model, row.vram_gb);
    const existing = byModel.get(model);
    if (!existing) {
      byModel.set(model, { ...row, gpu_model: model });
      continue;
    }
    existing.available += row.available;
    existing.total += row.total;
    existing.vram_gb = Math.max(existing.vram_gb, row.vram_gb);
    for (const loc of row.locations) {
      if (!existing.locations.includes(loc)) existing.locations.push(loc);
    }
    if (row.price_cad > 0 && (existing.price_cad === 0 || row.price_cad < existing.price_cad)) {
      existing.price_cad = row.price_cad;
    }
    if (row.spot_cad > 0 && (existing.spot_cad === 0 || row.spot_cad < existing.spot_cad)) {
      existing.spot_cad = row.spot_cad;
    }
  }

  const all = [...byModel.values()];
  const curated: MarketingGpuRow[] = [];

  for (const tier of MARKETING_GPU_PRIORITY) {
    const row = byModel.get(tier);
    if (row) curated.push(row);
    if (curated.length >= max) return curated;
  }

  const extras = all
    .filter((r) => !curated.some((c) => c.gpu_model === r.gpu_model))
    .sort(
      (a, b) =>
        (b.available > 0 ? 1 : 0) - (a.available > 0 ? 1 : 0) ||
        priorityIndex(a.gpu_model) - priorityIndex(b.gpu_model),
    );

  for (const row of extras) {
    if (curated.length >= max) break;
    if (row.available > 0 || curated.length < max - 2) {
      curated.push(row);
    }
  }

  return curated.sort(
    (a, b) =>
      (b.available > 0 ? 1 : 0) - (a.available > 0 ? 1 : 0) ||
      priorityIndex(a.gpu_model) - priorityIndex(b.gpu_model) ||
      a.price_cad - b.price_cad,
  );
}

export type PricingCardRow = {
  model: string;
  vram: number;
  onDemand: number;
  spot: number;
  reserved1m: number;
  reserved1y: number;
};

export function curatePricingGpus(rows: PricingCardRow[], max = 10): PricingCardRow[] {
  const normalized = rows.map((r) => ({
    ...r,
    model: normalizeGpuModel(r.model, r.vram),
  }));
  const byModel = new Map<string, PricingCardRow>();
  for (const row of normalized) {
    if (!byModel.has(row.model)) byModel.set(row.model, row);
  }

  const curated: PricingCardRow[] = [];
  for (const tier of MARKETING_GPU_PRIORITY) {
    const row = byModel.get(tier);
    if (row) curated.push(row);
    if (curated.length >= max) return curated;
  }
  for (const row of byModel.values()) {
    if (curated.some((c) => c.model === row.model)) continue;
    curated.push(row);
    if (curated.length >= max) break;
  }
  return curated;
}

export function gpuTierBadge(model: string): "flagship" | "datacenter" | "pro" | "value" {
  if (/^(B200|H200|H100)/i.test(model)) return "flagship";
  if (/^(A100|L40S|L40|L4|A40|MI)/i.test(model)) return "datacenter";
  if (/^RTX (5090|4090|6000)/i.test(model)) return "pro";
  return "value";
}