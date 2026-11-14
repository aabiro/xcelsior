// gpu-choices.ts — The GPUs a renter can pick, built from the live feed.
//
// The wizard used to list `/marketplace/search` rows: every listing ever
// written, including test listings for hosts that were offline or never
// existed (an H100 and a B200 at $0.20/hr). Launching on one failed, and its
// price was not the price the server charged. These choices come from
// `/api/v2/gpu/available`, the snapshot the scheduler places from, priced at
// the rate a launch is charged. A launch names the model and VRAM and the
// scheduler picks the machine, as the dashboard's launch dialog does.

import type { AvailableGpu } from "./api-client.js";

export interface GpuChoice {
    /** Select value and checkpoint key; it encodes the model and VRAM. */
    id: string;
    gpu_model: string;
    vram_gb: number;
    /** Lowest rate across the free GPUs of this kind, per hour. */
    price_per_hour: number;
    count_available: number;
    regions: string[];
}

export type GpuPreference = "best" | "cheapest" | "specific" | string;

const SEPARATOR = "|";

export function gpuChoiceId(gpuModel: string, vramGb: number): string {
    return `${gpuModel}${SEPARATOR}${vramGb}`;
}

/** The model and VRAM a pick names, readable even when the list is gone. */
export function parseGpuChoiceId(id: string | undefined): { gpu_model: string; vram_gb: number } | null {
    if (!id) return null;
    const at = id.lastIndexOf(SEPARATOR);
    if (at <= 0) return null;
    const vram = Number(id.slice(at + 1));
    const model = id.slice(0, at).trim();
    if (!model || !Number.isFinite(vram) || vram <= 0) return null;
    return { gpu_model: model, vram_gb: vram };
}

function roundVram(vramGb: number): number {
    return Math.round(vramGb * 10) / 10;
}

/**
 * Group the feed by model and VRAM. A GPU without a price is left out: the
 * server refuses to launch it, so offering it only leads to an error.
 */
export function buildGpuChoices(gpus: AvailableGpu[]): GpuChoice[] {
    const byKey = new Map<string, GpuChoice>();
    for (const gpu of gpus) {
        const model = String(gpu.gpu_model ?? "").trim();
        const vram = roundVram(Number(gpu.vram_gb));
        const price = Number(gpu.price_per_hour_cad);
        const count = Math.max(0, Math.floor(Number(gpu.count_available) || 0));
        if (!model || !(vram > 0) || count === 0) continue;
        if (gpu.service_tiers_available === false || !Number.isFinite(price) || price <= 0) continue;
        const id = gpuChoiceId(model, vram);
        const existing = byKey.get(id);
        const region = String(gpu.region ?? "").trim();
        if (existing) {
            existing.count_available += count;
            existing.price_per_hour = Math.min(existing.price_per_hour, price);
            if (region && !existing.regions.includes(region)) existing.regions.push(region);
        } else {
            byKey.set(id, {
                id, gpu_model: model, vram_gb: vram, price_per_hour: price,
                count_available: count, regions: region ? [region] : [],
            });
        }
    }
    return [...byKey.values()];
}

/** Count of free GPUs the feed reported that cannot be launched yet (no price). */
export function unpricedGpuCount(gpus: AvailableGpu[]): number {
    return gpus
        .filter((gpu) => gpu.service_tiers_available === false || !(Number(gpu.price_per_hour_cad) > 0))
        .reduce((sum, gpu) => sum + Math.max(0, Math.floor(Number(gpu.count_available) || 0)), 0);
}

/** Order for the chosen preference: cheapest first, most VRAM first, or by name. */
export function sortGpuChoices(choices: GpuChoice[], preference: GpuPreference): GpuChoice[] {
    const sorted = [...choices];
    if (preference === "cheapest") {
        sorted.sort((a, b) => a.price_per_hour - b.price_per_hour || b.vram_gb - a.vram_gb);
    } else if (preference === "best") {
        sorted.sort((a, b) => b.vram_gb - a.vram_gb || a.price_per_hour - b.price_per_hour);
    } else {
        sorted.sort((a, b) => a.gpu_model.localeCompare(b.gpu_model) || a.vram_gb - b.vram_gb);
    }
    return sorted;
}

/** Training wants 24 GB or more. */
export const TRAINING_MIN_VRAM_GB = 24;

export function gpuChoiceLabel(choice: GpuChoice): string {
    const where = choice.regions.length ? ` · ${choice.regions.join(", ")}` : "";
    return `${choice.gpu_model} · ${choice.vram_gb} GB · $${choice.price_per_hour.toFixed(2)}/hr · ${choice.count_available} free${where}`;
}
