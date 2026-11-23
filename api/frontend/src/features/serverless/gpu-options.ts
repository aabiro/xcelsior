import type { GpuAvailability, GpuInventoryRow } from "@/lib/api";
import { gpuDeployRegion, normalizeDeployRegion } from "./region-options";

/**
 * GPU and region choices for a serverless deploy, built only from what the API
 * reports. There is no fallback catalog: an empty fleet is an empty list, and
 * the form says so.
 *
 * `inventory` is every admitted GPU, free or busy; `gpus` is the free subset
 * with its price. An endpoint can target a busy model, because its workers
 * queue until one frees up, so the picker lists the whole inventory and labels
 * what is free.
 */
export interface DeployGpuOption {
  model: string;
  vramGb: number;
  available: number;
  total: number;
  regions: string[];
}

export interface DeployGpuData {
  gpus: GpuAvailability[];
  inventory?: GpuInventoryRow[];
  /** The request failed: say so, rather than "no GPUs". */
  failed?: boolean;
}

export const EMPTY_DEPLOY_GPU_DATA: DeployGpuData = { gpus: [], inventory: [] };

/** Load the inventory for a deploy form. A failure is reported, never an empty fleet. */
export async function loadDeployGpuData(
  fetcher: () => Promise<{ gpus?: GpuAvailability[]; inventory?: GpuInventoryRow[] }>,
): Promise<DeployGpuData> {
  try {
    const res = await fetcher();
    return { gpus: res.gpus ?? [], inventory: res.inventory ?? [] };
  } catch {
    return { gpus: [], inventory: [], failed: true };
  }
}

export function deployGpuOptions({ gpus, inventory }: DeployGpuData): DeployGpuOption[] {
  const rows: { model: string; vram: number; available: number; total: number; region: string }[] =
    inventory && inventory.length > 0
      ? inventory.map((row) => ({
          model: row.gpu_model,
          vram: row.vram_gb,
          available: row.count_available,
          total: row.count_total,
          region: gpuDeployRegion(row),
        }))
      : gpus.map((row) => ({
          model: row.gpu_model,
          vram: row.vram_gb,
          available: row.count_available,
          total: row.count_available,
          region: gpuDeployRegion(row),
        }));

  const byModel = new Map<string, DeployGpuOption>();
  for (const row of rows) {
    if (!row.model || row.total <= 0) continue;
    const option = byModel.get(row.model) ?? {
      model: row.model,
      vramGb: row.vram,
      available: 0,
      total: 0,
      regions: [],
    };
    option.vramGb = Math.max(option.vramGb, row.vram || 0);
    option.available += row.available;
    option.total += row.total;
    if (row.region && !option.regions.includes(row.region)) option.regions.push(row.region);
    byModel.set(row.model, option);
  }
  return [...byModel.values()]
    .map((option) => ({ ...option, regions: [...option.regions].sort() }))
    .sort((a, b) => (b.available > 0 ? 1 : 0) - (a.available > 0 ? 1 : 0) || a.model.localeCompare(b.model));
}

export function deployRegionOptions(options: DeployGpuOption[], model: string): string[] {
  return options.find((option) => option.model === model)?.regions ?? [];
}

/** The model to preselect: the first one with a free GPU, else none. */
export function defaultDeployGpu(options: DeployGpuOption[]): string {
  return options.find((option) => option.available > 0)?.model ?? "";
}

/** The free row behind a choice, for its price. Busy models have no price row. */
export function freeGpuRow(
  gpus: GpuAvailability[],
  model: string,
  region: string,
): GpuAvailability | undefined {
  const wanted = normalizeDeployRegion(region);
  return gpus.find((gpu) => gpu.gpu_model === model && (!wanted || gpuDeployRegion(gpu) === wanted));
}

/** Pick a region that exists for `model`, keeping the current one when it does. */
export function regionFor(options: DeployGpuOption[], model: string, current: string): string {
  const regions = deployRegionOptions(options, model);
  return regions.includes(current) ? current : regions[0] ?? "";
}
