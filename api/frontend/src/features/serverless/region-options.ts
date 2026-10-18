import type { GpuAvailability } from "@/lib/api";

export const DEFAULT_DEPLOY_REGION = "ca-east";

const CA_PROVINCES = new Set([
  "AB",
  "BC",
  "MB",
  "NB",
  "NL",
  "NS",
  "NT",
  "NU",
  "ON",
  "PE",
  "QC",
  "SK",
  "YT",
]);

export function normalizeDeployRegion(region?: string | null, province?: string | null): string {
  const raw = String(region || "").trim().replace(/_/g, "-");
  const provinceCode = String(province || "").trim().toUpperCase();

  if (raw) {
    const upper = raw.toUpperCase();
    if (CA_PROVINCES.has(upper)) return `ca-${upper.toLowerCase()}`;
    if (upper.startsWith("CA-") && CA_PROVINCES.has(upper.slice(3))) {
      return `ca-${upper.slice(3).toLowerCase()}`;
    }
    return raw.toLowerCase();
  }

  if (CA_PROVINCES.has(provinceCode)) return `ca-${provinceCode.toLowerCase()}`;
  return "";
}

export function gpuDeployRegion(gpu: Pick<GpuAvailability, "region" | "province">): string {
  return normalizeDeployRegion(gpu.region, gpu.province);
}

export function regionOptionsForGpus(gpus: GpuAvailability[], gpuTier = ""): string[] {
  const regions = new Set<string>();
  for (const gpu of gpus) {
    if (gpuTier && gpu.gpu_model !== gpuTier) continue;
    const region = gpuDeployRegion(gpu);
    if (region) regions.add(region);
  }
  return regions.size > 0 ? [...regions].sort() : [DEFAULT_DEPLOY_REGION];
}

export function findGpuInRegion(
  gpus: GpuAvailability[],
  gpuTier: string,
  region: string,
): GpuAvailability | undefined {
  const normalizedRegion = normalizeDeployRegion(region) || DEFAULT_DEPLOY_REGION;
  return gpus.find((gpu) => gpu.gpu_model === gpuTier && gpuDeployRegion(gpu) === normalizedRegion);
}
