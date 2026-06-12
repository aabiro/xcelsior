import * as api from "@/lib/api";
import { MANAGED_ENGINES } from "./constants";
import type { DeployStudioForm } from "./types";

export function envToRecord(rows: DeployStudioForm["envRows"]): Record<string, string> | undefined {
  const out: Record<string, string> = {};
  for (const row of rows) {
    const k = row.key.trim();
    if (k) out[k] = row.value;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

export function buildCreateServerlessPayload(form: DeployStudioForm) {
  const presetEngine = MANAGED_ENGINES.find((e) => e.id === form.managedEngine) ?? MANAGED_ENGINES[0];
  const image = form.method === "preset"
    ? presetEngine.image
    : form.customSource === "github"
      ? form.imageRef.trim()
      : form.imageRef.trim();

  return {
    name: form.name.trim() || form.modelRef || form.imageRef || form.githubRepo,
    mode: form.method === "preset" ? "preset" as const : "custom" as const,
    managed_engine: form.method === "preset" ? form.managedEngine : undefined,
    model_name: form.method === "preset" ? form.modelRef.trim() : undefined,
    model_ref: form.method === "preset" ? form.modelRef.trim() : undefined,
    source_type: form.method === "custom" && form.customSource === "github" ? "github" as const : undefined,
    source_ref: form.method === "custom" && form.customSource === "github" ? form.githubRepo.trim() : undefined,
    source_ref_branch: form.method === "custom" && form.customSource === "github" ? form.githubBranch.trim() : undefined,
    gpu_type: form.gpuTier,
    gpu_tier: form.gpuTier,
    gpu_count: form.gpuCount,
    region: form.region,
    docker_image: image || undefined,
    image_ref: image || undefined,
    min_workers: form.minWorkers,
    max_workers: form.maxWorkers,
    max_concurrency: form.maxConcurrency,
    idle_timeout_sec: form.idleTimeoutSec,
    scaling_policy_type: form.scalingPolicyType,
    scaling_policy_value: form.scalingPolicyValue,
    startup_command: form.method === "custom" ? form.startupCommand : undefined,
    http_port: form.method === "custom" ? form.httpPort : undefined,
    health_check_path: form.healthCheckPath,
    cuda_version: form.cudaVersion,
    env: envToRecord(form.envRows),
  };
}

export async function deployServerlessEndpoint(form: DeployStudioForm) {
  return api.createServerlessEndpoint(buildCreateServerlessPayload(form));
}