import type { DeployStudioForm } from "./types";

export const PRESET_IMAGE = "xcelsior/serverless-vllm:12.4";

/** Phase 15, additional managed engines (vLLM default; TGI/SGLang opt-in). */
export const MANAGED_ENGINES = [
  { id: "vllm", label: "vLLM (default)", image: "xcelsior/serverless-vllm:12.4" },
  { id: "tgi", label: "Text Generation Inference", image: "ghcr.io/huggingface/text-generation-inference:latest" },
  { id: "sglang", label: "SGLang", image: "lmsysorg/sglang:latest" },
] as const;

/** Preset model task: chat (default), embed (/v1/embeddings), or rerank (/v1/rerank). */
export type PresetTask = "chat" | "embed" | "rerank";

export interface TokenPricingQuote {
  input_price_cad_per_m?: number;
  output_price_cad_per_m?: number;
  cached_input_price_cad_per_m?: number;
  token_billing?: boolean;
}

/** Format per-million rates from API pricing (/api/v2/serverless/preset-token-pricing). */
export function formatTokenRateFromPricing(
  modelRef: string,
  pricing?: TokenPricingQuote | null,
): string | null {
  const task = PRESET_MODELS.find((m) => m.id === modelRef)?.task ?? "chat";
  if (task === "rerank") return null;
  const inRate = pricing?.input_price_cad_per_m;
  const outRate = pricing?.output_price_cad_per_m;
  if (inRate != null && outRate != null) {
    return `$${inRate.toFixed(2)} / 1M in · $${outRate.toFixed(2)} / 1M out`;
  }
  return null;
}

/** Preset LLM SKUs (chat, embed, rerank) — rates from /api/v2/serverless/preset-token-pricing. */
export const PRESET_MODELS = [
  { id: "Qwen/Qwen3-8B", label: "Qwen3 8B (default)", vram: 16, task: "chat" as PresetTask },
  { id: "BAAI/bge-m3", label: "BGE-M3 (embeddings)", vram: 8, task: "embed" },
  { id: "BAAI/bge-reranker-v2-m3", label: "BGE Reranker v2 (rerank)", vram: 8, task: "rerank" },
] as const;

export const DEPLOY_STUDIO_STEPS = [
  { id: "method", labelKey: "dash.serverless.step_method" },
  { id: "source", labelKey: "dash.serverless.step_source" },
  { id: "hardware", labelKey: "dash.serverless.step_hardware" },
  { id: "scaling", labelKey: "dash.serverless.step_scaling" },
  { id: "env", labelKey: "dash.serverless.step_env" },
  { id: "review", labelKey: "dash.serverless.step_review" },
] as const;

export const DEFAULT_FORM: DeployStudioForm = {
  name: "",
  method: "preset",
  managedEngine: "vllm",
  customSource: "docker",
  githubRepo: "",
  githubBranch: "main",
  modelRef: PRESET_MODELS[0].id,
  imageRef: PRESET_IMAGE,
  startupCommand: "",
  httpPort: 8000,
  healthCheckPath: "/",
  registryAuth: "",
  cudaVersion: "12.4",
  gpuTier: "",
  gpuCount: 1,
  region: "ca-east",
  minWorkers: 1,
  maxWorkers: 3,
  maxConcurrency: 1,
  idleTimeoutSec: 60,
  executionMode: "sync",
  queueTimeoutSec: 120,
  scalingPolicyType: "queue_delay",
  scalingPolicyValue: 4,
  envRows: [{ key: "", value: "" }],
};

export const IDLE_TIMEOUT_OPTIONS = [
  { value: 60, labelKey: "dash.serverless.idle_1m" },
  { value: 300, labelKey: "dash.serverless.idle_5m" },
  { value: 900, labelKey: "dash.serverless.idle_15m" },
  { value: 1800, labelKey: "dash.serverless.idle_30m" },
  { value: 3600, labelKey: "dash.serverless.idle_1h" },
] as const;
