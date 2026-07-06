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

/** Size-tiered token rates (CAD per 1M tokens) — mirrors serverless/metering.py bands. */
export function tokenRatesForModel(modelRef: string): {
  inputPerM: number;
  outputPerM: number;
  cachedInputPerM: number;
} {
  const s = modelRef.toLowerCase();
  const paramsMatch = s.match(/(\d+(?:\.\d+)?)\s*b\b/);
  let paramsB = paramsMatch ? parseFloat(paramsMatch[1]) : null;
  if (/deepseek|mixtral|8x7b|8x22b/.test(s)) paramsB = 999;
  let inRate = 0.5;
  let outRate = 1.5;
  if (paramsB != null) {
    if (paramsB <= 9) { inRate = 0.15; outRate = 0.45; }
    else if (paramsB <= 34) { inRate = 0.35; outRate = 1.05; }
    else if (paramsB <= 80) { inRate = 0.70; outRate = 2.10; }
    else { inRate = 1.10; outRate = 3.30; }
  }
  return { inputPerM: inRate, outputPerM: outRate, cachedInputPerM: inRate * 0.5 };
}

export function formatTokenRateLine(modelRef: string): string | null {
  const task = PRESET_MODELS.find((m) => m.id === modelRef)?.task ?? "chat";
  if (task === "rerank") return null;
  const { inputPerM, outputPerM } = tokenRatesForModel(modelRef);
  return `$${inputPerM.toFixed(2)} / 1M in · $${outputPerM.toFixed(2)} / 1M out`;
}

export const PRESET_MODELS = [
  // ── Chat / completions ──
  { id: "Qwen/Qwen3-8B", label: "Qwen3 8B (default)", vram: 16, task: "chat" as PresetTask },
  { id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", label: "DeepSeek R1 Distill 32B", vram: 80, task: "chat" },
  { id: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", label: "DeepSeek R1 Distill 8B", vram: 24, task: "chat" },
  { id: "meta-llama/Llama-3.3-70B-Instruct", label: "Llama 3.3 70B Instruct", vram: 80, task: "chat" },
  { id: "meta-llama/Llama-3.1-8B-Instruct", label: "Llama 3.1 8B Instruct", vram: 16, task: "chat" },
  { id: "Qwen/Qwen2.5-Coder-32B-Instruct", label: "Qwen 2.5 Coder 32B", vram: 80, task: "chat" },
  { id: "Qwen/Qwen2.5-7B-Instruct", label: "Qwen 2.5 7B Instruct", vram: 16, task: "chat" },
  { id: "mistralai/Mistral-7B-Instruct-v0.3", label: "Mistral 7B Instruct", vram: 16, task: "chat" },
  { id: "google/gemma-2-9b-it", label: "Gemma 2 9B IT", vram: 20, task: "chat" },
  // ── Embeddings ──
  { id: "BAAI/bge-m3", label: "BGE-M3 (embeddings)", vram: 8, task: "embed" },
  // ── Reranker ──
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
  httpPort: 8080,
  healthCheckPath: "/health",
  registryAuth: "",
  cudaVersion: "12.4",
  gpuTier: "",
  gpuCount: 1,
  region: "ca-east",
  minWorkers: 0,
  maxWorkers: 4,
  maxConcurrency: 4,
  idleTimeoutSec: 300,
  scalingPolicyType: "queue_request_count",
  scalingPolicyValue: 1,
  envRows: [{ key: "", value: "" }],
};

export const IDLE_TIMEOUT_OPTIONS = [
  { value: 60, labelKey: "dash.serverless.idle_1m" },
  { value: 300, labelKey: "dash.serverless.idle_5m" },
  { value: 900, labelKey: "dash.serverless.idle_15m" },
  { value: 1800, labelKey: "dash.serverless.idle_30m" },
  { value: 3600, labelKey: "dash.serverless.idle_1h" },
] as const;