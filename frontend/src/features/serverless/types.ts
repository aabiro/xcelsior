export type DeployMethod = "preset" | "custom";

export type ScalingPolicyType = "queue_request_count" | "queue_delay";

export interface EnvRow {
  key: string;
  value: string;
}

export interface DeployStudioForm {
  name: string;
  method: DeployMethod;
  modelRef: string;
  imageRef: string;
  startupCommand: string;
  httpPort: number;
  healthCheckPath: string;
  registryAuth: string;
  cudaVersion: string;
  gpuTier: string;
  gpuCount: number;
  region: string;
  minWorkers: number;
  maxWorkers: number;
  maxConcurrency: number;
  idleTimeoutSec: number;
  scalingPolicyType: ScalingPolicyType;
  scalingPolicyValue: number;
  envRows: EnvRow[];
}

export type DetailTab = "overview" | "workers" | "jobs" | "tryit" | "keys";