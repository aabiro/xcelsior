import type { ServerlessEndpoint } from "@/lib/api";

export const mockEndpoint: ServerlessEndpoint = {
  endpoint_id: "sep-test-1",
  owner_id: "owner-1",
  name: "test-llama",
  model_id: "meta-llama/Llama-3.1-8B-Instruct",
  model_name: "meta-llama/Llama-3.1-8B-Instruct",
  model_ref: "meta-llama/Llama-3.1-8B-Instruct",
  gpu_type: "RTX 4090",
  gpu_tier: "RTX 4090",
  gpu_count: 1,
  region: "ca-east",
  docker_image: "xcelsior/serverless-vllm:12.4",
  image_ref: "xcelsior/serverless-vllm:12.4",
  mode: "preset",
  status: "active",
  min_workers: 0,
  max_workers: 4,
  max_concurrency: 4,
  total_requests: 42,
  total_gpu_seconds: 120,
  total_cost_cad: 2.5,
  openai_base_url: "/v1/serverless/sep-test-1/openai/v1",
  created_at: 1_700_000_000,
  updated_at: 1_700_000_100,
};

export const writerUser = {
  user_id: "user-writer",
  email: "writer@xcelsior.ca",
  role: "user",
  customer_id: "cust-writer",
  team_can_write_instances: true,
};

export const viewerUser = {
  user_id: "user-viewer",
  email: "viewer@xcelsior.ca",
  role: "user",
  customer_id: "cust-viewer",
  team_id: "team-1",
  team_role: "viewer",
  team_name: "Acme",
  team_can_write_instances: false,
};

