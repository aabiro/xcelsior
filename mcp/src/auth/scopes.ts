export type McpScope =
  | "api"
  | "instances:read"
  | "instances:write"
  | "billing:read"
  | "billing:write"
  | "gpu:read"
  | "marketplace:read"
  | "hosts:read"
  | "events:read"
  | "inference:read"
  | "inference:write";

export const TOOL_SCOPES: Record<string, McpScope[]> = {
  list_available_gpus: ["gpu:read", "api"],
  get_spot_prices: ["marketplace:read", "api"],
  get_pricing_reference: ["gpu:read", "api"],
  search_marketplace: ["marketplace:read", "api"],
  list_tiers: ["gpu:read", "api"],
  list_instances: ["instances:read", "api"],
  get_instance: ["instances:read", "api"],
  get_instance_logs: ["instances:read", "api"],
  create_instance: ["instances:write", "api"],
  cancel_instance: ["instances:write", "api"],
  terminate_instance: ["instances:write", "api"],
  should_i_run_this: ["billing:read", "instances:read", "api"],
  run_training_job: ["instances:write", "billing:read", "api"],
  schedule_under_budget: ["instances:write", "gpu:read", "marketplace:read", "api"],
  watch_instance: ["instances:read", "api"],
  list_serverless_endpoints: ["inference:read", "api"],
  create_serverless_endpoint: ["inference:write", "api"],
  should_i_run_pel_job: ["billing:read", "inference:read", "api"],
  run_serverless_job: ["inference:write", "api"],
  get_serverless_job_status: ["inference:read", "api"],
  get_wallet_balance: ["billing:read", "api"],
  estimate_job_cost: ["billing:read", "api"],
  list_invoices: ["billing:read", "api"],
};

export function userHasScope(userScopes: string[] | undefined, required: McpScope[]): boolean {
  if (!userScopes?.length) return true;
  if (userScopes.includes("api")) return true;
  return required.some((s) => userScopes.includes(s));
}