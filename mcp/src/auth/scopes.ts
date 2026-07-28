export type McpScope =
  | "api"
  | "instances:read"
  | "instances:write"
  | "instances:operate"
  | "billing:read"
  | "billing:write"
  | "gpu:read"
  | "marketplace:read"
  | "hosts:read"
  | "hosts:operate"
  | "hosts:evict"
  | "control_plane:read"
  | "control_plane:operate"
  | "mcp_actions:approve"
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
  explain_instance_placement: ["instances:read", "api"],
  simulate_instance_placement: ["instances:read", "gpu:read", "api"],
  get_instance_timeline: ["instances:read", "api"],
  get_active_lease: ["instances:read", "api"],
  get_scheduler_health: ["control_plane:read", "api"],
  get_host_capacity: ["hosts:read", "api"],
  list_reconciliation_findings: ["instances:read", "control_plane:read", "api"],
  get_mcp_action_status: ["instances:read", "inference:read", "hosts:read", "api"],
  retry_instance: ["instances:operate", "api"],
  reconcile_instance: ["instances:operate", "api"],
  drain_host: ["hosts:operate", "api"],
  undrain_host: ["hosts:operate", "api"],
  evict_host_workloads: ["hosts:evict", "api"],
  retry_agent_command: ["control_plane:operate", "api"],
  get_wallet_balance: ["billing:read", "api"],
  estimate_job_cost: ["billing:read", "api"],
  list_invoices: ["billing:read", "api"],
};

export function userHasScope(userScopes: string[] | undefined, required: McpScope[]): boolean {
  if (!userScopes?.length || !required.length) return false;
  if (userScopes.includes("api")) return true;
  return required.some((s) => userScopes.includes(s));
}
