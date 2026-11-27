/**
 * Human-readable tool titles, in one reviewed place.
 *
 * The Connectors Directory review requires a `title` on every tool. Clients show
 * it instead of the snake_case name in confirmation prompts and tool pickers,
 * so it is the words a user reads when Claude asks "allow Terminate instance?".
 * Until 2026-11-24 no tool had one. A directory submission would have failed on
 * that alone, whatever else passed.
 *
 * Keyed by `ToolName` for the same reason descriptions are: a tool with no
 * title fails to compile, and so does a title for a tool that no longer exists.
 * `installToolAudit` puts the title on the registration, so no `registerTool`
 * call sets one itself.
 *
 * Style: sentence case, a verb first for writes, at most 40 characters, no
 * trailing punctuation. `tests/unit/titles.test.ts` enforces it.
 */

import type { ToolName } from "../auth/scopes.js";

const TITLES: Record<ToolName, string> = {
  // Discovery and pricing
  list_available_gpus: "List available GPUs",
  get_spot_prices: "Get spot prices",
  get_pricing_reference: "Get pricing reference",
  search_marketplace: "Search the GPU marketplace",
  get_host_evidence: "Get host evidence",
  list_tiers: "List service tiers",

  // Permissions
  get_permissions: "Get connection permissions",
  request_permission: "Request more permissions",

  // Instances
  list_instances: "List instances",
  get_instance: "Get instance",
  get_instance_logs: "Get instance logs",
  create_instance: "Launch a GPU instance",
  cancel_instance: "Cancel instance",
  terminate_instance: "Terminate instance",
  stop_instance: "Stop instance",
  should_i_run_this: "Check whether to run a job",
  run_training_job: "Run a training job",
  schedule_under_budget: "Schedule a job under budget",
  watch_instance: "Watch instance status",
  open_instance_access: "Open instance access",
  retry_instance: "Retry instance",
  reconcile_instance: "Reconcile instance",
  get_active_lease: "Get active lease",
  get_mcp_action_status: "Get approval status",

  // SSH keys
  register_ssh_key: "Add SSH key",
  list_ssh_keys: "List SSH keys",
  delete_ssh_key: "Remove SSH key",

  // Volumes and artifacts
  list_volumes: "List volumes",
  get_volume: "Get volume",
  create_volume: "Create volume",
  attach_volume: "Attach volume",
  detach_volume: "Detach volume",
  delete_volume: "Delete volume",
  snapshot_volume: "Snapshot volume",
  list_volume_snapshots: "List volume snapshots",
  restore_volume_snapshot: "Restore volume snapshot",
  get_artifact_expiry: "Get artifact expiry",
  promote_artifact_to_volume: "Copy artifact to volume",
  get_promotion_status: "Get artifact copy status",

  // Pipelines and launch plans
  run_pipeline: "Run a pipeline",
  revoke_launch_plan: "Withdraw launch plan",
  get_pipeline_status: "Get pipeline status",

  // Serverless inference
  list_serverless_endpoints: "List serverless endpoints",
  create_serverless_endpoint: "Create serverless endpoint",
  should_i_run_pel_job: "Check whether to run an inference job",
  run_serverless_job: "Run inference job",
  get_serverless_job_status: "Get inference job status",
  get_serverless_endpoint: "Get serverless endpoint",
  get_serverless_endpoint_health: "Get serverless endpoint health",
  list_serverless_endpoint_jobs: "List endpoint jobs",
  cancel_serverless_job: "Cancel inference job",
  delete_serverless_endpoint: "Delete serverless endpoint",

  // Placement and history
  explain_instance_placement: "Explain instance placement",
  simulate_instance_placement: "Simulate instance placement",
  evaluate_placement_preference: "Evaluate placement preference",
  get_instance_timeline: "Get instance timeline",
  get_instance_receipt: "Get instance receipt",
  get_event_history: "Get event history",

  // Providers and reputation
  list_providers: "List providers",
  get_provider_account: "Get provider account",
  get_provider_earnings: "Get provider earnings",
  get_my_reputation: "Get my reputation",
  get_reputation_journey: "Get reputation journey",
  get_trust_tiers: "Get trust tiers",
  get_reputation_breakdown: "Get reputation breakdown",
  get_reputation_leaderboard: "Get reputation leaderboard",
  get_reputation_history: "Get reputation history",
  claim_reputation_milestones: "Claim reputation milestones",
  get_paypal_status: "Get PayPal payout status",
  get_wise_status: "Get Wise payout status",
  register_provider: "Register as a provider",
  request_provider_payout: "Request provider payout",

  // Wallet and billing
  get_wallet_balance: "Get wallet balance",
  get_wallet_history: "Get wallet history",
  list_funding_options: "List funding options",
  create_crypto_deposit: "Create Bitcoin deposit",
  get_crypto_deposit: "Get Bitcoin deposit",
  create_lightning_deposit: "Create Lightning deposit",
  get_lightning_deposit: "Get Lightning deposit",
  get_spend_envelope: "Get spend limits",
  estimate_job_cost: "Estimate job cost",
  list_invoices: "List invoices",
  list_payment_methods: "List payment methods",
  get_auto_topup: "Get auto top-up settings",
  list_pending_verifications: "List pending verifications",
  top_up_wallet: "Top up wallet",
  configure_auto_topup: "Configure auto top-up",

  // Images
  create_instance_snapshot: "Snapshot instance to image",
  list_user_images: "List saved images",
  delete_user_image: "Delete saved image",
  create_image_sweep: "Compare an image across hosts",
  get_image_sweep: "Get image comparison",

  // Operator (never on the public connector)
  get_scheduler_health: "Get scheduler health",
  get_host_capacity: "Get host capacity",
  list_reconciliation_findings: "List reconciliation findings",
  list_hpc_clusters: "List HPC clusters",
  list_hpc_jobs: "List HPC jobs",
  get_hpc_job: "Get HPC job",
  get_hpc_job_output: "Get HPC job output",
  submit_hpc_job: "Submit HPC job",
  cancel_hpc_job: "Cancel HPC job",
  drain_host: "Drain host",
  undrain_host: "Undrain host",
  evict_host_workloads: "Evict host workloads",
  retry_agent_command: "Retry agent command",

  // Company knowledge (off by default)
  search: "Search Xcelsior docs",
  fetch: "Fetch Xcelsior doc",
};

export const TOOL_TITLES: Readonly<Record<string, string>> = TITLES;
