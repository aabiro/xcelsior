/**
 * Live rehearsal of every published tool.
 *
 * The Connectors Directory review requires every tool to succeed with valid
 * input, and a reviewer will try them through a real connector. So this calls
 * every tool on the customer profile, once, through the MCP SDK, against a
 * real API, PostgreSQL and Redis, with a token obtained the way a directory
 * connector obtains one: Dynamic Client Registration, then authorization code
 * with PKCE through the consent screen.
 *
 * Run it with `api/mcp/scripts/rehearse-live.sh`, which starts that stack from
 * this checkout. Gated on XCELSIOR_MCP_REHEARSAL=1, like the real-stack journey.
 *
 * Every tool must be in REHEARSALS, and the coverage test fails otherwise: a
 * new tool is not published until it has a rehearsal. A rehearsal passes only
 * on a successful result (no isError, no `error`, `ok` not false). Writes go
 * through the same preview, approve, confirm sequence a user sees.
 *
 * Then every tool is put through the ways it should refuse, while tenant A's
 * resources still exist: input that breaks its schema; a connection granted
 * one permission only; an id that does not exist; tenant A's ids used by a
 * second tenant; confirm with a plan nobody approved; and launches by a tenant
 * with no money or card. A refusal passes only if it names its reason and is
 * not a fault (5xx, stack trace, JavaScript error). Every value of every enum
 * option gets its own run. The tools that end things run last.
 */
import { generateKeyPairSync, randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { toolsInProfile } from "../../src/tools/profiles.js";
import { TOOL_SCOPES, satisfiesScope } from "../../src/auth/scopes.js";
import {
  approve,
  bearer,
  connectorToken,
  createSession,
  fixture,
  jsonFetch,
  startMcp,
  stopMcp,
  type Session,
} from "./live-stack.js";

const enabled = process.env.XCELSIOR_MCP_REHEARSAL === "1";
const mcpPort = Number(process.env.XCELSIOR_MCP_E2E_PORT || "18981");
const marker = `rehearsal-${randomUUID().slice(0, 8)}`;
const GPU = "RTX 4090";

type Ctx = Record<string, any>;
type Args = Record<string, unknown>;
type Rehearsal = {
  /** Arguments, built from what earlier rehearsals stored in ctx. */
  args: (ctx: Ctx) => Args;
  /**
   * Writes that take `confirm`: preview with confirm:false, approve the plan
   * if one is returned, then run with confirm:true.
   */
  confirm?: boolean;
  /** Checks on the result; store ids for later rehearsals here. */
  then?: (result: Record<string, any>, ctx: Ctx) => void | Promise<void>;
  /** Work done before the call, outside MCP (placing a job, for instance). */
  before?: (ctx: Ctx) => void | Promise<void>;
};

/** The first value found at any of these dotted paths. */
function pick(value: any, ...paths: string[]): any {
  for (const path of paths) {
    let cursor = value;
    for (const key of path.split(".")) cursor = cursor?.[key];
    if (cursor !== undefined && cursor !== null && cursor !== "") return cursor;
  }
  return undefined;
}

function need(value: any, what: string, result: unknown): any {
  if (value === undefined) throw new Error(`no ${what} in ${JSON.stringify(result).slice(0, 800)}`);
  return value;
}

/** An OpenSSH ed25519 public key, made here so no key file is needed. */
function sshPublicKey(): string {
  const { publicKey } = generateKeyPairSync("ed25519");
  const raw = publicKey.export({ format: "der", type: "spki" }).subarray(-32);
  const field = (b: Buffer) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(b.length);
    return Buffer.concat([len, b]);
  };
  const blob = Buffer.concat([field(Buffer.from("ssh-ed25519")), field(Buffer.from(raw))]);
  return `ssh-ed25519 ${blob.toString("base64")} ${marker}`;
}

/** The shared agent secret the API authenticates workers with in test. */
const agentHeaders = () => bearer(process.env.XCELSIOR_API_TOKEN || "test-token-not-for-production");

/** Report an attempt's status as the worker holding its lease. */
async function reportAttempt(worker: Record<string, any>, status: string, extra: Args = {}): Promise<void> {
  await jsonFetch("/agent/v2/attempts/status", {
    method: "POST",
    headers: agentHeaders(),
    body: JSON.stringify({
      job_id: worker.job_id,
      attempt_id: worker.attempt_id,
      host_id: worker.host_id,
      fencing_token: worker.fencing_token,
      status,
      ...extra,
    }),
  });
}

async function bringToRunning(jobId: string): Promise<Record<string, any>> {
  const worker = fixture("place", jobId, GPU);
  const headers = agentHeaders();
  const claimed = await jsonFetch("/agent/v2/commands/claim", {
    method: "POST",
    headers,
    body: JSON.stringify({ host_id: worker.host_id, worker_session_id: worker.worker_session_id }),
  });
  if (!claimed.commands?.some((c: any) => c.command_id === worker.command_id)) {
    throw new Error(
      `the worker was not handed the launch: placed ${JSON.stringify(worker)}, ` +
        `claimed ${JSON.stringify(claimed).slice(0, 800)}`,
    );
  }
  await jsonFetch("/agent/v2/leases/claim", {
    method: "POST",
    headers,
    body: JSON.stringify({
      lease_id: worker.lease_id,
      job_id: worker.job_id,
      attempt_id: worker.attempt_id,
      host_id: worker.host_id,
      fencing_token: worker.fencing_token,
      worker_session_id: worker.worker_session_id,
    }),
  });
  await reportAttempt(worker, "running");
  return worker;
}

const instanceSpec = { gpu_model: GPU, num_gpus: 1, vram_needed_gb: 8 };

/**
 * In the order they run. Later rehearsals read what earlier ones stored, so
 * the order is a lifecycle: look, launch, use, tidy up.
 */
const REHEARSALS: Record<string, Rehearsal> = {
  // ── Connection and discovery ────────────────────────────────────────
  get_permissions: { args: () => ({}) },
  request_permission: {
    args: () => ({ permissions: ["billing:write"], reason: "Live rehearsal" }),
  },
  list_available_gpus: { args: () => ({}) },
  get_spot_prices: { args: () => ({}) },
  get_pricing_reference: { args: () => ({}) },
  search_marketplace: { args: () => ({ gpu_model: GPU }) },
  list_tiers: { args: () => ({}) },
  estimate_job_cost: { args: () => ({ gpu_model: GPU, duration_hours: 1 }) },
  should_i_run_this: { args: () => ({ gpu_model: GPU, duration_hours: 1 }) },
  get_trust_tiers: { args: () => ({}) },
  get_reputation_leaderboard: { args: () => ({}) },
  get_reputation_journey: { args: () => ({}) },
  get_my_reputation: { args: () => ({}) },
  list_providers: { args: () => ({}) },
  simulate_instance_placement: { args: () => ({ spec: { name: `${marker}-sim`, ...instanceSpec } }) },
  evaluate_placement_preference: {
    args: () => ({
      spec: { name: `${marker}-sim`, ...instanceSpec },
      preference: { min_reliability_pct: 90 },
    }),
  },

  // ── Wallet and billing ──────────────────────────────────────────────
  get_wallet_balance: { args: () => ({}) },
  get_wallet_history: { args: () => ({}) },
  list_funding_options: { args: () => ({}) },
  get_spend_envelope: { args: () => ({}) },
  list_invoices: { args: () => ({}) },
  list_payment_methods: { args: () => ({}) },
  get_auto_topup: { args: () => ({}) },
  list_pending_verifications: { args: () => ({}) },

  // ── SSH keys ────────────────────────────────────────────────────────
  register_ssh_key: {
    confirm: true,
    args: (ctx) => ({ public_key: (ctx.sshKey = sshPublicKey()), name: `${marker}-key` }),
    then: (r, ctx) => {
      ctx.sshKeyId = need(pick(r, "key_id", "id", "key.id", "key.key_id"), "key id", r);
    },
  },
  list_ssh_keys: { args: () => ({}) },

  // ── An instance, end to end ─────────────────────────────────────────
  list_instances: { args: () => ({}) },
  create_instance: {
    confirm: true,
    args: () => ({ name: `${marker}-instance`, ...instanceSpec, idempotency_key: randomUUID() }),
    then: (r, ctx) => {
      ctx.jobId = need(pick(r, "job.job_id", "job_id"), "job id", r);
      ctx.instancePlanId = pick(r, "plan_id", "plan.plan_id");
    },
  },
  get_mcp_action_status: {
    args: (ctx) => ({ plan_id: ctx.instancePlanId }),
  },
  get_instance: {
    // Place it through the real scheduler and bring it to running, as the
    // real-stack journey does, so the tools that need a live workload have one.
    before: async (ctx) => {
      ctx.worker = await bringToRunning(ctx.jobId);
      ctx.hostId = ctx.worker.host_id;
    },
    args: (ctx) => ({ job_id: ctx.jobId }),
    then: (r, ctx) => {
      // retry_instance and reconcile_instance need it, and this is where an
      // agent would read it.
      ctx.jobVersion = need(
        pick(r, "job.version", "version", "instance.version", "instance.job_version"),
        "instance version for retry/reconcile",
        r,
      );
    },
  },
  watch_instance: {
    args: (ctx) => ({
      job_id: ctx.jobId,
      duration_minutes: 1,
      poll_interval_seconds: 10,
      return_on_phase: ["running"],
    }),
  },
  get_instance_logs: { args: (ctx) => ({ job_id: ctx.jobId }) },
  get_instance_timeline: { args: (ctx) => ({ job_id: ctx.jobId }) },
  get_active_lease: { args: (ctx) => ({ job_id: ctx.jobId }) },
  explain_instance_placement: { args: (ctx) => ({ job_id: ctx.jobId }) },
  get_event_history: { args: (ctx) => ({ entity_type: "job", entity_id: ctx.jobId }) },
  get_host_evidence: { args: (ctx) => ({ host_id: ctx.hostId }) },
  get_artifact_expiry: { args: (ctx) => ({ job_id: ctx.jobId }) },
  open_instance_access: { confirm: true, args: (ctx) => ({ job_id: ctx.jobId, method: "auto" }) },
  reconcile_instance: {
    confirm: true,
    args: (ctx) => ({
      job_id: ctx.jobId,
      expected_version: ctx.jobVersion,
      idempotency_key: randomUUID(),
    }),
  },

  // Needs a host on the marketplace, which placing the instance created.
  // The preview prepares a plan; callConfirmed approves it as the user, and
  // the confirm executes that plan, not a fresh search.
  schedule_under_budget: {
    confirm: true,
    args: () => ({ name: `${marker}-budget`, max_hourly_cad: 50, ...instanceSpec }),
    then: (r) => {
      expect(r.launched, JSON.stringify(r).slice(0, 400)).toBe(true);
      need(pick(r, "job_id", "job.job_id"), "budget job id", r);
    },
  },

  // ── Volumes ─────────────────────────────────────────────────────────
  list_volumes: { args: () => ({}) },
  create_volume: {
    confirm: true,
    args: () => ({ name: `${marker}-vol`, size_gb: 10 }),
    then: (r, ctx) => {
      ctx.volumeId = need(pick(r, "volume.volume_id", "volume_id", "volume.id", "id"), "volume id", r);
    },
  },
  get_volume: { args: (ctx) => ({ volume_id: ctx.volumeId }) },
  attach_volume: { confirm: true, args: (ctx) => ({ volume_id: ctx.volumeId, job_id: ctx.jobId }) },
  detach_volume: { confirm: true, args: (ctx) => ({ volume_id: ctx.volumeId }) },
  snapshot_volume: {
    confirm: true,
    args: (ctx) => ({ volume_id: ctx.volumeId, name: `${marker}-snap` }),
    then: (r, ctx) => {
      ctx.snapshotId = need(
        pick(r, "snapshot.snapshot_id", "snapshot_id", "snapshot.id"),
        "snapshot id",
        r,
      );
    },
  },
  list_volume_snapshots: { args: (ctx) => ({ volume_id: ctx.volumeId }) },
  restore_volume_snapshot: {
    confirm: true,
    args: (ctx) => ({ volume_id: ctx.volumeId, snapshot_id: ctx.snapshotId }),
  },
  promote_artifact_to_volume: {
    confirm: true,
    // A job has artifacts once something was uploaded for it. Upload one
    // through the routes the dashboard and the worker SDK use.
    before: async (ctx) => {
      const session = await jsonFetch("/api/artifacts/upload", {
        method: "POST",
        headers: bearer(tenant.browser),
        body: JSON.stringify({ job_id: ctx.jobId, filename: "rehearsal.txt", artifact_type: "job_output" }),
      });
      const sessionId = need(
        pick(session, "upload_session_id", "session_id", "session.session_id"),
        "upload session id",
        session,
      );
      await jsonFetch(`/api/artifacts/sessions/${encodeURIComponent(sessionId)}/content`, {
        method: "PUT",
        headers: { Authorization: `Bearer ${tenant.browser}`, "Content-Type": "text/plain" },
        body: "live rehearsal artifact\n",
      });
      await jsonFetch("/api/artifacts/finalize", {
        method: "POST",
        headers: bearer(tenant.browser),
        body: JSON.stringify({ upload_session_id: sessionId }),
      });
    },
    args: (ctx) => ({ job_id: ctx.jobId, volume_id: ctx.volumeId, idempotency_key: randomUUID() }),
    then: (r, ctx) => {
      ctx.promotionId = need(pick(r, "promotion_id", "promotion.promotion_id", "promotion.id"), "promotion id", r);
    },
  },
  get_promotion_status: {
    args: (ctx) => ({ volume_id: ctx.volumeId, promotion_id: ctx.promotionId }),
  },

  // ── Images ──────────────────────────────────────────────────────────
  create_instance_snapshot: {
    confirm: true,
    before: async () => {
      fixture("registry-up");
    },
    args: (ctx) => ({ job_id: ctx.jobId, name: `${marker}-image`, tag: "v1" }),
    then: (r, ctx) => {
      ctx.imageId = need(pick(r, "image.image_id", "image_id", "image.id", "id"), "image id", r);
    },
  },
  list_user_images: { args: () => ({ scope: "mine" }) },
  create_image_sweep: {
    confirm: true,
    // The worker commits and pushes, then reports the image ready. Do that
    // report, as the host that was asked for the image.
    before: async (ctx) => {
      await jsonFetch(`/user-images/${encodeURIComponent(ctx.imageId)}/complete`, {
        method: "POST",
        headers: agentHeaders(),
        body: JSON.stringify({
          status: "ready",
          size_bytes: 1024,
          image_digest: `registry.rehearsal.test/xcelsior/${marker}@sha256:${"a".repeat(64)}`,
        }),
      });
    },
    args: (ctx) => ({ image_id: ctx.imageId, count: 2, gpu_model: GPU }),
    then: (r, ctx) => {
      ctx.sweepId = need(pick(r, "sweep.sweep_id", "sweep_id", "sweep.id"), "sweep id", r);
    },
  },
  get_image_sweep: { args: (ctx) => ({ sweep_id: ctx.sweepId }) },

  // ── Serverless ──────────────────────────────────────────────────────
  list_serverless_endpoints: { args: () => ({}) },
  should_i_run_pel_job: {
    args: () => ({ model_ref: "test/model", estimated_input_tokens: 1000, estimated_output_tokens: 500 }),
  },
  create_serverless_endpoint: {
    confirm: true,
    args: () => ({
      name: `${marker}-endpoint`,
      model_ref: "test/model",
      gpu_tier: GPU,
      min_workers: 0,
      max_workers: 1,
      idempotency_key: randomUUID(),
    }),
    then: (r, ctx) => {
      ctx.endpointId = need(pick(r, "endpoint_id", "endpoint.endpoint_id", "endpoint.id"), "endpoint id", r);
    },
  },
  get_serverless_endpoint: { args: (ctx) => ({ endpoint_id: ctx.endpointId }) },
  get_serverless_endpoint_health: { args: (ctx) => ({ endpoint_id: ctx.endpointId }) },
  run_serverless_job: {
    confirm: true,
    args: (ctx) => ({ endpoint_id: ctx.endpointId, input: { prompt: "rehearsal" } }),
    then: (r, ctx) => {
      ctx.serverlessJobId = need(pick(r, "job_id", "id", "job.job_id", "job.id"), "serverless job id", r);
    },
  },
  get_serverless_job_status: {
    args: (ctx) => ({ endpoint_id: ctx.endpointId, job_id: ctx.serverlessJobId }),
  },
  list_serverless_endpoint_jobs: { args: (ctx) => ({ endpoint_id: ctx.endpointId }) },
  cancel_serverless_job: {
    confirm: true,
    args: (ctx) => ({ endpoint_id: ctx.endpointId, job_id: ctx.serverlessJobId }),
  },
  delete_serverless_endpoint: { confirm: true, args: (ctx) => ({ endpoint_id: ctx.endpointId }) },

  // ── Plans, pipelines and budget launches ────────────────────────────
  run_pipeline: {
    args: () => ({
      name: `${marker}-pipeline`,
      stages: [
        { name: "prepare", action_type: "create_instance", args: { name: `${marker}-p1`, ...instanceSpec } },
        { name: "train", action_type: "create_instance", args: { name: `${marker}-p2`, ...instanceSpec } },
      ],
    }),
    then: (r, ctx) => {
      ctx.pipelinePlanId = need(pick(r, "plan_id", "plan.plan_id", "pipeline.plan_id"), "pipeline plan id", r);
    },
  },
  get_pipeline_status: { args: (ctx) => ({ plan_id: ctx.pipelinePlanId }) },
  revoke_launch_plan: {
    args: (ctx) => ({ plan_id: ctx.pipelinePlanId, reason: "rehearsal" }),
  },
  run_training_job: {
    confirm: true,
    // The tool waits for running, so place the job while it waits, as a host
    // taking the work would.
    before: async (ctx) => {
      ctx.trainingPlacement = placeWhenQueued(`${marker}-train`);
    },
    then: async (r, ctx) => {
      await ctx.trainingPlacement;
      expect(r.timed_out, JSON.stringify(r).slice(0, 400)).toBe(false);
      ctx.trainJobId = need(pick(r, "job_id", "instance.job_id"), "training job id", r);
    },
    args: () => ({
      name: `${marker}-train`,
      ...instanceSpec,
      git_repo: "https://github.com/aabiro/xcelsior.git",
      wait_timeout_seconds: 30,
    }),
  },

  // ── Provider sign-up and payouts status ─────────────────────────────
  register_provider: {
    confirm: true,
    // The tool never takes an address or tax id: those stay out of model
    // context, and the user enters them in the dashboard. Do that step first,
    // through the route the dashboard uses.
    before: async () => {
      await jsonFetch("/api/providers/seller-profile", {
        method: "PUT",
        headers: bearer(tenant.browser),
        body: JSON.stringify({
          legal_name: "Rehearsal Provider",
          address_line1: "1 Test Street",
          city: "London",
          province: "ON",
          postal_code: "N6A 1A1",
          country: "CA",
          tax_id_kind: "sin",
          tax_id: "046454286",
          accepted_terms: true,
        }),
      });
    },
    args: () => ({ provider_type: "individual", legal_name: "Rehearsal Provider", country: "CA", province: "ON" }),
    then: (r, ctx) => {
      ctx.providerId = need(pick(r, "provider_id", "provider.provider_id", "provider.id"), "provider id", r);
    },
  },
  get_provider_account: { args: (ctx) => ({ provider_id: ctx.providerId }) },
  get_provider_earnings: { args: (ctx) => ({ provider_id: ctx.providerId }) },
  get_paypal_status: { args: (ctx) => ({ provider_id: ctx.providerId }) },
  get_wise_status: { args: (ctx) => ({ provider_id: ctx.providerId }) },
  get_reputation_breakdown: { args: (ctx) => ({ entity_id: ctx.customer }) },
  get_reputation_history: { args: (ctx) => ({ entity_id: ctx.customer }) },

  // ── Money in ────────────────────────────────────────────────────────
  top_up_wallet: {
    confirm: true,
    // Cards are added in the dashboard, never through MCP. Do what the
    // dashboard does, in Stripe test mode: create a SetupIntent and confirm it
    // with Stripe's test card.
    before: async () => saveTestCard(),
    args: () => ({ amount_cad: 10, idempotency_key: randomUUID() }),
  },
  configure_auto_topup: { args: () => ({ enabled: false, amount_cad: 20, threshold_cad: 5 }) },

  // ── Tidy up, last ───────────────────────────────────────────────────
  // A receipt exists once a run has finished, and a retry needs a failed one.
  // The worker reports the run failed, as a host does when the container exits
  // non-zero.
  get_instance_receipt: {
    before: async (ctx) => {
      await reportAttempt(ctx.worker, "failed", {
        failure_code: "container_exited",
        runtime_metadata: { error_message: "rehearsal: exited 1" },
      });
    },
    args: (ctx) => ({ job_id: ctx.jobId }),
  },
  retry_instance: {
    confirm: true,
    // Refreshes the version: reconcile above may have moved it.
    before: async (ctx) => {
      const r = await call("get_instance", { job_id: ctx.jobId });
      ctx.jobVersion = pick(r, "job.version", "version", "instance.version");
    },
    args: (ctx) => ({ job_id: ctx.jobId, expected_version: ctx.jobVersion, idempotency_key: randomUUID() }),
  },
  delete_user_image: { confirm: true, args: (ctx) => ({ image_id: ctx.imageId }) },
  delete_volume: { confirm: true, args: (ctx) => ({ volume_id: ctx.volumeId }) },
  delete_ssh_key: { confirm: true, args: (ctx) => ({ key_id: ctx.sshKeyId }) },
  // The retried instance is queued again: cancel it. The training job is
  // running: terminate it.
  cancel_instance: { confirm: true, args: (ctx) => ({ job_id: ctx.jobId }) },
  terminate_instance: { confirm: true, args: (ctx) => ({ job_id: ctx.trainJobId }) },
};

/** Run last, after the refusals and variants, because they end what those use. */
const TIDY = new Set([
  "cancel_serverless_job",
  "delete_serverless_endpoint",
  "get_instance_receipt",
  "retry_instance",
  "delete_user_image",
  "delete_volume",
  "delete_ssh_key",
  "cancel_instance",
  "terminate_instance",
]);

/** Wait for a launch to appear by name, then bring it to running. */
async function placeWhenQueued(name: string): Promise<void> {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    const listed = await call("list_instances", {});
    const rows = (pick(listed, "instances", "items", "jobs") ?? []) as any[];
    const job = rows.find((row) => row?.name === name);
    if (job?.job_id) {
      await bringToRunning(String(job.job_id));
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`${name} never appeared in list_instances`);
}

/** Save Stripe's test Visa on the tenant, as the dashboard's card form does. */
async function saveTestCard(): Promise<void> {
  const key =
    process.env.XCELSIOR_STRIPE_SANDBOX_SECRET_KEY || process.env.XCELSIOR_STRIPE_SECRET_KEY || "";
  if (!key.startsWith("sk_test_")) {
    throw new Error("top_up_wallet needs a Stripe test-mode secret key (sk_test_) in .env.test");
  }
  const intent = await jsonFetch("/api/billing/setup-intent", {
    method: "POST",
    headers: bearer(tenant.browser),
  });
  const id = need(pick(intent, "setup_intent_id"), "setup intent id", intent);
  const confirmed = await fetch(`https://api.stripe.com/v1/setup_intents/${id}/confirm`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: new URLSearchParams({ payment_method: "pm_card_visa" }),
  });
  const body = await confirmed.json();
  if (!confirmed.ok || body.status !== "succeeded") {
    throw new Error(`test card setup ${confirmed.status}: ${JSON.stringify(body).slice(0, 400)}`);
  }
}

let tenant: Session;
let client: Client;
let transport: StreamableHTTPClientTransport;
/** Tenant A again, through a connector that was granted one permission only. */
let narrow: Client;
/** A second tenant with no money and none of tenant A's resources. */
let other: Session;
let otherClient: Client;
const extraTransports: StreamableHTTPClientTransport[] = [];
const ctx: Ctx = {};
const outcomes: Array<{ tool: string; path: string; outcome: string; detail: string }> = [];
/** The one permission the narrow connector is granted. */
const NARROW_SCOPE = "gpu:read";
/** Input schemas from tools/list, read once. */
let schemas: Record<string, { properties?: Record<string, any>; required?: string[] }> = {};

async function connect(token: string, name: string): Promise<Client> {
  const c = new Client({ name, version: "1.0.0" });
  const t = new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${mcpPort}/mcp`), {
    requestInit: { headers: { Authorization: `Bearer ${token}` } },
  });
  extraTransports.push(t);
  await c.connect(t);
  return c;
}

async function call(name: string, args: Args, via: Client = client): Promise<Record<string, any>> {
  const result = (await via.callTool({ name, arguments: args })) as {
    structuredContent?: Record<string, any>;
    content?: unknown;
    isError?: boolean;
  };
  const body = result.structuredContent ?? {};
  const failed =
    result.isError === true || body.ok === false || (typeof body.error === "string" && body.error !== "");
  if (failed || !result.structuredContent) {
    throw new Error(
      `${name} did not succeed: ${JSON.stringify(result.structuredContent ?? result.content).slice(0, 1500)}`,
    );
  }
  return body;
}

/** Preview, approve the plan if there is one, then confirm. */
async function callConfirmed(
  name: string,
  args: Args,
  via: Client = client,
  approver: Session = tenant,
): Promise<Record<string, any>> {
  const preview = await call(name, { ...args, confirm: false }, via);
  // A preview that returns a plan needs the user's approval before it runs.
  const planId = pick(preview, "plan_id", "plan.plan_id");
  if (planId) await approve(planId, approver.browser);
  return call(name, { ...args, confirm: true, ...(planId ? { plan_id: planId } : {}) }, via);
}

// ── Refusals ──────────────────────────────────────────────────────────

type Attempt = { failed: boolean; text: string; body: Record<string, any> };

/** Call a tool and report what came back, success or not. */
async function attempt(via: Client, name: string, args: Args): Promise<Attempt> {
  try {
    const r = (await via.callTool({ name, arguments: args })) as {
      structuredContent?: Record<string, any>;
      content?: unknown;
      isError?: boolean;
    };
    const body = r.structuredContent ?? {};
    const failed =
      r.isError === true || body.ok === false || (typeof body.error === "string" && body.error !== "");
    return { failed, text: JSON.stringify(r.structuredContent ?? r.content ?? r), body };
  } catch (error) {
    // The SDK throws for protocol errors, input validation among them.
    return { failed: true, text: (error as Error).message, body: {} };
  }
}

/**
 * A refusal is an answer: it names the reason, so an agent can act on it. A
 * 5xx, a stack trace or a JavaScript error is a fault dressed as a refusal.
 */
const FAULT = /Internal Server Error|API error \(5\d\d\)|Traceback|is not a function|Cannot read propert|undefined is not/i;

async function refused(via: Client, name: string, args: Args, reason: RegExp): Promise<Attempt> {
  const got = await attempt(via, name, args);
  if (!got.failed) throw new Error(`was not refused: ${got.text.slice(0, 700)}`);
  if (FAULT.test(got.text)) throw new Error(`refused with a fault, not an answer: ${got.text.slice(0, 700)}`);
  if (!reason.test(got.text)) {
    throw new Error(`refused, but not for the expected reason ${reason}: ${got.text.slice(0, 700)}`);
  }
  return got;
}

/** The arguments the happy path used, as a preview for tools that confirm. */
function previewArgs(tool: string): Args {
  const rehearsal = REHEARSALS[tool];
  return { ...rehearsal.args(ctx), ...(rehearsal.confirm ? { confirm: false } : {}) };
}

/** The required `*_id` arguments of a tool, from its published schema. */
function requiredIds(tool: string): string[] {
  const schema = schemas[tool] ?? {};
  return (schema.required ?? []).filter((key) => key.endsWith("_id"));
}

/**
 * A value of the wrong type for one of the tool's properties, or undefined if
 * it takes no input.
 */
function wrongType(tool: string): Args | undefined {
  const properties = schemas[tool]?.properties ?? {};
  for (const [key, spec] of Object.entries(properties)) {
    const type = Array.isArray(spec?.type) ? spec.type[0] : spec?.type;
    if (type === "string") return { [key]: 12345 };
    if (type === "number" || type === "integer") return { [key]: "not-a-number" };
    if (type === "boolean") return { [key]: "yes" };
    if (type === "array") return { [key]: "not-a-list" };
    if (type === "object") return { [key]: "not-an-object" };
    if (spec?.enum) return { [key]: "not-one-of-the-values" };
  }
  return undefined;
}

/**
 * Reads another tenant may make by design: host evidence is the marketplace's
 * public trust signal. Everything else that takes an id is the owner's alone,
 * a user's reputation breakdown and history included.
 */
const PUBLIC_BY_DESIGN = new Set(["get_host_evidence"]);

const NOT_FOUND = /not.?found|404|no such|does not exist|unknown|not yours|forbidden|403|not authori[sz]ed|no access/i;

function record(tool: string, path: string, run: () => Promise<unknown>): () => Promise<void> {
  return async () => {
    try {
      await run();
      outcomes.push({ tool, path, outcome: "ok", detail: "" });
    } catch (error) {
      outcomes.push({ tool, path, outcome: "FAILED", detail: (error as Error).message.slice(0, 200) });
      throw error;
    }
  };
}

/** Variants: every value of every enum, beyond the one the happy path used. */
type Variant = { tool: string; label: string; via?: () => Client; run: () => Promise<unknown> };
const VARIANTS: Variant[] = [
  ...(["host", "job"] as const).map((entity_type) => ({
    tool: "get_event_history",
    label: `entity_type=${entity_type}`,
    run: () =>
      call("get_event_history", {
        entity_type,
        entity_id: entity_type === "host" ? ctx.hostId : ctx.jobId,
      }),
  })),
  ...(["host", "user"] as const).map((entity_type) => ({
    tool: "get_reputation_leaderboard",
    label: `entity_type=${entity_type}`,
    run: () => call("get_reputation_leaderboard", { entity_type }),
  })),
  ...(["on_demand", "spot"] as const).map((pricing_mode) => ({
    tool: "list_tiers",
    label: `pricing_mode=${pricing_mode}`,
    run: () => call("list_tiers", { pricing_mode }),
  })),
  ...(["mine", "public"] as const).map((scope) => ({
    tool: "list_user_images",
    label: `scope=${scope}`,
    run: () => call("list_user_images", { scope }),
  })),
  ...(["price", "region", "reputation", "vram"] as const).map((sort_by) => ({
    tool: "search_marketplace",
    label: `sort_by=${sort_by}`,
    run: () => call("search_marketplace", { gpu_model: GPU, sort_by }),
  })),
  ...(["ssh", "terminal"] as const).map((method) => ({
    tool: "open_instance_access",
    label: `method=${method}`,
    run: () => callConfirmed("open_instance_access", { job_id: ctx.jobId, method }),
  })),
  // Launch variants are priced, not launched: the preview is where the
  // pricing mode and tier change the answer, and one launch per variant would
  // only repeat the happy path's.
  ...(["on_demand", "spot"] as const).flatMap((pricing_mode) =>
    (["standard", "premium", "dedicated"] as const).map((tier) => ({
      tool: "create_instance",
      label: `preview pricing_mode=${pricing_mode} tier=${tier}`,
      run: async () => {
        const r = await call("create_instance", {
          name: `${marker}-v-${pricing_mode}-${tier}`,
          ...instanceSpec,
          pricing_mode,
          tier,
          idempotency_key: randomUUID(),
          confirm: false,
        });
        need(pick(r, "plan_id", "plan.plan_id", "estimate_cad", "preview"), "a priced preview", r);
      },
    })),
  ),
  ...(["on_demand", "spot"] as const).map((pricing_mode) => ({
    tool: "run_training_job",
    label: `preview pricing_mode=${pricing_mode}`,
    run: () =>
      call("run_training_job", {
        name: `${marker}-v-train-${pricing_mode}`,
        ...instanceSpec,
        pricing_mode,
        git_repo: "https://github.com/aabiro/xcelsior.git",
        confirm: false,
      }),
  })),
  ...(["on_demand", "spot"] as const).map((pricing_mode) => ({
    tool: "schedule_under_budget",
    label: `preview pricing_mode=${pricing_mode}`,
    run: () =>
      call("schedule_under_budget", {
        name: `${marker}-v-budget-${pricing_mode}`,
        max_hourly_cad: 50,
        ...instanceSpec,
        pricing_mode,
        confirm: false,
      }),
  })),
  {
    tool: "attach_volume",
    label: "mode=ro",
    run: async () => {
      await callConfirmed("attach_volume", { volume_id: ctx.volumeId, job_id: ctx.jobId, mode: "ro" });
      await callConfirmed("detach_volume", { volume_id: ctx.volumeId });
    },
  },
  {
    tool: "register_provider",
    label: "provider_type=company (second tenant)",
    run: async () => {
      await jsonFetch("/api/providers/seller-profile", {
        method: "PUT",
        headers: bearer(other.browser),
        body: JSON.stringify({
          legal_name: "Rehearsal Hosting Inc.",
          address_line1: "2 Test Street",
          city: "London",
          province: "ON",
          postal_code: "N6A 1A1",
          country: "CA",
          tax_id_kind: "bn",
          tax_id: "123456782",
          accepted_terms: true,
        }),
      });
      await callConfirmed(
        "register_provider",
        {
          provider_type: "company",
          legal_name: "Rehearsal Hosting Inc.",
          corporation_name: "Rehearsal Hosting Inc.",
          country: "CA",
          province: "ON",
        },
        otherClient,
      );
    },
  },
];

/** What an unfunded tenant is told when a launch needs money it does not have. */
const UNFUNDED: Array<{ tool: string; args: () => Args; reason: RegExp }> = [
  {
    tool: "create_instance",
    args: () => ({ name: `${marker}-broke`, ...instanceSpec, idempotency_key: randomUUID() }),
    reason: /fund|balance|insufficient|top.?up|wallet|402|payment/i,
  },
  {
    tool: "run_training_job",
    args: () => ({
      name: `${marker}-broke-train`,
      ...instanceSpec,
      git_repo: "https://github.com/aabiro/xcelsior.git",
      wait_timeout_seconds: 30,
    }),
    reason: /fund|balance|insufficient|top.?up|wallet|402|payment/i,
  },
  {
    tool: "top_up_wallet",
    args: () => ({ amount_cad: 10, idempotency_key: randomUUID() }),
    reason: /card|payment method|no saved|add a/i,
  },
];

beforeAll(async () => {
  if (!enabled) return;
  tenant = await createSession(marker, "tenant");
  ctx.customer = tenant.customer;
  // What a new user fills in during dashboard onboarding. Billing needs it
  // before a card can be saved.
  await jsonFetch("/api/auth/me/profile", {
    method: "PUT",
    headers: bearer(tenant.browser),
    body: JSON.stringify({
      name: "Rehearsal Tenant",
      account_type: "individual",
      country: "CA",
      province: "ON",
      address_line1: "1 Test Street",
      city: "London",
      postal_code: "N6A 1A1",
    }),
  });
  // Test money through the test-only deposit route; no processor is called.
  await jsonFetch(`/api/billing/wallet/${encodeURIComponent(tenant.customer)}/deposit`, {
    method: "POST",
    headers: bearer(tenant.browser),
    body: JSON.stringify({ amount_cad: 1000, description: "MCP live rehearsal" }),
  });
  await startMcp(mcpPort);
  const token = await connectorToken(tenant, `http://127.0.0.1:${mcpPort}/mcp`);
  client = new Client({ name: "live-rehearsal", version: "1.0.0" });
  transport = new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${mcpPort}/mcp`), {
    requestInit: { headers: { Authorization: `Bearer ${token}` } },
  });
  await client.connect(transport);
  schemas = Object.fromEntries(
    (await client.listTools()).tools.map((t) => [t.name, t.inputSchema as (typeof schemas)[string]]),
  );
  narrow = await connect(
    await connectorToken(tenant, `http://127.0.0.1:${mcpPort}/mcp`, [NARROW_SCOPE]),
    "live-rehearsal-narrow",
  );
  // The second tenant fills in the same profile but adds no money and no card.
  other = await createSession(marker, "other");
  await jsonFetch("/api/auth/me/profile", {
    method: "PUT",
    headers: bearer(other.browser),
    body: JSON.stringify({
      name: "Rehearsal Other",
      account_type: "individual",
      country: "CA",
      province: "ON",
      address_line1: "2 Test Street",
      city: "London",
      postal_code: "N6A 1A1",
    }),
  });
  otherClient = await connect(
    await connectorToken(other, `http://127.0.0.1:${mcpPort}/mcp`),
    "live-rehearsal-other",
  );
}, 90_000);

afterAll(async () => {
  if (!enabled) return;
  for (const t of [transport, ...extraTransports]) await t?.close().catch(() => undefined);
  await stopMcp();
  try {
    fixture("cleanup", marker);
  } catch (error) {
    console.warn(`[rehearsal] cleanup failed: ${(error as Error).message}`);
  }
  if (outcomes.length) {
    console.table(outcomes);
    // One line per tool: which paths it was put through.
    const byTool: Record<string, Set<string>> = {};
    for (const o of outcomes) (byTool[o.tool] ??= new Set()).add(o.path.split(" ")[0]);
    console.table(
      Object.entries(byTool)
        .sort()
        .map(([tool, paths]) => ({ tool, paths: [...paths].sort().join(", ") })),
    );
  }
});

const TOOLS = toolsInProfile("customer");
const happy = Object.entries(REHEARSALS).filter(([tool]) => !TIDY.has(tool));
const tidy = Object.entries(REHEARSALS).filter(([tool]) => TIDY.has(tool));

function rehearse([tool, rehearsal]: [string, Rehearsal]) {
  it(
    tool,
    record(tool, "happy", async () => {
      await rehearsal.before?.(ctx);
      const args = rehearsal.args(ctx);
      const result = rehearsal.confirm ? await callConfirmed(tool, args) : await call(tool, args);
      await rehearsal.then?.(result, ctx);
    }),
    120_000,
  );
}

describe.runIf(enabled)("live rehearsal of every published tool", () => {
  it("has a rehearsal for every tool a connector is shown, and for nothing else", async () => {
    const listed = (await client.listTools()).tools.map((t) => t.name).sort();
    expect(listed).toEqual(TOOLS);
    expect(Object.keys(REHEARSALS).sort()).toEqual(listed);
  });

  describe("happy path: each tool succeeds with valid input", () => {
    happy.forEach(rehearse);
  });

  // While tenant A's instance, volume, image, endpoint and plans still exist,
  // so a refusal of them means the check worked, not that they were gone.
  describe("sad path: input that breaks the schema is refused before it reaches the API", () => {
    for (const tool of TOOLS) {
      it(
        tool,
        record(tool, "invalid-input", async () => {
          const required = schemas[tool]?.required ?? [];
          const bad = required.length ? {} : wrongType(tool);
          if (!bad) {
            // Nothing to get wrong: the tool takes no input. Its refusal path
            // is the narrow-permission one below.
            expect(Object.keys(schemas[tool]?.properties ?? {})).toEqual([]);
            return;
          }
          await refused(client, tool, bad, /-32602|invalid|expected|required/i);
        }),
      );
    }
  });

  describe("sad path: a connection without the permission is refused, and told which", () => {
    for (const tool of TOOLS) {
      it(
        tool,
        record(tool, "permission", async () => {
          const allowed = satisfiesScope([NARROW_SCOPE], TOOL_SCOPES[tool]);
          if (allowed) {
            // The narrow connection may use it, so it must not be refused for
            // permission.
            const got = await attempt(narrow, tool, previewArgs(tool));
            expect(got.text, got.text.slice(0, 500)).not.toMatch(/insufficient_scope/);
            if (tool === "get_permissions") expect(got.failed, got.text.slice(0, 500)).toBe(false);
            return;
          }
          const got = await refused(narrow, tool, previewArgs(tool), /insufficient_scope/);
          const missing = (got.body.missing ?? []) as string[];
          expect(missing.length, got.text.slice(0, 500)).toBeGreaterThan(0);
          expect(missing).not.toContain(NARROW_SCOPE);
        }),
        60_000,
      );
    }
  });

  describe("sad path: an id that does not exist is refused, not a fault", () => {
    for (const tool of TOOLS) {
      it(
        tool,
        record(tool, "unknown-id", async () => {
          const ids = requiredIds(tool);
          if (!ids.length) return;
          const args = previewArgs(tool);
          for (const key of ids) args[key] = `missing-${marker}`;
          await refused(client, tool, args, new RegExp(`${NOT_FOUND.source}|-32602|invalid`, "i"));
        }),
        60_000,
      );
    }
  });

  describe("sad path: another tenant cannot reach tenant A's resources", () => {
    for (const tool of TOOLS) {
      it(
        tool,
        record(tool, "other-tenant", async () => {
          if (!requiredIds(tool).length) return;
          const args = previewArgs(tool);
          if (PUBLIC_BY_DESIGN.has(tool)) {
            const got = await attempt(otherClient, tool, args);
            expect(got.failed, got.text.slice(0, 500)).toBe(false);
            return;
          }
          const got = await refused(otherClient, tool, args, NOT_FOUND);
          expect(got.text).not.toContain(tenant.customer);
          if (REHEARSALS[tool].confirm) {
            // The preview refusing is not enough: confirm goes straight to the
            // API, which must refuse too. A plan the other tenant never had
            // is the only other acceptable reason.
            const run = await refused(
              otherClient,
              tool,
              { ...args, confirm: true },
              new RegExp(`${NOT_FOUND.source}|plan|approv`, "i"),
            );
            expect(run.text).not.toContain(tenant.customer);
          }
        }),
        60_000,
      );
    }
  });

  describe("sad path: confirm without an approved plan runs nothing", () => {
    for (const tool of TOOLS.filter((t) => REHEARSALS[t].confirm)) {
      it(
        tool,
        record(tool, "unapproved", async () => {
          // The happy path already ran, so a preview may now be refused for
          // what it would collide with (create_volume: the name is taken).
          // That refusal must still be an answer, not a fault; with no plan
          // there is nothing to confirm without approval.
          const preview = await attempt(client, tool, previewArgs(tool));
          if (FAULT.test(preview.text)) throw new Error(`preview faulted: ${preview.text.slice(0, 700)}`);
          const planId = pick(preview.body, "plan_id", "plan.plan_id");
          if (!planId) return; // Its confirm needs no plan: the preview is the check.
          await refused(
            client,
            tool,
            { ...previewArgs(tool), confirm: true, plan_id: planId },
            /approv|not approved|pending|awaiting/i,
          );
          await call("revoke_launch_plan", { plan_id: planId, reason: "rehearsal: never approved" });
        }),
        60_000,
      );
    }
  });

  describe("sad path: a tenant with no money or card is told so", () => {
    for (const { tool, args, reason } of UNFUNDED) {
      it(
        tool,
        record(tool, "unfunded", async () => {
          const preview = await attempt(otherClient, tool, { ...args(), confirm: false });
          const planId = pick(preview.body, "plan_id", "plan.plan_id");
          if (!preview.failed && planId) await approve(planId, other.browser);
          if (preview.failed) {
            // Refused at the preview: still has to say why.
            if (FAULT.test(preview.text) || !reason.test(preview.text)) {
              throw new Error(`preview refused without the reason ${reason}: ${preview.text.slice(0, 600)}`);
            }
            return;
          }
          await refused(otherClient, tool, { ...args(), confirm: true, ...(planId ? { plan_id: planId } : {}) }, reason);
        }),
        60_000,
      );
    }
  });

  describe("variants: every value of every option", () => {
    for (const variant of VARIANTS) {
      it(`${variant.tool} ${variant.label}`, record(variant.tool, `variant ${variant.label}`, variant.run), 60_000);
    }
    it("covers every enum value in the published schemas", () => {
      const covered = new Set(VARIANTS.map((v) => v.tool));
      const withEnums = TOOLS.filter((tool) =>
        Object.values(schemas[tool]?.properties ?? {}).some((spec: any) => Array.isArray(spec?.enum)),
      );
      expect(withEnums.filter((tool) => !covered.has(tool))).toEqual([]);
    });
  });

  describe("tidy up: the last tools, which end what the others used", () => {
    tidy.forEach(rehearse);
  });
});
