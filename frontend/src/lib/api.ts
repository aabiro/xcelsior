/** API client — all requests use httpOnly cookie auth (credentials: include). */

let _refreshing: Promise<boolean> | null = null;

async function _tryRefresh(): Promise<boolean> {
  try {
    const res = await fetch("/api/auth/refresh", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
    });
    return res.ok;
  } catch {
    return false;
  }
}

function _redirectToLogin() {
  if (typeof window !== "undefined" && !window.location.pathname.startsWith("/login")) {
    // Only redirect to login from protected routes (dashboard)
    // Marketing pages should silently handle unauthenticated state
    if (window.location.pathname.startsWith("/dashboard")) {
      window.location.href = `/login?redirect=${encodeURIComponent(window.location.pathname)}`;
    }
  }
}

async function apiFetch<T = unknown>(
  path: string,
  opts: RequestInit = {},
): Promise<T> {
  const { headers: extraHeaders, ...rest } = opts;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(extraHeaders as Record<string, string>),
  };
  const res = await fetch(path, { credentials: "include", headers, ...rest });
  if (res.status === 401) {
    // Auth endpoints return 401 for invalid credentials — don't intercept
    if (path.startsWith("/api/auth/")) {
      const body = await res.json().catch(() => ({}));
      throw new ApiError(
        401,
        body?.detail || body?.error?.message || body?.message || "Authentication failed",
        body,
      );
    }
    // Try refreshing the session once
    if (!_refreshing) _refreshing = _tryRefresh();
    const ok = await _refreshing;
    _refreshing = null;
    if (ok) {
      // Retry the original request
      const retry = await fetch(path, { credentials: "include", headers, ...rest });
      if (retry.ok) return retry.json();
      if (retry.status === 401) {
        _redirectToLogin();
        throw new ApiError(401, "Session expired");
      }
      const body = await retry.json().catch(() => ({}));
      throw new ApiError(
        retry.status,
        body?.detail || body?.error?.message || body?.message || retry.statusText,
        body,
      );
    }
    _redirectToLogin();
    throw new ApiError(401, "Session expired");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(
      res.status,
      body?.detail || body?.error?.message || body?.message || res.statusText,
      body,
    );
  }
  return res.json();
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public body?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ── Launch Error Classification ──────────────────────────────────────

export interface LaunchErrorInfo {
  message: string;
  action?: { label: string; href: string };
}

/**
 * Classify an instance-launch error into a user-friendly message and
 * optional call-to-action (like RunPod / Vast "add funds" prompts).
 */
export function classifyLaunchError(err: unknown): LaunchErrorInfo {
  if (err instanceof ApiError) {
    const detail =
      typeof err.body === "object" && err.body
        ? ((err.body as Record<string, string>).detail ?? err.message)
        : err.message;

    if (err.status === 402) {
      if (/suspend/i.test(detail)) {
        return {
          message: "Your wallet has been suspended.",
          action: { label: "Manage Wallet", href: "/dashboard/billing" },
        };
      }
      return {
        message: "Insufficient balance — add funds to launch instances.",
        action: { label: "Add Funds", href: "/dashboard/billing?topup=true" },
      };
    }
    if (err.status === 503) {
      return { message: "No GPU hosts available right now. Try again shortly." };
    }
    if (err.status === 422) {
      return { message: "Invalid configuration — check your instance settings." };
    }
    return { message: detail || `Request failed (${err.status})` };
  }
  return { message: err instanceof Error ? err.message : "Failed to launch instance" };
}

// ── Auth ──────────────────────────────────────────────────────────────
export async function login(email: string, password: string) {
  return apiFetch<{
    ok: boolean;
    access_token?: string;
    token_type?: string;
    expires_in?: number;
    user?: { user_id: string; email: string; role: string; is_admin?: boolean; name?: string; customer_id?: string };
    mfa_required?: boolean;
    challenge_id?: string;
    methods?: string[];
  }>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function register(email: string, password: string, name?: string) {
  return apiFetch<{
    ok: boolean;
    access_token?: string;
    email_verification_required?: boolean;
    message?: string;
    user?: { user_id: string; email: string; role: string; is_admin?: boolean };
  }>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password, name }),
  });
}

export async function oauthInitiate(provider: string) {
  return apiFetch<{ ok: boolean; auth_url: string }>(`/api/auth/oauth/${provider}`, {
    method: "POST",
  });
}

export async function getMe() {
  return apiFetch<{
    ok: boolean;
    user: {
      user_id: string;
      email: string;
      name?: string;
      role: string;
      is_admin?: boolean;
      country?: string;
      province?: string;
      avatar_url?: string;
      customer_id?: string;
      provider_id?: string;
    };
  }>("/api/auth/me");
}

export async function refreshToken() {
  return apiFetch<{ ok: boolean; access_token: string; expires_in: number }>(
    "/api/auth/refresh",
    { method: "POST" },
  );
}

export async function logout() {
  return apiFetch<{ ok: boolean }>("/api/auth/logout", { method: "POST" });
}

// ── Hosts ─────────────────────────────────────────────────────────────
export async function fetchHosts() {
  return apiFetch<{ ok: boolean; hosts: Host[] }>("/hosts?active_only=false");
}

export async function registerHost(data: Record<string, unknown>) {
  return apiFetch("/host", { method: "PUT", body: JSON.stringify(data) });
}

// ── Instances ─────────────────────────────────────────────────────────

/** Normalize backend instance → frontend Instance (image→docker_image, etc.) */
function normalizeInstance(inst: Instance): Instance {
  if (!inst.docker_image && inst.image) inst.docker_image = inst.image;
  return inst;
}

export async function fetchInstances() {
  const res = await apiFetch<{ ok: boolean; instances: Instance[] }>("/instances");
  res.instances?.forEach(normalizeInstance);
  return res;
}

/** Unified instance launch payload — all launch flows use this. */
export interface LaunchInstanceParams {
  name: string;
  image?: string;
  vram_needed_gb?: number;
  num_gpus?: number;
  priority?: number;
  tier?: string;
  host_id?: string;
  gpu_model?: string;
  max_bid?: number;
  nfs_path?: string;
  nfs_server?: string;
  nfs_mount_point?: string;
  interactive?: boolean;
  command?: string;
  ssh_port?: number;
}

/** Single entry-point for launching instances — marketplace, new-instance page, spot, on-demand.
 *  All go through POST /instance. No drift. */
export async function launchInstance(params: LaunchInstanceParams) {
  // Strip undefined values so Pydantic doesn't choke on explicit nulls
  const body: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) body[k] = v;
  }
  const res = await apiFetch<{ ok: boolean; instance: Instance }>(
    "/instance", { method: "POST", body: JSON.stringify(body) },
  );
  if (res.instance) normalizeInstance(res.instance);
  return res;
}

/** @deprecated Use launchInstance instead */
export async function submitInstance(data: Record<string, unknown>) {
  return apiFetch<{ ok: boolean; instance?: Record<string, unknown>; instance_id?: string; id?: string }>(
    "/instance", { method: "POST", body: JSON.stringify(data) },
  );
}

export interface ImageTemplate {
  id: string;
  label: string;
  image: string;
  default_vram_gb: number;
  icon: string;
  category: string;
  description: string;
}

export async function fetchImageTemplates() {
  return apiFetch<{ templates: ImageTemplate[] }>("/api/images/templates");
}

export async function cancelInstance(instanceId: string) {
  return apiFetch(`/instances/${encodeURIComponent(instanceId)}/cancel`, { method: "POST" });
}

export async function stopInstance(instanceId: string) {
  return apiFetch(`/instances/${encodeURIComponent(instanceId)}/stop`, { method: "POST" });
}

export async function startInstance(instanceId: string) {
  return apiFetch(`/instances/${encodeURIComponent(instanceId)}/start`, { method: "POST" });
}

export async function restartInstance(instanceId: string) {
  return apiFetch(`/instances/${encodeURIComponent(instanceId)}/restart`, { method: "POST" });
}

export async function terminateInstance(instanceId: string) {
  return apiFetch(`/instances/${encodeURIComponent(instanceId)}/terminate`, { method: "POST" });
}

// ── Billing ───────────────────────────────────────────────────────────
export async function fetchBilling() {
  return apiFetch<{ ok: boolean; records: BillingRecord[]; total_revenue_cad: number }>(
    "/billing",
  );
}

export async function fetchWallet(customerId: string) {
  return apiFetch<{ ok: boolean; wallet: Wallet }>(
    `/api/billing/wallet/${encodeURIComponent(customerId)}`,
  );
}

export async function depositWallet(customerId: string, amount: number, description?: string) {
  return apiFetch<{ ok: boolean; tx_id: string; balance_cad: number }>(
    `/api/billing/wallet/${encodeURIComponent(customerId)}/deposit`,
    { method: "POST", body: JSON.stringify({ amount_cad: amount, description: description || "Credit deposit" }) },
  );
}

export async function resetWalletTestingState(customerId: string) {
  return apiFetch<{ ok: boolean; wallet: Wallet; cleared_transactions: number; promo_available: boolean }>(
    `/api/billing/wallet/${encodeURIComponent(customerId)}/reset-testing`,
    { method: "POST" },
  );
}

export async function claimFreeCredits(customerId: string) {
  return apiFetch<{ ok: boolean; amount_cad: number; balance_cad: number; already_claimed: boolean }>(
    `/api/billing/free-credits/${encodeURIComponent(customerId)}`,
    { method: "POST" },
  );
}

export async function checkFreeCreditsStatus(customerId: string) {
  return apiFetch<{ ok: boolean; claimed: boolean }>(
    `/api/billing/free-credits/${encodeURIComponent(customerId)}/status`,
  );
}

export async function createPaymentIntent(customerId: string, amountCad: number, description?: string) {
  return apiFetch<{
    ok: boolean;
    intent: {
      intent_id: string;
      customer_id: string;
      amount_cents: number;
      currency: string;
      status: string;
      stripe_intent_id: string;
      client_secret?: string;
    };
  }>("/api/billing/payment-intent", {
    method: "POST",
    body: JSON.stringify({ customer_id: customerId, amount_cad: amountCad, description }),
  });
}

// ── Bitcoin Deposits ─────────────────────────────────────────────────

export async function checkCryptoEnabled(opts?: RequestInit) {
  return apiFetch<{
    ok: boolean;
    enabled: boolean;
    available?: boolean;
    reason?: string;
    wallet_name?: string;
    rpc_reachable?: boolean;
    wallet_ready?: boolean;
    network?: string;
    blocks?: number;
  }>("/api/billing/crypto/enabled", opts);
}

export async function createCryptoDeposit(customerId: string, amountCad: number) {
  return apiFetch<{
    ok: boolean;
    deposit_id: string;
    btc_address: string;
    amount_btc: number;
    amount_cad: number;
    btc_cad_rate: number;
    expires_at: number;
    qr_data: string;
  }>("/api/billing/crypto/deposit", {
    method: "POST",
    body: JSON.stringify({ customer_id: customerId, amount_cad: amountCad }),
  });
}

export async function checkCryptoDeposit(depositId: string) {
  return apiFetch<{
    ok: boolean;
    deposit_id: string;
    status: string;
    confirmations: number;
    amount_cad: number;
    amount_btc: number;
    balance_after_cad?: number;
  }>(`/api/billing/crypto/deposit/${encodeURIComponent(depositId)}`);
}

export async function getCryptoRate() {
  return apiFetch<{ ok: boolean; btc_cad: number; currency: string }>("/api/billing/crypto/rate");
}

export async function refreshCryptoDeposit(depositId: string) {
  return apiFetch<{
    ok: boolean;
    deposit_id: string;
    amount_btc: number;
    btc_cad_rate: number;
    expires_at: number;
    qr_data?: string;
  }>(`/api/billing/crypto/refresh/${encodeURIComponent(depositId)}`, {
    method: "POST",
  });
}

// ── Lightning Network Deposits ────────────────────────────────────────
export async function checkLightningEnabled(opts?: RequestInit) {
  return apiFetch<{
    ok: boolean;
    enabled: boolean;
    available?: boolean;
    reason?: string;
    node_alias?: string;
    node_id?: string;
    num_active_channels?: number;
    blockheight?: number;
    network?: string;
  }>("/api/billing/lightning/enabled", opts);
}

export async function createLightningDeposit(customerId: string, amountCad: number) {
  return apiFetch<{
    ok: boolean;
    deposit_id: string;
    bolt11: string;
    payment_hash: string;
    amount_sats: number;
    amount_btc: number;
    amount_cad: number;
    btc_cad_rate: number;
    expires_at: number;
  }>("/api/billing/lightning/deposit", {
    method: "POST",
    body: JSON.stringify({ customer_id: customerId, amount_cad: amountCad }),
  });
}

export async function checkLightningDeposit(depositId: string) {
  return apiFetch<{
    ok: boolean;
    deposit_id: string;
    status: string;
    amount_cad: number;
    amount_sats: number;
    amount_btc: number;
    payment_preimage?: string;
    paid_at?: number;
    credited_at?: number;
  }>(`/api/billing/lightning/deposit/${encodeURIComponent(depositId)}`);
}

export async function getLightningRate() {
  return apiFetch<{ ok: boolean; btc_cad: number; currency: string }>("/api/billing/lightning/rate");
}

export async function fetchWalletHistory(customerId: string, limit = 50) {
  return apiFetch<{ ok: boolean; customer_id: string; transactions: WalletTransaction[] }>(
    `/api/billing/wallet/${encodeURIComponent(customerId)}/history?limit=${limit}`,
  );
}

export async function fetchUsageSummary(customerId: string) {
  return apiFetch<{
    ok: boolean; customer_id: string;
    job_count: number; total_gpu_hours: number; total_cost_cad: number;
    canadian_compute_cad: number; non_canadian_compute_cad: number;
    hosts_used: number; currency: string;
  }>(`/api/billing/usage/${encodeURIComponent(customerId)}`);
}

export async function fetchInvoices(customerId: string, limit = 12) {
  return apiFetch<{ ok: boolean; invoices: Invoice[]; count: number }>(
    `/api/billing/invoices/${encodeURIComponent(customerId)}?limit=${limit}`,
  );
}

export async function downloadInvoice(
  customerId: string,
  format: "csv" | "txt",
  periodStart: number,
  periodEnd: number,
  taxRate = 0.13,
  customerName = "",
) {
  const qs = new URLSearchParams({
    format,
    period_start: String(periodStart),
    period_end: String(periodEnd),
    tax_rate: String(taxRate),
    customer_name: customerName,
  }).toString();
  const res = await fetch(
    `/api/billing/invoice/${encodeURIComponent(customerId)}/download?${qs}`,
    { credentials: "include" },
  );
  if (!res.ok) throw new ApiError(res.status, res.statusText);
  return res.blob();
}

export async function exportCaf(customerId: string, periodStart: number, periodEnd: number, format: "json" | "csv" = "json") {
  if (format === "csv") {
    const qs = new URLSearchParams({
      period_start: String(periodStart),
      period_end: String(periodEnd),
      format: "csv",
    }).toString();
    const res = await fetch(
      `/api/billing/export/caf/${encodeURIComponent(customerId)}?${qs}`,
      { credentials: "include" },
    );
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    return res.blob();
  }
  return apiFetch<{
    ok: boolean;
    summary: {
      total_jobs: number; total_cost_cad: number;
      canadian_eligible_reimbursement_cad: number;
      non_canadian_eligible_reimbursement_cad: number;
      total_eligible_reimbursement_cad: number;
      effective_cost_after_fund_cad: number;
    };
  }>(`/api/billing/export/caf/${encodeURIComponent(customerId)}?period_start=${periodStart}&period_end=${periodEnd}`);
}

export async function estimatePrice(data: {
  gpu_model: string; duration_hours: number; spot?: boolean;
  sovereignty?: boolean; is_canadian?: boolean;
}) {
  return apiFetch<{
    ok: boolean; base_cost_cad: number; spot_price_cad?: number;
    reserved_1mo_cad?: number; reserved_1yr_cad?: number;
    with_rebate_cad?: number; rebate_amount_cad?: number;
  }>("/api/pricing/estimate", { method: "POST", body: JSON.stringify(data) });
}

export async function createReservation(data: {
  customer_id: string; gpu_model: string;
  commitment_type: "1_month" | "3_month" | "1_year";
  quantity?: number; province?: string;
}) {
  return apiFetch<{
    ok: boolean; commitment_id: string;
    discounted_rate_cad: number; total_commitment_value_cad: number;
    start_date: string; end_date: string;
  }>("/api/pricing/reserve", { method: "POST", body: JSON.stringify(data) });
}

// ── Provider / Earnings ───────────────────────────────────────────────
export async function fetchProviderEarnings(providerId: string) {
  return apiFetch<{
    ok: boolean;
    earnings: { total_jobs: number; total_earned_cad: number; total_platform_cad: number; total_tax_cad: number };
    recent_payouts: Payout[];
  }>(`/api/providers/${encodeURIComponent(providerId)}/earnings`);
}

export async function registerProvider(data: {
  provider_id: string; email: string; provider_type?: string;
  corporation_name?: string; business_number?: string; province?: string; legal_name?: string;
}) {
  return apiFetch<{
    ok: boolean; provider_id: string; stripe_account_id: string;
    onboarding_url: string; status: string;
  }>("/api/providers/register", { method: "POST", body: JSON.stringify(data) });
}

export async function fetchProvider(providerId: string) {
  return apiFetch<{
    ok: boolean;
    provider: {
      provider_id: string; provider_type: string; status: string;
      corporation_name?: string; email: string; province?: string;
      created_at: string; onboarded_at?: string;
    };
  }>(`/api/providers/${encodeURIComponent(providerId)}`);
}

export async function resumeOnboarding(providerId: string) {
  return apiFetch<{
    ok: boolean; provider_id: string; onboarding_url?: string; status: string; message?: string;
  }>(`/api/providers/${encodeURIComponent(providerId)}/resume-onboarding`, { method: "POST" });
}

export async function abandonOnboarding(providerId: string) {
  return apiFetch<{
    ok: boolean; provider_id: string; status: string;
  }>(`/api/providers/${encodeURIComponent(providerId)}/abandon-onboarding`, { method: "POST" });
}

export async function requestPayout(providerId: string, jobId: string, totalCad: number) {
  return apiFetch<{
    ok: boolean; job_id: string; total_cad: number;
    provider_share_cad: number; platform_share_cad: number;
    gst_hst_cad: number; stripe_transfer_id?: string;
  }>(`/api/providers/${encodeURIComponent(providerId)}/payout?job_id=${encodeURIComponent(jobId)}&total_cad=${totalCad}`, {
    method: "POST",
  });
}

export async function fetchGstThreshold(providerId: string) {
  return apiFetch<{
    ok: boolean; provider_id: string; total_revenue_cad: number;
    threshold_cad: number; must_register: boolean; period_months: number;
  }>(`/api/billing/gst-threshold/${encodeURIComponent(providerId)}`);
}

// ── Marketplace ───────────────────────────────────────────────────────
export async function fetchMarketplace() {
  return apiFetch<{ ok: boolean; listings: MarketplaceListing[] }>("/marketplace");
}

export async function searchMarketplace(params: Record<string, string>) {
  const qs = new URLSearchParams(params).toString();
  return apiFetch<{ ok: boolean; listings: MarketplaceListing[] }>(
    `/marketplace/search?${qs}`,
  );
}

// ── Telemetry ─────────────────────────────────────────────────────────
export async function fetchTelemetry() {
  return apiFetch<{ ok: boolean; telemetry: Record<string, TelemetryData> }>(
    "/api/telemetry/all",
  );
}

// ── Pricing ───────────────────────────────────────────────────────────
export async function fetchPricingReference() {
  return apiFetch<{ ok: boolean; reference: PricingReference[] }>("/api/pricing/reference");
}

export async function fetchReservedPlans() {
  return apiFetch<{ ok: boolean; plans: ReservedPlan[] }>("/api/pricing/reserved-plans");
}

// ── Spot ──────────────────────────────────────────────────────────────
export async function fetchSpotPrices() {
  return apiFetch<{ ok: boolean; spot_prices: Record<string, number>; prices?: Record<string, number> }>("/spot-prices");
}

/** @deprecated Use launchInstance with max_bid instead */
export async function submitSpotInstance(data: {
  name: string; vram_needed_gb: number; max_bid: number;
  priority?: number; tier?: string; image?: string;
}) {
  return apiFetch<{ ok: boolean; instance: { job_id: string; name: string } }>(
    "/spot/instance", { method: "POST", body: JSON.stringify(data) },
  );
}

// ── Reputation ────────────────────────────────────────────────────────
export async function fetchLeaderboard() {
  return apiFetch<{ ok: boolean; leaderboard: ReputationEntry[] }>(
    "/api/reputation/leaderboard?limit=20",
  );
}

export async function fetchReputation(entityId: string) {
  return apiFetch<{ ok: boolean; reputation: ReputationEntry }>(
    `/api/reputation/${encodeURIComponent(entityId)}`,
  );
}

export async function fetchReputationBreakdown(entityId: string) {
  return apiFetch<{
    ok: boolean;
    total_score: number;
    breakdown: { jobs_completed: number; uptime_bonus: number; penalties: number; decay: number };
  }>(`/api/reputation/${encodeURIComponent(entityId)}/breakdown`);
}

export async function fetchReputationHistory(entityId: string) {
  return apiFetch<{ ok: boolean; events: { event_type: string; delta: number; timestamp: string; description?: string }[] }>(
    `/api/reputation/${encodeURIComponent(entityId)}/history?limit=50`,
  );
}

export async function fetchTrustTiers() {
  return apiFetch<{ ok: boolean; tiers: Record<string, { min_score: number; requirements: string[] }> }>(
    "/api/trust-tiers",
  );
}

// ── Analytics ─────────────────────────────────────────────────────────
export async function fetchAnalytics(params?: Record<string, string>) {
  const qs = params ? `?${new URLSearchParams(params).toString()}` : "";
  return apiFetch(`/api/analytics/usage${qs}`);
}

export interface EnhancedAnalytics {
  ok: boolean;
  days: number;
  role: "admin" | "provider" | "customer";
  customer_id: string;
  provider_id: string;
  cost_per_hour_trend: { date: string; cost_per_hour: number; gpu_hours: number; spend: number }[];
  cumulative_spend: { date: string; total: number }[];
  duration_histogram: { bucket: string; count: number; total_cost: number }[];
  daily_gpu_hours: { date: string; hours: number }[];
  hourly_heatmap: { dow: number; hour: number; count: number }[];
  top_entities: { entity: string; job_count: number; total_cost: number; gpu_hours: number }[];
  sovereignty: {
    total_jobs: number; canadian_jobs: number; canadian_pct: number;
    canadian_spend: number; international_spend: number;
  };
  gpu_performance: {
    gpu_model: string; jobs: number; avg_util: number; avg_duration_min: number;
    total_cost: number; gpu_hours: number; avg_cost_per_hour: number;
  }[];
  provider_daily?: { date: string; jobs_served: number; total_revenue: number; avg_util: number }[];
  provider_summary?: {
    total_jobs_served: number; total_revenue: number;
    total_gpu_hours: number; avg_util: number;
  };
  wallet_activity: { date: string; tx_type: string; total_amount: number; tx_count: number }[];
  peak_days: { date: string; jobs: number; gpu_hours: number; spend: number; avg_util: number }[];
}

export async function fetchEnhancedAnalytics(days = 30) {
  return apiFetch<EnhancedAnalytics>(`/api/analytics/enhanced?days=${days}`);
}

// ── Instance Detail ───────────────────────────────────────────────────
export async function fetchInstance(instanceId: string) {
  const res = await apiFetch<{ ok: boolean; instance: Instance }>(`/instance/${encodeURIComponent(instanceId)}`);
  if (res.instance) normalizeInstance(res.instance);
  return res;
}

export async function fetchInstanceLogs(instanceId: string, limit = 100) {
  return apiFetch<{ ok: boolean; instance_id: string; logs: InstanceLog[]; total: number }>(
    `/instances/${encodeURIComponent(instanceId)}/logs?limit=${limit}`,
  );
}

export function createInstanceLogStream(instanceId: string): EventSource {
  return new EventSource(`/instances/${encodeURIComponent(instanceId)}/logs/stream`, { withCredentials: true });
}

export async function requeueInstance(instanceId: string) {
  return apiFetch<{ ok: boolean; instance: Instance }>(`/instance/${encodeURIComponent(instanceId)}/requeue`, {
    method: "POST",
  });
}

// ── Host Detail ───────────────────────────────────────────────────────
export async function fetchHost(hostId: string) {
  return apiFetch<{ ok: boolean; host: Host }>(`/host/${encodeURIComponent(hostId)}`);
}

export async function fetchComputeScore(hostId: string) {
  return apiFetch<{ ok: boolean; host_id: string; score: number }>(
    `/compute-score/${encodeURIComponent(hostId)}`,
  );
}

export async function fetchSlaStatus(hostId: string) {
  return apiFetch<{ ok: boolean; uptime_30d_pct: number; monthly_record: Record<string, unknown> }>(
    `/api/sla/${encodeURIComponent(hostId)}`,
  );
}

export async function fetchVerificationStatus(hostId: string) {
  return apiFetch<{ ok: boolean; host_id: string; status: string }>(
    `/api/verify/${encodeURIComponent(hostId)}/status`,
  );
}

// ── Verification Admin ────────────────────────────────────────────────
export async function fetchVerifiedHosts() {
  return apiFetch<{
    ok: boolean;
    count: number;
    hosts: VerifiedHost[];
  }>("/api/verified-hosts");
}

export async function approveHost(hostId: string, notes?: string) {
  const qs = notes ? `?notes=${encodeURIComponent(notes)}` : "";
  return apiFetch<{ ok: boolean; status: string }>(`/api/verify/${encodeURIComponent(hostId)}/approve${qs}`, {
    method: "POST",
  });
}

export async function rejectHost(hostId: string, reason?: string) {
  const qs = reason ? `?reason=${encodeURIComponent(reason)}` : "";
  return apiFetch<{ ok: boolean; status: string }>(`/api/verify/${encodeURIComponent(hostId)}/reject${qs}`, {
    method: "POST",
  });
}

// ── Transparency ──────────────────────────────────────────────────────
export async function fetchTransparencyReport(months = 12) {
  return apiFetch<{
    ok: boolean;
    period_months: number;
    summary: {
      requests_received: number;
      complied: number;
      challenged: number;
      pending: number;
      by_type: Record<string, number>;
      by_jurisdiction: Record<string, number>;
    };
    cloud_act_note: string;
  }>(`/api/transparency/report?months=${months}`);
}

// ── Compliance ────────────────────────────────────────────────────────
export async function fetchProvinces() {
  return apiFetch<{ ok: boolean; provinces: Record<string, { tax_rate: number; description: string }> }>(
    "/api/compliance/provinces",
  );
}

export async function fetchTrustTierRequirements() {
  return apiFetch<{ ok: boolean; tiers: { tier: string; requirements: string[] }[] }>(
    "/api/compliance/trust-tier-requirements",
  );
}

// ── Password Reset ────────────────────────────────────────────────────
export async function requestPasswordReset(email: string) {
  return apiFetch<{ ok: boolean; message: string }>("/api/auth/password-reset", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export async function confirmPasswordReset(token: string, newPassword: string) {
  return apiFetch<{ ok: boolean; message: string }>("/api/auth/password-reset/confirm", {
    method: "POST",
    body: JSON.stringify({ token, new_password: newPassword }),
  });
}

// ── Change Password ───────────────────────────────────────────────────
export async function changePassword(currentPassword: string, newPassword: string) {
  return apiFetch<{ ok: boolean; message: string }>("/api/auth/change-password", {
    method: "POST",
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
}

// ── Account Deletion ──────────────────────────────────────────────────
export async function deleteAccount() {
  return apiFetch<{ ok: boolean }>("/api/auth/me", { method: "DELETE" });
}

// ── MFA / Two-Factor Authentication ───────────────────────────────────

export interface MfaMethod {
  id: number;
  type: string;
  enabled: boolean;
  device_name?: string;
  phone_number?: string;
  created_at: number;
}

export interface MfaStatusResponse {
  ok: boolean;
  mfa_enabled: boolean;
  methods: MfaMethod[];
  backup_codes_remaining: number;
}

export async function fetchMfaMethods() {
  return apiFetch<MfaStatusResponse>("/api/auth/mfa/methods");
}

// TOTP
export async function setupTotp() {
  return apiFetch<{ ok: boolean; secret: string; provisioning_uri: string; method_id: number }>(
    "/api/auth/mfa/totp/setup", { method: "POST" },
  );
}

export async function verifyTotp(code: string, methodId?: number) {
  return apiFetch<{ ok: boolean; message: string; backup_codes?: string[] }>(
    "/api/auth/mfa/totp/verify",
    { method: "POST", body: JSON.stringify({ code, method_id: methodId }) },
  );
}

export async function disableTotp() {
  return apiFetch<{ ok: boolean }>("/api/auth/mfa/totp", { method: "DELETE" });
}

// SMS
export async function setupSms(phoneNumber: string) {
  return apiFetch<{ ok: boolean; message: string }>(
    "/api/auth/mfa/sms/setup",
    { method: "POST", body: JSON.stringify({ phone_number: phoneNumber }) },
  );
}

export async function verifySms(code: string) {
  return apiFetch<{ ok: boolean; message: string; backup_codes?: string[] }>(
    "/api/auth/mfa/sms/verify",
    { method: "POST", body: JSON.stringify({ code }) },
  );
}

export async function disableSms() {
  return apiFetch<{ ok: boolean }>("/api/auth/mfa/sms", { method: "DELETE" });
}

// Passkeys (WebAuthn)
export async function passkeyRegisterOptions(deviceName: string = "Security Key") {
  return apiFetch<{ ok: boolean; options: Record<string, unknown>; state_id: string }>(
    "/api/auth/mfa/passkey/register-options",
    { method: "POST", body: JSON.stringify({ device_name: deviceName }) },
  );
}

export async function passkeyRegisterComplete(stateId: string, credential: Record<string, unknown>) {
  return apiFetch<{ ok: boolean; message: string; method_id: number; device_name: string; backup_codes?: string[] }>(
    "/api/auth/mfa/passkey/register-complete",
    { method: "POST", body: JSON.stringify({ state_id: stateId, credential }) },
  );
}

export async function deletePasskey(methodId: number) {
  return apiFetch<{ ok: boolean; message: string }>(
    "/api/auth/mfa/passkey/delete",
    { method: "POST", body: JSON.stringify({ method_id: methodId }) },
  );
}

export async function passkeyAuthenticateOptions(challengeId: string) {
  return apiFetch<{ ok: boolean; options: Record<string, unknown>; state_id: string }>(
    "/api/auth/mfa/passkey/authenticate-options",
    { method: "POST", body: JSON.stringify({ challenge_id: challengeId }) },
  );
}

export async function passkeyAuthenticateComplete(stateId: string, credential: Record<string, unknown>) {
  return apiFetch<{
    ok: boolean;
    access_token?: string;
    token_type?: string;
    expires_in?: number;
    user?: { user_id: string; email: string; name: string; role: string; is_admin?: boolean; customer_id: string; provider_id?: string };
  }>("/api/auth/mfa/passkey/authenticate-complete", {
    method: "POST",
    body: JSON.stringify({ state_id: stateId, credential }),
  });
}

// MFA Login verification
export async function verifyMfaLogin(challengeId: string, method: string, code: string) {
  return apiFetch<{
    ok: boolean;
    access_token?: string;
    token_type?: string;
    expires_in?: number;
    user?: { user_id: string; email: string; name: string; role: string; is_admin?: boolean; customer_id: string; provider_id?: string };
  }>("/api/auth/mfa/verify", {
    method: "POST",
    body: JSON.stringify({ challenge_id: challengeId, method, code }),
  });
}

export async function sendMfaSms(challengeId: string) {
  return apiFetch<{ ok: boolean; message: string }>(
    "/api/auth/mfa/sms/send",
    { method: "POST", body: JSON.stringify({ challenge_id: challengeId }) },
  );
}

// Backup codes
export async function regenerateBackupCodes() {
  return apiFetch<{ ok: boolean; backup_codes: string[] }>(
    "/api/auth/mfa/backup-codes/regenerate",
    { method: "POST" },
  );
}

export async function disableAllMfa() {
  return apiFetch<{ ok: boolean }>("/api/auth/mfa/all", { method: "DELETE" });
}

// ── API Keys ──────────────────────────────────────────────────────────
export async function generateApiKey(name: string, scope: "full-access" | "read-only" = "full-access") {
  return apiFetch<{ ok: boolean; key: string; name: string; scope: string; preview: string }>(
    "/api/keys/generate", { method: "POST", body: JSON.stringify({ name, scope }) },
  );
}

export async function fetchApiKeys() {
  return apiFetch<{ ok: boolean; keys: ApiKeyInfo[] }>("/api/keys");
}

export async function revokeApiKey(keyPreview: string) {
  return apiFetch<{ ok: boolean }>(`/api/keys/${encodeURIComponent(keyPreview)}`, { method: "DELETE" });
}

// ── SSH Keys ──────────────────────────────────────────────────────────
export async function generateSshKey() {
  return apiFetch<{ ok: boolean; key_path: string; public_key: string }>("/ssh/keygen", { method: "POST" });
}

export async function fetchSshPubKey() {
  return apiFetch<{ public_key: string }>("/api/ssh/pubkey");
}

export interface UserSshKey {
  id: string;
  name: string;
  fingerprint: string;
  public_key: string;
  created_at: number;
}

export async function uploadSshKey(name: string, publicKey: string) {
  return apiFetch<{ ok: boolean; id: string; name: string; fingerprint: string }>("/api/ssh/keys", {
    method: "POST",
    body: JSON.stringify({ name, public_key: publicKey }),
  });
}

export async function listSshKeys() {
  return apiFetch<{ ok: boolean; keys: UserSshKey[] }>("/api/ssh/keys");
}

export async function deleteSshKey(keyId: string) {
  return apiFetch<{ ok: boolean }>(`/api/ssh/keys/${encodeURIComponent(keyId)}`, { method: "DELETE" });
}

// ── Privacy / Consent ─────────────────────────────────────────────────
export async function fetchConsent(entityId: string) {
  return apiFetch<{ ok: boolean; consents: ConsentRecord[] }>(
    `/api/privacy/consent/${encodeURIComponent(entityId)}`,
  );
}

export async function recordConsent(entityId: string, consentType: string, details?: Record<string, unknown>) {
  return apiFetch<{ ok: boolean; consent_id: string }>("/api/privacy/consent", {
    method: "POST",
    body: JSON.stringify({ entity_id: entityId, consent_type: consentType, details }),
  });
}

export async function revokeConsent(entityId: string, consentType: string) {
  return apiFetch<{ ok: boolean }>(
    `/api/privacy/consent/${encodeURIComponent(entityId)}/${encodeURIComponent(consentType)}`,
    { method: "DELETE" },
  );
}

export async function fetchRetentionPolicies() {
  return apiFetch<{ ok: boolean; policies: Record<string, { retention_days: number; description: string }> }>(
    "/api/privacy/retention-policies",
  );
}

// ── Compliance ────────────────────────────────────────────────────────
export async function fetchTaxRates() {
  return apiFetch<{ ok: boolean; rates: Record<string, { rate: number; description: string; gst: number; pst: number; hst: number }> }>("/api/compliance/tax-rates");
}

export async function checkQuebecPia(data: { data_origin_province: string; processing_province: string; data_contains_pi: boolean }) {
  return apiFetch<{ ok: boolean; pia_required: boolean; reason: string }>(
    "/api/compliance/quebec-pia-check", { method: "POST", body: JSON.stringify(data) },
  );
}

export async function fetchSlaTargets() {
  return apiFetch<{ ok: boolean; tiers: Record<string, { uptime_pct: number; credit_pct_10: number; credit_pct_25: number; credit_pct_100: number }> }>(
    "/api/sla/targets",
  );
}

export async function fetchSlaHostsSummary() {
  return apiFetch<{ ok: boolean; hosts: { host_id: string; sla_tier: string; uptime_30d_pct: number; violation_count: number }[] }>(
    "/api/sla/hosts-summary",
  );
}

// ── Teams ─────────────────────────────────────────────────────────────
export interface TeamInfo {
  team_id: string;
  name: string;
  owner_email: string;
  plan: string;
  max_members: number;
  created_at: number;
}

export interface TeamMember {
  email: string;
  role: string;
  joined_at?: number;
}

export async function fetchMyTeams() {
  return apiFetch<{ ok: boolean; teams: TeamInfo[] }>("/api/teams/me");
}

export async function createTeam(data: { name: string; plan?: string }) {
  return apiFetch<{ ok: boolean; team_id: string; name: string; plan: string }>(
    "/api/teams", { method: "POST", body: JSON.stringify(data) },
  );
}

export async function fetchTeam(teamId: string) {
  return apiFetch<{ ok: boolean; team: TeamInfo; members: TeamMember[] }>(
    `/api/teams/${encodeURIComponent(teamId)}`,
  );
}

export async function addTeamMember(teamId: string, data: { email: string; role?: string }) {
  return apiFetch<{ ok: boolean; message: string }>(
    `/api/teams/${encodeURIComponent(teamId)}/members`,
    { method: "POST", body: JSON.stringify(data) },
  );
}

export async function removeTeamMember(teamId: string, email: string) {
  return apiFetch<{ ok: boolean }>(
    `/api/teams/${encodeURIComponent(teamId)}/members/${encodeURIComponent(email)}`,
    { method: "DELETE" },
  );
}

export async function deleteTeam(teamId: string) {
  return apiFetch<{ ok: boolean; message: string }>(
    `/api/teams/${encodeURIComponent(teamId)}`,
    { method: "DELETE" },
  );
}

export async function updateMemberRole(teamId: string, email: string, role: string) {
  return apiFetch<{ ok: boolean; message: string }>(
    `/api/teams/${encodeURIComponent(teamId)}/members/${encodeURIComponent(email)}`,
    { method: "PATCH", body: JSON.stringify({ role }) },
  );
}

// ── Artifacts ─────────────────────────────────────────────────────────
export async function uploadArtifact(data: { job_id: string; filename: string; artifact_type: string; residency_policy?: string }) {
  return apiFetch<{ ok: boolean; upload_url: string; artifact_id: string }>(
    "/api/artifacts/upload", { method: "POST", body: JSON.stringify(data) },
  );
}

export async function downloadArtifact(data: { job_id: string; filename: string; artifact_type: string }) {
  return apiFetch<{ ok: boolean; download_url: string }>(
    "/api/artifacts/download", { method: "POST", body: JSON.stringify(data) },
  );
}

export async function fetchArtifacts(jobId?: string) {
  const path = jobId ? `/api/artifacts/${encodeURIComponent(jobId)}` : "/api/artifacts";
  return apiFetch<{ ok: boolean; job_id?: string; artifacts: ArtifactEntry[] }>(path);
}

// ── Admin ─────────────────────────────────────────────────────────────
export async function fetchAdminStats() {
  return apiFetch<{ ok: boolean; total_users: number; active_hosts: number; running_jobs: number; revenue_mtd: number }>(
    "/api/admin/stats",
  );
}

export async function fetchAdminUsers() {
  return apiFetch<{
    ok: boolean;
    users: {
      email: string; role: string; is_admin?: boolean; is_active: boolean;
      created_at: string; wallet_balance_cad: number; total_jobs: number;
      province: string; country: string; team_id?: string | null;
    }[];
  }>("/api/admin/users");
}

export async function fetchAdminOverview(days = 30) {
  return apiFetch<{
    ok: boolean;
    days: number;
    kpis: {
      total_users: number; active_hosts: number; running_jobs: number;
      total_jobs: number; revenue_mtd: number; revenue_total: number;
      total_gpu_hours: number; gpu_utilization: number;
      job_failure_rate: number; arpu: number;
    };
    trends?: {
      users_pct: number; hosts_pct: number; jobs_pct: number; revenue_pct: number;
    };
    daily_revenue: { date: string; revenue: number }[];
    daily_signups: { date: string; signups: number }[];
    daily_jobs: { date: string; jobs: number }[];
  }>(`/api/admin/overview?days=${days}`);
}

export async function fetchAdminRevenue(days = 90) {
  return apiFetch<{
    ok: boolean; days: number;
    daily: { date: string; revenue: number; jobs: number; gpu_hours: number }[];
    by_gpu: { gpu_model: string; revenue: number; jobs: number }[];
    by_province: { province: string; revenue: number; jobs: number }[];
    top_customers: { email: string; total_spend: number; jobs: number }[];
    top_providers: { provider_id: string; earnings: number; jobs: number }[];
  }>(`/api/admin/revenue?days=${days}`);
}

export async function fetchAdminInfrastructure() {
  return apiFetch<{
    ok: boolean; total_hosts: number;
    by_state: { state: string; count: number }[];
    by_gpu: { gpu_model: string; count: number }[];
    by_province: { province: string; count: number }[];
    verification: { state: string; count: number }[];
    reputation_tiers: { tier: string; count: number }[];
  }>("/api/admin/infrastructure");
}

export async function fetchAdminActivity(days = 7) {
  return apiFetch<{
    ok: boolean; days: number;
    events: {
      event_id: string; event_type: string; entity_type: string;
      entity_id: string; timestamp: string; actor: string; data: Record<string, unknown>;
    }[];
    by_type: { event_type: string; count: number }[];
    daily_jobs: { date: string; submitted: number; completed: number; failed: number }[];
  }>(`/api/admin/activity?days=${days}`);
}

export async function verifyAuditChain() {
  return apiFetch<{ ok: boolean; chain_integrity: { valid: boolean; break_point: string | null } }>(
    "/api/audit/verify-chain",
  );
}

export async function fetchAlertConfig() {
  return apiFetch<Record<string, unknown>>(
    "/api/alerts/config",
  );
}

export async function updateAlertConfig(config: Record<string, unknown>) {
  return apiFetch<{ ok: boolean }>("/api/alerts/config", {
    method: "PUT",
    body: JSON.stringify(config),
  });
}

export async function fetchAdminVerificationQueue() {
  return apiFetch<{
    ok: boolean;
    queue: {
      host_id: string; state: string; overall_score: number;
      last_check_at: string | null; gpu_model: string;
      province: string; cost_per_hour: number;
    }[];
  }>("/api/admin/verification-queue");
}

export async function adminSetUserRole(email: string, role: string) {
  return apiFetch<{ ok: boolean; email: string; role: string }>(
    `/api/admin/users/${encodeURIComponent(email)}/role?role=${encodeURIComponent(role)}`,
    { method: "POST" },
  );
}

export async function adminToggleAdmin(email: string) {
  return apiFetch<{ ok: boolean; email: string; is_admin: number }>(
    `/api/admin/users/${encodeURIComponent(email)}/toggle-admin`,
    { method: "POST" },
  );
}

// ── Admin: Teams ─────────────────────────────────────────────────────
export interface AdminTeam {
  team_id: string;
  name: string;
  owner_email: string;
  plan: string;
  max_members: number;
  created_at: number;
  members: TeamMember[];
}

export async function fetchAdminTeams() {
  return apiFetch<{ ok: boolean; teams: AdminTeam[] }>("/api/admin/teams");
}

export async function adminRemoveTeamMember(teamId: string, email: string) {
  return apiFetch<{ ok: boolean; message: string }>(
    `/api/admin/teams/${encodeURIComponent(teamId)}/members/${encodeURIComponent(email)}`,
    { method: "DELETE" },
  );
}

// ── HPC / Slurm ──────────────────────────────────────────────────────
export async function fetchSlurmProfiles() {
  return apiFetch<{ ok: boolean; profiles: Record<string, { description: string; gpus: string[]; partitions: string[] }> }>(
    "/api/slurm/profiles",
  );
}

export async function submitSlurmInstance(data: {
  name: string; vram_needed_gb: number; priority: string;
  tier?: string; num_gpus?: number; image?: string; profile?: string; dry_run?: boolean;
}) {
  return apiFetch<{ ok: boolean; slurm_job_id: string; instance_id?: string }>(
    "/api/slurm/submit", { method: "POST", body: JSON.stringify(data) },
  );
}

export async function fetchSlurmJobStatus(slurmJobId: string) {
  return apiFetch<{ ok: boolean; slurm_job_id: string; state: string }>(
    `/api/slurm/status/${encodeURIComponent(slurmJobId)}`,
  );
}

export async function cancelSlurmJob(slurmJobId: string) {
  return apiFetch<{ ok: boolean; cancelled: boolean }>(
    `/api/slurm/${encodeURIComponent(slurmJobId)}`, { method: "DELETE" },
  );
}

// ── Events SSE ────────────────────────────────────────────────────────
export function createEventSource(url: string = "/api/stream"): EventSource {
  return new EventSource(url, { withCredentials: true });
}

// ── Notifications ─────────────────────────────────────────────────────
export interface Notification {
  id: string;
  user_email: string;
  type: string;
  title: string;
  body: string;
  data: Record<string, unknown>;
  read: number;
  created_at: number;
}

export async function fetchNotifications(unread = false, limit = 50) {
  const qs = `?unread=${unread}&limit=${limit}`;
  return apiFetch<{ ok: boolean; notifications: Notification[]; unread_count: number }>(
    `/api/notifications${qs}`,
  );
}

export async function fetchUnreadCount() {
  return apiFetch<{ ok: boolean; unread_count: number }>("/api/notifications/unread-count");
}

export async function markNotificationRead(notificationId: string) {
  return apiFetch<{ ok: boolean }>(
    `/api/notifications/${encodeURIComponent(notificationId)}/read`,
    { method: "POST" },
  );
}

export async function markAllNotificationsRead() {
  return apiFetch<{ ok: boolean; marked: number }>("/api/notifications/read-all", {
    method: "POST",
  });
}

export async function deleteNotification(notificationId: string) {
  return apiFetch<{ ok: boolean }>(
    `/api/notifications/${encodeURIComponent(notificationId)}`,
    { method: "DELETE" },
  );
}

// ── Types ─────────────────────────────────────────────────────────────
export interface Host {
  host_id: string;
  id?: string;
  hostname?: string;
  ip: string;
  gpu_model: string;
  vram_gb: number;
  status: string;
  cost_per_hour: number;
  price_per_hour?: number;
  country?: string;
  province?: string;
  region?: string;
  reputation_tier?: string;
  reputation_score?: number;
  verified?: boolean;
  compute_score?: number;
}

export interface Instance {
  job_id: string;
  id?: string;
  name?: string;
  host_id?: string;
  status: string;
  gpu_model: string;
  gpu_type?: string;
  docker_image: string;
  image?: string;  // raw backend field — normalizeInstance maps this to docker_image
  duration_sec?: number;
  elapsed_sec?: number;
  cost_cad?: number;
  tier?: string;
  submitted_at: string;
  created_at?: string;
  // Connection info (enriched by API when running/completed)
  host_ip?: string;
  host_gpu?: string;
  host_vram_gb?: number;
  container_id?: string;
  container_name?: string;
  started_at?: string;
  completed_at?: string;
  // Interactive instance fields
  interactive?: boolean;
  ssh_port?: number;
  command?: string;
}

/** @deprecated Use Instance instead */
export type Job = Instance;

export interface BillingRecord {
  id?: string;
  job_id: string;
  host_id: string;
  amount_cad: number;
  amount?: number;
  gpu_model: string;
  duration_sec: number;
  billed_at: string;
  created_at?: string;
  description?: string;
  type?: string;
}

export interface Wallet {
  customer_id: string;
  balance_cad: number;
  balance?: number;
  currency: string;
}

export interface MarketplaceListing {
  id?: string;
  host_id: string;
  hostname?: string;
  gpu_model: string;
  vram_gb: number;
  price_per_hour_cad: number;
  price_per_hour?: number;
  status: string;
  country?: string;
  province?: string;
  region?: string;
  reputation_tier?: string;
  reputation_score?: number;
}

export interface TelemetryData {
  gpu_utilization_pct: number;
  gpu_util?: number;
  gpu_temp_c: number;
  temperature?: number;
  gpu_memory_used_mb: number;
  mem_used_mb?: number;
  gpu_memory_total_mb: number;
  gpu_power_draw_w: number;
  power_draw_w?: number;
  ecc_errors: number;
  timestamp: string;
}

export interface PricingReference {
  gpu_model: string;
  on_demand_cad: number;
  spot_cad?: number;
  reserved_1mo_cad?: number;
  reserved_3mo_cad?: number;
  reserved_1yr_cad?: number;
}

export interface ReservedPlan {
  plan_id: string;
  name: string;
  duration_months: number;
  discount_pct: number;
  gpu_model: string;
  price_per_hour_cad: number;
}

export interface ReputationEntry {
  entity_id: string;
  user_id?: string;
  score: number;
  tier: string;
  rank: number;
  jobs_completed?: number;
  uptime?: number;
  gpu_model?: string;
  reliability_score?: number;
}

export interface InstanceLog {
  timestamp: number | string;
  level?: string;
  message: string;
  line?: string;
  stream?: string;
}

/** @deprecated Use InstanceLog instead */
export type JobLog = InstanceLog;

export interface VerifiedHost {
  host_id: string;
  gpu_model?: string;
  overall_score: number;
  last_check: string;
  gpu_fingerprint?: string;
  status: string;
  deverify_reason?: string;
}

export interface WalletTransaction {
  tx_id: string;
  type: string;
  amount_cad: number;
  description?: string;
  job_id?: string;
  created_at: string;
  balance_after?: number;
}

export interface Invoice {
  invoice_id: string;
  period_start: string;
  period_end: string;
  total_cad: number;
  subtotal_cad: number;
  tax_cad: number;
  tax_rate: number;
  line_items: number;
  caf_eligible_cad?: number;
  status: string;
}

export interface Payout {
  job_id: string;
  provider_id: string;
  total_cad: number;
  provider_share_cad: number;
  platform_share_cad: number;
  gst_hst_cad: number;
  stripe_transfer_id?: string;
  created_at: string;
}

export interface ApiKeyInfo {
  name: string;
  scope: string;
  preview: string;
  created_at: string;
}

export interface ConsentRecord {
  consent_id: string;
  entity_id: string;
  consent_type: string;
  granted_at: string;
  details?: Record<string, unknown>;
}

export interface ArtifactEntry {
  artifact_id: string;
  job_id: string;
  filename: string;
  artifact_type: string;
  size_bytes?: number;
  residency_policy?: string;
  created_at: string;
}

// ── v2 Types ──────────────────────────────────────────────────────────

export interface GPUOffer {
  offer_id: string;
  provider_id: string;
  host_id: string;
  gpu_model: string;
  gpu_count_total: number;
  gpu_count_available: number;
  vram_gb: number;
  ask_cents_per_hour: number;
  region: string;
  spot_enabled: boolean;
  spot_min_cents: number;
  available: boolean;
}

export interface InferenceEndpoint {
  endpoint_id: string;
  owner_id: string;
  model_id: string;
  model_name?: string;
  gpu_type: string;
  region: string;
  docker_image: string;
  mode: string;
  health_endpoint: string;
  api_format: string;
  status: string;
  min_workers: number;
  max_workers: number;
  vram_required_gb: number;
  worker_job_id: string | null;
  total_requests: number;
  total_tokens_generated: number;
  total_cost_cad: number;
  cost_per_hour_cad?: number;
  avg_latency_ms?: number;
  created_at: number;
  updated_at: number;
}

export interface GpuAvailability {
  gpu_model: string;
  vram_gb: number;
  region: string;
  province: string;
  count_available: number;
  price_per_hour_cad: number;
}

export interface Volume {
  volume_id: string;
  owner_id: string;
  name: string;
  size_gb: number;
  region: string;
  encrypted: boolean;
  status: string;
  created_at: number;
  price_per_gb_month_cad?: number;
  monthly_cost_cad?: number;
}

export interface SpotPricePoint {
  gpu_model: string;
  spot_cents: number;
  recorded_at: number;
}

// ── v2 Marketplace API ────────────────────────────────────────────────

export async function searchMarketplaceV2(params: {
  gpu_model?: string;
  min_vram_gb?: number;
  max_price_cents?: number;
  region?: string;
  canada_only?: boolean;
  sort_by?: string;
  limit?: number;
}) {
  return apiFetch<{ ok: boolean; offers: GPUOffer[]; count: number }>(
    "/api/v2/marketplace/search",
    { method: "POST", body: JSON.stringify(params) },
  );
}

export async function fetchSpotPricesV2() {
  return apiFetch<{ ok: boolean; spot_prices: SpotPricePoint[] }>(
    "/api/v2/marketplace/spot-prices",
  );
}

export async function fetchSpotHistory(gpuModel: string, hours = 24) {
  return apiFetch<{ ok: boolean; gpu_model: string; history: SpotPricePoint[] }>(
    `/api/v2/marketplace/spot-prices/${encodeURIComponent(gpuModel)}/history?hours=${hours}`,
  );
}

export async function fetchMarketplaceStatsV2() {
  return apiFetch<{ ok: boolean; total_offers: number; total_gpus: number; avg_price: number }>(
    "/api/v2/marketplace/stats",
  );
}

export async function createMarketplaceReservation(data: {
  gpu_model: string;
  gpu_count?: number;
  period_months: number;
}) {
  return apiFetch<{ ok: boolean; reservation: Record<string, unknown> }>(
    "/api/v2/marketplace/reservations",
    { method: "POST", body: JSON.stringify(data) },
  );
}

// ── v2 Inference API ──────────────────────────────────────────────────

export async function fetchAvailableGPUs() {
  return apiFetch<{ ok: boolean; gpus: GpuAvailability[] }>(
    "/api/v2/gpu/available",
  );
}

export async function createInferenceEndpoint(data: {
  model_name: string;
  gpu_type?: string;
  region?: string;
  docker_image?: string;
  min_workers?: number;
  max_workers?: number;
  max_batch_size?: number;
  max_concurrent?: number;
  scaledown_window_sec?: number;
  mode?: string;
  health_endpoint?: string;
  api_format?: string;
}) {
  return apiFetch<{ ok: boolean; endpoint: InferenceEndpoint }>(
    "/api/v2/inference/endpoints",
    { method: "POST", body: JSON.stringify(data) },
  );
}

export async function listInferenceEndpoints() {
  return apiFetch<{ ok: boolean; endpoints: InferenceEndpoint[] }>(
    "/api/v2/inference/endpoints",
  );
}

export async function getInferenceEndpointHealth(endpointId: string) {
  return apiFetch<{ ok: boolean; health: Record<string, unknown> }>(
    `/api/v2/inference/endpoints/${encodeURIComponent(endpointId)}/health`,
  );
}

export async function getInferenceEndpointUsage(endpointId: string) {
  return apiFetch<{ ok: boolean; usage: Record<string, unknown> }>(
    `/api/v2/inference/endpoints/${encodeURIComponent(endpointId)}/usage`,
  );
}

export async function deleteInferenceEndpoint(endpointId: string) {
  return apiFetch<{ ok: boolean }>(
    `/api/v2/inference/endpoints/${encodeURIComponent(endpointId)}`,
    { method: "DELETE" },
  );
}

// ── v2 Volumes API ────────────────────────────────────────────────────

export async function createVolume(data: {
  name: string;
  size_gb?: number;
  region?: string;
  encrypted?: boolean;
}) {
  return apiFetch<{ ok: boolean; volume: Volume }>(
    "/api/v2/volumes",
    { method: "POST", body: JSON.stringify(data) },
  );
}

export async function listVolumes() {
  return apiFetch<{ ok: boolean; volumes: Volume[] }>("/api/v2/volumes");
}

export async function deleteVolume(volumeId: string) {
  return apiFetch<{ ok: boolean }>(
    `/api/v2/volumes/${encodeURIComponent(volumeId)}`,
    { method: "DELETE" },
  );
}

export async function attachVolume(volumeId: string, instanceId: string, mountPath = "/workspace") {
  return apiFetch<{ ok: boolean; attachment: Record<string, unknown> }>(
    `/api/v2/volumes/${encodeURIComponent(volumeId)}/attach`,
    { method: "POST", body: JSON.stringify({ instance_id: instanceId, mount_path: mountPath }) },
  );
}

export async function detachVolume(volumeId: string) {
  return apiFetch<{ ok: boolean }>(
    `/api/v2/volumes/${encodeURIComponent(volumeId)}/detach`,
    { method: "POST" },
  );
}

// ── v2 Billing API ────────────────────────────────────────────────────

export async function configureAutoTopup(data: {
  enabled: boolean;
  amount_cad?: number;
  threshold_cad?: number;
  stripe_payment_method_id?: string;
}) {
  return apiFetch<{ ok: boolean }>(
    "/api/v2/billing/auto-topup",
    { method: "POST", body: JSON.stringify(data) },
  );
}

export async function fetchAutoTopup() {
  return apiFetch<{ ok: boolean; auto_topup: { enabled: boolean; amount_cad: number; threshold_cad: number } }>(
    "/api/v2/billing/auto-topup",
  );
}

// ── v2 Cloud Burst API ────────────────────────────────────────────────

export async function fetchBurstStatus() {
  return apiFetch<{ ok: boolean; active_instances: number; total_spending: number; budget_remaining: number }>(
    "/api/v2/burst/status",
  );
}

// ── Email Verification ────────────────────────────────────────────────

export async function verifyEmail(token: string) {
  return apiFetch<{
    ok: boolean;
    access_token?: string;
    user?: { user_id: string; email: string; role: string; is_admin?: boolean; name?: string; customer_id?: string };
  }>("/api/auth/verify-email", {
    method: "POST",
    body: JSON.stringify({ token }),
  });
}

export async function resendVerification(email: string) {
  return apiFetch<{ ok: boolean; message: string }>("/api/auth/resend-verification", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

// ── Session Management ────────────────────────────────────────────────

export interface SessionInfo {
  token_prefix: string;
  is_current: boolean;
  ip_address: string;
  user_agent: string;
  created_at: number;
  last_active: number;
  expires_at: number;
}

export async function fetchSessions() {
  return apiFetch<{ ok: boolean; sessions: SessionInfo[] }>("/api/auth/sessions");
}

export async function revokeSession(tokenPrefix: string) {
  return apiFetch<{ ok: boolean; message: string }>(`/api/auth/sessions/${tokenPrefix}`, {
    method: "DELETE",
  });
}

// ── Admin AI Insights ─────────────────────────────────────────────────

export interface AdminAiMessage {
  role: string;
  content: string;
  tool_name: string | null;
  tokens_in: number;
  tokens_out: number;
  created_at: number;
}

export interface AdminAiConversation {
  conversation_id: string;
  source: string;
  user: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
  total_input_tokens: number;
  total_output_tokens: number;
  messages: AdminAiMessage[];
}

export interface AdminAiSourceStat {
  source: string;
  conversations: number;
  messages: number;
  input_tokens: number;
  output_tokens: number;
}

export interface AdminAiTopUser {
  user_id: string;
  conversations: number;
  total_tokens: number;
}

export interface AdminAiStats {
  ok: boolean;
  total_conversations: number;
  total_messages: number;
  total_input_tokens: number;
  total_output_tokens: number;
  estimated_cost: number;
  by_source: AdminAiSourceStat[];
  daily: Record<string, string | number>[];
  top_users: AdminAiTopUser[];
}

export async function fetchAdminAiStats(days = 30) {
  return apiFetch<AdminAiStats>(`/api/admin/ai-stats?days=${days}`);
}

export async function fetchAdminAiConversations(
  source = "all",
  days = 7,
  search = "",
  page = 1,
  perPage = 30,
) {
  const params = new URLSearchParams({
    source,
    days: String(days),
    page: String(page),
    per_page: String(perPage),
  });
  if (search) params.set("search", search);
  return apiFetch<{
    ok: boolean;
    conversations: AdminAiConversation[];
    total: number;
    page: number;
    per_page: number;
  }>(`/api/admin/ai-conversations?${params}`);
}
