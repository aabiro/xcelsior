/** API client — all requests use httpOnly cookie auth (credentials: include). */

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
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body?.error?.message || res.statusText, body);
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

// ── Auth ──────────────────────────────────────────────────────────────
export async function login(email: string, password: string) {
  return apiFetch<{
    ok: boolean;
    access_token: string;
    token_type: string;
    expires_in: number;
    user: { user_id: string; email: string; role: string; name?: string; customer_id?: string };
  }>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function register(email: string, password: string, name?: string) {
  return apiFetch<{
    ok: boolean;
    access_token: string;
    user: { user_id: string; email: string; role: string };
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
      country?: string;
      province?: string;
      avatar_url?: string;
      customer_id?: string;
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
export async function fetchInstances() {
  return apiFetch<{ ok: boolean; instances: Instance[] }>("/instances");
}

export async function submitInstance(data: Record<string, unknown>) {
  return apiFetch<{ ok: boolean; instance_id?: string; id?: string }>(
    "/instance", { method: "POST", body: JSON.stringify(data) },
  );
}

export async function cancelInstance(instanceId: string) {
  return apiFetch(`/instances/${encodeURIComponent(instanceId)}/cancel`, { method: "POST" });
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

export async function checkCryptoEnabled() {
  return apiFetch<{ ok: boolean; enabled: boolean }>("/api/billing/crypto/enabled");
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

export async function submitSpotInstance(data: {
  name: string; vram_needed_gb: number; max_bid: number;
  priority?: number; tier?: string;
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

// ── Instance Detail ───────────────────────────────────────────────────
export async function fetchInstance(instanceId: string) {
  return apiFetch<{ ok: boolean; instance: Instance }>(`/instance/${encodeURIComponent(instanceId)}`);
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
  return apiFetch<{ ok: boolean; users: { email: string; role: string; is_active: boolean; created_at: string }[] }>(
    "/api/admin/users",
  );
}

export async function verifyAuditChain() {
  return apiFetch<{ ok: boolean; chain_integrity: { valid: boolean; break_point: string | null } }>(
    "/api/audit/verify-chain",
  );
}

export async function fetchAlertConfig() {
  return apiFetch<{ ok: boolean; email_enabled: boolean; smtp_host: string }>(
    "/api/alerts/config",
  );
}

export async function updateAlertConfig(config: Record<string, unknown>) {
  return apiFetch<{ ok: boolean }>("/api/alerts/config", {
    method: "PUT",
    body: JSON.stringify(config),
  });
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
  duration_sec?: number;
  elapsed_sec?: number;
  cost_cad?: number;
  tier?: string;
  submitted_at: string;
  created_at?: string;
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
}

export interface InstanceLog {
  timestamp: string;
  level?: string;
  message: string;
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
