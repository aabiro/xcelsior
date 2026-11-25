"use client";

/**
 * A host's evidence, as the API's one read model returns it
 * (`hosts/host_evidence.py`): verification freshness, the latest xcelsior-bench
 * run, reputation and the 30-day job success rate. The listing card shows the
 * badge; the host page shows it in full. MCP's `search_marketplace` and
 * `get_host_evidence` return the same fields, so a renter and an agent read
 * the same facts.
 */
import { Activity, Gauge, ShieldAlert, ShieldCheck, ShieldQuestion, Star } from "lucide-react";
import { Card } from "@/components/ui/card";
import type { HostEvidence, HostEvidenceBadge } from "@/lib/api";
import { formatEpochDate } from "@/lib/format-units";
import { useLocale } from "@/lib/locale";

const FRESHNESS: Record<string, { key: string; tone: string; Icon: typeof ShieldCheck }> = {
  verified: { key: "dash.evidence.verified", tone: "text-emerald bg-emerald/10", Icon: ShieldCheck },
  overdue: { key: "dash.evidence.overdue", tone: "text-accent-gold bg-accent-gold/10", Icon: ShieldAlert },
  deverified: { key: "dash.evidence.deverified", tone: "text-accent-red bg-accent-red/10", Icon: ShieldAlert },
  pending: { key: "dash.evidence.pending", tone: "text-text-muted bg-surface-hover", Icon: ShieldQuestion },
  unverified: { key: "dash.evidence.unverified", tone: "text-text-muted bg-surface-hover", Icon: ShieldQuestion },
};

const VERDICT: Record<string, { key: string; tone: string }> = {
  pass: { key: "dash.evidence.verdict_pass", tone: "text-emerald" },
  fail: { key: "dash.evidence.verdict_fail", tone: "text-accent-red" },
  unknown: { key: "dash.evidence.verdict_unknown", tone: "text-accent-gold" },
};

const FAULTS: Record<string, string> = {
  thermal_throttling: "dash.evidence.fault_thermal",
  power_throttling: "dash.evidence.fault_power",
  allocatable_vram_below_listing: "dash.evidence.fault_vram",
};

function freshness(state: string | undefined) {
  return FRESHNESS[state ?? "unverified"] ?? FRESHNESS.unverified;
}

function num(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function pct(share: number | null | undefined, digits = 0): string {
  return share === null || share === undefined ? "—" : `${(share * 100).toFixed(digits)}%`;
}

/** `new_user` → "New User": the tier as the reputation page names it. */
function tierLabel(tier: string): string {
  return tier.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function VerificationChip({ state }: { state: string | undefined }) {
  const { t } = useLocale();
  const f = freshness(state);
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide ${f.tone}`}
      data-testid="verification-chip"
    >
      <f.Icon className="h-3 w-3" />
      {t(f.key)}
    </span>
  );
}

/** The short form on a listing card: one line of benchmark, one of standing. */
export function EvidenceSummary({ badge }: { badge: HostEvidenceBadge | undefined }) {
  const { t, locale } = useLocale();
  if (!badge) return null;
  const bench = badge.benchmark;
  const verdict = bench ? VERDICT[bench.verdict] : undefined;
  return (
    <div className="space-y-1 text-[11px] text-text-muted" data-testid="evidence-summary">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="inline-flex items-center gap-1">
          <Gauge className="h-3 w-3" />
          {bench
            ? t("dash.evidence.benchmarked", { date: formatEpochDate(bench.ran_at, locale) })
            : t("dash.evidence.no_benchmark")}
        </span>
        {verdict && <span className={`font-medium ${verdict.tone}`}>{t(verdict.key)}</span>}
      </div>
      {(bench?.fp16_tflops != null || bench?.temp_peak_c != null) && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono">
          {bench?.fp16_tflops != null && <span>{num(bench.fp16_tflops)} TFLOPS</span>}
          {bench?.temp_peak_c != null && <span>{num(bench.temp_peak_c, 0)}°C</span>}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        {badge.reputation_tier && (
          <span className="inline-flex items-center gap-1">
            <Star className="h-3 w-3 text-accent-gold" />
            <span>{tierLabel(badge.reputation_tier)}</span>
            {badge.reputation_score != null && <span className="font-mono">{num(badge.reputation_score, 0)}</span>}
          </span>
        )}
        <span className="inline-flex items-center gap-1">
          <Activity className="h-3 w-3" />
          {badge.job_success_rate != null
            ? t("dash.evidence.success_rate", { pct: pct(badge.job_success_rate) })
            : t("dash.evidence.no_jobs")}
        </span>
      </div>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <dt className="text-xs text-text-muted">{label}</dt>
      <dd className="mt-0.5 font-mono font-medium">{value}</dd>
      {sub && <dd className="text-xs text-text-muted">{sub}</dd>}
    </div>
  );
}

/** The full evidence for the host page. */
export function HostEvidenceCard({ evidence }: { evidence: HostEvidence | null | undefined }) {
  const { t, locale } = useLocale();
  if (!evidence) return null;
  const v = evidence.verification;
  const bench = evidence.benchmark;
  const verdict = bench ? VERDICT[bench.verdict] : undefined;
  const jobs = evidence.jobs;
  return (
    <Card data-testid="host-evidence-card">
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <ShieldCheck className="h-4 w-4 text-emerald" />
        <h2 className="text-sm font-semibold text-text-secondary">{t("dash.evidence.title")}</h2>
        <VerificationChip state={v.freshness} />
        <span className="ml-auto text-xs text-text-muted">
          {t("dash.evidence.last_check", { date: formatEpochDate(v.last_check_at, locale) })}
          {" · "}
          {t("dash.evidence.next_check", { date: formatEpochDate(v.next_check_at, locale) })}
        </span>
      </div>

      {evidence.outside_use && (
        <p
          data-testid="host-evidence-outside-use"
          className="mb-3 flex items-start gap-2 rounded-lg bg-accent-gold/10 px-3 py-2 text-sm text-accent-gold"
        >
          <Activity className="mt-0.5 h-4 w-4 shrink-0" />
          <span>
            {t("dash.evidence.outside_use", {
              processes: evidence.outside_use.processes,
              vram: num(evidence.outside_use.vram_mb / 1024, 1),
            })}
          </span>
        </p>
      )}

      {bench ? (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
            <Gauge className="h-4 w-4 text-ice-blue" />
            <span>{t("dash.evidence.benchmarked", { date: formatEpochDate(bench.ran_at, locale) })}</span>
            <span className="text-xs font-mono text-text-muted">xcelsior-bench {bench.bench_version}</span>
            {verdict && <span className={`text-xs font-semibold ${verdict.tone}`}>{t(verdict.key)}</span>}
            {bench.faults.map((f) => (
              <span key={f} className="rounded-full bg-accent-red/10 px-2 py-0.5 text-[11px] text-accent-red">
                {FAULTS[f] ? t(FAULTS[f]) : f}
              </span>
            ))}
          </div>
          <dl className="grid gap-3 text-sm sm:grid-cols-4">
            <Stat label={t("dash.evidence.fp16")} value={`${num(bench.fp16_tflops)} TFLOPS`} />
            <Stat label={t("dash.evidence.bf16")} value={bench.bf16_tflops != null ? `${num(bench.bf16_tflops)} TFLOPS` : "—"} />
            <Stat label={t("dash.evidence.bandwidth")} value={`${num(bench.memory_bandwidth_gbps, 0)} GB/s`} />
            <Stat label={t("dash.evidence.vram")} value={`${num(bench.allocatable_vram_gb, 2)} GB`} />
            <Stat
              label={t("dash.evidence.pcie")}
              value={`${num(bench.pcie_h2d_gbps)} / ${num(bench.pcie_d2h_gbps)} GB/s`}
            />
            <Stat
              label={t("dash.evidence.sustained")}
              value={bench.sustained.duration_s != null ? `${num(bench.sustained.duration_s, 0)} s` : "—"}
              sub={t("dash.evidence.clock", {
                p50: num(bench.sustained.sm_clock_p50_mhz, 0),
                min: num(bench.sustained.sm_clock_min_mhz, 0),
              })}
            />
            <Stat label={t("dash.evidence.temp")} value={`${num(bench.sustained.temp_peak_c, 0)}°C`} />
            <Stat
              label={t("dash.evidence.throttle")}
              value={pct(bench.sustained.thermal_throttle_share, 1)}
              sub={t("dash.evidence.power_faults", { pct: pct(bench.sustained.power_fault_share, 1) })}
            />
          </dl>
          {evidence.benchmark_history && evidence.benchmark_history.length > 0 && (
            <p className="mt-3 text-xs text-text-muted">
              {t("dash.evidence.history")}{" "}
              {evidence.benchmark_history
                .map((r) => `${formatEpochDate(r.ran_at, locale)} (${r.verdict || "—"})`)
                .join(" · ")}
            </p>
          )}
        </>
      ) : (
        <p className="text-sm text-text-muted">{t("dash.evidence.no_benchmark")}</p>
      )}

      <dl className="mt-4 grid gap-3 border-t border-border/50 pt-4 text-sm sm:grid-cols-2">
        <Stat
          label={t("dash.evidence.reputation")}
          value={evidence.reputation.score != null ? num(evidence.reputation.score, 0) : "—"}
          sub={evidence.reputation.tier ? tierLabel(evidence.reputation.tier) : undefined}
        />
        <Stat
          label={t("dash.evidence.jobs")}
          value={jobs.success_rate != null ? pct(jobs.success_rate) : "—"}
          sub={
            jobs.runs_ended > 0
              ? t("dash.evidence.host_faults", { faults: jobs.host_fault_failures, runs: jobs.runs_ended })
              : t("dash.evidence.no_jobs")
          }
        />
      </dl>
    </Card>
  );
}
