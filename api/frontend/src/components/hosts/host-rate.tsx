"use client";

/**
 * What renters pay on a host and what its operator keeps. Rates are set per
 * GPU model (the API's `host_rate_fields`); a host does not choose a price.
 */
import type { Host } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";

/** $0.12, $0.096: per-hour rates under a dollar keep the third decimal. */
export function formatHourlyRate(value: number): string {
  if (value >= 1) return `$${value.toFixed(2)}`;
  const text = value.toFixed(3);
  return `$${text.endsWith("0") ? text.slice(0, -1) : text}`;
}

type RateFields = Pick<Host, "renter_rate_per_hour" | "host_earns_per_hour" | "platform_cut_pct">;

export function HostRate({ host, className }: { host: RateFields; className?: string }) {
  const { t } = useLocale();
  const rate = host.renter_rate_per_hour;
  if (rate == null) {
    return (
      <span className={cn("text-sm text-text-muted", className)} title={t("dash.hosts.rate_unpriced_hint")}>
        {t("dash.hosts.rate_unpriced")}
      </span>
    );
  }
  const earns = host.host_earns_per_hour;
  return (
    <span
      className={cn("inline-flex flex-col items-center leading-tight", className)}
      title={t("dash.hosts.rate_hint", {
        rate: formatHourlyRate(rate),
        cut: host.platform_cut_pct ?? 0,
      })}
      data-testid="host-rate"
    >
      <span className="font-mono text-sm font-medium text-emerald">{formatHourlyRate(rate)}/hr</span>
      {earns != null && (
        <span className="text-[11px] text-text-muted">
          {t("dash.hosts.rate_earn", { amount: formatHourlyRate(earns) })}
        </span>
      )}
    </span>
  );
}
