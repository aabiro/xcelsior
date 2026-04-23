"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge, Badge } from "@/components/ui/badge";
import {
  ArrowLeft, Server, Cpu, MapPin, Gauge, ShieldCheck, Star, Zap, Thermometer, HardDrive,
  ArrowDownToLine, ArrowUpFromLine, Database,
} from "lucide-react";
import {
  fetchHost, fetchComputeScore, fetchSlaStatus, fetchVerificationStatus,
  fetchTelemetry, fetchReputation,
} from "@/lib/api";
import type { Host, TelemetryData, ReputationEntry } from "@/lib/api";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";

export default function HostDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { t } = useLocale();
  const [host, setHost] = useState<Host | null>(null);
  const [loading, setLoading] = useState(true);
  const [computeScore, setComputeScore] = useState<number | null>(null);
  const [sla, setSla] = useState<{ uptime_30d_pct: number } | null>(null);
  const [verification, setVerification] = useState<{ status: string } | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [reputation, setReputation] = useState<ReputationEntry | null>(null);

  useEffect(() => {
    setLoading(true);

    fetchHost(id)
      .then((r) => setHost(r.host))
      .catch(() => toast.error("Failed to load host"))
      .finally(() => setLoading(false));

    // Parallel secondary fetches — each can fail independently
    fetchComputeScore(id).then((r) => setComputeScore(r.score)).catch((e) => console.error("Failed to load compute score", e));
    fetchSlaStatus(id).then((r) => setSla(r)).catch((e) => console.error("Failed to load SLA status", e));
    fetchVerificationStatus(id).then((r) => setVerification(r)).catch((e) => console.error("Failed to load verification", e));
    fetchReputation(id).then((r) => setReputation(r.reputation)).catch((e) => console.error("Failed to load reputation", e));
    fetchTelemetry()
      .then((r) => {
        const data = r.telemetry?.[id];
        if (data) setTelemetry(data);
      })
      .catch((e) => console.error("Failed to load telemetry", e));
  }, [id]);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-48 rounded bg-surface skeleton-pulse" />
        <div className="h-40 rounded-xl bg-surface skeleton-pulse" />
        <div className="h-64 rounded-xl bg-surface skeleton-pulse" />
      </div>
    );
  }

  if (!host) {
    return (
      <Card className="p-12 text-center">
        <h2 className="text-xl font-semibold mb-2">{t("dash.hosts.not_found")}</h2>
        <p className="text-text-secondary mb-4">{t("dash.hosts.not_found_desc")}</p>
        <Link href="/dashboard/hosts"><Button variant="outline">{t("dash.hosts.back")}</Button></Link>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link href="/dashboard/hosts">
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold">{host.hostname || host.host_id}</h1>
            <p className="text-sm font-mono text-text-muted">{host.host_id}</p>
          </div>
        </div>
        <StatusBadge status={host.status} />
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-ice-blue/10">
            <Cpu className="h-5 w-5 text-ice-blue" />
          </div>
          <div>
            <p className="text-xs text-text-muted">{t("dash.hosts.label_gpu")}</p>
            <p className="text-xs text-text-muted">{host.vram_gb}GB VRAM</p>
          </div>
        </Card>

        <Card className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald/10">
            <MapPin className="h-5 w-5 text-emerald" />
          </div>
          <div>
            <p className="text-xs text-text-muted">{t("dash.hosts.label_location")}</p>
            <p className="font-medium">{host.province || "—"}, {host.country || "CA"}</p>
          </div>
        </Card>

        <Card className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent-gold/10">
            <Zap className="h-5 w-5 text-accent-gold" />
          </div>
          <div>
            <p className="text-xs text-text-muted">{t("dash.hosts.label_price")}</p>
            <p className="font-medium font-mono">${Number(host.cost_per_hour || host.price_per_hour || 0).toFixed(2)}/hr</p>
          </div>
        </Card>

        <Card className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent-red/10">
            <Gauge className="h-5 w-5 text-accent-red" />
          </div>
          <div>
            <p className="text-xs text-text-muted">{t("dash.hosts.label_compute")}</p>
            <p className="font-medium font-mono">{computeScore != null ? computeScore.toFixed(0) : "—"}</p>
          </div>
        </Card>
      </div>

      {/* Verification & Reputation */}
      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <ShieldCheck className="h-4 w-4 text-emerald" />
            <h2 className="text-sm font-semibold text-text-secondary">{t("dash.hosts.label_verification")}</h2>
          </div>
          {verification ? (
            <div className="flex items-center gap-3">
              <Badge variant={verification.status === "verified" ? "active" : verification.status === "deverified" ? "failed" : "default"}>
                {verification.status}
              </Badge>
              <span className="text-sm text-text-secondary">
                {verification.status === "verified"
                  ? t("dash.hosts.verified")
                  : verification.status === "deverified"
                    ? t("dash.hosts.deverified")
                    : t("dash.hosts.pending")}
              </span>
            </div>
          ) : (
            <p className="text-sm text-text-muted">{t("dash.hosts.no_verification")}</p>
          )}
        </Card>

        <Card>
          <div className="flex items-center gap-2 mb-4">
            <Star className="h-4 w-4 text-accent-gold" />
            <h2 className="text-sm font-semibold text-text-secondary">{t("dash.hosts.label_reputation")}</h2>
          </div>
          {reputation ? (
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <span className="text-2xl font-bold font-mono">{reputation.score}</span>
                <Badge variant={
                  reputation.tier === "gold" || reputation.tier === "platinum" ? "active" :
                  reputation.tier === "silver" ? "info" : "default"
                }>
                  {reputation.tier}
                </Badge>
              </div>
              {reputation.jobs_completed != null && (
                <p className="text-xs text-text-muted">{t("dash.hosts.instances_completed", { count: reputation.jobs_completed })}</p>
              )}
            </div>
          ) : (
            <p className="text-sm text-text-muted">{t("dash.hosts.no_reputation")}</p>
          )}
        </Card>
      </div>

      {/* SLA */}
      <Card>
        <div className="flex items-center gap-2 mb-4">
          <Gauge className="h-4 w-4 text-ice-blue" />
          <h2 className="text-sm font-semibold text-text-secondary">{t("dash.hosts.label_sla")}</h2>
        </div>
        {sla ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-secondary">{t("dash.hosts.uptime_30d")}</span>
              <span className="text-lg font-bold font-mono">
                {sla.uptime_30d_pct.toFixed(2)}%
              </span>
            </div>
            <div className="h-2 rounded-full bg-navy overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  sla.uptime_30d_pct >= 99.5 ? "bg-emerald" :
                  sla.uptime_30d_pct >= 97 ? "bg-ice-blue" :
                  sla.uptime_30d_pct >= 95 ? "bg-accent-gold" : "bg-accent-red"
                }`}
                style={{ width: `${Math.min(sla.uptime_30d_pct, 100)}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-text-muted">
              <span>{t("dash.hosts.tier_bronze")}</span>
              <span>{t("dash.hosts.tier_silver")}</span>
              <span>{t("dash.hosts.tier_gold")}</span>
              <span>{t("dash.hosts.tier_platinum")}</span>
            </div>
          </div>
        ) : (
          <p className="text-sm text-text-muted">{t("dash.hosts.no_sla")}</p>
        )}
      </Card>

      {/* Telemetry */}
      <Card>
        <h2 className="text-sm font-semibold text-text-secondary mb-4">{t("dash.hosts.live_telemetry")}</h2>
        {telemetry ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <TelemetryGauge
              icon={<Cpu className="h-4 w-4 text-ice-blue" />}
              label={t("dash.hosts.gpu_util")}
              value={telemetry.gpu_utilization_pct ?? telemetry.gpu_util ?? 0}
              unit="%"
              max={100}
              color="bg-ice-blue"
            />
            <TelemetryGauge
              icon={<Thermometer className="h-4 w-4 text-accent-gold" />}
              label={t("dash.hosts.temperature")}
              value={telemetry.gpu_temp_c ?? telemetry.temperature ?? 0}
              unit="°C"
              max={100}
              color={
                (telemetry.gpu_temp_c ?? telemetry.temperature ?? 0) > 85 ? "bg-accent-red" :
                (telemetry.gpu_temp_c ?? telemetry.temperature ?? 0) > 70 ? "bg-accent-gold" : "bg-emerald"
              }
            />
            <TelemetryGauge
              icon={<HardDrive className="h-4 w-4 text-emerald" />}
              label={t("dash.hosts.mem_used")}
              value={telemetry.gpu_memory_used_mb ?? telemetry.mem_used_mb ?? 0}
              unit="MB"
              max={telemetry.gpu_memory_total_mb || 1}
              color="bg-emerald"
            />
            <TelemetryGauge
              icon={<Zap className="h-4 w-4 text-accent-red" />}
              label={t("dash.hosts.power_draw")}
              value={telemetry.gpu_power_draw_w ?? telemetry.power_draw_w ?? 0}
              unit="W"
              max={400}
              color="bg-accent-red"
            />
            {/* P2.4: network + disk bandwidth — only render if the agent reports them. */}
            {telemetry.net_rx_mbps !== undefined && (
              <TelemetryGauge
                icon={<ArrowDownToLine className="h-4 w-4 text-ice-blue" />}
                label={t("dash.hosts.net_rx")}
                value={telemetry.net_rx_mbps}
                unit=" Mbps"
                max={1000}
                color="bg-ice-blue"
              />
            )}
            {telemetry.net_tx_mbps !== undefined && (
              <TelemetryGauge
                icon={<ArrowUpFromLine className="h-4 w-4 text-ice-blue" />}
                label={t("dash.hosts.net_tx")}
                value={telemetry.net_tx_mbps}
                unit=" Mbps"
                max={1000}
                color="bg-ice-blue"
              />
            )}
            {telemetry.disk_read_mb_s !== undefined && (
              <TelemetryGauge
                icon={<Database className="h-4 w-4 text-emerald" />}
                label={t("dash.hosts.disk_read")}
                value={telemetry.disk_read_mb_s}
                unit=" MB/s"
                max={1000}
                color="bg-emerald"
              />
            )}
            {telemetry.disk_write_mb_s !== undefined && (
              <TelemetryGauge
                icon={<Database className="h-4 w-4 text-accent-gold" />}
                label={t("dash.hosts.disk_write")}
                value={telemetry.disk_write_mb_s}
                unit=" MB/s"
                max={1000}
                color="bg-accent-gold"
              />
            )}
          </div>
        ) : (
          <p className="text-sm text-text-muted">{t("dash.hosts.no_telemetry")}</p>
        )}
      </Card>
    </div>
  );
}

function TelemetryGauge({
  icon,
  label,
  value,
  unit,
  max,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
  unit: string;
  max: number;
  color: string;
}) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-xs text-text-muted">{label}</span>
      </div>
      <div className="text-lg font-bold font-mono">
        {typeof value === "number" ? value.toLocaleString() : value}{unit}
      </div>
      <div className="h-1.5 rounded-full bg-navy overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
