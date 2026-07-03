"use client";

import { StatCard } from "@/components/ui/stat-card";
import { DollarSign, Zap, Clock, BarChart3 } from "lucide-react";
import type { ServerlessEndpoint, ServerlessEndpointMetrics } from "@/lib/api";
import { useLocale } from "@/lib/locale";

interface CostUsagePanelProps {
  endpoint: ServerlessEndpoint;
  metrics: ServerlessEndpointMetrics | null;
}

export function CostUsagePanel({ endpoint, metrics }: CostUsagePanelProps) {
  const { t } = useLocale();
  const pricing = endpoint.pricing;

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label={t("dash.serverless.total_cost")}
          value={`$${(metrics?.total_cost_cad ?? endpoint.total_cost_cad ?? 0).toFixed(2)} CAD`}
          icon={DollarSign}
          glow="gold"
        />
        <StatCard
          label={t("dash.serverless.gpu_seconds")}
          value={(metrics?.total_gpu_seconds ?? endpoint.total_gpu_seconds ?? 0).toLocaleString()}
          icon={Zap}
          glow="violet"
        />
        <StatCard
          label={t("dash.serverless.avg_exec")}
          value={metrics?.avg_execution_ms ? `${Math.round(metrics.avg_execution_ms)}ms` : "-"}
          icon={Clock}
          glow="cyan"
        />
        <StatCard
          label={t("dash.serverless.total_requests")}
          value={(metrics?.total_requests ?? endpoint.total_requests ?? 0).toLocaleString()}
          icon={BarChart3}
          glow="emerald"
        />
      </div>

      {pricing && (
        <div className="glow-card brand-top-accent rounded-xl border border-border bg-surface p-5">
          <p className="text-sm font-semibold mb-3">{t("dash.serverless.pricing_title")}</p>
          <div className="grid gap-2 text-sm">
            <div className="flex justify-between">
              <span className="text-text-muted">{t("dash.serverless.rate_hour")}</span>
              <span className="font-mono">${pricing.rate_per_hour_cad.toFixed(2)} CAD/hr</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">{t("dash.serverless.rate_second")}</span>
              <span className="font-mono">{pricing.rate_cents_per_second_per_worker}¢/s/worker</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">{t("dash.serverless.gpu_count")}</span>
              <span className="font-mono">{pricing.gpu_count}× GPU</span>
            </div>
            <p className="text-xs text-text-muted pt-2 border-t border-border">{pricing.formula}</p>
          </div>
        </div>
      )}
    </div>
  );
}