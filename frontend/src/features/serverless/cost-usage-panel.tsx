"use client";

import { StatCard } from "@/components/ui/stat-card";
import { DollarSign, Zap, Clock, BarChart3 } from "lucide-react";
import type { ServerlessEndpoint, ServerlessEndpointMetrics } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { formatTokenRateFromPricing } from "./constants";
import { formatWorkerSecondPrice } from "./format";

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
	              <span className="text-text-muted">Worker rate</span>
	              <span className="font-mono">{formatWorkerSecondPrice(pricing).replace("Price: ", "")}</span>
	            </div>
	            <div className="flex justify-between">
	              <span className="text-text-muted">{t("dash.serverless.gpu_count")}</span>
	              <span className="font-mono">{pricing.gpu_count}× GPU</span>
            </div>
            {pricing.token_billing && (
              <div className="flex justify-between">
                <span className="text-text-muted">Tokens</span>
                <span className="font-mono text-accent-cyan">
                  {formatTokenRateFromPricing(
                    endpoint.model_ref || endpoint.model_id || "",
                    pricing,
                  )}
                </span>
              </div>
            )}
            {metrics?.kv_cache_hit_rate != null && metrics.kv_cache_hit_rate > 0 && (
              <div className="flex justify-between">
                <span className="text-text-muted">KV cache hit</span>
                <span className="font-mono">{(metrics.kv_cache_hit_rate * 100).toFixed(1)}%</span>
              </div>
            )}
            {metrics?.tokens_per_sec != null && metrics.tokens_per_sec > 0 && (
              <div className="flex justify-between">
                <span className="text-text-muted">Throughput</span>
                <span className="font-mono">{metrics.tokens_per_sec.toLocaleString()} tok/s</span>
              </div>
            )}
            {metrics?.ttft_p95_ms != null && metrics.ttft_p95_ms > 0 && (
              <div className="flex justify-between">
                <span className="text-text-muted">TTFT p95</span>
                <span className="font-mono">{Math.round(metrics.ttft_p95_ms)}ms</span>
              </div>
            )}
	            <p className="text-xs text-text-muted pt-2 border-t border-border">
	              Charged for running workers. With min workers set to 0, idle scaled-down endpoints do not keep a worker running.
	            </p>
	          </div>
        </div>
      )}
    </div>
  );
}
