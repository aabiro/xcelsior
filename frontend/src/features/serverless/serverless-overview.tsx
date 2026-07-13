"use client";

import type { ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import type { ServerlessEndpoint, ServerlessEndpointMetrics } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { formatServerlessChip, formatWorkerSecondPrice } from "./format";
import { CopyableText } from "./copyable-text";

interface ServerlessOverviewProps {
  endpoint: ServerlessEndpoint;
  metrics: ServerlessEndpointMetrics | null;
}

function metric(value: number | undefined | null, suffix = "") {
  if (value == null) return "-";
  return `${value.toLocaleString()}${suffix}`;
}

function pct(value: number | undefined | null) {
  if (value == null) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function Row({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-border/50 py-2 last:border-0">
      <span className="text-xs text-text-muted">{label}</span>
      <span className="min-w-0 truncate text-right text-sm font-medium">{value}</span>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-xl border border-border bg-surface p-4">
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      <div>{children}</div>
    </section>
  );
}

export function ServerlessOverview({ endpoint, metrics }: ServerlessOverviewProps) {
  const { t } = useLocale();
  const activeWorkers = metrics?.active_workers ?? 0;
  const bootingWorkers = metrics?.booting_workers ?? 0;
  const workerConfig = `Min ${endpoint.min_workers} - Max ${endpoint.max_workers}`;
  const gpuConfig = `${endpoint.gpu_count || 1} * ${endpoint.gpu_type || endpoint.gpu_tier || "Auto"}`;

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Section title="Summary">
        <Row
          label="Status"
          value={<Badge variant={endpoint.status === "active" ? "active" : "default"}>{formatServerlessChip(endpoint.status)}</Badge>}
        />
        <Row label="Endpoint ID" value={<CopyableText text={endpoint.endpoint_id} />} />
        <Row label="URL" value={endpoint.openai_base_url || endpoint.invoke_path || "-"} />
        <Row label="Type" value={formatServerlessChip(endpoint.execution_mode || "sync")} />
      </Section>

      <Section title="Usage">
        <Row label={t("dash.serverless.metric_requests")} value={metric(metrics?.total_requests ?? endpoint.total_requests)} />
        <Row label={t("dash.serverless.metric_success")} value={pct(metrics?.success_rate)} />
        <Row label={t("dash.serverless.metric_queue_depth")} value={metric(metrics?.queue_depth)} />
        <Row label="Workers running" value={`${activeWorkers} running${bootingWorkers ? `, ${bootingWorkers} starting` : ""}`} />
      </Section>

      <Section title="Cost">
        <Row label="Price" value={formatWorkerSecondPrice(endpoint.pricing)} />
        <Row label="Total cost" value={`$${(metrics?.total_cost_cad ?? endpoint.total_cost_cad ?? 0).toFixed(2)} CAD`} />
        <Row label="Worker seconds" value={metric(metrics?.total_gpu_seconds ?? endpoint.total_gpu_seconds)} />
        <Row label="Billing" value={endpoint.min_workers === 0 ? "No idle worker charge when scaled down" : "Minimum workers stay billable while running"} />
      </Section>

      <Section title="Config">
        <Row label="GPU config" value={gpuConfig} />
        <Row label="Workers" value={workerConfig} />
        <Row label="Idle timeout" value={`${endpoint.idle_timeout_sec ?? 60}s`} />
        <Row label="Region" value={endpoint.region || "-"} />
        <Row label="Image" value={endpoint.image_ref || endpoint.docker_image || "-"} />
      </Section>
    </div>
  );
}
