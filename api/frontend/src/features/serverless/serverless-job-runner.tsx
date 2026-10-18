"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, Play, Square } from "lucide-react";
import { toast } from "sonner";
import posthog from "posthog-js";
import type { ServerlessEndpoint, ServerlessWarmStatus } from "@/lib/api";
import * as api from "@/lib/api";
import { WorkerTelemetryStrip } from "./resource-telemetry";
import { WorkerLogsModal } from "./worker-logs-modal";
import { formatModelDisplayName, formatServerlessChip } from "./format";

interface ServerlessJobRunnerProps {
  endpoint: ServerlessEndpoint;
  canWrite: boolean;
  defaultPayload?: string;
  compact?: boolean;
}

function bestWorker(warm: ServerlessWarmStatus | null) {
  const workers = warm?.workers || [];
  return workers.find((w) => ["ready", "idle", "booting"].includes(String(w.state))) || workers[0] || null;
}

export function ServerlessJobRunner({
  endpoint,
  canWrite,
  defaultPayload = '{"message": "hello"}',
  compact = false,
}: ServerlessJobRunnerProps) {
  const [payload, setPayload] = useState(defaultPayload);
  const [output, setOutput] = useState("");
  const [running, setRunning] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [warm, setWarm] = useState<ServerlessWarmStatus | null>(null);
  const [logsOpen, setLogsOpen] = useState(false);
  const esRef = useRef<EventSource | null>(null);
  const worker = useMemo(() => bestWorker(warm), [warm]);

  useEffect(() => () => esRef.current?.close(), []);

  const parsePayload = () => {
    try {
      return JSON.parse(payload) as Record<string, unknown>;
    } catch {
      toast.error("Invalid JSON payload");
      return null;
    }
  };

  const warmWorker = async () => {
    setWarm((prev) => prev ?? {
      endpoint_id: endpoint.endpoint_id,
      state: "starting",
      ready_count: 0,
      booting_count: 0,
      active_count: 0,
      workers: [],
    });
    const res = await api.warmServerlessEndpoint(endpoint.endpoint_id);
    setWarm(res.warm);
    return res.warm;
  };

  const stop = () => {
    esRef.current?.close();
    esRef.current = null;
    if (activeJobId) api.cancelServerlessJob(endpoint.endpoint_id, activeJobId).catch(() => {});
    setActiveJobId(null);
    setRunning(false);
  };

  const runAsync = async () => {
    if (!canWrite) return toast.error("Viewer access cannot run jobs");
    const input = parsePayload();
    if (!input) return;
    setOutput("");
    setRunning(true);
    try {
      await warmWorker();
      const res = await api.runServerlessTestJob(endpoint.endpoint_id, input);
      if (res.warm) setWarm(res.warm);
      setActiveJobId(res.id);
      setOutput(`Job ${res.id} ${res.status}\n`);
      posthog.capture("serverless_inference_run", { mode: "job_async", model: endpoint.model_ref || endpoint.model_name });

      const es = api.createServerlessJobStream(endpoint.endpoint_id, res.id);
      esRef.current = es;
      const append = (text: string) => setOutput((prev) => prev + text);
      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const chunk = data?.payload?.text ?? data?.payload?.output ?? data?.text ?? "";
          if (chunk) append(typeof chunk === "string" ? chunk : JSON.stringify(chunk, null, 2));
          if (data?.event_type === "done" || data?.event_type === "error") stop();
        } catch {
          if (event.data && event.data !== "[DONE]") append(`${event.data}\n`);
        }
      };
      es.onerror = () => {
        api.getServerlessJobStatus(endpoint.endpoint_id, res.id)
          .then((st) => {
            if (st.output) append(JSON.stringify(st.output, null, 2));
            if (st.error) append(JSON.stringify(st.error, null, 2));
          })
          .finally(stop);
      };
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Run failed");
      setRunning(false);
    }
  };

  const runSync = async () => {
    if (!canWrite) return toast.error("Viewer access cannot run jobs");
    const input = parsePayload();
    if (!input) return;
    setOutput("");
    setRunning(true);
    try {
      await warmWorker();
      const res = await api.runServerlessTestJobSync(endpoint.endpoint_id, input);
      setOutput(JSON.stringify(res.output ?? res, null, 2));
      posthog.capture("serverless_inference_run", { mode: "job_sync", model: endpoint.model_ref || endpoint.model_name });
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Run failed");
      setOutput(error instanceof Error ? error.message : String(error));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="glow-card rounded-xl border border-border bg-surface p-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="text-sm font-medium">Run test job</p>
          <p className="text-xs text-text-muted">{formatModelDisplayName(endpoint.model_ref || endpoint.model_name || endpoint.endpoint_id)}</p>
        </div>
        {warm && <Badge variant={warm.state === "ready" ? "active" : "default"}>{formatServerlessChip(warm.state)}</Badge>}
      </div>
      <textarea
        className="w-full min-h-[100px] rounded-lg border border-border bg-background px-3 py-2 font-mono text-xs"
        value={payload}
        onChange={(e) => setPayload(e.target.value)}
      />
      <div className="flex flex-wrap gap-2">
        <Button onClick={runSync} disabled={running || !canWrite}>
          {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          Run Sync
        </Button>
        <Button variant="outline" onClick={runAsync} disabled={running || !canWrite}>
          Run Async
        </Button>
        {running && (
          <Button variant="ghost" onClick={stop}>
            <Square className="h-4 w-4" /> Stop
          </Button>
        )}
        {worker && (
          <Button variant="ghost" onClick={() => setLogsOpen(true)}>
            Logs
          </Button>
        )}
      </div>
      {warm && (
        <div className="rounded-lg border border-border/60 bg-surface-hover/30 p-3">
          <div className="mb-2 flex flex-wrap gap-3 text-xs text-text-muted">
            <span>Ready {warm.ready_count}</span>
            <span>Booting {warm.booting_count}</span>
            <span>Active {warm.active_count}</span>
          </div>
          {worker ? (
            <WorkerTelemetryStrip endpointId={endpoint.endpoint_id} workerId={worker.worker_id} compact={compact} />
          ) : (
            <span className="text-xs text-text-muted">Starting worker...</span>
          )}
        </div>
      )}
      {output && (
        <pre className="max-h-80 overflow-auto rounded-lg border border-border bg-black/60 p-3 font-mono text-xs text-slate-100">
          {output}
        </pre>
      )}
      <WorkerLogsModal
        endpointId={endpoint.endpoint_id}
        workers={warm?.workers || []}
        selectedWorkerId={worker?.worker_id}
        open={logsOpen && !!worker}
        onClose={() => setLogsOpen(false)}
      />
    </div>
  );
}
