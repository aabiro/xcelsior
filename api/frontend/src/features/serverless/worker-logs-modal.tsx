"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Dialog } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import type { ServerlessWorker, ServerlessWorkerLogLine } from "@/lib/api";
import * as api from "@/lib/api";
import { ServerlessSelect } from "./serverless-ui";
import { WorkerTelemetryStrip } from "./resource-telemetry";
import { formatServerlessChip } from "./format";

function lineText(line: ServerlessWorkerLogLine): string {
  return String(line.message || line.line || "");
}

function lineTime(line: ServerlessWorkerLogLine): string {
  const ts = Number(line.timestamp || line.ts || 0);
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleTimeString();
}

export function WorkerLogsModal({
  endpointId,
  workers,
  open,
  selectedWorkerId,
  onClose,
}: {
  endpointId: string;
  workers: ServerlessWorker[];
  open: boolean;
  selectedWorkerId?: string | null;
  onClose: () => void;
}) {
  const initial = selectedWorkerId || workers[0]?.worker_id || "";
  const [workerId, setWorkerId] = useState(initial);
  const [logs, setLogs] = useState<ServerlessWorkerLogLine[]>([]);
  const esRef = useRef<EventSource | null>(null);
  const selected = useMemo(
    () => workers.find((w) => w.worker_id === workerId) || workers[0],
    [workers, workerId],
  );

  useEffect(() => {
    if (open) setWorkerId(selectedWorkerId || workers[0]?.worker_id || "");
  }, [open, selectedWorkerId, workers]);

  useEffect(() => {
    esRef.current?.close();
    esRef.current = null;
    setLogs([]);
    if (!open || !workerId) return;

    let cancelled = false;
    api.getServerlessWorkerLogs(endpointId, workerId, 200)
      .then((res) => {
        if (!cancelled) setLogs(res.logs || []);
      })
      .catch(() => {
        if (!cancelled) setLogs([]);
      });

    const es = api.createServerlessWorkerLogStream(endpointId, workerId);
    esRef.current = es;
    const onLog: EventListener = (event) => {
      const message = event as MessageEvent;
      try {
        const data = JSON.parse(message.data);
        setLogs((prev) => [...prev.slice(-499), data]);
      } catch {
        if (message.data) setLogs((prev) => [...prev.slice(-499), { message: message.data }]);
      }
    };
    es.addEventListener("job_log", onLog);
    es.onerror = () => {};

    return () => {
      cancelled = true;
      es.close();
      esRef.current = null;
    };
  }, [endpointId, open, workerId]);

  return (
    <Dialog open={open} onClose={onClose} title="Worker logs" maxWidth="max-w-4xl" bodyClassName="px-6 pb-6 overflow-y-auto space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <ServerlessSelect value={workerId} onChange={(e) => setWorkerId(e.target.value)} className="max-w-sm">
          {workers.map((worker) => (
            <option key={worker.worker_id} value={worker.worker_id}>
              {worker.worker_id}
            </option>
          ))}
        </ServerlessSelect>
        {selected && <Badge variant={selected.state === "ready" || selected.state === "idle" ? "active" : "default"}>{formatServerlessChip(selected.state)}</Badge>}
      </div>

      {selected && <WorkerTelemetryStrip endpointId={endpointId} workerId={selected.worker_id} compact={false} />}

      <pre className="min-h-[320px] max-h-[55vh] overflow-auto rounded-lg border border-border bg-black/60 p-3 font-mono text-xs text-slate-100">
        {logs.length === 0
          ? "Waiting for logs..."
          : logs.map((line) => {
            const prefix = [lineTime(line), line.level].filter(Boolean).join(" ");
            return `${prefix ? `[${prefix}] ` : ""}${lineText(line) || JSON.stringify(line)}\n`;
          }).join("")}
      </pre>
    </Dialog>
  );
}
