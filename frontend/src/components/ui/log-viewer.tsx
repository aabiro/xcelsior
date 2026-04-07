"use client";

import { useEffect, useRef, useState } from "react";
import { createInstanceLogStream, fetchInstanceLogs } from "@/lib/api";
import type { InstanceLog } from "@/lib/api";
import { Download } from "lucide-react";

interface LogViewerProps {
  jobId: string;
  live?: boolean;
  /** Logs delivered from an external source (e.g. WebSocket) – skips SSE when provided. */
  wsLogs?: InstanceLog[];
  /** Connection status from external source. */
  wsConnected?: boolean;
}

const LEVEL_COLORS: Record<string, string> = {
  error: "text-accent-red",
  stderr: "text-accent-red",
  warn: "text-accent-gold",
  warning: "text-accent-gold",
  info: "text-ice-blue",
  debug: "text-text-muted",
};

export function LogViewer({ jobId, live = false, wsLogs, wsConnected }: LogViewerProps) {
  const [logs, setLogs] = useState<InstanceLog[]>([]);
  const [connected, setConnected] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  // When WS logs are provided, use those instead of internal state
  const useWs = wsLogs !== undefined;
  const displayLogs = useWs ? [...logs, ...wsLogs] : logs;
  const displayConnected = useWs ? !!wsConnected : connected;

  // Load historical logs
  useEffect(() => {
    fetchInstanceLogs(jobId, 500)
      .then((r) => setLogs(r.logs || []))
      .catch((e) => console.error("Failed to load logs", e));
  }, [jobId]);

  // SSE live stream — skip when WS logs are provided
  useEffect(() => {
    if (!live || useWs) return;

    const es = createInstanceLogStream(jobId);

    es.addEventListener("connected", () => setConnected(true));

    es.addEventListener("job_log", (e) => {
      try {
        const data = JSON.parse(e.data);
        setLogs((prev) => [...prev, data].slice(-5000));
      } catch {}
    });

    es.addEventListener("job_status", (e) => {
      try {
        const data = JSON.parse(e.data);
        setLogs((prev) => [
          ...prev,
          { timestamp: new Date().toISOString(), level: "info", message: `Status changed to ${data.status}` },
        ].slice(-5000));
      } catch {}
    });

    es.onerror = () => setConnected(false);

    return () => es.close();
  }, [jobId, live, useWs]);

  // Auto-scroll
  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [displayLogs, autoScroll]);

  function handleScroll() {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    setAutoScroll(scrollHeight - scrollTop - clientHeight < 40);
  }

  function handleDownload() {
    const text = displayLogs.map((log) => {
      const rawTs = typeof log.timestamp === "number" && log.timestamp < 1e12
        ? log.timestamp * 1000
        : log.timestamp;
      const d = rawTs ? new Date(rawTs) : null;
      const ts = d && !isNaN(d.getTime()) ? d.toISOString() : "";
      const level = log.level ? `[${log.level.toUpperCase()}]` : "";
      return `${ts} ${level} ${log.message}`;
    }).join("\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${jobId}-logs.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-2">
      {live && (
        <div className="flex items-center gap-2 text-xs">
          <span className={`h-2 w-2 rounded-full ${displayConnected ? "bg-emerald animate-pulse" : "bg-text-muted"}`} />
          <span className="text-text-muted">{displayConnected ? "Live" : "Disconnected"}</span>
          {displayLogs.length > 0 && (
            <button
              onClick={handleDownload}
              title="Download logs"
              className="ml-auto flex items-center gap-1 text-text-muted hover:text-text-primary transition-colors"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Download</span>
            </button>
          )}
        </div>
      )}
      {!live && displayLogs.length > 0 && (
        <div className="flex justify-end">
          <button
            onClick={handleDownload}
            title="Download logs"
            className="flex items-center gap-1 text-xs text-text-muted hover:text-text-primary transition-colors"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Download</span>
          </button>
        </div>
      )}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="max-h-96 overflow-y-auto rounded-lg bg-navy border border-border p-4 font-mono text-xs leading-relaxed"
      >
        {displayLogs.length === 0 ? (
          <p className="text-text-muted">No logs yet.</p>
        ) : (
          displayLogs.map((log, i) => {
            const color = LEVEL_COLORS[log.level?.toLowerCase() || ""] || "text-text-secondary";
            const rawTs = typeof log.timestamp === "number" && log.timestamp < 1e12
              ? log.timestamp * 1000
              : log.timestamp;
            const d = rawTs ? new Date(rawTs) : null;
            const ts = d && !isNaN(d.getTime())
              ? d.toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })
              : "";
            return (
              <div key={i} className="flex gap-2 hover:bg-surface-hover -mx-1 px-1 rounded">
                {ts && <span className="text-text-muted shrink-0 select-none">{ts}</span>}
                {log.level && (
                  <span className={`shrink-0 uppercase w-12 ${color}`}>{log.level}</span>
                )}
                <span className={color}>{log.message}</span>
              </div>
            );
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
