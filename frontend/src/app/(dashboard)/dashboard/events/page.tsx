"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/input";
import { Calendar, RefreshCw, Radio, Download, Trash2, Wifi, WifiOff } from "lucide-react";
import { createEventSource } from "@/lib/api";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";

interface Event {
  id?: string;
  type: string;
  severity?: "info" | "warning" | "error" | "critical";
  data?: any;
  timestamp: string;
  message?: string;
}

const SEVERITY_COLORS: Record<string, { dot: string; badge: "info" | "warning" | "failed" | "completed" }> = {
  info: { dot: "bg-ice-blue", badge: "info" },
  warning: { dot: "bg-accent-gold", badge: "warning" },
  error: { dot: "bg-accent-red", badge: "failed" },
  critical: { dot: "bg-accent-red animate-pulse", badge: "failed" },
};

const MAX_RECONNECT_DELAY = 30000;

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "reconnecting";

export default function EventsPage() {
  const { t } = useLocale();
  const [events, setEvents] = useState<Event[]>([]);
  const [filter, setFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [live, setLive] = useState(false);
  const [connStatus, setConnStatus] = useState<ConnectionStatus>("disconnected");
  const esRef = useRef<EventSource | null>(null);
  const reconnectAttempt = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Load historical events
  const loadHistory = useCallback(() => {
    fetch("/api/events", { credentials: "include" })
      .then((r) => r.ok ? r.json() : Promise.reject())
      .then((d) => setEvents(Array.isArray(d.events) ? d.events : []))
      .catch(() => toast.error("Failed to load events"));
  }, []);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  // SSE live stream with exponential backoff reconnect
  const connectSSE = useCallback(() => {
    if (esRef.current) esRef.current.close();
    setConnStatus(reconnectAttempt.current > 0 ? "reconnecting" : "connecting");

    const es = createEventSource();
    es.onopen = () => { reconnectAttempt.current = 0; setConnStatus("connected"); };
    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data);
        setEvents((prev) => [event, ...prev].slice(0, 500));
      } catch {}
    };
    es.onerror = () => {
      es.close();
      if (live) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempt.current), MAX_RECONNECT_DELAY);
        reconnectAttempt.current++;
        setConnStatus("reconnecting");
        reconnectTimer.current = setTimeout(connectSSE, delay);
      } else {
        setConnStatus("disconnected");
      }
    };
    esRef.current = es;
  }, [live]);

  useEffect(() => {
    if (live) {
      connectSSE();
    } else {
      esRef.current?.close();
      esRef.current = null;
      setConnStatus("disconnected");
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    }
    return () => {
      esRef.current?.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [live, connectSSE]);

  const filtered = events.filter((e) => {
    if (filter !== "all" && e.type !== filter) return false;
    if (severityFilter !== "all" && (e.severity || "info") !== severityFilter) return false;
    return true;
  });

  const eventTypes = [...new Set(events.map((e) => e.type).filter(Boolean))];

  // Download events as JSON log
  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(filtered, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `xcelsior-events-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getSeverity = (e: Event) => e.severity || "info";

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold">{t("dash.events.title")}</h1>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleDownload} disabled={filtered.length === 0}>
            <Download className="h-3.5 w-3.5" /> {t("dash.events.export")}
          </Button>
          <Button variant="outline" size="sm" onClick={loadHistory}>
            <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
          </Button>
          <Button
            variant={live ? "success" : "outline"}
            size="sm"
            onClick={() => setLive(!live)}
          >
            <Radio className={`h-3.5 w-3.5 ${live ? "animate-pulse" : ""}`} />
            {live ? t("dash.events.live") : t("dash.events.connect")}
          </Button>
          {/* Connection status indicator */}
          {live && (
            <span className="flex items-center gap-1.5 text-xs">
              {connStatus === "connected" ? (
                <><Wifi className="h-3.5 w-3.5 text-emerald" /><span className="text-emerald">Connected</span></>
              ) : connStatus === "reconnecting" ? (
                <><WifiOff className="h-3.5 w-3.5 text-accent-gold animate-pulse" /><span className="text-accent-gold">Reconnecting…</span></>
              ) : connStatus === "connecting" ? (
                <><Wifi className="h-3.5 w-3.5 text-text-muted animate-pulse" /><span className="text-text-muted">Connecting…</span></>
              ) : null}
            </span>
          )}
        </div>
      </div>

      <div className="flex gap-3 flex-wrap">
        <Select value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="all">All Types</option>
          {eventTypes.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </Select>
        <Select value={severityFilter} onChange={(e) => setSeverityFilter(e.target.value)}>
          <option value="all">All Severity</option>
          <option value="info">Info</option>
          <option value="warning">Warning</option>
          <option value="error">Error</option>
          <option value="critical">Critical</option>
        </Select>
        {events.length > 0 && (
          <span className="flex items-center text-xs text-text-muted">
            {filtered.length} of {events.length} events
          </span>
        )}
      </div>

      <Card>
        <CardContent className="p-0">
          {filtered.length === 0 ? (
            <div className="p-12 text-center">
              <Calendar className="mx-auto h-12 w-12 text-text-muted mb-4" />
              <h3 className="text-lg font-semibold mb-1">No events</h3>
              <p className="text-sm text-text-secondary">Events will stream here in real-time.</p>
            </div>
          ) : (
            <div className="divide-y divide-border max-h-[600px] overflow-y-auto">
              {filtered.map((event, i) => {
                const sev = getSeverity(event);
                const colors = SEVERITY_COLORS[sev] || SEVERITY_COLORS.info;
                return (
                  <div key={event.id || i} className="flex items-start gap-3 p-4 hover:bg-surface-hover">
                    <div className={`mt-1.5 h-2 w-2 rounded-full shrink-0 ${colors.dot}`} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                        <Badge variant={colors.badge}>{event.type}</Badge>
                        {sev !== "info" && (
                          <Badge variant={colors.badge} className="text-[10px] px-1.5 py-0">{sev}</Badge>
                        )}
                        <span className="text-xs text-text-muted">
                          {event.timestamp ? new Date(event.timestamp).toLocaleString() : "—"}
                        </span>
                      </div>
                      <p className="text-sm text-text-secondary truncate">
                        {event.message || (event.data
                          ? Object.entries(event.data)
                              .map(([k, v]) => `${k}: ${v}`)
                              .join(" · ")
                          : event.type?.replace(/_/g, " ") || "—")}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
