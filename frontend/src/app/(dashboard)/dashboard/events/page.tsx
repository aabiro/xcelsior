"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/input";
import { Calendar, RefreshCw, Radio, Download, Wifi, WifiOff, EyeOff } from "lucide-react";
import { createEventSource, isEventStreamAvailable, apiFetch } from "@/lib/api";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";

interface Event {
  id?: string;
  type: string;
  severity?: "info" | "warning" | "error" | "critical";
  data?: any;
  timestamp: string | number;
  event_type?: string;
  event_id?: string;
  message?: string;
}

const SEVERITY_COLORS: Record<string, { dot: string; badge: "info" | "warning" | "failed" | "completed" }> = {
  info: { dot: "bg-ice-blue", badge: "info" },
  warning: { dot: "bg-accent-gold", badge: "warning" },
  error: { dot: "bg-accent-red", badge: "failed" },
  critical: { dot: "bg-accent-red animate-pulse", badge: "failed" },
};

/** Event types that are verbose logs, not meaningful lifecycle events. */
const VERBOSE_EVENT_TYPES = new Set(["job_log", "spot_prices", "host_update"]);

const MAX_RECONNECT_DELAY = 30000;

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "reconnecting";

export default function EventsPage() {
  const { t } = useLocale();
  const [events, setEvents] = useState<Event[]>([]);
  const [filter, setFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [showVerbose, setShowVerbose] = useState(false);
  const [live, setLive] = useState(false);
  const [connStatus, setConnStatus] = useState<ConnectionStatus>("disconnected");
  const [cursors, setCursors] = useState<(string | null)[]>([null]);
  const [pageIndex, setPageIndex] = useState(0);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [allTypes, setAllTypes] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [newEvents, setNewEvents] = useState(0);
  const latest = useRef(0);
  const before = cursors[pageIndex];
  const esRef = useRef<EventSource | null>(null);
  const reconnectAttempt = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const resetPage = () => { setCursors([null]); setPageIndex(0); };
  const loadHistory = useCallback(async () => {
    const request = ++latest.current;
    setLoading(true);
    setFailed(false);
    const query = new URLSearchParams({ limit: "25", include_verbose: String(showVerbose) });
    if (before) query.set("before", before);
    if (filter !== "all") query.set("event_type", filter);
    if (severityFilter !== "all") query.set("severity", severityFilter);
    try {
      const data = await apiFetch<{ events: Event[]; total: number; next_cursor: string | null; event_types: string[] }>(`/api/events?${query}`);
      if (request !== latest.current) return;
      setEvents((data.events || []).map((event) => ({ ...event, id: event.event_id ?? event.id, type: event.event_type ?? event.type })));
      setTotal(data.total);
      setNextCursor(data.next_cursor);
      setAllTypes(data.event_types || []);
      setNewEvents(0);
    } catch {
      if (request === latest.current) { setFailed(true); toast.error("Failed to load events"); }
    } finally {
      if (request === latest.current) setLoading(false);
    }
  }, [before, filter, severityFilter, showVerbose]);

  useEffect(() => { void loadHistory(); return () => { latest.current++; }; }, [loadHistory]);

  useEffect(() => {
    if (!live) { setConnStatus("disconnected"); return; }
    let active = true;
    reconnectAttempt.current = 0;
    const connect = async () => {
      setConnStatus(reconnectAttempt.current ? "reconnecting" : "connecting");
      try {
        const available = await isEventStreamAvailable("/api/stream", { force: true });
        if (!active) return;
        if (!available) {
          setLive(false);
          toast.error("Live event stream is unavailable right now.");
          return;
        }
        const es = createEventSource();
        esRef.current = es;
        es.onopen = () => { if (active) { reconnectAttempt.current = 0; setConnStatus("connected"); } };
        es.onmessage = (message) => {
          if (!active) return;
          try {
            const event = JSON.parse(message.data);
            if (event.type || event.event_type) setNewEvents((count) => count + 1);
          } catch { /* Ignore stream keepalives. */ }
        };
        es.onerror = () => {
          es.close();
          if (active) reconnect();
        };
      } catch { if (active) reconnect(); }
    };
    const reconnect = () => {
      const delay = Math.min(1000 * 2 ** reconnectAttempt.current++, MAX_RECONNECT_DELAY);
      setConnStatus("reconnecting");
      reconnectTimer.current = setTimeout(() => void connect(), delay);
    };
    void connect();
    return () => {
      active = false;
      esRef.current?.close();
      esRef.current = null;
      clearTimeout(reconnectTimer.current);
    };
  }, [live]);

  const filtered = events.filter((e) => {
    if (!showVerbose && VERBOSE_EVENT_TYPES.has(e.type)) return false;
    if (filter !== "all" && e.type !== filter) return false;
    if (severityFilter !== "all" && (e.severity || "info") !== severityFilter) return false;
    return true;
  });

  const eventTypes = allTypes.filter((type) => showVerbose || !VERBOSE_EVENT_TYPES.has(type));

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
            <Download className="h-3.5 w-3.5" /> {t("dash.events.export")} page
          </Button>
          <Button variant="outline" size="sm" onClick={() => void loadHistory()} disabled={loading}>
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
        <Select value={filter} aria-label="Event type" onChange={(e) => { setFilter(e.target.value); resetPage(); }}>
          <option value="all">All Types</option>
          {eventTypes.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </Select>
        <Select value={severityFilter} aria-label="Severity" onChange={(e) => { setSeverityFilter(e.target.value); resetPage(); }}>
          <option value="all">All Severity</option>
          <option value="info">Info</option>
          <option value="warning">Warning</option>
          <option value="error">Error</option>
          <option value="critical">Critical</option>
        </Select>
        <button
          type="button"
          onClick={() => { setShowVerbose((v) => !v); setFilter("all"); resetPage(); }}
          className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition-colors ${
            showVerbose
              ? "border-accent-violet/40 bg-accent-violet/10 text-accent-violet"
              : "border-border text-text-muted hover:text-text-primary hover:bg-surface-hover"
          }`}
        >
          <EyeOff className="h-3 w-3" />
          {showVerbose ? "Hide verbose" : "Show verbose"}
        </button>
        {events.length > 0 && (
          <span className="flex items-center text-xs text-text-muted">
            {total} matching events
          </span>
        )}
      </div>

      {newEvents > 0 && <button type="button" onClick={() => { resetPage(); if (pageIndex === 0) void loadHistory(); }} className="w-full rounded-xl border border-accent-cyan/30 bg-accent-cyan/5 px-4 py-3 text-sm text-accent-cyan">{newEvents} new events · Show latest</button>}
      <Card>
        <CardContent className="p-0">
          {loading ? <p role="status" className="p-12 text-center text-text-muted">Loading events…</p> : failed ? <div className="p-12 text-center"><p className="mb-3 text-text-secondary">Couldn’t load this page.</p><Button onClick={() => void loadHistory()}>Retry</Button></div> : filtered.length === 0 ? (
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
                          {event.timestamp ? new Date(typeof event.timestamp === "number" ? event.timestamp * 1000 : event.timestamp).toLocaleString() : "-"}
                        </span>
                      </div>
                      <p className="text-sm text-text-secondary truncate">
                        {event.message || (event.data
                          ? Object.entries(event.data)
                              .map(([k, v]) => `${k}: ${v}`)
                              .join(" · ")
                          : event.type?.replace(/_/g, " ") || "-")}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
      <nav aria-label="Event pagination" className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-text-muted">Page {pageIndex + 1} · {total} matching events</p>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" disabled={loading || pageIndex === 0} onClick={() => setPageIndex((page) => page - 1)}>Newer events</Button>
          <Button variant="outline" size="sm" disabled={loading || failed || !nextCursor} onClick={() => { setCursors((current) => [...current.slice(0, pageIndex + 1), nextCursor]); setPageIndex((page) => page + 1); }}>Older events</Button>
        </div>
      </nav>
    </div>
  );
}
