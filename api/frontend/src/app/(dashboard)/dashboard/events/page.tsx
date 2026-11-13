"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input, Select } from "@/components/ui/input";
import { Calendar, RefreshCw, Radio, Download, Wifi, WifiOff, EyeOff } from "lucide-react";
import { createEventSource, isEventStreamAvailable, apiFetch } from "@/lib/api";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";

interface Event {
  id?: string;
  type: string;
  severity?: "info" | "warning" | "error" | "critical";
  data?: Record<string, unknown>;
  entity_type?: string;
  entity_id?: string;
  actor?: string;
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
const VERBOSE_EVENT_TYPES = new Set(["job_log", "spot_prices", "host_update", "job.lease.renewed", "host.updated", "security.version_check"]);

function relatedUrl(event: Event): string | null {
  if (!event.entity_id) return null;
  const id = encodeURIComponent(event.entity_id);
  if (event.entity_type === "job" || event.entity_type === "instance") return `/dashboard/instances/${id}`;
  if (event.entity_type === "host") return `/dashboard/hosts/${id}`;
  if (event.entity_type === "billing" || event.entity_type === "wallet") return "/dashboard/billing";
  if (event.entity_type === "volume") return "/dashboard/volumes";
  return null;
}

function eventMessage(event: Event): string {
  if (event.message) return event.message;
  const data = event.data ?? {};
  for (const key of ["message", "reason", "error", "detail", "description", "failure_reason"]) {
    if (typeof data[key] === "string" && data[key]) return data[key];
  }
  const transition = [data.from_state ?? data.old_state, data.to_state ?? data.new_state ?? data.status].filter(Boolean);
  if (transition.length) return transition.join(" → ");
  return event.type.replace(/[_.]/g, " ");
}

const MAX_RECONNECT_DELAY = 30000;

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "reconnecting";

export default function EventsPage() {
  const { t } = useLocale();
  const [events, setEvents] = useState<Event[]>([]);
  const [filter, setFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const [severityCounts, setSeverityCounts] = useState<Record<string, number>>({});
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
  const invalidateRequest = useCallback(() => { latest.current++; }, []);
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
    if (search.trim()) query.set("query", search.trim());
    try {
      const data = await apiFetch<{ events: Event[]; total: number; next_cursor: string | null; event_types: string[]; severity_counts?: Record<string, number> }>(`/api/events?${query}`);
      if (request !== latest.current) return;
      setEvents((data.events || []).map((event) => ({ ...event, id: event.event_id ?? event.id, type: event.event_type ?? event.type })));
      setTotal(data.total);
      setNextCursor(data.next_cursor);
      setAllTypes(data.event_types || []);
      setSeverityCounts(data.severity_counts || {});
      setNewEvents(0);
    } catch {
      if (request === latest.current) { setFailed(true); toast.error("Failed to load events"); }
    } finally {
      if (request === latest.current) setLoading(false);
    }
  }, [before, filter, severityFilter, showVerbose, search]);

  useEffect(() => { void loadHistory(); return invalidateRequest; }, [loadHistory, invalidateRequest]);

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
        const receive = (message: MessageEvent) => {
          if (!active) return;
          try {
            const event = JSON.parse(message.data);
            const type = event.type || event.event_type || event.event || message.type;
            if (type && (showVerbose || !VERBOSE_EVENT_TYPES.has(type))) setNewEvents((count) => count + 1);
          } catch { /* Ignore stream keepalives. */ }
        };
        es.onmessage = receive;
        // The API emits named SSE events; onmessage receives only unnamed ones.
        for (const type of new Set([...allTypes, "job_status", "job_submitted", "job_cancelled", "host_update", "host_removed", "queue_processed", "billing_update", "user_registered", "team_created", "team_member_added", "team_deleted", "spot_prices"])) {
          es.addEventListener(type, receive as EventListener);
        }
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
  }, [live, showVerbose, allTypes]);

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

      <p className="text-sm text-text-muted">Browse recorded instance, account and infrastructure events. Routine lease renewals and telemetry are hidden unless you show verbose events.</p>
      <div className="flex flex-wrap gap-3" aria-label="Event severity totals">
        {Object.entries(severityCounts).map(([severity, count]) => (
          <span key={severity} className="rounded-lg border border-border px-3 py-2 text-sm"><span className="capitalize">{severity}</span> <strong className="ml-2 tabular-nums">{count}</strong></span>
        ))}
      </div>
      <form className="flex gap-2" onSubmit={(event) => { event.preventDefault(); setSearch(query); resetPage(); }}>
        <Input aria-label="Search events" placeholder="Search by instance, host, actor or message" value={query} onChange={(event) => setQuery(event.target.value)} />
        <Button type="submit" variant="outline">Search</Button>
        {search && <Button type="button" variant="ghost" onClick={() => { setQuery(""); setSearch(""); resetPage(); }}>Clear</Button>}
      </form>
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
              <p className="text-sm text-text-secondary">No recorded events match these filters. Change the filters or show verbose events.</p>
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
                      <p className="break-words text-sm text-text-secondary">{eventMessage(event)}</p>
                      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-text-muted">
                        {event.actor && <span>Actor: {event.actor}</span>}
                        {event.entity_id && (relatedUrl(event)
                          ? <Link href={relatedUrl(event)!} className="break-all text-ice-blue hover:underline">{event.entity_type}: {event.entity_id}</Link>
                          : <span className="break-all">{event.entity_type}: {event.entity_id}</span>)}
                      </div>
                      {event.data && Object.keys(event.data).length > 0 && (
                        <details className="mt-2 text-xs text-text-muted">
                          <summary className="cursor-pointer hover:text-text-primary">Event details</summary>
                          <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-background p-3">{JSON.stringify(event.data, null, 2)}</pre>
                        </details>
                      )}
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
