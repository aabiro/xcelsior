"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Select } from "@/components/ui/input";
import {
  Bell, CheckCheck, Search, Inbox, Server, Cpu, AlertTriangle, DollarSign, Shield, Info,
  Trash2, ArrowLeft, ExternalLink, X, Check,
} from "lucide-react";
import {
  fetchNotifications,
  markNotificationRead,
  markAllNotificationsRead,
  deleteNotification,
} from "@/lib/api";
import type { Notification } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

const TYPE_META: Record<string, { icon: React.ReactNode; color: string; label: string }> = {
  job_submitted:        { icon: <Cpu className="h-4 w-4" />,            color: "text-blue-400",   label: "Job Submitted" },
  job_completed:        { icon: <Cpu className="h-4 w-4" />,            color: "text-green-400",  label: "Job Completed" },
  job_failed:           { icon: <AlertTriangle className="h-4 w-4" />,  color: "text-red-400",    label: "Job Failed" },
  job_status:           { icon: <Cpu className="h-4 w-4" />,            color: "text-yellow-400", label: "Job Status" },
  host_registered:      { icon: <Server className="h-4 w-4" />,         color: "text-green-400",  label: "Host Registered" },
  host_removed:         { icon: <Server className="h-4 w-4" />,         color: "text-red-400",    label: "Host Removed" },
  preemption_scheduled: { icon: <AlertTriangle className="h-4 w-4" />,  color: "text-orange-400", label: "Preemption Scheduled" },
  billing_alert:        { icon: <DollarSign className="h-4 w-4" />,     color: "text-yellow-400", label: "Billing Alert" },
  security_alert:       { icon: <Shield className="h-4 w-4" />,         color: "text-red-400",    label: "Security Alert" },
};

function getTypeMeta(type: string) {
  return TYPE_META[type] ?? { icon: <Info className="h-4 w-4" />, color: "text-text-muted", label: type.replace(/_/g, " ") };
}

function timeAgo(epoch: number): string {
  const diff = Math.max(0, Math.floor(Date.now() / 1000 - epoch));
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function formatTimestamp(epoch: number): string {
  return new Date(epoch * 1000).toLocaleString();
}

function notificationHref(n: Notification): string | null {
  const data = n.data ?? {};
  if (data.job_id) return `/dashboard/instances/${data.job_id}`;
  if (data.host_id) return `/dashboard/hosts/${data.host_id}`;
  return null;
}

function routeForType(type: string): string {
  if (type.startsWith("job") || type === "preemption_scheduled") return "/dashboard/instances";
  if (type.startsWith("host")) return "/dashboard/hosts";
  if (type.startsWith("billing")) return "/dashboard/billing";
  if (type.startsWith("security")) return "/dashboard/settings";
  return "/dashboard/notifications";
}

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "unread">("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    fetchNotifications(filter === "unread", 200)
      .then((r) => setNotifications(Array.isArray(r.notifications) ? r.notifications : []))
      .catch((e) => console.error("Failed to load notifications", e))
      .finally(() => setLoading(false));
  }, [filter]);

  useEffect(() => { load(); }, [load]);

  // Live updates
  useEventStream({
    eventTypes: [
      "job_submitted", "job_status", "job_completed", "job_failed",
      "host_registered", "host_removed", "preemption_scheduled",
    ],
    onEvent: () => { load(); },
  });

  async function handleMarkRead(id: string) {
    await markNotificationRead(id);
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: 1 } : n)),
    );
  }

  async function handleMarkAllRead() {
    await markAllNotificationsRead();
    setNotifications((prev) => prev.map((n) => ({ ...n, read: 1 })));
  }

  async function handleDelete(id: string) {
    try {
      await deleteNotification(id);
      setNotifications((prev) => prev.filter((n) => n.id !== id));
      if (selectedId === id) setSelectedId(null);
      toast.success("Notification deleted");
    } catch {
      toast.error("Failed to delete notification");
    }
  }

  // Collect unique types for the filter dropdown
  const types = Array.from(new Set(notifications.map((n) => n.type)));

  const filtered = notifications.filter((n) => {
    if (typeFilter !== "all" && n.type !== typeFilter) return false;
    if (search) {
      const q = search.toLowerCase();
      return n.title.toLowerCase().includes(q) || n.body.toLowerCase().includes(q);
    }
    return true;
  });

  const unreadCount = notifications.filter((n) => !n.read).length;
  const selected = selectedId ? notifications.find((n) => n.id === selectedId) ?? null : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary flex items-center gap-2">
            <Bell className="h-6 w-6" />
            Notifications
            {unreadCount > 0 && (
              <span className="ml-2 rounded-full bg-accent-red px-2.5 py-0.5 text-xs font-semibold text-white">
                {unreadCount} unread
              </span>
            )}
          </h1>
          <p className="text-sm text-text-muted mt-1">
            Stay informed about your instances, hosts, and platform activity.
          </p>
        </div>
        <Button variant="secondary" size="sm" onClick={handleMarkAllRead} disabled={unreadCount === 0}>
          <CheckCheck className="h-4 w-4 mr-1" />
          Mark all read
        </Button>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="py-3 px-4 flex flex-wrap items-center gap-3">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-text-muted pointer-events-none" />
            <Input
              placeholder="Search notifications..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-10"
            />
          </div>
          <Select value={filter} onChange={(e) => setFilter(e.target.value as "all" | "unread")}>
            <option value="all">All</option>
            <option value="unread">Unread only</option>
          </Select>
          <Select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
            <option value="all">All types</option>
            {types.map((t) => (
              <option key={t} value={t}>
                {t.replace(/_/g, " ")}
              </option>
            ))}
          </Select>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Notification list */}
        <Card className={cn(selected ? "lg:col-span-2" : "lg:col-span-3")}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-text-muted">
              {loading ? "Loading..." : `${filtered.length} Notification${filtered.length !== 1 ? "s" : ""}`}
            </CardTitle>
          </CardHeader>
          <CardContent className="divide-y divide-border">
            {!loading && filtered.length === 0 && (
              <div className="flex flex-col items-center justify-center py-16 text-text-muted">
                <Inbox className="h-10 w-10 mb-3 opacity-40" />
                <p className="text-sm">No notifications to show.</p>
              </div>
            )}
            {filtered.map((n) => {
              const meta = getTypeMeta(n.type);
              const isSelected = selectedId === n.id;

              return (
                <div
                  key={n.id}
                  onClick={() => { handleMarkRead(n.id); setSelectedId(n.id); }}
                  className={cn(
                    "flex items-start gap-3 py-3 px-2 rounded-md transition-colors hover:bg-surface-hover cursor-pointer",
                    !n.read && "bg-accent-blue/5",
                    isSelected && "ring-1 ring-accent-cyan/40 bg-accent-cyan/5",
                  )}
                >
                  {/* Icon */}
                  <div className={`mt-0.5 flex-shrink-0 ${meta.color}`}>{meta.icon}</div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={cn("text-sm font-medium truncate", !n.read ? "text-text-primary" : "text-text-muted")}>
                        {n.title}
                      </span>
                      {!n.read && <span className="h-2 w-2 rounded-full bg-accent-blue flex-shrink-0" />}
                    </div>
                    <p className="text-xs text-text-muted mt-0.5 line-clamp-1">{n.body}</p>
                    <div className="flex items-center gap-3 mt-1">
                      <span className="text-[11px] text-text-muted">{timeAgo(n.created_at)}</span>
                      <span className="text-[11px] text-text-muted bg-surface-hover rounded px-1.5 py-0.5">
                        {n.type.replace(/_/g, " ")}
                      </span>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-1 flex-shrink-0 mt-0.5">
                    {!n.read && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleMarkRead(n.id); }}
                        className="p-1 rounded hover:bg-surface text-text-muted hover:text-text-primary"
                        title="Mark as read"
                      >
                        <Check className="h-3.5 w-3.5" />
                      </button>
                    )}
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDelete(n.id); }}
                      className="p-1 rounded hover:bg-surface text-text-muted hover:text-red-400"
                      title="Delete notification"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>

        {/* Detail panel */}
        {selected && (
          <Card className="lg:col-span-1 self-start sticky top-6">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-semibold truncate text-text-muted">Details</CardTitle>
                <button
                  onClick={() => setSelectedId(null)}
                  className="p-1 rounded hover:bg-surface-hover text-text-muted hover:text-text-primary"
                  title="Close"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Type badge + timestamp */}
              <div className="flex items-center gap-2 flex-wrap">
                <span className={cn("flex items-center gap-1.5 text-xs font-medium rounded-full px-2.5 py-1 bg-surface-hover", getTypeMeta(selected.type).color)}>
                  {getTypeMeta(selected.type).icon}
                  {getTypeMeta(selected.type).label}
                </span>
                {!selected.read && (
                  <span className="text-xs font-medium text-accent-blue bg-accent-blue/10 rounded-full px-2 py-0.5">Unread</span>
                )}
              </div>

              {/* Title */}
              <h3 className="text-base font-semibold text-text-primary leading-snug">{selected.title}</h3>

              {/* Full body */}
              <div className="text-sm text-text-secondary leading-relaxed whitespace-pre-wrap break-words">
                {selected.body || <span className="text-text-muted italic">No additional details.</span>}
              </div>

              {/* Metadata */}
              <div className="space-y-1.5 text-xs text-text-muted border-t border-border pt-3">
                <div className="flex justify-between">
                  <span>Time</span>
                  <span>{formatTimestamp(selected.created_at)}</span>
                </div>
                {selected.data && Object.keys(selected.data).length > 0 && (
                  <>
                    {Object.entries(selected.data).map(([k, v]) => (
                      <div key={k} className="flex justify-between gap-4">
                        <span className="shrink-0">{k.replace(/_/g, " ")}</span>
                        <span className="truncate text-right font-mono text-[11px]">{String(v)}</span>
                      </div>
                    ))}
                  </>
                )}
              </div>

              {/* Actions */}
              <div className="flex flex-col gap-2 border-t border-border pt-3">
                {notificationHref(selected) && (
                  <Link
                    href={notificationHref(selected)!}
                    className="flex items-center justify-center gap-1.5 rounded-lg bg-accent-cyan/10 hover:bg-accent-cyan/20 text-accent-cyan text-sm font-medium py-2 transition-colors"
                  >
                    <ExternalLink className="h-3.5 w-3.5" />
                    View Related Resource
                  </Link>
                )}
                {!selected.read && (
                  <button
                    onClick={() => handleMarkRead(selected.id)}
                    className="flex items-center justify-center gap-1.5 rounded-lg bg-surface-hover hover:bg-surface text-text-secondary text-sm font-medium py-2 transition-colors"
                  >
                    <CheckCheck className="h-3.5 w-3.5" />
                    Mark as Read
                  </button>
                )}
                <button
                  onClick={() => handleDelete(selected.id)}
                  className="flex items-center justify-center gap-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 text-sm font-medium py-2 transition-colors"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Delete Notification
                </button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
