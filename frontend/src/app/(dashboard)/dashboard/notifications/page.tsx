"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Select } from "@/components/ui/input";
import {
  Bell, CheckCheck, Search, Inbox, Server, Cpu, AlertTriangle, DollarSign, Shield, Info,
} from "lucide-react";
import {
  fetchNotifications,
  markNotificationRead,
  markAllNotificationsRead,
} from "@/lib/api";
import type { Notification } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";

const TYPE_META: Record<string, { icon: React.ReactNode; color: string }> = {
  job_submitted:        { icon: <Cpu className="h-4 w-4" />,            color: "text-blue-400" },
  job_completed:        { icon: <Cpu className="h-4 w-4" />,            color: "text-green-400" },
  job_failed:           { icon: <AlertTriangle className="h-4 w-4" />,  color: "text-red-400" },
  job_status:           { icon: <Cpu className="h-4 w-4" />,            color: "text-yellow-400" },
  host_registered:      { icon: <Server className="h-4 w-4" />,         color: "text-green-400" },
  host_removed:         { icon: <Server className="h-4 w-4" />,         color: "text-red-400" },
  preemption_scheduled: { icon: <AlertTriangle className="h-4 w-4" />,  color: "text-orange-400" },
  billing_alert:        { icon: <DollarSign className="h-4 w-4" />,     color: "text-yellow-400" },
  security_alert:       { icon: <Shield className="h-4 w-4" />,         color: "text-red-400" },
};

function getTypeMeta(type: string) {
  return TYPE_META[type] ?? { icon: <Info className="h-4 w-4" />, color: "text-text-muted" };
}

function timeAgo(epoch: number): string {
  const diff = Math.max(0, Math.floor(Date.now() / 1000 - epoch));
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function notificationHref(n: Notification): string | null {
  const data = n.data ?? {};
  if (data.job_id) return `/dashboard/instances/${data.job_id}`;
  if (data.host_id) return `/dashboard/hosts/${data.host_id}`;
  return null;
}

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "unread">("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [search, setSearch] = useState("");

  const load = useCallback(() => {
    setLoading(true);
    fetchNotifications(filter === "unread", 200)
      .then((r) => setNotifications(r.notifications))
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

      {/* Notification list */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-text-muted">
            {loading ? "Loading..." : `${filtered.length} notification${filtered.length !== 1 ? "s" : ""}`}
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
            const href = notificationHref(n);

            const inner = (
              <>
                {/* Icon */}
                <div className={`mt-0.5 flex-shrink-0 ${meta.color}`}>{meta.icon}</div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-medium ${!n.read ? "text-text-primary" : "text-text-muted"}`}>
                      {n.title}
                    </span>
                    {!n.read && <span className="h-2 w-2 rounded-full bg-accent-blue flex-shrink-0" />}
                  </div>
                  <p className="text-xs text-text-muted mt-0.5 line-clamp-2">{n.body}</p>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-[11px] text-text-muted">{timeAgo(n.created_at)}</span>
                    <span className="text-[11px] text-text-muted bg-surface-hover rounded px-1.5 py-0.5">
                      {n.type.replace(/_/g, " ")}
                    </span>
                  </div>
                </div>

                {/* Mark-read button */}
                {!n.read && (
                  <button
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      handleMarkRead(n.id);
                    }}
                    className="flex-shrink-0 text-xs text-text-muted hover:text-text-primary mt-1"
                    title="Mark as read"
                  >
                    <CheckCheck className="h-4 w-4" />
                  </button>
                )}
              </>
            );

            const cls = `flex items-start gap-3 py-3 px-2 rounded-md transition-colors ${
              !n.read ? "bg-accent-blue/5" : ""
            } ${href ? "hover:bg-surface-hover cursor-pointer" : ""}`;

            return href ? (
              <Link key={n.id} href={href} className={cls}>
                {inner}
              </Link>
            ) : (
              <div key={n.id} className={cls}>
                {inner}
              </div>
            );
          })}
        </CardContent>
      </Card>
    </div>
  );
}
