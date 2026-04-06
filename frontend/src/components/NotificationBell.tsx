"use client";

import { useState, useRef, useEffect } from "react";
import { Bell, Check, CheckCheck, ExternalLink, Trash2 } from "lucide-react";
import { useNotifications } from "@/hooks/useNotifications";
import { cn } from "@/lib/utils";
import Link from "next/link";

const TYPE_ICONS: Record<string, string> = {
  instance: "🖥️",
  host: "🖧",
  billing: "💳",
  security: "🔒",
};

function timeAgo(ts: number): string {
  const diff = (Date.now() / 1000) - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function notifHref(notif: { type: string; data: Record<string, unknown> }): string {
  const jobId = notif.data?.job_id as string | undefined;
  const hostId = notif.data?.host_id as string | undefined;
  if (notif.type === "instance" && jobId) return `/dashboard/instances/${jobId}`;
  if (notif.type === "host" && hostId) return `/dashboard/hosts/${hostId}`;
  return "/dashboard/notifications";
}

export function NotificationBell() {
  const { notifications, unreadCount, loading, markRead, markAllRead, deleteNotification, refresh } =
    useNotifications();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    if (open) document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  // Fetch full list when opening
  function handleToggle() {
    if (!open) refresh();
    setOpen(!open);
  }

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={handleToggle}
        className="relative text-text-muted hover:text-text-primary transition-colors p-1"
        aria-label={`Notifications${unreadCount > 0 ? ` (${unreadCount} unread)` : ""}`}
      >
        <Bell className="h-6 w-6" />
        {unreadCount > 0 && (
          <span className="absolute -top-2 -right-2 flex h-[18px] min-w-[18px] items-center justify-center rounded-full bg-accent-red px-1 text-[9px] font-bold text-white ring-2 ring-surface">
            {unreadCount > 99 ? "99+" : unreadCount}
          </span>
        )}
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-80 sm:w-96 rounded-xl border border-border bg-surface shadow-xl z-50 overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold">Notifications</h3>
            {unreadCount > 0 && (
              <button
                onClick={() => markAllRead()}
                className="flex items-center gap-1 text-xs text-ice-blue hover:underline"
              >
                <CheckCheck className="h-3 w-3" /> Mark all read
              </button>
            )}
          </div>

          {/* List */}
          <div className="max-h-[24rem] overflow-y-auto divide-y divide-border">
            {loading && notifications.length === 0 ? (
              <div className="py-8 text-center text-sm text-text-muted">Loading…</div>
            ) : notifications.length === 0 ? (
              <div className="py-8 text-center">
                <Bell className="mx-auto h-8 w-8 text-text-muted mb-2 opacity-40" />
                <p className="text-sm text-text-muted">No notifications yet</p>
              </div>
            ) : (
              notifications.map((n) => {
                const href = notifHref(n);
                const wrapperProps = { href, onClick: () => { markRead(n.id); setOpen(false); } };
                return (
                  <Link
                    key={n.id}
                    {...wrapperProps}
                    className={cn(
                      "flex items-start gap-3 px-4 py-3 transition-colors cursor-pointer hover:bg-surface-hover",
                      !n.read && "bg-ice-blue/5",
                    )}
                  >
                    <span className="mt-0.5 text-base">{TYPE_ICONS[n.type] || "📋"}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <p className={cn("text-sm truncate", !n.read && "font-semibold")}>
                          {n.title}
                        </p>
                        {!n.read && (
                          <span className="h-2 w-2 rounded-full bg-ice-blue shrink-0" />
                        )}
                      </div>
                      {n.body && (
                        <p className="text-xs text-text-muted truncate mt-0.5">{n.body}</p>
                      )}
                      <p className="text-[11px] text-text-muted mt-1">{timeAgo(n.created_at)}</p>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      {!n.read && (
                        <button
                          onClick={(e) => { e.preventDefault(); e.stopPropagation(); markRead(n.id); }}
                          className="p-1 rounded hover:bg-surface text-text-muted hover:text-text-primary"
                          title="Mark as read"
                        >
                          <Check className="h-3.5 w-3.5" />
                        </button>
                      )}
                      <button
                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); deleteNotification(n.id); }}
                        className="p-1 rounded hover:bg-surface text-text-muted hover:text-red-400"
                        title="Delete notification"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </Link>
                );
              })
            )}
          </div>

          {/* Footer */}
          <div className="border-t border-border px-4 py-2 text-center">
            <Link
              href="/dashboard/notifications"
              onClick={() => setOpen(false)}
              className="text-xs text-ice-blue hover:underline"
            >
              View all notifications
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
