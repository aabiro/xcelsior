"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import {
  fetchNotifications,
  fetchUnreadCount,
  markNotificationRead,
  markAllNotificationsRead,
} from "@/lib/api";
import type { Notification } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";

export interface UseNotificationsReturn {
  notifications: Notification[];
  unreadCount: number;
  loading: boolean;
  markRead: (id: string) => Promise<void>;
  markAllRead: () => Promise<void>;
  refresh: () => void;
}

/**
 * Hook that manages the user's notification state.
 * Polls unread count via SSE events and fetches the list on demand.
 */
export function useNotifications(): UseNotificationsReturn {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const hasFetched = useRef(false);

  const refreshCount = useCallback(() => {
    fetchUnreadCount()
      .then((r) => setUnreadCount(r.unread_count))
      .catch(() => {});
  }, []);

  const refresh = useCallback(() => {
    setLoading(true);
    fetchNotifications(false, 50)
      .then((r) => {
        setNotifications(r.notifications);
        setUnreadCount(r.unread_count);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  // Initial load — just the count (lightweight)
  useEffect(() => {
    if (!hasFetched.current) {
      hasFetched.current = true;
      refreshCount();
    }
  }, [refreshCount]);

  // Live updates — bump unread count when relevant events arrive
  useEventStream({
    eventTypes: [
      "job_submitted", "job_status", "job_completed", "job_failed",
      "host_registered", "host_removed", "preemption_scheduled",
    ],
    onEvent: () => {
      refreshCount();
    },
  });

  const markRead = useCallback(async (id: string) => {
    await markNotificationRead(id);
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: 1 } : n)),
    );
    setUnreadCount((c) => Math.max(0, c - 1));
  }, []);

  const markAllRead = useCallback(async () => {
    await markAllNotificationsRead();
    setNotifications((prev) => prev.map((n) => ({ ...n, read: 1 })));
    setUnreadCount(0);
  }, []);

  return { notifications, unreadCount, loading, markRead, markAllRead, refresh };
}
