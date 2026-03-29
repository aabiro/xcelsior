"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { createEventSource } from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────

export type ConnectionStatus = "disconnected" | "connecting" | "connected" | "reconnecting";

export interface UseEventStreamOptions {
  /** Event types to listen for (e.g. ["job_status", "job_submitted"]). Empty = all. */
  eventTypes?: string[];
  /** Called for each matching event. */
  onEvent?: (eventType: string, data: Record<string, unknown>) => void;
  /** Disable connection. */
  enabled?: boolean;
}

const MAX_RECONNECT_DELAY = 30_000;
const BASE_DELAY = 1_000;

/**
 * Subscribe to the platform-wide `/api/stream` SSE bus.
 *
 * Returns connection status and a manual `refresh` noop (for symmetry with WS hook).
 * Multiple components mounting this hook each open their own EventSource; for
 * heavy pages consider lifting to a shared context.
 */
export function useEventStream(options: UseEventStreamOptions = {}): {
  status: ConnectionStatus;
} {
  const { enabled = true } = options;
  const cbRef = useRef(options);
  cbRef.current = options;

  const [status, setStatus] = useState<ConnectionStatus>("disconnected");

  useEffect(() => {
    if (!enabled) {
      setStatus("disconnected");
      return;
    }

    let es: EventSource | null = null;
    let retries = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let unmounted = false;

    function connect() {
      if (unmounted) return;
      setStatus(retries > 0 ? "reconnecting" : "connecting");

      es = createEventSource();

      es.onopen = () => {
        if (unmounted) return;
        retries = 0;
        setStatus("connected");
      };

      es.onmessage = (e) => {
        if (unmounted) return;
        try {
          const msg = JSON.parse(e.data);
          const eventType: string = msg.event || msg.type || "message";
          const data: Record<string, unknown> = msg.data || msg;

          const filter = cbRef.current.eventTypes;
          if (filter && filter.length > 0 && !filter.includes(eventType)) return;

          cbRef.current.onEvent?.(eventType, data);
        } catch {
          // malformed payload
        }
      };

      es.onerror = () => {
        if (unmounted) return;
        es?.close();
        es = null;

        if (retries < 20) {
          const delay = Math.min(BASE_DELAY * 2 ** retries, MAX_RECONNECT_DELAY);
          retries++;
          setStatus("reconnecting");
          timer = setTimeout(connect, delay);
        } else {
          setStatus("disconnected");
        }
      };
    }

    connect();

    return () => {
      unmounted = true;
      if (timer) clearTimeout(timer);
      if (es) {
        es.onmessage = null;
        es.onerror = null;
        es.close();
      }
    };
  }, [enabled]);

  return { status };
}
