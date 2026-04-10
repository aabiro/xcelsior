"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, createInstanceStreamTicket, type Instance } from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────

interface InstanceWsEvent {
  event: string;
  data: Record<string, unknown>;
}

export interface UseInstanceWebSocketOptions {
  /** Called when a full instance snapshot arrives (connect + status change). */
  onInstance?: (instance: Instance) => void;
  /** Called for each log line. */
  onLog?: (log: { job_id: string; timestamp: number; line: string; level: string }) => void;
  /** Called when the job status changes. */
  onStatusChange?: (jobId: string, status: string) => void;
  /** Called when a job error event arrives (e.g. no hosts, image pull failure). */
  onJobError?: (error: { job_id: string; error: string; message: string }) => void;
  /** Disable the connection (e.g. when the job is terminal). */
  enabled?: boolean;
}

export interface InstanceWebSocketState {
  connected: boolean;
  reconnecting: boolean;
  error: string | null;
}

// ── Hook ──────────────────────────────────────────────────────────────

const MAX_RETRIES = 20;
const BASE_DELAY_MS = 1000;
const MAX_DELAY_MS = 30_000;
const NON_RETRYABLE_WS_CLOSE_CODES = new Set([1000, 1008, 4001, 4003, 4004, 4429]);

export function useInstanceWebSocket(
  jobId: string | undefined,
  options: UseInstanceWebSocketOptions = {},
): InstanceWebSocketState {
  const { enabled = true } = options;

  // Store callbacks in a ref so reconnection doesn't trigger on
  // every render when callers pass inline arrow functions.
  const cbRef = useRef(options);
  cbRef.current = options;

  const [state, setState] = useState<InstanceWebSocketState>({
    connected: false,
    reconnecting: false,
    error: null,
  });

  useEffect(() => {
    if (!enabled || !jobId) return;
    const streamJobId = jobId;

    let ws: WebSocket | null = null;
    let retries = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let unmounted = false;

    function getUrl(ticket: string): string {
      // Allow explicit override for local dev (e.g. NEXT_PUBLIC_WS_URL=ws://localhost:9500)
      const override = process.env.NEXT_PUBLIC_WS_URL;
      if (override) {
        return `${override.replace(/\/$/, "")}/ws/instances/${encodeURIComponent(streamJobId)}?ticket=${encodeURIComponent(ticket)}`;
      }
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      return `${proto}//${window.location.host}/ws/instances/${encodeURIComponent(streamJobId)}?ticket=${encodeURIComponent(ticket)}`;
    }

    function scheduleReconnect(message?: string) {
      if (retries < MAX_RETRIES) {
        const delay = Math.min(BASE_DELAY_MS * 2 ** retries, MAX_DELAY_MS);
        retries++;
        setState((s) => ({
          ...s,
          connected: false,
          reconnecting: true,
          error: message ?? s.error,
        }));
        timer = setTimeout(() => {
          void connect();
        }, delay);
      } else {
        setState({
          connected: false,
          reconnecting: false,
          error: message || "Max reconnection attempts reached",
        });
      }
    }

    async function connect() {
      if (unmounted) return;
      let ticketResp: { ok: boolean; ticket: string; expires_in: number };
      try {
        ticketResp = await createInstanceStreamTicket(streamJobId);
      } catch (err) {
        if (unmounted) return;
        const message = err instanceof Error ? err.message : "Failed to open instance stream";
        if (err instanceof ApiError && [401, 403, 404].includes(err.status)) {
          setState({ connected: false, reconnecting: false, error: message });
          return;
        }
        scheduleReconnect(message);
        return;
      }
      if (unmounted) return;

      ws = new WebSocket(getUrl(ticketResp.ticket));

      ws.onopen = () => {
        if (unmounted) return;
        retries = 0;
        setState({ connected: true, reconnecting: false, error: null });
      };

      ws.onmessage = (e) => {
        if (unmounted) return;
        try {
          const msg: InstanceWsEvent = JSON.parse(e.data);
          switch (msg.event) {
            case "instance":
              cbRef.current.onInstance?.(msg.data as unknown as Instance);
              break;
            case "job_log":
              cbRef.current.onLog?.(
                msg.data as unknown as { job_id: string; timestamp: number; line: string; level: string },
              );
              break;
            case "job_status":
              cbRef.current.onStatusChange?.(
                (msg.data as { job_id: string }).job_id,
                (msg.data as { status: string }).status,
              );
              break;
            case "job_error":
              cbRef.current.onJobError?.(
                msg.data as unknown as { job_id: string; error: string; message: string },
              );
              break;
            case "ping":
              ws?.send(JSON.stringify({ event: "pong" }));
              break;
            case "error":
              setState((s) => ({ ...s, error: String((msg.data as { message?: string }).message) }));
              break;
          }
        } catch {
          // malformed message — ignore
        }
      };

      ws.onclose = (e) => {
        if (unmounted) return;
        setState((s) => ({ ...s, connected: false }));
        ws = null;

        if (NON_RETRYABLE_WS_CLOSE_CODES.has(e.code)) {
          setState((s) => ({
            ...s,
            reconnecting: false,
            error: e.code === 1000 ? s.error : e.reason || s.error,
          }));
          return;
        }
        scheduleReconnect();
      };

      ws.onerror = () => {
        // onerror is always followed by onclose — just flag the error
        if (!unmounted) {
          setState((s) => ({ ...s, error: "WebSocket error" }));
        }
      };
    }

    void connect();

    return () => {
      unmounted = true;
      if (timer) clearTimeout(timer);
      if (ws) {
        ws.onclose = null; // prevent reconnect on intentional close
        ws.close(1000);
      }
    };
  }, [jobId, enabled]);

  return state;
}
