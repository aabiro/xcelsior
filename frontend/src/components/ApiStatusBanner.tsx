"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const POLL_INTERVAL_MS = 60_000; // 60 seconds when healthy
const BACKOFF_SCHEDULE = [5_000, 10_000, 20_000, 40_000, 60_000]; // escalating retry
const FETCH_TIMEOUT_MS = 5_000;

export function ApiStatusBanner() {
  const [down, setDown] = useState(false);
  const [attempts, setAttempts] = useState(0);
  const retryCount = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);
  const abortRef = useRef<AbortController>(undefined);
  const mountedRef = useRef(true);

  const check = useCallback(async () => {
    // Abort any in-flight check before starting a new one
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch("/healthz", {
        method: "GET",
        cache: "no-store",
        signal: AbortSignal.any([
          controller.signal,
          AbortSignal.timeout(FETCH_TIMEOUT_MS),
        ]),
      });
      if (!mountedRef.current) return;
      if (res.ok) {
        const wasDown = retryCount.current > 0;
        retryCount.current = 0;
        setDown(false);
        setAttempts(0);
        if (wasDown) {
          window.dispatchEvent(new CustomEvent("xcelsior:api-recovered"));
        }
        timerRef.current = setTimeout(check, POLL_INTERVAL_MS);
        return;
      }
      throw new Error("non-ok");
    } catch (err) {
      if (!mountedRef.current) return;
      // Don't count intentional aborts as failures
      if (err instanceof DOMException && err.name === "AbortError") return;
      retryCount.current = Math.min(retryCount.current + 1, BACKOFF_SCHEDULE.length);
      setDown(true);
      setAttempts(retryCount.current);
      const delay = BACKOFF_SCHEDULE[Math.min(retryCount.current - 1, BACKOFF_SCHEDULE.length - 1)];
      timerRef.current = setTimeout(check, delay);
    }
  }, []);

  const retryNow = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    check();
  }, [check]);

  useEffect(() => {
    mountedRef.current = true;
    // Start first check after a short delay to avoid flash on fast loads
    timerRef.current = setTimeout(check, 3_000);
    return () => {
      mountedRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
      abortRef.current?.abort();
    };
  }, [check]);

  // Always render the aria-live container so screen readers can detect transitions
  return (
    <div aria-live="assertive" aria-atomic="true">
      {down && (
        <div
          role="alert"
          className="flex items-center justify-center gap-2 bg-amber-600 px-4 py-2 text-sm font-medium text-white"
        >
          <span className="relative flex h-2.5 w-2.5" aria-hidden="true">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-white opacity-75" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-white" />
          </span>
          <span>
            Xcelsior API is temporarily unavailable &mdash; reconnecting
            {attempts > 1 && <span className="opacity-75"> (attempt {attempts})</span>}
            &hellip;
          </span>
          <button
            type="button"
            onClick={retryNow}
            className="ml-2 rounded bg-white/20 px-2 py-0.5 text-xs font-semibold hover:bg-white/30 transition-colors"
          >
            Retry now
          </button>
        </div>
      )}
    </div>
  );
}
