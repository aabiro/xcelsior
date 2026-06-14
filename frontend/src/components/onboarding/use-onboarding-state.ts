"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { OnboardingStepKey } from "./onboarding-steps";

export type OnboardingStatus = "idle" | "loading" | "ready" | "error";

type OnboardingUser = { name?: string; email?: string } | null;

const PERSIST_DEBOUNCE_MS = 500;

async function fetchJson<T>(url: string): Promise<T | null> {
  const res = await fetch(url, { credentials: "include" });
  if (!res.ok) return null;
  return res.json() as Promise<T>;
}

function persistOnboarding(onboarding: Record<string, boolean>) {
  return fetch("/api/users/me/preferences", {
    method: "PUT",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ preferences: { onboarding } }),
  });
}

export function useOnboardingState(user: OnboardingUser, pathname: string) {
  const [completed, setCompleted] = useState<Record<string, boolean>>({});
  const [status, setStatus] = useState<OnboardingStatus>("idle");
  const persistTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastPersistedRef = useRef<string>("");
  const detectGenRef = useRef(0);

  const schedulePersist = useCallback((next: Record<string, boolean>) => {
    const serialized = JSON.stringify(next);
    if (serialized === lastPersistedRef.current) return;

    if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
    persistTimerRef.current = setTimeout(() => {
      persistOnboarding(next)
        .then((res) => {
          if (res.ok) lastPersistedRef.current = serialized;
        })
        .catch(() => {});
    }, PERSIST_DEBOUNCE_MS);
  }, []);

  const detect = useCallback(async () => {
    if (!user) return;

    const gen = ++detectGenRef.current;
    setStatus((s) => (s === "ready" ? s : "loading"));

    try {
      const prefs = await fetchJson<{
        canada_only_routing?: boolean;
        preferences?: { onboarding?: Record<string, boolean> };
      }>("/api/users/me/preferences");

      if (gen !== detectGenRef.current) return;

      const serverOnboarding = prefs?.preferences?.onboarding ?? {};
      const autoDetected: Record<string, boolean> = {};

      autoDetected.profile = !!(user.name && user.name.trim().length > 0);
      autoDetected.jurisdiction = !!(
        serverOnboarding.jurisdiction || prefs?.canada_only_routing
      );
      autoDetected.browse = !!(
        serverOnboarding.browse || pathname.startsWith("/dashboard/marketplace")
      );

      if (serverOnboarding.api_key) {
        autoDetected.api_key = true;
      } else {
        const oauth = await fetchJson<{ clients?: unknown[] }>("/api/oauth/clients");
        if (gen !== detectGenRef.current) return;
        autoDetected.api_key =
          Array.isArray(oauth?.clients) && oauth.clients.length > 0;
      }

      if (serverOnboarding.instance) {
        autoDetected.instance = true;
      } else {
        const instances = await fetchJson<{ instances?: unknown[] }>("/instances");
        if (gen !== detectGenRef.current) return;
        autoDetected.instance =
          Array.isArray(instances?.instances) && instances.instances.length > 0;
      }

      setCompleted(autoDetected);
      setStatus("ready");

      const changed = Object.keys(autoDetected).some(
        (k) => autoDetected[k] !== serverOnboarding[k],
      );
      if (changed) schedulePersist(autoDetected);
    } catch {
      if (gen === detectGenRef.current) setStatus("error");
    }
  }, [user, pathname, schedulePersist]);

  const userSig = `${user?.email ?? ""}|${user?.name ?? ""}`;

  useEffect(() => {
    if (!user) {
      setCompleted({});
      setStatus("idle");
      return;
    }
    void detect();
  }, [user, userSig, detect]);

  useEffect(() => {
    if (!pathname.startsWith("/dashboard/marketplace")) return;
    setCompleted((prev) => {
      if (prev.browse) return prev;
      const next = { ...prev, browse: true };
      schedulePersist(next);
      return next;
    });
  }, [pathname, schedulePersist]);

  useEffect(() => {
    return () => {
      if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
    };
  }, []);

  const toggle = useCallback(
    (key: string) => {
      setCompleted((prev) => {
        const next = { ...prev, [key]: !prev[key] };
        schedulePersist(next);
        return next;
      });
    },
    [schedulePersist],
  );

  const retry = useCallback(() => {
    void detect();
  }, [detect]);

  return { completed, toggle, status, retry };
}