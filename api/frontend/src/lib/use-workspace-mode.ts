"use client";

import { useCallback, useEffect, useSyncExternalStore } from "react";
import { usePathname } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { hostingEnabled, routeWorkspaceMode, type WorkspaceMode } from "@/lib/hosting";
import { WORKSPACE_MODE_KEY_PREFIX } from "@/lib/sidebar-state";

const CHANGE_EVENT = "xcelsior-workspace-mode";
function subscribe(onChange: () => void) {
  window.addEventListener("storage", onChange);
  window.addEventListener(CHANGE_EVENT, onChange);
  return () => {
    window.removeEventListener("storage", onChange);
    window.removeEventListener(CHANGE_EVENT, onChange);
  };
}
function read(key: string): WorkspaceMode {
  try { return localStorage.getItem(key) === "host" ? "host" : "rent"; }
  catch { return "rent"; }
}
function write(key: string, mode: WorkspaceMode) {
  try { localStorage.setItem(key, mode); } catch { /* The links still work without storage. */ }
  window.dispatchEvent(new Event(CHANGE_EVENT));
}
const serverSnapshot = (): WorkspaceMode => "rent";

/** Route changes select a workspace; shared pages preserve the account's choice. */
export function useWorkspaceMode() {
  const { user } = useAuth();
  const pathname = usePathname();
  const key = `${WORKSPACE_MODE_KEY_PREFIX}${user?.user_id || "signed-out"}`;
  const getSnapshot = useCallback(() => read(key), [key]);
  const stored = useSyncExternalStore(subscribe, getSnapshot, serverSnapshot);
  const enabled = hostingEnabled(user);
  const route = routeWorkspaceMode(pathname);
  const mode = enabled ? route || stored : "rent";
  useEffect(() => {
    if (enabled && route && read(key) !== route) write(key, route);
  }, [enabled, route, key]);
  const selectMode = useCallback((next: WorkspaceMode) => write(key, next), [key]);
  return { mode, enabled, selectMode };
}
