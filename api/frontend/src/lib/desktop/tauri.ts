"use client";

import type { DesktopRuntimeState } from "@/lib/desktop/contract";

export interface DesktopBridge {
  invoke<T>(command: string, args?: Record<string, unknown>): Promise<T>;
  listen<T>(event: string, handler: (payload: T) => void): Promise<() => void>;
}

declare global {
  interface Window {
    __TAURI__?: unknown;
    __TAURI_INTERNALS__?: unknown;
  }
}

export function isNativeDesktopEnvironment() {
  if (typeof window === "undefined") return false;
  return Boolean(window.__TAURI__ || window.__TAURI_INTERNALS__);
}

export async function loadDesktopBridge(): Promise<DesktopBridge | null> {
  if (!isNativeDesktopEnvironment()) return null;

  const [{ invoke }, { listen }] = await Promise.all([
    import("@tauri-apps/api/core"),
    import("@tauri-apps/api/event"),
  ]);

  return {
    invoke,
    async listen<T>(event: string, handler: (payload: T) => void) {
      const unlisten = await listen<T>(event, (tauriEvent) => {
        handler(tauriEvent.payload);
      });
      return () => {
        void unlisten();
      };
    },
  };
}

export function createBrowserDesktopState(): DesktopRuntimeState {
  return {
    isNativeDesktop: false,
    isStandalonePwa: false,
    canInstall: false,
    isInstalled: false,
    isOnline: typeof navigator === "undefined" ? true : navigator.onLine,
    notificationsEnabled: typeof Notification !== "undefined" && Notification.permission === "granted",
    trayConnected: false,
    autostartEnabled: false,
    updateAvailable: false,
    currentDesktopRoute: "/desktop",
    lastRemoteRoute: "/dashboard",
    unreadCount: 0,
    criticalAlertCount: 0,
    hideToTray: true,
    defaultDesktopRoute: "/desktop",
    updaterChannel: "stable",
    currentVersion: null,
    updateVersion: null,
    remoteOrigin: "",
    devOrigin: "",
    authRequired: false,
    pendingDeepLinks: [],
    recentNotifications: [],
  };
}

