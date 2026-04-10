import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import {
  DESKTOP_STATE_CHANGED_EVENT,
  type DesktopRuntimeState,
} from "@/lib/desktop/contract";

const runtimeMocks = vi.hoisted(() => ({
  usePwaInstallPrompt: vi.fn(),
  loadDesktopBridge: vi.fn(),
}));

vi.mock("@/hooks/usePwaInstallPrompt", () => ({
  usePwaInstallPrompt: runtimeMocks.usePwaInstallPrompt,
}));

vi.mock("@/lib/desktop/tauri", async () => {
  const actual = await vi.importActual<typeof import("@/lib/desktop/tauri")>("@/lib/desktop/tauri");
  return {
    ...actual,
    loadDesktopBridge: runtimeMocks.loadDesktopBridge,
  };
});

import { DesktopRuntimeProvider, useDesktopRuntime } from "@/lib/desktop/runtime";

function createNativeDesktopState(overrides: Partial<DesktopRuntimeState> = {}): DesktopRuntimeState {
  return {
    isNativeDesktop: true,
    isStandalonePwa: false,
    canInstall: false,
    isInstalled: true,
    isOnline: true,
    notificationsEnabled: true,
    trayConnected: true,
    autostartEnabled: true,
    updateAvailable: false,
    currentDesktopRoute: "/desktop",
    lastRemoteRoute: "/dashboard",
    unreadCount: 2,
    criticalAlertCount: 1,
    hideToTray: true,
    defaultDesktopRoute: "/desktop",
    updaterChannel: "stable",
    currentVersion: "0.1.0",
    updateVersion: null,
    remoteOrigin: "https://xcelsior.ca",
    devOrigin: "http://localhost:3000",
    authRequired: false,
    pendingDeepLinks: [],
    recentNotifications: [],
    ...overrides,
  };
}

function wrapper({ children }: { children: ReactNode }) {
  return <DesktopRuntimeProvider>{children}</DesktopRuntimeProvider>;
}

describe("DesktopRuntimeProvider", () => {
  const originalNotification = globalThis.Notification;

  beforeEach(() => {
    runtimeMocks.usePwaInstallPrompt.mockReset();
    runtimeMocks.loadDesktopBridge.mockReset();
    Object.defineProperty(window.navigator, "onLine", {
      configurable: true,
      value: true,
    });
    Object.defineProperty(globalThis, "Notification", {
      configurable: true,
      value: { permission: "granted" },
    });
  });

  afterEach(() => {
    Object.defineProperty(globalThis, "Notification", {
      configurable: true,
      value: originalNotification,
    });
    document.documentElement.dataset.nativeDesktop = "";
    document.documentElement.dataset.standalonePwa = "";
    document.documentElement.dataset.desktopMode = "";
    document.documentElement.dataset.desktopOnline = "";
    document.documentElement.dataset.desktopNotifications = "";
  });

  it("maps installed browser state into standalone desktop mode and tracks online/offline state", async () => {
    runtimeMocks.usePwaInstallPrompt.mockReturnValue({
      canInstall: false,
      isInstalled: true,
      isDesktopDevice: true,
      promptToInstall: vi.fn(),
    });
    runtimeMocks.loadDesktopBridge.mockResolvedValue(null);

    const { result } = renderHook(() => useDesktopRuntime(), { wrapper });

    await waitFor(() => {
      expect(result.current.state.isStandalonePwa).toBe(true);
    });

    expect(result.current.state.isNativeDesktop).toBe(false);
    expect(result.current.state.isInstalled).toBe(true);
    expect(document.documentElement.dataset.desktopMode).toBe("1");
    expect(document.documentElement.dataset.standalonePwa).toBe("1");

    await act(async () => {
      Object.defineProperty(window.navigator, "onLine", {
        configurable: true,
        value: false,
      });
      window.dispatchEvent(new Event("offline"));
    });

    expect(result.current.state.isOnline).toBe(false);
    expect(document.documentElement.dataset.desktopOnline).toBe("0");
  });

  it("hydrates native desktop state from the Tauri bridge and reacts to runtime events", async () => {
    const nativeState = createNativeDesktopState();
    const eventUnlisten = vi.fn();
    let stateListener: ((payload: DesktopRuntimeState) => void) | undefined;
    const invoke = vi.fn(async (command: string) => {
      if (command === "desktop_get_state") {
        return nativeState;
      }
      return nativeState;
    });

    runtimeMocks.usePwaInstallPrompt.mockReturnValue({
      canInstall: false,
      isInstalled: false,
      isDesktopDevice: true,
      promptToInstall: vi.fn(),
    });
    runtimeMocks.loadDesktopBridge.mockResolvedValue({
      invoke,
      listen: vi.fn(async (event: string, handler: (payload: DesktopRuntimeState) => void) => {
        expect(event).toBe(DESKTOP_STATE_CHANGED_EVENT);
        stateListener = handler;
        return eventUnlisten;
      }),
    });

    const { result, unmount } = renderHook(() => useDesktopRuntime(), { wrapper });

    await waitFor(() => {
      expect(result.current.state.isNativeDesktop).toBe(true);
    });

    expect(invoke).toHaveBeenCalledWith("desktop_get_state");
    expect(document.documentElement.dataset.nativeDesktop).toBe("1");
    expect(document.documentElement.dataset.desktopNotifications).toBe("1");

    await act(async () => {
      stateListener?.(
        createNativeDesktopState({
          updateAvailable: true,
          updateVersion: "0.2.0",
          notificationsEnabled: false,
          currentDesktopRoute: "/desktop/settings",
        }),
      );
    });

    expect(result.current.state.updateAvailable).toBe(true);
    expect(result.current.state.updateVersion).toBe("0.2.0");
    expect(result.current.state.notificationsEnabled).toBe(false);
    expect(result.current.state.currentDesktopRoute).toBe("/desktop/settings");
    expect(document.documentElement.dataset.desktopNotifications).toBe("0");

    await act(async () => {
      await result.current.openControlCenter("/desktop/activity");
      await result.current.checkForUpdates();
    });

    expect(invoke).toHaveBeenCalledWith("desktop_show_control_center", {
      route: "/desktop/activity",
    });
    expect(invoke).toHaveBeenCalledWith("desktop_check_for_updates");

    unmount();
    expect(eventUnlisten).toHaveBeenCalledTimes(1);
  });
});
