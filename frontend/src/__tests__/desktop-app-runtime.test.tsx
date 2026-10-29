import { render, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DesktopRuntimeState } from "@/lib/desktop/contract";

const appRuntimeMocks = vi.hoisted(() => ({
  pathname: "/dashboard",
  replace: vi.fn(),
  syncNativeState: vi.fn(),
  state: {
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
    unreadCount: 0,
    criticalAlertCount: 0,
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
  } satisfies DesktopRuntimeState,
}));

vi.mock("next/navigation", () => ({
  usePathname: () => appRuntimeMocks.pathname,
  useRouter: () => ({
    replace: appRuntimeMocks.replace,
  }),
}));

vi.mock("@/lib/desktop/runtime", () => ({
  useDesktopRuntime: () => ({
    state: appRuntimeMocks.state,
    syncNativeState: appRuntimeMocks.syncNativeState,
  }),
}));

import { DesktopAppRuntime } from "@/components/DesktopAppRuntime";

describe("DesktopAppRuntime", () => {
  beforeEach(() => {
    appRuntimeMocks.pathname = "/dashboard";
    appRuntimeMocks.replace.mockReset();
    appRuntimeMocks.syncNativeState.mockReset();
    appRuntimeMocks.syncNativeState.mockResolvedValue(undefined);
    appRuntimeMocks.state = {
      ...appRuntimeMocks.state,
      isNativeDesktop: true,
      lastRemoteRoute: "/dashboard",
    };
    localStorage.clear();
    Object.defineProperty(window.navigator, "onLine", {
      configurable: true,
      value: true,
    });
    window.history.replaceState({}, "", "/dashboard");
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("syncs auth and online state to the native shell", async () => {
    render(<DesktopAppRuntime />);

    await waitFor(() => {
      expect(appRuntimeMocks.syncNativeState).toHaveBeenCalledWith(
        expect.objectContaining({ isOnline: true }),
      );
    });

    Object.defineProperty(window.navigator, "onLine", {
      configurable: true,
      value: false,
    });
    window.dispatchEvent(new Event("offline"));

    await waitFor(() => {
      expect(appRuntimeMocks.syncNativeState).toHaveBeenCalledWith(
        expect.objectContaining({ isOnline: false }),
      );
    });
  });

  it("marks auth-required routes for the native shell", async () => {
    appRuntimeMocks.pathname = "/login";

    render(<DesktopAppRuntime />);

    await waitFor(() => {
      expect(appRuntimeMocks.syncNativeState).toHaveBeenCalledWith(
        expect.objectContaining({ authRequired: true }),
      );
    });
  });
});
