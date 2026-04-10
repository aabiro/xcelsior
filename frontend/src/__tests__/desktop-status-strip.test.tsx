import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { DesktopRuntimeState } from "@/lib/desktop/contract";

const stripMocks = vi.hoisted(() => ({
  useDesktopRuntime: vi.fn(),
  openControlCenter: vi.fn(),
  checkForUpdates: vi.fn(),
  installUpdate: vi.fn(),
}));

vi.mock("@/lib/desktop/runtime", () => ({
  useDesktopRuntime: stripMocks.useDesktopRuntime,
}));

import { DesktopStatusStrip } from "@/components/DesktopStatusStrip";

function createDesktopState(overrides: Partial<DesktopRuntimeState> = {}): DesktopRuntimeState {
  return {
    isNativeDesktop: false,
    isStandalonePwa: false,
    canInstall: false,
    isInstalled: false,
    isOnline: true,
    notificationsEnabled: true,
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
    ...overrides,
  };
}

describe("DesktopStatusStrip", () => {
  beforeEach(() => {
    stripMocks.openControlCenter.mockReset();
    stripMocks.checkForUpdates.mockReset();
    stripMocks.installUpdate.mockReset();
    stripMocks.useDesktopRuntime.mockReturnValue({
      state: createDesktopState(),
      openControlCenter: stripMocks.openControlCenter,
      checkForUpdates: stripMocks.checkForUpdates,
      installUpdate: stripMocks.installUpdate,
    });
  });

  it("does not render outside desktop mode", () => {
    const { container } = render(<DesktopStatusStrip />);
    expect(container.firstChild).toBeNull();
  });

  it("renders native desktop state and routes control actions through the runtime", () => {
    stripMocks.useDesktopRuntime.mockReturnValue({
      state: createDesktopState({
        isNativeDesktop: true,
        isInstalled: true,
        isOnline: false,
        notificationsEnabled: false,
      }),
      openControlCenter: stripMocks.openControlCenter,
      checkForUpdates: stripMocks.checkForUpdates,
      installUpdate: stripMocks.installUpdate,
    });

    render(<DesktopStatusStrip />);

    expect(screen.getByText("Native Desktop")).toBeInTheDocument();
    expect(screen.getByText("Offline")).toBeInTheDocument();
    expect(screen.getByText("Notifications Off")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /control center/i }));
    fireEvent.click(screen.getByRole("button", { name: /check updates/i }));

    expect(stripMocks.openControlCenter).toHaveBeenCalledWith("/desktop");
    expect(stripMocks.checkForUpdates).toHaveBeenCalledTimes(1);
  });

  it("shows the install CTA for native updates and reload CTA for installed PWA updates", () => {
    stripMocks.useDesktopRuntime.mockReturnValue({
      state: createDesktopState({
        isNativeDesktop: true,
        isInstalled: true,
        updateAvailable: true,
      }),
      openControlCenter: stripMocks.openControlCenter,
      checkForUpdates: stripMocks.checkForUpdates,
      installUpdate: stripMocks.installUpdate,
    });

    const { rerender } = render(<DesktopStatusStrip />);
    fireEvent.click(screen.getByRole("button", { name: /install update/i }));
    expect(stripMocks.installUpdate).toHaveBeenCalledTimes(1);

    stripMocks.useDesktopRuntime.mockReturnValue({
      state: createDesktopState({
        isStandalonePwa: true,
        isInstalled: true,
        updateAvailable: true,
      }),
      openControlCenter: stripMocks.openControlCenter,
      checkForUpdates: stripMocks.checkForUpdates,
      installUpdate: stripMocks.installUpdate,
    });

    rerender(<DesktopStatusStrip />);
    expect(screen.getByRole("button", { name: /reload update/i })).toBeInTheDocument();
  });
});
