import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import App from "../App";
import * as native from "../lib/native";

describe("Control Center shell", () => {
  it("renders the control center chrome", async () => {
    vi.spyOn(native, "getDesktopState").mockResolvedValue({
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
    });
    vi.spyOn(native, "listenForDesktopState").mockResolvedValue(() => {});

    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>,
    );

    expect(await screen.findByText("Control Center")).toBeInTheDocument();
    expect(screen.getByText("Open Shared App")).toBeInTheDocument();
  });
});
