import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const tauriMocks = vi.hoisted(() => ({
  invoke: vi.fn(),
  listen: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: tauriMocks.invoke,
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: tauriMocks.listen,
}));

import { createBrowserDesktopState, isNativeDesktopEnvironment, loadDesktopBridge } from "@/lib/desktop/tauri";

describe("desktop tauri bridge", () => {
  const originalNotification = globalThis.Notification;

  beforeEach(() => {
    tauriMocks.invoke.mockReset();
    tauriMocks.listen.mockReset();
    delete window.__TAURI__;
    delete window.__TAURI_INTERNALS__;
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
    delete window.__TAURI__;
    delete window.__TAURI_INTERNALS__;
  });

  it("detects the native desktop environment from Tauri globals", () => {
    expect(isNativeDesktopEnvironment()).toBe(false);

    window.__TAURI__ = {};
    expect(isNativeDesktopEnvironment()).toBe(true);

    delete window.__TAURI__;
    window.__TAURI_INTERNALS__ = {};
    expect(isNativeDesktopEnvironment()).toBe(true);
  });

  it("builds the browser fallback desktop state from current browser capabilities", () => {
    Object.defineProperty(window.navigator, "onLine", {
      configurable: true,
      value: false,
    });

    const state = createBrowserDesktopState();

    expect(state.isNativeDesktop).toBe(false);
    expect(state.isInstalled).toBe(false);
    expect(state.isOnline).toBe(false);
    expect(state.notificationsEnabled).toBe(true);
    expect(state.currentDesktopRoute).toBe("/desktop");
    expect(state.lastRemoteRoute).toBe("/dashboard");
  });

  it("wraps Tauri invoke and listen helpers behind a stable desktop bridge", async () => {
    window.__TAURI__ = {};

    const unlisten = vi.fn();
    let eventHandler: ((event: { payload: { route: string } }) => void) | undefined;

    tauriMocks.listen.mockImplementation(async (_event, handler) => {
      eventHandler = handler;
      return unlisten;
    });

    const bridge = await loadDesktopBridge();

    expect(bridge).not.toBeNull();
    await bridge?.invoke("desktop_get_state", { force: true });
    expect(tauriMocks.invoke).toHaveBeenCalledWith("desktop_get_state", { force: true });

    const payloadHandler = vi.fn();
    const dispose = await bridge?.listen("xcelsior://state-changed", payloadHandler);

    eventHandler?.({ payload: { route: "/desktop/settings" } });
    expect(payloadHandler).toHaveBeenCalledWith({ route: "/desktop/settings" });

    dispose?.();
    expect(unlisten).toHaveBeenCalledTimes(1);
  });

  it("returns null for the bridge when the browser is not running inside Tauri", async () => {
    await expect(loadDesktopBridge()).resolves.toBeNull();
    expect(tauriMocks.invoke).not.toHaveBeenCalled();
    expect(tauriMocks.listen).not.toHaveBeenCalled();
  });
});
