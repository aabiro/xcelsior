"use client";

import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  DEFAULT_DESKTOP_ROUTE,
  DESKTOP_STATE_CHANGED_EVENT,
  type DesktopRoute,
  type DesktopPreferencesUpdate,
  type DesktopRemoteStateSyncPayload,
  type DesktopRuntimeState,
} from "@/lib/desktop/contract";
import { createBrowserDesktopState, loadDesktopBridge } from "@/lib/desktop/tauri";
import { usePwaInstallPrompt } from "@/hooks/usePwaInstallPrompt";

type DesktopRuntimeContextValue = {
  state: DesktopRuntimeState;
  refresh: () => Promise<void>;
  updatePreferences: (updates: DesktopPreferencesUpdate) => Promise<void>;
  syncNativeState: (payload: DesktopRemoteStateSyncPayload) => Promise<void>;
  openControlCenter: (route?: DesktopRoute) => Promise<void>;
  openMainWindow: (route?: string) => Promise<void>;
  checkForUpdates: () => Promise<void>;
  installUpdate: () => Promise<void>;
};

const DesktopRuntimeContext = createContext<DesktopRuntimeContextValue | null>(null);

function withBrowserState(
  current: DesktopRuntimeState,
  options: { canInstall: boolean; isInstalled: boolean; isDesktopDevice: boolean },
): DesktopRuntimeState {
  const next = { ...current };
  next.canInstall = !current.isNativeDesktop && options.canInstall;
  next.isStandalonePwa = !current.isNativeDesktop && options.isDesktopDevice && options.isInstalled;
  next.isInstalled = current.isNativeDesktop ? true : options.isInstalled;
  next.isOnline = typeof navigator === "undefined" ? next.isOnline : navigator.onLine;
  if (!current.isNativeDesktop) {
    next.notificationsEnabled = typeof Notification !== "undefined" && Notification.permission === "granted";
  }
  return next;
}

export function DesktopRuntimeProvider({ children }: { children: ReactNode }) {
  const { canInstall, isDesktopDevice, isInstalled } = usePwaInstallPrompt();
  const [state, setState] = useState<DesktopRuntimeState>(() =>
    withBrowserState(createBrowserDesktopState(), {
      canInstall: false,
      isInstalled: false,
      isDesktopDevice: true,
    }),
  );
  const bridgePromiseRef = useRef<Promise<Awaited<ReturnType<typeof loadDesktopBridge>>> | null>(null);

  function getBridge() {
    if (!bridgePromiseRef.current) {
      bridgePromiseRef.current = loadDesktopBridge();
    }
    return bridgePromiseRef.current;
  }

  async function refresh() {
    const bridge = await getBridge();
    if (!bridge) {
      setState((current) =>
        withBrowserState(
          {
            ...current,
            ...createBrowserDesktopState(),
          },
          { canInstall, isInstalled, isDesktopDevice },
        ),
      );
      return;
    }

    const nativeState = await bridge.invoke<DesktopRuntimeState>("desktop_get_state");
    setState(
      withBrowserState(
        {
          ...nativeState,
          isNativeDesktop: true,
          isInstalled: true,
          canInstall: false,
        },
        { canInstall: false, isInstalled: true, isDesktopDevice },
      ),
    );
  }

  async function updatePreferences(updates: DesktopPreferencesUpdate) {
    const bridge = await getBridge();
    if (!bridge) return;

    const nextState = await bridge.invoke<DesktopRuntimeState>("desktop_update_preferences", {
      updates,
    });
    setState(withBrowserState(nextState, { canInstall: false, isInstalled: true, isDesktopDevice }));
  }

  async function syncNativeState(payload: DesktopRemoteStateSyncPayload) {
    const bridge = await getBridge();
    if (!bridge) return;

    const nextState = await bridge.invoke<DesktopRuntimeState>("desktop_sync_remote_state", {
      payload,
    });
    setState(withBrowserState(nextState, { canInstall: false, isInstalled: true, isDesktopDevice }));
  }

  async function openControlCenter(route = DEFAULT_DESKTOP_ROUTE) {
    const bridge = await getBridge();
    if (!bridge) return;
    await bridge.invoke("desktop_show_control_center", { route });
  }

  async function openMainWindow(route = state.lastRemoteRoute || "/dashboard") {
    const bridge = await getBridge();
    if (!bridge) return;
    await bridge.invoke("desktop_open_main_window", { route });
  }

  async function checkForUpdates() {
    const bridge = await getBridge();
    if (!bridge) return;
    const nextState = await bridge.invoke<DesktopRuntimeState>("desktop_check_for_updates");
    setState(withBrowserState(nextState, { canInstall: false, isInstalled: true, isDesktopDevice }));
  }

  async function installUpdate() {
    const bridge = await getBridge();
    if (!bridge) {
      window.location.reload();
      return;
    }
    const nextState = await bridge.invoke<DesktopRuntimeState>("desktop_install_update");
    setState(withBrowserState(nextState, { canInstall: false, isInstalled: true, isDesktopDevice }));
  }

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canInstall, isInstalled, isDesktopDevice]);

  useEffect(() => {
    let unlisten: (() => void) | undefined;

    void (async () => {
      const bridge = await getBridge();
      if (!bridge) return;
      unlisten = await bridge.listen<DesktopRuntimeState>(DESKTOP_STATE_CHANGED_EVENT, (nextState) => {
        setState(withBrowserState(nextState, { canInstall: false, isInstalled: true, isDesktopDevice }));
      });
    })();

    return () => {
      unlisten?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isDesktopDevice]);

  useEffect(() => {
    const updateOnlineState = () => {
      setState((current) => ({
        ...current,
        isOnline: navigator.onLine,
      }));
    };

    window.addEventListener("online", updateOnlineState);
    window.addEventListener("offline", updateOnlineState);
    return () => {
      window.removeEventListener("online", updateOnlineState);
      window.removeEventListener("offline", updateOnlineState);
    };
  }, []);

  useEffect(() => {
    if (typeof document === "undefined") return;

    const root = document.documentElement;
    root.dataset.nativeDesktop = state.isNativeDesktop ? "1" : "0";
    root.dataset.standalonePwa = state.isStandalonePwa ? "1" : "0";
    root.dataset.desktopMode = state.isNativeDesktop || state.isStandalonePwa ? "1" : "0";
    root.dataset.desktopOnline = state.isOnline ? "1" : "0";
    root.dataset.desktopNotifications = state.notificationsEnabled ? "1" : "0";
  }, [state.isNativeDesktop, state.isStandalonePwa, state.isOnline, state.notificationsEnabled]);

  return (
    <DesktopRuntimeContext.Provider
      value={{
        state,
        refresh,
        updatePreferences,
        syncNativeState,
        openControlCenter,
        openMainWindow,
        checkForUpdates,
        installUpdate,
      }}
    >
      {children}
    </DesktopRuntimeContext.Provider>
  );
}

export function useDesktopRuntime() {
  const context = useContext(DesktopRuntimeContext);
  if (!context) {
    throw new Error("useDesktopRuntime must be used inside DesktopRuntimeProvider");
  }
  return context;
}
