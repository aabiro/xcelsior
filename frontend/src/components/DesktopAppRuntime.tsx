"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useDesktopRuntime } from "@/lib/desktop/runtime";

const LAST_ROUTE_KEY = "xcelsior-desktop-last-route";

function shouldPersistRoute(pathname: string) {
  return (
    pathname !== "/" &&
    pathname !== "/~offline" &&
    pathname !== "/login" &&
    pathname !== "/register" &&
    !pathname.startsWith("/desktop")
  );
}

export function DesktopAppRuntime() {
  const pathname = usePathname();
  const router = useRouter();
  const { state, syncNativeState } = useDesktopRuntime();
  const desktopMode = state.isNativeDesktop || state.isStandalonePwa;

  useEffect(() => {
    if (!desktopMode) return;

    if (pathname === "/") {
      const lastRoute = state.isNativeDesktop
        ? state.lastRemoteRoute
        : window.localStorage.getItem(LAST_ROUTE_KEY);
      if (lastRoute && lastRoute.startsWith("/") && lastRoute !== "/") {
        router.replace(lastRoute);
      }
      return;
    }

    if (!shouldPersistRoute(pathname)) return;

    const query = window.location.search.replace(/^\?/, "");
    const route = query ? `${pathname}?${query}` : pathname;
    window.localStorage.setItem(LAST_ROUTE_KEY, route);
    void syncNativeState({
      authRequired: false,
      lastRemoteRoute: route,
    });
  }, [desktopMode, pathname, router, state.isNativeDesktop, state.lastRemoteRoute, syncNativeState]);

  useEffect(() => {
    void syncNativeState({
      authRequired: pathname === "/login" || pathname === "/register",
    });
  }, [pathname, syncNativeState]);

  useEffect(() => {
    if (!state.isNativeDesktop) return;

    const syncConnectionState = () => {
      void syncNativeState({
        isOnline: navigator.onLine,
      });
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        syncConnectionState();
      }
    };

    syncConnectionState();
    window.addEventListener("online", syncConnectionState);
    window.addEventListener("offline", syncConnectionState);
    window.addEventListener("focus", syncConnectionState);
    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      window.removeEventListener("online", syncConnectionState);
      window.removeEventListener("offline", syncConnectionState);
      window.removeEventListener("focus", syncConnectionState);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [state.isNativeDesktop, syncNativeState]);

  return null;
}
