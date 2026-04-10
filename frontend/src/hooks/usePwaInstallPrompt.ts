"use client";

import { useCallback, useEffect, useState } from "react";

declare global {
  interface BeforeInstallPromptEvent extends Event {
    prompt(): Promise<void>;
    userChoice: Promise<{ outcome: "accepted" | "dismissed"; platform: string }>;
  }

  interface Navigator {
    standalone?: boolean;
  }
}

export function getInstalledState() {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
}

export function getDesktopState() {
  if (typeof window === "undefined") return false;

  const ua = window.navigator.userAgent.toLowerCase();
  return !/android|iphone|ipad|ipod/.test(ua);
}

export function usePwaInstallPrompt() {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [isInstalled, setIsInstalled] = useState(false);
  const [isDesktopDevice, setIsDesktopDevice] = useState(false);

  useEffect(() => {
    const updateInstalledState = () => {
      const installed = getInstalledState();
      setIsInstalled(installed);

      if (installed) {
        window.localStorage.setItem("xcelsior-pwa-installed", "1");
      }
    };

    updateInstalledState();
    setIsDesktopDevice(getDesktopState());

    const mediaQuery = window.matchMedia("(display-mode: standalone)");
    const handleDisplayModeChange = () => updateInstalledState();
    const handleBeforeInstallPrompt = (event: Event) => {
      event.preventDefault();
      setDeferredPrompt(event as BeforeInstallPromptEvent);
    };
    const handleInstalled = () => {
      setDeferredPrompt(null);
      updateInstalledState();
    };

    mediaQuery.addEventListener("change", handleDisplayModeChange);
    window.addEventListener("beforeinstallprompt", handleBeforeInstallPrompt);
    window.addEventListener("appinstalled", handleInstalled);

    return () => {
      mediaQuery.removeEventListener("change", handleDisplayModeChange);
      window.removeEventListener("beforeinstallprompt", handleBeforeInstallPrompt);
      window.removeEventListener("appinstalled", handleInstalled);
    };
  }, []);

  const promptToInstall = useCallback(async () => {
    if (!deferredPrompt) return "unavailable" as const;

    await deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    setDeferredPrompt(null);

    if (outcome === "accepted") {
      window.localStorage.removeItem("xcelsior-pwa-dismissed");
    }

    return outcome;
  }, [deferredPrompt]);

  return {
    canInstall: Boolean(deferredPrompt) && !isInstalled,
    isDesktopDevice,
    isInstalled,
    promptToInstall,
  };
}
