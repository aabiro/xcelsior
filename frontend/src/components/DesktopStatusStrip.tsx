"use client";

import { AppWindowMac, BellRing, CloudOff, Download, Wifi } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useDesktopRuntime } from "@/lib/desktop/runtime";
import { cn } from "@/lib/utils";

function StatusChip({
  label,
  tone,
}: {
  label: string;
  tone: "neutral" | "good" | "warn";
}) {
  const toneClass =
    tone === "good"
      ? "border-emerald/30 bg-emerald/10 text-emerald"
      : tone === "warn"
        ? "border-accent-gold/30 bg-accent-gold/10 text-accent-gold"
        : "border-border/70 bg-background/50 text-text-secondary";

  return (
    <span className={cn("rounded-full border px-2.5 py-1 text-[11px] font-medium", toneClass)}>
      {label}
    </span>
  );
}

export function DesktopStatusStrip({ className }: { className?: string }) {
  const { state, checkForUpdates, installUpdate, openControlCenter } = useDesktopRuntime();
  const desktopMode = state.isNativeDesktop || state.isStandalonePwa;

  if (!desktopMode) return null;

  return (
    <div className={cn("items-center gap-2", className)}>
      <StatusChip label={state.isNativeDesktop ? "Native Desktop" : "Installed PWA"} tone="neutral" />
      <StatusChip label={state.isOnline ? "Online" : "Offline"} tone={state.isOnline ? "good" : "warn"} />
      <StatusChip
        label={state.notificationsEnabled ? "Notifications On" : "Notifications Off"}
        tone={state.notificationsEnabled ? "good" : "warn"}
      />
      {state.isNativeDesktop ? (
        <Button type="button" variant="outline" size="sm" onClick={() => void openControlCenter("/desktop")}>
          <AppWindowMac className="h-4 w-4" />
          Control Center
        </Button>
      ) : null}
      {state.updateAvailable ? (
        <Button type="button" variant="secondary" size="sm" onClick={() => void installUpdate()}>
          <Download className="h-4 w-4" />
          {state.isNativeDesktop ? "Install Update" : "Reload Update"}
        </Button>
      ) : state.isNativeDesktop ? (
        <Button type="button" variant="ghost" size="sm" onClick={() => void checkForUpdates()}>
          <BellRing className="h-4 w-4" />
          Check Updates
        </Button>
      ) : null}
      {!state.isOnline ? <CloudOff className="h-4 w-4 text-accent-gold" /> : <Wifi className="h-4 w-4 text-emerald" />}
    </div>
  );
}

