"use client";

import { useState } from "react";
import type { DesktopPreferencesUpdate, DesktopRoute } from "@/lib/desktop/contract";
import {
  AppWindowMac,
  BellRing,
  ExternalLink,
  Loader2,
  PlugZap,
  RefreshCw,
  Rocket,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { useDesktopRuntime } from "@/lib/desktop/runtime";
import { cn } from "@/lib/utils";

const DESKTOP_ROUTE_OPTIONS: Array<{ route: DesktopRoute; label: string; description: string }> = [
  { route: "/desktop", label: "Overview", description: "Health, tray state, and startup posture." },
  { route: "/desktop/activity", label: "Activity", description: "Desktop alerts and notification stream." },
  { route: "/desktop/launch", label: "Launch", description: "Recent work and shared-app handoff." },
  { route: "/desktop/settings", label: "Settings", description: "Native preferences and release controls." },
  { route: "/desktop/links", label: "Links", description: "Deep-link intake and route debugging." },
];

const UPDATER_CHANNEL_OPTIONS: Array<{ channel: "stable" | "beta"; label: string; description: string }> = [
  { channel: "stable", label: "Stable", description: "Production release track." },
  { channel: "beta", label: "Beta", description: "Earlier desktop validation builds." },
];

function InlineToggle({
  enabled,
  onToggle,
}: {
  enabled: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className={cn(
        "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors duration-200",
        enabled ? "bg-emerald" : "bg-border",
      )}
    >
      <span
        className={cn(
          "inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform duration-200",
          enabled ? "translate-x-6" : "translate-x-1",
        )}
      />
    </button>
  );
}

function OptionButton({
  label,
  description,
  active,
  onClick,
}: {
  label: string;
  description: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-xl border px-3 py-3 text-left transition-colors",
        active
          ? "border-accent-cyan/30 bg-accent-cyan/10"
          : "border-border/70 bg-background/40 hover:bg-background/60",
      )}
    >
      <p className="text-sm font-medium text-text-primary">{label}</p>
      <p className="mt-1 text-xs text-text-secondary">{description}</p>
    </button>
  );
}

export function NativeDesktopPreferences() {
  const { state, updatePreferences, openControlCenter, openMainWindow, checkForUpdates, installUpdate } =
    useDesktopRuntime();
  const [saving, setSaving] = useState<string | null>(null);
  const [checkingUpdate, setCheckingUpdate] = useState(false);

  if (!state.isNativeDesktop) return null;

  async function handlePreferenceUpdate(key: string, updates: DesktopPreferencesUpdate, successMessage = "Desktop preference updated.") {
    setSaving(key);
    try {
      await updatePreferences(updates);
      toast.success(successMessage);
    } catch (error) {
      console.error("Failed to update native desktop preference", error);
      toast.error("Failed to update native desktop preference.");
    } finally {
      setSaving(null);
    }
  }

  async function handleCheckForUpdates() {
    setCheckingUpdate(true);
    try {
      await checkForUpdates();
      toast.success(state.updateAvailable ? "Desktop update is ready to install." : "Desktop app is current.");
    } catch (error) {
      console.error("Failed to check for desktop updates", error);
      toast.error("Failed to check for desktop updates.");
    } finally {
      setCheckingUpdate(false);
    }
  }

  async function handleInstallUpdate() {
    try {
      await installUpdate();
      toast.success("Desktop update installed. Restart the app if prompted.");
    } catch (error) {
      console.error("Failed to install desktop update", error);
      toast.error("Failed to install desktop update.");
    }
  }

  return (
    <div className="mt-5 border-t border-border/60 pt-5">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <AppWindowMac className="h-4 w-4 text-accent-cyan" />
            <p className="text-sm font-medium text-text-primary">Native desktop shell</p>
            <span className="rounded-full border border-emerald/30 bg-emerald/10 px-2 py-0.5 text-[11px] font-medium text-emerald">
              Connected
            </span>
          </div>
          <p className="max-w-2xl text-xs text-text-secondary">
            Tauri keeps the shared Xcelsior app in a native window, adds tray orchestration, deep links, launch-on-login,
            and a local Control Center without changing the website layout.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="secondary" size="sm" onClick={() => void openControlCenter("/desktop")}>
            <PlugZap className="h-4 w-4" />
            Open Control Center
          </Button>
          <Button type="button" variant="outline" size="sm" onClick={() => void openMainWindow(state.lastRemoteRoute)}>
            <ExternalLink className="h-4 w-4" />
            Reopen Main Window
          </Button>
        </div>
      </div>

      <div className="mt-5 grid gap-3 md:grid-cols-2">
        <div className="rounded-xl border border-border/70 bg-background/50 p-4">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-text-primary">Launch on login</p>
              <p className="mt-1 text-xs text-text-secondary">Start the desktop shell automatically so tray controls are ready.</p>
            </div>
            {saving === "launchOnLogin" ? (
              <Loader2 className="h-4 w-4 animate-spin text-text-secondary" />
            ) : (
              <InlineToggle
                enabled={state.autostartEnabled}
                onToggle={() => void handlePreferenceUpdate("launchOnLogin", { launchOnLogin: !state.autostartEnabled })}
              />
            )}
          </div>
        </div>

        <div className="rounded-xl border border-border/70 bg-background/50 p-4">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-text-primary">Hide to tray on close</p>
              <p className="mt-1 text-xs text-text-secondary">Keep infrastructure controls alive in the tray instead of quitting the app.</p>
            </div>
            {saving === "hideToTray" ? (
              <Loader2 className="h-4 w-4 animate-spin text-text-secondary" />
            ) : (
              <InlineToggle
                enabled={state.hideToTray}
                onToggle={() => void handlePreferenceUpdate("hideToTray", { hideToTray: !state.hideToTray })}
              />
            )}
          </div>
        </div>

        <div className="rounded-xl border border-border/70 bg-background/50 p-4">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium text-text-primary">Native desktop notifications</p>
              <p className="mt-1 text-xs text-text-secondary">Mirror important in-app events into the native desktop shell and tray.</p>
            </div>
            {saving === "notificationsEnabled" ? (
              <Loader2 className="h-4 w-4 animate-spin text-text-secondary" />
            ) : (
              <InlineToggle
                enabled={state.notificationsEnabled}
                onToggle={() =>
                  void handlePreferenceUpdate("notificationsEnabled", {
                    notificationsEnabled: !state.notificationsEnabled,
                  })
                }
              />
            )}
          </div>
        </div>

        <div className="rounded-xl border border-border/70 bg-background/50 p-4">
          <div className="space-y-3">
            <div>
              <p className="text-sm font-medium text-text-primary">Default desktop route</p>
              <p className="mt-1 text-xs text-text-secondary">Pick where the Control Center should land on startup or tray reopen.</p>
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              {DESKTOP_ROUTE_OPTIONS.map((option) => (
                <OptionButton
                  key={option.route}
                  label={option.label}
                  description={option.description}
                  active={state.defaultDesktopRoute === option.route}
                  onClick={() =>
                    void handlePreferenceUpdate(
                      "defaultDesktopRoute",
                      { defaultDesktopRoute: option.route },
                      "Default desktop route updated.",
                    )
                  }
                />
              ))}
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-border/70 bg-background/50 p-4">
          <div className="space-y-4">
            <div className="flex items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <BellRing className="h-4 w-4 text-accent-gold" />
                  <p className="text-sm font-medium text-text-primary">Updater</p>
                </div>
                <p className="text-xs text-text-secondary">
                  Channel: <span className="font-mono text-text-primary">{state.updaterChannel}</span>
                  {state.updateVersion ? `, ready: ${state.updateVersion}` : ", current"}
                </p>
              </div>

              <div className="flex gap-2">
                <Button type="button" variant="outline" size="sm" onClick={() => void handleCheckForUpdates()} disabled={checkingUpdate}>
                  {checkingUpdate ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
                  Check
                </Button>
                {state.updateAvailable ? (
                  <Button type="button" variant="secondary" size="sm" onClick={() => void handleInstallUpdate()}>
                    <Rocket className="h-4 w-4" />
                    Install
                  </Button>
                ) : null}
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              {UPDATER_CHANNEL_OPTIONS.map((option) => (
                <OptionButton
                  key={option.channel}
                  label={option.label}
                  description={option.description}
                  active={state.updaterChannel === option.channel}
                  onClick={() =>
                    void handlePreferenceUpdate(
                      "updaterChannel",
                      { updaterChannel: option.channel },
                      "Desktop release channel updated.",
                    )
                  }
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-border/70 bg-background/40 p-4 text-xs text-text-secondary">
        <p>Deep links: <span className="font-mono text-text-primary">xcelsior://</span></p>
        <p className="mt-1">Last remote route: <span className="font-mono text-text-primary">{state.lastRemoteRoute}</span></p>
        <p className="mt-1">Default desktop route: <span className="font-mono text-text-primary">{state.defaultDesktopRoute}</span></p>
      </div>
    </div>
  );
}
