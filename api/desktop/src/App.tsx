import { useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import {
  AppWindowMac,
  ArrowRight,
  BellRing,
  Cable,
  ExternalLink,
  LayoutDashboard,
  Link2,
  Loader2,
  Play,
  PanelsTopLeft,
  RefreshCw,
  Rocket,
  Settings,
  ShieldCheck,
} from "lucide-react";
import type { DesktopPreferencesUpdate, DesktopRuntimeState, DesktopRoute } from "./lib/contract";
import {
  checkForUpdates,
  clearDeepLinks,
  getDesktopState,
  installUpdate,
  listenForDesktopState,
  openExternalUrl,
  openMainWindow,
  showControlCenter,
  updatePreferences,
} from "./lib/native";
import { parseDesktopDeepLink } from "./lib/deep-links";
import { getCurrentWindow } from "@tauri-apps/api/window";

const NAV_ITEMS: Array<{ route: DesktopRoute; label: string; icon: typeof LayoutDashboard }> = [
  { route: "/desktop", label: "Overview", icon: LayoutDashboard },
  { route: "/desktop/activity", label: "Activity", icon: BellRing },
  { route: "/desktop/launch", label: "Launch", icon: Rocket },
  { route: "/desktop/settings", label: "Settings", icon: Settings },
  { route: "/desktop/links", label: "Links", icon: Link2 },
];

const DESKTOP_ROUTE_OPTIONS: Array<{ route: DesktopRoute; label: string; description: string }> = [
  { route: "/desktop", label: "Overview", description: "Startup health and tray posture." },
  { route: "/desktop/activity", label: "Activity", description: "Notification and alert stream." },
  { route: "/desktop/launch", label: "Launch", description: "Recent work and shared-app handoff." },
  { route: "/desktop/settings", label: "Settings", description: "Native preferences and updater state." },
  { route: "/desktop/links", label: "Links", description: "Deep-link intake and route debugging." },
];

const UPDATER_CHANNEL_OPTIONS: Array<{ channel: "stable" | "beta"; label: string; description: string }> = [
  { channel: "stable", label: "Stable", description: "Production release train for operators." },
  { channel: "beta", label: "Beta", description: "Earlier builds for desktop validation." },
];

function isDesktopRoute(pathname: string): pathname is DesktopRoute {
  return NAV_ITEMS.some((item) => item.route === pathname);
}

function formatTimestamp(timestamp: number | undefined) {
  if (!timestamp) return "Waiting";
  return new Date(timestamp * 1000).toLocaleString();
}

function formatNotificationTimestamp(timestamp: number) {
  return new Date(timestamp * 1000).toLocaleString();
}

function StatusPill({ label, tone }: { label: string; tone: "good" | "warn" | "neutral" }) {
  return <span className={`pill pill-${tone}`}>{label}</span>;
}

function PreferenceToggle({
  title,
  description,
  enabled,
  disabled,
  onToggle,
}: {
  title: string;
  description: string;
  enabled: boolean;
  disabled?: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="metric-card">
      <div>
        <p className="section-title">{title}</p>
        <p className="section-copy">{description}</p>
      </div>
      <button type="button" className={`toggle ${enabled ? "toggle-on" : ""}`} onClick={onToggle} disabled={disabled}>
        <span />
      </button>
    </div>
  );
}

function QuickLink({
  title,
  description,
  onClick,
}: {
  title: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <button type="button" className="quick-link" onClick={onClick}>
      <div>
        <p>{title}</p>
        <span>{description}</span>
      </div>
      <ArrowRight size={16} />
    </button>
  );
}

function OptionChip({
  title,
  description,
  active,
  onClick,
}: {
  title: string;
  description: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`option-chip ${active ? "option-chip-active" : ""}`}
      onClick={onClick}
    >
      <strong>{title}</strong>
      <span>{description}</span>
    </button>
  );
}

function OverviewPage({
  state,
  onOpenRemote,
  onCheckForUpdates,
  checkingUpdates,
}: {
  state: DesktopRuntimeState;
  onOpenRemote: (route: string) => void;
  onCheckForUpdates: () => void;
  checkingUpdates: boolean;
}) {
  return (
    <div className="page-grid">
      <section className="hero-card">
        <div>
          <p className="eyebrow">Native Desktop Infrastructure</p>
          <h1>Xcelsior Control Center</h1>
          <p className="section-copy">
            Tray orchestration, launch routing, update readiness, deep-link intake, and desktop notification state stay
            here while the main Xcelsior product UI remains the shared hosted app.
          </p>
        </div>
        <div className="hero-actions">
          <button type="button" className="primary-button" onClick={() => onOpenRemote(state.lastRemoteRoute || "/dashboard")}>
            <AppWindowMac size={16} />
            Open Main App
          </button>
          <button type="button" className="secondary-button" onClick={onCheckForUpdates} disabled={checkingUpdates}>
            {checkingUpdates ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
            Check Updates
          </button>
        </div>
      </section>

      <section className="metrics-grid">
        <div className="metric-card">
          <p className="metric-label">Tray</p>
          <p className="metric-value">{state.trayConnected ? "Connected" : "Disconnected"}</p>
          <p className="section-copy">Hide-to-tray is {state.hideToTray ? "enabled" : "disabled"}.</p>
        </div>
        <div className="metric-card">
          <p className="metric-label">Unread</p>
          <p className="metric-value">{state.unreadCount}</p>
          <p className="section-copy">Critical alerts: {state.criticalAlertCount}</p>
        </div>
        <div className="metric-card">
          <p className="metric-label">Updater</p>
          <p className="metric-value">{state.updateAvailable ? state.updateVersion || "Ready" : "Current"}</p>
          <p className="section-copy">Channel: {state.updaterChannel}</p>
        </div>
        <div className="metric-card">
          <p className="metric-label">Network</p>
          <p className="metric-value">{state.isOnline ? "Online" : "Offline"}</p>
          <p className="section-copy">Remote origin: {state.remoteOrigin || "Not configured"}</p>
        </div>
      </section>

      <section className="panel-card">
        <div className="panel-header">
          <div>
            <p className="section-title">Quick Actions</p>
            <p className="section-copy">Jump into the highest-value infrastructure surfaces in the shared app.</p>
          </div>
        </div>
        <div className="quick-link-grid">
          <QuickLink title="Marketplace" description="Launch new infrastructure" onClick={() => onOpenRemote("/dashboard/marketplace")} />
          <QuickLink title="Instances" description="Inspect active workloads" onClick={() => onOpenRemote("/dashboard/instances")} />
          <QuickLink title="Notifications" description="Review the live event stream" onClick={() => onOpenRemote("/dashboard/notifications")} />
          <QuickLink title="Admin" description="Open platform controls" onClick={() => onOpenRemote("/dashboard/admin")} />
        </div>
      </section>
    </div>
  );
}

function ActivityPage({ state, onOpenRemote }: { state: DesktopRuntimeState; onOpenRemote: (route: string) => void }) {
  return (
    <div className="stack">
      <section className="panel-card">
        <div className="panel-header">
          <div>
            <p className="section-title">Recent Notifications</p>
            <p className="section-copy">Mirrored from the shared notification model so the desktop shell can surface them in tray and native alerts.</p>
          </div>
          <StatusPill label={`${state.unreadCount} unread`} tone={state.unreadCount > 0 ? "warn" : "good"} />
        </div>

        {state.recentNotifications.length === 0 ? (
          <div className="empty-state">No notification activity has been synced from the main app yet.</div>
        ) : (
          <div className="notification-list">
            {state.recentNotifications.map((notification) => (
              <button
                key={notification.id}
                type="button"
                className="notification-row"
                onClick={() => onOpenRemote(notification.actionUrl)}
              >
                <div>
                  <p>{notification.title}</p>
                  <span>{notification.body || "Open this route in the main app."}</span>
                </div>
                <div className="notification-meta">
                  <strong>{notification.read ? "Read" : "Unread"}</strong>
                  <span>{formatNotificationTimestamp(notification.createdAt)}</span>
                </div>
              </button>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function LaunchPage({ state, onOpenRemote }: { state: DesktopRuntimeState; onOpenRemote: (route: string) => void }) {
  return (
    <div className="page-grid">
      <section className="panel-card">
        <div className="panel-header">
          <div>
            <p className="section-title">Resume Work</p>
            <p className="section-copy">The native shell remembers your last shared-app route and the default desktop landing route.</p>
          </div>
        </div>
        <div className="quick-link-grid">
          <QuickLink title="Last Remote Route" description={state.lastRemoteRoute} onClick={() => onOpenRemote(state.lastRemoteRoute)} />
          <QuickLink title="Marketplace" description="Open the launch flow" onClick={() => onOpenRemote("/dashboard/marketplace")} />
          <QuickLink title="Billing" description="Review balance and funding" onClick={() => onOpenRemote("/dashboard/billing")} />
          <QuickLink title="Telemetry" description="Inspect live platform performance" onClick={() => onOpenRemote("/dashboard/telemetry")} />
        </div>
      </section>

      <section className="panel-card">
        <div className="panel-header">
          <div>
            <p className="section-title">Desktop Defaults</p>
            <p className="section-copy">These routes are owned by the native shell and can stay independent from the shared web navigation.</p>
          </div>
        </div>
        <div className="detail-list">
          <div><strong>Control Center route</strong><span>{state.currentDesktopRoute}</span></div>
          <div><strong>Default desktop route</strong><span>{state.defaultDesktopRoute}</span></div>
          <div><strong>Deep-link scheme</strong><span>xcelsior://</span></div>
        </div>
      </section>
    </div>
  );
}

function SettingsPage({
  state,
  savingPreference,
  onPreferenceUpdate,
  onInstallUpdate,
}: {
  state: DesktopRuntimeState;
  savingPreference: boolean;
  onPreferenceUpdate: (updates: DesktopPreferencesUpdate) => void;
  onInstallUpdate: () => void;
}) {
  return (
    <div className="stack">
      <PreferenceToggle
        title="Launch on login"
        description="Start the desktop shell automatically so tray and route recovery are available immediately."
        enabled={state.autostartEnabled}
        disabled={savingPreference}
        onToggle={() => onPreferenceUpdate({ launchOnLogin: !state.autostartEnabled })}
      />
      <PreferenceToggle
        title="Hide to tray"
        description="Closing the main window keeps the infrastructure client alive in the background."
        enabled={state.hideToTray}
        disabled={savingPreference}
        onToggle={() => onPreferenceUpdate({ hideToTray: !state.hideToTray })}
      />
      <PreferenceToggle
        title="Native notifications"
        description="Allow the desktop shell to surface mirrored in-app events through the tray and OS notification layer."
        enabled={state.notificationsEnabled}
        disabled={savingPreference}
        onToggle={() => onPreferenceUpdate({ notificationsEnabled: !state.notificationsEnabled })}
      />
      <section className="panel-card">
        <div className="panel-header">
          <div>
            <p className="section-title">Default Desktop Route</p>
            <p className="section-copy">Choose where the native Control Center should land on startup and tray reopen.</p>
          </div>
          <StatusPill label={state.defaultDesktopRoute} tone="neutral" />
        </div>
        <div className="option-grid">
          {DESKTOP_ROUTE_OPTIONS.map((option) => (
            <OptionChip
              key={option.route}
              title={option.label}
              description={option.description}
              active={state.defaultDesktopRoute === option.route}
              onClick={() => onPreferenceUpdate({ defaultDesktopRoute: option.route })}
            />
          ))}
        </div>
      </section>
      <section className="panel-card">
        <div className="panel-header">
          <div>
            <p className="section-title">Updater</p>
            <p className="section-copy">Native release metadata comes from the Tauri updater configuration.</p>
          </div>
          {state.updateAvailable ? <StatusPill label="Update Ready" tone="warn" /> : <StatusPill label="Current" tone="good" />}
        </div>
        <div className="detail-list">
          <div><strong>Current version</strong><span>{state.currentVersion || "Unknown"}</span></div>
          <div><strong>Available version</strong><span>{state.updateVersion || "None"}</span></div>
          <div><strong>Channel</strong><span>{state.updaterChannel}</span></div>
        </div>
        <div className="option-grid option-grid-compact">
          {UPDATER_CHANNEL_OPTIONS.map((option) => (
            <OptionChip
              key={option.channel}
              title={option.label}
              description={option.description}
              active={state.updaterChannel === option.channel}
              onClick={() => onPreferenceUpdate({ updaterChannel: option.channel })}
            />
          ))}
        </div>
        {state.updateAvailable ? (
          <button type="button" className="primary-button" onClick={onInstallUpdate}>
            <Rocket size={16} />
            Install Update
          </button>
        ) : null}
      </section>
    </div>
  );
}

function LinksPage({
  state,
  onOpenDesktop,
  onOpenRemote,
  onClear,
  clearing,
}: {
  state: DesktopRuntimeState;
  onOpenDesktop: (route: DesktopRoute) => void;
  onOpenRemote: (route: string) => void;
  onClear: () => void;
  clearing: boolean;
}) {
  const pendingLinks = state.pendingDeepLinks.map((link) => ({
    raw: link,
    preview: parseDesktopDeepLink(link),
  }));

  return (
    <div className="stack">
      <section className="panel-card">
        <div className="panel-header">
          <div>
            <p className="section-title">Deep Links</p>
            <p className="section-copy">Incoming `xcelsior://` links are queued here for visibility and debugging.</p>
          </div>
          <button type="button" className="secondary-button" onClick={onClear} disabled={clearing}>
            {clearing ? <Loader2 className="spin" size={16} /> : <ShieldCheck size={16} />}
            Clear
          </button>
        </div>

        {state.pendingDeepLinks.length === 0 ? (
          <div className="empty-state">No deep links have been opened in this desktop session.</div>
        ) : (
          <div className="detail-list">
            {pendingLinks.map(({ raw, preview }) => (
              <div key={raw} className="detail-item-split">
                <div className="detail-copy">
                  <strong>{raw}</strong>
                  <span>
                    {preview
                      ? `Targets ${preview.target === "desktop" ? "control center" : "shared app"} route ${preview.route}.`
                      : "Unable to parse this deep link. Inspect the raw value before replaying it."}
                  </span>
                </div>
                <div className="detail-actions">
                  <StatusPill
                    label={preview ? `${preview.target === "desktop" ? "Desktop" : "Remote"} Route` : "Unknown"}
                    tone={preview ? "good" : "warn"}
                  />
                  {preview ? (
                    <button
                      type="button"
                      className="secondary-button secondary-button-small"
                      onClick={() =>
                        preview.target === "desktop" ? onOpenDesktop(preview.route) : onOpenRemote(preview.route)
                      }
                    >
                      <ArrowRight size={16} />
                      Open Target
                    </button>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

export default function App() {
  const [state, setState] = useState<DesktopRuntimeState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [checkingUpdates, setCheckingUpdates] = useState(false);
  const [savingPreference, setSavingPreference] = useState(false);
  const [clearingLinks, setClearingLinks] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    let unlisten: (() => void) | undefined;

    void (async () => {
      try {
        const current = await getDesktopState();
        setState(current);
        navigate(current.currentDesktopRoute, { replace: true });
        unlisten = await listenForDesktopState((nextState) => {
          setState(nextState);
          if (location.pathname !== nextState.currentDesktopRoute) {
            navigate(nextState.currentDesktopRoute, { replace: true });
          }
        });
      } catch (nextError) {
        console.error("Failed to load desktop state", nextError);
        setError("Failed to load native desktop state.");
      } finally {
        setLoading(false);
      }
    })();

    return () => {
      unlisten?.();
    };
  }, [location.pathname, navigate]);

  useEffect(() => {
    if (!state) return;
    if (!isDesktopRoute(location.pathname)) return;
    if (location.pathname === state.currentDesktopRoute) return;

    let cancelled = false;

    void updatePreferences({ currentDesktopRoute: location.pathname }).then(
      (nextState) => {
        if (!cancelled) {
          setState(nextState);
        }
      },
      (nextError) => {
        console.error("Failed to sync current desktop route", nextError);
        if (!cancelled) {
          setError("Failed to sync the current control center route.");
        }
      },
    );

    return () => {
      cancelled = true;
    };
  }, [location.pathname, state]);

  // ── Keyboard shortcuts ─────────────────────────────────────────────
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const mod = e.metaKey || e.ctrlKey;
      if (!mod) return;

      // Ctrl/Cmd + 1–5 → sidebar nav
      const numKey = parseInt(e.key, 10);
      if (numKey >= 1 && numKey <= NAV_ITEMS.length) {
        e.preventDefault();
        const target = NAV_ITEMS[numKey - 1];
        if (target) {
          navigate(target.route, { replace: true });
          void showControlCenter(target.route);
        }
        return;
      }

      // Ctrl/Cmd + , → Settings
      if (e.key === ",") {
        e.preventDefault();
        navigate("/desktop/settings", { replace: true });
        void showControlCenter("/desktop/settings");
        return;
      }

      // Ctrl/Cmd + W → hide window (tray-aware)
      if (e.key === "w" || e.key === "W") {
        e.preventDefault();
        void getCurrentWindow().hide();
        return;
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [navigate]);

  if (loading || !state) {
    return (
      <div className="splash-screen">
        <Loader2 className="spin" size={20} />
        <span>Loading Control Center...</span>
      </div>
    );
  }

  async function handlePreferenceUpdate(updates: DesktopPreferencesUpdate) {
    setSavingPreference(true);
    try {
      const nextState = await updatePreferences(updates);
      setState(nextState);
      if (updates.currentDesktopRoute) {
        navigate(updates.currentDesktopRoute, { replace: true });
      }
    } catch (nextError) {
      console.error("Failed to update desktop preferences", nextError);
      setError("Failed to update desktop preferences.");
    } finally {
      setSavingPreference(false);
    }
  }

  async function handleOpenRemote(route: string) {
    try {
      await openMainWindow(route);
    } catch (nextError) {
      console.error("Failed to open remote app window", nextError);
      setError("Failed to open the shared app window.");
    }
  }

  async function handleOpenDesktop(route: DesktopRoute) {
    try {
      await showControlCenter(route);
      navigate(route, { replace: true });
    } catch (nextError) {
      console.error("Failed to open control center route", nextError);
      setError("Failed to open the control center route.");
    }
  }

  async function handleCheckForUpdates() {
    setCheckingUpdates(true);
    try {
      const nextState = await checkForUpdates();
      setState(nextState);
    } catch (nextError) {
      console.error("Failed to check for updates", nextError);
      setError("Failed to check for updates.");
    } finally {
      setCheckingUpdates(false);
    }
  }

  async function handleInstallUpdate() {
    try {
      const nextState = await installUpdate();
      setState(nextState);
    } catch (nextError) {
      console.error("Failed to install update", nextError);
      setError("Failed to install the desktop update.");
    }
  }

  async function handleClearDeepLinks() {
    setClearingLinks(true);
    try {
      const nextState = await clearDeepLinks();
      setState(nextState);
    } catch (nextError) {
      console.error("Failed to clear deep links", nextError);
      setError("Failed to clear pending deep links.");
    } finally {
      setClearingLinks(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar titlebar-safe">
        <div className="brand">
          <div className="brand-mark">X</div>
          <div>
            <p className="eyebrow">Xcelsior</p>
            <h2>Control Center</h2>
          </div>
        </div>
        <nav className="nav-list">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.route}
              to={item.route}
              className={({ isActive }) => `nav-item ${isActive ? "nav-item-active" : ""}`}
              onClick={() => void showControlCenter(item.route)}
            >
              <item.icon size={18} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <StatusPill label={state.isOnline ? "Online" : "Offline"} tone={state.isOnline ? "good" : "warn"} />
          <StatusPill label={state.trayConnected ? "Tray Ready" : "Tray Offline"} tone={state.trayConnected ? "good" : "warn"} />
        </div>
      </aside>

      <main className="content">
        <header className="topbar" data-tauri-drag-region="true">
          <div>
            <p className="eyebrow">Desktop Infrastructure Shell</p>
            <h1>{NAV_ITEMS.find((item) => item.route === location.pathname)?.label || "Overview"}</h1>
          </div>
          <div className="topbar-actions">
            <button type="button" className="secondary-button" onClick={() => void openExternalUrl("https://docs.xcelsior.ca")}>
              <ExternalLink size={16} />
              Docs
            </button>
            <button type="button" className="primary-button" onClick={() => void handleOpenRemote(state.lastRemoteRoute || "/dashboard")}>
              <Play size={16} />
              Open Shared App
            </button>
          </div>
        </header>

        <section className="status-bar">
          <StatusPill label={state.notificationsEnabled ? "Notifications On" : "Notifications Off"} tone={state.notificationsEnabled ? "good" : "warn"} />
          <StatusPill label={`Unread ${state.unreadCount}`} tone={state.unreadCount > 0 ? "warn" : "neutral"} />
          <StatusPill label={`Critical ${state.criticalAlertCount}`} tone={state.criticalAlertCount > 0 ? "warn" : "neutral"} />
          <StatusPill label={`Last Route ${state.lastRemoteRoute}`} tone="neutral" />
          <StatusPill label={state.updateAvailable ? `Update ${state.updateVersion || "Ready"}` : "Version Current"} tone={state.updateAvailable ? "warn" : "good"} />
        </section>

        {error ? <div className="error-banner">{error}</div> : null}

        <Routes>
          <Route
            path="/desktop"
            element={
              <OverviewPage
                state={state}
                onOpenRemote={handleOpenRemote}
                onCheckForUpdates={handleCheckForUpdates}
                checkingUpdates={checkingUpdates}
              />
            }
          />
          <Route path="/desktop/activity" element={<ActivityPage state={state} onOpenRemote={handleOpenRemote} />} />
          <Route path="/desktop/launch" element={<LaunchPage state={state} onOpenRemote={handleOpenRemote} />} />
          <Route
            path="/desktop/settings"
            element={
              <SettingsPage
                state={state}
                savingPreference={savingPreference}
                onPreferenceUpdate={handlePreferenceUpdate}
                onInstallUpdate={handleInstallUpdate}
              />
            }
          />
          <Route
            path="/desktop/links"
            element={
              <LinksPage
                state={state}
                onOpenDesktop={handleOpenDesktop}
                onOpenRemote={handleOpenRemote}
                onClear={handleClearDeepLinks}
                clearing={clearingLinks}
              />
            }
          />
          <Route path="*" element={<OverviewPage state={state} onOpenRemote={handleOpenRemote} onCheckForUpdates={handleCheckForUpdates} checkingUpdates={checkingUpdates} />} />
        </Routes>

        <footer className="footer">
          <div className="footer-item">
            <PanelsTopLeft size={16} />
            <span>{state.trayConnected ? "Tray connected" : "Tray disconnected"}</span>
          </div>
          <div className="footer-item">
            <Cable size={16} />
            <span>Remote origin {state.remoteOrigin || "not configured"}</span>
          </div>
          <div className="footer-item">
            <BellRing size={16} />
            <span>Last notification sync {state.recentNotifications[0] ? formatTimestamp(state.recentNotifications[0].createdAt) : "never"}</span>
          </div>
        </footer>
      </main>
    </div>
  );
}
