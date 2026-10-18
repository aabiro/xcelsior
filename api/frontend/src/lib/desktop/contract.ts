export const DESKTOP_STATE_CHANGED_EVENT = "xcelsior://state-changed";

export type DesktopRoute =
  | "/desktop"
  | "/desktop/activity"
  | "/desktop/launch"
  | "/desktop/settings"
  | "/desktop/links";

export interface DesktopNotificationSummary {
  id: string;
  title: string;
  body: string;
  actionUrl: string;
  createdAt: number;
  read: boolean;
  type: string;
  priority: number;
}

export interface DesktopRuntimeState {
  isNativeDesktop: boolean;
  isStandalonePwa: boolean;
  canInstall: boolean;
  isInstalled: boolean;
  isOnline: boolean;
  notificationsEnabled: boolean;
  trayConnected: boolean;
  autostartEnabled: boolean;
  updateAvailable: boolean;
  currentDesktopRoute: DesktopRoute;
  lastRemoteRoute: string;
  unreadCount: number;
  criticalAlertCount: number;
  hideToTray: boolean;
  defaultDesktopRoute: DesktopRoute;
  updaterChannel: "stable" | "beta";
  currentVersion: string | null;
  updateVersion: string | null;
  remoteOrigin: string;
  devOrigin: string;
  authRequired: boolean;
  pendingDeepLinks: string[];
  recentNotifications: DesktopNotificationSummary[];
}

export interface DesktopPreferencesUpdate {
  launchOnLogin?: boolean;
  hideToTray?: boolean;
  notificationsEnabled?: boolean;
  defaultDesktopRoute?: DesktopRoute;
  updaterChannel?: "stable" | "beta";
  currentDesktopRoute?: DesktopRoute;
}

export interface DesktopRemoteStateSyncPayload {
  isOnline?: boolean;
  notificationsEnabled?: boolean;
  lastRemoteRoute?: string;
  unreadCount?: number;
  criticalAlertCount?: number;
  authRequired?: boolean;
  recentNotifications?: DesktopNotificationSummary[];
}

export const DEFAULT_DESKTOP_ROUTE: DesktopRoute = "/desktop";
export const DEFAULT_REMOTE_ROUTE = "/dashboard";

