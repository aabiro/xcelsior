use serde::{Deserialize, Serialize};

pub const DESKTOP_STATE_CHANGED_EVENT: &str = "xcelsior://state-changed";
pub const DEFAULT_DESKTOP_ROUTE: &str = "/desktop";
pub const DEFAULT_REMOTE_ROUTE: &str = "/dashboard";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DesktopNotificationSummary {
    pub id: String,
    pub title: String,
    pub body: String,
    pub action_url: String,
    pub created_at: i64,
    pub read: bool,
    pub r#type: String,
    pub priority: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DesktopRuntimeState {
    pub is_native_desktop: bool,
    pub is_standalone_pwa: bool,
    pub can_install: bool,
    pub is_installed: bool,
    pub is_online: bool,
    pub notifications_enabled: bool,
    pub tray_connected: bool,
    pub autostart_enabled: bool,
    pub update_available: bool,
    pub current_desktop_route: String,
    pub last_remote_route: String,
    pub unread_count: i32,
    pub critical_alert_count: i32,
    pub hide_to_tray: bool,
    pub default_desktop_route: String,
    pub updater_channel: String,
    pub current_version: Option<String>,
    pub update_version: Option<String>,
    pub remote_origin: String,
    pub dev_origin: String,
    pub auth_required: bool,
    pub pending_deep_links: Vec<String>,
    pub recent_notifications: Vec<DesktopNotificationSummary>,
}

impl DesktopRuntimeState {
    pub fn new(remote_origin: String, dev_origin: String, current_version: String) -> Self {
        Self {
            is_native_desktop: true,
            is_standalone_pwa: false,
            can_install: false,
            is_installed: true,
            is_online: true,
            notifications_enabled: true,
            tray_connected: false,
            autostart_enabled: false,
            update_available: false,
            current_desktop_route: DEFAULT_DESKTOP_ROUTE.to_string(),
            last_remote_route: DEFAULT_REMOTE_ROUTE.to_string(),
            unread_count: 0,
            critical_alert_count: 0,
            hide_to_tray: true,
            default_desktop_route: DEFAULT_DESKTOP_ROUTE.to_string(),
            updater_channel: "stable".to_string(),
            current_version: Some(current_version),
            update_version: None,
            remote_origin,
            dev_origin,
            auth_required: false,
            pending_deep_links: Vec::new(),
            recent_notifications: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DesktopPreferencesUpdate {
    pub launch_on_login: Option<bool>,
    pub hide_to_tray: Option<bool>,
    pub notifications_enabled: Option<bool>,
    pub default_desktop_route: Option<String>,
    pub updater_channel: Option<String>,
    pub current_desktop_route: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DesktopRemoteStateSyncPayload {
    pub is_online: Option<bool>,
    pub notifications_enabled: Option<bool>,
    pub last_remote_route: Option<String>,
    pub unread_count: Option<i32>,
    pub critical_alert_count: Option<i32>,
    pub auth_required: Option<bool>,
    pub recent_notifications: Option<Vec<DesktopNotificationSummary>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredDesktopState {
    pub last_remote_route: String,
    pub current_desktop_route: String,
    pub hide_to_tray: bool,
    pub notifications_enabled: bool,
    pub default_desktop_route: String,
    pub updater_channel: String,
    pub unread_count: i32,
    pub critical_alert_count: i32,
    pub auth_required: bool,
    pub pending_deep_links: Vec<String>,
    pub recent_notifications: Vec<DesktopNotificationSummary>,
    pub last_native_notification_id: Option<String>,
}

impl Default for StoredDesktopState {
    fn default() -> Self {
        Self {
            last_remote_route: DEFAULT_REMOTE_ROUTE.to_string(),
            current_desktop_route: DEFAULT_DESKTOP_ROUTE.to_string(),
            hide_to_tray: true,
            notifications_enabled: true,
            default_desktop_route: DEFAULT_DESKTOP_ROUTE.to_string(),
            updater_channel: "stable".to_string(),
            unread_count: 0,
            critical_alert_count: 0,
            auth_required: false,
            pending_deep_links: Vec::new(),
            recent_notifications: Vec::new(),
            last_native_notification_id: None,
        }
    }
}
