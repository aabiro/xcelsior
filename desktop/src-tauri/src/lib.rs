mod config;
mod contract;
mod routing;
mod startup;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use anyhow::{anyhow, Context, Result};
use config::{load_shell_config, DesktopShellConfig};
use contract::{
    DesktopPreferencesUpdate, DesktopRemoteStateSyncPayload, DesktopRuntimeState,
    StoredDesktopState, DESKTOP_STATE_CHANGED_EVENT, DEFAULT_DESKTOP_ROUTE,
};
use routing::{deep_link_target, normalize_desktop_route, normalize_remote_route, DeepLinkTarget};
use startup::{launch_mode_from_env, LaunchMode};
use tauri::{
    menu::{MenuBuilder, MenuItemBuilder},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager, State, WebviewUrl, WebviewWindow, WebviewWindowBuilder, WindowEvent,
    Wry,
};
use tauri_plugin_autostart::ManagerExt as _;
use tauri_plugin_deep_link::DeepLinkExt as _;
use tauri_plugin_opener::OpenerExt as _;
use tauri_plugin_store::{Store, StoreBuilder};
use tauri_plugin_updater::UpdaterExt as _;
use url::Url;

const CONTROL_CENTER_WINDOW_LABEL: &str = "control-center";
const MAIN_WINDOW_LABEL: &str = "main-app";
const STORE_PATH: &str = "desktop-shell.json";
const STORE_KEY: &str = "desktop";
const TRAY_ID: &str = "xcelsior-desktop";
const TRAY_OPEN_CONTROL_CENTER: &str = "tray-open-control-center";
const TRAY_OPEN_MAIN: &str = "tray-open-main";
const TRAY_OPEN_MARKETPLACE: &str = "tray-open-marketplace";
const TRAY_OPEN_INSTANCES: &str = "tray-open-instances";
const TRAY_OPEN_NOTIFICATIONS: &str = "tray-open-notifications";
const TRAY_QUIT: &str = "tray-quit";

struct DesktopShellState {
    runtime: Mutex<DesktopRuntimeState>,
    config: DesktopShellConfig,
    store: Arc<Store<Wry>>,
    is_quitting: AtomicBool,
    last_native_notification_id: Mutex<Option<String>>,
}

impl DesktopShellState {
    fn snapshot(&self) -> DesktopRuntimeState {
        self.runtime
            .lock()
            .expect("desktop runtime lock poisoned")
            .clone()
    }

    fn update<F>(&self, updater: F) -> DesktopRuntimeState
    where
        F: FnOnce(&mut DesktopRuntimeState),
    {
        let mut runtime = self.runtime.lock().expect("desktop runtime lock poisoned");
        updater(&mut runtime);
        runtime.clone()
    }

    fn persist(&self) -> Result<()> {
        let runtime = self.snapshot();
        let stored = StoredDesktopState {
            last_remote_route: runtime.last_remote_route,
            current_desktop_route: runtime.current_desktop_route,
            hide_to_tray: runtime.hide_to_tray,
            notifications_enabled: runtime.notifications_enabled,
            default_desktop_route: runtime.default_desktop_route,
            updater_channel: runtime.updater_channel,
            unread_count: runtime.unread_count,
            critical_alert_count: runtime.critical_alert_count,
            auth_required: runtime.auth_required,
            pending_deep_links: runtime.pending_deep_links,
            recent_notifications: runtime.recent_notifications,
            last_native_notification_id: self
                .last_native_notification_id
                .lock()
                .expect("desktop runtime notification lock poisoned")
                .clone(),
        };

        self.store
            .set(
                STORE_KEY,
                serde_json::to_value(stored).context("failed to serialize desktop store state")?,
            );
        self.store.save().context("failed to save desktop store")?;
        Ok(())
    }
}

fn load_stored_state(store: &Store<Wry>) -> StoredDesktopState {
    store
        .get(STORE_KEY)
        .and_then(|value| serde_json::from_value::<StoredDesktopState>(value).ok())
        .unwrap_or_default()
}

fn active_remote_origin(state: &DesktopShellState) -> String {
    if tauri::is_dev() {
        if !state.config.dev_origin.trim().is_empty() {
            return state.config.dev_origin.clone();
        }
    }

    let runtime = state.snapshot();
    if runtime.remote_origin.trim().is_empty() {
        state.config.remote_origin.clone()
    } else {
        runtime.remote_origin
    }
}

fn remote_url(app: &AppHandle<Wry>, route: &str) -> Result<Url> {
    let state = app.state::<DesktopShellState>();
    let origin = active_remote_origin(state.inner());
    let normalized_route = normalize_remote_route(Some(route));
    Url::parse(&format!("{origin}{normalized_route}")).context("failed to construct remote app URL")
}

fn updater_endpoint_with_channel(base: &str, channel: &str) -> Result<Url> {
    let mut url = Url::parse(base).context("failed to parse updater endpoint")?;
    url.query_pairs_mut().append_pair("channel", channel);
    Ok(url)
}

fn emit_state(app: &AppHandle<Wry>, state: &DesktopRuntimeState) -> Result<()> {
    app.emit(DESKTOP_STATE_CHANGED_EVENT, state)
        .context("failed to emit desktop runtime state")
}

fn maybe_notify_native(app: &AppHandle<Wry>, shared: &DesktopShellState, state: &DesktopRuntimeState) -> Result<()> {
    if !state.notifications_enabled {
        return Ok(());
    }

    let next_notification = state
        .recent_notifications
        .iter()
        .find(|notification| !notification.read)
        .cloned();
    let Some(notification) = next_notification else {
        return Ok(());
    };

    let mut last_notification_id = shared
        .last_native_notification_id
        .lock()
        .expect("desktop notification lock poisoned");
    if last_notification_id.as_deref() == Some(notification.id.as_str()) {
        return Ok(());
    }

    let body_text = if notification.body.is_empty() {
        "Open Xcelsior Desktop to review this event.".to_string()
    } else {
        notification.body.clone()
    };

    // Use notify-rust directly for desktop notification click handling.
    // The tauri-plugin-notification 2.x desktop API doesn't support click events.
    let mut n = notify_rust::Notification::new();
    n.summary(&notification.title)
        .body(&body_text)
        .auto_icon();

    #[cfg(target_os = "linux")]
    let action_url = notification.action_url.clone();
    #[cfg(target_os = "linux")]
    let click_app = app.clone();
    #[cfg(target_os = "linux")]
    if !action_url.is_empty() {
        n.action("open", "Open in Xcelsior");
    }

    std::thread::spawn(move || {
        match n.show() {
            #[cfg(target_os = "linux")]
            Ok(handle) => {
                handle.wait_for_action(|action| {
                    if (action == "open" || action == "default") && !action_url.is_empty() {
                        let _ = open_remote_route(&click_app, Some(&action_url));
                    } else if action == "default" {
                        let _ = show_control_center_window(&click_app, None);
                    }
                });
            }
            #[cfg(not(target_os = "linux"))]
            Ok(_) => {}
            Err(e) => {
                eprintln!("failed to show native notification: {e}");
            }
        }
    });

    *last_notification_id = Some(notification.id);
    Ok(())
}

/// Notification click handling is now inline in `maybe_notify_native` via
/// `notify-rust`'s `wait_for_action` callback.
fn setup_notification_click_handler(_app: &AppHandle<Wry>) -> Result<()> {
    Ok(())
}

fn refresh_tray(app: &AppHandle<Wry>, state: &DesktopRuntimeState) {
    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        let tooltip = if state.critical_alert_count > 0 {
            format!(
                "Xcelsior Desktop • {} unread • {} critical",
                state.unread_count, state.critical_alert_count
            )
        } else {
            format!("Xcelsior Desktop • {} unread", state.unread_count)
        };
        let _ = tray.set_tooltip(Some(tooltip));
        let _ = tray.set_title(Some(state.unread_count.to_string()));
    }
}

fn apply_runtime_state(app: &AppHandle<Wry>, state: DesktopRuntimeState) -> Result<DesktopRuntimeState> {
    let shared = app.state::<DesktopShellState>();
    {
        let mut runtime = shared
            .runtime
            .lock()
            .expect("desktop runtime lock poisoned");
        *runtime = state.clone();
    }

    maybe_notify_native(app, shared.inner(), &state)?;
    shared.persist()?;
    refresh_tray(app, &state);
    emit_state(app, &state)?;
    Ok(state)
}

fn show_control_center_window(app: &AppHandle<Wry>, route: Option<&str>) -> Result<()> {
    let next_state = {
        let shared = app.state::<DesktopShellState>();
        let fallback = {
            let current = shared.snapshot();
            current.default_desktop_route
        };
        shared.update(|runtime| {
            runtime.current_desktop_route = normalize_desktop_route(route, &fallback);
        })
    };

    let window = app
        .get_webview_window(CONTROL_CENTER_WINDOW_LABEL)
        .ok_or_else(|| anyhow!("control center window is not available"))?;
    let _ = window.show();
    let _ = window.unminimize();
    let _ = window.set_focus();
    apply_runtime_state(app, next_state)?;
    Ok(())
}

fn attach_hide_to_tray_behavior(window: WebviewWindow<Wry>) {
    let app = window.app_handle().clone();
    let label = window.label().to_string();
    window.on_window_event(move |event| {
        if let WindowEvent::CloseRequested { api, .. } = event {
            let state = app.state::<DesktopShellState>();
            if state.is_quitting.load(Ordering::SeqCst) {
                return;
            }

            let should_hide = state.snapshot().hide_to_tray;
            if should_hide {
                api.prevent_close();
                if let Some(window) = app.get_webview_window(&label) {
                    let _ = window.hide();
                }
            }
        }
    });
}

fn focus_window(app: &AppHandle<Wry>, label: &str) -> bool {
    let Some(window) = app.get_webview_window(label) else {
        return false;
    };

    let _ = window.show();
    let _ = window.unminimize();
    let _ = window.set_focus();
    true
}

fn hide_window(app: &AppHandle<Wry>, label: &str) -> bool {
    let Some(window) = app.get_webview_window(label) else {
        return false;
    };

    let _ = window.hide();
    true
}

fn ensure_main_window(app: &AppHandle<Wry>, route: &str) -> Result<WebviewWindow<Wry>> {
    let url = remote_url(app, route)?;
    if let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) {
        let _ = window.navigate(url);
        let _ = focus_window(app, MAIN_WINDOW_LABEL);
        return Ok(window);
    }

    let mut builder = WebviewWindowBuilder::new(app, MAIN_WINDOW_LABEL, WebviewUrl::External(url));
    builder = builder
        .title("Xcelsior")
        .center()
        .resizable(true)
        .inner_size(1600.0, 980.0)
        .min_inner_size(1100.0, 720.0)
        .visible(true)
        .focused(true);

    let window = builder
        .build()
        .context("failed to create the shared app window")?;
    attach_hide_to_tray_behavior(window.clone());
    Ok(window)
}

fn open_remote_route(app: &AppHandle<Wry>, route: Option<&str>) -> Result<()> {
    let normalized_route = normalize_remote_route(route);
    let next_state = {
        let shared = app.state::<DesktopShellState>();
        shared.update(|runtime| {
            runtime.last_remote_route = normalized_route.clone();
            runtime.auth_required = normalized_route == "/login" || normalized_route == "/register";
        })
    };

    ensure_main_window(app, &normalized_route)?;
    apply_runtime_state(app, next_state)?;
    Ok(())
}

fn handle_deep_link(app: &AppHandle<Wry>, url: &Url) -> Result<()> {
    let target = deep_link_target(url);
    let serialized = url.as_str().to_string();

    let next_state = {
        let shared = app.state::<DesktopShellState>();
        shared.update(|runtime| {
            if !runtime.pending_deep_links.contains(&serialized) {
                runtime.pending_deep_links.insert(0, serialized.clone());
                runtime.pending_deep_links.truncate(20);
            }
        })
    };
    apply_runtime_state(app, next_state)?;

    match target {
        DeepLinkTarget::Desktop(route) => show_control_center_window(app, Some(&route)),
        DeepLinkTarget::Remote(route) => open_remote_route(app, Some(&route)),
    }
}

fn handle_tray_menu(app: &AppHandle<Wry>, id: &str) {
    let _ = match id {
        TRAY_OPEN_CONTROL_CENTER => show_control_center_window(app, None),
        TRAY_OPEN_MAIN => {
            let route = app.state::<DesktopShellState>().snapshot().last_remote_route;
            open_remote_route(app, Some(&route))
        }
        TRAY_OPEN_MARKETPLACE => open_remote_route(app, Some("/dashboard/marketplace")),
        TRAY_OPEN_INSTANCES => open_remote_route(app, Some("/dashboard/instances")),
        TRAY_OPEN_NOTIFICATIONS => open_remote_route(app, Some("/dashboard/notifications")),
        TRAY_QUIT => {
            let shared = app.state::<DesktopShellState>();
            shared.is_quitting.store(true, Ordering::SeqCst);
            app.exit(0);
            Ok(())
        }
        _ => Ok(()),
    };
}

fn build_tray(app: &AppHandle<Wry>) -> Result<()> {
    let open_control_center = MenuItemBuilder::with_id(TRAY_OPEN_CONTROL_CENTER, "Open Control Center")
        .build(app)?;
    let open_main = MenuItemBuilder::with_id(TRAY_OPEN_MAIN, "Open Main App").build(app)?;
    let open_marketplace = MenuItemBuilder::with_id(TRAY_OPEN_MARKETPLACE, "Marketplace").build(app)?;
    let open_instances = MenuItemBuilder::with_id(TRAY_OPEN_INSTANCES, "Instances").build(app)?;
    let open_notifications =
        MenuItemBuilder::with_id(TRAY_OPEN_NOTIFICATIONS, "Notifications").build(app)?;
    let quit = MenuItemBuilder::with_id(TRAY_QUIT, "Quit Desktop").build(app)?;

    let menu = MenuBuilder::new(app)
        .item(&open_control_center)
        .item(&open_main)
        .separator()
        .item(&open_marketplace)
        .item(&open_instances)
        .item(&open_notifications)
        .separator()
        .item(&quit)
        .build()?;

    let icon = app
        .default_window_icon()
        .cloned()
        .ok_or_else(|| anyhow!("desktop tray icon is not configured"))?;

    TrayIconBuilder::with_id(TRAY_ID)
        .icon(icon)
        .menu(&menu)
        .tooltip("Xcelsior Desktop")
        .show_menu_on_left_click(false)
        .on_menu_event(move |app, event| {
            handle_tray_menu(app, event.id().0.as_ref());
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                let _ = show_control_center_window(&app, None);
            }
        })
        .build(app)
        .context("failed to create tray icon")?;

    let next_state = {
        let shared = app.state::<DesktopShellState>();
        shared.update(|runtime| {
            runtime.tray_connected = true;
        })
    };
    apply_runtime_state(app, next_state)?;
    Ok(())
}

fn initialize_runtime_state(app: &AppHandle<Wry>, config: &DesktopShellConfig, store: Arc<Store<Wry>>) -> Result<DesktopRuntimeState> {
    let mut runtime = DesktopRuntimeState::new(
        config.remote_origin.clone(),
        config.dev_origin.clone(),
        app.package_info().version.to_string(),
    );
    let stored = load_stored_state(&store);

    runtime.last_remote_route = normalize_remote_route(Some(&stored.last_remote_route));
    runtime.current_desktop_route =
        normalize_desktop_route(Some(&stored.current_desktop_route), DEFAULT_DESKTOP_ROUTE);
    runtime.hide_to_tray = stored.hide_to_tray;
    runtime.notifications_enabled = stored.notifications_enabled;
    runtime.default_desktop_route =
        normalize_desktop_route(Some(&stored.default_desktop_route), DEFAULT_DESKTOP_ROUTE);
    runtime.updater_channel = if stored.updater_channel == "beta" {
        "beta".to_string()
    } else {
        "stable".to_string()
    };
    runtime.unread_count = stored.unread_count;
    runtime.critical_alert_count = stored.critical_alert_count;
    runtime.auth_required = stored.auth_required;
    runtime.pending_deep_links = stored.pending_deep_links;
    runtime.recent_notifications = stored.recent_notifications;
    runtime.autostart_enabled = app
        .autolaunch()
        .is_enabled()
        .context("failed to read desktop autostart state")?;

    Ok(runtime)
}

#[tauri::command]
fn desktop_get_state(state: State<'_, DesktopShellState>) -> DesktopRuntimeState {
    state.snapshot()
}

#[tauri::command]
fn desktop_update_preferences(
    app: AppHandle<Wry>,
    state: State<'_, DesktopShellState>,
    updates: DesktopPreferencesUpdate,
) -> Result<DesktopRuntimeState, String> {
    if let Some(launch_on_login) = updates.launch_on_login {
        if launch_on_login {
            app.autolaunch().enable().map_err(|error| error.to_string())?;
        } else {
            app.autolaunch().disable().map_err(|error| error.to_string())?;
        }
    }

    let next_state = state.update(|runtime| {
        if let Some(hide_to_tray) = updates.hide_to_tray {
            runtime.hide_to_tray = hide_to_tray;
        }
        if let Some(notifications_enabled) = updates.notifications_enabled {
            runtime.notifications_enabled = notifications_enabled;
        }
        if let Some(default_desktop_route) = updates.default_desktop_route.as_deref() {
            runtime.default_desktop_route =
                normalize_desktop_route(Some(default_desktop_route), DEFAULT_DESKTOP_ROUTE);
        }
        if let Some(current_desktop_route) = updates.current_desktop_route.as_deref() {
            runtime.current_desktop_route =
                normalize_desktop_route(Some(current_desktop_route), &runtime.default_desktop_route);
        }
        if let Some(updater_channel) = updates.updater_channel.as_deref() {
            runtime.updater_channel = if updater_channel == "beta" {
                "beta".to_string()
            } else {
                "stable".to_string()
            };
        }
        runtime.autostart_enabled = app.autolaunch().is_enabled().unwrap_or(false);
    });

    apply_runtime_state(&app, next_state).map_err(|error| error.to_string())
}

#[tauri::command]
fn desktop_sync_remote_state(
    app: AppHandle<Wry>,
    state: State<'_, DesktopShellState>,
    payload: DesktopRemoteStateSyncPayload,
) -> Result<DesktopRuntimeState, String> {
    let next_state = state.update(|runtime| {
        if let Some(is_online) = payload.is_online {
            runtime.is_online = is_online;
        }
        if let Some(last_remote_route) = payload.last_remote_route.as_deref() {
            runtime.last_remote_route = normalize_remote_route(Some(last_remote_route));
        }
        if let Some(unread_count) = payload.unread_count {
            runtime.unread_count = unread_count.max(0);
        }
        if let Some(critical_alert_count) = payload.critical_alert_count {
            runtime.critical_alert_count = critical_alert_count.max(0);
        }
        if let Some(auth_required) = payload.auth_required {
            runtime.auth_required = auth_required;
        }
        if let Some(recent_notifications) = payload.recent_notifications {
            runtime.recent_notifications = recent_notifications.into_iter().take(8).collect();
        }
    });

    apply_runtime_state(&app, next_state).map_err(|error| error.to_string())
}

#[tauri::command]
fn desktop_show_control_center(app: AppHandle<Wry>, route: Option<String>) -> Result<(), String> {
    show_control_center_window(&app, route.as_deref()).map_err(|error| error.to_string())
}

#[tauri::command]
fn desktop_open_main_window(app: AppHandle<Wry>, route: Option<String>) -> Result<(), String> {
    open_remote_route(&app, route.as_deref()).map_err(|error| error.to_string())
}

#[tauri::command]
async fn desktop_check_for_updates(
    app: AppHandle<Wry>,
    state: State<'_, DesktopShellState>,
) -> Result<DesktopRuntimeState, String> {
    let channel = state.snapshot().updater_channel;
    let endpoint = updater_endpoint_with_channel(&state.config.updater_endpoint, &channel)
        .map_err(|error| error.to_string())?;
    let updater = app
        .updater_builder()
        .endpoints(vec![endpoint])
        .map_err(|error| error.to_string())?
        .build()
        .map_err(|error| error.to_string())?;

    let update = updater.check().await.map_err(|error| error.to_string())?;
    let next_state = state.update(|runtime| {
        runtime.update_available = update.is_some();
        runtime.update_version = update.as_ref().map(|release| release.version.to_string());
    });
    apply_runtime_state(&app, next_state).map_err(|error| error.to_string())
}

#[tauri::command]
async fn desktop_install_update(
    app: AppHandle<Wry>,
    state: State<'_, DesktopShellState>,
) -> Result<DesktopRuntimeState, String> {
    let channel = state.snapshot().updater_channel;
    let endpoint = updater_endpoint_with_channel(&state.config.updater_endpoint, &channel)
        .map_err(|error| error.to_string())?;
    let updater = app
        .updater_builder()
        .endpoints(vec![endpoint])
        .map_err(|error| error.to_string())?
        .build()
        .map_err(|error| error.to_string())?;

    if let Some(update) = updater.check().await.map_err(|error| error.to_string())? {
        update
            .download_and_install(|_, _| {}, || {})
            .await
            .map_err(|error| error.to_string())?;
    }

    let next_state = state.update(|runtime| {
        runtime.update_available = false;
        runtime.update_version = None;
    });
    apply_runtime_state(&app, next_state).map_err(|error| error.to_string())
}

#[tauri::command]
fn desktop_open_external_url(app: AppHandle<Wry>, url: String) -> Result<(), String> {
    let parsed = Url::parse(&url).map_err(|error| error.to_string())?;
    match parsed.scheme() {
        "https" | "http" | "mailto" => app
            .opener()
            .open_url(url, None::<&str>)
            .map_err(|error| error.to_string()),
        _ => Err("unsupported external URL scheme".to_string()),
    }
}

#[tauri::command]
fn desktop_clear_deep_links(
    app: AppHandle<Wry>,
    state: State<'_, DesktopShellState>,
) -> Result<DesktopRuntimeState, String> {
    let next_state = state.update(|runtime| {
        runtime.pending_deep_links.clear();
    });
    apply_runtime_state(&app, next_state).map_err(|error| error.to_string())
}

fn setup_shell(app: &AppHandle<Wry>, launch_mode: LaunchMode) -> Result<()> {
    let control_center = app
        .get_webview_window(CONTROL_CENTER_WINDOW_LABEL)
        .ok_or_else(|| anyhow!("control center window is not available"))?;
    attach_hide_to_tray_behavior(control_center.clone());
    build_tray(app)?;
    setup_notification_click_handler(app)?;

    let mut had_initial_deep_links = false;
    if let Ok(Some(initial_urls)) = app.deep_link().get_current() {
        had_initial_deep_links = !initial_urls.is_empty();
        for url in initial_urls {
            handle_deep_link(app, &url)?;
        }
    }

    let deep_link_app = app.clone();
    app.deep_link().on_open_url(move |event| {
        for url in event.urls() {
            let _ = handle_deep_link(&deep_link_app, &url);
        }
    });

    match (launch_mode, had_initial_deep_links) {
        (LaunchMode::Minimized, false) => {
            let _ = hide_window(app, CONTROL_CENTER_WINDOW_LABEL);
        }
        (LaunchMode::Normal, false) => {
            let last_route = app.state::<DesktopShellState>().snapshot().last_remote_route;
            open_remote_route(app, Some(&last_route))?;
        }
        _ => {}
    }

    Ok(())
}

fn handle_second_instance(app: &AppHandle<Wry>) {
    if focus_window(app, MAIN_WINDOW_LABEL) {
        return;
    }

    let current_route = app.state::<DesktopShellState>().snapshot().current_desktop_route;
    let _ = show_control_center_window(app, Some(&current_route));
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let shell_config = load_shell_config().expect("failed to load desktop shell config");
    let single_instance_dbus_id = shell_config.bundle_id.clone();
    let launch_mode = launch_mode_from_env();
    let context = tauri::generate_context!();

    tauri::Builder::default()
        .plugin(
            tauri_plugin_single_instance::Builder::new()
                .dbus_id(single_instance_dbus_id)
                .callback(|app, _args, _cwd| {
                    handle_second_instance(app);
                })
                .build(),
        )
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_opener::Builder::new().build())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_deep_link::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            Some(vec!["--minimized"]),
        ))
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(move |app| {
            let store = StoreBuilder::new(app, STORE_PATH).build()?;
            let runtime_state = initialize_runtime_state(app.handle(), &shell_config, store.clone())?;
            let last_native_notification_id = load_stored_state(&store).last_native_notification_id;

            app.manage(DesktopShellState {
                runtime: Mutex::new(runtime_state.clone()),
                config: shell_config.clone(),
                store,
                is_quitting: AtomicBool::new(false),
                last_native_notification_id: Mutex::new(last_native_notification_id),
            });

            setup_shell(app.handle(), launch_mode)?;
            let current_state = app.state::<DesktopShellState>().snapshot();
            emit_state(app.handle(), &current_state)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            desktop_get_state,
            desktop_update_preferences,
            desktop_sync_remote_state,
            desktop_show_control_center,
            desktop_open_main_window,
            desktop_check_for_updates,
            desktop_install_update,
            desktop_open_external_url,
            desktop_clear_deep_links
        ])
        .build(context)
        .expect("error while running Xcelsior Desktop")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { api, .. } = event {
                let state = app.state::<DesktopShellState>();
                if !state.is_quitting.load(Ordering::SeqCst) {
                    let should_hide = state.snapshot().hide_to_tray;
                    if should_hide {
                        api.prevent_exit();
                        for window in app.webview_windows().values() {
                            let _ = window.hide();
                        }
                    }
                }
            }
        });
}
