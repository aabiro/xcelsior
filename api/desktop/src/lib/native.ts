import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  DESKTOP_STATE_CHANGED_EVENT,
  type DesktopRoute,
  type DesktopPreferencesUpdate,
  type DesktopRuntimeState,
} from "./contract";

export async function getDesktopState() {
  return invoke<DesktopRuntimeState>("desktop_get_state");
}

export async function listenForDesktopState(handler: (state: DesktopRuntimeState) => void) {
  const unlisten = await listen<DesktopRuntimeState>(DESKTOP_STATE_CHANGED_EVENT, (event) => {
    handler(event.payload);
  });

  return () => {
    void unlisten();
  };
}

export async function showControlCenter(route?: DesktopRoute) {
  return invoke("desktop_show_control_center", { route });
}

export async function openMainWindow(route?: string) {
  return invoke("desktop_open_main_window", { route });
}

export async function updatePreferences(updates: DesktopPreferencesUpdate) {
  return invoke<DesktopRuntimeState>("desktop_update_preferences", { updates });
}

export async function checkForUpdates() {
  return invoke<DesktopRuntimeState>("desktop_check_for_updates");
}

export async function installUpdate() {
  return invoke<DesktopRuntimeState>("desktop_install_update");
}

export async function openExternalUrl(url: string) {
  return invoke("desktop_open_external_url", { url });
}

export async function clearDeepLinks() {
  return invoke<DesktopRuntimeState>("desktop_clear_deep_links");
}
