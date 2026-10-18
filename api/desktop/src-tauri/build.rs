fn main() {
    println!("cargo:rerun-if-changed=capabilities");
    println!("cargo:rerun-if-changed=permissions");
    println!("cargo:rerun-if-changed=tauri.conf.json");

    let attributes = tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "desktop_get_state",
            "desktop_update_preferences",
            "desktop_sync_remote_state",
            "desktop_show_control_center",
            "desktop_open_main_window",
            "desktop_check_for_updates",
            "desktop_install_update",
            "desktop_open_external_url",
            "desktop_clear_deep_links",
        ]),
    );

    tauri_build::try_build(attributes).expect("failed to build Xcelsior desktop bindings");
}
