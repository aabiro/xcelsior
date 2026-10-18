use anyhow::{Context, Result};
use serde::Deserialize;
use url::Url;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DesktopShellConfig {
    pub remote_origin: String,
    pub dev_origin: String,
    pub control_center_dev_url: String,
    pub updater_endpoint: String,
    pub bundle_id: String,
    pub deep_link_scheme: String,
    pub bundle_targets: Vec<String>,
}

impl DesktopShellConfig {
    fn validate(&self) -> Result<()> {
        Url::parse(&self.remote_origin).context("desktop remoteOrigin must be a valid URL")?;
        Url::parse(&self.dev_origin).context("desktop devOrigin must be a valid URL")?;
        Url::parse(&self.control_center_dev_url)
            .context("desktop controlCenterDevUrl must be a valid URL")?;

        if self.bundle_id.trim().is_empty() {
            anyhow::bail!("desktop bundleId must not be empty");
        }
        if self.deep_link_scheme.trim().is_empty() {
            anyhow::bail!("desktop deepLinkScheme must not be empty");
        }
        if self.bundle_targets.is_empty() {
            anyhow::bail!("desktop bundleTargets must not be empty");
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct TauriConfigFile {
    plugins: PluginSection,
}

#[derive(Debug, Deserialize)]
struct PluginSection {
    xcelsior: DesktopShellConfig,
}

pub fn load_shell_config() -> Result<DesktopShellConfig> {
    let config: TauriConfigFile = serde_json::from_str(include_str!("../tauri.conf.json"))
        .context("failed to parse desktop tauri.conf.json")?;
    config.plugins.xcelsior.validate()?;
    Ok(config.plugins.xcelsior)
}

#[cfg(test)]
mod tests {
    use super::load_shell_config;

    #[test]
    fn loads_xcelsior_plugin_config() {
        let config = load_shell_config().expect("desktop config should parse");
        assert_eq!(config.remote_origin, "https://xcelsior.ca");
        assert_eq!(config.dev_origin, "http://localhost:3000");
        assert_eq!(config.deep_link_scheme, "xcelsior");
        assert!(config.bundle_targets.iter().any(|target| target == "msi"));
    }
}
