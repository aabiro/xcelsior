use url::Url;

use crate::contract::{DEFAULT_DESKTOP_ROUTE, DEFAULT_REMOTE_ROUTE};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeepLinkTarget {
    Desktop(String),
    Remote(String),
}

pub fn normalize_desktop_route(route: Option<&str>, fallback: &str) -> String {
    let candidate = route.unwrap_or(fallback).trim();
    let value = if candidate.is_empty() { fallback } else { candidate };
    match value {
        "/desktop" | "/desktop/activity" | "/desktop/launch" | "/desktop/settings" | "/desktop/links" => {
            value.to_string()
        }
        _ => fallback.to_string(),
    }
}

pub fn normalize_remote_route(route: Option<&str>) -> String {
    let candidate = route.unwrap_or(DEFAULT_REMOTE_ROUTE).trim();
    if candidate.is_empty() || !candidate.starts_with('/') || candidate.starts_with("/desktop") {
        return DEFAULT_REMOTE_ROUTE.to_string();
    }

    if candidate == "/" {
        return DEFAULT_REMOTE_ROUTE.to_string();
    }

    candidate.to_string()
}

pub fn deep_link_target(url: &Url) -> DeepLinkTarget {
    let route = route_from_url(url);
    if route.starts_with("/desktop") {
        DeepLinkTarget::Desktop(normalize_desktop_route(Some(&route), DEFAULT_DESKTOP_ROUTE))
    } else {
        DeepLinkTarget::Remote(normalize_remote_route(Some(&route)))
    }
}

fn route_from_url(url: &Url) -> String {
    let explicit_route = url
        .query_pairs()
        .find_map(|(key, value)| (key == "route").then_some(value.into_owned()));

    let query = url
        .query_pairs()
        .filter(|(key, _)| key != "route")
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("&");

    if let Some(route) = explicit_route {
        return append_query(route, &query);
    }

    let mut segments = Vec::new();
    if let Some(host) = url.host_str() {
        segments.push(host.trim_matches('/'));
    }
    segments.extend(url.path_segments().into_iter().flatten().filter(|segment| !segment.is_empty()));

    if segments.is_empty() {
        return DEFAULT_REMOTE_ROUTE.to_string();
    }

    append_query(format!("/{}", segments.join("/")), &query)
}

fn append_query(route: String, query: &str) -> String {
    if query.is_empty() || route.starts_with("/desktop") || route.contains('?') {
        return route;
    }

    format!("{route}?{query}")
}

#[cfg(test)]
mod tests {
    use super::{deep_link_target, normalize_desktop_route, normalize_remote_route, DeepLinkTarget};

    #[test]
    fn normalizes_remote_routes() {
        assert_eq!(normalize_remote_route(Some("/dashboard/instances")), "/dashboard/instances");
        assert_eq!(normalize_remote_route(Some("/desktop/settings")), "/dashboard");
        assert_eq!(normalize_remote_route(Some("")), "/dashboard");
        assert_eq!(normalize_remote_route(Some("dashboard")), "/dashboard");
    }

    #[test]
    fn normalizes_desktop_routes() {
        assert_eq!(normalize_desktop_route(Some("/desktop/links"), "/desktop"), "/desktop/links");
        assert_eq!(normalize_desktop_route(Some("/dashboard"), "/desktop"), "/desktop");
        assert_eq!(normalize_desktop_route(None, "/desktop"), "/desktop");
    }

    #[test]
    fn resolves_desktop_deep_links() {
        let url = url::Url::parse("xcelsior://desktop/settings").expect("valid url");
        assert_eq!(deep_link_target(&url), DeepLinkTarget::Desktop("/desktop/settings".to_string()));
    }

    #[test]
    fn resolves_remote_deep_links() {
        let url = url::Url::parse("xcelsior://dashboard/instances?tab=active").expect("valid url");
        assert_eq!(
            deep_link_target(&url),
            DeepLinkTarget::Remote("/dashboard/instances?tab=active".to_string())
        );
    }

    #[test]
    fn resolves_query_based_route_overrides() {
        let url = url::Url::parse("xcelsior://open?route=/desktop/activity&mode=debug").expect("valid url");
        assert_eq!(deep_link_target(&url), DeepLinkTarget::Desktop("/desktop/activity".to_string()));
    }

    #[test]
    fn merges_query_params_with_explicit_remote_routes() {
        let url = url::Url::parse("xcelsior://open?route=/dashboard/billing&invoice=inv_123").expect("valid url");
        assert_eq!(
            deep_link_target(&url),
            DeepLinkTarget::Remote("/dashboard/billing?invoice=inv_123".to_string())
        );
    }
}
