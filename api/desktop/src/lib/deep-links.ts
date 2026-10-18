import type { DesktopRoute } from "./contract";

export type DeepLinkPreview =
  | { target: "desktop"; route: DesktopRoute }
  | { target: "remote"; route: string };

const DESKTOP_ROUTES: DesktopRoute[] = [
  "/desktop",
  "/desktop/activity",
  "/desktop/launch",
  "/desktop/settings",
  "/desktop/links",
];

function normalizeDesktopRoute(route: string | null | undefined, fallback: DesktopRoute = "/desktop"): DesktopRoute {
  const candidate = (route ?? "").trim();
  if (DESKTOP_ROUTES.includes(candidate as DesktopRoute)) {
    return candidate as DesktopRoute;
  }
  return fallback;
}

function normalizeRemoteRoute(route: string | null | undefined) {
  const candidate = (route ?? "").trim();
  if (!candidate || candidate === "/" || !candidate.startsWith("/") || candidate.startsWith("/desktop")) {
    return "/dashboard";
  }
  return candidate;
}

export function parseDesktopDeepLink(raw: string): DeepLinkPreview | null {
  try {
    const url = new URL(raw);
    if (url.protocol !== "xcelsior:") return null;

    const explicitRoute = url.searchParams.get("route");
    const query = new URLSearchParams(url.search);
    query.delete("route");
    const segments = [
      url.hostname,
      ...url.pathname.split("/").filter(Boolean),
    ].filter(Boolean);
    const route = withQuery(explicitRoute ?? `/${segments.join("/")}`, query);

    if (route.startsWith("/desktop")) {
      return {
        target: "desktop",
        route: normalizeDesktopRoute(route),
      };
    }

    return {
      target: "remote",
      route: normalizeRemoteRoute(route),
    };
  } catch {
    return null;
  }
}

function withQuery(route: string, query: URLSearchParams) {
  if (!query.toString() || route.startsWith("/desktop") || route.includes("?")) {
    return route;
  }

  return `${route}?${query.toString()}`;
}
