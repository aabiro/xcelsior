import type { RuntimeCaching } from "serwist";
import { CacheFirst, ExpirationPlugin, NetworkFirst, NetworkOnly, RangeRequestsPlugin, StaleWhileRevalidate } from "serwist";

const STATIC_ASSET_MAX_AGE = 24 * 60 * 60;
const SHORT_NETWORK_TIMEOUT_SECONDS = 3;

const PUBLIC_DATA_PATHS = new Set([
  "/api/images/templates",
  "/api/pricing/reference",
  "/compute-scores",
  "/healthz",
  "/marketplace",
  "/spot-prices",
]);

const PUBLIC_DATA_PREFIXES = [
  "/compute-score/",
  "/marketplace/search",
];

const SENSITIVE_PATH_PREFIXES = [
  "/api/admin",
  "/api/alerts",
  "/api/analytics/",
  "/api/artifacts/",
  "/api/auth/",
  "/api/billing/",
  "/api/chat/",
  "/api/compliance/",
  "/api/connect/",
  "/api/events",
  "/api/keys",
  "/api/notifications",
  "/api/providers/",
  "/api/reputation/",
  "/api/sla/",
  "/api/slurm/",
  "/api/ssh/",
  "/api/teams",
  "/api/telemetry/",
  "/api/users/",
  "/api/verify/",
  "/billing",
  "/host",
  "/hosts",
  "/instance",
  "/instances",
  "/job",
  "/jobs",
  "/spot/instance",
];

export function isPublicDataPath(pathname: string) {
  return PUBLIC_DATA_PATHS.has(pathname) || PUBLIC_DATA_PREFIXES.some((prefix) => pathname.startsWith(prefix));
}

export function isSensitiveDataPath(pathname: string) {
  return SENSITIVE_PATH_PREFIXES.some((prefix) => pathname.startsWith(prefix));
}

export function isEventStreamRequest(request: Request, pathname: string) {
  const accept = request.headers.get("accept") ?? "";
  return accept.includes("text/event-stream") || pathname.endsWith("/stream");
}

export const desktopRuntimeCaching: RuntimeCaching[] =
  process.env.NODE_ENV !== "production"
    ? [
      {
        matcher: /.*/i,
        handler: new NetworkOnly(),
      },
    ]
    : [
      // Ensure /oauth/callback is always network-only (never cached or precached)
      {
        matcher: ({ sameOrigin, url }) => sameOrigin && url.pathname.startsWith("/oauth/callback"),
        method: "GET",
        handler: new NetworkOnly(),
      },
      {
        matcher: /^https:\/\/fonts\.(?:gstatic)\.com\/.*/i,
        handler: new CacheFirst({
          cacheName: "google-fonts-webfonts",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 4,
              maxAgeSeconds: 365 * 24 * 60 * 60,
              maxAgeFrom: "last-used",
            }),
          ],
        }),
      },
      {
        matcher: /^https:\/\/fonts\.(?:googleapis)\.com\/.*/i,
        handler: new StaleWhileRevalidate({
          cacheName: "google-fonts-stylesheets",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 4,
              maxAgeSeconds: 7 * 24 * 60 * 60,
              maxAgeFrom: "last-used",
            }),
          ],
        }),
      },
      {
        matcher: ({ sameOrigin, url }) => sameOrigin && /\.(?:eot|otf|ttc|ttf|woff|woff2|font\.css)$/i.test(url.pathname),
        handler: new StaleWhileRevalidate({
          cacheName: "static-font-assets",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 4,
              maxAgeSeconds: 7 * 24 * 60 * 60,
              maxAgeFrom: "last-used",
            }),
          ],
        }),
      },
      {
        matcher: ({ sameOrigin, url }) => sameOrigin && /\.(?:jpg|jpeg|gif|png|svg|ico|webp)$/i.test(url.pathname),
        handler: new StaleWhileRevalidate({
          cacheName: "static-image-assets",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 64,
              maxAgeSeconds: 30 * 24 * 60 * 60,
              maxAgeFrom: "last-used",
            }),
          ],
        }),
      },
      {
        matcher: ({ sameOrigin, url }) => sameOrigin && /\/_next\/static.+\.js$/i.test(url.pathname),
        handler: new CacheFirst({
          cacheName: "next-static-js-assets",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 64,
              maxAgeSeconds: STATIC_ASSET_MAX_AGE,
              maxAgeFrom: "last-used",
            }),
          ],
        }),
      },
      {
        matcher: ({ sameOrigin, url }) => sameOrigin && url.pathname === "/_next/image",
        handler: new StaleWhileRevalidate({
          cacheName: "next-image",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 64,
              maxAgeSeconds: STATIC_ASSET_MAX_AGE,
              maxAgeFrom: "last-used",
            }),
          ],
        }),
      },
      {
        matcher: ({ sameOrigin, url }) => sameOrigin && /\.(?:mp3|wav|ogg)$/i.test(url.pathname),
        handler: new CacheFirst({
          cacheName: "static-audio-assets",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 32,
              maxAgeSeconds: STATIC_ASSET_MAX_AGE,
              maxAgeFrom: "last-used",
            }),
            new RangeRequestsPlugin(),
          ],
        }),
      },
      {
        matcher: ({ sameOrigin, url }) => sameOrigin && /\.(?:mp4|webm)$/i.test(url.pathname),
        handler: new CacheFirst({
          cacheName: "static-video-assets",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 32,
              maxAgeSeconds: STATIC_ASSET_MAX_AGE,
              maxAgeFrom: "last-used",
            }),
            new RangeRequestsPlugin(),
          ],
        }),
      },
      {
        matcher: ({ sameOrigin, url }) => sameOrigin && /\.(?:js)$/i.test(url.pathname),
        handler: new StaleWhileRevalidate({
          cacheName: "static-js-assets",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 48,
              maxAgeSeconds: STATIC_ASSET_MAX_AGE,
              maxAgeFrom: "last-used",
            }),
          ],
        }),
      },
      {
        matcher: ({ sameOrigin, url }) => sameOrigin && /\.(?:css|less)$/i.test(url.pathname),
        handler: new StaleWhileRevalidate({
          cacheName: "static-style-assets",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 32,
              maxAgeSeconds: STATIC_ASSET_MAX_AGE,
              maxAgeFrom: "last-used",
            }),
          ],
        }),
      },
      {
        matcher: ({ sameOrigin, url }) => sameOrigin && /\/_next\/data\/.+\/.+\.json$/i.test(url.pathname),
        handler: new NetworkFirst({
          cacheName: "next-data",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 32,
              maxAgeSeconds: STATIC_ASSET_MAX_AGE,
              maxAgeFrom: "last-used",
            }),
          ],
        }),
      },
      {
        matcher: ({ request, sameOrigin, url }) => sameOrigin && isEventStreamRequest(request, url.pathname),
        method: "GET",
        handler: new NetworkOnly(),
      },
      {
        matcher: ({ sameOrigin, url }) => sameOrigin && isSensitiveDataPath(url.pathname),
        method: "GET",
        handler: new NetworkOnly(),
      },
      {
        matcher: ({ sameOrigin, url }) => sameOrigin && isPublicDataPath(url.pathname),
        method: "GET",
        handler: new NetworkFirst({
          cacheName: "public-read-models",
          networkTimeoutSeconds: SHORT_NETWORK_TIMEOUT_SECONDS,
          plugins: [
            new ExpirationPlugin({
              maxEntries: 24,
              maxAgeSeconds: 10 * 60,
              maxAgeFrom: "last-used",
            }),
          ],
        }),
      },
      {
        matcher: ({ request, sameOrigin, url }) =>
          request.headers.get("RSC") === "1" &&
          request.headers.get("Next-Router-Prefetch") === "1" &&
          sameOrigin &&
          !url.pathname.startsWith("/api/"),
        handler: new NetworkFirst({
          cacheName: "pages-rsc-prefetch",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 32,
              maxAgeSeconds: STATIC_ASSET_MAX_AGE,
            }),
          ],
        }),
      },
      {
        matcher: ({ request, sameOrigin, url }) =>
          request.headers.get("RSC") === "1" &&
          sameOrigin &&
          !url.pathname.startsWith("/api/"),
        handler: new NetworkFirst({
          cacheName: "pages-rsc",
          plugins: [
            new ExpirationPlugin({
              maxEntries: 32,
              maxAgeSeconds: STATIC_ASSET_MAX_AGE,
            }),
          ],
        }),
      },
      {
        matcher: ({ request, sameOrigin, url }) =>
          request.mode === "navigate" &&
          sameOrigin &&
          !url.pathname.startsWith("/api/"),
        handler: new NetworkFirst({
          cacheName: "pages",
          networkTimeoutSeconds: SHORT_NETWORK_TIMEOUT_SECONDS,
          plugins: [
            new ExpirationPlugin({
              maxEntries: 32,
              maxAgeSeconds: STATIC_ASSET_MAX_AGE,
            }),
          ],
        }),
      },
      {
        matcher: ({ sameOrigin }) => sameOrigin,
        method: "GET",
        handler: new NetworkOnly(),
      },
    ];
