import type { NextConfig } from "next";
import bundleAnalyzer from "@next/bundle-analyzer";

const withBundleAnalyzer = bundleAnalyzer({
  enabled: process.env.ANALYZE === "true",
});

// INTERNAL_API_URL: server-side only (rewrites/SSR), never shipped to browser.
// Falls back to NEXT_PUBLIC_API_URL (baked into client JS at build time).
const BACKEND = process.env.INTERNAL_API_URL || process.env.NEXT_PUBLIC_API_URL || "https://xcelsior.ca";
const PUBLIC_ASSET_CACHE = "public, max-age=3600, stale-while-revalidate=86400";
const CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  "script-src 'self' https://www.googletagmanager.com https://www.google-analytics.com https://js.stripe.com https://*.js.stripe.com https://connect-js.stripe.com https://hcaptcha.com https://*.hcaptcha.com https://static.cloudflareinsights.com https://connect.facebook.net 'unsafe-inline'" +
    (process.env.NODE_ENV === "development" ? " 'unsafe-eval'" : ""),
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob: https:",
  "font-src 'self' data:",
  "connect-src 'self' https://www.google-analytics.com https://*.google-analytics.com https://*.analytics.google.com https://www.googletagmanager.com https://*.googletagmanager.com https://www.google.com https://static.cloudflareinsights.com wss://xcelsior.ca https://api.web3modal.org https://*.walletconnect.org wss://relay.walletconnect.org https://pulse.walletconnect.org https://api.stripe.com https://js.stripe.com https://*.js.stripe.com https://connect-js.stripe.com https://hooks.stripe.com https://m.stripe.network https://r.stripe.com https://q.stripe.com https://merchant-ui-api.stripe.com https://link.com https://*.link.com https://hcaptcha.com https://*.hcaptcha.com https://us.i.posthog.com https://us-assets.i.posthog.com https://connect.facebook.net https://*.facebook.net https://www.facebook.com https://*.facebook.com https://md-eecad2978f7a43f5b7838c919258e6de.ecs.us-east-2.on.aws https://cvw425zxroht.cloud.run",
  "frame-src 'self' https://js.stripe.com https://*.js.stripe.com https://connect-js.stripe.com https://hooks.stripe.com https://m.stripe.network https://link.com https://*.link.com https://hcaptcha.com https://*.hcaptcha.com https://verify.walletconnect.org",
  "frame-ancestors 'self'",
].join("; ");

const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
  { key: "Content-Security-Policy", value: CONTENT_SECURITY_POLICY },
];

const nextConfig: NextConfig = {
  output: "standalone",
  compress: true,
  poweredByHeader: false,
  typescript: {
    // Type-checking runs in CI; skip in Docker builds where Fern SDK .js imports
    // fail under Node's module resolution despite working locally with bundler.
    ignoreBuildErrors: !!process.env.DOCKER_BUILD,
  },
  async headers() {
    return [
      { source: "/(.*)", headers: securityHeaders },
      {
        source: "/sw.js",
        headers: [
          { key: "Content-Type", value: "application/javascript; charset=utf-8" },
          { key: "Cache-Control", value: "no-cache, no-store, must-revalidate" },
          { key: "Content-Security-Policy", value: "default-src 'self'; script-src 'self'" },
        ],
      },
      {
        source: "/manifest.webmanifest",
        headers: [{ key: "Cache-Control", value: "public, max-age=0, must-revalidate" }],
      },
      // Files in /public keep their names when they are redrawn, so they must not be
      // `immutable`: a browser (and the CDN) would keep the old art for a year. An hour
      // fresh, then revalidate in the background. Hashed build output is below.
      {
        source: "/:path*.svg",
        headers: [{ key: "Cache-Control", value: PUBLIC_ASSET_CACHE }],
      },
      {
        source: "/:path*.png",
        headers: [{ key: "Cache-Control", value: PUBLIC_ASSET_CACHE }],
      },
      {
        source: "/:path*.woff2",
        headers: [{ key: "Cache-Control", value: "public, max-age=31536000, immutable" }],
      },
      {
        source: "/_next/static/:path*",
        headers: [{ key: "Cache-Control", value: "public, max-age=31536000, immutable" }],
      },
    ];
  },
  async redirects() {
    return [
      {
        source: "/blog/mcp-server-release.png",
        destination: "/blog/mcp-server-release/image",
        permanent: true,
      },
      {
        // An endpoint's page is /dashboard/inference?endpoint=<id>; Quick Deploy
        // and the endpoint sidebar used to link to this path, a 404 (prod,
        // 2026-11-22). The [id] page's redirect() only fires once the client has
        // rendered through the auth shell and the serverless gate, so it could
        // sit on a blank page; this answers with a 307 before anything renders.
        // `new` is the deploy studio, a real route.
        source: "/dashboard/inference/:id((?!new$)[^/]+)",
        destination: "/dashboard/inference?endpoint=:id",
        permanent: false,
      },
      {
        source: "/docs",
        destination: "https://docs.xcelsior.ca",
        permanent: false,
      },
      {
        source: "/docs/:path*",
        destination: "https://docs.xcelsior.ca/:path*",
        permanent: false,
      },
    ];
  },
  skipTrailingSlashRedirect: true,
  async rewrites() {
    return [
      {
        source: "/ingest/static/:path*",
        destination: "https://us-assets.i.posthog.com/static/:path*",
      },
      {
        source: "/ingest/array/:path*",
        destination: "https://us-assets.i.posthog.com/array/:path*",
      },
      {
        source: "/ingest/:path*",
        destination: "https://us.i.posthog.com/:path*",
      },
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
      { source: "/oauth/:path*", destination: `${BACKEND}/oauth/:path*` },
      { source: "/.well-known/:path*", destination: `${BACKEND}/.well-known/:path*` },
      { source: "/hosts", destination: `${BACKEND}/hosts` },
      { source: "/host", destination: `${BACKEND}/host` },
      { source: "/host/:path*", destination: `${BACKEND}/host/:path*` },
      { source: "/jobs", destination: `${BACKEND}/jobs` },
      { source: "/jobs/:path*", destination: `${BACKEND}/jobs/:path*` },
      { source: "/job", destination: `${BACKEND}/job` },
      { source: "/job/:path*", destination: `${BACKEND}/job/:path*` },
      { source: "/instance", destination: `${BACKEND}/instance` },
      { source: "/instance/:path*", destination: `${BACKEND}/instance/:path*` },
      { source: "/instances/:path*", destination: `${BACKEND}/instances/:path*` },
      { source: "/marketplace", destination: `${BACKEND}/marketplace` },
      { source: "/marketplace/:path*", destination: `${BACKEND}/marketplace/:path*` },
      { source: "/billing", destination: `${BACKEND}/billing` },
      { source: "/spot-prices", destination: `${BACKEND}/spot-prices` },
      { source: "/spot/:path*", destination: `${BACKEND}/spot/:path*` },
      { source: "/compute-score/:path*", destination: `${BACKEND}/compute-score/:path*` },
      { source: "/compute-scores", destination: `${BACKEND}/compute-scores` },
      { source: "/healthz", destination: `${BACKEND}/healthz` },
    ];
  },
};

export default withBundleAnalyzer(nextConfig);
