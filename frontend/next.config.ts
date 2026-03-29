import type { NextConfig } from "next";

const BACKEND = process.env.NEXT_PUBLIC_API_URL || "https://xcelsior.ca";

const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "X-XSS-Protection", value: "1; mode=block" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
];

const nextConfig: NextConfig = {
  output: "standalone",
  typescript: {
    // Type-checking runs in CI; skip in Docker builds where Fern SDK .js imports
    // fail under Node's module resolution despite working locally with bundler.
    ignoreBuildErrors: !!process.env.DOCKER_BUILD,
  },
  async headers() {
    return [{ source: "/(.*)", headers: securityHeaders }];
  },
  async redirects() {
    return [
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
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND}/api/:path*` },
      { source: "/hosts", destination: `${BACKEND}/hosts` },
      { source: "/host", destination: `${BACKEND}/host` },
      { source: "/host/:path*", destination: `${BACKEND}/host/:path*` },
      { source: "/jobs", destination: `${BACKEND}/jobs` },
      { source: "/jobs/:path*", destination: `${BACKEND}/jobs/:path*` },
      { source: "/job", destination: `${BACKEND}/job` },
      { source: "/job/:path*", destination: `${BACKEND}/job/:path*` },
      { source: "/instance/:path*", destination: `${BACKEND}/instance/:path*` },
      { source: "/instances/:path*", destination: `${BACKEND}/instances/:path*` },
      { source: "/marketplace", destination: `${BACKEND}/marketplace` },
      { source: "/marketplace/:path*", destination: `${BACKEND}/marketplace/:path*` },
      { source: "/billing", destination: `${BACKEND}/billing` },
      { source: "/spot-prices", destination: `${BACKEND}/spot-prices` },
      { source: "/compute-score/:path*", destination: `${BACKEND}/compute-score/:path*` },
      { source: "/compute-scores", destination: `${BACKEND}/compute-scores` },
      { source: "/healthz", destination: `${BACKEND}/healthz` },
    ];
  },
};

export default nextConfig;
