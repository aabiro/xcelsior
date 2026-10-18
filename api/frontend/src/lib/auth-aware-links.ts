export type MarketingCtaIntent =
  | "launch"
  | "start"
  | "mcp"
  | "serverless"
  | "instances"
  | "hosting"
  | "xcelai"
  | "volumes";

const CTA_TARGETS: Record<MarketingCtaIntent, { signedOut: string; signedIn: string }> = {
  launch: { signedOut: "/register", signedIn: "/dashboard/marketplace" },
  start: { signedOut: "/register", signedIn: "/dashboard" },
  mcp: {
    signedOut: "/login?redirect=%2Fdashboard%2Fsettings%23mcp",
    signedIn: "/dashboard/settings#mcp",
  },
  serverless: { signedOut: "/register", signedIn: "/dashboard/inference" },
  instances: { signedOut: "/register", signedIn: "/dashboard/instances" },
  hosting: { signedOut: "/register", signedIn: "/dashboard/hosts" },
  xcelai: { signedOut: "/register", signedIn: "/dashboard/ai" },
  volumes: { signedOut: "/register", signedIn: "/dashboard/volumes" },
};

export function marketingCtaHref(intent: MarketingCtaIntent, signedIn: boolean): string {
  return signedIn ? CTA_TARGETS[intent].signedIn : CTA_TARGETS[intent].signedOut;
}