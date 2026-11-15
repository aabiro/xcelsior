import type { User } from "@/lib/auth";
import type { Host } from "@/lib/api";

export type WorkspaceMode = "rent" | "host";

/** Hosting is additive: all accounts retain their customer id and rental access. */
export function hostingEnabled(user: Pick<User, "role" | "provider_id" | "onboarding_intent"> | null): boolean {
  return Boolean(user && (user.role === "provider" || user.provider_id || ["provide", "both"].includes(user.onboarding_intent || "")));
}

export function routeWorkspaceMode(pathname: string): WorkspaceMode | null {
  if (/^\/dashboard\/(hosting|hosts|earnings|telemetry)(\/|$)/.test(pathname)) return "host";
  if (/^\/dashboard\/(instances|templates|inference|hpc|volumes|artifacts|billing|marketplace|spot-pricing)(\/|$)/.test(pathname)) return "rent";
  return null;
}

export function hostingProgress(hosts: Host[]) {
  const machines = hosts.filter(host => host.status !== "dead");
  // Registration and a running browser are not evidence of machine verification.
  const verified = machines.filter(host => host.admitted === true);
  return { machines, verified, step: machines.length === 0 ? 1 : verified.length === 0 ? 2 : 3 };
}
