import type { User } from "@/lib/auth";

export type BillingUserRef = Pick<User, "billing_customer_id" | "customer_id" | "user_id">;

export type TeamRole = "admin" | "member" | "viewer";

export interface TeamContext {
  isTeamMember: boolean;
  teamId?: string;
  teamName?: string;
  teamRole?: TeamRole;
  teamPlan?: string;
  billingCustomerId: string;
  canManageBilling: boolean;
  canWriteInstances: boolean;
}

export interface InstanceConcurrency {
  active: number;
  cap: number;
  shared: boolean;
}

/** Effective wallet / job owner id (team wallet when in a team). */
export function getBillingCustomerId(user: BillingUserRef | null): string {
  if (!user) return "";
  return user.billing_customer_id || user.customer_id || user.user_id || "";
}

export function getTeamContext(user: User | null): TeamContext {
  const billingCustomerId = getBillingCustomerId(user);
  const teamId = user?.team_id?.trim() || undefined;
  return {
    isTeamMember: !!teamId,
    teamId,
    teamName: user?.team_name?.trim() || undefined,
    teamRole: user?.team_role as TeamRole | undefined,
    teamPlan: user?.team_plan?.trim() || undefined,
    billingCustomerId,
    canManageBilling: user?.team_can_manage_billing ?? true,
    canWriteInstances: user?.team_can_write_instances ?? true,
  };
}

export function formatTeamRoleLabel(role?: string): string {
  if (!role) return "";
  return role.charAt(0).toUpperCase() + role.slice(1);
}

/** Persist active team on server and refresh auth context for billing/instance scope. */
export async function applyActiveTeamSwitch(
  teamId: string | null,
  refreshUser: () => Promise<void>,
): Promise<void> {
  const { switchActiveTeam } = await import("@/lib/api");
  await switchActiveTeam(teamId);
  await refreshUser();
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("xcelsior-team-changed", { detail: { teamId } }));
  }
}