/**
 * Trust-surface profiles (adoption plan §4a).
 *
 * One codebase, one image, two exposed surfaces:
 *
 *  - `customer` — what `mcp.xcelsior.ca/mcp` serves and what a directory
 *    reviewer sees. Tenant-scoped workflows only.
 *  - `operator` — the same build at an unlisted hostname, adding the
 *    platform-global host and control-plane tools.
 *
 * The split is a security boundary *and* a model-quality one. A provider's
 * frozen tool snapshot becomes a promise about what every end user can see, so
 * `drain_host` appearing once in a public listing is a promise we would rather
 * not make. Filtering happens at registration, not at call time, because
 * `tools/list` is what gets snapshotted.
 */
import { TOOL_CONTRACTS } from "./contracts.js";

export type ToolProfile = "customer" | "operator";

/** Scopes that mark a principal as platform-operator rather than tenant. */
export const OPERATOR_SCOPE_PREFIXES = ["hosts:", "control_plane:"] as const;

export function isOperatorTool(name: string): boolean {
  return TOOL_CONTRACTS[name]?.tenantClass === "operator";
}

export function principalIsOperator(scopes: string[] | undefined): boolean {
  if (!scopes?.length) return false;
  return scopes.some((scope) => OPERATOR_SCOPE_PREFIXES.some((prefix) => scope.startsWith(prefix)));
}

/**
 * Whether this deployment + principal may see a tool at all.
 *
 * Two independent gates, deliberately: the deployment profile is the
 * structural boundary, and the principal check means a token without operator
 * scopes never enumerates operator tools even on the operator host — so a
 * misconfigured deployment cannot leak the operator surface into a directory
 * listing on its own.
 */
export function toolIsVisible(
  name: string,
  profile: ToolProfile,
  principalScopes: string[] | undefined,
): boolean {
  if (!TOOL_CONTRACTS[name]) return false;
  if (!isOperatorTool(name)) return true;
  if (profile !== "operator") return false;
  // `undefined` scopes means "not resolved" (stdio with no token yet), which
  // must not be read as operator authority.
  return principalIsOperator(principalScopes);
}

/**
 * Optional company-knowledge tools (adoption plan X1.11).
 *
 * They have contracts like every other tool, but they are only *registered*
 * when the deployment opts in. Anything describing the surface — the snapshot,
 * the advertised scopes, the E2E's expected listing — has to account for that,
 * or it describes a surface the server does not actually serve.
 */
export const COMPANY_KNOWLEDGE_TOOLS = new Set(["search", "fetch"]);

export interface ProfileOptions {
  /** Defaults to false, matching the deployment default. */
  companyKnowledge?: boolean;
}

export function toolsInProfile(
  profile: ToolProfile,
  options: ProfileOptions = {},
): string[] {
  return Object.keys(TOOL_CONTRACTS)
    .filter((name) => profile === "operator" || !isOperatorTool(name))
    .filter((name) => options.companyKnowledge || !COMPANY_KNOWLEDGE_TOOLS.has(name))
    .sort();
}

/** Scopes the profile's tools actually require, for resource metadata. */
export function scopesForProfile(profile: ToolProfile, options: ProfileOptions = {}): string[] {
  const scopes = new Set<string>();
  for (const name of toolsInProfile(profile, options)) {
    for (const scope of TOOL_CONTRACTS[name].requiredScopes) {
      // `api` is the blanket automation scope, not something a connector should
      // request during consent.
      if (scope !== "api") scopes.add(scope);
    }
  }
  return [...scopes].sort();
}
