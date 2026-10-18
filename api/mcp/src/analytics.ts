import { instrument, type McpAnalytics, type MCPAnalyticsOptions } from "@posthog/mcp";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { PostHog } from "posthog-node";
import type { AuthUser } from "./auth/bearer.js";
import type { PostHogAnalyticsConfig } from "./config.js";
import type { ToolProfile } from "./tools/profiles.js";

export type McpAnalyticsTransport = "streamable_http" | "stdio";

export interface McpAnalyticsContext {
  transport: McpAnalyticsTransport;
  profile: ToolProfile;
  user?: AuthUser;
}

type PostHogCaptureEvent = Parameters<NonNullable<MCPAnalyticsOptions["beforeSend"]>>[0];

const PRIVATE_EVENT_PROPERTIES = [
  "$mcp_parameters",
  "$mcp_response",
  "$mcp_error_message",
  "$exception_list",
  "$exception_source",
] as const;

let posthog: PostHog | undefined;
let shutdownPromise: Promise<void> | undefined;
let warnedInvalidKey = false;

function analyticsLogger(message: string): void {
  // The PostHog SDK also reports successful captures to its logger. A fresh MCP
  // server is constructed per HTTP request, so forwarding those messages would
  // turn ordinary traffic into noisy logs. Warnings still go to stderr, which
  // is safe for both HTTP and STDIO transports.
  if (/warn|fail|error/i.test(message)) {
    process.stderr.write(`[xcelsior-mcp] PostHog analytics: ${message}\n`);
  }
}

export function posthogIdentity(user?: AuthUser): MCPAnalyticsOptions["identify"] {
  // Match the browser's PostHog identity first so MCP activity joins the same
  // person. Machine grants generally have a subject but no user_id; tenant id
  // is the final stable fallback. Email is deliberately never an identifier.
  const distinctId = user?.user_id || user?.subject || user?.client_id || user?.customer_id;
  return distinctId ? { distinctId } : null;
}

/**
 * Keep MCP analytics metadata-only.
 *
 * @posthog/mcp captures arguments, results, and exception messages by default.
 * Xcelsior tools can carry init scripts, environment variables, SSH material,
 * and upstream error bodies, so even the SDK's key-based sanitizer is not a
 * sufficient privacy boundary here. Tool name, duration, outcome, client, and
 * session metadata remain available without retaining payload content.
 */
export function stripPrivateMcpPayloads(event: PostHogCaptureEvent): PostHogCaptureEvent {
  const properties = { ...event.properties };
  for (const property of PRIVATE_EVENT_PROPERTIES) delete properties[property];
  return { ...event, properties };
}

export function buildMcpAnalyticsOptions(context: McpAnalyticsContext): MCPAnalyticsOptions {
  return {
    logger: analyticsLogger,
    identify: posthogIdentity(context.user),
    // These options mutate the public tool schemas. Keep the reviewed v2 tool
    // surface stable; stateless correlation comes from auth identity plus the
    // SDK's signed Mcp-Session-Id token instead.
    context: false,
    enableConversationId: false,
    reportMissing: false,
    // Exception messages can include upstream response bodies. The primary
    // tool event still records the safe error flag/type.
    enableExceptionAutocapture: false,
    eventProperties: () => ({
      xcelsior_transport: context.transport,
      xcelsior_tool_profile: context.profile,
      xcelsior_environment: process.env.XCELSIOR_ENV || process.env.NODE_ENV || "development",
    }),
    beforeSend: stripPrivateMcpPayloads,
  };
}

/** Exported separately so integration tests exercise the real SDK without network I/O. */
export function instrumentMcpAnalyticsWithClient(
  server: McpServer,
  client: PostHog,
  context: McpAnalyticsContext,
): McpAnalytics {
  return instrument(server, client, buildMcpAnalyticsOptions(context));
}

/** Attach analytics when a project API key is configured; otherwise remain a no-op. */
export function instrumentMcpAnalytics(
  server: McpServer,
  config: PostHogAnalyticsConfig,
  context: McpAnalyticsContext,
): boolean {
  if (!config.projectApiKey) return false;
  if (!config.projectApiKey.startsWith("phc_")) {
    if (!warnedInvalidKey) {
      warnedInvalidKey = true;
      analyticsLogger("Warning: project API key must start with phc_; analytics disabled");
    }
    return false;
  }

  posthog ??= new PostHog(config.projectApiKey, {
    host: config.host,
    // MCP analytics does not need IP enrichment, and disabling it reduces the
    // amount of personal data sent to PostHog.
    disableGeoip: true,
  });
  instrumentMcpAnalyticsWithClient(server, posthog, context);
  return true;
}

/** Flush the module-scoped posthog-node client exactly once during shutdown. */
export async function shutdownMcpAnalytics(): Promise<void> {
  if (!posthog) return;
  if (!shutdownPromise) {
    const client = posthog;
    shutdownPromise = client.shutdown().finally(() => {
      posthog = undefined;
      shutdownPromise = undefined;
    });
  }
  await shutdownPromise;
}
