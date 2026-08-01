/**
 * A machine-comparable description of the published tool surface.
 *
 * `_meta["xcelsior/toolVersion"]` already existed but meant nothing — nothing
 * checked it, so nothing stopped a tool from changing shape while its version
 * stayed at 2.0.0. This module turns it into a contract: the surface is
 * snapshotted to `mcp/tool-surface.json`, and a change that would break an
 * existing caller fails the build unless the tool's version was bumped in the
 * same commit (adoption plan X6.28, gate GX6).
 *
 * The descriptor deliberately captures *shape*, not prose. A reworded
 * description is not a breaking change; a newly-required input is.
 */
import { z } from "zod";
import { TOOL_CONTRACTS } from "./contracts.js";
import { TOOL_DESCRIPTIONS } from "./descriptions.js";
import { installToolAudit } from "../audit/context.js";
import { registerAllTools, type ToolRegistrationOptions } from "./index.js";
import { toolsInProfile, type ToolProfile } from "./profiles.js";
import type { AuthUser } from "../auth/bearer.js";
import type { XcelsiorApiClient } from "../client/api.js";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";

export interface FieldShape {
  type: string;
  optional: boolean;
  /** Present for enums — narrowing an enum breaks callers that used a value. */
  values?: string[];
}

export interface ToolSurfaceEntry {
  name: string;
  version: string;
  tenantClass: "tenant" | "operator";
  requiredScopes: string[];
  idempotency: string;
  retry: string;
  annotations: Record<string, boolean>;
  input: Record<string, FieldShape>;
  hasOutputSchema: boolean;
}

/** Structural summary of a Zod schema — enough to classify a change. */
export function zodShape(schema: unknown): Record<string, FieldShape> {
  const object = schema as z.ZodObject<z.ZodRawShape> | undefined;
  const shape = object?._def?.shape?.();
  if (!shape) return {};
  const result: Record<string, FieldShape> = {};
  for (const [key, value] of Object.entries(shape)) {
    result[key] = describeField(value as z.ZodTypeAny);
  }
  return result;
}

function describeField(field: z.ZodTypeAny): FieldShape {
  let current: z.ZodTypeAny = field;
  let optional = false;
  // Unwrap the decorators that change optionality or add a default without
  // changing the underlying type.
  for (let depth = 0; depth < 8; depth += 1) {
    const typeName = current?._def?.typeName;
    if (typeName === "ZodOptional" || typeName === "ZodDefault" || typeName === "ZodNullable") {
      optional = true;
      current = (current._def as { innerType: z.ZodTypeAny }).innerType;
      continue;
    }
    if (typeName === "ZodEffects") {
      current = (current._def as { schema: z.ZodTypeAny }).schema;
      continue;
    }
    break;
  }
  const typeName = String(current?._def?.typeName ?? "unknown");
  const shape: FieldShape = { type: typeName.replace(/^Zod/, "").toLowerCase(), optional };
  if (typeName === "ZodEnum") {
    shape.values = [...((current._def as { values: string[] }).values ?? [])].sort();
  }
  return shape;
}

/**
 * Run the real registration against a recorder.
 *
 * Re-deriving the surface from the registry rather than from a parallel list
 * is the point: a tool that changes only in `registerTool` still shows up.
 */
export function describeToolSurface(
  profile: ToolProfile = "customer",
  options: ToolRegistrationOptions = {},
): ToolSurfaceEntry[] {
  const recorded = new Map<string, { config: Record<string, unknown> }>();
  const recorder = {
    registerTool(name: string, config: Record<string, unknown>) {
      recorded.set(name, { config });
      return undefined;
    },
  };
  // A principal holding every scope, so nothing is filtered out for lack of
  // authority — the snapshot is of the profile, not of one token.
  const user: AuthUser = {
    scopes: [...new Set(Object.values(TOOL_CONTRACTS).flatMap((c) => [...c.requiredScopes]))],
  };
  installToolAudit(
    recorder as unknown as McpServer,
    {} as XcelsiorApiClient,
    user,
    "streamable_http",
    profile,
  );
  registerAllTools(recorder as unknown as McpServer, {} as XcelsiorApiClient, user, options);

  const inProfile = new Set(
    toolsInProfile(profile, { companyKnowledge: Boolean(options.companyKnowledge) }),
  );
  return [...recorded.entries()]
    .filter(([name]) => inProfile.has(name))
    .map(([name, { config }]) => {
      const contract = TOOL_CONTRACTS[name];
      return {
        name,
        version: contract.version,
        tenantClass: contract.tenantClass,
        requiredScopes: [...contract.requiredScopes].sort(),
        idempotency: contract.idempotency,
        retry: contract.retry,
        annotations: { ...contract.annotations },
        input: zodShape(config.inputSchema),
        hasOutputSchema: Boolean(config.outputSchema),
      } satisfies ToolSurfaceEntry;
    })
    .sort((a, b) => a.name.localeCompare(b.name));
}

export interface SurfaceChange {
  tool: string;
  breaking: boolean;
  detail: string;
}

/**
 * Classify snapshot → current.
 *
 * "Breaking" means an integration that worked yesterday stops working today:
 * the tool is gone, an input it never sent is now required, a value it passed
 * is no longer accepted, it needs a scope it was not granted, or an annotation
 * it relied on now claims something different. Everything else — a new optional
 * input, a widened enum, a reworded description — is additive.
 */
export function diffSurface(
  before: ToolSurfaceEntry[],
  after: ToolSurfaceEntry[],
): SurfaceChange[] {
  const previous = new Map(before.map((entry) => [entry.name, entry]));
  const current = new Map(after.map((entry) => [entry.name, entry]));
  const changes: SurfaceChange[] = [];

  for (const [name, was] of previous) {
    const now = current.get(name);
    if (!now) {
      changes.push({ tool: name, breaking: true, detail: "tool removed from the published profile" });
      continue;
    }
    if (was.tenantClass !== now.tenantClass) {
      changes.push({
        tool: name, breaking: true,
        detail: `tenantClass ${was.tenantClass} → ${now.tenantClass}`,
      });
    }
    const addedScopes = now.requiredScopes.filter((s) => !was.requiredScopes.includes(s));
    const removedScopes = was.requiredScopes.filter((s) => !now.requiredScopes.includes(s));
    // The gateway grants access if the caller holds *any* required scope, so
    // removing an alternative narrows access and is the breaking direction.
    if (removedScopes.length) {
      changes.push({
        tool: name, breaking: true,
        detail: `no longer accepts scope(s): ${removedScopes.join(", ")}`,
      });
    }
    if (addedScopes.length) {
      changes.push({
        tool: name, breaking: false,
        detail: `additionally accepts scope(s): ${addedScopes.join(", ")}`,
      });
    }
    for (const [key, value] of Object.entries(was.annotations)) {
      if (now.annotations[key] !== value) {
        changes.push({
          tool: name, breaking: true,
          detail: `annotation ${key} ${value} → ${now.annotations[key]}`,
        });
      }
    }
    for (const [field, wasField] of Object.entries(was.input)) {
      const nowField = now.input[field];
      if (!nowField) {
        changes.push({ tool: name, breaking: true, detail: `input '${field}' removed` });
        continue;
      }
      if (nowField.type !== wasField.type) {
        changes.push({
          tool: name, breaking: true,
          detail: `input '${field}' type ${wasField.type} → ${nowField.type}`,
        });
      }
      if (wasField.optional && !nowField.optional) {
        changes.push({
          tool: name, breaking: true,
          detail: `input '${field}' became required`,
        });
      }
      const dropped = (wasField.values ?? []).filter((v) => !(nowField.values ?? []).includes(v));
      if (wasField.values && dropped.length) {
        changes.push({
          tool: name, breaking: true,
          detail: `input '${field}' no longer accepts: ${dropped.join(", ")}`,
        });
      }
    }
    for (const [field, nowField] of Object.entries(now.input)) {
      if (was.input[field]) continue;
      changes.push({
        tool: name,
        breaking: !nowField.optional,
        detail: nowField.optional
          ? `new optional input '${field}'`
          : `new REQUIRED input '${field}'`,
      });
    }
    if (was.hasOutputSchema && !now.hasOutputSchema) {
      changes.push({ tool: name, breaking: true, detail: "output schema removed" });
    }
  }

  for (const name of current.keys()) {
    if (!previous.has(name)) {
      changes.push({ tool: name, breaking: false, detail: "new tool" });
    }
  }
  return changes;
}
