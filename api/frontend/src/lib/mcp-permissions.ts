// Pure logic behind the MCP permission picker (Read / Write / Custom).
//
// Nothing here names a scope. The rows are built from whatever the server
// says a key may hold, so a scope added to the Quick Connect set shows up as a
// checkbox without this file changing.

export type PermissionPreset = "read" | "write" | "custom";

export interface PermissionPresets {
  /** Every scope that only reads. */
  read: string[];
  /** Everything the key may hold: the ceiling Custom chooses from. */
  write: string[];
}

export interface PermissionRow {
  /** The part before the colon: `instances`, `billing`, … */
  resource: string;
  /** The read scope first, then the ones that change something. */
  scopes: string[];
}

const resourceOf = (scope: string) => scope.split(":", 1)[0];
export const isReadScope = (scope: string) => scope.endsWith(":read");

/** One row per resource, in the order the server lists them. */
export function permissionRows(ceiling: string[]): PermissionRow[] {
  const rows = new Map<string, string[]>();
  for (const scope of ceiling) {
    const resource = resourceOf(scope);
    rows.set(resource, [...(rows.get(resource) ?? []), scope]);
  }
  return [...rows].map(([resource, scopes]) => ({
    resource,
    scopes: [...scopes.filter(isReadScope), ...scopes.filter((s) => !isReadScope(s))],
  }));
}

/**
 * Tick or untick one box, keeping the selection coherent.
 *
 * Ticking anything that changes a resource ticks that resource's read box too.
 * Unticking read unticks everything on the row that depends on it, otherwise
 * the box could never be cleared. The server applies the same rule
 * (`normalize_mcp_permission_scopes`), so this is a courtesy, not the guard.
 */
export function toggleScope(selected: string[], scope: string, ceiling: string[]): string[] {
  const next = new Set(selected);
  const resource = resourceOf(scope);
  const read = `${resource}:read`;
  if (next.has(scope)) {
    next.delete(scope);
    if (scope === read) {
      for (const other of ceiling) if (resourceOf(other) === resource) next.delete(other);
    }
  } else {
    next.add(scope);
    if (scope !== read && ceiling.includes(read)) next.add(read);
  }
  return ceiling.filter((s) => next.has(s));
}

export function sameScopes(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const set = new Set(a);
  return b.every((s) => set.has(s));
}

/** What a tab starts with. Custom starts with everything ticked. */
export function presetScopes(preset: PermissionPreset, presets: PermissionPresets): string[] {
  return preset === "read" ? [...presets.read] : [...presets.write];
}

/** Read and Write for a plain list of scopes, for callers with no server copy. */
export function presetsFrom(ceiling: readonly string[]): PermissionPresets {
  return { read: ceiling.filter(isReadScope), write: [...ceiling] };
}
