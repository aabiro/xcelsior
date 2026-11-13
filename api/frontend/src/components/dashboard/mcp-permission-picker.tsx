"use client";

import {
  Activity,
  Building2,
  Check,
  Cpu,
  Eye,
  HardDrive,
  KeyRound,
  Package,
  PencilLine,
  Server,
  Shield,
  SlidersHorizontal,
  Star,
  Store,
  Wallet,
  Zap,
  type LucideIcon,
} from "lucide-react";
import { useLocale } from "@/lib/locale";
import {
  isReadScope,
  permissionRows,
  presetScopes,
  toggleScope,
  type PermissionPreset,
  type PermissionPresets,
} from "@/lib/mcp-permissions";
import { cn } from "@/lib/utils";

export interface PermissionSelection {
  preset: PermissionPreset;
  scopes: string[];
}

const TAB_ICON = { read: Eye, write: PencilLine, custom: SlidersHorizontal } as const;

// Decoration only: a resource with no icon here gets the shield.
const RESOURCE_ICON: Record<string, LucideIcon> = {
  instances: Server,
  ssh: KeyRound,
  volumes: HardDrive,
  artifacts: Package,
  billing: Wallet,
  gpu: Cpu,
  marketplace: Store,
  inference: Zap,
  events: Activity,
  providers: Building2,
  reputation: Star,
};

/**
 * Read / Write / Custom, with every box already ticked the way each tab means.
 *
 * Read and Write are fixed sets. Editing a box on either one moves to Custom
 * with that edit applied, so there is no dead control and no way to hold a
 * selection labelled "Read" that is not the Read set.
 */
export function McpPermissionPicker({
  presets,
  value,
  onChange,
  customStart,
  extras,
  disabled,
  stacked,
  className,
}: {
  presets: PermissionPresets;
  value: PermissionSelection;
  onChange: (next: PermissionSelection) => void;
  /** What the Custom tab opens with; everything ticked when not given. */
  customStart?: string[];
  /**
   * Permissions the key holds that no tab offers, each granted on the approval
   * page. Shown so they can be removed; a tab never adds one.
   */
  extras?: string[];
  disabled?: boolean;
  /** Put each row's boxes under its name. For a narrow column on a wide screen. */
  stacked?: boolean;
  className?: string;
}) {
  const { t } = useLocale();
  const approved = new Set(extras ?? []);
  // Everything a box exists for: the tabs' set, then what was approved on top.
  const ceiling = [...presets.write, ...(extras ?? []).filter((s) => !presets.write.includes(s))];
  const rows = permissionRows(ceiling);
  const selected = new Set(value.scopes);

  // A message with no translation comes back as its own key.
  const label = (key: string, fallback: string) => {
    const text = t(key);
    return text === key ? fallback : text;
  };

  const pickTab = (id: string) => {
    const preset = id as PermissionPreset;
    if (preset === value.preset) return;
    onChange({
      preset,
      scopes: preset === "custom" ? [...(customStart ?? ceiling)] : presetScopes(preset, presets),
    });
  };

  return (
    <div className={cn("space-y-3", className)}>
      <div role="tablist" aria-label={t("dash.mcp.perm.title")} className="grid grid-cols-3 gap-2">
        {(["read", "write", "custom"] as const).map((id) => {
          const Icon = TAB_ICON[id];
          const active = value.preset === id;
          return (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={active}
              aria-label={t(`dash.mcp.perm.tab_${id}`)}
              disabled={disabled}
              onClick={() => pickTab(id)}
              className={cn(
                "group relative flex flex-col items-center gap-1 rounded-xl border px-2 py-2.5 text-center transition-all duration-200 disabled:opacity-60",
                active
                  ? "border-accent-cyan/50 bg-gradient-to-b from-accent-cyan/10 to-accent-violet/10 shadow-sm ring-1 ring-accent-cyan/25"
                  : "border-border/60 bg-surface/40 hover:border-border hover:bg-surface-hover",
              )}
            >
              <span
                className={cn(
                  "flex h-7 w-7 items-center justify-center rounded-lg transition-colors",
                  active
                    ? "bg-gradient-to-br from-accent-cyan to-accent-violet text-white"
                    : "bg-surface-hover text-text-muted group-hover:text-text-primary",
                )}
              >
                <Icon aria-hidden className="h-3.5 w-3.5" />
              </span>
              <span className={cn("text-sm font-semibold", active ? "text-text-primary" : "text-text-secondary")}>
                {t(`dash.mcp.perm.tab_${id}`)}
              </span>
              <span className="text-[11px] leading-tight text-text-muted">{t(`dash.mcp.perm.hint_${id}`)}</span>
            </button>
          );
        })}
      </div>
      <p className="text-center text-xs leading-relaxed text-text-secondary">
        {t(`dash.mcp.perm.desc_${value.preset}`)}
      </p>

      <div
        role="group"
        aria-label={t("dash.mcp.perm.title")}
        className="divide-y divide-border/40 overflow-hidden rounded-xl border border-border/60 bg-surface/30 text-left"
      >
        {rows.map((row) => {
          const RowIcon = RESOURCE_ICON[row.resource] ?? Shield;
          const name = label(`dash.mcp.perm.res_${row.resource}`, row.resource);
          const held = row.scopes.some((scope) => selected.has(scope));
          return (
            <div
              key={row.resource}
              className={cn(
                "flex flex-col gap-2 px-3 py-2",
                !stacked && "sm:flex-row sm:items-center sm:justify-between sm:gap-4",
              )}
            >
              <span
                className={cn(
                  "flex items-center gap-2 text-[13px] font-medium transition-colors",
                  held ? "text-text-primary" : "text-text-muted",
                )}
              >
                <RowIcon aria-hidden className={cn("h-3.5 w-3.5 shrink-0", held ? "text-accent-cyan" : "text-text-muted")} />
                {name}
              </span>
              <div className={cn("flex flex-wrap gap-1.5", !stacked && "sm:justify-end")}>
                {row.scopes.map((scope) => {
                  const verb = scope.split(":")[1] ?? scope;
                  const checked = selected.has(scope);
                  const read = isReadScope(scope);
                  const extra = approved.has(scope);
                  const verbLabel = label(`dash.mcp.perm.verb_${verb}`, verb);
                  return (
                    <label
                      key={scope}
                      title={extra ? t("dash.mcp.perm.extra_hint") : scope}
                      className={cn(
                        "inline-flex cursor-pointer select-none items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-accent-cyan/60",
                        checked && read && "border-accent-cyan/40 bg-accent-cyan/10 text-accent-cyan",
                        checked && !read && !extra && "border-accent-violet/40 bg-accent-violet/10 text-accent-violet",
                        checked && extra && "border-accent-gold/50 bg-accent-gold/10 text-accent-gold",
                        !checked && "border-border/60 text-text-muted hover:border-border hover:text-text-secondary",
                        disabled && "pointer-events-none opacity-50",
                      )}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        disabled={disabled}
                        onChange={() =>
                          onChange({ preset: "custom", scopes: toggleScope(value.scopes, scope, ceiling) })
                        }
                        aria-label={`${name}: ${verbLabel}`}
                        className="sr-only"
                      />
                      <span
                        aria-hidden
                        className={cn(
                          "flex h-3.5 w-3.5 items-center justify-center rounded-full border transition-colors",
                          checked && read && "border-accent-cyan bg-accent-cyan text-white",
                          checked && !read && !extra && "border-accent-violet bg-accent-violet text-white",
                          checked && extra && "border-accent-gold bg-accent-gold text-white",
                          !checked && "border-border",
                        )}
                      >
                        {checked && <Check className="h-2.5 w-2.5" strokeWidth={3.5} />}
                      </span>
                      {verbLabel}
                    </label>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
