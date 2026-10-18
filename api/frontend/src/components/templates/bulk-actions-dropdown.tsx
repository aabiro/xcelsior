"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown, Check, Square, CheckSquare, Star, StarOff } from "lucide-react";
import { cn } from "@/lib/utils";

// Gmail-style "select + bulk action" split-button that lives in the
// table header beside the "select all" checkbox. Clicking the caret
// opens a small menu with selection helpers (All, None, Starred, Unstarred)
// plus any caller-supplied bulk actions. Kept deliberately small so it
// doesn't need the full headless-ui / radix primitive surface.

export type BulkSelectionHelper =
  | "all-page"
  | "none"
  | "starred"
  | "unstarred";

export type BulkAction = {
  id: string;
  label: string;
  icon?: React.ReactNode;
  danger?: boolean;
  disabled?: boolean;
};

type Props = {
  // Selection controls
  allSelected: boolean;
  someSelected: boolean;
  onToggleAll: () => void;
  onSelectHelper: (helper: BulkSelectionHelper) => void;
  // Bulk actions shown below the selection helpers
  actions: BulkAction[];
  onAction: (actionId: string) => void;
  // Disabled state (scope !== "mine")
  disabled?: boolean;
};

export function BulkActionsDropdown({
  allSelected,
  someSelected,
  onToggleAll,
  onSelectHelper,
  actions,
  onAction,
  disabled,
}: Props) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent) {
      if (!containerRef.current?.contains(e.target as Node)) setOpen(false);
    }
    function onEsc(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onEsc);
    };
  }, [open]);

  return (
    <div ref={containerRef} className="relative inline-flex items-center">
      <button
        type="button"
        onClick={onToggleAll}
        disabled={disabled}
        className={cn(
          "flex h-7 w-7 items-center justify-center rounded-l border border-border bg-surface transition-colors",
          !disabled && "hover:bg-surface-hover",
          disabled && "opacity-40 cursor-not-allowed",
        )}
        title={allSelected ? "Deselect all" : "Select all on page"}
        aria-label="Toggle select all"
      >
        {allSelected ? (
          <CheckSquare className="h-4 w-4 text-ice-blue" />
        ) : someSelected ? (
          <div className="h-2.5 w-2.5 rounded-[2px] bg-ice-blue" />
        ) : (
          <Square className="h-4 w-4 text-text-muted" />
        )}
      </button>
      <button
        type="button"
        onClick={() => !disabled && setOpen((v) => !v)}
        disabled={disabled}
        className={cn(
          "flex h-7 w-5 items-center justify-center rounded-r border border-l-0 border-border bg-surface transition-colors",
          !disabled && "hover:bg-surface-hover",
          disabled && "opacity-40 cursor-not-allowed",
        )}
        title="Bulk actions"
        aria-label="Open bulk actions"
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <ChevronDown className="h-3.5 w-3.5 text-text-muted" />
      </button>
      {open && (
        <div
          role="menu"
          className="absolute left-0 top-full z-50 mt-1 w-56 overflow-hidden rounded-lg border border-border bg-surface shadow-lg"
        >
          <div className="py-1 text-xs text-text-muted px-3 pt-2">Select</div>
          <MenuItem
            onClick={() => { onSelectHelper("all-page"); setOpen(false); }}
            icon={<CheckSquare className="h-3.5 w-3.5" />}
          >
            All on page
          </MenuItem>
          <MenuItem
            onClick={() => { onSelectHelper("none"); setOpen(false); }}
            icon={<Square className="h-3.5 w-3.5" />}
          >
            None
          </MenuItem>
          <MenuItem
            onClick={() => { onSelectHelper("starred"); setOpen(false); }}
            icon={<Star className="h-3.5 w-3.5" />}
          >
            Starred
          </MenuItem>
          <MenuItem
            onClick={() => { onSelectHelper("unstarred"); setOpen(false); }}
            icon={<StarOff className="h-3.5 w-3.5" />}
          >
            Unstarred
          </MenuItem>
          {actions.length > 0 && (
            <>
              <div className="border-t border-border my-1" />
              <div className="py-1 text-xs text-text-muted px-3">Actions</div>
              {actions.map((a) => (
                <MenuItem
                  key={a.id}
                  onClick={() => { if (!a.disabled) { onAction(a.id); setOpen(false); } }}
                  icon={a.icon}
                  danger={a.danger}
                  disabled={a.disabled}
                >
                  {a.label}
                </MenuItem>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function MenuItem({
  onClick,
  icon,
  danger,
  disabled,
  children,
}: {
  onClick: () => void;
  icon?: React.ReactNode;
  danger?: boolean;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm transition-colors",
        !disabled && !danger && "text-text-primary hover:bg-surface-hover",
        !disabled && danger && "text-red-400 hover:bg-red-500/10",
        disabled && "text-text-muted/50 cursor-not-allowed",
      )}
    >
      {icon && <span className="flex h-4 w-4 items-center justify-center">{icon}</span>}
      <span className="flex-1">{children}</span>
    </button>
  );
}
