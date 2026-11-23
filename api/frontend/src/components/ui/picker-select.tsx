"use client";

import { useEffect, useId, useRef, useState, useSyncExternalStore } from "react";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

export interface PickerOption {
  value: string;
  label: string;
}

const PHONE_QUERY = "(max-width: 639px)";

function subscribePhone(onChange: () => void) {
  if (typeof window === "undefined" || !window.matchMedia) return () => {};
  const query = window.matchMedia(PHONE_QUERY);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

/** Below the `sm` breakpoint. The server render (and the first client render) say no. */
export function usePhoneWidth(): boolean {
  return useSyncExternalStore(
    subscribePhone,
    () => typeof window !== "undefined" && !!window.matchMedia && window.matchMedia(PHONE_QUERY).matches,
    () => false,
  );
}

interface PickerSelectProps {
  id?: string;
  value: string;
  options: PickerOption[];
  onChange: (value: string) => void;
  /** Shown when nothing is chosen, or when there is nothing to choose. */
  placeholder?: string;
  disabled?: boolean;
  /** The native `<select>` on wider screens, so desktop stays exactly as it was. */
  renderNative: (select: {
    id?: string;
    value: string;
    disabled?: boolean;
    onChange: (event: React.ChangeEvent<HTMLSelectElement>) => void;
    children: React.ReactNode;
  }) => React.ReactNode;
  /** Classes for the phone trigger (height, text size) to match the form's fields. */
  className?: string;
  "aria-labelledby"?: string;
}

/**
 * A select that is a solid list on a phone.
 *
 * iOS draws a native `<select>` as a translucent glass menu over the page; in
 * the deploy sheet the page showed through it, and an empty select opened a
 * glass bubble reading "No Options" (owner review, 2026-11-22). On a phone this
 * opens an opaque list in the form's own colours, inline under the field so a
 * scrolling sheet never clips it. Wider screens keep the native select.
 */
export function PickerSelect({
  id,
  value,
  options,
  onChange,
  placeholder,
  disabled,
  renderNative,
  className,
  "aria-labelledby": labelledBy,
}: PickerSelectProps) {
  const phone = usePhoneWidth();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const listId = useId();
  const empty = options.length === 0;
  const selected = options.find((option) => option.value === value);

  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        setOpen(false);
      }
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey, true);
    };
  }, [open]);

  if (!phone) {
    return (
      <>
        {renderNative({
          id,
          value,
          disabled: disabled || empty,
          onChange: (event) => onChange(event.target.value),
          children: (
            <>
              {(!selected || empty) && <option value="">{placeholder ?? ""}</option>}
              {options.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </>
          ),
        })}
      </>
    );
  }

  const inert = disabled || empty;
  return (
    <div ref={rootRef} className="relative">
      <button
        id={id}
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        aria-labelledby={labelledBy}
        disabled={inert}
        onClick={() => setOpen((value) => !value)}
        className={cn(
          "flex h-11 w-full items-center justify-between gap-2 rounded-lg border border-border bg-navy px-3 text-left text-base text-text-primary",
          "focus:outline-none focus:ring-2 focus:ring-ice-blue focus:border-transparent",
          "disabled:cursor-not-allowed disabled:opacity-50",
          className,
        )}
      >
        <span className={cn("min-w-0 truncate", !selected && "text-text-muted")}>
          {selected?.label ?? placeholder ?? ""}
        </span>
        <ChevronDown className={cn("h-4 w-4 shrink-0 text-text-muted transition-transform", open && "rotate-180")} aria-hidden />
      </button>
      {open && (
        <div
          id={listId}
          role="listbox"
          aria-labelledby={labelledBy}
          className="mt-1.5 max-h-64 overflow-y-auto overscroll-contain rounded-xl border border-border bg-[var(--popover-solid)] p-1 shadow-xl"
        >
          {options.map((option) => {
            const active = option.value === value;
            return (
              <button
                key={option.value}
                type="button"
                role="option"
                aria-selected={active}
                onClick={() => {
                  onChange(option.value);
                  setOpen(false);
                }}
                className={cn(
                  "flex min-h-11 w-full items-center justify-between gap-3 rounded-lg px-3 text-left text-base",
                  active ? "bg-ice-blue/15 font-medium text-text-primary" : "text-text-secondary active:bg-surface-hover",
                )}
              >
                <span className="min-w-0 truncate">{option.label}</span>
                {active && <Check className="h-4 w-4 shrink-0 text-ice-blue" aria-hidden />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
