"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GPU_MODELS, getGpusByCategory, type GpuModel, type GpuCategory } from "@/lib/gpu-models";

interface GpuModelSelectorProps {
  value: string;
  onChange: (selection: { model: string; vram_gb: number }) => void;
  placeholder?: string;
  disabled?: boolean;
}

const LISTBOX_ID = "gpu-model-listbox";

export function GpuModelSelector({
  value,
  onChange,
  placeholder = "Search GPUs...",
  disabled = false,
}: GpuModelSelectorProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [highlightIndex, setHighlightIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const selectedLabel = useMemo(() => {
    const entry = GPU_MODELS.find((g) => g.value === value);
    return entry ? entry.label : "";
  }, [value]);

  const filtered = useMemo(() => {
    if (!search.trim()) return getGpusByCategory();
    const q = search.toLowerCase();
    const result = {} as Record<GpuCategory, GpuModel[]>;
    const grouped = getGpusByCategory();
    for (const [cat, models] of Object.entries(grouped) as [GpuCategory, GpuModel[]][]) {
      const matches = models.filter(
        (m) => m.label.toLowerCase().includes(q) || m.value.toLowerCase().includes(q)
      );
      if (matches.length > 0) result[cat] = matches;
    }
    return result;
  }, [search]);

  // Pre-compute flat list with stable indices (no mutable render variable)
  const flatFiltered = useMemo(() => {
    const items: GpuModel[] = [];
    for (const models of Object.values(filtered)) {
      items.push(...models);
    }
    return items;
  }, [filtered]);

  // Map model value → flat index for O(1) lookup during render
  const modelIndexMap = useMemo(() => {
    const map = new Map<string, number>();
    flatFiltered.forEach((m, i) => map.set(m.value, i));
    return map;
  }, [flatFiltered]);

  useEffect(() => {
    setHighlightIndex(0);
  }, [search]);

  // Clamp highlightIndex when filtered list shrinks (prevents stale index)
  useEffect(() => {
    if (flatFiltered.length > 0 && highlightIndex >= flatFiltered.length) {
      setHighlightIndex(flatFiltered.length - 1);
    }
  }, [flatFiltered.length, highlightIndex]);

  // Click-outside handler
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [open]);

  // Scroll highlighted item into view
  useEffect(() => {
    if (!open || !listRef.current) return;
    const highlighted = listRef.current.querySelector("[data-highlighted=true]");
    if (highlighted) highlighted.scrollIntoView({ block: "nearest" });
  }, [highlightIndex, open]);

  const selectItem = useCallback(
    (model: GpuModel) => {
      onChange({ model: model.value, vram_gb: model.vram_gb });
      setSearch("");
      setOpen(false);
    },
    [onChange]
  );

  function handleKeyDown(e: React.KeyboardEvent) {
    if (!open) {
      if (e.key === "ArrowDown" || e.key === "Enter") {
        e.preventDefault();
        setOpen(true);
      }
      return;
    }

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setHighlightIndex((i) => Math.min(i + 1, flatFiltered.length - 1));
        break;
      case "ArrowUp":
        e.preventDefault();
        setHighlightIndex((i) => Math.max(i - 1, 0));
        break;
      case "Enter":
        e.preventDefault();
        if (flatFiltered.length > 0 && flatFiltered[highlightIndex]) {
          selectItem(flatFiltered[highlightIndex]);
        }
        break;
      case "Escape":
      case "Tab":
        setOpen(false);
        if (e.key === "Escape") {
          e.preventDefault();
          inputRef.current?.blur();
        }
        break;
    }
  }

  const activeDescendant = flatFiltered[highlightIndex]
    ? `gpu-opt-${flatFiltered[highlightIndex].value}`
    : undefined;

  return (
    <div ref={containerRef} className="relative">
      <input
        ref={inputRef}
        type="text"
        role="combobox"
        aria-label="GPU model"
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-controls={LISTBOX_ID}
        aria-activedescendant={open ? activeDescendant : undefined}
        aria-autocomplete="list"
        value={open ? search : selectedLabel}
        placeholder={placeholder}
        disabled={disabled}
        className="w-full rounded-lg border border-border bg-navy px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-2 focus:ring-ice-blue focus:border-transparent disabled:cursor-not-allowed disabled:opacity-50"
        onFocus={() => {
          if (!disabled) {
            setOpen(true);
            setSearch("");
          }
        }}
        onChange={(e) => setSearch(e.target.value)}
        onKeyDown={handleKeyDown}
        autoComplete="off"
      />
      {/* Chevron */}
      <div className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-text-muted">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
          <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      {open && (
        <div
          ref={listRef}
          id={LISTBOX_ID}
          role="listbox"
          aria-label="GPU models"
          className="absolute z-50 mt-1 max-h-72 w-full overflow-y-auto rounded-lg border border-border bg-navy shadow-lg"
        >
          {Object.keys(filtered).length === 0 ? (
            <div className="px-3 py-2 text-sm text-text-muted">No GPUs match &ldquo;{search}&rdquo;</div>
          ) : (
            Object.entries(filtered).map(([category, models]) => (
              <div key={category} role="group" aria-label={category}>
                <div className="sticky top-0 bg-surface-hover/80 px-3 py-1.5 text-xs font-semibold text-text-muted backdrop-blur-sm">
                  {category}
                </div>
                {(models as GpuModel[]).map((model) => {
                  const idx = modelIndexMap.get(model.value) ?? 0;
                  const isHighlighted = idx === highlightIndex;
                  const isSelected = model.value === value;
                  return (
                    <button
                      key={model.value}
                      id={`gpu-opt-${model.value}`}
                      type="button"
                      role="option"
                      aria-selected={isSelected}
                      data-highlighted={isHighlighted}
                      className={`flex w-full items-center justify-between px-3 py-2 text-sm transition-colors ${
                        isHighlighted ? "bg-ice-blue/15 text-text-primary" : "text-text-secondary"
                      } ${isSelected ? "font-medium text-ice-blue" : ""} hover:bg-ice-blue/10`}
                      onMouseEnter={() => setHighlightIndex(idx)}
                      onMouseDown={(e) => {
                        e.preventDefault();
                        selectItem(model);
                      }}
                    >
                      <span className="flex items-center gap-2">
                        {isSelected && (
                          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true" className="text-ice-blue shrink-0">
                            <path d="M11.5 3.5L5.5 10L2.5 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        )}
                        <span>{model.label}</span>
                      </span>
                      <span className="ml-2 text-xs text-text-muted">{model.vram_gb} GB</span>
                    </button>
                  );
                })}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
