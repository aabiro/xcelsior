"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { apiFetch } from "@/lib/api";
import { GPU_CATEGORIES, getGpuModel } from "@/lib/gpu-models";

/** One GPU a host can list: a model with a published rate on the backend rate card. */
interface GpuModel {
  value: string;
  label: string;
  vram_gb: number;
  category: string;
}

type RateCard = { pricing?: Record<string, { vram_gb?: number; base_rate_cad?: number }> };

/**
 * The models a host can register are the ones the backend publishes a rate
 * for (`/api/pricing/reference`), not a list compiled into the page. The
 * frontend catalog only supplies the display label and the group heading.
 */
function useRateCardModels() {
  const [models, setModels] = useState<GpuModel[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    apiFetch<RateCard>("/api/pricing/reference")
      .then((card) => {
        if (!active) return;
        const rows = Object.entries(card.pricing || {})
          .filter(([, rate]) => (rate.base_rate_cad ?? 0) > 0 && (rate.vram_gb ?? 0) > 0)
          .map(([value, rate]) => {
            const known = getGpuModel(value);
            return { value, label: known?.label ?? value, vram_gb: rate.vram_gb as number, category: known?.category ?? "Other" };
          });
        setModels(rows);
        setFailed(false);
      })
      .catch(() => { if (active) setFailed(true); });
    return () => { active = false; };
  }, [attempt]);
  const retry = useCallback(() => { setFailed(false); setAttempt((n) => n + 1); }, []);
  return { models, failed, retry };
}

function groupByCategory(models: GpuModel[]): Record<string, GpuModel[]> {
  const order = [...GPU_CATEGORIES, "Other"];
  const grouped: Record<string, GpuModel[]> = {};
  for (const category of order) {
    const inCategory = models.filter((m) => m.category === category);
    if (inCategory.length) grouped[category] = inCategory;
  }
  for (const model of models) {
    if (!order.includes(model.category)) (grouped[model.category] ??= []).push(model);
  }
  return grouped;
}

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

  const { models: rateCard, failed, retry } = useRateCardModels();

  const selectedLabel = useMemo(() => {
    if (!value) return "";
    return rateCard?.find((g) => g.value === value)?.label ?? getGpuModel(value)?.label ?? value;
  }, [rateCard, value]);

  const filtered = useMemo(() => {
    const grouped = groupByCategory(rateCard ?? []);
    if (!search.trim()) return grouped;
    const q = search.toLowerCase();
    const result: Record<string, GpuModel[]> = {};
    for (const [cat, models] of Object.entries(grouped)) {
      const matches = models.filter(
        (m) => m.label.toLowerCase().includes(q) || m.value.toLowerCase().includes(q)
      );
      if (matches.length > 0) result[cat] = matches;
    }
    return result;
  }, [rateCard, search]);

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

  const effectiveHighlightIndex =
    flatFiltered.length === 0 ? 0 : Math.min(highlightIndex, flatFiltered.length - 1);

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
  }, [effectiveHighlightIndex, open]);

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
        setHighlightIndex((i) => Math.min(i + 1, Math.max(flatFiltered.length - 1, 0)));
        break;
      case "ArrowUp":
        e.preventDefault();
        setHighlightIndex((i) => Math.max(i - 1, 0));
        break;
      case "Enter":
        e.preventDefault();
        if (flatFiltered.length > 0 && flatFiltered[effectiveHighlightIndex]) {
          selectItem(flatFiltered[effectiveHighlightIndex]);
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

  const activeDescendant = flatFiltered[effectiveHighlightIndex]
    ? `gpu-opt-${flatFiltered[effectiveHighlightIndex].value}`
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
            setHighlightIndex(0);
          }
        }}
        onChange={(e) => {
          setSearch(e.target.value);
          setHighlightIndex(0);
        }}
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
          {failed ? (
            <div className="flex items-center justify-between gap-2 px-3 py-2 text-sm text-text-muted" role="status">
              <span>Couldn&rsquo;t load the GPU list.</span>
              <button type="button" className="min-h-9 rounded px-2 font-medium text-ice-blue hover:underline"
                onMouseDown={(e) => e.preventDefault()} onClick={retry}>Retry</button>
            </div>
          ) : rateCard === null ? (
            <div className="px-3 py-2 text-sm text-text-muted" role="status">Loading GPUs&hellip;</div>
          ) : Object.keys(filtered).length === 0 ? (
            <div className="px-3 py-2 text-sm text-text-muted">
              {search.trim() ? <>No GPUs match &ldquo;{search}&rdquo;</> : "No GPU models are published yet."}
            </div>
          ) : (
            Object.entries(filtered).map(([category, models]) => (
              <div key={category} role="group" aria-label={category}>
                <div className="sticky top-0 bg-surface-hover/80 px-3 py-1.5 text-xs font-semibold text-text-muted backdrop-blur-sm">
                  {category}
                </div>
                {models.map((model) => {
                  const idx = modelIndexMap.get(model.value) ?? 0;
                  const isHighlighted = idx === effectiveHighlightIndex;
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
