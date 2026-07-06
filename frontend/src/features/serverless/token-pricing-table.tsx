"use client";

import type { TokenPricingQuote } from "./constants";
import { PRESET_MODELS } from "./constants";

interface TokenPricingTableProps {
  quotes: Record<string, TokenPricingQuote>;
  selectedModel?: string;
}

/** Novita/RunPod-style per-million token rate table for preset LLM models. */
export function TokenPricingTable({ quotes, selectedModel }: TokenPricingTableProps) {
  const rows = PRESET_MODELS.filter((m) => m.task !== "rerank");
  if (!rows.length) return null;

  return (
    <div className="rounded-xl border border-border bg-surface/60 overflow-hidden">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border bg-surface-hover/50 text-text-muted">
            <th className="text-left font-medium px-3 py-2">Model</th>
            <th className="text-right font-medium px-3 py-2">Input / 1M</th>
            <th className="text-right font-medium px-3 py-2">Output / 1M</th>
            <th className="text-right font-medium px-3 py-2 hidden sm:table-cell">Cached in / 1M</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => {
            const q = quotes[m.id];
            const inRate = q?.input_price_cad_per_m;
            const outRate = q?.output_price_cad_per_m;
            const cached = q?.cached_input_price_cad_per_m;
            const selected = selectedModel === m.id;
            if (inRate == null || outRate == null) return null;
            return (
              <tr
                key={m.id}
                className={selected ? "bg-accent-cyan/10" : "hover:bg-surface-hover/40"}
              >
                <td className="px-3 py-2 font-medium truncate max-w-[10rem]">{m.label}</td>
                <td className="px-3 py-2 text-right font-mono text-accent-cyan">${inRate.toFixed(2)}</td>
                <td className="px-3 py-2 text-right font-mono">${outRate.toFixed(2)}</td>
                <td className="px-3 py-2 text-right font-mono text-text-muted hidden sm:table-cell">
                  {cached != null ? `$${cached.toFixed(3)}` : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="text-[10px] text-text-muted px-3 py-1.5 border-t border-border">
        Token rates billed in parallel with worker uptime (¢/s). Instances &amp; volumes: GPU time only.
      </p>
    </div>
  );
}