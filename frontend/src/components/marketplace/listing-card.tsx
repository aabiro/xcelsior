"use client";

import { Cpu, MapPin, Zap, ArrowRight } from "lucide-react";
import type { MarketplaceListing } from "@/lib/api";

interface ListingCardProps {
  listing: MarketplaceListing;
  onClick: () => void;
}

export function ListingCard({ listing, onClick }: ListingCardProps) {
  const price = listing.price_per_hour_cad ?? listing.price_per_hour ?? 0;
  const spot = price * 0.7;
  const region = listing.region || "Canada";
  const model = listing.gpu_model || "GPU";
  const vram = listing.vram_gb ? `${listing.vram_gb}GB` : null;

  return (
    <button
      type="button"
      role="button"
      onClick={onClick}
      aria-label={`Launch ${model} in ${region} for $${price.toFixed(2)} per hour`}
      className="group relative flex w-full flex-col gap-3 rounded-xl border border-border/60 bg-surface p-6 text-left transition-all duration-200 hover:border-ice-blue/40 hover:bg-surface-hover/40 hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ice-blue focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
    >
      {/* Top accent strip on hover */}
      <div
        className="absolute inset-x-0 top-0 h-px rounded-t-xl opacity-0 group-hover:opacity-100 transition-opacity"
        style={{
          background:
            "linear-gradient(90deg, transparent 0%, var(--color-accent-cyan) 30%, var(--color-accent-violet) 60%, var(--color-accent-red) 90%, transparent 100%)",
        }}
      />

      {/* Header: GPU name + status */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          <Cpu className="h-5 w-5 text-ice-blue shrink-0" />
          <div className="min-w-0">
            <h3 className="text-[18px] font-semibold leading-tight truncate">
              {model}
              {vram && <span className="ml-1.5 text-text-muted font-normal">· {vram}</span>}
            </h3>
          </div>
        </div>
        <span className="shrink-0 rounded-full bg-emerald/10 px-2 py-0.5 text-[10px] font-medium text-emerald uppercase tracking-wide">
          {listing.status || "available"}
        </span>
      </div>

      {/* Meta row */}
      <div className="flex items-center gap-4 text-xs text-text-muted">
        <span className="inline-flex items-center gap-1">
          <MapPin className="h-3.5 w-3.5" />
          {region}
        </span>
        {listing.reputation_score != null && (
          <span className="inline-flex items-center gap-1">
            <span className="text-accent-gold">★</span>
            {listing.reputation_score.toFixed(1)}
          </span>
        )}
        {listing.hostname && (
          <span className="truncate text-text-muted/70">{listing.hostname}</span>
        )}
      </div>

      {/* Price block */}
      <div className="mt-1 flex items-end justify-between gap-3">
        <div>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold font-mono tracking-tight">
              ${price.toFixed(2)}
            </span>
            <span className="text-xs text-text-muted">/hr</span>
          </div>
          <div className="mt-0.5 flex items-center gap-1 text-[11px] font-mono text-emerald/80">
            <Zap className="h-3 w-3" />
            ~${spot.toFixed(2)}/hr spot
          </div>
        </div>
        <span className="inline-flex items-center gap-1 rounded-full border border-ice-blue/30 bg-ice-blue/5 px-2.5 py-1 text-[11px] font-medium text-ice-blue opacity-0 group-hover:opacity-100 transition-opacity">
          Launch <ArrowRight className="h-3 w-3" />
        </span>
      </div>
    </button>
  );
}
