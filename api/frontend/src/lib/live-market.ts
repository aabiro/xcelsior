"use client";

/**
 * The cheapest GPU a renter can launch right now, read from the public
 * availability endpoint (`GET /api/v2/gpu/available`). Marketing copy used a
 * fixed "from $0.30/hr" (the RTX 4090 reference rate) while the fleet had no
 * RTX 4090; a from-price now comes only from live, priced supply, and copy
 * falls back to saying no number at all.
 */
import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "";

export interface LiveGpu {
  gpuModel: string;
  vramGb: number;
  /** The listed rate a launch is billed, per GPU-hour. */
  pricePerHour: number;
  available: number;
  hosts: number;
  verifiedHosts: number;
  latestBenchmarkAt: number | null;
}

export type LiveMarket =
  | { state: "loading" }
  | { state: "empty" }
  | { state: "ready"; cheapest: LiveGpu; gpus: LiveGpu[] };

type Row = Record<string, unknown> & {
  evidence?: { hosts?: number; verified?: number; latest_benchmark_at?: number | null };
};

/** Priced rows with a GPU free now, cheapest first. Pure, so it is tested alone. */
export function liveGpus(rows: Row[]): LiveGpu[] {
  const out: LiveGpu[] = [];
  for (const row of rows) {
    const price = Number(row.price_per_hour_cad) || 0;
    const available = Number(row.count_available) || 0;
    if (price <= 0 || available <= 0) continue;
    out.push({
      gpuModel: String(row.gpu_model || ""),
      vramGb: Number(row.vram_gb) || 0,
      pricePerHour: price,
      available,
      hosts: Number(row.evidence?.hosts) || 0,
      verifiedHosts: Number(row.evidence?.verified) || 0,
      latestBenchmarkAt: row.evidence?.latest_benchmark_at ?? null,
    });
  }
  // Cheapest first; at the same price, verified supply first.
  return out.sort(
    (a, b) => a.pricePerHour - b.pricePerHour || b.verifiedHosts - a.verifiedHosts,
  );
}

export function formatFromPrice(value: number): string {
  return `$${value.toFixed(2)}`;
}

export function useLiveMarket(): LiveMarket {
  const [market, setMarket] = useState<LiveMarket>({ state: "loading" });
  useEffect(() => {
    let cancelled = false;
    fetch(`${API}/api/v2/gpu/available`, { credentials: "omit" })
      .then((res) => (res.ok ? res.json() : { gpus: [] }))
      .then((body: { gpus?: Row[] }) => {
        if (cancelled) return;
        const gpus = liveGpus(body.gpus || []);
        setMarket(gpus.length ? { state: "ready", cheapest: gpus[0], gpus } : { state: "empty" });
      })
      .catch(() => {
        if (!cancelled) setMarket({ state: "empty" });
      });
    return () => {
      cancelled = true;
    };
  }, []);
  return market;
}
