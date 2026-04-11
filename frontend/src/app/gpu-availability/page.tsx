"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Navbar } from "@/components/marketing/navbar";
import { Footer } from "@/components/marketing/footer";
import { Cpu, Activity, MapPin, Clock, RefreshCw } from "lucide-react";

interface HostSummary {
  gpu_model: string;
  available: number;
  total: number;
  vram_gb: number;
  price_cad: number;
  spot_cad: number;
  locations: string[];
}

const API = process.env.NEXT_PUBLIC_API_URL ?? "";

export default function GPUAvailabilityPage() {
  const [gpus, setGpus] = useState<HostSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  async function load() {
    setLoading(true);
    try {
      const [hostsRes, pricingRes] = await Promise.all([
        fetch(`${API}/hosts?active_only=true`).then((r) => r.json()),
        fetch(`${API}/api/pricing/reference`).then((r) => r.json()),
      ]);

      const hosts: Record<string, unknown>[] = hostsRes.hosts || [];
      const pricing: Record<string, Record<string, number>> = pricingRes.pricing || {};

      // Group hosts by GPU model
      const byModel = new Map<string, { available: number; total: number; vram: number; locations: Set<string> }>();
      for (const h of hosts) {
        const model = (h.gpu_model as string) || "Unknown";
        const vram = (h.vram_gb as number) || 0;
        const status = h.status as string;
        const province = (h.province as string) || "";
        if (!byModel.has(model)) {
          byModel.set(model, { available: 0, total: 0, vram, locations: new Set() });
        }
        const entry = byModel.get(model)!;
        entry.total++;
        if (status === "idle" || status === "active") entry.available++;
        if (province) entry.locations.add(province);
      }

      // Merge with pricing to also show GPUs with pricing but no active hosts
      for (const [model, info] of Object.entries(pricing)) {
        if (!byModel.has(model)) {
          const vram = model.includes("H100") ? 80 : model.includes("A100") ? 80 : model.includes("L40") ? 48 : model.includes("4090") ? 24 : model.includes("4080") ? 16 : model.includes("3090") ? 24 : 24;
          byModel.set(model, { available: 0, total: 0, vram, locations: new Set() });
        }
      }

      const summaries: HostSummary[] = [];
      for (const [model, entry] of byModel) {
        const ref = pricing[model] || {};
        summaries.push({
          gpu_model: model,
          available: entry.available,
          total: entry.total,
          vram_gb: entry.vram,
          price_cad: (ref.base_rate_cad as number) || 0,
          spot_cad: (ref.base_rate_cad as number) ? (ref.base_rate_cad as number) * 0.6 : 0,
          locations: Array.from(entry.locations),
        });
      }

      // Sort: available first, then by price
      summaries.sort((a, b) => (b.available > 0 ? 1 : 0) - (a.available > 0 ? 1 : 0) || a.price_cad - b.price_cad);

      setGpus(summaries);
      setLastUpdated(new Date());
    } catch {
      // fail silently, show empty state
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    const interval = setInterval(load, 30_000); // refresh every 30s
    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <Navbar />
      <main className="min-h-screen bg-navy">
        <div className="mx-auto max-w-6xl px-6 py-16">
          {/* Header */}
          <div className="text-center space-y-4 mb-12">
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-4 py-1.5 text-sm text-emerald-400">
              <Activity className="h-3.5 w-3.5" />
              Live Availability
            </div>
            <h1 className="text-4xl sm:text-5xl font-bold text-text-primary tracking-tight">
              GPU Availability
            </h1>
            <p className="text-lg text-text-secondary max-w-2xl mx-auto">
              Real-time GPU availability and pricing across our Canada-first compute network, built for teams worldwide.
              All prices in CAD. Eligible for the AI Compute Access Fund rebate.
            </p>
          </div>

          {/* Refresh bar */}
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2 text-sm text-text-muted">
              <Clock className="h-3.5 w-3.5" />
              {lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString()}` : "Loading..."}
            </div>
            <button
              onClick={load}
              disabled={loading}
              className="flex items-center gap-1.5 text-sm text-text-secondary hover:text-text-primary transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </button>
          </div>

          {/* GPU Cards */}
          {loading && gpus.length === 0 ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="h-48 rounded-xl border border-border bg-navy-light animate-pulse" />
              ))}
            </div>
          ) : gpus.length === 0 ? (
            <div className="text-center py-20 text-text-muted">
              <Cpu className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No GPUs currently listed. Check back soon.</p>
            </div>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {gpus.map((gpu) => (
                <div
                  key={gpu.gpu_model}
                  className="rounded-xl border border-border bg-navy-light p-6 space-y-4 hover:border-text-muted/30 transition-colors"
                >
                  {/* GPU Name + Availability */}
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent-red/10">
                        <Cpu className="h-5 w-5 text-accent-red" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-text-primary">{gpu.gpu_model}</h3>
                        <p className="text-xs text-text-muted">{gpu.vram_gb} GB VRAM</p>
                      </div>
                    </div>
                    <span
                      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${
                        gpu.available > 0
                          ? "bg-emerald-500/10 text-emerald-400"
                          : "bg-yellow-500/10 text-yellow-400"
                      }`}
                    >
                      <span className={`h-1.5 w-1.5 rounded-full ${gpu.available > 0 ? "bg-emerald-400" : "bg-yellow-400"}`} />
                      {gpu.available > 0 ? `${gpu.available} available` : "On request"}
                    </span>
                  </div>

                  {/* Pricing */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-lg bg-surface-hover/50 p-3">
                      <p className="text-xs text-text-muted mb-0.5">On-Demand</p>
                      <p className="text-lg font-bold text-text-primary">
                        {gpu.price_cad > 0 ? `$${gpu.price_cad.toFixed(2)}` : "—"}
                        <span className="text-xs font-normal text-text-muted">/hr</span>
                      </p>
                    </div>
                    <div className="rounded-lg bg-surface-hover/50 p-3">
                      <p className="text-xs text-text-muted mb-0.5">Spot</p>
                      <p className="text-lg font-bold text-ice-blue">
                        {gpu.spot_cad > 0 ? `$${gpu.spot_cad.toFixed(2)}` : "—"}
                        <span className="text-xs font-normal text-text-muted">/hr</span>
                      </p>
                    </div>
                  </div>

                  {/* Locations */}
                  {gpu.locations.length > 0 && (
                    <div className="flex items-center gap-1.5 text-xs text-text-muted">
                      <MapPin className="h-3 w-3" />
                      {gpu.locations.join(", ")}
                    </div>
                  )}

                  {/* CTA */}
                  <Link
                    href="/register"
                    className="block w-full text-center rounded-lg bg-accent-red px-4 py-2 text-sm font-medium text-white hover:bg-accent-red-hover transition-colors"
                  >
                    Deploy Now
                  </Link>
                </div>
              ))}
            </div>
          )}

          {/* Bottom CTA */}
          <div className="mt-16 text-center space-y-4">
            <h2 className="text-2xl font-bold text-text-primary">Need a GPU not listed here?</h2>
            <p className="text-text-secondary max-w-xl mx-auto">
              We&apos;re constantly adding new hardware to the network. Register as a provider to list your GPUs,
              or contact us for bulk availability.
            </p>
            <div className="flex justify-center gap-4">
              <Link
                href="/register"
                className="inline-flex h-10 items-center rounded-lg bg-accent-red px-6 text-sm font-medium text-white hover:bg-accent-red-hover transition-colors"
              >
                Get Started
              </Link>
              <Link
                href="/pricing"
                className="inline-flex h-10 items-center rounded-lg border border-border px-6 text-sm font-medium text-text-secondary hover:bg-surface-hover hover:text-text-primary transition-colors"
              >
                View Pricing
              </Link>
            </div>
          </div>
        </div>
      </main>
      <Footer />
    </>
  );
}
