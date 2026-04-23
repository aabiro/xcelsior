"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import Image from "next/image";
import { useSearchParams, useRouter } from "next/navigation";
import { Search, RefreshCw, Zap, X, Cpu, Globe2 } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { MarketplaceListing } from "@/lib/api";
import { LaunchInstanceModal } from "@/components/instances/launch-instance-modal";
import { ListingCard } from "@/components/marketplace/listing-card";
import { Pagination, usePagination } from "@/components/ui/pagination";
import { toast } from "sonner";

export default function MarketplacePage() {
  const [listings, setListings] = useState<MarketplaceListing[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [gpuFilter, setGpuFilter] = useState("all");
  const [regionFilter, setRegionFilter] = useState("all");
  const [sortBy, setSortBy] = useState("price_asc");
  const [rentListing, setRentListing] = useState<MarketplaceListing | null>(null);
  const [page, setPage] = useState(1);
  const api = useApi();
  const { t } = useLocale();
  const searchParams = useSearchParams();
  const router = useRouter();

  const templateId = searchParams.get("template");

  const clearTemplate = () => {
    const params = new URLSearchParams(searchParams.toString());
    params.delete("template");
    router.replace(
      `/dashboard/marketplace${params.size > 0 ? `?${params}` : ""}`,
    );
  };

  const load = useCallback(() => {
    setLoading(true);
    api
      .fetchMarketplace()
      .then((res) => setListings(res.listings || []))
      .catch(() => toast.error("Failed to load marketplace"))
      .finally(() => setLoading(false));
  }, [api]);

  useEffect(() => {
    load();
  }, [load]);

  const gpuModels = useMemo(
    () => [...new Set(listings.map((l) => l.gpu_model).filter((v): v is string => !!v))],
    [listings],
  );
  const regions = useMemo(
    () => [...new Set(listings.map((l) => l.region).filter((v): v is string => !!v))],
    [listings],
  );

  const filtered = useMemo(() => {
    return listings
      .filter((l) => {
        if (gpuFilter !== "all" && l.gpu_model !== gpuFilter) return false;
        if (regionFilter !== "all" && l.region !== regionFilter) return false;
        if (search) {
          const q = search.toLowerCase();
          if (
            !l.gpu_model?.toLowerCase().includes(q) &&
            !l.hostname?.toLowerCase().includes(q) &&
            !l.region?.toLowerCase().includes(q)
          )
            return false;
        }
        return true;
      })
      .sort((a, b) => {
        const priceA = a.price_per_hour_cad || a.price_per_hour || 0;
        const priceB = b.price_per_hour_cad || b.price_per_hour || 0;
        if (sortBy === "price_asc") return priceA - priceB;
        if (sortBy === "price_desc") return priceB - priceA;
        if (sortBy === "vram") return (b.vram_gb || 0) - (a.vram_gb || 0);
        return 0;
      });
  }, [listings, gpuFilter, regionFilter, search, sortBy]);

  // Stats — derived from full listings, not filtered
  const stats = useMemo(() => {
    const prices = listings
      .map((l) => l.price_per_hour_cad ?? l.price_per_hour ?? 0)
      .filter((p) => p > 0);
    return {
      lowestPrice: prices.length ? Math.min(...prices) : 0,
      totalGpus: listings.length,
      models: gpuModels.length,
      regions: regions.length,
    };
  }, [listings, gpuModels, regions]);

  const { paginate, totalPages } = usePagination(filtered, 12);
  const pageItems = paginate(page);

  useEffect(() => {
    setPage(1);
  }, [search, gpuFilter, regionFilter, sortBy]);

  const clearFilters = () => {
    setSearch("");
    setGpuFilter("all");
    setRegionFilter("all");
    setSortBy("price_asc");
  };

  const hasFilters = !!search || gpuFilter !== "all" || regionFilter !== "all";

  return (
    <div className="space-y-6">
      {/* ── Hero band ────────────────────────────────────────────── */}
      <section className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-surface via-surface to-surface-hover/40 px-6 py-6 sm:px-10 sm:py-8">
        <div className="flex flex-col-reverse items-start justify-between gap-6 sm:flex-row sm:items-center">
          <div className="flex-1 space-y-2">
            <h1 className="text-3xl font-bold tracking-tight">
              {t("dash.market.title")}
            </h1>
            <p className="max-w-xl text-sm text-text-secondary">
              GPU compute on Canadian-owned hardware. Transparent prices, hourly
              billing, no commitment.
            </p>
            <div className="flex items-center gap-2 pt-2">
              <button
                onClick={load}
                className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-surface px-3 py-1.5 text-xs font-medium text-text-secondary hover:border-ice-blue/40 hover:text-text-primary transition-colors"
              >
                <RefreshCw className="h-3 w-3" />
                {t("common.refresh")}
              </button>
            </div>
          </div>
          <div className="opacity-90">
            <Image
              src="/gpu.svg"
              alt=""
              width={280}
              height={160}
              priority
              className="hidden dark:block w-auto h-32 sm:h-40"
            />
            <Image
              src="/gpu-light.svg"
              alt=""
              width={280}
              height={160}
              priority
              className="block dark:hidden w-auto h-32 sm:h-40"
            />
          </div>
        </div>

        {/* Stat chips */}
        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatChip
            label="From"
            value={stats.lowestPrice > 0 ? `$${stats.lowestPrice.toFixed(2)}` : "—"}
            sub="/hr on-demand"
            accent="ice-blue"
          />
          <StatChip
            label="GPUs"
            value={stats.totalGpus.toString()}
            sub="available now"
          />
          <StatChip
            label="Models"
            value={stats.models.toString()}
            sub="distinct"
          />
          <StatChip
            label="Regions"
            value={stats.regions.toString()}
            sub="Canada-wide"
          />
        </div>
      </section>

      {/* Brand-line separator */}
      <div className="brand-line" />

      {/* ── Template banner ─────────────────────────────────────── */}
      {templateId && (
        <div className="flex items-center gap-3 rounded-xl border border-ice-blue/30 bg-ice-blue/5 px-4 py-3 text-sm">
          <Zap className="h-4 w-4 text-ice-blue shrink-0" />
          <span className="flex-1">
            Launching from template{" "}
            <code className="rounded bg-surface px-1.5 py-0.5 font-mono text-xs">
              {templateId}
            </code>
            {" "}— pick a GPU to continue.
          </span>
          <button
            onClick={clearTemplate}
            className="text-text-muted hover:text-text-primary transition-colors"
            aria-label="Cancel template launch"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* ── Filter bar (pill controls) ──────────────────────────── */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-[220px] flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-muted" />
          <input
            className="w-full rounded-full border border-border/60 bg-surface py-2 pl-9 pr-3 text-sm placeholder:text-text-muted focus:border-ice-blue/50 focus:outline-none focus:ring-1 focus:ring-ice-blue/30"
            placeholder={t("dash.market.search")}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <PillSelect
          value={gpuFilter}
          onChange={setGpuFilter}
          icon={<Cpu className="h-3 w-3" />}
          options={[
            { value: "all", label: "All GPUs" },
            ...gpuModels.map((m) => ({ value: m, label: m })),
          ]}
        />

        <PillSelect
          value={regionFilter}
          onChange={setRegionFilter}
          icon={<Globe2 className="h-3 w-3" />}
          options={[
            { value: "all", label: "All regions" },
            ...regions.map((r) => ({ value: r, label: r })),
          ]}
        />

        <PillSelect
          value={sortBy}
          onChange={setSortBy}
          options={[
            { value: "price_asc", label: t("dash.market.price_low") },
            { value: "price_desc", label: t("dash.market.price_high") },
            { value: "vram", label: t("dash.market.vram_most") },
          ]}
        />

        {hasFilters && (
          <button
            onClick={clearFilters}
            className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-surface px-3 py-1.5 text-xs text-text-muted hover:text-text-primary hover:border-ice-blue/40 transition-colors"
          >
            <X className="h-3 w-3" />
            Clear
          </button>
        )}

        <span className="ml-auto text-xs text-text-muted">
          {filtered.length} of {listings.length}
        </span>
      </div>

      {/* Brand-line separator */}
      <div className="brand-line" />

      {/* ── Grid ────────────────────────────────────────────────── */}
      {loading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-52 rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-border/60 bg-surface/40 px-6 py-16 text-center">
          <Image
            src="/gpu.svg"
            alt=""
            width={120}
            height={70}
            className="mb-5 opacity-25 hidden dark:block"
          />
          <Image
            src="/gpu-light.svg"
            alt=""
            width={120}
            height={70}
            className="mb-5 opacity-25 block dark:hidden"
          />
          <h3 className="text-lg font-semibold mb-1">
            {hasFilters ? "No matches" : t("dash.market.empty")}
          </h3>
          <p className="text-sm text-text-secondary mb-4 max-w-sm">
            {hasFilters
              ? "Try widening your search or removing a filter."
              : t("dash.market.empty_desc")}
          </p>
          {hasFilters && (
            <button
              onClick={clearFilters}
              className="inline-flex items-center gap-1.5 rounded-full border border-ice-blue/40 bg-ice-blue/5 px-4 py-2 text-sm font-medium text-ice-blue hover:bg-ice-blue/10 transition-colors"
            >
              <X className="h-3.5 w-3.5" />
              Clear filters
            </button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {pageItems.map((listing) => (
            <ListingCard
              key={listing.host_id}
              listing={listing}
              onClick={() => setRentListing(listing)}
            />
          ))}
        </div>
      )}

      {!loading && filtered.length > 0 && totalPages > 1 && (
        <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
      )}

      {/* Launch Instance Modal */}
      <LaunchInstanceModal
        open={!!rentListing}
        onClose={() => setRentListing(null)}
        listing={rentListing ?? undefined}
        templateId={templateId ?? undefined}
        onLaunched={() => setRentListing(null)}
      />
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────

function StatChip({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: "ice-blue";
}) {
  return (
    <div className="rounded-xl border border-border/60 bg-surface/60 px-4 py-3">
      <div className="text-[10px] font-medium uppercase tracking-wider text-text-muted">
        {label}
      </div>
      <div className="mt-0.5 flex items-baseline gap-1.5">
        <span
          className={`text-xl font-bold font-mono ${
            accent === "ice-blue" ? "text-ice-blue" : "text-text-primary"
          }`}
        >
          {value}
        </span>
        {sub && <span className="text-xs text-text-muted">{sub}</span>}
      </div>
    </div>
  );
}

function PillSelect({
  value,
  onChange,
  options,
  icon,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  icon?: React.ReactNode;
}) {
  return (
    <div className="relative">
      {icon && (
        <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-text-muted">
          {icon}
        </span>
      )}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`appearance-none rounded-full border border-border/60 bg-surface py-2 pr-7 text-xs font-medium text-text-primary focus:border-ice-blue/50 focus:outline-none focus:ring-1 focus:ring-ice-blue/30 ${
          icon ? "pl-8" : "pl-3"
        }`}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-text-muted text-[10px]">
        ▾
      </span>
    </div>
  );
}
