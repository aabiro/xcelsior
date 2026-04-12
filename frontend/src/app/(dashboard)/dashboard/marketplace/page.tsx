"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input, Select } from "@/components/ui/input";
import { Store, Search, RefreshCw, Cpu, MapPin, Zap } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { MarketplaceListing } from "@/lib/api";
import { LaunchInstanceModal } from "@/components/instances/launch-instance-modal";
import { Pagination, usePagination } from "@/components/ui/pagination";
import { toast } from "sonner";

export default function MarketplacePage() {
  const [listings, setListings] = useState<MarketplaceListing[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [gpuFilter, setGpuFilter] = useState("all");
  const [sortBy, setSortBy] = useState("price_asc");
  const [rentListing, setRentListing] = useState<MarketplaceListing | null>(null);
  const [page, setPage] = useState(1);
  const api = useApi();
  const { t } = useLocale();

  const load = useCallback(() => {
    setLoading(true);
    api.fetchMarketplace()
      .then((res) => setListings(res.listings || []))
      .catch(() => toast.error("Failed to load marketplace"))
      .finally(() => setLoading(false));
  }, [api]);

  useEffect(() => { load(); }, [load]);

  const filtered = listings
    .filter((l) => {
      if (gpuFilter !== "all" && l.gpu_model !== gpuFilter) return false;
      if (search) {
        const q = search.toLowerCase();
        if (
          !l.gpu_model?.toLowerCase().includes(q) &&
          !l.hostname?.toLowerCase().includes(q) &&
          !l.region?.toLowerCase().includes(q)
        ) return false;
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

  const gpuModels = [...new Set(listings.map((l) => l.gpu_model).filter(Boolean))];

  const { paginate, totalPages } = usePagination(filtered, 12);
  const pageItems = paginate(page);

  useEffect(() => { setPage(1); }, [search, gpuFilter, sortBy]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">{t("dash.market.title")}</h1>
          <p className="text-sm text-text-muted">{t("dash.market.available", { count: filtered.length, s: filtered.length !== 1 ? "s" : "" })}</p>
        </div>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
        </Button>
      </div>

      {/* Filters */}
      <div className="flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" />
          <Input className="pl-9" placeholder={t("dash.market.search")} value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <Select value={gpuFilter} onChange={(e) => setGpuFilter(e.target.value)}>
          <option value="all">{t("dash.market.all_gpus")}</option>
          {gpuModels.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </Select>
        <Select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
          <option value="price_asc">{t("dash.market.price_low")}</option>
          <option value="price_desc">{t("dash.market.price_high")}</option>
          <option value="vram">{t("dash.market.vram_most")}</option>
        </Select>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-48 rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <Card className="p-12 text-center">
          <Store className="mx-auto h-12 w-12 text-text-muted mb-4" />
          <h3 className="text-lg font-semibold mb-1">{t("dash.market.empty")}</h3>
          <p className="text-sm text-text-secondary">{t("dash.market.empty_desc")}</p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {pageItems.map((listing) => (
            <Card key={listing.host_id} className="p-5 card-hover">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Cpu className="h-5 w-5 text-ice-blue" />
                  <span className="font-semibold">{listing.gpu_model || "GPU"}</span>
                </div>
                <Badge variant="active">{listing.status || "available"}</Badge>
              </div>
              <div className="space-y-2 mb-4">
                <div className="flex items-center gap-2 text-sm text-text-secondary">
                  <MapPin className="h-3.5 w-3.5" />
                  {listing.region || "Canada"}
                </div>
                <div className="flex items-center gap-2 text-sm text-text-secondary">
                  <Zap className="h-3.5 w-3.5" />
                  {listing.vram_gb ? `${listing.vram_gb}GB VRAM` : "—"}
                </div>
                {listing.reputation_score != null && (
                  <div className="flex items-center gap-2 text-sm text-text-secondary">
                    <span className="text-accent-gold">★</span>
                    {listing.reputation_score.toFixed(1)}
                    {listing.reputation_tier && <span className="text-xs capitalize text-text-muted">({listing.reputation_tier})</span>}
                  </div>
                )}
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-lg font-bold font-mono">
                    ${(listing.price_per_hour_cad || listing.price_per_hour)?.toFixed(2) || "—"}<span className="text-xs text-text-muted">{t("dash.market.per_hr")}</span>
                  </span>
                  <p className="text-xs text-emerald font-mono">
                    ~${((listing.price_per_hour_cad || listing.price_per_hour || 0) * 0.7).toFixed(2)}/hr spot
                  </p>
                </div>
                <Button size="sm" onClick={() => setRentListing(listing)}>{t("dash.market.rent")}</Button>
              </div>
            </Card>
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
        onLaunched={() => setRentListing(null)}
      />
    </div>
  );
}
