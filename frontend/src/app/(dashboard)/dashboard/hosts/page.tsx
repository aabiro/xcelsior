"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge, StatusBadge } from "@/components/ui/badge";
import { Input, Label, Select } from "@/components/ui/input";
import { Pagination, usePagination } from "@/components/ui/pagination";
import { Server, Plus, Search, RefreshCw, ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { Host } from "@/lib/api";
import { toast } from "sonner";

type SortKey = "hostname" | "gpu_model" | "status" | "vram_gb" | "cost_per_hour";
type SortDir = "asc" | "desc";

export default function HostsPage() {
  const [hosts, setHosts] = useState<Host[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("hostname");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [page, setPage] = useState(1);
  const [showRegister, setShowRegister] = useState(false);
  const api = useApi();
  const { t } = useLocale();

  const load = () => {
    setLoading(true);
    api.fetchHosts()
      .then((res) => setHosts(res.hosts || []))
      .catch(() => toast.error("Failed to load hosts"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const filtered = hosts
    .filter((h) => {
      if (statusFilter !== "all" && h.status !== statusFilter) return false;
      if (search && !h.hostname?.toLowerCase().includes(search.toLowerCase()) && !h.host_id?.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    })
    .sort((a, b) => {
      const av = (a as unknown as Record<string, unknown>)[sortKey] ?? "";
      const bv = (b as unknown as Record<string, unknown>)[sortKey] ?? "";
      const cmp = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv));
      return sortDir === "asc" ? cmp : -cmp;
    });

  const { paginate, totalPages } = usePagination(filtered, 10);
  const pageItems = paginate(page);

  // Reset to page 1 when filters change
  useEffect(() => { setPage(1); }, [search, statusFilter, sortKey, sortDir]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir(sortDir === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("asc"); }
  }

  function SortIcon({ col }: { col: SortKey }) {
    if (sortKey !== col) return <ArrowUpDown className="h-3 w-3 opacity-40" />;
    return sortDir === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.hosts.title")}</h1>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
          </Button>
          <Button size="sm" onClick={() => setShowRegister(!showRegister)}>
            <Plus className="h-3.5 w-3.5" /> {t("dash.hosts.register")}
          </Button>
        </div>
      </div>

      {showRegister && <RegisterHostForm api={api} onDone={() => { setShowRegister(false); load(); }} />}

      {/* Filters */}
      <div className="flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" />
          <Input className="pl-9" placeholder={t("dash.hosts.search")} value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <Select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
          <option value="all">{t("dash.hosts.all_status")}</option>
          <option value="active">{t("dash.hosts.active")}</option>
          <option value="offline">{t("dash.hosts.offline")}</option>
          <option value="maintenance">{t("dash.hosts.maintenance")}</option>
        </Select>
      </div>

      {/* Table */}
      {loading ? (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-16 rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <Card className="p-12 text-center">
          <Server className="mx-auto h-12 w-12 text-text-muted mb-4" />
          <h3 className="text-lg font-semibold mb-1">{t("dash.hosts.empty")}</h3>
          <p className="text-sm text-text-secondary">{t("dash.hosts.empty_desc")}</p>
        </Card>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-text-secondary">
                <th className="py-3 pr-4 text-left font-medium cursor-pointer select-none" onClick={() => toggleSort("hostname")}>
                  <span className="inline-flex items-center gap-1">{t("dash.hosts.col_hostname")} <SortIcon col="hostname" /></span>
                </th>
                <th className="py-3 px-4 text-left font-medium cursor-pointer select-none" onClick={() => toggleSort("gpu_model")}>
                  <span className="inline-flex items-center gap-1">{t("dash.hosts.col_gpu")} <SortIcon col="gpu_model" /></span>
                </th>
                <th className="py-3 px-4 text-center font-medium cursor-pointer select-none" onClick={() => toggleSort("status")}>
                  <span className="inline-flex items-center gap-1 justify-center">{t("dash.hosts.col_status")} <SortIcon col="status" /></span>
                </th>
                <th className="py-3 px-4 text-center font-medium cursor-pointer select-none" onClick={() => toggleSort("vram_gb")}>
                  <span className="inline-flex items-center gap-1 justify-center">{t("dash.hosts.col_vram")} <SortIcon col="vram_gb" /></span>
                </th>
                <th className="py-3 px-4 text-center font-medium cursor-pointer select-none" onClick={() => toggleSort("cost_per_hour")}>
                  <span className="inline-flex items-center gap-1 justify-center">{t("dash.hosts.col_price")} <SortIcon col="cost_per_hour" /></span>
                </th>
                <th className="py-3 px-4 text-right font-medium">{t("dash.hosts.col_actions")}</th>
              </tr>
            </thead>
            <tbody>
              {pageItems.map((host) => (
                <tr key={host.host_id} className="border-b border-border/50 hover:bg-surface-hover transition-colors">
                  <td className="py-3 pr-4">
                    <Link href={`/dashboard/hosts/${host.host_id}`} className="font-medium text-ice-blue hover:underline">
                      {host.hostname || host.host_id}
                    </Link>
                  </td>
                  <td className="py-3 px-4 text-text-secondary">{host.gpu_model || "—"}</td>
                  <td className="py-3 px-4 text-center"><StatusBadge status={host.status} /></td>
                  <td className="py-3 px-4 text-center font-mono">{host.vram_gb ? `${host.vram_gb}GB` : "—"}</td>
                  <td className="py-3 px-4 text-center font-mono">{host.cost_per_hour ? `$${host.cost_per_hour}/hr` : "—"}</td>
                  <td className="py-3 px-4 text-right">
                    <Link href={`/dashboard/hosts/${host.host_id}`}>
                      <Button variant="ghost" size="sm">View</Button>
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-4 flex items-center justify-between">
            <span className="text-xs text-text-muted">{t("dash.hosts.count", { count: filtered.length })}</span>
            <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
          </div>
        </div>
      )}
    </div>
  );
}

function RegisterHostForm({ api, onDone }: { api: ReturnType<typeof useApi>; onDone: () => void }) {
  const { t } = useLocale();
  const [hostname, setHostname] = useState("");
  const [gpuModel, setGpuModel] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    try {
      await api.registerHost({ hostname, gpu_model: gpuModel });
      toast.success("Host registered");
      onDone();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to register host");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card className="p-6">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4 sm:flex-row sm:items-end">
        <div className="flex-1 space-y-2">
          <Label>{t("dash.hosts.form_hostname")}</Label>
          <Input value={hostname} onChange={(e) => setHostname(e.target.value)} required placeholder={t("dash.hosts.hostname_placeholder")} />
        </div>
        <div className="flex-1 space-y-2">
          <Label>{t("dash.hosts.form_gpu")}</Label>
          <Input value={gpuModel} onChange={(e) => setGpuModel(e.target.value)} required placeholder={t("dash.hosts.gpu_placeholder")} />
        </div>
        <Button type="submit" disabled={loading}>{loading ? t("dash.hosts.registering") : t("dash.hosts.register_btn")}</Button>
      </form>
    </Card>
  );
}
