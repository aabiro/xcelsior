"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/badge";
import { Input, Select } from "@/components/ui/input";
import { Pagination, usePagination } from "@/components/ui/pagination";
import { Briefcase, Plus, Search, RefreshCw, XCircle, ArrowUpDown, ArrowUp, ArrowDown } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { Instance } from "@/lib/api";
import { toast } from "sonner";
import { useEventStream } from "@/hooks/useEventStream";

type SortKey = "name" | "gpu_type" | "status" | "created_at";
type SortDir = "asc" | "desc";

export default function InstancesPage() {
  const [instances, setInstances] = useState<Instance[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [page, setPage] = useState(1);
  const api = useApi();
  const { t } = useLocale();

  const load = () => {
    setLoading(true);
    api.fetchInstances()
      .then((res) => setInstances(res.instances || []))
      .catch(() => toast.error("Failed to load instances"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  // Live updates — re-fetch list on job status changes
  useEventStream({
    eventTypes: ["job_status", "job_submitted"],
    onEvent: () => { load(); },
  });

  async function handleCancel(id: string) {
    try {
      await api.cancelInstance(id);
      toast.success("Instance cancelled");
      load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Cancel failed");
    }
  }

  const filtered = instances
    .filter((j) => {
      if (statusFilter !== "all" && j.status !== statusFilter) return false;
      if (search) {
        const q = search.toLowerCase();
        if (!j.name?.toLowerCase().includes(q) && !j.job_id?.toLowerCase().includes(q)) return false;
      }
      return true;
    })
    .sort((a, b) => {
      let av: string | number | undefined, bv: string | number | undefined;
      if (sortKey === "created_at") {
        av = a.created_at || a.submitted_at || "";
        bv = b.created_at || b.submitted_at || "";
      } else if (sortKey === "gpu_type") {
        av = a.gpu_type || a.gpu_model || "";
        bv = b.gpu_type || b.gpu_model || "";
      } else {
        av = (a as unknown as Record<string, unknown>)[sortKey] as string ?? "";
        bv = (b as unknown as Record<string, unknown>)[sortKey] as string ?? "";
      }
      const cmp = typeof av === "number" && typeof bv === "number" ? av - bv : String(av).localeCompare(String(bv));
      return sortDir === "asc" ? cmp : -cmp;
    });

  const { paginate, totalPages } = usePagination(filtered, 10);
  const pageItems = paginate(page);

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
        <h1 className="text-2xl font-bold">{t("dash.instances.title")}</h1>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
          </Button>
          <Link href="/dashboard/instances/new">
            <Button size="sm"><Plus className="h-3.5 w-3.5" /> {t("dash.instances.submit")}</Button>
          </Link>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" />
          <Input className="pl-9" placeholder={t("dash.instances.search")} value={search} onChange={(e) => setSearch(e.target.value)} />
        </div>
        <Select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
          <option value="all">{t("dash.instances.all_status")}</option>
          <option value="queued">{t("dash.instances.queued")}</option>
          <option value="running">{t("dash.instances.running")}</option>
          <option value="completed">{t("dash.instances.completed")}</option>
          <option value="failed">{t("dash.instances.failed")}</option>
          <option value="cancelled">{t("dash.instances.cancelled")}</option>
        </Select>
      </div>

      {loading ? (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-16 rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <Card className="p-12 text-center">
          <Briefcase className="mx-auto h-12 w-12 text-text-muted mb-4" />
          <h3 className="text-lg font-semibold mb-1">{t("dash.instances.empty")}</h3>
          <p className="text-sm text-text-secondary">{t("dash.instances.empty_desc")}</p>
        </Card>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-text-secondary">
                <th className="py-3 pr-4 text-left font-medium cursor-pointer select-none" onClick={() => toggleSort("name")}>
                  <span className="inline-flex items-center gap-1">{t("dash.instances.col_job")} <SortIcon col="name" /></span>
                </th>
                <th className="py-3 px-4 text-left font-medium cursor-pointer select-none" onClick={() => toggleSort("gpu_type")}>
                  <span className="inline-flex items-center gap-1">{t("dash.instances.col_gpu")} <SortIcon col="gpu_type" /></span>
                </th>
                <th className="py-3 px-4 text-center font-medium cursor-pointer select-none" onClick={() => toggleSort("status")}>
                  <span className="inline-flex items-center gap-1 justify-center">{t("dash.instances.col_status")} <SortIcon col="status" /></span>
                </th>
                <th className="py-3 px-4 text-center font-medium cursor-pointer select-none" onClick={() => toggleSort("created_at")}>
                  <span className="inline-flex items-center gap-1 justify-center">{t("dash.instances.col_created")} <SortIcon col="created_at" /></span>
                </th>
                <th className="py-3 px-4 text-right font-medium">{t("dash.instances.col_actions")}</th>
              </tr>
            </thead>
            <tbody>
              {pageItems.map((inst) => (
                <tr key={inst.job_id} className="border-b border-border/50 hover:bg-surface-hover transition-colors">
                  <td className="py-3 pr-4">
                    <Link href={`/dashboard/instances/${inst.job_id}`} className="font-medium text-ice-blue hover:underline">
                      {inst.name || inst.job_id}
                    </Link>
                  </td>
                  <td className="py-3 px-4 text-text-secondary">{inst.gpu_type || inst.gpu_model || "—"}</td>
                  <td className="py-3 px-4 text-center"><StatusBadge status={inst.status} /></td>
                  <td className="py-3 px-4 text-center text-text-muted">
                    {(inst.created_at || inst.submitted_at) ? new Date(inst.created_at || inst.submitted_at).toLocaleDateString() : "—"}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <div className="flex justify-end gap-1">
                      <Link href={`/dashboard/instances/${inst.job_id}`}>
                        <Button variant="ghost" size="sm">View</Button>
                      </Link>
                      {(inst.status === "queued" || inst.status === "running") && (
                        <Button variant="ghost" size="sm" onClick={() => handleCancel(inst.job_id)} className="text-accent-red hover:text-accent-red">
                          <XCircle className="h-3.5 w-3.5" />
                        </Button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-4 flex items-center justify-between">
            <span className="text-xs text-text-muted">{t("dash.instances.count", { count: filtered.length })}</span>
            <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
          </div>
        </div>
      )}
    </div>
  );
}
