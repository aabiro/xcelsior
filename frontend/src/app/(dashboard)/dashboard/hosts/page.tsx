"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge, StatusBadge } from "@/components/ui/badge";
import { Input, Label, Select } from "@/components/ui/input";
import { Pagination, usePagination } from "@/components/ui/pagination";
import { Dialog } from "@/components/ui/dialog";
import { Server, Plus, Search, RefreshCw, ArrowUpDown, ArrowUp, ArrowDown, Terminal, Download, Cpu, Copy, Check } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { Host } from "@/lib/api";
import { toast } from "sonner";
import { useEventStream } from "@/hooks/useEventStream";

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
  const [showInstall, setShowInstall] = useState(false);
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

  // Live updates — re-fetch list on host changes
  useEventStream({
    eventTypes: ["host_registered", "host_removed", "job_status"],
    onEvent: () => { load(); },
  });

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

  const activeHosts = hosts.filter((h) => h.status === "active").length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold">{t("dash.hosts.title")}</h1>
          <p className="text-sm text-text-secondary mt-1">
            {hosts.length > 0
              ? `${activeHosts} active of ${hosts.length} registered host${hosts.length !== 1 ? "s" : ""}`
              : "Register a GPU host to start earning"}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
          </Button>
          <Button size="sm" onClick={() => setShowRegister(true)}>
            <Plus className="h-3.5 w-3.5" /> {t("dash.hosts.register")}
          </Button>
          <Button variant="outline" size="sm" onClick={() => setShowInstall(true)}>
            <Terminal className="h-3.5 w-3.5" /> Install Worker
          </Button>
        </div>
      </div>

      {/* Register Host Modal */}
      <Dialog
        open={showRegister}
        onClose={() => setShowRegister(false)}
        title="Register a New Host"
        description="Add a GPU machine to the Xcelsior network. You'll need the hostname and GPU model."
      >
        <RegisterHostForm api={api} onDone={() => { setShowRegister(false); load(); }} />
      </Dialog>

      {/* Install Worker Modal */}
      <Dialog
        open={showInstall}
        onClose={() => setShowInstall(false)}
        title="Install Worker Agent"
        description="Set up the Xcelsior worker agent on your GPU host to start accepting compute jobs."
        maxWidth="max-w-2xl"
      >
        <InstallWorkerSection />
      </Dialog>

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
          <p className="text-sm text-text-secondary mb-6">{t("dash.hosts.empty_desc")}</p>
          <div className="flex justify-center gap-3">
            <Button size="sm" onClick={() => setShowRegister(true)}>
              <Plus className="h-3.5 w-3.5" /> {t("dash.hosts.register")}
            </Button>
            <Button variant="outline" size="sm" onClick={() => setShowInstall(true)}>
              <Terminal className="h-3.5 w-3.5" /> Install Worker
            </Button>
          </div>
        </Card>
      ) : (
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-surface-hover/50">
                  <th className="py-3 pl-4 pr-3 text-left font-medium text-text-secondary cursor-pointer select-none" onClick={() => toggleSort("hostname")}>
                    <span className="inline-flex items-center gap-1">{t("dash.hosts.col_hostname")} <SortIcon col="hostname" /></span>
                  </th>
                  <th className="py-3 px-3 text-left font-medium text-text-secondary cursor-pointer select-none" onClick={() => toggleSort("gpu_model")}>
                    <span className="inline-flex items-center gap-1">{t("dash.hosts.col_gpu")} <SortIcon col="gpu_model" /></span>
                  </th>
                  <th className="py-3 px-3 text-center font-medium text-text-secondary cursor-pointer select-none" onClick={() => toggleSort("status")}>
                    <span className="inline-flex items-center gap-1 justify-center">{t("dash.hosts.col_status")} <SortIcon col="status" /></span>
                  </th>
                  <th className="py-3 px-3 text-center font-medium text-text-secondary cursor-pointer select-none" onClick={() => toggleSort("vram_gb")}>
                    <span className="inline-flex items-center gap-1 justify-center">{t("dash.hosts.col_vram")} <SortIcon col="vram_gb" /></span>
                  </th>
                  <th className="py-3 px-3 text-center font-medium text-text-secondary cursor-pointer select-none" onClick={() => toggleSort("cost_per_hour")}>
                    <span className="inline-flex items-center gap-1 justify-center">{t("dash.hosts.col_price")} <SortIcon col="cost_per_hour" /></span>
                  </th>
                  <th className="py-3 pl-3 pr-4 text-right font-medium text-text-secondary">{t("dash.hosts.col_actions")}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40">
                {pageItems.map((host) => (
                  <tr key={host.host_id} className="group hover:bg-surface-hover/60 transition-colors">
                    <td className="py-3.5 pl-4 pr-3">
                      <Link href={`/dashboard/hosts/${host.host_id}`} className="font-medium text-ice-blue hover:underline">
                        {host.hostname || host.host_id}
                      </Link>
                    </td>
                    <td className="py-3.5 px-3">
                      <div className="flex items-center gap-1.5 text-text-secondary">
                        <Cpu className="h-3.5 w-3.5 shrink-0 opacity-50" />
                        {host.gpu_model || "—"}
                      </div>
                    </td>
                    <td className="py-3.5 px-3 text-center"><StatusBadge status={host.status} /></td>
                    <td className="py-3.5 px-3 text-center font-mono text-text-secondary">{host.vram_gb ? `${host.vram_gb} GB` : "—"}</td>
                    <td className="py-3.5 px-3 text-center font-mono text-text-secondary">{host.cost_per_hour ? `$${host.cost_per_hour}/hr` : "—"}</td>
                    <td className="py-3.5 pl-3 pr-4 text-right">
                      <Link href={`/dashboard/hosts/${host.host_id}`}>
                        <Button variant="ghost" size="sm">View</Button>
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="border-t border-border px-4 py-3 flex items-center justify-between">
            <span className="text-xs text-text-muted">{t("dash.hosts.count", { count: filtered.length })}</span>
            <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
          </div>
        </Card>
      )}
    </div>
  );
}

/* ── Register Host Form (renders inside Dialog) ─────────────────── */

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
    <form onSubmit={handleSubmit} className="space-y-4 mt-2">
      <div className="space-y-2">
        <Label>{t("dash.hosts.form_hostname")}</Label>
        <Input value={hostname} onChange={(e) => setHostname(e.target.value)} required placeholder={t("dash.hosts.hostname_placeholder")} />
      </div>
      <div className="space-y-2">
        <Label>{t("dash.hosts.form_gpu")}</Label>
        <Input value={gpuModel} onChange={(e) => setGpuModel(e.target.value)} required placeholder={t("dash.hosts.gpu_placeholder")} />
      </div>
      <div className="flex justify-end gap-3 pt-2">
        <Button type="submit" disabled={loading}>
          {loading ? t("dash.hosts.registering") : t("dash.hosts.register_btn")}
        </Button>
      </div>
    </form>
  );
}

/* ── Install Worker Section (renders inside Dialog) ──────────────── */

function InstallWorkerSection() {
  const [copied, setCopied] = useState<string | null>(null);

  function copy(label: string, text: string) {
    navigator.clipboard.writeText(text);
    setCopied(label);
    setTimeout(() => setCopied(null), 2000);
  }

  const envTemplate = `XCELSIOR_HOST_ID=<your-host-id>
XCELSIOR_SCHEDULER_URL=https://api.xcelsior.ai
XCELSIOR_API_TOKEN=<your-api-token>
XCELSIOR_COST_PER_HOUR=0.50`;

  const installCmd = `npx xcelsior setup --mode provide`;

  const systemdUnit = `[Unit]
Description=Xcelsior Worker Agent
After=network-online.target docker.service
Requires=docker.service

[Service]
EnvironmentFile=/root/.xcelsior/.env
ExecStart=/usr/local/bin/xcelsior-worker
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target`;

  function CopyBtn({ label, text }: { label: string; text: string }) {
    return (
      <button
        onClick={() => copy(label, text)}
        className="absolute top-2 right-2 flex items-center gap-1 text-xs px-2 py-1 rounded bg-surface border border-border hover:bg-surface-hover transition-colors"
      >
        {copied === label ? <><Check className="h-3 w-3 text-emerald" /> Copied</> : <><Copy className="h-3 w-3" /> Copy</>}
      </button>
    );
  }

  return (
    <div className="space-y-5 mt-2">
      <div>
        <p className="text-sm font-medium mb-1.5">1. Run the setup wizard</p>
        <p className="text-xs text-text-secondary mb-2">
          Execute this on your GPU host to auto-detect hardware and register with Xcelsior:
        </p>
        <div className="relative">
          <pre className="bg-surface-hover rounded-lg p-3 text-sm font-mono overflow-x-auto">{installCmd}</pre>
          <CopyBtn label="install" text={installCmd} />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium mb-1.5">2. Configure environment</p>
        <p className="text-xs text-text-secondary mb-2">
          The wizard handles this automatically. To configure manually, create{" "}
          <code className="text-xs bg-surface-hover px-1 py-0.5 rounded">~/.xcelsior/.env</code>:
        </p>
        <div className="relative">
          <pre className="bg-surface-hover rounded-lg p-3 text-sm font-mono overflow-x-auto">{envTemplate}</pre>
          <CopyBtn label="env" text={envTemplate} />
        </div>
      </div>

      <div>
        <p className="text-sm font-medium mb-1.5">3. Enable as a service</p>
        <p className="text-xs text-text-secondary mb-2">
          Set up automatic startup with systemd:
        </p>
        <div className="relative">
          <pre className="bg-surface-hover rounded-lg p-3 text-[13px] font-mono overflow-x-auto leading-relaxed">{systemdUnit}</pre>
          <CopyBtn label="systemd" text={systemdUnit} />
        </div>
      </div>
    </div>
  );
}
