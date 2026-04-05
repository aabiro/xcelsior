"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge, StatusBadge } from "@/components/ui/badge";
import { Input, Label, Select, TextArea } from "@/components/ui/input";
import { Pagination, usePagination } from "@/components/ui/pagination";
import { Dialog } from "@/components/ui/dialog";
import { StatCard } from "@/components/ui/stat-card";
import {
  Server, Plus, Search, RefreshCw, ArrowUpDown, ArrowUp, ArrowDown, Terminal,
  Cpu, Copy, Check, DollarSign, Activity, Zap, ChevronRight, Info, AlertCircle,
  HardDrive, Globe, Shield,
} from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { Host } from "@/lib/api";
import { toast } from "sonner";
import { useEventStream } from "@/hooks/useEventStream";
import { cn } from "@/lib/utils";

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
  const offlineHosts = hosts.filter((h) => h.status === "offline").length;
  const totalVram = hosts.reduce((sum, h) => sum + (h.vram_gb || 0), 0);

  return (
    <div className="space-y-6">
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-surface via-surface to-accent-cyan/5 p-6 md:p-8">
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-accent-cyan/10 blur-3xl" />
        <div className="absolute -bottom-10 -left-10 h-48 w-48 rounded-full bg-accent-violet/10 blur-3xl" />
        <div className="relative z-10 flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-accent-cyan/10 ring-1 ring-accent-cyan/20">
                <Server className="h-6 w-6 text-accent-cyan" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">{t("dash.hosts.title")}</h1>
                <p className="text-sm text-text-secondary">
                  Manage your GPU compute infrastructure
                </p>
              </div>
            </div>
            <p className="text-sm text-text-muted max-w-lg mt-3">
              Register GPU machines to earn revenue by providing compute power to the Xcelsior network.
              Each host runs the worker agent to accept and execute jobs.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button
              variant="outline"
              className="h-10"
              onClick={load}
            >
              <RefreshCw className="h-4 w-4" />
              {t("common.refresh")}
            </Button>
            <Button
              className="h-10 bg-accent-cyan hover:bg-accent-cyan/90 text-navy"
              onClick={() => setShowRegister(true)}
            >
              <Plus className="h-4 w-4" />
              Register Host
            </Button>
            <Button
              variant="outline"
              className="h-10"
              onClick={() => setShowInstall(true)}
            >
              <Terminal className="h-4 w-4" />
              Install Worker
            </Button>
          </div>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard label="Total Hosts" value={hosts.length} icon={Server} glow="cyan" />
        <StatCard label="Active" value={activeHosts} icon={Activity} glow="emerald" />
        <StatCard label="Offline" value={offlineHosts} icon={AlertCircle} glow="gold" />
        <StatCard label="Total VRAM" value={`${totalVram} GB`} icon={HardDrive} glow="violet" />
      </div>

      {/* Register Host Modal */}
      <Dialog
        open={showRegister}
        onClose={() => setShowRegister(false)}
        title="Register a GPU Host"
        description="Add your machine to the Xcelsior network and start earning from compute jobs."
        maxWidth="max-w-xl"
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

      {/* Filters Card */}
      <Card className="border-border/60">
        <CardContent className="py-4 px-5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted pointer-events-none" />
              <Input 
                className="pl-10 h-10" 
                placeholder="Search by hostname or ID..." 
                value={search} 
                onChange={(e) => setSearch(e.target.value)} 
              />
            </div>
            <Select 
              className="h-10 min-w-[160px]" 
              value={statusFilter} 
              onChange={(e) => setStatusFilter(e.target.value)}
            >
              <option value="all">All Statuses</option>
              <option value="active">Active</option>
              <option value="offline">Offline</option>
              <option value="maintenance">Maintenance</option>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Hosts Table */}
      {loading ? (
        <div className="space-y-3">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-[72px] rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <Card className="border-border/60 overflow-hidden">
          <div className="relative py-16 px-8 text-center">
            <div className="absolute inset-0 bg-gradient-to-b from-accent-cyan/5 to-transparent" />
            <div className="relative">
              <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-2xl bg-surface-hover ring-1 ring-border">
                <Server className="h-10 w-10 text-text-muted" />
              </div>
              <h3 className="text-xl font-semibold mb-2">{t("dash.hosts.empty")}</h3>
              <p className="text-sm text-text-secondary mb-8 max-w-md mx-auto">
                {t("dash.hosts.empty_desc")} Get started by registering your first GPU host
                or installing the worker agent on your machine.
              </p>
              <div className="flex flex-col sm:flex-row justify-center gap-3">
                <Button
                  className="h-11 bg-accent-cyan hover:bg-accent-cyan/90 text-navy"
                  onClick={() => setShowRegister(true)}
                >
                  <Plus className="h-4 w-4" />
                  Register Your First Host
                </Button>
                <Button variant="outline" className="h-11" onClick={() => setShowInstall(true)}>
                  <Terminal className="h-4 w-4" />
                  View Installation Guide
                </Button>
              </div>
            </div>
          </div>
        </Card>
      ) : (
        <Card className="border-border/60 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/60 bg-gradient-to-r from-surface-hover/80 to-surface-hover/40">
                  <th 
                    className="h-12 px-5 text-left font-semibold text-text-primary cursor-pointer select-none hover:bg-surface-hover/60 transition-colors" 
                    onClick={() => toggleSort("hostname")}
                  >
                    <span className="inline-flex items-center gap-2">
                      <Server className="h-3.5 w-3.5 text-text-muted" />
                      Host
                      <SortIcon col="hostname" />
                    </span>
                  </th>
                  <th 
                    className="h-12 px-4 text-left font-semibold text-text-primary cursor-pointer select-none hover:bg-surface-hover/60 transition-colors" 
                    onClick={() => toggleSort("gpu_model")}
                  >
                    <span className="inline-flex items-center gap-2">
                      <Cpu className="h-3.5 w-3.5 text-text-muted" />
                      GPU Model
                      <SortIcon col="gpu_model" />
                    </span>
                  </th>
                  <th 
                    className="h-12 px-4 text-center font-semibold text-text-primary cursor-pointer select-none hover:bg-surface-hover/60 transition-colors" 
                    onClick={() => toggleSort("status")}
                  >
                    <span className="inline-flex items-center gap-2 justify-center">
                      Status
                      <SortIcon col="status" />
                    </span>
                  </th>
                  <th 
                    className="h-12 px-4 text-center font-semibold text-text-primary cursor-pointer select-none hover:bg-surface-hover/60 transition-colors" 
                    onClick={() => toggleSort("vram_gb")}
                  >
                    <span className="inline-flex items-center gap-2 justify-center">
                      <HardDrive className="h-3.5 w-3.5 text-text-muted" />
                      VRAM
                      <SortIcon col="vram_gb" />
                    </span>
                  </th>
                  <th 
                    className="h-12 px-4 text-center font-semibold text-text-primary cursor-pointer select-none hover:bg-surface-hover/60 transition-colors" 
                    onClick={() => toggleSort("cost_per_hour")}
                  >
                    <span className="inline-flex items-center gap-2 justify-center">
                      <DollarSign className="h-3.5 w-3.5 text-text-muted" />
                      Rate
                      <SortIcon col="cost_per_hour" />
                    </span>
                  </th>
                  <th className="h-12 px-5 text-right font-semibold text-text-primary">Actions</th>
                </tr>
              </thead>
              <tbody>
                {pageItems.map((host, idx) => (
                  <tr 
                    key={host.host_id} 
                    className={cn(
                      "group transition-colors hover:bg-accent-cyan/5",
                      idx !== pageItems.length - 1 && "border-b border-border/40"
                    )}
                  >
                    <td className="py-4 px-5">
                      <Link href={`/dashboard/hosts/${host.host_id}`} className="group/link">
                        <div className="flex items-center gap-3">
                          <div className={cn(
                            "flex h-10 w-10 items-center justify-center rounded-lg ring-1 transition-colors",
                            host.status === "active" 
                              ? "bg-emerald/10 ring-emerald/20 text-emerald" 
                              : "bg-surface-hover ring-border text-text-muted"
                          )}>
                            <Server className="h-5 w-5" />
                          </div>
                          <div>
                            <p className="font-medium text-text-primary group-hover/link:text-accent-cyan transition-colors">
                              {host.hostname || "Unnamed Host"}
                            </p>
                            <p className="text-xs text-text-muted font-mono mt-0.5">
                              {host.host_id?.slice(0, 12)}...
                            </p>
                          </div>
                        </div>
                      </Link>
                    </td>
                    <td className="py-4 px-4">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="font-mono text-xs px-2 py-0.5">
                          {host.gpu_model || "Unknown"}
                        </Badge>
                      </div>
                    </td>
                    <td className="py-4 px-4 text-center">
                      <StatusBadge status={host.status} />
                    </td>
                    <td className="py-4 px-4 text-center">
                      <span className="font-mono text-sm text-text-secondary">
                        {host.vram_gb ? `${host.vram_gb} GB` : "—"}
                      </span>
                    </td>
                    <td className="py-4 px-4 text-center">
                      <span className={cn(
                        "font-mono text-sm font-medium",
                        host.cost_per_hour ? "text-emerald" : "text-text-muted"
                      )}>
                        {host.cost_per_hour ? `$${host.cost_per_hour}/hr` : "—"}
                      </span>
                    </td>
                    <td className="py-4 px-5 text-right">
                      <Link href={`/dashboard/hosts/${host.host_id}`}>
                        <Button 
                          variant="ghost" 
                          size="sm"
                          className="h-8 text-text-secondary hover:text-accent-cyan hover:bg-accent-cyan/10"
                        >
                          View
                          <ChevronRight className="h-3.5 w-3.5 ml-1" />
                        </Button>
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="border-t border-border/60 bg-surface-hover/30 px-5 py-3 flex items-center justify-between">
              <span className="text-xs text-text-muted">
                Showing {pageItems.length} of {filtered.length} host{filtered.length !== 1 ? "s" : ""}
              </span>
              <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

/* ── Register Host Form (renders inside Dialog) ─────────────────── */

function RegisterHostForm({ api, onDone }: { api: ReturnType<typeof useApi>; onDone: () => void }) {
  const { t } = useLocale();
  const [step, setStep] = useState<1 | 2>(1);
  const [hostname, setHostname] = useState("");
  const [gpuModel, setGpuModel] = useState("");
  const [vramGb, setVramGb] = useState("");
  const [costPerHour, setCostPerHour] = useState("");
  const [location, setLocation] = useState("");
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    try {
      await api.registerHost({ 
        hostname, 
        gpu_model: gpuModel,
        vram_gb: vramGb ? parseFloat(vramGb) : undefined,
        cost_per_hour: costPerHour ? parseFloat(costPerHour) : undefined,
        location: location || undefined,
        notes: notes || undefined,
      });
      toast.success("Host registered successfully! Install the worker agent to bring it online.");
      onDone();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to register host");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="mt-4">
      {/* Progress indicator */}
      <div className="flex items-center gap-2 mb-6">
        <div className={cn(
          "flex h-7 w-7 items-center justify-center rounded-full text-xs font-semibold transition-colors",
          step >= 1 ? "bg-accent-cyan text-navy" : "bg-surface-hover text-text-muted"
        )}>1</div>
        <div className={cn("h-0.5 flex-1 rounded-full transition-colors", step >= 2 ? "bg-accent-cyan" : "bg-border")} />
        <div className={cn(
          "flex h-7 w-7 items-center justify-center rounded-full text-xs font-semibold transition-colors",
          step >= 2 ? "bg-accent-cyan text-navy" : "bg-surface-hover text-text-muted"
        )}>2</div>
      </div>

      {step === 1 && (
        <div className="space-y-5">
          <div className="rounded-lg bg-accent-cyan/5 border border-accent-cyan/20 p-4">
            <div className="flex gap-3">
              <Info className="h-5 w-5 text-accent-cyan shrink-0 mt-0.5" />
              <div className="text-sm">
                <p className="font-medium text-text-primary mb-1">What is host registration?</p>
                <p className="text-text-secondary leading-relaxed">
                  Registering creates a record of your GPU machine in the Xcelsior network. 
                  After registration, install the worker agent on your machine to start 
                  accepting compute jobs and earning revenue.
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium">
              Hostname <span className="text-accent-red">*</span>
            </Label>
            <Input 
              value={hostname} 
              onChange={(e) => setHostname(e.target.value)} 
              required 
              placeholder="e.g., gpu-server-01 or my-workstation"
              className="h-10"
            />
            <p className="text-xs text-text-muted">
              A friendly name to identify this machine in your dashboard
            </p>
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium">
              GPU Model <span className="text-accent-red">*</span>
            </Label>
            <Select
              value={gpuModel}
              onChange={(e) => setGpuModel(e.target.value)}
              required
              className="h-10"
            >
              <option value="">Select GPU model...</option>
              <optgroup label="NVIDIA Data Center">
                <option value="H100 80GB">H100 80GB</option>
                <option value="A100 80GB">A100 80GB</option>
                <option value="A100 40GB">A100 40GB</option>
                <option value="A40">A40 48GB</option>
                <option value="A30">A30 24GB</option>
                <option value="A10">A10 24GB</option>
                <option value="L40S">L40S 48GB</option>
                <option value="L4">L4 24GB</option>
              </optgroup>
              <optgroup label="NVIDIA Consumer">
                <option value="RTX 4090">RTX 4090 24GB</option>
                <option value="RTX 4080">RTX 4080 16GB</option>
                <option value="RTX 4070 Ti">RTX 4070 Ti 12GB</option>
                <option value="RTX 3090">RTX 3090 24GB</option>
                <option value="RTX 3080">RTX 3080 10GB</option>
              </optgroup>
              <option value="Other">Other (specify in notes)</option>
            </Select>
            <p className="text-xs text-text-muted">
              The worker agent will auto-detect this, but we need it for initial registration
            </p>
          </div>

          <div className="flex justify-end pt-2">
            <Button type="button" onClick={() => setStep(2)} disabled={!hostname || !gpuModel}>
              Continue
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label className="text-sm font-medium">VRAM (GB)</Label>
              <Input 
                type="number"
                value={vramGb} 
                onChange={(e) => setVramGb(e.target.value)} 
                placeholder="e.g., 24"
                className="h-10"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-sm font-medium">Hourly Rate ($)</Label>
              <Input 
                type="number"
                step="0.01"
                value={costPerHour} 
                onChange={(e) => setCostPerHour(e.target.value)} 
                placeholder="e.g., 0.50"
                className="h-10"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium">Location</Label>
            <Select
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              className="h-10"
            >
              <option value="">Select region (optional)...</option>
              <optgroup label="North America">
                <option value="us-east">US East</option>
                <option value="us-west">US West</option>
                <option value="ca-central">Canada Central</option>
                <option value="ca-west">Canada West</option>
              </optgroup>
              <optgroup label="Europe">
                <option value="eu-west">EU West</option>
                <option value="eu-central">EU Central</option>
                <option value="uk">United Kingdom</option>
              </optgroup>
              <optgroup label="Asia Pacific">
                <option value="ap-southeast">Asia Pacific Southeast</option>
                <option value="ap-northeast">Asia Pacific Northeast</option>
              </optgroup>
            </Select>
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium">Notes</Label>
            <TextArea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Any additional info about this host (CPU, RAM, network speed, etc.)"
              rows={3}
              className="resize-none"
            />
          </div>

          <div className="rounded-lg bg-surface-hover border border-border p-4">
            <div className="flex gap-3">
              <Shield className="h-5 w-5 text-emerald shrink-0" />
              <div className="text-sm">
                <p className="font-medium text-text-primary">What happens next?</p>
                <p className="text-text-muted mt-1">
                  After registration, run the worker agent installer on your machine. 
                  It will automatically connect this host and verify your GPU specs.
                </p>
              </div>
            </div>
          </div>

          <div className="flex justify-between pt-2">
            <Button type="button" variant="ghost" onClick={() => setStep(1)}>
              Back
            </Button>
            <Button type="submit" disabled={loading} className="bg-accent-cyan hover:bg-accent-cyan/90 text-navy">
              {loading ? "Registering..." : "Register Host"}
            </Button>
          </div>
        </div>
      )}
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
