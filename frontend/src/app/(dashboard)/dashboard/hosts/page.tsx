"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useState, useRef } from "react";
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
  HardDrive, Globe, Shield, Package, Code2, Clipboard, ArrowRight, ArrowLeft,
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
              data-action="register"
              onClick={() => setShowRegister(true)}
            >
              <Plus className="h-4 w-4" />
              Register Host
            </Button>
            <Button
              variant="outline"
              className="h-10 border-accent-cyan/30 text-accent-cyan hover:bg-accent-cyan/10 hover:border-accent-cyan/50 shadow-sm shadow-accent-cyan/10 hover:shadow-accent-cyan/20 transition-all duration-200"
              data-action="install"
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

      <HostSetupGuideCard onRegister={() => setShowRegister(true)} />

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
        maxWidth="max-w-3xl"
        className="border-accent-cyan/20 shadow-lg shadow-accent-cyan/5 h-[82vh]"
        bodyClassName="flex-1 min-h-0 flex flex-col overflow-hidden"
      >
        <InstallWorkerSection />
      </Dialog>

      {/* Architecture — always visible */}
      <ArchitectureCard />

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
                        <Badge variant="default" className="font-mono text-xs px-2 py-0.5">
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

      <ProviderTipsCard hosts={hosts} />
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

const LLM_INSTALL_PROMPT = `I am setting up an Xcelsior GPU worker node to join the distributed GPU compute marketplace at xcelsior.ca. Walk me through setup step-by-step. Mark any values I need to fill in with placeholder comments.

## Option A: SDK + AI Onboarding Wizard (Recommended)

\`\`\`bash
npm install -g @xcelsior-gpu/sdk @xcelsior-gpu/wizard
xcelsior-wizard setup
\`\`\`

The AI Onboarding Wizard will ask whether you want to rent GPUs, provide GPUs, or both — then handle hardware detection, host registration, pricing, and worker service setup automatically.

### SDK Commands (available after setup)

\`\`\`bash
xcelsior status                                  # Worker status
xcelsior jobs --watch                            # Live job queue
xcelsior pricing set --gpu "RTX 4090" --rate 0.45  # Update pricing
xcelsior diagnostics --full                      # Run diagnostics
xcelsior earnings --period 30d                   # Earnings summary
\`\`\`

### Requirements
- Node.js >= 18, NVIDIA drivers >= 535, Docker >= 24.0, Ubuntu 22.04+ or WSL2

## Option B: Manual Setup

### 1. Install worker agent

\`\`\`bash
curl -fsSL https://xcelsior.ca/install.sh | bash
\`\`\`

### 2. Create environment file

Create \`~/.xcelsior/worker.env\`:

\`\`\`bash
XCELSIOR_HOST_ID=<your-host-id>            # From dashboard after registering
XCELSIOR_SCHEDULER_URL=https://xcelsior.ca
# Choose one auth method below. If both are set, the worker prefers OAuth.
XCELSIOR_API_TOKEN=<your-api-token>        # Settings → API & SSH
XCELSIOR_OAUTH_CLIENT_ID=<your-client-id>  # Settings → API & SSH
XCELSIOR_OAUTH_CLIENT_SECRET=<your-client-secret>
XCELSIOR_HOST_IP=<your-host-ip>            # Tailscale/Headscale mesh or public
\`\`\`

### 3. Enable as a systemd service

\`\`\`bash
sudo tee /etc/systemd/system/xcelsior-worker.service << 'EOF'
[Unit]
Description=Xcelsior Worker Agent
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
EnvironmentFile=$HOME/.xcelsior/worker.env
ExecStart=/usr/bin/python3 $HOME/.xcelsior/worker_agent.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now xcelsior-worker
\`\`\`

### 4. Verify

\`\`\`bash
sudo systemctl status xcelsior-worker
journalctl -u xcelsior-worker -f
\`\`\`

## Notes
- Register your host first at https://xcelsior.ca/dashboard/hosts → "Register Host" (the AI Onboarding Wizard handles this automatically).
- \`nvidia-smi\` not found → \`sudo apt install nvidia-driver-535\`
- Docker permission denied → \`sudo usermod -aG docker $USER\`
- Can't connect → check firewall allows outbound HTTPS to xcelsior.ca:443
- Not picking up jobs → \`xcelsior pricing compare\` to check competitiveness`;

function CodeSnippet({
  label,
  text,
  copied,
  onCopy,
  className,
}: {
  label: string;
  text: string;
  copied: string | null;
  onCopy: (label: string, text: string) => void;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-2xl border border-slate-200/90 bg-[linear-gradient(180deg,rgba(255,255,255,0.96),rgba(242,247,255,0.96))] px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.75),0_8px_24px_rgba(15,23,42,0.06)] dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(5,11,22,0.98),rgba(4,8,18,0.94))] dark:shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]",
        className,
      )}
    >
      <button
        onClick={() => onCopy(label, text)}
        className="absolute right-3 top-3 flex items-center gap-1 rounded-lg border border-slate-200/90 bg-white/80 px-2 py-1 text-xs text-text-secondary transition-colors hover:border-accent-cyan/35 hover:text-accent-cyan dark:border-white/10 dark:bg-white/5"
      >
        {copied === label ? (
          <>
            <Check className="h-3 w-3 text-emerald" /> Copied
          </>
        ) : (
          <>
            <Copy className="h-3 w-3" /> Copy
          </>
        )}
      </button>
      <pre className="overflow-x-auto pr-20 text-[13px] font-mono leading-relaxed text-slate-800 dark:text-[#d9e8ff]">{text}</pre>
    </div>
  );
}

function HostSetupGuideCard({ onRegister }: { onRegister: () => void }) {
  const [copied, setCopied] = useState<string | null>(null);
  const sdkInstall = "npm install -g @xcelsior-gpu/sdk @xcelsior-gpu/wizard";
  const wizardCmd = "xcelsior-wizard setup";

  function handleCopy(label: string, text: string) {
    navigator.clipboard.writeText(text);
    setCopied(label);
    toast.success("Copied to clipboard");
    setTimeout(() => setCopied((current) => current === label ? null : current), 2000);
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[0.98fr_1.02fr]">
      {/* Steps card */}
      <Card className="relative overflow-hidden border-border/60 bg-gradient-to-br from-surface via-surface to-accent-cyan/[0.04]">
        <div className="absolute -right-16 top-0 h-40 w-40 rounded-full bg-accent-cyan/10 blur-3xl" />
        <div className="absolute bottom-0 left-12 h-32 w-32 rounded-full bg-accent-violet/10 blur-3xl" />
        <div className="relative p-6 md:p-7">
          <p className="text-sm font-semibold uppercase tracking-[0.24em] text-accent-cyan">Become a Host</p>

          <div className="mt-5 space-y-5">
            {/* Step 1 */}
            <div className="flex items-start gap-3">
              <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-accent-cyan/25 bg-accent-cyan/10 text-sm font-semibold text-accent-cyan">
                1
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-text-primary">Register host</p>
                <p className="mt-0.5 text-xs text-text-muted">Create the machine record.</p>
                <Button
                  className="mt-3 h-10 rounded-full bg-accent-cyan px-4 text-navy hover:bg-accent-cyan/90"
                  onClick={onRegister}
                >
                  <Plus className="h-4 w-4" />
                  Register Host
                </Button>
              </div>
            </div>

            <div className="ml-4 border-t border-border/40" />

            {/* Step 2 */}
            <div className="flex items-start gap-3">
              <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-accent-violet/25 bg-accent-violet/10 text-sm font-semibold text-accent-violet">
                2
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-text-primary">Install SDK</p>
                <p className="mt-0.5 text-xs text-text-muted">Add the setup tools globally.</p>
                <div className="mt-3">
                  <CodeSnippet label="host-card-sdk" text={sdkInstall} copied={copied} onCopy={handleCopy} />
                </div>
              </div>
            </div>

            <div className="ml-4 border-t border-border/40" />

            {/* Step 3 */}
            <div className="flex items-start gap-3">
              <span className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-accent-gold/25 bg-accent-gold/10 text-sm font-semibold text-accent-gold">
                3
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium text-text-primary">Run AI Onboarding Wizard</p>
                <p className="mt-0.5 text-xs text-text-muted">Detect, register, and configure.</p>
                <div className="mt-3">
                  <CodeSnippet label="host-card-wizard" text={wizardCmd} copied={copied} onCopy={handleCopy} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Image — fully transparent background */}
      <div className="relative hidden min-h-[360px] overflow-hidden rounded-2xl xl:block">
        <img
          src="/xcelsior-hosts-setup-transparent.svg"
          alt=""
          aria-hidden
          width={1200}
          height={900}
          loading="eager"
          decoding="sync"
          fetchPriority="high"
          className="h-full w-full object-cover"
        />
        <div className="pointer-events-none absolute inset-0" style={{ background: "linear-gradient(to bottom, var(--background) 0%, color-mix(in srgb, var(--background) 60%, transparent) 20%, color-mix(in srgb, var(--background) 25%, transparent) 45%, transparent 70%)" }} />
        <div className="pointer-events-none absolute inset-0" style={{ background: "linear-gradient(to right, var(--background) 0%, color-mix(in srgb, var(--background) 60%, transparent) 15%, color-mix(in srgb, var(--background) 25%, transparent) 35%, transparent 60%)" }} />
      </div>
    </div>
  );
}

function ProviderTipsCard({ hosts }: { hosts: Host[] }) {
  const [dismissed, setDismissed] = useState(() => {
    if (typeof window !== "undefined") return localStorage.getItem("xcelsior_tips_dismissed") === "1";
    return false;
  });

  const dismiss = () => {
    setDismissed(true);
    localStorage.setItem("xcelsior_tips_dismissed", "1");
  };

  const restore = () => {
    setDismissed(false);
    localStorage.removeItem("xcelsior_tips_dismissed");
  };

  if (dismissed) {
    return (
      <button
        onClick={restore}
        className="flex items-center gap-1.5 text-xs text-text-muted hover:text-accent-cyan transition-colors"
      >
        <Info className="h-3 w-3" /> Show tips
      </button>
    );
  }

  const activeHosts = hosts.filter((h) => h.status === "active");
  const hasHosts = hosts.length > 0;
  const hasActive = activeHosts.length > 0;

  // Determine state
  type TipState = "empty" | "registered" | "active";
  const state: TipState = hasActive ? "active" : hasHosts ? "registered" : "empty";

  const tips: Record<TipState, { icon: typeof Zap; color: string; title: string; description: string; items: string[]; cta?: { label: string; action: string } }> = {
    empty: {
      icon: Server,
      color: "accent-cyan",
      title: "Get started: register your first GPU",
      description: "You haven't registered any hosts yet. Add your machine to the Xcelsior network to start earning revenue from GPU compute jobs.",
      items: [
        "Click Register Host above to add your GPU machine",
        "You'll need: hostname, GPU model, VRAM, and a price per hour",
        "After registering, install the worker agent to go online",
        "Competitive pricing tip: check Spot Pricing for current market rates",
      ],
      cta: { label: "Register Host", action: "register" },
    },
    registered: {
      icon: Terminal,
      color: "accent-violet",
      title: "Almost there: install the worker agent",
      description: `You have ${hosts.length} host${hosts.length > 1 ? "s" : ""} registered but none are active. Install the worker agent to bring your GPU online and start accepting jobs.`,
      items: [
        "Click Install Worker above for setup instructions",
        "The agent runs as a background process and sends heartbeats every 30s",
        "Make sure Docker is installed and your user is in the docker group",
        "Allow outbound HTTPS to xcelsior.ca:443 through your firewall",
        "Once running, your host status will change to 'active' automatically",
      ],
      cta: { label: "Install Worker", action: "install" },
    },
    active: {
      icon: Zap,
      color: "emerald",
      title: "You're live! Tips to maximize earnings",
      description: `${activeHosts.length} of ${hosts.length} host${hosts.length > 1 ? "s" : ""} active. Your GPU is accepting jobs from the marketplace. Here are tips to maximize your revenue.`,
      items: [
        "Keep your worker agent running 24/7 for highest uptime score",
        "Competitive pricing improves job assignment priority — check Spot Pricing",
        "Monitor GPU utilization in your host detail page for performance insights",
        "Complete identity verification to unlock higher trust tiers and premium jobs",
        "Set up Stripe payouts in Earnings to receive your revenue",
      ],
    },
  };

  const tip = tips[state];
  const Icon = tip.icon;

  return (
    <Card className="border-border/60 relative overflow-hidden bg-gradient-to-br from-surface via-surface to-accent-cyan/[0.03]">
      <div className={cn("absolute top-0 left-0 right-0 h-[2px]", {
        "bg-accent-cyan": state === "empty",
        "bg-accent-violet": state === "registered",
        "bg-emerald": state === "active",
      })} />
      <div className="px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <div className={cn("flex h-8 w-8 items-center justify-center rounded-lg shrink-0 mt-0.5", {
              "bg-accent-cyan/10 text-accent-cyan": state === "empty",
              "bg-accent-violet/10 text-accent-violet": state === "registered",
              "bg-emerald/10 text-emerald": state === "active",
            })}>
              <Icon className="h-4 w-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-text-secondary">{tip.title}</h3>
              <p className="text-xs text-text-muted mt-1 max-w-xl leading-relaxed">{tip.description}</p>
            </div>
          </div>
          <button
            onClick={dismiss}
            className="text-text-muted hover:text-text-secondary text-xs shrink-0 mt-1 opacity-60 hover:opacity-100 transition-opacity"
            title="Dismiss tips"
          >
            ✕
          </button>
        </div>

        <ul className="mt-3 ml-11 space-y-1.5">
          {tip.items.map((item, i) => (
            <li key={i} className="flex items-start gap-2 text-xs text-text-muted leading-relaxed">
              <ChevronRight className="h-3 w-3 text-text-muted/60 shrink-0 mt-0.5" />
              {item}
            </li>
          ))}
        </ul>

        {tip.cta && (
          <div className="mt-4 ml-11">
            <Button
              variant="outline"
              className={cn("h-8 text-xs", {
                "border-accent-cyan/30 text-accent-cyan hover:bg-accent-cyan/5": state === "empty",
                "border-accent-violet/30 text-accent-violet hover:bg-accent-violet/5": state === "registered",
              })}
              onClick={() => {
                const el = document.querySelector(`[data-action="${tip.cta!.action}"]`);
                if (el) (el as HTMLButtonElement).click();
              }}
            >
              <ArrowRight className="h-3 w-3" />
              {tip.cta.label}
            </Button>
          </div>
        )}
      </div>
    </Card>
  );
}

/* ── Architecture Card — always visible (not dismissible) ────────── */

function ArchitectureCard() {
  return (
    <Card className="border-border/60 relative overflow-hidden bg-gradient-to-br from-surface via-surface to-accent-cyan/[0.03]">
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-accent-cyan via-accent-violet to-accent-gold" />
      <div className="px-5 py-4">
        <p className="text-xs font-semibold text-text-secondary mb-3 flex items-center gap-1.5">
          <Shield className="h-3.5 w-3.5 text-accent-cyan" /> How it works
        </p>
        <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { n: "1", label: "Worker Agent", desc: "Lightweight process sends heartbeats every 30s", color: "accent-cyan" },
            { n: "2", label: "Job Assignment", desc: "Scheduler matches renters to your GPU via lease", color: "accent-violet" },
            { n: "3", label: "Secure Container", desc: "Sandboxed Docker with GPU access + SSH keys", color: "emerald" },
            { n: "4", label: "Earn Revenue", desc: "Per-second billing, real-time telemetry", color: "accent-gold" },
          ].map((step) => (
            <div key={step.n} className={cn("rounded-lg border border-border/40 p-3 flex items-start gap-2.5 bg-gradient-to-br from-transparent", `to-${step.color}/[0.03]`)}>
              <div className={cn("flex h-6 w-6 items-center justify-center rounded-md text-xs font-bold shrink-0", `bg-${step.color}/10 text-${step.color} ring-1 ring-${step.color}/20`)}>
                {step.n}
              </div>
              <div>
                <p className="text-xs font-medium text-text-secondary">{step.label}</p>
                <p className="text-[11px] text-text-muted leading-tight mt-0.5">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

type InstallView = "sdk" | "quickstart";

function InstallWorkerSection() {
  const [view, setView] = useState<InstallView>("sdk");
  const [copied, setCopied] = useState<string | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);

  function copy(label: string, text: string) {
    navigator.clipboard.writeText(text);
    setCopied(label);
    toast.success(label === "llm-prompt" ? "LLM prompt copied to clipboard" : "Copied to clipboard");
    setTimeout(() => setCopied(null), 2000);
  }

  function switchView() {
    setView(view === "sdk" ? "quickstart" : "sdk");
    contentRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Scrollable content with subtle inner side borders */}
      <div
        ref={contentRef}
        className="flex-1 min-h-0 overflow-y-auto px-6 pt-4 pb-2 border-l border-r border-border/20"
      >
        <div className="rounded-lg border border-white/5 ring-1 ring-inset ring-border/20 p-4">
          {view === "sdk" ? <SdkSetupView copied={copied} onCopy={copy} /> : <ManualQuickstartView copied={copied} onCopy={copy} />}
        </div>
      </div>

      {/* Pinned footer — brand-line separator + side borders */}
      <div className="border-l border-r border-b border-border/20 rounded-b-xl overflow-hidden">
        <div className="brand-line" />
        <div className="bg-surface/95 backdrop-blur-sm px-6 py-4">
          <div className="flex items-center justify-between gap-4">
          <p className="text-xs text-text-muted max-w-sm leading-relaxed">
            Get started with a code quickstart or copy these setup steps as a prompt.
          </p>
          <div className="flex items-center gap-2.5 shrink-0">
            <button
              onClick={() => copy("llm-prompt", LLM_INSTALL_PROMPT)}
              className="flex items-center gap-2 rounded-lg border border-accent-cyan/30 bg-gradient-to-r from-accent-cyan/5 to-accent-violet/5 px-3.5 py-2 text-xs font-medium text-text-secondary hover:border-accent-cyan/50 hover:text-accent-cyan hover:from-accent-cyan/10 hover:to-accent-violet/10 transition-all duration-200"
            >
              {copied === "llm-prompt" ? <Check className="h-3.5 w-3.5 text-emerald" /> : <Clipboard className="h-3.5 w-3.5" />}
              Copy prompt for LLM
            </button>
            <button
              onClick={switchView}
              className="flex items-center gap-2 rounded-lg bg-gradient-to-r from-accent-cyan to-accent-cyan/90 text-navy px-3.5 py-2 text-xs font-semibold hover:from-accent-cyan/90 hover:to-accent-cyan transition-all duration-200 shadow-sm shadow-accent-cyan/25"
            >
              {view === "sdk" ? (
                <>View Quickstart <ArrowRight className="h-3.5 w-3.5" /></>
              ) : (
                <><ArrowLeft className="h-3.5 w-3.5" /> View SDK Setup</>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
  );
}

/* ── SDK + AI Onboarding Wizard View ──────────────────────────────── */

function SdkSetupView({ copied, onCopy }: { copied: string | null; onCopy: (label: string, text: string) => void }) {
  const sdkInstall = `npm install -g @xcelsior-gpu/sdk @xcelsior-gpu/wizard`;
  const wizardCmd = `xcelsior-wizard setup`;
  const quickCmds = `# Check worker status
xcelsior status

# View live job queue
xcelsior jobs --watch

# Update pricing dynamically
xcelsior pricing set --gpu "RTX 4090" --rate 0.45

# Run diagnostics
xcelsior diagnostics --full

# View earnings summary
xcelsior earnings --period 30d`;

  return (
    <div className="space-y-5 mt-2">
      {/* Header pill */}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1.5 rounded-full bg-accent-cyan/10 border border-accent-cyan/20 px-3 py-1 text-xs font-medium text-accent-cyan">
          <Package className="h-3.5 w-3.5" />
          SDK &amp; AI Onboarding Wizard
        </div>
        <span className="text-[11px] text-text-muted">Recommended</span>
      </div>

      {/* Step 1: Install */}
      <div>
        <p className="text-sm font-medium mb-1.5">1. Install the SDK</p>
        <p className="text-xs text-text-secondary mb-2">
          Install the Xcelsior CLI tools globally. Requires Node.js &ge; 18.
        </p>
        <CodeSnippet label="sdk-install" text={sdkInstall} copied={copied} onCopy={onCopy} />
      </div>

      {/* Step 2: Run AI Onboarding Wizard */}
      <div>
        <p className="text-sm font-medium mb-1.5">2. Run the AI Onboarding Wizard</p>
        <p className="text-xs text-text-secondary mb-2">
          The AI Onboarding Wizard walks you through setup — it will ask whether you want to rent, provide, or both, then handle everything from there.
        </p>
        <CodeSnippet label="wizard-cmd" text={wizardCmd} copied={copied} onCopy={onCopy} />
      </div>

      {/* What the wizard does */}
      <div className="rounded-lg border border-border/60 bg-surface-hover/30 p-3.5">
        <p className="text-xs font-medium text-text-primary mb-2">The AI Onboarding Wizard will:</p>
        <ul className="text-xs text-text-secondary space-y-1.5">
          <li className="flex items-start gap-2">
            <span className="text-accent-cyan mt-0.5">&#x2022;</span>
            Ask your intent (rent GPUs, provide GPUs, or both) and tailor the flow accordingly
          </li>
          <li className="flex items-start gap-2">
            <span className="text-accent-cyan mt-0.5">&#x2022;</span>
            Auto-detect hardware, register your host, configure pricing, and install the worker service
          </li>
        </ul>
      </div>

      {/* Step 3: Quick-start commands */}
      <div>
        <p className="text-sm font-medium mb-1.5">3. Quick-start commands</p>
        <p className="text-xs text-text-secondary mb-2">
          After setup, use these SDK commands to manage your worker:
        </p>
        <CodeSnippet label="quick-cmds" text={quickCmds} copied={copied} onCopy={onCopy} />
      </div>

      {/* Pre-install requirements */}
      <div className="rounded-lg bg-surface-hover/30 border border-border/40 p-3.5">
        <div className="flex gap-2.5">
          <Info className="h-4 w-4 text-accent-cyan shrink-0 mt-0.5" />
          <div>
            <p className="text-xs font-medium text-text-primary mb-1.5">Pre-install requirements</p>
            <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs text-text-secondary">
              <span>Node.js &ge; 18</span>
              <span>NVIDIA drivers &ge; 535</span>
              <span>Docker Engine &ge; 24.0</span>
              <span>Ubuntu 22.04+ or WSL2</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Manual Quickstart View ──────────────────────────────────────── */

function ManualQuickstartView({ copied, onCopy }: { copied: string | null; onCopy: (label: string, text: string) => void }) {
  const installCmd = `curl -fsSL https://xcelsior.ca/install.sh | bash`;

  const envTemplate = `XCELSIOR_HOST_ID=<your-host-id>
XCELSIOR_SCHEDULER_URL=https://xcelsior.ca
# Choose one auth method below. If both are set, the worker prefers OAuth.
XCELSIOR_API_TOKEN=<your-api-token>
XCELSIOR_OAUTH_CLIENT_ID=<your-client-id>
XCELSIOR_OAUTH_CLIENT_SECRET=<your-client-secret>
XCELSIOR_HOST_IP=<your-host-ip>`;


  const systemdUnit = `[Unit]
Description=Xcelsior Worker Agent
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
EnvironmentFile=$HOME/.xcelsior/worker.env
ExecStart=/usr/bin/python3 $HOME/.xcelsior/worker_agent.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target`;

  const verifyCmd = `sudo systemctl daemon-reload
sudo systemctl enable --now xcelsior-worker
sudo systemctl status xcelsior-worker`;

  return (
    <div className="space-y-5 mt-2">
      {/* Header pill */}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1.5 rounded-full bg-accent-violet/10 border border-accent-violet/20 px-3 py-1 text-xs font-medium text-accent-violet">
          <Code2 className="h-3.5 w-3.5" />
          Manual Setup
        </div>
        <span className="text-[11px] text-text-muted">Step-by-step</span>
      </div>

      <div>
        <p className="text-sm font-medium mb-1.5">1. Run the setup script</p>
        <p className="text-xs text-text-secondary mb-2">
          Execute this on your GPU host to auto-detect hardware and register with Xcelsior:
        </p>
        <CodeSnippet label="install" text={installCmd} copied={copied} onCopy={onCopy} />
      </div>

      <div>
        <p className="text-sm font-medium mb-1.5">2. Configure environment</p>
        <p className="text-xs text-text-secondary mb-2">
          The AI Onboarding Wizard handles this automatically. To configure manually, create{" "}
          <code className="text-xs bg-surface-hover px-1 py-0.5 rounded">~/.xcelsior/worker.env</code>:
        </p>
        <CodeSnippet label="env" text={envTemplate} copied={copied} onCopy={onCopy} />
      </div>

      <div>
        <p className="text-sm font-medium mb-1.5">3. Enable as a service</p>
        <p className="text-xs text-text-secondary mb-2">
          Set up automatic startup with systemd:
        </p>
        <CodeSnippet label="systemd" text={systemdUnit} copied={copied} onCopy={onCopy} />
      </div>

      <div>
        <p className="text-sm font-medium mb-1.5">4. Start and verify</p>
        <p className="text-xs text-text-secondary mb-2">
          Enable the service and confirm it&apos;s running:
        </p>
        <CodeSnippet label="verify" text={verifyCmd} copied={copied} onCopy={onCopy} />
      </div>
    </div>
  );
}
