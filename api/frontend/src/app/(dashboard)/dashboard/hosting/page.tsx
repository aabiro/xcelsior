"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Check, ChevronRight, Loader2, RefreshCw, Server, ShieldCheck, SlidersHorizontal } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { fetchHosts, type Host } from "@/lib/api";
import { hostingEnabled, hostingProgress } from "@/lib/hosting";
import { StartHostingButton } from "@/components/hosting/start-hosting-button";
import { HostAvailability } from "@/components/hosting/host-availability";
import { HostSpotSettings } from "@/components/hosts/host-spot-settings";
import { Button } from "@/components/ui/button";
import { useEventStream } from "@/hooks/useEventStream";
import { cn } from "@/lib/utils";

const STEPS = [
  { title: "Connect GPU", icon: Server, description: "Register your machine and install the worker on it." },
  { title: "Verify machine", icon: ShieldCheck, description: "The worker reports hardware and runtime checks for admission." },
  { title: "Set pricing and availability", icon: SlidersHorizontal, description: "Choose your spot minimum and when to accept new jobs." },
];

export default function HostingPage() {
  const { user } = useAuth();
  const enabled = hostingEnabled(user);
  const [hosts, setHosts] = useState<Host[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const load = useCallback(async () => {
    if (!enabled || !user) return;
    setError("");
    try {
      const response = await fetchHosts();
      // Operators see the platform fleet from /hosts; hosting setup is personal.
      const ids = new Set([user.user_id, user.customer_id, user.provider_id, user.email].filter(Boolean));
      setHosts(user.is_admin || user.role === "admin"
        ? response.hosts.filter(host => [host.owner, host.owner_id, host.provider_id].some(owner => owner && ids.has(owner)))
        : response.hosts);
    } catch (error) { setError(error instanceof Error ? error.message : "Could not load your machines."); }
    finally { setLoading(false); }
  }, [enabled, user]);
  useEffect(() => {
    if (!enabled) return;
    void load();
    const timer = setInterval(() => { if (document.visibilityState === "visible") void load(); }, 30000);
    return () => clearInterval(timer);
  }, [enabled, load]);
  useEventStream({ eventTypes: ["host_registered", "host_removed", "host_update"], onEvent: () => { if (enabled) void load(); } });
  const { machines, verified, step } = hostingProgress(hosts);
  const update = (host: Host) => setHosts(current => current.map(existing => existing.host_id === host.host_id ? { ...existing, ...host } : existing));
  return <div className="mx-auto max-w-5xl space-y-7">
    <header className="space-y-3">
      <p className="text-xs font-semibold uppercase tracking-widest text-accent-cyan">Hosting</p>
      <h1 className="text-3xl font-semibold tracking-tight">{enabled ? "Your hosting setup" : "Put your GPUs to work"}</h1>
      <p className="max-w-2xl text-sm leading-relaxed text-text-secondary">Connect a GPU, finish machine verification, then choose pricing and availability. You can keep renting with the same account.</p>
    </header>
    <ol aria-label="Hosting setup progress" className="grid gap-3 md:grid-cols-3">
      {STEPS.map(({ title, icon: Icon, description }, index) => {
        const complete = enabled && !loading && !error && step > index + 1;
        const current = enabled && !loading && !error && step === index + 1;
        return <li key={title} aria-current={current ? "step" : undefined} className={cn("rounded-2xl border p-5", current ? "border-accent-cyan/40 bg-accent-cyan/5" : "border-border bg-surface")}>
          <div className="mb-4 flex items-center gap-3"><span className={cn("flex h-9 w-9 items-center justify-center rounded-lg", complete ? "bg-emerald/10 text-emerald" : "bg-accent-cyan/10 text-accent-cyan")}>
            {complete ? <Check className="h-5 w-5" aria-label="Complete" /> : <Icon className="h-5 w-5" />}
          </span><span className="text-xs text-text-muted">Step {index + 1}</span></div>
          <h2 className="text-base font-semibold">{title}</h2><p className="mt-2 text-sm leading-relaxed text-text-secondary">{description}</p>
        </li>;
      })}
    </ol>
    {!enabled ? <StartHostingButton /> : loading ? <p role="status" className="flex items-center gap-2 text-sm text-text-secondary"><Loader2 className="h-4 w-4 animate-spin" /> Loading your machines…</p> : error ?
      <div role="alert" className="space-y-3 rounded-xl border border-red-400/30 p-5"><p className="text-sm text-red-400">{error}</p><Button onClick={load} variant="outline">Retry</Button></div> : <>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <Link href="/dashboard/hosts?setup=connect" className="inline-flex items-center gap-2 text-sm font-medium text-accent-cyan">{machines.length ? "Connect another GPU" : "Connect your first GPU"}<ChevronRight className="h-4 w-4" /></Link>
          <Button variant="outline" size="sm" onClick={load}><RefreshCw className="h-4 w-4" /> Refresh status</Button>
        </div>
        {machines.length > 0 && verified.length === 0 && <section className="space-y-4 rounded-2xl border border-border bg-surface p-5">
          <h2 className="font-semibold">Waiting for machine verification</h2>
          <p className="text-sm text-text-secondary">Install and run the worker on your GPU machine. Registration alone does not verify it. This page updates as the machine reports in.</p>
          {machines.map(host => <div key={host.host_id} className="space-y-1 border-t border-border pt-3">
            <Link href={`/dashboard/hosts/${encodeURIComponent(host.host_id)}`} className="text-sm font-medium text-accent-cyan">{host.hostname || host.host_id}</Link>
            <p className="text-sm text-text-secondary">{host.gpu_model} · {host.status}</p>
            {host.admission_details?.rejection_reasons?.map(reason => <p key={reason} className="text-sm text-amber-400">{reason}</p>)}
          </div>)}
          <Link href="/dashboard/hosts?setup=connect" className="inline-block text-sm text-accent-cyan underline">Worker setup instructions</Link>
        </section>}
        {verified.map(host => <section key={host.host_id} className="space-y-4 rounded-2xl border border-border bg-surface p-5">
          <div><Link href={`/dashboard/hosts/${encodeURIComponent(host.host_id)}`} className="font-semibold text-accent-cyan">{host.hostname || host.host_id}</Link><p className="mt-1 text-sm text-text-secondary">{host.gpu_model} · Machine verified</p></div>
          <div className="grid items-start gap-4 lg:grid-cols-2"><HostSpotSettings host={host} onUpdated={update} /><HostAvailability host={host} onUpdated={update} /></div>
        </section>)}
        {verified.length > 0 && <p className="text-sm text-text-secondary">Standard rates follow the platform catalog. Spot settings control your interruptible capacity. <Link href="/dashboard/earnings" className="text-accent-cyan underline">Set up payouts on Earnings</Link>.</p>}
      </>}
  </div>;
}
