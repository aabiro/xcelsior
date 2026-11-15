"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { updateHostAvailability, type Host } from "@/lib/api";
import { Loader2, Pause, Play } from "lucide-react";

export function HostAvailability({ host, onUpdated }: { host: Host; onUpdated: (host: Host) => void }) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const paused = host.status === "draining";
  async function save() {
    if (host.version === undefined) return;
    setPending(true); setError("");
    try {
      const res = await updateHostAvailability(host.host_id, paused, host.version);
      onUpdated(res.host);
    } catch (error) {
      setError(error instanceof Error ? error.message : "Could not update availability. Refresh and try again.");
    } finally { setPending(false); }
  }
  return <div className="space-y-3 rounded-xl border border-border bg-navy/30 p-4">
    <div>
      <h3 className="text-sm font-semibold">Availability</h3>
      <p className="mt-1 text-sm text-text-secondary">{paused ? "Paused for new jobs. Running workloads can finish." : host.status === "active" ? "Accepting new jobs while this machine is online." : "Machine offline. It must reconnect before it can accept jobs."}</p>
    </div>
    <Button variant="outline" size="sm" onClick={save} disabled={pending || host.version === undefined || (!paused && host.status !== "active")}>
      {pending ? <Loader2 className="h-4 w-4 animate-spin" /> : paused ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
      {pending ? "Saving…" : paused ? "Resume new jobs" : "Pause new jobs"}
    </Button>
    {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
  </div>;
}
