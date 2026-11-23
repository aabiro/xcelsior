"use client";

import { useEffect, useState } from "react";
import { Loader2, ShieldCheck } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Input, Label, Select, TextArea } from "@/components/ui/input";
import * as api from "@/lib/api";
import type { HpcClusterRegistration } from "@/lib/api";

const CLUSTER_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/;
/** What the API accepts: a named env var, or a file in its own key directory. */
const CREDENTIAL_REF = /^(env:XCELSIOR_SLURM_KEY_[A-Z0-9_]+|file:[^\s]+)$/;

interface Props {
  open: boolean;
  onClose: () => void;
  onRegistered: () => void;
}

/**
 * Register a cluster through POST /api/v1/hpc/clusters (platform admin, audited).
 * It is always registered disabled here: polling starts only after a connection
 * check has proven the transport, and that is a second, deliberate step.
 */
export function RegisterClusterDialog({ open, onClose, onRegistered }: Props) {
  const [profiles, setProfiles] = useState<{ id: string; name: string }[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [profile, setProfile] = useState("");
  const [clusterId, setClusterId] = useState("");
  const [viaSsh, setViaSsh] = useState(true);
  const [host, setHost] = useState("");
  const [port, setPort] = useState("22");
  const [user, setUser] = useState("");
  const [hostKey, setHostKey] = useState("");
  const [credential, setCredential] = useState("");
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!open) return;
    let live = true;
    api.fetchSlurmProfiles()
      .then((res) => {
        if (!live) return;
        const list = Object.entries(res.details || {})
          .filter(([, d]) => !d.retired)
          .map(([id, d]) => ({ id, name: d.name }));
        setProfiles(list);
        setProfile((cur) => cur || list[0]?.id || "");
        setLoadError(null);
      })
      .catch((err) => {
        if (live) setLoadError(err instanceof Error ? err.message : "Could not load cluster profiles");
      });
    return () => { live = false; };
  }, [open]);

  const portNumber = Number(port);
  const errors: Record<string, string | null> = {
    clusterId: clusterId && !CLUSTER_ID.test(clusterId) ? "Letters, digits, dot, dash or underscore; up to 64." : null,
    host: viaSsh && !host.trim() ? "Enter the login node's host name or address." : null,
    port: viaSsh && !(Number.isInteger(portNumber) && portNumber >= 1 && portNumber <= 65535) ? "A port from 1 to 65535." : null,
    hostKey: viaSsh && !hostKey.trim() ? "Paste the known_hosts line you verified out of band." : null,
    credential: viaSsh && !CREDENTIAL_REF.test(credential.trim())
      ? "env:XCELSIOR_SLURM_KEY_<NAME>, or file:<name> in the API's key directory."
      : null,
  };
  const valid = !!profile && Object.values(errors).every((e) => !e);
  const show = (field: string) => (touched[field] ? errors[field] : null);
  const touch = (field: string) => setTouched((t) => ({ ...t, [field]: true }));

  const submit = async () => {
    if (!valid) return;
    const body: HpcClusterRegistration = { profile_key: profile, enabled: false };
    if (clusterId.trim()) body.cluster_id = clusterId.trim();
    if (viaSsh) {
      body.ssh_host = host.trim();
      body.ssh_port = portNumber;
      if (user.trim()) body.remote_user = user.trim();
      body.known_hosts_line = hostKey.trim();
      body.credential_ref = credential.trim();
    }
    setSaving(true);
    try {
      const res = await api.registerHpcCluster(body);
      toast.success(`${res.cluster.cluster_id} registered, disabled. Run a connection check next.`);
      setTouched({});
      onRegistered();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not register the cluster");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Register a Slurm cluster"
      description="Registered disabled. Nothing is polled until a connection check passes."
      maxWidth="max-w-xl"
    >
      <form
        className="space-y-4 pt-4"
        onSubmit={(e) => { e.preventDefault(); void submit(); }}
        noValidate
      >
        {loadError && <p className="text-sm text-accent-red">{loadError}</p>}

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="space-y-1.5">
            <Label htmlFor="hpc-reg-profile" className="text-xs">Adapter profile</Label>
            <Select id="hpc-reg-profile" value={profile} onChange={(e) => setProfile(e.target.value)}>
              {profiles.length === 0 && <option value="">Loading…</option>}
              {profiles.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
            </Select>
            <p className="text-[11px] text-text-muted">
              Research-only, marketplace and agent access come from the profile.
            </p>
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="hpc-reg-id" className="text-xs">Cluster ID (optional)</Label>
            <Input autoComplete="off" autoCapitalize="none" autoCorrect="off" spellCheck={false}
              id="hpc-reg-id"
              placeholder="The profile's own ID"
              value={clusterId}
              onChange={(e) => setClusterId(e.target.value)}
              onBlur={() => touch("clusterId")}
              aria-invalid={!!show("clusterId")}
              className="font-mono"
            />
            {show("clusterId") && <p className="text-[11px] text-accent-red">{show("clusterId")}</p>}
          </div>
        </div>

        <label className="flex cursor-pointer items-start gap-2.5 rounded-lg border border-border p-3">
          <input
            type="checkbox"
            className="mt-0.5 h-4 w-4 accent-[var(--color-accent-cyan,#00d4ff)]"
            checked={viaSsh}
            onChange={(e) => setViaSsh(e.target.checked)}
          />
          <span>
            <span className="block text-sm font-medium">Reach it over SSH with a pinned host key</span>
            <span className="block text-xs text-text-muted">
              Off: the profile&apos;s own transport is used, as for a cluster on this machine.
            </span>
          </span>
        </label>

        {viaSsh && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-[1fr_6rem]">
              <div className="space-y-1.5">
                <Label htmlFor="hpc-reg-host" className="text-xs">Login node</Label>
                <Input
                  id="hpc-reg-host"
                  placeholder="login.cluster.example"
                  value={host}
                  onChange={(e) => setHost(e.target.value)}
                  onBlur={() => touch("host")}
                  aria-invalid={!!show("host")}
                  className="font-mono"
                  autoComplete="off"
                />
                {show("host") && <p className="text-[11px] text-accent-red">{show("host")}</p>}
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="hpc-reg-port" className="text-xs">Port</Label>
                <Input autoComplete="off"
                  id="hpc-reg-port"
                  inputMode="numeric"
                  value={port}
                  onChange={(e) => setPort(e.target.value)}
                  onBlur={() => touch("port")}
                  aria-invalid={!!show("port")}
                  className="font-mono"
                />
                {show("port") && <p className="text-[11px] text-accent-red">{show("port")}</p>}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-1.5">
                <Label htmlFor="hpc-reg-user" className="text-xs">Remote user (optional)</Label>
                <Input
                  id="hpc-reg-user"
                  value={user}
                  onChange={(e) => setUser(e.target.value)}
                  className="font-mono"
                  autoComplete="off"
                />
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="hpc-reg-cred" className="text-xs">Credential reference</Label>
                <Input
                  id="hpc-reg-cred"
                  placeholder="env:XCELSIOR_SLURM_KEY_NAME"
                  value={credential}
                  onChange={(e) => setCredential(e.target.value)}
                  onBlur={() => touch("credential")}
                  aria-invalid={!!show("credential")}
                  className="font-mono"
                  autoComplete="off"
                />
                {show("credential") && <p className="text-[11px] text-accent-red">{show("credential")}</p>}
              </div>
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="hpc-reg-key" className="text-xs">Host key to pin</Label>
              <TextArea autoComplete="off" autoCapitalize="none" autoCorrect="off"
                id="hpc-reg-key"
                rows={3}
                placeholder="login.cluster.example ssh-ed25519 AAAA…"
                value={hostKey}
                onChange={(e) => setHostKey(e.target.value)}
                onBlur={() => touch("hostKey")}
                aria-invalid={!!show("hostKey")}
                className="resize-y font-mono text-xs"
                spellCheck={false}
              />
              {show("hostKey") && <p className="text-[11px] text-accent-red">{show("hostKey")}</p>}
            </div>

            <p className="flex items-start gap-2 rounded-lg bg-surface-hover px-3 py-2 text-xs text-text-secondary">
              <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald" />
              The private key never passes through this page. Only the reference to where the API keeps it is stored,
              and the host key is shown afterwards as its fingerprint.
            </p>
          </div>
        )}

        <div className="flex flex-wrap justify-end gap-2 pt-1">
          <Button type="button" variant="outline" onClick={onClose} disabled={saving}>Cancel</Button>
          <Button type="submit" disabled={!valid || saving}>
            {saving && <Loader2 className="h-4 w-4 animate-spin" />}
            {saving ? "Registering…" : "Register disabled"}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
