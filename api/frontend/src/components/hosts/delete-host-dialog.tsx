"use client";

import { useState } from "react";
import { AlertTriangle, Loader2, Trash2 } from "lucide-react";
import { Dialog } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import type { Host } from "@/lib/api";

export function DeleteHostDialog({
  host,
  open,
  onClose,
  onConfirm,
}: {
  host: Host | null;
  open: boolean;
  onClose: () => void;
  onConfirm: (hostId: string) => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);

  if (!host) return null;

  const label = host.hostname || host.host_id;

  async function handleDelete() {
    if (!host) return;
    setBusy(true);
    try {
      await onConfirm(host.host_id);
      onClose();
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog
      open={open}
      onClose={busy ? () => {} : onClose}
      title="Remove this host?"
      description={`You are about to permanently remove "${label}" from your account.`}
      maxWidth="max-w-md"
      className="border-accent-red/20"
    >
      <div className="space-y-4">
        <div className="flex gap-3 rounded-xl border border-accent-red/20 bg-accent-red/5 px-4 py-3">
          <AlertTriangle className="h-5 w-5 shrink-0 text-accent-red mt-0.5" />
          <div className="space-y-2 text-sm text-text-secondary">
            <p>
              <span className="font-medium text-text-primary">This cannot be undone.</span> The host
              registration will be deleted from Xcelsior and the worker will no longer receive jobs for this machine.
            </p>
            <ul className="list-disc pl-4 space-y-1 text-xs">
              <li>Active or queued jobs on this host may fail or need to be relaunched elsewhere.</li>
              <li>Telemetry, reputation, and marketplace listings tied to this host will stop updating.</li>
              <li>To offer compute again, you must register the host and reinstall the worker agent.</li>
            </ul>
          </div>
        </div>
        <p className="text-xs text-text-muted font-mono truncate">Host ID: {host.host_id}</p>
        <div className="flex justify-end gap-2 pt-1">
          <Button type="button" variant="outline" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button
            type="button"
            disabled={busy}
            onClick={() => void handleDelete()}
            className="gap-1.5 bg-accent-red hover:bg-accent-red/90"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
            Remove host
          </Button>
        </div>
      </div>
    </Dialog>
  );
}