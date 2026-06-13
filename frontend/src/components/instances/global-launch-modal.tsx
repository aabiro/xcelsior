"use client";

import { useEffect, useState } from "react";
import { LaunchInstanceModal } from "@/components/instances/launch-instance-modal";
import { LAUNCH_MODAL_EVENT, type LaunchModalOptions } from "@/lib/launch-modal";

/**
 * Single dashboard-level host for the Launch Instance modal. Listening for the
 * global event means any "Launch Instance" button opens the modal in place —
 * no navigation to the instances list or the detail page.
 */
export function GlobalLaunchModal() {
  const [open, setOpen] = useState(false);
  const [opts, setOpts] = useState<LaunchModalOptions>({});

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<LaunchModalOptions>).detail || {};
      setOpts(detail);
      setOpen(true);
    };
    window.addEventListener(LAUNCH_MODAL_EVENT, handler);
    return () => window.removeEventListener(LAUNCH_MODAL_EVENT, handler);
  }, []);

  return (
    <LaunchInstanceModal
      open={open}
      onClose={() => setOpen(false)}
      initialGpuModel={opts.gpu}
      initialPricingMode={opts.mode}
      preSelectedVolumeIds={opts.volumeIds}
      // Success step handles next steps; nothing to do here but allow a refresh
      // signal for any listeners that care.
      onLaunched={() => {
        window.dispatchEvent(new CustomEvent("xcelsior-instance-launched"));
      }}
    />
  );
}
