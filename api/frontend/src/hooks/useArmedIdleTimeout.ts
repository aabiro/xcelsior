"use client";

import { useEffect } from "react";

/** Auto-disarm after idle period so armed states do not persist indefinitely. */
export function useArmedIdleTimeout(
  armed: boolean,
  onTimeout: () => void,
  timeoutMs = 45_000,
) {
  useEffect(() => {
    if (!armed) return;
    const id = window.setTimeout(onTimeout, timeoutMs);
    return () => window.clearTimeout(id);
  }, [armed, onTimeout, timeoutMs]);
}