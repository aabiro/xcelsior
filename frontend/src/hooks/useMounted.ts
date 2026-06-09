"use client";

import { useSyncExternalStore } from "react";

function subscribe(onStoreChange: () => void) {
  if (typeof window === "undefined") return () => {};
  const id = requestAnimationFrame(onStoreChange);
  return () => cancelAnimationFrame(id);
}

function getSnapshot() {
  return true;
}

function getServerSnapshot() {
  return false;
}

/** True after the component has mounted on the client (avoids hydration mismatch). */
export function useMounted(): boolean {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}