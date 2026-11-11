"use client";

import { useEffect, useSyncExternalStore } from "react";

/**
 * Display names for breadcrumb segments that only a page knows.
 *
 * The app bar's breadcrumb is built from the URL, so a detail page like
 * `/dashboard/instances/ae1d03c3` would otherwise show its id. The page that
 * has loaded the record registers the name for its own path while mounted.
 */
const labels = new Map<string, string>();
const listeners = new Set<() => void>();
const EMPTY: ReadonlyMap<string, string> = new Map();
let snapshot: ReadonlyMap<string, string> = EMPTY;

function emit() {
  snapshot = new Map(labels);
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Labels keyed by the crumb's full href, e.g. "/dashboard/instances/ae1d03c3". */
export function useBreadcrumbLabels(): ReadonlyMap<string, string> {
  return useSyncExternalStore(subscribe, () => snapshot, () => EMPTY);
}

/** Show `label` instead of the URL segment for `href` while this component is mounted. */
export function useBreadcrumbLabel(href: string, label: string | null | undefined) {
  useEffect(() => {
    if (!label) return;
    labels.set(href, label);
    emit();
    return () => {
      if (labels.get(href) !== label) return;
      labels.delete(href);
      emit();
    };
  }, [href, label]);
}
