"use client";

import { LazyMotion, m, AnimatePresence } from "framer-motion";
import type { ReactNode } from "react";

const loadDomAnimation = () =>
  import("framer-motion").then((mod) => mod.domAnimation);

/**
 * Lazy-load framer-motion features for marketing routes (smaller initial JS).
 *
 * Note: `strict` is intentionally disabled. framer-motion has a long-standing
 * false-positive bug (framer/motion#2037) where legitimate `m.*` components
 * (e.g. from dynamically-imported widgets like ChatWidget) can still trip the
 * "rendered a `motion` component within `LazyMotion`" invariant and crash the
 * tree, even though only `m` components are used here. The check is dev-only
 * tree-shaking guidance, not a production behavior difference, so it's safe
 * to disable.
 */
export function MarketingMotion({ children }: { children: ReactNode }) {
  return (
    <LazyMotion features={loadDomAnimation}>
      {children}
    </LazyMotion>
  );
}

export { m, AnimatePresence };