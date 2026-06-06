"use client";

import { LazyMotion, m, AnimatePresence } from "framer-motion";
import type { ReactNode } from "react";

const loadDomAnimation = () =>
  import("framer-motion").then((mod) => mod.domAnimation);

/** Lazy-load framer-motion features for marketing routes (smaller initial JS). */
export function MarketingMotion({ children }: { children: ReactNode }) {
  return (
    <LazyMotion features={loadDomAnimation} strict>
      {children}
    </LazyMotion>
  );
}

export { m, AnimatePresence };