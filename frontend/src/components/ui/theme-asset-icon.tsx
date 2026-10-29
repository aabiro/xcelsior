"use client";

import { siteIcon } from "@/lib/brand-assets";
import { cn } from "@/lib/utils";

/** Theme-paired SVG icon — swaps with `.dashboard-shell[data-theme]` or marketing `data-theme`. */
export function ThemeAssetIcon({ name, className }: { name: string; className?: string }) {
  return (
    <>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={siteIcon(name, "dark")} className={cn("site-theme-dark", className)} alt="" aria-hidden />
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={siteIcon(name, "light")} className={cn("site-theme-light", className)} alt="" aria-hidden />
    </>
  );
}