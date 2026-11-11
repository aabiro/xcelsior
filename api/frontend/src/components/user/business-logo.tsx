"use client";

import { useState } from "react";
import { Building2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { BusinessLogoTile } from "@/lib/auth";
import {
  businessName,
  monogram,
  monogramGradient,
  normalizeTile,
  type BusinessIdentity,
} from "@/lib/business-profile";

/**
 * A business is drawn as a rounded square and a person as a circle, so the two
 * read differently at a glance wherever they appear side by side.
 */
const SIZE_MAP = {
  "2xs": { box: "h-4 w-4 rounded-[5px]", text: "text-[8px]", letters: 1 },
  xs: { box: "h-7 w-7 rounded-[8px]", text: "text-[11px]", letters: 1 },
  sm: { box: "h-8 w-8 rounded-[9px]", text: "text-[12px]", letters: 1 },
  md: { box: "h-10 w-10 rounded-[11px]", text: "text-[14px]", letters: 2 },
  lg: { box: "h-12 w-12 rounded-[13px]", text: "text-[16px]", letters: 2 },
  xl: { box: "h-24 w-24 rounded-[24px]", text: "text-[30px]", letters: 2 },
} as const;

export type BusinessLogoSize = keyof typeof SIZE_MAP;

export function BusinessLogo({
  business,
  size = "md",
  className,
  src,
  tile,
  label,
}: {
  business?: Partial<BusinessIdentity> | null;
  size?: BusinessLogoSize;
  className?: string;
  /** Override the image (a local preview while uploading). */
  src?: string | null;
  /** Override the tile (with `src`). */
  tile?: BusinessLogoTile | null;
  /** Accessible name, when no visible name sits beside the logo. */
  label?: string;
}) {
  const dim = SIZE_MAP[size];
  const imageSrc = src ?? business?.business_logo_url ?? null;
  // Remember which address failed rather than a flag, so a new logo gets its
  // own chance to load without resetting state in an effect.
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  const failed = imageSrc !== null && failedSrc === imageSrc;

  const name = businessName(business);
  const showImage = Boolean(imageSrc) && !failed;
  const resolvedTile = showImage ? normalizeTile(tile ?? business?.business_logo_tile) : "monogram";

  return (
    <span
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      data-tile={resolvedTile}
      className={cn(
        "business-logo relative inline-flex shrink-0 select-none items-center justify-center overflow-hidden",
        dim.box,
        className,
      )}
      style={resolvedTile === "monogram" ? { backgroundImage: monogramGradient(name || "business") } : undefined}
    >
      {showImage ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={imageSrc!}
          alt=""
          draggable={false}
          decoding="async"
          className="h-full w-full object-cover"
          onError={() => setFailedSrc(imageSrc)}
        />
      ) : name ? (
        <span className={cn("relative font-semibold leading-none tracking-[-0.02em]", dim.text)}>
          {monogram(name, dim.letters)}
        </span>
      ) : (
        <Building2 className="relative h-[46%] w-[46%]" strokeWidth={1.75} />
      )}
    </span>
  );
}

/** The small "Business" pill shown beside a business's name. */
export function BusinessBadge({ label, className }: { label: string; className?: string }) {
  return (
    <span
      className={cn(
        "business-badge inline-flex shrink-0 items-center gap-1 rounded-[6px] px-1.5 py-[3px]",
        "text-[10px] font-semibold uppercase leading-none tracking-[0.07em]",
        className,
      )}
    >
      <Building2 className="h-2.5 w-2.5" strokeWidth={2.25} aria-hidden />
      {label}
    </span>
  );
}
