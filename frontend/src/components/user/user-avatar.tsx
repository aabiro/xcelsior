"use client";

import { cn } from "@/lib/utils";
import type { User } from "@/lib/auth";

const SIZE_MAP = {
  xs: { box: "h-7 w-7", text: "text-[10px]", ring: "p-[1.5px]" },
  sm: { box: "h-8 w-8", text: "text-xs", ring: "p-[1.5px]" },
  md: { box: "h-10 w-10", text: "text-base", ring: "p-[2px]" },
  lg: { box: "h-16 w-16", text: "text-xl", ring: "p-[2.5px]" },
  xl: { box: "h-24 w-24", text: "text-3xl", ring: "p-[3px]" },
} as const;

export type UserAvatarSize = keyof typeof SIZE_MAP;

function initialsFor(user?: Pick<User, "name" | "email"> | null): string {
  const ch = user?.name?.[0] || user?.email?.[0] || "?";
  return ch.toUpperCase();
}

export function UserAvatar({
  user,
  size = "md",
  className,
  showRing = true,
  src,
}: {
  user?: Pick<User, "name" | "email" | "avatar_url"> | null;
  size?: UserAvatarSize;
  className?: string;
  showRing?: boolean;
  /** Override image URL (e.g. local preview while uploading). */
  src?: string | null;
}) {
  const dim = SIZE_MAP[size];
  const imageSrc = src ?? user?.avatar_url ?? null;

  return (
    <div
      className={cn(
        "relative shrink-0 rounded-full",
        showRing && [
          dim.ring,
          "bg-gradient-to-br from-accent-cyan via-accent-violet to-accent-cyan",
          "shadow-[0_0_18px_rgba(0,212,255,0.22)]",
        ],
        className,
      )}
    >
      <div
        className={cn(
          "relative flex items-center justify-center overflow-hidden rounded-full bg-navy-light",
          dim.box,
          !showRing && "ring-1 ring-border/80",
        )}
      >
        {imageSrc ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={imageSrc}
            alt=""
            className="h-full w-full object-cover"
            referrerPolicy="no-referrer"
          />
        ) : (
          <span
            className={cn("font-semibold text-accent-cyan select-none", dim.text)}
            aria-hidden
          >
            {initialsFor(user)}
          </span>
        )}
      </div>
    </div>
  );
}