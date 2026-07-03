"use client";

import Link from "next/link";
import type { ComponentProps, ReactNode } from "react";
import { useAuth } from "@/lib/auth";
import { marketingCtaHref, type MarketingCtaIntent } from "@/lib/auth-aware-links";
import { useMounted } from "@/hooks/useMounted";
import { cn } from "@/lib/utils";

type AuthAwareLinkProps = Omit<ComponentProps<typeof Link>, "href"> & {
  intent: MarketingCtaIntent;
};

function PendingCta({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <span className={cn(className)} aria-busy="true" style={{ pointerEvents: "none", opacity: 0.72 }}>
      {children}
    </span>
  );
}

export function AuthAwareLink({ intent, className, children, ...rest }: AuthAwareLinkProps) {
  const { user, loading } = useAuth();
  const mounted = useMounted();
  const authReady = mounted && !loading;

  if (!authReady) {
    return <PendingCta className={className}>{children}</PendingCta>;
  }

  const href = marketingCtaHref(intent, Boolean(user));

  return (
    <Link href={href} className={className} {...rest}>
      {children}
    </Link>
  );
}