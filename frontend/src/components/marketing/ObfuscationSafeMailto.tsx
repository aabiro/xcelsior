"use client";

import { useEffect, useState } from "react";

/**
 * Cloudflare Email Obfuscation rewrites mailto text in the HTML edge response
 * after SSR, causing React #418 on legal pages. Render link text only after mount
 * so server HTML and the initial hydration pass stay aligned.
 */
export function ObfuscationSafeMailto({
  href,
  children,
  className,
}: {
  href: string;
  children: React.ReactNode;
  className?: string;
}) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <span className={className} aria-hidden="true">
        {"\u00a0"}
      </span>
    );
  }

  return (
    <a href={href} className={className}>
      {children}
    </a>
  );
}