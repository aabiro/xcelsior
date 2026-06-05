/**
 * Mailto links on pages behind Cloudflare Email Obfuscation: CF rewrites
 * addresses in the HTML stream after SSR, which triggers React #418 text mismatches.
 * suppressHydrationWarning keeps hydration aligned with the obfuscated DOM.
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
  return (
    <a href={href} className={className} suppressHydrationWarning>
      {children}
    </a>
  );
}