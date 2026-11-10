import { SiteAuthRouteShell } from "@/components/marketing/SiteAuthShell";
import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata("Unsubscribe", "/unsubscribe");

export default function UnsubscribeLayout({ children }: { children: React.ReactNode }) {
  return <SiteAuthRouteShell>{children}</SiteAuthRouteShell>;
}
