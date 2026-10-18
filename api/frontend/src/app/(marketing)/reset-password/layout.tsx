import { SiteAuthRouteShell } from "@/components/marketing/SiteAuthShell";
import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata("Reset Password", "/reset-password");

export default function ResetPasswordLayout({ children }: { children: React.ReactNode }) {
  return <SiteAuthRouteShell>{children}</SiteAuthRouteShell>;
}
