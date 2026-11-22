import { AuthProviderShell } from "@/components/AuthProviderShell";
import { SiteAuthRouteShell } from "@/components/marketing/SiteAuthShell";
import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata(
  "Create Account",
  "/register",
  "Create your Xcelsior account and launch verified GPUs billed by the second.",
);

export default function RegisterLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProviderShell>
      <SiteAuthRouteShell>{children}</SiteAuthRouteShell>
    </AuthProviderShell>
  );
}
