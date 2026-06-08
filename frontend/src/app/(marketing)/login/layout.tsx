import { AuthProviderShell } from "@/components/AuthProviderShell";
import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata("Sign In", "/login", "Sign in to your Xcelsior account.");

export default function LoginLayout({ children }: { children: React.ReactNode }) {
  return <AuthProviderShell>{children}</AuthProviderShell>;
}