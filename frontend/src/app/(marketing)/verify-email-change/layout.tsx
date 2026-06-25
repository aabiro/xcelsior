import { AuthProviderShell } from "@/components/AuthProviderShell";
import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata("Confirm Email Change", "/verify-email-change");

export default function VerifyEmailChangeLayout({ children }: { children: React.ReactNode }) {
  return <AuthProviderShell>{children}</AuthProviderShell>;
}
