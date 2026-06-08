import { AuthProviderShell } from "@/components/AuthProviderShell";
import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata("Accept Team Invite", "/accept-invite");

export default function AcceptInviteLayout({ children }: { children: React.ReactNode }) {
  return <AuthProviderShell>{children}</AuthProviderShell>;
}