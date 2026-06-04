import { privatePageMetadata } from "@/lib/page-metadata";

export const metadata = privatePageMetadata("Accept Team Invite", "/accept-invite");

export default function AcceptInviteLayout({ children }: { children: React.ReactNode }) {
  return children;
}